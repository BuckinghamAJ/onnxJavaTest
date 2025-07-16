package com.method.onnxTest.service;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.method.onnxTest.dto.ScalerParams;
import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class ClassifierModelPrediction {

    private OrtEnvironment environment; // Probably would need to separate class to handle and passaround in future
    private OrtSession session;
    private final ReentrantLock lock = new ReentrantLock(); // Thread safety for tensor creation
    private String resourcePath = "models/sklearn_classifier.onnx"; //
    private String inputName;
    private String outputName;
    private final ScalerParams scalerParams;

    private static final Logger logger = LoggerFactory.getLogger(
        ClassifierModelPrediction.class
    );

    public ClassifierModelPrediction() throws OrtException, IOException {
        environment = OrtEnvironment.getEnvironment();

        try (
            InputStream modelStream = getClass()
                .getClassLoader()
                .getResourceAsStream(resourcePath);
        ) {
            if (modelStream == null) {
                throw new RuntimeException(
                    "Model file not found: " + resourcePath
                );
            }
            byte[] modelBytes = modelStream.readAllBytes();

            session = environment.createSession(modelBytes);
            logger.info("session up!");

            // Cache input/output names to avoid repeated lookups
            inputName = session.getInputNames().iterator().next();
            outputName = session.getOutputNames().iterator().next();
            logger.info(
                "Input name: {}, Output name: {}",
                inputName,
                outputName
            );
        }

        // Load scaler parameters from sklearn preprocessing JSON
        try (
            InputStream scalerStream = getClass()
                .getClassLoader()
                .getResourceAsStream("static/sklearn_preprocessing.json");
        ) {
            if (scalerStream == null) {
                throw new RuntimeException(
                    "Scaler params file not found in resources"
                );
            }
            ObjectMapper mapper = new ObjectMapper();

            // Parse the sklearn preprocessing JSON structure
            var jsonNode = mapper.readTree(scalerStream);
            var classifierNode = jsonNode.get("classifier");

            logger.info("Classifier info: " + classifierNode);

            double[] mean = mapper.convertValue(
                classifierNode.get("scaler_mean"),
                double[].class
            );
            double[] scale = mapper.convertValue(
                classifierNode.get("scaler_scale"),
                double[].class
            );

            logger.debug("Scaler mean: " + java.util.Arrays.toString(mean));
            logger.debug("Scaler scale: " + java.util.Arrays.toString(scale));

            this.scalerParams = new ScalerParams(mean, scale);
        }

        logger.info("Scaler parameters loaded: " + scalerParams);
    }

    public float predict(double[] features)
        throws OrtException, IllegalArgumentException {
        if (features.length != 10) {
            throw new IllegalArgumentException(
                "Expected 10 features, got " + features.length
            );
        }

        float[] standardizedFeatures = standardize(features);

        lock.lock();

        try {
            float[][] tensorInput = { standardizedFeatures }; // Wrap in 2D array

            try (
                OnnxTensor inputTensor = OnnxTensor.createTensor(
                    environment,
                    tensorInput
                );
            ) {
                // Run inference with cached input/output names
                Map<String, OnnxTensor> inputs = Collections.singletonMap(
                    inputName,
                    inputTensor
                );

                try (OrtSession.Result result = session.run(inputs)) {
                    // Get output tensor
                    OnnxValue value = result
                        .get(outputName)
                        .orElseThrow(() ->
                            new OrtException("No Prediction Value")
                        );

                    logger.debug(
                        "output_label, class: " +
                        result.get(outputName).get().getValue().getClass()
                    );
                    logger.debug(
                        "output_label, str: " +
                        result.get(outputName).get().getValue().toString()
                    );
                    logger.debug(
                        "output_label, getType: " +
                        result.get(outputName).get().getType()
                    );
                    logger.debug(
                        "output_label, getInfo: " +
                        result.get(outputName).get().getInfo()
                    );

                    long[] output = (long[]) value.getValue();

                    return (float) output[0]; // Return the prediction
                }
            }
        } finally {
            lock.unlock();
        }
    }

    private float[] standardize(double[] features) {
        logger.info(
            "standardize Method, Scaler parameters: " + this.scalerParams
        );

        double[] mean = scalerParams.mean(); // Full precision
        double[] scale = scalerParams.scaler(); // Full precision

        float[] result = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            result[i] = (float) ((features[i] - mean[i]) / scale[i]);
        }
        return result;
    }

    @PreDestroy
    public void cleanup() throws OrtException {
        if (session != null) {
            session.close();
        }
        if (environment != null) {
            environment.close();
        }
    }
}
