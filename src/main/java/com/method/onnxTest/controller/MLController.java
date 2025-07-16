package com.method.onnxTest.controller;

import ai.onnxruntime.OrtException;
import com.method.onnxTest.dto.request.PredictionRequest;
import com.method.onnxTest.dto.response.ClassificationEnum;
import com.method.onnxTest.dto.response.PredictionResponse;
import com.method.onnxTest.service.ClassifierModelPrediction;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/ml")
public class MLController {

    @Autowired
    private ClassifierModelPrediction classifierModelPrediction;

    @PostMapping("/classifier")
    public PredictionResponse classifierPrediction(
        @Valid @RequestBody PredictionRequest predictionReq
    ) throws OrtException {
        double prediction = classifierModelPrediction.predict(
            predictionReq.features()
        );
        return new PredictionResponse(
            prediction,
            ClassificationEnum.values()[(int) prediction].name()
        );
    }
}
