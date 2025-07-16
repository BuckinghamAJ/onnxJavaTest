package com.method.onnxTest.dto.request;

import jakarta.validation.constraints.Size;

public record PredictionRequest(
    @Size(
        min = 10,
        max = 10,
        message = "Features array must have exactly 10 elements"
    )
    double[] features
) {}
