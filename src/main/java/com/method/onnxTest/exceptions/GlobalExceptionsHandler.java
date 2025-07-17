package com.method.onnxTest.exceptions;

import ai.onnxruntime.OrtException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

@ControllerAdvice
public class GlobalExceptionsHandler {

    private static final Logger logger = LoggerFactory.getLogger(
        GlobalExceptionsHandler.class
    );

    @ExceptionHandler(OrtException.class)
    public ResponseEntity<ErrorResponse> handleORTException(OrtException ex) {
        logger.error("ORT Exception occurred: ", ex);
        ErrorResponse errorResponse = new ErrorResponse(
            "MODEL_ERROR",
            ex.getMessage(),
            System.currentTimeMillis()
        );
        return new ResponseEntity<>(
            errorResponse,
            HttpStatus.INTERNAL_SERVER_ERROR
        );
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(Exception ex) {
        logger.error("Unexpected error occurred: ", ex);
        ErrorResponse errorResponse = new ErrorResponse(
            "INTERNAL_ERROR",
            "An unexpected error occurred",
            System.currentTimeMillis()
        );
        return new ResponseEntity<>(
            errorResponse,
            HttpStatus.INTERNAL_SERVER_ERROR
        );
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgumentException(
        IllegalArgumentException ex
    ) {
        logger.warn("Invalid argument: ", ex);
        ErrorResponse errorResponse = new ErrorResponse(
            "INVALID_ARGUMENT",
            ex.getMessage(),
            System.currentTimeMillis()
        );
        return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ErrorResponse> handleMessageNotReadableException(
        HttpMessageNotReadableException ex
    ) {
        logger.warn("Invalid Request Provided: ", ex);
        ErrorResponse errorResponse = new ErrorResponse(
            "INVALID_REQUEST",
            ex.getMessage(),
            System.currentTimeMillis()
        );
        return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleMessageNotReadableException(
        MethodArgumentNotValidException ex
    ) {
        logger.warn(
            "MethodArgumentNotValidException: Invalid Request Provided"
        );

        // Collect all validation errors
        List<String> errors = ex
            .getBindingResult()
            .getFieldErrors()
            .stream()
            .map(fieldError -> fieldError.getDefaultMessage())
            .collect(Collectors.toList());

        logger.info("Errors: " + errors);

        ErrorResponse errorResponse = new ErrorResponse(
            "INVALID_REQUEST",
            System.currentTimeMillis(),
            errors
        );

        return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
    }

    public static record ErrorResponse(
        String errorCode,
        String message,
        long timestamp,
        List<String> errors
    ) {
        public ErrorResponse(String errorCode, String message, long timestamp) {
            this(errorCode, message, timestamp, null);
        }

        public ErrorResponse(
            String errorCode,
            long timestamp,
            List<String> errors
        ) {
            this(errorCode, null, timestamp, errors);
        }
    }
}
