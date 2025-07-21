import time
import numpy as np
import tensorflow as tf
import os
from utils import get_logger
from config import NUM_INFERENCE_RUNS, ACCEPTABLE_ACCURACY_DROP_PERCENT # Import ACCEPTABLE_ACCURACY_DROP_PERCENT

logger = get_logger(__name__)

def evaluate_keras_model(model, x_test, y_test, num_classes):
    """
    Evaluates a Keras model for accuracy, inference time, and size.
    Args:
        model (tf.keras.Model): The Keras model to evaluate.
        x_test (np.array): Test input data.
        y_test (np.array): True labels for test data.
        num_classes (int): Number of output classes.
    Returns:
        dict: Dictionary containing 'accuracy', 'inference_time_ms', 'model_size_mb'.
    """
    logger.info("Evaluating Keras model...")

    # Calculate accuracy
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Keras model accuracy: {accuracy:.4f}")

    # Measure inference time
    start_time = time.perf_counter()
    for _ in range(NUM_INFERENCE_RUNS):
        # Predict on a small batch to simulate real-world inference
        _ = model.predict(x_test[0:1], verbose=0)
    end_time = time.perf_counter()
    avg_inference_time_ms = ((end_time - start_time) / NUM_INFERENCE_RUNS) * 1000
    logger.info(f"Keras model average inference time: {avg_inference_time_ms:.3f} ms")

    # Model size will be obtained from the file system in main.py
    model_size_mb = 0 # Placeholder, will be updated in main.py

    return {
        "accuracy": accuracy,
        "inference_time_ms": avg_inference_time_ms,
        "model_size_mb": model_size_mb # This will be filled by main.py
    }

def evaluate_tflite_model(interpreter, x_test, y_test, num_classes):
    """
    Evaluates a TFLite model for accuracy, inference time, and size.
    Args:
        interpreter (tf.lite.Interpreter): The TFLite interpreter to evaluate.
        x_test (np.array): Test input data.
        y_test (np.array): True labels for test data.
        num_classes (int): Number of output classes.
    Returns:
        dict: Dictionary containing 'accuracy', 'inference_time_ms', 'model_size_mb'.
    """
    logger.info("Evaluating TFLite model...")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input tensor type and quantization parameters
    input_dtype = input_details[0]['dtype']
    # Ensure input_scale and input_zero_point are handled correctly if they are empty
    input_scale = input_details[0]['quantization_parameters']['scales'][0] if input_details[0]['quantization_parameters']['scales'].size > 0 else 1.0
    input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0] if input_details[0]['quantization_parameters']['zero_points'].size > 0 else 0


    predictions = []
    inference_times = []

    for i in range(len(x_test)):
        # CRITICAL FIX: Do NOT divide by 255.0 again if x_test is already normalized.
        # x_test from data_handler.py is already normalized to [0, 1].
        input_data_float = x_test[i:i+1].astype(np.float32)

        # Quantize input data if the model expects INT8
        if input_dtype == np.int8 or input_dtype == np.uint8:
            # Apply quantization formula: quantized_value = (float_value / scale) + zero_point
            input_data_quantized = (input_data_float / input_scale) + input_zero_point
            input_data = np.round(input_data_quantized).astype(input_dtype)
        else:
            input_data = input_data_float # Use float32 if model expects it

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        inference_times.append((end_time - start_time) * 1000) # in milliseconds

        output_data = interpreter.get_tensor(output_details[0]['index'])

        # If output is quantized, dequantize it to get actual probabilities/logits
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.int8 or output_dtype == np.uint8:
            output_scale = output_details[0]['quantization_parameters']['scales'][0] if output_details[0]['quantization_parameters']['scales'].size > 0 else 1.0
            output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0] if output_details[0]['quantization_parameters']['zero_points'].size > 0 else 0
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale


        predictions.append(np.argmax(output_data))

    avg_inference_time_ms = np.mean(inference_times)
    logger.info(f"TFLite model average inference time: {avg_inference_time_ms:.3f} ms")

    # Calculate accuracy
    correct_predictions = np.sum(np.array(predictions) == y_test[:len(predictions)])
    accuracy = correct_predictions / len(predictions)
    logger.info(f"TFLite model accuracy: {accuracy:.4f}")

    # Model size will be obtained from the file system in main.py
    model_size_mb = 0 # Placeholder

    return {
        "accuracy": accuracy,
        "inference_time_ms": avg_inference_time_ms,
        "model_size_mb": model_size_mb # This will be filled by main.py
    }

def get_file_size_mb(file_path):
    """Returns the size of a file in MB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0
