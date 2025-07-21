import os
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np

from config import MODEL_CONFIGS, MODELS_DIR, OUTPUT_DIR, REPORTS_DIR, NUM_INFERENCE_RUNS, ACCEPTABLE_ACCURACY_DROP_PERCENT
from utils import get_logger, create_simple_cnn_model, representative_dataset_gen
from data_handler import load_mnist_data
from model_loader import load_keras_model, load_tflite_model
from evaluator import evaluate_keras_model, evaluate_tflite_model
from optimizers.quantization import apply_post_training_quantization
from reporter import generate_report

logger = get_logger(__name__)

def train_and_save_mnist_model(model_name="mnist_cnn"):
    """
    Trains a simple CNN model on MNIST and saves it.
    This is for initial setup/testing, typically users will provide their own models.
    """
    model_config = MODEL_CONFIGS.get(model_name)
    if not model_config:
        logger.error(f"Model configuration for '{model_name}' not found.")
        return None, None, None

    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    model = create_simple_cnn_model(model_config["input_shape"], model_config["num_classes"])
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    logger.info("Training Keras model...")
    model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=0) # Set verbose to 0 for cleaner logs during training

    model_path = model_config["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    logger.info(f"Keras model saved to: {model_path}")
    return model_path, x_test, y_test

def get_file_size_mb(file_path):
    """Returns the size of a file in MB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def optimize_model(model_name="mnist_cnn", optimization_type="quantization"):
    """
    Main function to orchestrate the model optimization process.
    Args:
        model_name (str): The name of the model to optimize as defined in config.py.
        optimization_type (str): The type of optimization to apply (e.g., "quantization").
    """
    logger.info(f"Starting optimization for model: {model_name} using {optimization_type}...")

    model_config = MODEL_CONFIGS.get(model_name)
    if not model_config:
        logger.error(f"Model configuration for '{model_name}' not found. Exiting.")
        return

    # 1. Load Data
    _, (x_test, y_test) = load_mnist_data()

    # 2. Load Original Keras Model
    original_keras_model_path = model_config["model_path"]
    if not os.path.exists(original_keras_model_path):
        logger.warning(f"Original Keras model not found at {original_keras_model_path}. Training a new one...")
        original_keras_model_path, x_test, y_test = train_and_save_mnist_model(model_name)
        if not original_keras_model_path:
            logger.error("Failed to train and save original model. Exiting.")
            return

    original_keras_model = load_keras_model(original_keras_model_path)
    if original_keras_model is None:
        logger.error("Failed to load original Keras model. Exiting.")
        return

    # 3. Evaluate Original Model
    original_model_size_mb = get_file_size_mb(original_keras_model_path)
    original_metrics = evaluate_keras_model(original_keras_model, x_test, y_test, model_config["num_classes"])
    original_metrics["model_size_mb"] = original_model_size_mb
    logger.info(f"Original Model Metrics: {original_metrics}")

    optimized_model_path = None
    optimized_metrics = {}

    # 4. Apply Optimization
    if optimization_type == "quantization":
        # Use a subset of x_test for calibration (e.g., first 100 samples)
        x_test_for_calibration = x_test[:100]
        optimized_model_path = apply_post_training_quantization(
            original_keras_model,
            x_test_for_calibration,
            model_config["quantized_model_path"]
        )
        if optimized_model_path:
            # 5. Evaluate Optimized Model (TFLite)
            optimized_tflite_interpreter = load_tflite_model(optimized_model_path)
            if optimized_tflite_interpreter:
                optimized_model_size_mb = get_file_size_mb(optimized_model_path)
                optimized_metrics = evaluate_tflite_model(optimized_tflite_interpreter, x_test, y_test, model_config["num_classes"])
                optimized_metrics["model_size_mb"] = optimized_model_size_mb
                logger.info(f"Optimized Model Metrics: {optimized_metrics}")
            else:
                logger.error("Failed to load optimized TFLite model for evaluation.")
                return
        else:
            logger.error("Quantization failed. Exiting.")
            return
    else:
        logger.error(f"Unsupported optimization type: {optimization_type}")
        return

    # 6. Generate Report
    if optimized_model_path and optimized_metrics:
        generate_report(original_metrics, optimized_metrics, model_name, optimization_type)
        logger.info("Optimization process completed successfully!")
    else:
        logger.error("Optimization or evaluation failed, report not generated.")

if __name__ == "__main__":
    # Example usage:
    # Run this script to train a model (if not exists), optimize it, and generate a report.
    optimize_model(model_name="mnist_cnn", optimization_type="quantization")