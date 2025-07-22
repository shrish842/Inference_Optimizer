import os
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import importlib
import torch
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM

from config import MODEL_CONFIGS, MODELS_DIR, OUTPUT_DIR, REPORTS_DIR, NUM_INFERENCE_RUNS, ACCEPTABLE_ACCURACY_DROP_PERCENT
from utils import get_logger, create_simple_cnn_model, representative_dataset_gen
from data_handler import load_mnist_data, load_summarizer_data
from model_loader import load_keras_model, load_pytorch_model, convert_keras_to_onnx, convert_pytorch_to_onnx, load_onnx_model
from evaluator import evaluate_keras_model, evaluate_onnx_model, evaluate_tflite_model, get_file_size_mb
from optimizers.quantization import apply_onnx_quantization
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

def optimize_model(model_name="mnist_cnn", optimization_type="quantization",quantization_mode = "static"):
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
    
    data_loader_module = importlib.import_module("data_handler")
    data_loader_func = getattr(data_loader_module,model_config["data_loader_function"])
    
    _, (x_test_inputs, y_test_actual) = data_loader_func(model_name=model_name)

    if x_test_inputs is None or y_test_actual is None:
        logger.error(f"Data loading failed for model '{model_name}'. Exiting.")
        return

    original_model = None
    original_onnx_path = model_config.get("onnx_path", os.path.join(OUTPUT_DIR, f"{model_name}_original.onnx"))
    quantized_onnx_path = model_config.get("quantized_onnx_path", os.path.join(OUTPUT_DIR, f"{model_name}_quantized.onnx"))

    if model_config["framework"] == "tensorflow":
        original_keras_model_path = model_config["model_path"]
        if not os.path.exists(original_keras_model_path) and model_name == "mnist_cnn":
            logger.warning(f"Original Keras model not found at {original_keras_model_path}. Training a new one...")
            original_keras_model_path, x_test_temp, y_test_temp = train_and_save_mnist_model(model_name)
            if x_test_temp is not None:
                 x_test_inputs = x_test_temp
                 y_test_actual = y_test_temp
            if not original_keras_model_path:
                logger.error("Failed to train and save original model. Exiting.")
                return
            original_model = load_keras_model(original_keras_model_path)
        elif os.path.exists(original_keras_model_path):
            original_model = load_keras_model(original_keras_model_path)
        else:
            logger.error(f"Original Keras model '{original_keras_model_path}' not found. Please train and save it or update model_config. Exiting.")
            return

        if original_model is None:
            logger.error("Failed to load original Keras model. Exiting.")
            return

        # Convert Keras to ONNX
        original_onnx_path = convert_keras_to_onnx(original_model, original_onnx_path)
        if original_onnx_path is None:
            logger.error("Keras to ONNX conversion failed. Exiting.")
            return

    elif model_config["framework"] == "pytorch":
        original_pytorch_model_path = model_config["model_path"]
        # Pass the specific model class (e.g., AutoModelForSeq2SeqLM)
        original_pytorch_model = load_pytorch_model(original_pytorch_model_path,
                                                    model_class=AutoModelForSeq2SeqLM, # <--- IMPORTANT: Specify your model class
                                                    tokenizer_name=model_config.get("tokenizer_name")) # Used for HF hub loading
        if original_pytorch_model is None:
            logger.error("Failed to load original PyTorch model. Exiting.")
            return

        # Convert PyTorch to ONNX
        original_onnx_path = convert_pytorch_to_onnx(
            original_pytorch_model,
            original_onnx_path,
            model_config["input_shape"],
            model_config["tokenizer_name"],
            model_config["max_input_length"]
        )
        if original_onnx_path is None:
            logger.error("PyTorch to ONNX conversion failed. Exiting.")
            return
    else:
        logger.error(f"Unsupported framework: {model_config['framework']}. Exiting.")
        return

    original_onnx_session = load_onnx_model(original_onnx_path)
    if original_onnx_session is None:
        logger.error("Failed to load original ONNX model for evaluation. Exiting.")
        return

    original_metrics = evaluate_onnx_model(original_onnx_session, (x_test_inputs, y_test_actual), model_name) # Pass model_name
    original_metrics["model_size_mb"] = get_file_size_mb(original_onnx_path)
    logger.info(f"Original ONNX Model Metrics: {original_metrics}")
    
    if optimization_type == "quantization":
        optimized_model_path = apply_onnx_quantization(
            original_onnx_path,
            quantized_onnx_path,
            x_test_inputs, # Pass the actual input data for calibration
            quantization_mode=quantization_mode,
            num_calibration_samples=100 # Adjust as needed
        )
        if optimized_model_path is None:
            logger.error("ONNX Quantization failed. Exiting.")
            return
    else:
        logger.error(f"Unsupported optimization type: {optimization_type}")
        return
    
    optimized_onnx_session = load_onnx_model(optimized_model_path)
    if optimized_onnx_session is None:
        logger.error("Failed to load quantized ONNX model for evaluation. Exiting.")
        return

    optimized_metrics = evaluate_onnx_model(optimized_onnx_session, (x_test_inputs, y_test_actual), model_name) # Pass model_name
    optimized_metrics["model_size_mb"] = get_file_size_mb(optimized_model_path)
    logger.info(f"Optimized ONNX Model Metrics: {optimized_metrics}")

   
    if optimized_model_path and optimized_metrics:
        generate_report(original_metrics, optimized_metrics, model_name, f"onnx_{optimization_type}_{quantization_mode}")
        logger.info("Optimization process completed successfully!")
    else:
        logger.error("Optimization or evaluation failed, report not generated.")
  
  
if __name__ == "__main__":
    optimize_model(model_name="text_summarizer",optimization_type="quantization",quantization_mode="static")  