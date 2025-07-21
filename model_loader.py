# inference_optimizer/model_loader.py

import tensorflow as tf
import os
from utils import get_logger

logger = get_logger(__name__)

def load_keras_model(model_path):
    """
    Loads a Keras model from a specified path.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Keras model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading Keras model from {model_path}: {e}")
        raise
    
def load_tflite_model(model_path):
    """
    Loads a TFLite model from a specified path.
    """
    if not os.path.exists(model_path):
        logger.error(f"TFLite model file not found at: {model_path}")
        raise FileNotFoundError(f"TFLite model file not found at: {model_path}")

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        logger.info(f"TFLite model loaded successfully from: {model_path}")
        return interpreter
    except Exception as e:
        logger.error(f"Error loading TFLite model from {model_path}: {e}")
        raise

# We can add functions for other frameworks (PyTorch, ONNX) as per our requiremenets.
