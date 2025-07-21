# inference_optimizer/model_loader.py

import tensorflow as tf
import os
import torch
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig # Added AutoConfig
import numpy as np
import tf2onnx
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

def load_pytorch_model(model_path, model_class=AutoModelForSeq2SeqLM, tokenizer_name=None):
    try:
        if os.path.exists(model_path) and os.path.isdir(model_path):
            logger.info(f"Loading PyTorch model from local directory (save_pretrained): {model_path}")
            model = model_class.from_pretrained(model_path)
            
        elif tokenizer_name:
            logger.info(f"Loading PyTorch model '{model_path}' from Hugging Face hub (using {model_class.__name__})...")
            model = model_class.from_pretrained(model_path)
        else:
            raise ValueError("model path is not a directory and tokenizer_name is not provided for Hugging Face hub loading")
            
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading PyTorch model from {model_path}: {e}")
        raise 
    
    
def convert_pytorch_to_onnx(pytorch_model, onnx_path, input_shape, tokenizer_name, max_input_length):
    logger.info(f"Converting PyTorch model to ONNX: {onnx_path}...")
    try:
        dummy_input_ids = torch.randint(0,1000,input_shape, dtype=torch.long)
        dummy_attention_mask = torch.ones(input_shape, dtype=torch.long)
        
        dummy_inputs = (dummy_input_ids,dummy_attention_mask)
        input_names = ["input_ids", "attention_mask"]
        output_names = ["output"]
        
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        }
        
        os.makedirs(os.path.dirname(onnx_path, exist_ok=True))
        
        torch.onnx.export(
            pytorch_model,
            dummy_inputs,
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13, # Choose a compatible opset version
            do_constant_folding=True,
       )
        logger.info(f"PyTorch model successfully converted to ONNX and saved to: {onnx_path}")
        return onnx_path
    except Exception as e:
        logger.error(f"Error converting PyTorch model to ONNX: {e}")
        return None
    
def convert_keras_to_onnx(keras_model, onnx_path, input_signature=None):
    logger.info(f"Converting Keras model to ONNX: {onnx_path}...")
    try:
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        model_proto, _ = tf2onnx.convert.from_keras(
            keras_model,
            input_signature=input_signature,
            output_path=onnx_path
        )
        logger.info(f"Keras model successfully converted to ONNX and saved to: {onnx_path}")
        return onnx_path
    except Exception as e:
        logger.error(f"Error converting Keras model to ONNX: {e}")
        return None

def load_onnx_model(onnx_path):
    """
    Loads an ONNX model using ONNX Runtime.
    Args:
        onnx_path (str): Path to the ONNX model file (.onnx).
    Returns:
        onnxruntime.InferenceSession: ONNX Runtime inference session.
    """
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model file not found at: {onnx_path}")
        raise FileNotFoundError(f"ONNX model file not found at: {onnx_path}")

    try:
        session = ort.InferenceSession(onnx_path)
        logger.info(f"ONNX model loaded successfully from: {onnx_path}")
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX model from {onnx_path}: {e}")
        raise
