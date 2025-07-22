# inference_optimizer/data_handler.py

import tensorflow as tf
import numpy as np

from utils import get_logger
from transformers import AutoTokenizer 
from config import MODEL_CONFIGS 


logger = get_logger(__name__)

def load_mnist_data():
    """
    Loads and preprocesses the MNIST dataset.
    """
    logger.info("Placeholder: Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Implement actual data loading and preprocessing here
    # For template, return dummy data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def load_summarizer_data(model_name="text_summarizer"):
    model_config = MODEL_CONFIGS.get(model_name)
    if not model_config:
        logger.error(f"Model configuration for '{model_name}' not found in config.py.")
        return (None, None), (None, None)
    tokenizer_name = model_config.get("tokenizer_name","t5-small")
    max_input_length = model_config.get("max_input_length",512)
    max_output_length = model_config.get("max_output_length",150)
    logger.info(f"Loading summarizer data using tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,use_fast=True)
    
    sample_texts = []
    
    sample_summaries = []
    
    inputs = tokenizer(sample_texts,
                       max_length = max_input_length,
                       padding = "max_length",
                       truncation = True,
                       return_tensors = "np")
    
    x_test_input_ids =  inputs["input_ids"]
    x_test_attention_mask = inputs["attention_mask"]
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(sample_summaries,
                           max_length = max_output_length,
                           padding = "max_length",
                           truncation = True,
                           return_tensors = "np")
        
    y_test_labels = labels["input_ids"]
    logger.info(f"Summarizer input data loaded. x_test_input_ids shape: {x_test_input_ids.shape}")
    logger.info(f"Summarizer reference data loaded. y_test_labels shape: {y_test_labels.shape}")


    # Return input_ids and attention_mask as a list, and actual labels
    return (None, None), ([x_test_input_ids, x_test_attention_mask], y_test_labels)
