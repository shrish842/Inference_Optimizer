# inference_optimizer/optimizers/quantization.py

import tensorflow as tf
import os
from utils import get_logger

logger = get_logger(__name__)

def apply_post_training_quantization(keras_model, x_test_for_calibration, quantized_model_path):
    """
    Applies post-training integer quantization to a Keras model.
    The model is converted to a TFLite model with int8 weights and activations.
    Args:
        keras_model (tf.keras.Model): The Keras model to quantize.
        x_test_for_calibration (np.array): A subset of the test data (e.g., first 100 samples)
                                           used for calibration during quantization.
                                           Should be preprocessed (e.g., normalized).
        quantized_model_path (str): Path to save the quantized TFLite model.
    Returns:
        str: Path to the saved quantized TFLite model, or None if conversion fails.
    """
    logger.info("Applying post-training quantization...")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # This line is crucial for full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # Or tf.uint8 depending on model
    converter.inference_output_type = tf.int8 # Or tf.uint8 depending on model

    # Provide a representative dataset for calibration
    # The generator should yield input tensors in the expected format (e.g., float32)
    # The converter will use this to determine the min/max ranges for quantization.
    def representative_dataset():
        for i in range(x_test_for_calibration.shape[0]):
            # Ensure the data is in the expected format (e.g., float32 and normalized)
            img = x_test_for_calibration[i:i+1].astype("float32")
            yield [img]

    converter.representative_dataset = representative_dataset

    try:
        tflite_quantized_model = converter.convert()
        os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)
        with open(quantized_model_path, 'wb') as f:
            f.write(tflite_quantized_model)
        logger.info(f"Quantized TFLite model saved to: {quantized_model_path}")
        return quantized_model_path
    except Exception as e:
        logger.error(f"Error during TFLite quantization: {e}")
        return None

# Add placeholders for other optimization techniques
# def apply_pruning(...): pass
# def apply_knowledge_distillation(...): pass