# inference_optimizer/config.py

import os

# Base directory for models and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'optimized_models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure directories exist (will be created by template.py, but good to have here)
# os.makedirs(MODELS_DIR, exist_ok=True)
# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(REPORTS_DIR, exist_ok=True)

# Evaluation settings
NUM_INFERENCE_RUNS = 100
ACCEPTABLE_ACCURACY_DROP_PERCENT = 1.0

# Model specific configurations (extend as needed)
MODEL_CONFIGS = {
    "mnist_cnn": {
        "input_shape": (28, 28, 1),
        "num_classes": 10,
        "framework": "tensorflow",
        "model_path": os.path.join(MODELS_DIR, "mnist_cnn_model.h5"),
        "quantized_model_path": os.path.join(OUTPUT_DIR, "mnist_cnn_quantized.tflite")
    }
}

# Supported optimization types
OPTIMIZATION_TYPES = ["quantization"] # Add "pruning", "distillation" later
