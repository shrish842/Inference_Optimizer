import os
import sys

# Base directory for models and data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'optimized_models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Evaluation settings
NUM_INFERENCE_RUNS = 100  # Number of runs to average for latency
ACCEPTABLE_ACCURACY_DROP_PERCENT = 5.0  # Max acceptable accuracy drop (e.g., 5% of original ROUGE-L)

# --- Model specific configurations ---
MODEL_CONFIGS = {
    "mnist_cnn": {
        "model_path": os.path.join(MODELS_DIR, "mnist_cnn_model.h5"),
        "framework": "tensorflow",
        "input_shape": (1, 28, 28, 1), # Input shape for ONNX conversion (batch_size, H, W, C)
        "num_classes": 10,
        "data_loader_function": "load_mnist_data",
        "onnx_path": os.path.join(OUTPUT_DIR, "mnist_cnn_original.onnx"),
        "quantized_onnx_path": os.path.join(OUTPUT_DIR, "mnist_cnn_quantized.onnx")
    },
    "text_summarizer": {
        # IMPORTANT: This should be the path to the directory saved by model.save_pretrained()
        "model_path": os.path.join(MODELS_DIR, "pegasus-samsum-model"),
        "framework": "pytorch",
        "input_shape": (1, 512), # Common input shape for token IDs (batch_size, sequence_length)
                                 # Verify your model's actual max input length
        "num_classes": None, # Summarization is not a classification task
        "data_loader_function": "load_summarizer_data",
        "onnx_path": os.path.join(OUTPUT_DIR, "text_summarizer_original.onnx"),
        "quantized_onnx_path": os.path.join(OUTPUT_DIR, "text_summarizer_quantized.onnx"),
        # IMPORTANT: This should be the name/path of the tokenizer used in your project.
        # If you saved it locally with tokenizer.save_pretrained(), use that path:
        # os.path.join(MODELS_DIR, "tokenizer")
        # Otherwise, use the Hugging Face model ID:
        "tokenizer_name": "google/pegasus-samsum", # Example: Replace with your actual tokenizer name/path
        "max_input_length": 512, # Max length for input text (source)
        "max_output_length": 150 # Max length for generated summary (target)
    }
}

# Supported optimization types
OPTIMIZATION_TYPES = ["quantization"] # Add "pruning", "distillation" later
