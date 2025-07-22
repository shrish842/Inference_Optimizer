import time
import numpy as np
import tensorflow as tf
import os
import onnxruntime as ort
from utils import get_logger
from config import NUM_INFERENCE_RUNS, ACCEPTABLE_ACCURACY_DROP_PERCENT, MODEL_CONFIGS # Import MODEL_CONFIGS
from transformers import AutoTokenizer 
from rogue_score import rogue_scorer

logger = get_logger(__name__)


def greedy_decode_onnx_tflite(model_session_or_interpreter, tokenizer, input_ids, attention_mask , max_output_length, model_type="onnx"):
    batch_size = input_ids.shape[0]
    decoder_input_ids = np.full((batch_size, 1), tokenizer.pad_token_id, dtype = np.int64)
    
    if hasattr(tokenizer, 'decoder_start_token_id') and tokenizer.decoder_start_token_id is not None:
        decoder_start_token_id = tokenizer.decoder_start_token_id
    elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        decoder_start_token_id = tokenizer.bos_token_id
    else:
        logger.warning("Could not find decoder_start_token_id or bos_token_id in tokenizer. Using pad_token_id as start.")
        decoder_start_token_id = tokenizer.pad_token_id

    decoder_input_ids = np.full((batch_size, 1), decoder_start_token_id, dtype=np.int64)
    
    generated_ids = []
    for _ in range(max_output_length):
        if model_type == "onnx":
            # ONNX Runtime expects input names as a dictionary
            input_feed = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids # Pass decoder input for current step
            }
            # Assuming output is logits (batch_size, 1, vocab_size)
            outputs = model_session_or_interpreter.run(None, input_feed)
            logits = outputs[0] # Get logits from the first output
        elif model_type == "tflite":
            # TFLite expects inputs to be set by index
            input_details = model_session_or_interpreter.get_input_details()
            output_details = model_session_or_interpreter.get_output_details()

            # Map input names to indices for clarity
            input_map = {detail['name']: detail['index'] for detail in input_details}
            input_ids_detail = next(d for d in input_details if d['name'] == 'input_ids')
            attention_mask_detail = next(d for d in input_details if d['name'] == 'attention_mask')
            decoder_input_ids_detail = next(d for d in input_details if d['name'] == 'decoder_input_ids')
            
            
            def prepare_tflite_input(data_array, detail):
                data_float = data_array.astype(np.float32)
                input_dtype = detail['dtype']
                input_scale = detail['quantization_parameters']['scales'][0] if detail['quantization_parameters']['scales'].size > 0 else 1.0
                input_zero_point = detail['quantization_parameters']['zero_points'][0] if detail['quantization_parameters']['zero_points'].size > 0 else 0
                
                if input_dtype == np.int8 or input_dtype==np.uint8:
                    quantized_data = (data_float/input_scale) + input_zero_point
                    return np.round(quantized_data).astype(input_dtype)
                
                else:
                    return data_array.astype(input_dtype)
                
            model_session_or_interpreter.set_tensor(input_ids_detail['index'], prepare_tflite_input(input_ids, input_ids_detail))
            model_session_or_interpreter.set_tensor(attention_mask_detail['index'], prepare_tflite_input(attention_mask, attention_mask_detail))
            model_session_or_interpreter.set_tensor(decoder_input_ids_detail['index'], prepare_tflite_input(decoder_input_ids, decoder_input_ids_detail))


            model_session_or_interpreter.invoke()
            logits = model_session_or_interpreter.get_tensor(output_details[0]['index'])
            output_dtype = output_details[0]['dtype']
            if output_dtype == np.int8 or output_dtype == np.uint8:
                output_scale = output_details[0]['quantization_parameters']['scales'][0] if output_details[0]['quantization_parameters']['scales'].size > 0 else 1.0
                output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0] if output_details[0]['quantization_parameters']['zero_points'].size > 0 else 0
                logits = (logits.astype(np.float32) - output_zero_point) * output_scale

        next_token_logits = logits[:,-1,:]
        next_token_ids = np.argmax(next_token_logits, axis=-1)
        
        generated_ids.append(next_token_ids)
        
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_ids[:, None]], axis=-1)

        if tokenizer.eos_token_id is not None and (next_token_ids == tokenizer.eos_token_id).all():
            break
    generated_ids = np.stack(generated_ids, axis=1)

    return generated_ids




def evaluate_keras_model(model, x_test_tuple, y_test_tuple, model_name):
    logger.info("Evaluating Keras model...")

    model_config = MODEL_CONFIGS.get(model_name)
    is_summarization_model = model_config.get("num_classes") is None

    if is_summarization_model:
        logger.info("Performing text generation for summarization model (Keras)...")
        tokenizer_name = model_config.get("tokenizer_name")
        max_input_length = model_config.get("max_input_length")
        max_output_length = model_config.get("max_output_length")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        input_ids = x_test_tuple[0][0] # Assuming input_ids is the first element
        attention_mask = x_test_tuple[0][1] # Assuming attention_mask is the second

        generated_texts = []
        inference_times = []
        
        logger.warning("Keras model generation for summarization is simplified. For full Hugging Face Keras models, use model.generate().")
        # Simplified greedy decode for generic Keras model
        for i in range(input_ids.shape[0]):
            current_input_ids = input_ids[i:i+1]
            current_attention_mask = attention_mask[i:i+1]
            # Initialize decoder_input_ids
            decoder_start_token_id = tokenizer.pad_token_id # Fallback
            if hasattr(tokenizer, 'decoder_start_token_id') and tokenizer.decoder_start_token_id is not None:
                decoder_start_token_id = tokenizer.decoder_start_token_id
            elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
                decoder_start_token_id = tokenizer.bos_token_id
            current_decoder_input_ids = np.full((1, 1), decoder_start_token_id, dtype=np.int64)

            generated_token_ids = []
            start_time = time.perf_counter()
            for _ in range(max_output_length):
                # Keras models often take inputs as a list for multiple inputs
                outputs = model.predict([current_input_ids, current_attention_mask, current_decoder_input_ids], verbose=0)
                logits = outputs[0] # Assuming first output is logits
                next_token_logits = logits[:, -1, :]
                next_token_id = np.argmax(next_token_logits, axis=-1)
                generated_token_ids.append(next_token_id[0]) # Get scalar ID

                if next_token_id[0] == tokenizer.eos_token_id:
                    break
                current_decoder_input_ids = np.concatenate([current_decoder_input_ids, next_token_id[:, None]], axis=-1)
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)
            generated_texts.append(tokenizer.decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

        avg_inference_time_ms = np.mean(inference_times)
        logger.info(f"Keras model average inference time (generation): {avg_inference_time_ms:.3f} ms")

        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        num_samples = min(len(generated_texts), len(y_test_tuple)) # Use min to avoid index error

        # Decode reference summaries for ROUGE evaluation
        reference_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in y_test_tuple[:num_samples]]


        for i in range(num_samples):
            scores = scorer.score(reference_texts[i], generated_texts[i])
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure

        for key in rouge_scores:
            rouge_scores[key] /= num_samples

        logger.info(f"Keras model ROUGE scores: {rouge_scores}")
        accuracy_metric = rouge_scores['rougeL'] # Use ROUGE-L as the primary "accuracy" for reporting

    else: # Classification model (like MNIST)
        loss, accuracy = model.evaluate(x_test_tuple, y_test_tuple, verbose=0)
        logger.info(f"Keras model accuracy: {accuracy:.4f}")
        accuracy_metric = accuracy

        start_time = time.perf_counter()
        for _ in range(NUM_INFERENCE_RUNS):
            if isinstance(x_test_tuple, list) or isinstance(x_test_tuple, tuple):
                sample_input = [inp[0:1] for inp in x_test_tuple[0]] # Get first sample from input list
            else:
                sample_input = x_test_tuple[0:1]
            _ = model.predict(sample_input, verbose=0)
        end_time = time.perf_counter()
        avg_inference_time_ms = ((end_time - start_time) / NUM_INFERENCE_RUNS) * 1000
        logger.info(f"Keras model average inference time: {avg_inference_time_ms:.3f} ms")


    model_size_mb = 0 # Placeholder, will be updated in main.py

    return {
        "accuracy": accuracy_metric, # This will be ROUGE-L for summarization
        "inference_time_ms": avg_inference_time_ms,
        "model_size_mb": model_size_mb,
        "rouge_scores": rouge_scores if is_summarization_model else None # Include full ROUGE scores
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


def evaluate_onnx_model(session, x_test_tuple, y_test_actual, model_name):
    """
    Evaluates an ONNX model for accuracy (for classification) or generates text (for summarization).
    Args:
        session (onnxruntime.InferenceSession): The ONNX Runtime inference session.
        x_test_tuple (tuple): A tuple where the first element is a list/tuple of input arrays
                              (e.g., [input_ids, attention_mask]) and the second is y_test.
        y_test_actual (np.array or list of str): True labels for classification or reference summaries for summarization.
        model_name (str): Name of the model from config.py.
    Returns:
        dict: Dictionary containing 'accuracy' (or ROUGE scores), 'inference_time_ms', 'model_size_mb'.
    """
    logger.info("Evaluating ONNX model...")

    model_config = MODEL_CONFIGS.get(model_name)
    is_summarization_model = model_config.get("num_classes") is None

    x_test_inputs = x_test_tuple[0] # This is the list of input arrays (input_ids, attention_mask)

    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    predictions = []
    inference_times = []

    if is_summarization_model:
        logger.info("Performing text generation for summarization model (ONNX)...")
        tokenizer_name = model_config.get("tokenizer_name")
        max_output_length = model_config.get("max_output_length")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        input_ids = x_test_inputs[0]
        attention_mask = x_test_inputs[1]

        generated_texts = []
        num_samples = input_ids.shape[0]

        for i in range(num_samples):
            current_input_ids = input_ids[i:i+1]
            current_attention_mask = attention_mask[i:i+1]

            start_time = time.perf_counter()
            # Perform greedy decoding using the ONNX session
            generated_token_ids = greedy_decode_onnx_tflite(
                session,
                tokenizer,
                current_input_ids,
                current_attention_mask,
                max_output_length,
                model_type="onnx"
            )
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)

            # Decode generated token IDs to text
            generated_texts.append(tokenizer.decode(generated_token_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

        avg_inference_time_ms = np.mean(inference_times)
        logger.info(f"ONNX model average inference time (generation): {avg_inference_time_ms:.3f} ms")

        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        num_samples = min(len(generated_texts), len(y_test_actual))

        reference_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in y_test_actual[:num_samples]]

        for i in range(num_samples):
            scores = scorer.score(reference_texts[i], generated_texts[i])
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure

        for key in rouge_scores:
            rouge_scores[key] /= num_samples

        logger.info(f"ONNX model ROUGE scores: {rouge_scores}")
        accuracy_metric = rouge_scores['rougeL'] # Use ROUGE-L as the primary "accuracy" for reporting

    else: # Classification model
        num_samples = x_test_inputs[0].shape[0] if isinstance(x_test_inputs, (list, tuple)) else x_test_inputs.shape[0]
        for i in range(num_samples):
            input_feed = {}
            if isinstance(x_test_inputs, (list, tuple)):
                for j, name in enumerate(input_names):
                    input_feed[name] = x_test_inputs[j][i:i+1]
            else:
                input_feed[input_names[0]] = x_test_inputs[i:i+1] # Assuming single input for classification

            start_time = time.perf_counter()
            outputs = session.run(output_names, input_feed)
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)

            output_data = outputs[0]
            predictions.append(np.argmax(output_data))

        avg_inference_time_ms = np.mean(inference_times)
        logger.info(f"ONNX model average inference time: {avg_inference_time_ms:.3f} ms")

        correct_predictions = np.sum(np.array(predictions) == y_test_actual[:len(predictions)])
        accuracy_metric = correct_predictions / len(predictions)
        logger.info(f"ONNX model accuracy: {accuracy_metric:.4f}")

    model_size_mb = 0 # Placeholder, will be updated in main.py

    return {
        "accuracy": accuracy_metric,
        "inference_time_ms": avg_inference_time_ms,
        "model_size_mb": model_size_mb,
        "rouge_scores": rouge_scores if is_summarization_model else None
    }

def evaluate_tflite_model(interpreter, x_test_tuple, y_test_actual, model_name):
    """
    Evaluates a TFLite model for accuracy (for classification) or generates text (for summarization).
    Args:
        interpreter (tf.lite.Interpreter): The TFLite interpreter to evaluate.
        x_test_tuple (tuple): A tuple where the first element is a list/tuple of input arrays
                              (e.g., [input_ids, attention_mask]) and the second is y_test.
        y_test_actual (np.array or list of str): True labels for classification or reference summaries for summarization.
        model_name (str): Name of the model from config.py.
    Returns:
        dict: Dictionary containing 'accuracy' (or ROUGE scores), 'inference_time_ms', 'model_size_mb'.
    """
    logger.info("Evaluating TFLite model...")

    model_config = MODEL_CONFIGS.get(model_name)
    is_summarization_model = model_config.get("num_classes") is None

    x_test_inputs = x_test_tuple[0] # This is the list of input arrays (input_ids, attention_mask)

    predictions = []
    inference_times = []

    if is_summarization_model:
        logger.info("Performing text generation for summarization model (TFLite)...")
        tokenizer_name = model_config.get("tokenizer_name")
        max_output_length = model_config.get("max_output_length")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        input_ids = x_test_inputs[0]
        attention_mask = x_test_inputs[1]

        generated_texts = []
        num_samples = input_ids.shape[0]

        for i in range(num_samples):
            current_input_ids = input_ids[i:i+1]
            current_attention_mask = attention_mask[i:i+1]

            start_time = time.perf_counter()
            # Perform greedy decoding using the TFLite interpreter
            generated_token_ids = greedy_decode_onnx_tflite(
                interpreter,
                tokenizer,
                current_input_ids,
                current_attention_mask,
                max_output_length,
                model_type="tflite"
            )
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)

            # Decode generated token IDs to text
            generated_texts.append(tokenizer.decode(generated_token_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

        avg_inference_time_ms = np.mean(inference_times)
        logger.info(f"TFLite model average inference time (generation): {avg_inference_time_ms:.3f} ms")

        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        num_samples = min(len(generated_texts), len(y_test_actual))

        reference_texts = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in y_test_actual[:num_samples]]

        for i in range(num_samples):
            scores = scorer.score(reference_texts[i], generated_texts[i])
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure

        for key in rouge_scores:
            rouge_scores[key] /= num_samples

        logger.info(f"TFLite model ROUGE scores: {rouge_scores}")
        accuracy_metric = rouge_scores['rougeL'] # Use ROUGE-L as the primary "accuracy" for reporting

    else: # Classification model
        num_samples = x_test_inputs[0].shape[0] if isinstance(x_test_inputs, (list, tuple)) else x_test_inputs.shape[0]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_dtype = input_details[0]['dtype']
        input_scale = input_details[0]['quantization_parameters']['scales'][0] if input_details[0]['quantization_parameters']['scales'].size > 0 else 1.0
        input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0] if input_details[0]['quantization_parameters']['zero_points'].size > 0 else 0

        for i in range(num_samples):
            input_data_float = x_test_inputs[i:i+1].astype(np.float32)

            if input_dtype == np.int8 or input_dtype == np.uint8:
                input_data = np.round((input_data_float / input_scale) + input_zero_point).astype(input_dtype)
            else:
                input_data = input_data_float

            interpreter.set_tensor(input_details[0]['index'], input_data)

            start_time = time.perf_counter()
            interpreter.invoke()
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)

            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_dtype = output_details[0]['dtype']
            if output_dtype == np.int8 or output_dtype == np.uint8:
                output_scale = output_details[0]['quantization_parameters']['scales'][0] if output_details[0]['quantization_parameters']['scales'].size > 0 else 1.0
                output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0] if output_details[0]['quantization_parameters']['zero_points'].size > 0 else 0
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

            predictions.append(np.argmax(output_data))

        avg_inference_time_ms = np.mean(inference_times)
        logger.info(f"TFLite model average inference time: {avg_inference_time_ms:.3f} ms")

        correct_predictions = np.sum(np.array(predictions) == y_test_actual[:len(predictions)])
        accuracy_metric = correct_predictions / len(predictions)
        logger.info(f"TFLite model accuracy: {accuracy_metric:.4f}")

    model_size_mb = 0 # Placeholder

    return {
        "accuracy": accuracy_metric,
        "inference_time_ms": avg_inference_time_ms,
        "model_size_mb": model_size_mb,
        "rouge_scores": rouge_scores if is_summarization_model else None
    }




def get_file_size_mb(file_path):
    """Returns the size of a file in MB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0
