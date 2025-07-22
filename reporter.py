# inference_optimizer/reporter.py

import os
import datetime
from utils import get_logger
from config import REPORTS_DIR,ACCEPTABLE_ACCURACY_DROP_PERCENT,MODEL_CONFIGS

logger = get_logger(__name__)

def generate_report(original_metrics, optimized_metrics, model_name, optimization_type):
    """
    Generates a markdown report comparing original and optimized model metrics.
    Args:
        original_metrics (dict): Metrics for the original model.
        optimized_metrics (dict): Metrics for the optimized model.
        model_name (str): Name of the model.
        optimization_type (str): Type of optimization applied.
    Returns:
        str: Path to the generated report file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{model_name}_{optimization_type}_report_{timestamp}.md"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    model_config = MODEL_CONFIGS.get(model_name)
    is_summarization_model = model_config.get("num_classes") is None
    
    accuracy_change = optimized_metrics["accuracy"] - original_metrics["accuracy"]
    inference_time_change = original_metrics["inference_time_ms"] - optimized_metrics["inference_time_ms"]
    model_size_change = original_metrics["model_size_mb"] - optimized_metrics["model_size_mb"]

    # Determine accuracy status based on ACCEPTABLE_ACCURACY_DROP_PERCENT
    original_accuracy = original_metrics["accuracy"]
    acceptable_drop = original_accuracy * (ACCEPTABLE_ACCURACY_DROP_PERCENT / 100.0)
    accuracy_status_emoji = '✅'
    accuracy_status_text = 'No significant drop'
    if accuracy_change < -acceptable_drop: # If accuracy dropped more than acceptable_drop
        accuracy_status_emoji = '❌'
        accuracy_status_text = 'Significant drop!'
    elif accuracy_change < 0: # If accuracy dropped but within acceptable limits
        accuracy_status_emoji = '⚠️'
        accuracy_status_text = 'Slight drop (within acceptable limit)'


    report_content = f"""
# Inference Optimization Report: {model_name}

## Optimization Type: {optimization_type.replace('_', ' ').title()}

**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Original Model Metrics:

"""
    if is_summarization_model and original_metrics.get("rouge_scores"):
        report_content += f"""
* **ROUGE-1 F-measure:** {original_metrics["rouge_scores"]["rouge1"]:.4f}
* **ROUGE-2 F-measure:** {original_metrics["rouge_scores"]["rouge2"]:.4f}
* **ROUGE-L F-measure:** {original_metrics["rouge_scores"]["rougeL"]:.4f}
"""
    else:
        report_content += f"""
* **Accuracy:** {original_metrics["accuracy"]:.4f}
"""
    report_content += f"""
* **Average Inference Time:** {original_metrics["inference_time_ms"]:.3f} ms
* **Model Size:** {original_metrics["model_size_mb"]:.2f} MB

---

## Optimized Model Metrics:

"""
    if is_summarization_model and optimized_metrics.get("rouge_scores"):
        report_content += f"""
* **ROUGE-1 F-measure:** {optimized_metrics["rouge_scores"]["rouge1"]:.4f}
* **ROUGE-2 F-measure:** {optimized_metrics["rouge_scores"]["rouge2"]:.4f}
* **ROUGE-L F-measure:** {optimized_metrics["rouge_scores"]["rougeL"]:.4f}
"""
    else:
        report_content += f"""
* **Accuracy:** {optimized_metrics["accuracy"]:.4f}
"""
    report_content += f"""
* **Average Inference Time:** {optimized_metrics["inference_time_ms"]:.3f} ms
* **Model Size:** {optimized_metrics["model_size_mb"]:.2f} MB

---

## Optimization Summary:

* **Primary Metric Change ({'ROUGE-L' if is_summarization_model else 'Accuracy'}):** {accuracy_change:.4f} (Absolute change)
    * **Status:** {accuracy_status_emoji} {accuracy_status_text}
* **Inference Time Reduction:** {inference_time_change:.3f} ms ({((inference_time_change / original_metrics["inference_time_ms"]) * 100):.2f}% reduction)
* **Model Size Reduction:** {model_size_change:.2f} MB ({((model_size_change / original_metrics["model_size_mb"]) * 100):.2f}% reduction)

---

### Recommendations:

* Review the primary metric change. If it's outside the acceptable threshold, consider different optimization parameters or techniques.
* The optimized model offers significant improvements in inference speed and size.
* Further optimizations (e.g., pruning, more advanced quantization techniques, optimized generation pipelines) could be explored for more gains.
"""
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Report generated and saved to: {report_path}")
    return report_path













