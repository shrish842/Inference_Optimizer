
# Inference Optimization Report: mnist_cnn

## Optimization Type: Quantization

**Date:** 2025-07-21 19:48:43

---

## Original Model Metrics:

* **Accuracy:** 0.9887
* **Average Inference Time:** 37.919 ms
* **Model Size:** 0.43 MB

---

## Optimized Model Metrics:

* **Accuracy:** 0.9885
* **Average Inference Time:** 0.096 ms
* **Model Size:** 0.04 MB

---

## Optimization Summary:

* **Accuracy Change:** -0.0002 (Absolute change)
    * **Status:** ⚠️ Slight drop (within acceptable limit)
* **Inference Time Reduction:** 37.823 ms (99.75% reduction)
* **Model Size Reduction:** 0.39 MB (90.92% reduction)

---

### Recommendations:

* Review the accuracy change. If it's outside the acceptable threshold, consider different optimization parameters or techniques.
* The optimized model offers significant improvements in inference speed and size.
* Further optimizations (e.g., pruning, knowledge distillation) could be explored for more gains.
