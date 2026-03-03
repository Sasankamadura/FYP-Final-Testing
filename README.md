# VisDrone RT-DETR Evaluation Project

This repository contains the evaluation framework, model weights, and datasets for the RT-DETR object detection project focused on the VisDrone 2019 dataset.

## Directory Structure

- **`eval/`**: The core directory containing all the validation, testing, and benchmarking scripts.
  - `config.yaml`: The master configuration file where all datasets, model paths, and evaluation metrics are registered. It also categorizes models as either `experiments` or `final`.
  - `run_validation.py`: Evaluates models on the validation set. Computes standard COCO metrics (mAP, mAR) and raw precision/recall curves, separating outputs based on model category.
  - `run_test_inference.py`: Generates predictions on the test set and exports visualization images, isolated by model category.
  - `run_benchmark.py`: Benchmarks model latency and throughput (FPS), saving results to `experiments` or `final` outputs.
  - `baseline_class_analysis.py`: Extracts per-class metrics and generates visualization heatmaps comparing models to the baseline.
  - `generate_report.py`: Aggregates the results into grouped PDF/CSV reports and plots Precision-Recall curves for crucial classes.
  - `requirements.txt`: Python package dependencies.
  - `results/`: Automatically generated parent folder where categorized prediction JSON files, PR data, and visualizations are saved.
- **`Experiments/`**: Contains exported ONNX model checkpoints for the baseline RT-DETR and various architectural trials (9-class trained models).
- **`Final Models/`**: A designated folder to hold the chosen production-ready models (10-class trained models).
- **`VisDrone Val image set/`**: Contains the validation images and the `annotations_VisDrone_val.json` ground-truth file.
- **`VisDrone test image set/`**: Contains the test images without ground-truth annotations (for inference only).
- **Notebooks**:
  - `VisDrone_RTDETR_Colab.ipynb`: Jupyter notebook configured for model training/experimentation on Google Colab.
  - `VisDrone_RTDETR_Kaggle.ipynb`: Jupyter notebook configured for model training/experimentation on Kaggle.

## Getting Started

### 1. Requirements Installation

To run the tools in the `eval/` directory, ensure you have a Python environment set up with the necessary dependencies:

```bash
cd eval
pip install -r requirements.txt
```

*(You may also need `onnxruntime-gpu` if you intend to run inference leveraging an NVIDIA GPU.)*

### 2. Configuration (`eval/config.yaml`)

The `eval/config.yaml` is the single source of truth for the project. 

- Automatically maps internal aliases (e.g., `baseline_rtdetr_r18`, `p2_layer`) to the corresponding ONNX files in the `Experiments/` directory.
- Controls NMS thresholds, confidence thresholds, input sizes, and custom metrics.
- Should a dataset directory or a model's path change, update this file exclusively.

## Running Evaluation Scripts

All Python scripts in the `eval/` folder should ideally be executed from the project root (`d:\Final Testing`) so that relative paths defined in `config.yaml` resolve properly.

### Running Validation

Compute the mAP over the full validation set for any model defined in the config. Results and Precision-Recall data will automatically route to `results/GPU_XXX/validation/experiments` or `results/GPU_XXX/validation/final`.

```bash
# General validation (will evaluate the default or all models depending on script setup)
python eval/run_validation.py

# To evaluate a specific model registered in config.yaml, pass its key:
python eval/run_validation.py --model p2p3_fusion
```

### Running Test Inference (Visualization)

Run inference on the test set. Isolated folders (`experiments` vs `final`) will be populated automatically based on categories designated in `config.yaml`.

```bash
# Run over all test images
python eval/run_test_inference.py --model baseline_rtdetr_r18

# Run with visualization (saves drawn images to eval/results)
python eval/run_test_inference.py --model p2p3_fusion --visualize --max_vis 20
```

### Benchmarking Models

To test the FPS, latency, and throughput of your models during a mock warm-up and inference benchmark loop (saved to respective `experiments` or `final` folders):

```bash
python eval/run_benchmark.py
```

### Full Reporting & Plot Generation

Generate an aggregated series of CSV reports and visual plots. This newly decoupled pipeline maps accuracy vs latency graphs and **Precision-Recall curves** separately for `experiments` and `final` components.

```bash
# Generate overall comparison reports across all evaluated categories
python eval/generate_report.py
```

### Class-Specific Baseline Delta Heatmaps

Compare deep metrics class-by-class against the original R18 baseline:

```bash
# Check experimental models against the baseline
python eval/baseline_class_analysis.py --mode experiments

# Check production-ready models against the baseline
python eval/baseline_class_analysis.py --mode final
```
