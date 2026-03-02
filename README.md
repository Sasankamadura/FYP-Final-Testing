# VisDrone RT-DETR Evaluation Project

This repository contains the evaluation framework, model weights, and datasets for the RT-DETR object detection project focused on the VisDrone 2019 dataset.

## Directory Structure

- **`eval/`**: The core directory containing all the validation, testing, and benchmarking scripts.
  - `config.yaml`: The master configuration file where all datasets, model paths, and evaluation metrics are registered.
  - `run_validation.py`: Script to evaluate models on the validation set and compute standard COCO metrics (mAP, mAR).
  - `run_test_inference.py`: Script to generate predictions on the test set and optionally visualize the outputs bounding boxes.
  - `run_benchmark.py`: Script to measure model latency and throughput (FPS).
  - `baseline_class_analysis.py`: Script dedicated to extracting per-class metrics.
  - `generate_report.py`: Aggregates the results into a final report.
  - `requirements.txt`: Python package dependencies required to run the evaluation scripts.
  - `results/`: Automatically generated output folder where prediction JSON files and visualizations are saved.
- **`Experiments/`**: Contains exported ONNX model checkpoints for the baseline RT-DETR and various architectural improvements (e.g., P2-P3 fusion, query importance, gnConv integrations). 
- **`Final Models/`**: A designated folder to hold the chosen production-ready models once finalized.
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

Compute the mAP over the full validation set for any model defined in the config:

```bash
# General validation (will evaluate the default or all models depending on script setup)
python eval/run_validation.py

# To evaluate a specific model registered in config.yaml, pass its key:
python eval/run_validation.py --model p2p3_fusion
```

### Running Test Inference (Visualization)

Run inference on the test set. You can optionally export side-by-side visual comparisons of the model's bounding box predictions.

```bash
# Run over all test images
python eval/run_test_inference.py --model baseline_rtdetr_r18

# Run with visualization (saves drawn images to eval/results)
python eval/run_test_inference.py --model p2p3_fusion --visualize --max_vis 20
```

### Benchmarking Models

To test the FPS, latency, and throughput of your models during a mock warm-up and inference benchmark loop:

```bash
python eval/run_benchmark.py
```

### Full Reporting

To generate an aggregated markdown or CSV report of evaluation runs across multiple improvements:

```bash
python eval/generate_report.py
```
