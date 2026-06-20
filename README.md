# Language Model Training Project (lm-project)

This repository contains a complete pipeline for downloading, training, fine-tuning, evaluating, and generating text with language models like GPT-2. It leverages the Hugging Face Transformers and Datasets libraries, and is heavily optimized for consumer-grade GPUs with limited VRAM (e.g., NVIDIA RTX 4060 8GB).

## Features

- **Hardware Optimized**: Built-in support for FP16 Mixed Precision and Gradient Checkpointing to maximize performance and minimize memory usage on 8GB VRAM GPUs.
- **Offline Mode Support**: Provides scripts and utilities to pre-download models and datasets, completely disabling Hugging Face online checks for air-gapped or restricted network environments.
- **Centralized Configuration**: All hyperparameters, hardware settings, and paths are managed centrally in `config.py`.
- **End-to-End Pipeline**: Includes tools for everything from dataset verification and model downloading to training, perplexity evaluation, and interactive inference.

## Project Structure

- **`config.py`**: The central configuration file. Defines data paths, model settings, hardware configuration, and training hyperparameters.
- **Training Scripts**:
  - `train.py` / `train_gpt.py`: Main language model training/fine-tuning scripts.
  - `offline_gpt2_medium_train.py`: Specifically tailored script for training GPT-2 Medium in a fully offline environment.
- **Evaluation & Inference**:
  - `evaluate_model.py`: Calculates model perplexity and assesses text generation quality on evaluation datasets.
  - `generate_text.py`: An interactive CLI script for text generation using your fine-tuned model.
- **Utilities**:
  - `download_gpt2_medium.py`: Downloads and caches the GPT-2 Medium model and tokenizer for offline use.
  - `verify_env.py` / `verify_dataset.py`: Utility scripts to verify CUDA availability, dependencies, and dataset integrity.
  - `run_training.sh` / `monitor_training.sh`: Bash scripts for managing and monitoring the training process.

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed along with PyTorch (with CUDA support) and the Hugging Face ecosystem.

```bash
pip install torch transformers datasets accelerate
```

### 2. Configuration

Review and modify `config.py` as needed. By default, it uses the `wikitext-103-raw-v1` dataset and optimizes for an 8GB VRAM GPU with a batch size of 4 and 8 gradient accumulation steps.

### 3. Downloading Models (Optional / Offline Mode)

If you plan to run the training offline or want to pre-cache the model, run:

```bash
python download_gpt2_medium.py
```

### 4. Training

To start the training process, you can use the provided bash script or run the Python script directly:

```bash
bash run_training.sh
# OR
python train.py
```

Check the `logs/` directory for training metrics and the `results/` directory for saved checkpoints.

### 5. Evaluation

After training, evaluate your model's perplexity and generation quality:

```bash
python evaluate_model.py
```

### 6. Text Generation

Test your model interactively by running the generation script:

```bash
python generate_text.py
```
*Note: You may need to update the `model_dir` variable in `generate_text.py` to point to your latest checkpoint in the `models/` or `results/` directory.*
