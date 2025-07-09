# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a fork of the Flame framework (Flash Language Modeling Made Easy) customized for the LT25 project. It's a minimal and efficient training framework built on torchtitan for language models and sequence modeling tasks, with specific adaptations for:

- **Speech Recognition (ASR)**: HGRN-based ASR models with CTC loss
- **Time Series Classification**: HGRN models for sequential data like MNIST and CIFAR-10
- **Language Modeling**: Standard FLA-based language modeling with various architectures

## Key Architecture Components

### Core Framework (`flame/`)
- **`flame/train.py`**: Main training orchestrator using torchtitan infrastructure
- **`flame/config_manager.py`**: Configuration management system
- **`flame/data.py`**: Data loading and preprocessing utilities
- **`flame/components/`**: Checkpoint management and other training components

### Model Implementations
- **`flame/models/hgrn_timeseries.py`**: HGRN model for time series classification
- **`flame/models/hgrn_asr.py`**: HGRN model for automatic speech recognition with CTC
- **`flame/models/parallelize_fla.py`**: Parallelization utilities for FLA models
- **`flame/models/pipeline_fla.py`**: Pipeline parallelism for FLA models

### Training Scripts
- **`train_timeseries.py`**: Standalone training for time series tasks (MNIST, CIFAR-10)
- **`train_stt.py`**: Standalone training for speech recognition tasks
- **`data.py`**: Dataset creation utilities for various tasks
- **`preprocess_ctc.py`**: CTC preprocessing for ASR tasks

### Custom Models (`custom_models/`)
- **`custom_models/sba/`**: Example of custom model integration (Stick-Breaking Attention)
- Models integrate with Hugging Face transformers and can be registered with AutoModel

## Development Commands

### Setup and Installation
```bash
# Initial setup with custom flash-linear-attention fork
uv sync
uv pip install -e ./flash-linear-attention --force-reinstall
chmod +x *.sh
```

### Training Commands

#### Language Model Training
```bash
# Basic language model training
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer-340M \
  --model.config configs/transformer_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.steps 20480
```

#### Time Series Training
```bash
# MNIST classification
./train_timeseries.sh
# Or manually:
CUDA_VISIBLE_DEVICES=0 uv run python train_timeseries.py \
  --dataset mnist --batch_size 32 --epochs 100 \
  --model_name hgrn_timeseries --use_wandb
```

#### Speech Recognition Training
```bash
# ASR training
uv run python train_stt.py \
  --batch_size 32 --epochs 10 \
  --model_name hgrn_asr --use_wandb
```

### Environment Variables
- `NNODE`: Number of nodes (default: 1)
- `NGPU`: Number of GPUs (default: 8, set to 1 for single-GPU debugging)
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices
- `WANDB_PROJECT`: W&B project name (default: "fla")

## Configuration System

### Model Configurations (`configs/`)
- JSON files defining model architectures (transformer, mamba, hgrn, etc.)
- Sizes from 340M to 7B parameters
- Example: `configs/transformer_340M.json`, `configs/hgrn_timeseries.json`

### Training Configuration
- Uses torchtitan's argument parsing system
- Key parameters: batch_size, seq_len, learning_rate, warmup_steps, steps
- Supports variable-length sequences (`--training.varlen`)
- Multi-dataset training with sampling probabilities

## Parallelization Support

### Distributed Training
- Data parallelism (FSDP/DDP)
- Tensor parallelism
- Pipeline parallelism
- Context parallelism
- Hybrid approaches (HSDP)

### Multi-node Training
- Set `MASTER_ADDR` and `MASTER_PORT` environment variables
- Use torchrun for distributed execution
- Supports Slurm job schedulers

## Dependencies and Package Management

### Package Management
- Uses `uv` for dependency management (uv.lock file)
- Custom flash-linear-attention fork at `git@github.com:TristanTorchet/lt25-flash-linear-attention.git`
- Torchtitan pinned to specific commit: `5e2033c`

### Key Dependencies
- torch >= 2.5
- triton >= 3.0
- transformers >= 4.45.0
- flash-linear-attention (custom fork)
- torchtitan (specific commit)

## Testing

This repository does not include a formal test suite. Testing is primarily done through:
- Manual training runs on small datasets
- Checkpoint loading/saving verification
- Model inference validation

## Logging and Monitoring

### Weights & Biases Integration
- Enable with `--use_wandb` flag
- Project configurable via `WANDB_PROJECT` environment variable
- Automatic run naming based on model and configuration

### TensorBoard Support
- Built-in TensorBoard logging
- Metrics include loss, learning rate, memory usage
- Logs saved to experiment dump folder

## Checkpoint Management

### Automatic Conversion
- Training script automatically converts DCPs to HuggingFace format
- Manual conversion: `python -m flame.utils.convert_dcp_to_hf`
- Reverse conversion: `python -m flame.utils.convert_hf_to_dcp`

### Checkpoint Loading
- `--checkpoint.load_step -1` loads latest checkpoint
- `--checkpoint.interval` controls save frequency
- `--checkpoint.keep_latest_k` manages checkpoint retention