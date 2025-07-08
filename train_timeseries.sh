CUDA_VISIBLE_DEVICES=0 uv run python train_timeseries.py \
    --dataset mnist \
    --batch_size 32 \
    --data_root ./data \
    --hidden_size 64 \
    --num_layers 4 \
    --num_heads 1 \
    --expand_ratio 1 \
    --attn_mode fused_recurrent \
    --epochs 100 \
    --lr 5e-3 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --device cuda \
    --log_interval 100 \
    --save_dir ./checkpoints \
    --model_name hgrn_timeseries \
    --use_wandb \

# Arguments explanation:
# --dataset: Dataset to use for training
#   Possible values: mnist, cifar10
#   Default: mnist
#
# --batch_size: Batch size for training
#   Possible values: any positive integer
#   Default: 32
#
# --data_root: Root directory for dataset storage
#   Possible values: any valid directory path
#   Default: ./data
#
# --hidden_size: Hidden dimension size of the model
#   Possible values: any positive integer
#   Default: 64
#
# --num_layers: Number of layers in the model
#   Possible values: any positive integer
#   Default: 4
#
# --num_heads: Number of attention heads
#   Possible values: any positive integer
#   Default: 1
#
# --expand_ratio: Expansion ratio for feedforward layers
#   Possible values: any positive integer
#   Default: 1
#
# --attn_mode: Attention mechanism mode
#   Possible values: chunk, fused_recurrent
#   Default: chunk
#
# --epochs: Number of training epochs
#   Possible values: any positive integer
#   Default: 10
#
# --lr: Learning rate for optimizer
#   Possible values: any positive float
#   Default: 1e-3
#
# --warmup_steps: Number of warmup steps for learning rate scheduler
#   Possible values: any non-negative integer
#   Default: 100
#
# --weight_decay: Weight decay for optimizer
#   Possible values: any non-negative float
#   Default: 0.01
#
# --max_grad_norm: Maximum gradient norm for clipping
#   Possible values: any positive float
#   Default: 1.0
#
# --device: Device to use for training
#   Possible values: cuda, cpu, or specific device like cuda:0
#   Default: cuda if available, else cpu
#
# --log_interval: Interval for logging training progress
#   Possible values: any positive integer
#   Default: 100
#
# --save_dir: Directory to save model checkpoints
#   Possible values: any valid directory path
#   Default: ./checkpoints
#
# --model_name: Name of the model for saving (used in folder naming)
#   Possible values: any valid string
#   Default: hgrn_timeseries
#
# --use_wandb: Enable Weights & Biases logging
#   Flag argument (no value needed)
#   Default: False (wandb disabled)
#
# --disable_colors: Disable colored output in console
#   Flag argument (no value needed)
#   Default: False (colors enabled)
