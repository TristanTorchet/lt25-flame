#!/usr/bin/env python3
"""
Training script for HGRN time series classification.
Adapted from the main training script but simplified for time series classification.
"""

import argparse
import os
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
import wandb
import shutil
import json

from data import create_mnist_classification_dataset, create_cifar_gs_classification_dataset
from flame.models.hgrn_timeseries import HGRNTimeSeriesConfig, HGRNTimeSeriesForSequenceClassification
from flame.logging_timeseries import TimeSeriesLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train HGRN on time series classification")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="./data")
    
    # Model args
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--expand_ratio", type=int, default=1)
    parser.add_argument("--attn_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Device args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging args
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="hgrn_timeseries", help="Name of the model for saving")
    parser.add_argument("--disable_colors", action="store_true", help="Disable colored output")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    
    return parser.parse_args()


def create_model(config):
    """Create HGRN time series classification model."""
    return HGRNTimeSeriesForSequenceClassification(config)


def evaluate_model(model, dataloader, device):
    """Evaluate model on given dataloader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.clone().detach().to(dtype=torch.float32, device=device)
            labels = labels.clone().detach().to(dtype=torch.long, device=device)
            
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval, logger, max_grad_norm=1.0, use_wandb=False):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.clone().detach().to(dtype=torch.float32, device=device)
        labels = labels.clone().detach().to(dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0]
            global_step = (epoch - 1) * num_batches + batch_idx
            total_steps = epoch * num_batches  # Approximate for current epoch
            logger.log_training_step(epoch, global_step, total_steps, avg_loss, lr, grad_norm.item())
            
            # Log to wandb if enabled
            if use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/epoch": epoch,
                    "train/step": global_step
                })
    
    return total_loss / num_batches


def main():
    args = parse_args()
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(args.save_dir, f"{args.model_name}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_api_key = None
        if os.path.exists("wandb_api_key.txt"):
            with open("wandb_api_key.txt", "r") as f:
                wandb_api_key = f.read().strip()
            os.environ["WANDB_API_KEY"] = wandb_api_key
        
        wandb.init(
            project="flame-timeseries",
            name=f"{args.model_name}_{timestamp}",
            config={
                "model_name": args.model_name,
                "dataset": args.dataset,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "expand_ratio": args.expand_ratio,
                "attn_mode": args.attn_mode,
                "epochs": args.epochs,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "timestamp": timestamp,
                "save_dir": model_save_dir,
            }
        )
    
    # Initialize logger
    logger = TimeSeriesLogger(enable_colors=not args.disable_colors, device=args.device)
    
    # Set device
    device = torch.device(args.device)
    
    # Create datasets
    if args.dataset == "mnist":
        train_loader, val_loader, test_loader, n_classes, seq_len, input_dim = create_mnist_classification_dataset(
            bsz=args.batch_size,
            root=args.data_root,
            version="sequential"
        )
    elif args.dataset == "cifar10":
        train_loader, val_loader, test_loader, n_classes, seq_len, input_dim = create_cifar_gs_classification_dataset(
            bsz=args.batch_size,
            root=args.data_root
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Log dataset info
    logger.log_dataset_info(
        dataset_name=args.dataset,
        n_classes=n_classes,
        seq_len=seq_len,
        input_dim=input_dim,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset)
    )
    
    # Log dataset info to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "dataset/name": args.dataset,
            "dataset/n_classes": n_classes,
            "dataset/seq_len": seq_len,
            "dataset/input_dim": input_dim,
            "dataset/train_size": len(train_loader.dataset),
            "dataset/val_size": len(val_loader.dataset),
            "dataset/test_size": len(test_loader.dataset)
        })
    
    # Create model config
    config = HGRNTimeSeriesConfig(
        input_size=input_dim,
        num_classes=n_classes,
        max_sequence_length=seq_len,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        expand_ratio=args.expand_ratio,
        attn_mode=args.attn_mode,
        use_short_conv=False,
        conv_size=4,
        use_lower_bound=True,
        hidden_ratio=4,
        hidden_act="swish",
        elementwise_affine=True,
        norm_eps=1e-4,
        fuse_norm=True,
        fuse_swiglu=True,
        initializer_range=0.02,
    )
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    
    # Copy training scripts to checkpoint folder
    shutil.copy2("train_timeseries.py", model_save_dir)
    if os.path.exists("train_timeseries.sh"):
        shutil.copy2("train_timeseries.sh", model_save_dir)
    
    # Log training start
    logger.log_training_start(
        total_params=total_params,
        trainable_params=trainable_params,
        total_epochs=args.epochs,
        total_steps=total_steps,
        batch_size=args.batch_size
    )
    
    # Log model info to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "model/total_epochs": args.epochs,
            "model/total_steps": total_steps,
            "model/batch_size": args.batch_size
        })
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        logger.log_epoch_start(epoch + 1, args.epochs)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch + 1, 
            args.log_interval, logger, args.max_grad_norm, args.use_wandb
        )
        
        # Validate
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'args': vars(args),
                'timestamp': timestamp,
            }, os.path.join(model_save_dir, f'best_model_{args.dataset}.pt'))
        
        # Log validation results
        logger.log_validation(epoch + 1, val_loss, val_acc, is_best)
        
        # Log validation results to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/epoch": epoch + 1,
                "val/is_best": is_best,
                "val/best_accuracy": best_val_acc
            })
    
    # Final evaluation on test set
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    
    # Calculate total training time
    total_time = time.time() - logger.start_time
    
    # Log final results
    logger.log_test_results(test_loss, test_acc)
    logger.log_training_complete(total_time, best_val_acc)
    
    # Save final model info
    final_info = {
        "model_name": args.model_name,
        "timestamp": timestamp,
        "dataset": args.dataset,
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "total_training_time": total_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "config": config.__dict__ if hasattr(config, '__dict__') else config,
        "args": vars(args)
    }
    
    with open(os.path.join(model_save_dir, "training_info.json"), "w") as f:
        json.dump(final_info, f, indent=2, default=str)
    
    # Log final results to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "training/total_time": total_time,
            "training/best_val_accuracy": best_val_acc
        })
        
        # Finish wandb run
        wandb.finish()
    
    print(f"\nTraining completed! Model saved to: {model_save_dir}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()