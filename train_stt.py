#!/usr/bin/env python3
"""
Training script for HGRN ASR with CTC loss.
Adapted from the main training script but simplified for ASR tasks.
"""

import argparse
import os
import time
from datetime import datetime
import torch
from transformers import get_cosine_schedule_with_warmup
import wandb
import shutil
import json

from data import create_librosa_raw_classification_dataset
from flame.models.hgrn_asr import HGRNASRConfig, HGRNASRForCTC
from flame.logging_timeseries import TimeSeriesLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train HGRN on ASR")
    
    # Dataset args
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples for ASR dataset")
    parser.add_argument("--num_mfcc", type=int, default=80, help="Number of MFCC features")
    parser.add_argument("--cache_dir", type=str, default="/export/work/apierro/datasets/cache", help="Cache directory for datasets")
    
    # Model args
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--expand_ratio", type=int, default=1)
    parser.add_argument("--attn_mode", type=str, default="chunk", choices=["chunk", "fused_recurrent"])

    # Training args
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Device args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging args
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_name", type=str, default="hgrn_asr", help="Name of the model for saving")
    parser.add_argument("--disable_colors", action="store_true", help="Disable colored output")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--print_samples", action="store_true", help="Print sample predictions during training")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    
    return parser.parse_args()


def create_model(config):
    """Create HGRN ASR model."""
    return HGRNASRForCTC(config)


def levenshtein_distance(seq1: list, seq2: list) -> int:
    """Calculate Levenshtein distance between two sequences."""
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    
    if len(seq2) == 0:
        return len(seq1)
    
    previous_row = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        current_row = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_sequence_metrics(predictions: list, targets: list, idx_to_char: dict = None) -> dict:
    """Calculate comprehensive sequence-level metrics."""
    metrics = {
        'cer': 0.0,
        'exact_match': 0.0,
        'token_accuracy': 0.0,
        'total_chars': 0,
        'total_sequences': 0
    }
    
    if len(predictions) == 0:
        return metrics
    
    total_char_errors = 0
    total_chars = 0
    exact_matches = 0
    token_correct = 0
    token_total = 0
    
    for pred, target in zip(predictions, targets):
        # Convert to lists if they're tensors
        if hasattr(pred, 'tolist'):
            pred = pred.tolist()
        if hasattr(target, 'tolist'):
            target = target.tolist()
        
        # If idx_to_char is provided, convert tokens to character sequences for true CER
        if idx_to_char is not None:
            pred_chars = []
            target_chars = []
            
            for token_idx in pred:
                if token_idx in idx_to_char:
                    char = idx_to_char[token_idx]
                    if char != '<blank>':  # Skip blank tokens
                        pred_chars.append(char)
            
            for token_idx in target:
                if token_idx in idx_to_char:
                    char = idx_to_char[token_idx]
                    if char != '<blank>':  # Skip blank tokens
                        target_chars.append(char)
            
            # Character Error Rate (CER) - true character-level comparison
            char_errors = levenshtein_distance(pred_chars, target_chars)
            total_char_errors += char_errors
            total_chars += len(target_chars)
        else:
            # Fallback to token-level comparison if no character mapping
            char_errors = levenshtein_distance(pred, target)
            total_char_errors += char_errors
            total_chars += len(target)
        
        # Exact sequence match
        if pred == target:
            exact_matches += 1
        
        # Token-level accuracy
        min_len = min(len(pred), len(target))
        token_correct += sum(1 for i in range(min_len) if pred[i] == target[i])
        token_total += max(len(pred), len(target))
    
    metrics['cer'] = total_char_errors / max(total_chars, 1)
    metrics['exact_match'] = exact_matches / len(predictions)
    metrics['token_accuracy'] = token_correct / max(token_total, 1)
    metrics['total_chars'] = total_chars
    metrics['total_sequences'] = len(predictions)
    
    return metrics


def decode_tokens_to_text(tokens, idx_to_char):
    """Convert list of token indices back to readable text."""
    if not tokens:
        return ""
    
    text = ""
    for token_idx in tokens:
        if token_idx in idx_to_char:
            char = idx_to_char[token_idx]
            if char == '<blank>':
                continue  # Skip blank tokens
            text += char
        else:
            text += f"<UNK:{token_idx}>"
    return text


def print_sample_predictions(predictions, targets, idx_to_char, num_samples=3, prefix=""):
    """Print sample predictions and targets in readable format."""
    print(f"\n{prefix} Sample Predictions vs Targets:")
    print("=" * 80)
    
    for i in range(min(num_samples, len(predictions))):
        pred_text = decode_tokens_to_text(predictions[i], idx_to_char)
        target_text = decode_tokens_to_text(targets[i], idx_to_char)
        
        print(f"Sample {i+1}:")
        print(f"  Prediction: '{pred_text}'")
        print(f"  Target:     '{target_text}'")
        print(f"  Match:      {predictions[i] == targets[i]}")
        print()


def evaluate_model(model, dataloader, device, idx_to_char=None, print_samples=False):
    """Evaluate ASR model with comprehensive sequence-level metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle ASR batch dictionary format
            inputs = batch['features'].to(dtype=torch.float32, device=device)
            labels = batch['targets'].to(dtype=torch.long, device=device)
            input_lengths = batch['feature_lengths'].to(dtype=torch.long, device=device)
            target_lengths = batch['target_lengths'].to(dtype=torch.long, device=device)
            
            # Transpose features to match model input format [batch, seq_len, feature_dim]
            inputs = inputs.transpose(1, 2)
            
            outputs = model(
                input_ids=inputs, 
                labels=labels,
                input_lengths=input_lengths,
                target_lengths=target_lengths
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            # Decode sequences for evaluation
            decoded_seqs = model.decode(inputs, input_lengths, use_beam_search=True)
            
            # Extract target sequences from batched format
            batch_targets = []
            start_idx = 0
            for i in range(len(decoded_seqs)):
                target_len = target_lengths[i].item()
                target_tokens = labels[start_idx:start_idx + target_len].tolist()
                batch_targets.append(target_tokens)
                start_idx += target_len
            
            all_predictions.extend(decoded_seqs)
            all_targets.extend(batch_targets)
    
    # Calculate comprehensive metrics
    metrics = calculate_sequence_metrics(all_predictions, all_targets, idx_to_char)
    avg_loss = total_loss / len(dataloader)
    
    # Print sample predictions if requested
    if print_samples and idx_to_char is not None:
        print_sample_predictions(all_predictions, all_targets, idx_to_char, num_samples=3, prefix="EVALUATION")
    
    # Return main accuracy metric for backward compatibility
    accuracy = 1.0 - metrics['cer']  # Use CER as primary metric (lower is better)
    
    return avg_loss, accuracy, metrics


def compute_training_metrics(model, batch, device, idx_to_char=None, print_samples=False):
    """Compute sequence metrics for a training batch."""
    model.eval()
    
    with torch.no_grad():
        inputs = batch['features'].to(dtype=torch.float32, device=device)
        labels = batch['targets'].to(dtype=torch.long, device=device)
        input_lengths = batch['feature_lengths'].to(dtype=torch.long, device=device)
        target_lengths = batch['target_lengths'].to(dtype=torch.long, device=device)
        
        # Transpose features to match model input format [batch, seq_len, feature_dim]
        inputs = inputs.transpose(1, 2)
        
        # Decode sequences for evaluation
        decoded_seqs = model.decode(inputs, input_lengths, use_beam_search=False)
        
        # Extract target sequences from batched format
        batch_targets = []
        start_idx = 0
        for i in range(len(decoded_seqs)):
            target_len = target_lengths[i].item()
            target_tokens = labels[start_idx:start_idx + target_len].tolist()
            batch_targets.append(target_tokens)
            start_idx += target_len
        
        # Print sample predictions if requested
        if print_samples and idx_to_char is not None:
            print_sample_predictions(decoded_seqs, batch_targets, idx_to_char, num_samples=2, prefix="TRAINING")
        
        # Calculate metrics for this batch
        metrics = calculate_sequence_metrics(decoded_seqs, batch_targets, idx_to_char)
    
    model.train()
    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval, logger, max_grad_norm=1.0, use_wandb=False, idx_to_char=None, print_samples=False, gradient_accumulation_steps=1):
    """Train ASR model for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Accumulate training metrics
    train_metrics_accumulator = {
        'cer': 0.0,
        'exact_match': 0.0,
        'token_accuracy': 0.0,
        'total_sequences': 0,
        'total_chars': 0,
        'batch_count': 0
    }
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle ASR batch dictionary format
        inputs = batch['features'].to(dtype=torch.float32, device=device)
        labels = batch['targets'].to(dtype=torch.long, device=device)

        input_lengths = batch['feature_lengths'].to(dtype=torch.long, device=device)
        target_lengths = batch['target_lengths'].to(dtype=torch.long, device=device)
        
        # Transpose features to match model input format [batch, seq_len, feature_dim]
        inputs = inputs.transpose(1, 2)
        
        outputs = model(
            input_ids=inputs, 
            labels=labels,
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )
        
        loss = outputs.loss
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Only update optimizer and scheduler every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            # For logging purposes, still calculate grad norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    
        avg_loss = total_loss / (batch_idx + 1)
        lr = scheduler.get_last_lr()[0]
        global_step = (epoch - 1) * num_batches + batch_idx
        total_steps = epoch * num_batches  # Approximate for current epoch
        logger.log_training_step(epoch, global_step, total_steps, avg_loss, lr, grad_norm.item())

        if use_wandb:
            wandb.log({
                "train/loss": avg_loss,
                "train/learning_rate": lr,
                "train/grad_norm": grad_norm.item(),
                "train/epoch": epoch,
                "train/step": global_step,
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
                "dataset": "ASR",
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

    # Create ASR dataset
    train_loader, val_loader, test_loader, n_classes, seq_len, input_dim, char_to_idx, idx_to_char = create_librosa_raw_classification_dataset(
        bsz=args.batch_size,
        max_samples=args.max_samples,
        num_mfcc=args.num_mfcc,
        cache_dir=args.cache_dir
    )
    
    # Log dataset info
    logger.log_dataset_info(
        dataset_name="ASR",
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
            "dataset/name": "ASR",
            "dataset/n_classes": n_classes,
            "dataset/seq_len": seq_len,
            "dataset/input_dim": input_dim,
            "dataset/train_size": len(train_loader.dataset),
            "dataset/val_size": len(val_loader.dataset),
            "dataset/test_size": len(test_loader.dataset)
        })
    
    # Create ASR model config
    max_seq_len = seq_len if seq_len > 0 else 1000  # Default for variable-length
    
    config = HGRNASRConfig(
        input_size=input_dim,
        vocab_size=n_classes,
        max_sequence_length=max_seq_len,
        blank_id=0,  # Assuming blank token is at index 0
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
    
    # Initialize training state
    start_epoch = 0
    best_val_acc = 0.0
    
    # Load checkpoint if resuming
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"Loading checkpoint from {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch} with best validation accuracy: {best_val_acc:.4f}")
        else:
            print(f"Checkpoint file {args.resume_checkpoint} not found. Starting from scratch.")
            args.resume_checkpoint = None
    
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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Load optimizer and scheduler state if resuming
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    
    # Copy training scripts to checkpoint folder
    shutil.copy2("train_stt.py", model_save_dir)
    if os.path.exists("train_stt.sh"):
        shutil.copy2("train_stt.sh", model_save_dir)
    
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
    
    # Training loop starts from start_epoch
    
    print(f"[DEBUG] Sample printing enabled: {args.print_samples}")
    if args.print_samples:
        print(f"[DEBUG] Will print training samples every {args.log_interval} batches")
        print(f"[DEBUG] Will print validation samples every 2 epochs")
    
    for epoch in range(start_epoch, args.epochs):
        logger.log_epoch_start(epoch + 1, args.epochs)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch + 1, 
            args.log_interval, logger, args.max_grad_norm, args.use_wandb, idx_to_char, args.print_samples, args.gradient_accumulation_steps
        )
        
        # Compute training batch metrics for one batch
        train_batch_iter = iter(train_loader)
        train_batch = next(train_batch_iter)
        train_batch_metrics = compute_training_metrics(model, train_batch, device, idx_to_char, print_samples=True)
        
        # Log training batch metrics to wandb
        if args.use_wandb:
            wandb.log({
                "train_batch/cer": train_batch_metrics['cer'],
                "train_batch/exact_match": train_batch_metrics['exact_match'],
                "train_batch/token_accuracy": train_batch_metrics['token_accuracy'],
                "train_batch/total_sequences": train_batch_metrics['total_sequences'],
                "train_batch/epoch": epoch + 1,
            })
        
        # Validate (print samples every 2 epochs if enabled)
        print_val_samples = args.print_samples
        if print_val_samples:
            print(f"\n[DEBUG] Printing validation samples for epoch {epoch + 1}")
        val_loss, val_acc, val_metrics = evaluate_model(model, val_loader, device, idx_to_char, print_val_samples)
        
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
            }, os.path.join(model_save_dir, f'best_model_asr.pt'))
        
        # Log validation results with detailed metrics
        logger.log_validation(epoch + 1, val_loss, val_acc, is_best)
        logger.log_sequence_metrics(epoch + 1, val_metrics, "validation")
        
        # Log epoch results to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "train/epoch_loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "val/cer": val_metrics['cer'],
                "val/exact_match": val_metrics['exact_match'],
                "val/token_accuracy": val_metrics['token_accuracy'],
                "val/total_sequences": val_metrics['total_sequences'],
                "val/epoch": epoch + 1,
                "val/is_best": is_best,
                "val/best_accuracy": best_val_acc
            })
    
    # Final evaluation on test set (print samples if enabled)
    test_loss, test_acc, test_metrics = evaluate_model(model, test_loader, device, idx_to_char, print_samples=args.print_samples)
    
    # Calculate total training time
    total_time = time.time() - logger.start_time
    
    # Log final results
    logger.log_test_results(test_loss, test_acc)
    logger.log_sequence_metrics(0, test_metrics, "final test")
    logger.log_training_complete(total_time, best_val_acc)
    
    # Save final model info
    final_info = {
        "model_name": args.model_name,
        "timestamp": timestamp,
        "dataset": "ASR",
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc,
        "final_test_metrics": test_metrics,
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
            "test/cer": test_metrics['cer'],
            "test/exact_match": test_metrics['exact_match'],
            "test/token_accuracy": test_metrics['token_accuracy'],
            "test/total_sequences": test_metrics['total_sequences'],
            "training/total_time": total_time,
            "training/best_val_accuracy": best_val_acc
        })
        
        # Finish wandb run
        wandb.finish()
    
    print(f"\nTraining completed! Model saved to: {model_save_dir}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test CER: {test_metrics['cer']:.4f} ({test_metrics['cer']*100:.2f}%)")
    print(f"Final test exact match: {test_metrics['exact_match']:.4f} ({test_metrics['exact_match']*100:.2f}%)")


if __name__ == "__main__":
    main()
