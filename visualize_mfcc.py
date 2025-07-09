#!/usr/bin/env python3

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from preprocess_ctc import LibriSpeechASRDataset
import argparse

def create_mfcc_heatmap(mfcc_features, save_path="mfcc_heatmap.png", title="MFCC Features Heatmap"):
    """
    Create and save a heatmap of MFCC features
    
    Args:
        mfcc_features: Tensor of shape (n_mfcc, time_steps)
        save_path: Path to save the heatmap image
        title: Title for the heatmap
    """
    # Convert to numpy if tensor
    if isinstance(mfcc_features, torch.Tensor):
        mfcc_features = mfcc_features.detach().cpu().numpy()
    
    # Create figure and heatmap
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with time on x-axis and MFCC coefficients on y-axis
    im = plt.imshow(mfcc_features, aspect='auto', origin='lower', cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, label='MFCC Coefficient Value')
    
    # Set labels and title
    plt.xlabel('Time Steps')
    plt.ylabel('MFCC Coefficient Index')
    plt.title(title)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MFCC heatmap saved to: {save_path}")
    return save_path

def visualize_sample_mfcc(dataset_split="train.100", max_samples=1, num_mfcc=13, save_dir="mfcc_visualizations"):
    """
    Extract and visualize MFCC features from LibriSpeech dataset
    
    Args:
        dataset_split: LibriSpeech split to use
        max_samples: Number of samples to visualize
        num_mfcc: Number of MFCC coefficients
        save_dir: Directory to save visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset
    print(f"Loading dataset with {num_mfcc} MFCC features...")
    dataset = LibriSpeechASRDataset(
        split=dataset_split,
        max_samples=max_samples,
        num_mfcc=num_mfcc,
        use_mfcc=True,
        streaming=False
    )
    
    # Process samples
    for i in range(min(max_samples, len(dataset))):
        sample = dataset[i]
        mfcc_features = sample['features']  # Shape: (n_mfcc, time_steps)
        text = sample['text']
        sample_id = sample['id']
        
        print(f"\nSample {i+1}/{max_samples}:")
        print(f"  ID: {sample_id}")
        print(f"  Text: '{text}'")
        print(f"  MFCC shape: {mfcc_features.shape}")
        print(f"  Duration: {mfcc_features.shape[1] * 160 / 16000:.2f} seconds")
        
        # Create filename
        safe_text = text.replace(' ', '_').replace("'", "")[:30]
        filename = f"mfcc_sample_{i+1}_{sample_id}_{safe_text}.png"
        save_path = os.path.join(save_dir, filename)
        
        # Create heatmap
        title = f"MFCC Features - Sample {i+1}\nText: '{text}'"
        create_mfcc_heatmap(mfcc_features, save_path, title)
        
        # Print statistics
        print(f"  Min MFCC value: {mfcc_features.min():.3f}")
        print(f"  Max MFCC value: {mfcc_features.max():.3f}")
        print(f"  Mean MFCC value: {mfcc_features.mean():.3f}")
        print(f"  Std MFCC value: {mfcc_features.std():.3f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize MFCC features as heatmaps')
    parser.add_argument('--split', default='train.100', help='LibriSpeech split to use')
    parser.add_argument('--max_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--num_mfcc', type=int, default=13, help='Number of MFCC coefficients')
    parser.add_argument('--save_dir', default='mfcc_visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("Starting MFCC visualization...")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples}")
    print(f"Number of MFCC coefficients: {args.num_mfcc}")
    print(f"Save directory: {args.save_dir}")
    
    visualize_sample_mfcc(
        dataset_split=args.split,
        max_samples=args.max_samples,
        num_mfcc=args.num_mfcc,
        save_dir=args.save_dir
    )
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()