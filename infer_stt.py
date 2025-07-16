#!/usr/bin/env python3
"""
Inference script for HGRN ASR with CTC loss.
Performs speech-to-text inference on audio files using trained checkpoints.
"""

import argparse
import os
import torch
import torchaudio
import librosa
import numpy as np
import json
from pathlib import Path

from flame.models.hgrn_asr import HGRNASRConfig, HGRNASRForCTC

SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Perform ASR inference with HGRN")
    
    # Required args
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to audio file for inference")
    
    # Optional args
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for inference")
    parser.add_argument("--beam_size", type=int, default=5,
                       help="Beam size for beam search decoding")
    parser.add_argument("--max_length", type=int, default=1000,
                       help="Maximum sequence length for decoding")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                       help="Length penalty for beam search")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save transcription (optional)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    return parser.parse_args()


class AudioPreprocessor:
    """Audio preprocessing pipeline for ASR inference"""
    
    def __init__(self, num_mfcc=80, sample_rate=SAMPLE_RATE):
        self.num_mfcc = num_mfcc
        self.sample_rate = sample_rate
        self.n_fft = 400
        self.hop_length = 160
        self.win_length = 400
        self.n_mels = max(num_mfcc, 40)
        self.f_min = 0.0
        self.f_max = sample_rate // 2
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio using torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to numpy for librosa processing
        waveform_np = waveform.squeeze().numpy()
        
        return waveform_np
    
    def extract_features(self, waveform):
        """Extract MFCC features from waveform"""
        # Use librosa for MFCC extraction (same as training)
        mfcc = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        # Apply same normalization as training
        mfcc = 20 * np.log10(mfcc + 1e-10)
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) + 1e-8
        
        # Convert to tensor and add batch dimension
        features = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]
        
        return features
    
    def preprocess(self, audio_path):
        """Complete preprocessing pipeline"""
        waveform = self.load_audio(audio_path)
        features = self.extract_features(waveform)
        
        # Create input lengths tensor
        input_lengths = torch.tensor([features.size(2)], dtype=torch.long)
        
        return features, input_lengths


class ASRInferenceEngine:
    """ASR inference engine with checkpoint loading and beam search"""
    
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        self.model = None
        self.config = None
        self.idx_to_char = None
        self.char_to_idx = None
        
        self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load trained model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint with weights_only=False for compatibility
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract config and create model
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint. Please ensure checkpoint contains model config.")
        
        # Create model
        self.model = HGRNASRForCTC(self.config)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Model state dict not found in checkpoint")
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Load character mappings if available
        if 'char_to_idx' in checkpoint:
            self.char_to_idx = checkpoint['char_to_idx']
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        else:
            # Create default character mapping if not in checkpoint
            print("Warning: Character mappings not found in checkpoint. Creating default mapping.")
            self.create_default_char_mapping()
        
        print(f"Model loaded successfully. Vocabulary size: {len(self.char_to_idx)}")
    
    def create_default_char_mapping(self):
        """Create default character mapping (same as training)"""
        chars = [
            '<blank>',  # 0 - CTC blank token
            ' ',        # 1 - space
            "'",        # 2 - apostrophe
        ] + [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 3-28 - letters
        
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    def decode_tokens_to_text(self, tokens):
        """Convert list of token indices back to readable text"""
        if not tokens:
            return ""
        
        text = ""
        for token_idx in tokens:
            if token_idx in self.idx_to_char:
                char = self.idx_to_char[token_idx]
                if char == '<blank>':
                    continue  # Skip blank tokens
                text += char
            else:
                text += f"<UNK:{token_idx}>"
        return text
    
    def beam_search_decode(self, features, input_lengths, beam_size=5, max_length=1000, length_penalty=1.0):
        """Perform beam search decoding"""
        with torch.no_grad():
            # Transpose features to match model input format [batch, seq_len, feature_dim]
            features = features.transpose(1, 2)
            
            # Use model's decode method with beam search
            decoded_seqs = self.model.decode(
                features, 
                input_lengths, 
                use_beam_search=True
            )
            
            return decoded_seqs[0] if decoded_seqs else []
    
    def greedy_decode(self, features, input_lengths):
        """Perform greedy decoding (faster but potentially less accurate)"""
        with torch.no_grad():
            # Transpose features to match model input format [batch, seq_len, feature_dim]
            features = features.transpose(1, 2)
            
            # Use model's decode method without beam search
            decoded_seqs = self.model.decode(
                features, 
                input_lengths, 
                use_beam_search=False
            )
            
            return decoded_seqs[0] if decoded_seqs else []
    
    def inference(self, features, input_lengths, use_beam_search=True, beam_size=5, max_length=1000, length_penalty=1.0):
        """Perform inference on audio features"""
        # Move inputs to device
        features = features.to(self.device)
        input_lengths = input_lengths.to(self.device)
        
        # Decode
        if use_beam_search:
            tokens = self.beam_search_decode(features, input_lengths, beam_size, max_length, length_penalty)
        else:
            tokens = self.greedy_decode(features, input_lengths)
        
        # Convert tokens to text
        text = self.decode_tokens_to_text(tokens)
        
        return text, tokens


def perform_inference(audio_path, checkpoint_path, device='cpu', beam_size=5, max_length=1000, 
                     length_penalty=1.0, verbose=False, output_file=None):
    """
    Complete inference pipeline for ASR
    
    Args:
        audio_path: Path to input audio file
        checkpoint_path: Path to trained model checkpoint
        device: Device for inference ('cpu' or 'cuda')
        beam_size: Beam size for beam search decoding
        max_length: Maximum sequence length for decoding
        length_penalty: Length penalty for beam search
        verbose: Enable verbose output
        output_file: Optional output file to save transcription
    
    Returns:
        tuple: (transcribed_text, confidence_score)
    """
    
    # Initialize preprocessor
    if verbose:
        print("Initializing audio preprocessor...")
    
    # Load checkpoint to get num_mfcc from config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    num_mfcc = getattr(checkpoint.get('config', None), 'input_size', 80)
    
    preprocessor = AudioPreprocessor(num_mfcc=num_mfcc)
    
    # Initialize inference engine
    if verbose:
        print("Loading model checkpoint...")
    
    inference_engine = ASRInferenceEngine(checkpoint_path, device)
    
    # Preprocess audio
    if verbose:
        print(f"Processing audio file: {audio_path}")
    
    features, input_lengths = preprocessor.preprocess(audio_path)
    
    if verbose:
        print(f"Audio features shape: {features.shape}")
        print(f"Audio length: {input_lengths.item()} frames ({input_lengths.item() * 0.01:.2f} seconds)")
    
    # Perform inference
    if verbose:
        print(f"Performing inference with beam_size={beam_size}...")
    
    transcription, tokens = inference_engine.inference(
        features, 
        input_lengths, 
        use_beam_search=(beam_size > 1),
        beam_size=beam_size,
        max_length=max_length,
        length_penalty=length_penalty
    )
    
    # Clean up transcription
    transcription = transcription.strip()
    
    if verbose:
        print(f"Raw tokens: {tokens}")
        print(f"Transcription: '{transcription}'")
    
    # Save to output file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            "audio_file": str(Path(audio_path).absolute()),
            "transcription": transcription,
            "tokens": tokens,
            "beam_size": beam_size,
            "audio_length_frames": input_lengths.item(),
            "audio_length_seconds": input_lengths.item() * 0.01
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        if verbose:
            print(f"Results saved to: {output_file}")
    
    return transcription, tokens


def main():
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    try:
        # Perform inference
        transcription, tokens = perform_inference(
            audio_path=args.audio,
            checkpoint_path=args.checkpoint,
            device=args.device,
            beam_size=args.beam_size,
            max_length=args.max_length,
            length_penalty=args.length_penalty,
            verbose=args.verbose,
            output_file=args.output_file
        )
        
        # Print results
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT")
        print("="*50)
        print(f"Audio file: {args.audio}")
        print(f"Transcription: '{transcription}'")
        print("="*50)
        
        if args.verbose:
            print(f"Token sequence: {tokens}")
            print(f"Number of tokens: {len(tokens)}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()