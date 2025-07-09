import os
import random
import torch
import torchaudio
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from datasets import load_dataset
import re
from collections import defaultdict

SAMPLE_RATE = 16000

os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = "3600"  # 1 hour
os.environ["FSSPEC_HTTP_TIMEOUT"] = "3600"
os.environ['HF_DATASETS_CACHE'] = os.path.expanduser('~/.cache/huggingface/datasets')


class LibriSpeechASRDataset(Dataset):
    def __init__(self, split="train.100", 
                 max_audio_length=16000*30,  # 30 seconds max
                 min_audio_length=16000*1,   # 1 second min
                 background_frequency=0.2,   # Lower for ASR
                 background_volume=0.05,     # Lower for ASR
                 time_shift_ms=100.0,
                 num_mfcc=80,
                 use_mfcc=True,
                 max_samples=None,
                 tokenizer=None,
                 streaming=False,):
        """
        LibriSpeech dataset for ASR with audio preprocessing
        
        Args:
            split: LibriSpeech split ("train.100", "train.360", "train.500", "validation.clean", "test.clean")
            max_audio_length: Maximum audio length in samples
            min_audio_length: Minimum audio length in samples
            background_frequency: Probability of adding background noise
            background_volume: Volume of background noise
            time_shift_ms: Time shift augmentation in milliseconds
            use_mfcc: Whether to use MFCC features or mel-spectrograms
            max_samples: Maximum number of samples to process
            tokenizer: Text tokenizer for labels (optional)
            streaming: Whether to use streaming mode for large datasets
        """
        
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.background_frequency = background_frequency
        self.background_volume = background_volume
        self.time_shift_samples = int(SAMPLE_RATE * time_shift_ms / 1000)
        self.use_mfcc = use_mfcc
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self.streaming = streaming
        
        # Load LibriSpeech dataset
        print(f"Loading LibriSpeech {split} split...")
        self.dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split=split,
            trust_remote_code=True,
            streaming=streaming,
        )
        
        # Initialize transforms
        # Store MFCC parameters for librosa
        self.num_mfcc = num_mfcc
        self.n_fft = 400
        self.hop_length = 160
        self.win_length = 400
        self.n_mels = max(num_mfcc, 40)
        self.f_min = 0.0
        self.f_max = SAMPLE_RATE // 2

        # Prepare dataset
        self.prepare_dataset()
        
        print(f"Dataset prepared with {len(self.samples)} samples")
        print(f"Audio length stats: min={self.min_length:.2f}s, max={self.max_length:.2f}s, avg={self.avg_length:.2f}s")

    def create_character_tokenizer(self):
        """Create a simple character-level tokenizer"""
        chars = [
            '<blank>',  # 0 - CTC blank token
            ' ',        # 1 - space
            "'",        # 2 - apostrophe
        ] + [chr(i) for i in range(ord('a'), ord('z') + 1)]  # 3-28 - letters
        
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        return char_to_idx, idx_to_char

    def prepare_dataset(self):
        """Prepare the dataset by filtering and processing samples"""
        self.samples = []
        audio_lengths = []
        
        print("Processing LibriSpeech samples...")
        
        # Create character tokenizer if none provided
        if self.tokenizer is None:
            self.char_to_idx, self.idx_to_char = self.create_character_tokenizer()
        
        for idx, sample in enumerate(self.dataset):
            if self.max_samples and idx >= self.max_samples:
                break
                
            audio_array = sample['audio']['array']
            text = sample['text']
            
            # Filter by audio length
            audio_length = len(audio_array)
            if audio_length < self.min_audio_length or audio_length > self.max_audio_length:
                continue

            text = text.lower()
            
            # Tokenize text if using character tokenizer
            if self.tokenizer is None:
                text_tokens = self.tokenize_text(text)
            else:
                text_tokens = self.tokenizer(text)
            
            self.samples.append({
                'audio': audio_array,
                'text': text,
                'text_tokens': text_tokens,
                'speaker_id': sample['speaker_id'],
                'id': sample['id']
            })
            
            audio_lengths.append(audio_length / SAMPLE_RATE)  # Convert to seconds
        
        # Calculate statistics
        self.min_length = min(audio_lengths) if audio_lengths else 0
        self.max_length = max(audio_lengths) if audio_lengths else 0
        self.avg_length = sum(audio_lengths) / len(audio_lengths) if audio_lengths else 0
        
        if not self.streaming:
            print(f"Filtered dataset: {len(self.samples)} samples (from {len(self.dataset)} original)")
    
    def tokenize_text(self, text):
        """tokenization"""
        tokens = []
        for char in text:
            if char in self.char_to_idx:
                tokens.append(char)
            else:
                print(f"Unknown character '{char}' in text: '{text}'")
                tokens.append('<unk>')  # Only if you want OOV handling
        
        return [self.char_to_idx[token] for token in tokens]

    def time_shift(self, waveform):
        """Apply time shift augmentation"""
        if self.time_shift_samples > 0 and len(waveform) > self.time_shift_samples:
            shift = random.randint(-self.time_shift_samples, self.time_shift_samples)
            return torch.roll(waveform, shifts=shift)
        return waveform

    def add_background_noise(self, waveform):
        """Add subtle background noise for ASR"""
        if random.random() < self.background_frequency:
            # Generate subtle white noise
            noise = torch.randn_like(waveform) * self.background_volume
            waveform = waveform + noise
        return waveform

    def process_audio(self, audio_array):
        """Process audio array and apply augmentations"""
        # Convert to tensor
        waveform = torch.tensor(audio_array, dtype=torch.float32)
        
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process audio
        waveform = self.process_audio(sample['audio'])
        
        # Extract features
        if self.use_mfcc:
            # Use librosa for MFCC extraction
            waveform_np = waveform.numpy()
            mfcc = librosa.feature.melspectrogram(
                y=waveform_np,
                sr=SAMPLE_RATE,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.n_fft,
                fmin=self.f_min,
                fmax=self.f_max
            )
            # Add normalization for better visualization
            mfcc = 20 * np.log10(mfcc + 1e-10)
            mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) + 1e-8

            # # Periodically save MFCC visualizations
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 4))
            # plt.imshow(mfcc, aspect='auto', origin='lower', interpolation='none', cmap='jet')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title(f'MFCC - {sample["id"]}')
            # plt.ylabel('MFCC Coefficients')
            # plt.xlabel('Time Frames')
            # plt.tight_layout()
            
            # # Create directory if it doesn't exist
            # os.makedirs('mfcc_plots', exist_ok=True)
            # plt.savefig(f'mfcc_plots/mfcc_{sample["id"]}.png', dpi=150)
            # plt.close()
            features = torch.tensor(mfcc, dtype=torch.float32)
        else:
            features = AmplitudeToDB()(MelSpectrogram(sample_rate=SAMPLE_RATE)(waveform))
        
        return {
            'features': features,
            'text': sample['text'],
            'text_tokens': torch.tensor(sample['text_tokens'], dtype=torch.long),
            'speaker_id': sample['speaker_id'],
            'id': sample['id']
        }



def collate_fn_ctc(batch):
    """CTC-specific collate function"""
    # Sort batch by feature sequence length (descending)
    batch = sorted(batch, key=lambda x: x['features'].size(-1), reverse=True)
    
    # Get dimensions
    batch_size = len(batch)
    feature_dim = batch[0]['features'].size(0)
    max_feature_len = batch[0]['features'].size(1)
    
    # Initialize tensors
    features = torch.zeros(batch_size, feature_dim, max_feature_len)
    feature_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # CTC requires concatenated targets
    all_targets = []
    target_lengths = []
    
    # Fill tensors
    texts = []
    speaker_ids = []
    ids = []
    
    for i, item in enumerate(batch):
        # Features
        feat_len = item['features'].size(1)
        features[i, :, :feat_len] = item['features']
        feature_lengths[i] = feat_len
        
        # For CTC: concatenate targets (without <sos>/<eos>)
        # Remove special tokens for CTC
        tokens = item['text_tokens']

        
        all_targets.extend(tokens)
        target_lengths.append(len(tokens))
        
        # Other info
        texts.append(item['text'])
        speaker_ids.append(item['speaker_id'])
        ids.append(item['id'])
    
    return {
        'features': features,
        'feature_lengths': feature_lengths,
        'targets': torch.tensor(all_targets, dtype=torch.long),  # Concatenated
        'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
        'texts': texts,
        'speaker_ids': speaker_ids,
        'ids': ids
    }


# Example usage and testing
def create_asr_dataloaders(batch_size=16, max_samples=1000, use_ctc=False, num_mfcc=256, streaming=False):
    """Create train and validation dataloaders for ASR
    
    Args:
        batch_size: Batch size for dataloaders
        max_samples: Maximum number of samples to load
        use_ctc: Whether to use CTC-specific collate function
        num_mfcc: Number of MFCC features
        streaming: Whether to use streaming mode
    """
    
    # Create datasets
    train_dataset = LibriSpeechASRDataset(
        split="train.100",
        max_samples=max_samples,
        num_mfcc=num_mfcc,
        streaming=streaming,
    )
    
    val_dataset = LibriSpeechASRDataset(
        split="validation",
        max_samples=max_samples//5,
        num_mfcc=num_mfcc,
        streaming=streaming,
    )
    
    test_dataset = LibriSpeechASRDataset(
        split="test",
        max_samples=max_samples//5,
        num_mfcc=num_mfcc,
        streaming=streaming,
    )
    
    collate_function = collate_fn_ctc 
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_function
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_function
    )
    
    return train_loader, val_loader, test_dataset, train_dataset.char_to_idx, train_dataset.idx_to_char


def create_ctc_loss_function(char_to_idx):
    """Create CTC loss function with blank token"""
    # CTC needs a blank token (usually index 0)
    blank_idx = char_to_idx['<blank>']  
    return torch.nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)


# Test the dataset with CTC
if __name__ == "__main__":
    # Test with CTC collate function
    train_loader, val_loader, char_to_idx, idx_to_char = create_asr_dataloaders(
        max_samples=5, 
        use_ctc=True,
        streaming=True,
        num_mfcc=13
    )
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Blank token index: {char_to_idx['<blank>']}")
    
    # Print sample batch for CTC
    for batch in train_loader:
        print(f"Features shape: {batch['features'].shape}")
        print(f"Feature lengths: {batch['feature_lengths']}")
        print(f"Targets shape after concat: {batch['targets'].shape}")  # Concatenated
        print(f"Target lengths element one: {batch['target_lengths'][0]}")
        print(f"Target lengths: {batch['target_lengths']}")
        print(f"Sample texts: {batch['texts'][:2]}")
        
        # Example CTC loss calculation
        ctc_loss = create_ctc_loss_function(char_to_idx)
        
        # Simulate model output (random logits)
        batch_size = batch['features'].size(0)
        vocab_size = len(char_to_idx)
        max_time = batch['features'].size(2)
        
        # Fake model output: [time, batch, vocab]
        log_probs = torch.randn(max_time, batch_size, vocab_size).log_softmax(2)
        
        # CTC loss
        loss = ctc_loss(
            log_probs,
            batch['targets'],
            batch['feature_lengths'], 
            batch['target_lengths']
        )
        print(f"Example CTC loss: {loss.item():.4f}")
        break
