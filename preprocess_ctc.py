import os
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC, MelSpectrogram, AmplitudeToDB
from datasets import load_dataset
import re
from collections import defaultdict
import aiohttp

SAMPLE_RATE = 16000

os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = "3600"  # 1 hour
os.environ["FSSPEC_HTTP_TIMEOUT"] = "3600"

class LibriSpeechASRDataset(Dataset):
    def __init__(self, split="train.100", 
                 max_audio_length=16000*10,  # 10 seconds max
                 min_audio_length=16000*1,   # 1 second min
                 background_frequency=0.2,   # Lower for ASR
                 background_volume=0.05,     # Lower for ASR
                 time_shift_ms=100.0,
                 num_mfcc=256,
                 use_mfcc=True,
                 max_samples=None,
                 tokenizer=None,
                 cache_dir="/export/work/apierro/datasets/cache"):
        """
        LibriSpeech dataset for ASR with audio preprocessing
        
        Args:
            split: LibriSpeech split ("train.100", "train.360", "train.500", "validation", "test")
            max_audio_length: Maximum audio length in samples
            min_audio_length: Minimum audio length in samples
            background_frequency: Probability of adding background noise
            background_volume: Volume of background noise
            time_shift_ms: Time shift augmentation in milliseconds
            use_mfcc: Whether to use MFCC features or mel-spectrograms
            max_samples: Maximum number of samples to process
            tokenizer: Text tokenizer for labels (optional)
        """
        
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.background_frequency = background_frequency
        self.background_volume = background_volume
        self.time_shift_samples = int(SAMPLE_RATE * time_shift_ms / 1000)
        self.use_mfcc = use_mfcc
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir

        # Load LibriSpeech dataset
        print(f"Loading LibriSpeech {split} split...")
        self.dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split=split,
            storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
            cache_dir=cache_dir
        )
        
        # Initialize transforms
        self.mfcc_transform = MFCC(sample_rate=SAMPLE_RATE, n_mfcc=256, melkwargs={'n_mels': 256}) if use_mfcc else None
        
        # Prepare dataset
        self.prepare_dataset()
        
        print(f"Dataset prepared with {len(self.samples)} samples")
        print(f"Audio length stats: min={self.min_length:.2f}s, max={self.max_length:.2f}s, avg={self.avg_length:.2f}s")

    def create_character_tokenizer(self):
        """Create a simple character-level tokenizer"""
        # Collect all characters from the dataset
        chars = set()
        for sample in self.dataset:
            chars.update(sample['text'].lower())
        
        # Create vocab (add special tokens)
        vocab = ['<pad>', '<unk>', '<sos>', '<eos>'] + sorted(list(chars))
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        
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
            
            # Clean text
            text = self.clean_text(text)
            
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
        
        print(f"Filtered dataset: {len(self.samples)} samples (from {len(self.dataset)} original)")

    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\?\!\-\']', '', text)
        
        return text

    def tokenize_text(self, text):
        """Convert text to token indices using character tokenizer"""
        tokens = ['<sos>']
        for char in text:
            if char in self.char_to_idx:
                tokens.append(char)
            else:
                tokens.append('<unk>')
        tokens.append('<eos>')
        
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
        
        # Ensure single channel
        if waveform.dim() > 1:
            waveform = waveform.mean(0)
        
        # Apply augmentations (only during training)
        if self.dataset.split._name.startswith('train'):
            waveform = self.time_shift(waveform)
            waveform = self.add_background_noise(waveform)
        
        # Normalize audio
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process audio
        waveform = self.process_audio(sample['audio'])
        
        # Extract features
        if self.use_mfcc:
            features = self.mfcc_transform(waveform)
        else:
            features = AmplitudeToDB()(MelSpectrogram(sample_rate=SAMPLE_RATE)(waveform))
        
        return {
            'features': features,
            'text': sample['text'],
            'text_tokens': torch.tensor(sample['text_tokens'], dtype=torch.long),
            'speaker_id': sample['speaker_id'],
            'id': sample['id']
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    # Sort batch by feature sequence length (descending)
    batch = sorted(batch, key=lambda x: x['features'].size(-1), reverse=True)
    
    # Get dimensions
    batch_size = len(batch)
    feature_dim = batch[0]['features'].size(0)
    max_feature_len = batch[0]['features'].size(1)
    max_text_len = max([len(item['text_tokens']) for item in batch])
    
    # Initialize tensors
    features = torch.zeros(batch_size, feature_dim, max_feature_len)
    feature_lengths = torch.zeros(batch_size, dtype=torch.long)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    texts = []
    speaker_ids = []
    ids = []
    
    for i, item in enumerate(batch):
        # Features
        feat_len = item['features'].size(1)
        features[i, :, :feat_len] = item['features']
        feature_lengths[i] = feat_len
        
        # Text tokens
        text_len = len(item['text_tokens'])
        text_tokens[i, :text_len] = item['text_tokens']
        text_lengths[i] = text_len
        
        # Other info
        texts.append(item['text'])
        speaker_ids.append(item['speaker_id'])
        ids.append(item['id'])
    
    return {
        'features': features,
        'feature_lengths': feature_lengths,
        'text_tokens': text_tokens,
        'text_lengths': text_lengths,
        'texts': texts,
        'speaker_ids': speaker_ids,
        'ids': ids
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
        # Filter out special tokens (assuming they are first few indices)
        clean_tokens = [t for t in tokens if t >= 4]  # Skip <pad>, <unk>, <sos>, <eos>
        
        all_targets.extend(clean_tokens)
        target_lengths.append(len(clean_tokens))
        
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
def create_asr_dataloaders(batch_size=16, max_samples=1000, use_ctc=False, num_mfcc=256, cache_dir="/export/work/apierro/datasets/cache"):
    """Create train and validation dataloaders for ASR"""
    
    # Create datasets
    train_dataset = LibriSpeechASRDataset(
        split="train.360",
        max_samples=max_samples,
        num_mfcc=num_mfcc,
        cache_dir=cache_dir,
    )
    
    val_dataset = LibriSpeechASRDataset(
        split="test",
        max_samples=max_samples//5,
        num_mfcc=num_mfcc,
        cache_dir=cache_dir,
    )
    
    # Choose collate function based on model type
    collate_function = collate_fn_ctc if use_ctc else collate_fn
    
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
    
    return train_loader, val_loader, train_dataset.char_to_idx, train_dataset.idx_to_char


def create_ctc_loss_function(char_to_idx):
    """Create CTC loss function with blank token"""
    # CTC needs a blank token (usually index 0)
    blank_idx = char_to_idx['<pad>']  # Use pad as blank
    return torch.nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)


# Test the dataset with CTC
if __name__ == "__main__":
    # Test with CTC collate function
    train_loader, val_loader, char_to_idx, idx_to_char = create_asr_dataloaders(
        max_samples=100, 
        use_ctc=True
    )
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Blank token index: {char_to_idx['<pad>']}")
    
    # Print sample batch for CTC
    for batch in train_loader:
        print(f"Features shape: {batch['features'].shape}")
        print(f"Feature lengths: {batch['feature_lengths']}")
        print(f"Targets shape: {batch['targets'].shape}")  # Concatenated
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
