import torch
import torchvision
import numpy as np
from torchvision import transforms

# PX = 1/plt.rcParams['figure.dpi']



# WARNING: this code is from QSSM project and won't be updated 
def create_mnist_classification_dataset(bsz=128, root="./data", version="sequential"):
    print("[*] Generating MNIST Classification Dataset...")
    assert version in ["sequential", "row"], "Invalid version for MNIST dataset"

    # Constants
    if version == "sequential":
        SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 10, 1
    elif version == "row":
        SEQ_LENGTH, N_CLASSES, IN_DIM = 28, 10, 28
    tf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ]

    tf.append(transforms.Lambda(lambda x: x.view(SEQ_LENGTH, IN_DIM)))
    tf = transforms.Compose(tf)

    train = torchvision.datasets.MNIST(
        root, train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        root, train=False, download=True, transform=tf
    )

    # split the dataset into train and val 
    train, val = torch.utils.data.random_split(train, [50000, 10000])      


    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True, 
        drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=bsz, shuffle=False, 
        drop_last=True, pin_memory=True, num_workers=2, persistent_workers=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False, 
        drop_last=True, pin_memory=True, num_workers=2, persistent_workers=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM


def create_cifar_gs_classification_dataset(bsz=128, root="./data"):
    
    print("[*] Generating CIFAR-10 Classification Dataset")

    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 32 * 32, 10, 1
    tf = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(1, SEQ_LENGTH).t()),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )
    train, val = torch.utils.data.random_split(train, [40000, 10000])

    # Return data loaders, with the provided batch size
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=bsz, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=4, persistent_workers=True
    )
    valloader = torch.utils.data.DataLoader(
        val, batch_size=bsz, shuffle=False, drop_last=True,
        pin_memory=True, num_workers=2, persistent_workers=True
    )
    testloader = torch.utils.data.DataLoader(
        test, batch_size=bsz, shuffle=False, drop_last=True,
        pin_memory=True, num_workers=2, persistent_workers=True
    )

    return trainloader, valloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM

def create_librosa_raw_classification_dataset(bsz=128, root="./data", max_samples=1000, num_mfcc=256, cache_dir="/export/work/apierro/datasets/cache"):
    
    print("[*] Generating LibriSpeech ASR Dataset")

    # Import here to avoid circular imports
    from preprocess_ctc import create_asr_dataloaders
    
    train_loader, val_loader, test_loader, char_to_idx, idx_to_char = create_asr_dataloaders(
                        batch_size=bsz,
                        max_samples=max_samples, 
                        use_ctc=True,
                        num_mfcc=num_mfcc,
                        cache_dir=cache_dir)

    # Return: train_loader, val_loader, test_loader, n_classes, seq_len, input_dim, char_to_idx, idx_to_char
    # For ASR: seq_len is variable, so we use -1, input_dim is num_mfcc
    return train_loader, val_loader, test_loader, len(char_to_idx), -1, num_mfcc, char_to_idx, idx_to_char
