#!/usr/bin/env python3
"""Enhanced Dataset Loader for Cascade Gate v7 Training."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path

class EnhancedDataset(Dataset):
    """Dataset with enhanced feature loading."""
    
    def __init__(self, text_features, audio_features, labels, lengths, 
                 cross_modal_features=None, augmentation=False):
        self.text_features = text_features
        self.audio_features = audio_features
        self.labels = labels
        self.lengths = lengths
        self.cross_modal_features = cross_modal_features
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.text_features)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_features[idx], dtype=torch.float32)
        audio = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        length = self.lengths[idx]
        
        # Optional augmentation
        if self.augmentation and np.random.random() < 0.3:
            # Light feature noise
            text = text + torch.randn_like(text) * 0.02
            audio = audio + torch.randn_like(audio) * 0.02
            
        cross_modal = None
        if self.cross_modal_features is not None:
            cross_modal = torch.tensor(self.cross_modal_features[idx], dtype=torch.float32)
        
        return (text, audio, label, length, cross_modal if cross_modal is not None else text)


def enhanced_collate_fn(batch):
    """Collate function that handles enhanced features."""
    if len(batch[0]) == 4:
        # Without cross-modal features
        text_feats, audio_feats, labels, lengths = zip(*batch)
        return (torch.stack(text_feats), torch.stack(audio_feats), 
                torch.stack(labels), torch.tensor(lengths, dtype=torch.long))
    else:
        # With cross-modal features
        text_feats, audio_feats, labels, lengths, cross_feats = zip(*batch)
        return (torch.stack(text_feats), torch.stack(audio_feats), 
                torch.stack(labels), torch.tensor(lengths, dtype=torch.long),
                torch.stack(cross_feats))


def build_kfold_dataloaders(text_arr, audio_arr, labels_arr, batch_size=32, 
                             augmentation=True, include_cross_modal=False, 
                             cross_modal_features=None):
    """Build k-fold dataloaders with optional cross-modal features."""
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(text_arr)):
        # Training dataset
        train_ds = EnhancedDataset(
            text_arr[train_idx], audio_arr[train_idx], 
            labels_arr[train_idx], np.array([len(t) for t in text_arr[train_idx]]),
            cross_features=cross_modal_features[train_idx] if include_cross_modal else None,
            augmentation=augmentation
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=enhanced_collate_fn
        )
        
        # Validation dataset (no augmentation)
        val_ds = EnhancedDataset(
            text_arr[val_idx], audio_arr[val_idx], 
            labels_arr[val_idx], np.array([len(t) for t in text_arr[val_idx]]),
            cross_features=cross_modal_features[val_idx] if include_cross_modal else None,
            augmentation=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=enhanced_collate_fn
        )
        
        yield train_loader, val_loader, fold_idx

if __name__ == "__main__":
    # Test with dummy data
    n_samples = 100
    text_features = np.random.randn(n_samples, 20, 768).astype(np.float32)
    audio_features = np.random.randn(n_samples, 20, 791).astype(np.float32)
    labels = np.random.rand(n_samples).astype(np.float32)
    lengths = np.random.randint(15, 21, size=n_samples)
    
    print("Creating dataset with:")
    print(f"  Samples: {n_samples}")
    print(f"  Text features: {text_features.shape}")
    print(f"  Audio features: {audio_features.shape}")
    
    # Test data loader
    from torch.utils.data import DataLoader
    loader = DataLoader(
        EnhancedDataset(text_features[:20], audio_features[:20], labels[:20], lengths[:20]),
        batch_size=4, collate_fn=enhanced_collate_fn
    )
    
    for batch in loader:
        print(f"Batch shapes: {len(batch)} tensors")
        for i, tensor in enumerate(batch):
            print(f"  Tensor {i}: {tensor.shape}")
        break
    
    print("✅ Enhanced dataset test passed")
