# src/data/dataloader.py

from torch.utils.data import DataLoader, ConcatDataset
from .dataset import BrainToTextDataset, collate_fn
import os

def create_dataloaders(config):
    """
    Create train/val/test dataloaders from config.
    Automatically detects mode from loss type.
    
    Args:
        config: Dict with data configuration
    """
    data_root = config['data']['data_root']
    
    # Detect mode from loss type (if available in config)
    # Otherwise default to 'ctc'
    loss_type = config['training']['loss']['type']
    mode = 'frame_level' if loss_type in ['cross_entropy', 'frame_ce'] else 'ctc'
    
    print(f"\n{'='*60}")
    print(f"Creating DataLoaders in '{mode}' mode")
    print(f"{'='*60}\n")
    
    # Collect training files
    train_datasets = []
    for session in config['data']['train_sessions']:
        hdf5_path = os.path.join(data_root, session, 'data_train.hdf5')
        if os.path.exists(hdf5_path):
            train_datasets.append(BrainToTextDataset(hdf5_path, mode=mode))
        else:
            print(f"Warning: {hdf5_path} not found")
    
    train_dataset = ConcatDataset(train_datasets)
    
    # Collect validation files
    val_datasets = []
    for session in config['data']['val_sessions']:
        hdf5_path = os.path.join(data_root, session, 'data_train.hdf5')
        if os.path.exists(hdf5_path):
            val_datasets.append(BrainToTextDataset(hdf5_path, mode=mode))
    
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=config['data'].get('shuffle_train', True),
        collate_fn=collate_fn,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config['data'].get('num_workers', 0),
            pin_memory=config['data'].get('pin_memory', True)
        )
    
    return train_loader, val_loader