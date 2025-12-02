# src/data/dataloader.py
from torch.utils.data import DataLoader, ConcatDataset
from src.data.dataset import BrainToTextDataset, collate_fn
import os

def create_dataloaders(config):
    """
    Create train/val/test dataloaders from multiple sessions.
    
    Args:
        config: Dict with keys:
            - data_root: Path to hdf5_data_final folder
            - train_sessions: List of session folder names
            - val_sessions: List of session folder names
            - batch_size: int
            - num_workers: int
    """
    data_root = config['data_root']
    
    # Collect all training files
    train_datasets = []
    for session in config['train_sessions']:
        hdf5_path = os.path.join(data_root, session, 'data_train.hdf5')
        if os.path.exists(hdf5_path):
            train_datasets.append(BrainToTextDataset(hdf5_path))
        else:
            print(f"Warning: {hdf5_path} not found")
    
    # Combine all training sessions
    train_dataset = ConcatDataset(train_datasets)
    
    # Same for validation
    val_datasets = []
    for session in config['val_sessions']:
        hdf5_path = os.path.join(data_root, session, 'data_train.hdf5')
        if os.path.exists(hdf5_path):
            val_datasets.append(BrainToTextDataset(hdf5_path))
    
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    
    # Create DataLoaders
    pin_memory = config.get('pin_memory', False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 0),
        pin_memory=pin_memory
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0),
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader

