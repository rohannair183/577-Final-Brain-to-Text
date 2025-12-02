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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 0),
            pin_memory=True
        )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    config = {
        'data_root': 'data/raw/hdf5_data_final',
        'train_sessions': ['t15.2023.08.11', 't15.2023.08.13'],
        'val_sessions': ['t15.2023.09.24'],
        'batch_size': 16,
        'num_workers': 4
    }
    
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")