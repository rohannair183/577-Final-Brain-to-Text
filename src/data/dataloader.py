# src/data/dataloader.py

import os
from torch.utils.data import DataLoader, ConcatDataset
from .dataset import BrainToTextDataset, collate_fn


def create_dataloaders(config):
    """
    Create train/val dataloaders from multiple sessions.

    Args:
        config: Dict that either contains data settings at the top level or
                under a ``data`` key (preferred).
    """
    # Support both legacy flat configs and newer nested configs
    data_cfg = config.get('data', config)

    data_root = data_cfg['data_root']
    
    # Collect all training files
    train_datasets = []
    for session in data_cfg['train_sessions']:
        hdf5_path = os.path.join(data_root, session, 'data_train.hdf5')
        if os.path.exists(hdf5_path):
            print(f"[DataLoader] Using train file: {hdf5_path}")
            train_datasets.append(BrainToTextDataset(hdf5_path))
        else:
            print(f"[DataLoader] Warning: {hdf5_path} not found")

    if len(train_datasets) == 0:
        raise RuntimeError(
            f"No training data found under {data_root} for sessions={config['train_sessions']}"
        )

    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else ConcatDataset(train_datasets)
    )

    # -------------------------
    # Build VAL dataset(s)
    # -------------------------
    val_dataset = None
    val_datasets = []
    for session in data_cfg.get('val_sessions', []):
        hdf5_path = os.path.join(data_root, session, 'data_train.hdf5')
        if os.path.exists(hdf5_path):
            print(f"[DataLoader] Using val file: {hdf5_path}")
            val_datasets.append(BrainToTextDataset(hdf5_path))
    
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None
    
    # Create DataLoaders
    pin_memory = data_cfg.get('pin_memory', False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=data_cfg.get('shuffle_train', True),
        collate_fn=collate_fn,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=pin_memory
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=data_cfg.get('num_workers', 0),
            pin_memory=pin_memory
        )

    return train_loader, val_loader
