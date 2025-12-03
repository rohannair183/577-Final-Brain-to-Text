#!/usr/bin/env python3
import sys
sys.path.append('.')

import os
import argparse
import torch
from src.data.dataloader import create_dataloaders
from src.training.trainer import Trainer
from src.models import get_model
from src.utils.config import load_config


def main(config_path, checkpoint_path, override_total_epochs=None):
    # Load config
    config = load_config(config_path)

    # Set seed
    torch.manual_seed(config['experiment']['seed'])

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    model = get_model(config['model'])
    print(f"Created model: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)

    # Restore model state
    try:
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state.")
    except Exception as e:
        print(f"Warning: could not fully load model state: {e}")

    # Restore optimizer state if present
    if 'optimizer_state_dict' in checkpoint and hasattr(trainer, 'optimizer'):
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state.")
        except Exception as e:
            print(f"Warning: could not load optimizer state: {e}")

    # Restore scheduler state if saved
    if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
        try:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state.")
        except Exception as e:
            print(f"Warning: could not load scheduler state: {e}")

    # Restore tracked metrics
    trainer.train_losses = checkpoint.get('train_losses', [])
    trainer.val_losses = checkpoint.get('val_losses', [])
    trainer.val_pers = checkpoint.get('val_pers', [])
    trainer.best_val_per = checkpoint.get('best_val_per', trainer.best_val_per)

    ckpt_epoch = checkpoint.get('epoch', None)
    if ckpt_epoch is None:
        print("Warning: checkpoint has no 'epoch' entry; resuming from start of config epochs.")
        start_epoch = 0
    else:
        start_epoch = ckpt_epoch + 1
        print(f"Checkpoint epoch: {ckpt_epoch}. Resuming from epoch {start_epoch}.")

    # Determine total desired epochs
    total_epochs = override_total_epochs if override_total_epochs is not None else config['training']['num_epochs']

    remaining = total_epochs - start_epoch
    if remaining <= 0:
        print(f"No remaining epochs to train (start={start_epoch}, total={total_epochs}). Exiting.")
        return

    print(f"Continuing training for {remaining} more epoch(s) to reach {total_epochs} total epochs.")

    # Call trainer.train with remaining epochs
    trainer.train(num_epochs=remaining)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resume training from a checkpoint')
    parser.add_argument('--config', '-c', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', '-ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--total-epochs', '-e', type=int, default=None,
                        help='Optional: override total number of epochs to train to (default from config)')

    args = parser.parse_args()
    main(args.config, args.checkpoint, override_total_epochs=args.total_epochs)
