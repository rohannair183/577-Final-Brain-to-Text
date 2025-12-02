# scripts/train.py
import sys
sys.path.append('.')

import torch
from src.data.dataloader import create_dataloaders
from src.training.trainer import Trainer
from src.models import get_model  # Factory function
from src.utils.config import load_config
import argparse

def main(config_path):
    # Load config
    config = load_config(config_path)
    
    # Set seed
    torch.manual_seed(config['experiment']['seed'])
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config['data'])
    
    # Create model (automatically picks the right one!)
    model = get_model(config['model'])
    print(f"Created model: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)