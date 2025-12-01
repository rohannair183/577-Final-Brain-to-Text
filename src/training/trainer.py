# src/training/trainer.py
import torch
from tqdm import tqdm
import os
from .metrics import PhonemeMetrics
from .losses import get_loss_function
from .optimizers import get_optimizer, get_scheduler

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create loss function from config
        loss_config = config['training']['loss']
        self.criterion = get_loss_function(
            loss_config['type'],
            **{k: v for k, v in loss_config.items() if k != 'type'}
        )
        
        # Create optimizer from config
        opt_config = config['training']['optimizer']
        self.optimizer = get_optimizer(
            model,
            opt_config['type'],
            opt_config['learning_rate'],
            **{k: v for k, v in opt_config.items() if k not in ['type', 'learning_rate']}
        )
        
        # Create scheduler from config
        sched_config = config['training'].get('scheduler', {})
        self.scheduler = get_scheduler(
            self.optimizer,
            sched_config.get('type'),
            **{k: v for k, v in sched_config.items() if k != 'type'}
        )
        
        # Metrics
        phoneme_to_char = config.get('phoneme_to_char_map', None)
        self.metrics = PhonemeMetrics(phoneme_to_char)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_per = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_pers = []
        
        print(f"Using loss: {loss_config['type']}")
        print(f"Using optimizer: {opt_config['type']}")
        print(f"Using scheduler: {sched_config.get('type', 'none')}")
    
    def train_epoch(self):
        """Run one training epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            inputs = batch['input_features'].to(self.device)
            targets = batch['target_ids'].to(self.device)
            input_lengths = batch['input_lengths']
            target_lengths = batch['target_lengths']
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, input_lengths)
            
            # Compute loss (handles both CTC and CE)
            loss = self.criterion(outputs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_clip = self.config['training'].get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # Step scheduler if it's not plateau-based
        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        return avg_loss
    
    def validate(self):
        """Run validation"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['input_features'].to(self.device)
                targets = batch['target_ids'].to(self.device)
                input_lengths = batch['input_lengths']
                target_lengths = batch['target_lengths']
                transcriptions = batch['transcriptions']
                
                # Forward pass
                outputs = self.model(inputs, input_lengths)
                
                # Compute loss
                loss = self.criterion(outputs, targets, input_lengths, target_lengths)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Compute metrics
                batch_metrics = self.metrics.compute_metrics(
                    outputs, targets, target_lengths, transcriptions
                )
                all_metrics.append(batch_metrics)
        
        # Aggregate
        avg_loss = total_loss / num_batches
        avg_per = sum(m['per'] for m in all_metrics) / len(all_metrics)
        
        result = {
            'val_loss': avg_loss,
            'val_per': avg_per
        }
        
        if 'wer' in all_metrics[0]:
            result['val_wer'] = sum(m['wer'] for m in all_metrics) / len(all_metrics)
            result['val_cer'] = sum(m['cer'] for m in all_metrics) / len(all_metrics)
        
        self.val_losses.append(avg_loss)
        self.val_pers.append(avg_per)
        
        # Step plateau scheduler
        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)
        
        return result
    
    def train(self, num_epochs):
        """Full training loop"""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"Val Loss = {val_metrics['val_loss']:.4f}, "
                      f"PER = {val_metrics['val_per']:.3f}")
                
                if 'val_wer' in val_metrics:
                    print(f"WER = {val_metrics['val_wer']:.3f}, "
                          f"CER = {val_metrics['val_cer']:.3f}")
                
                # Save best model based on PER
                if val_metrics['val_per'] < self.best_val_per:
                    self.best_val_per = val_metrics['val_per']
                    self.save_checkpoint('best_model.pt')
                    print(f"✓ New best PER: {self.best_val_per:.3f}")
            
            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_pers': self.val_pers,
            'best_val_per': self.best_val_per,
            'config': self.config
        }
        
        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")