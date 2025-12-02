# src/training/optimizers.py
import torch.optim as optim

def get_optimizer(model, optimizer_type, learning_rate, **kwargs):
    """
    Factory function to create optimizer based on config.
    
    Args:
        model: The model to optimize
        optimizer_type: 'adam', 'adamw', 'sgd', 'radam'
        learning_rate: Learning rate
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_type == 'adam':
        return optim.Adam(
            params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    
    elif optimizer_type == 'sgd':
        return optim.SGD(
            params,
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0),
            nesterov=kwargs.get('nesterov', False)
        )
    
    elif optimizer_type == 'radam':
        # Rectified Adam - more stable at start of training
        return optim.RAdam(
            params,
            lr=learning_rate,
            betas=kwargs.get('betas', (0.9, 0.999)),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        scheduler_type: 'cosine', 'step', 'plateau', 'onecycle'
        **kwargs: Scheduler-specific arguments
    """
    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    
    elif scheduler_type == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=kwargs.get('total_steps', 1000),
            pct_start=kwargs.get('pct_start', 0.3)
        )
    
    elif scheduler_type is None or scheduler_type == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")