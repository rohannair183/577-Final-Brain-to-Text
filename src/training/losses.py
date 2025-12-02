# src/training/losses.py
import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    """
    CTC Loss for brain-to-phoneme decoding.
    Handles variable-length input/output without explicit alignment.
    """
    def __init__(self, blank_id=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        self.blank_id = blank_id
    
    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits: (batch, time, num_classes) - raw predictions
            targets: (batch, target_len) - target phoneme IDs
            input_lengths: (batch,) - length of each neural sequence
            target_lengths: (batch,) - length of each target sequence
        
        Returns:
            loss value
        """
        # CTC expects: (time, batch, num_classes)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (time, batch, classes)
        
        # Flatten targets (CTC expects 1D concatenated targets)
        targets_flat = []
        for i, length in enumerate(target_lengths):
            target_seq = targets[i][:length]
            # Remove padding (-1)
            target_seq = target_seq[target_seq != -1]
            targets_flat.append(target_seq)
        
        targets_flat = torch.cat(targets_flat)
        
        loss = self.ctc_loss(
            log_probs,
            targets_flat,
            input_lengths,
            target_lengths
        )
        
        return loss


# src/training/losses.py

class CrossEntropyLoss(nn.Module):
    """Frame-level cross-entropy loss."""
    def __init__(self, blank_id=0, ignore_index=-1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.first_call = True
    
    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        """
        Args:
            logits: (batch, time, num_classes)
            targets: (batch, time)
        """
        # Reshape and compute loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        
        return self.loss_fn(logits_flat, targets_flat)


class Seq2SeqLoss(nn.Module):
    """
    Sequence-to-sequence loss with attention.
    For encoder-decoder architectures.
    """
    def __init__(self, ignore_index=-1, label_smoothing=0.0):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(self, logits, targets, input_lengths=None, target_lengths=None):
        # Similar to CrossEntropy but with label smoothing option
        return self.loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )


def get_loss_function(loss_type, **kwargs):
    """
    Factory function to create loss based on config.
    
    Args:
        loss_type: 'ctc', 'cross_entropy', or 'seq2seq'
        **kwargs: Additional arguments for the loss
    
    Returns:
        Loss function instance
    """
    loss_functions = {
        'ctc': CTCLoss,
        'cross_entropy': CrossEntropyLoss,
        'seq2seq': Seq2SeqLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)