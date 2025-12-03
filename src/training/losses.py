# src/training/losses.py
import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    """
    CTC Loss for brain-to-phoneme decoding.
    Expects:
      - logits: (batch, time, num_classes)
      - targets: (batch, max_target_len), padded with anything
      - input_lengths: (batch,)
      - target_lengths: (batch,)  # number of valid labels per sample
    """
    def __init__(self, blank_id=0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            reduction='mean',
            zero_infinity=True
        )
        self.blank_id = blank_id

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits: (B, T, C) raw scores
            targets: (B, max_target_len) label IDs
            input_lengths: (B,) lengths of each input sequence (time steps)
            target_lengths: (B,) number of labels per sample (no padding)
        """
        # 1) CTC expects log-probabilities of shape (T, B, C)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T, B, C)

        # 2) Flatten targets according to target_lengths
        batch_size = targets.size(0)
        targets_flat = []
        for i in range(batch_size):
            L = int(target_lengths[i].item())
            if L > 0:
                targets_flat.append(targets[i, :L])
        if len(targets_flat) == 0:
            # No valid targets? Return zero loss.
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        targets_flat = torch.cat(targets_flat, dim=0)

        # 3) CTCLoss expects:
        #    - log_probs: (T, B, C)
        #    - targets_flat: (sum(target_lengths),)
        #    - input_lengths: (B,)
        #    - target_lengths: (B,)
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