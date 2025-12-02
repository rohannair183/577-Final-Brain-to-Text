# src/models/tcn.py
import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """
    Removes extra timesteps from the right to keep causal conv length consistent.
    Used when padding for dilated causal convolutions.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    A single TCN residual block:
    - Dilated causal Conv1d
    - Normalization + ReLU + Dropout
    - Second dilated causal Conv1d
    - Residual connection (with 1x1 conv if channels change)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # For causal conv, we pad on the left (implemented via padding + chomp)
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # If channel dims change, use 1x1 conv for the residual
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, channels, time)
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        return self.final_relu(out)


class TCNDecoder(nn.Module):
    """
    Pure Temporal Convolutional Network (TCN) decoder for brain-to-phoneme.

    Expects inputs of shape (batch, time, input_dim) and outputs
    logits of shape (batch, time, num_phonemes),
    compatible with your existing CTC loss + metrics.
    """
    def __init__(
        self,
        input_dim=512,
        num_phonemes=50,
        num_levels=5,
        hidden_dim=256,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        # First layer projects from input_dim to hidden_dim
        layers = []
        in_channels = input_dim
        out_channels = hidden_dim

        for i in range(num_levels):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            # After first block, keep using hidden_dim
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, num_phonemes)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, time, input_dim)
            lengths: (batch,) tensor with valid lengths (not strictly needed
                     inside TCN, but kept for interface consistency)

        Returns:
            logits: (batch, time, num_phonemes)
        """
        # Convert to (batch, channels, time) for Conv1d
        x = x.transpose(1, 2)  # (B, input_dim, T)

        x = self.tcn(x)        # (B, hidden_dim, T)

        # Back to (batch, time, channels)
        x = x.transpose(1, 2)  # (B, T, hidden_dim)

        logits = self.fc(x)    # (B, T, num_phonemes)
        return logits
