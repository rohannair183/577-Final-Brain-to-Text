# src/models/cnn_transformer.py
import math
import torch
import torch.nn as nn


def _expand_to_list(value, length):
    """Expand scalar or list/tuple to a list of given length."""
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected {length} values, got {len(value)}")
        return list(value)
    return [value for _ in range(length)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CNNTransformerEncoder(nn.Module):
    """
    Convolutional front-end for local feature extraction + Transformer encoder
    for global context aggregation. Returns logits and updated sequence lengths.
    """

    def __init__(
        self,
        input_dim=512,
        num_phonemes=50,
        cnn_channels=(256, 256),
        cnn_kernel_sizes=5,
        cnn_strides=1,
        d_model=256,
        num_layers=4,
        num_heads=4,
        dim_feedforward=None,
        dropout=0.1,
        max_len=5000,
        blank_penalty=0.0,
        blank_id=0,
    ):
        super().__init__()

        cnn_channels = list(cnn_channels)
        num_cnn_layers = len(cnn_channels)
        kernel_sizes = _expand_to_list(cnn_kernel_sizes, num_cnn_layers)
        strides = _expand_to_list(cnn_strides, num_cnn_layers)

        # Convolutional feature extractor
        conv_layers = []
        in_channels = input_dim
        for out_channels, kernel, stride in zip(cnn_channels, kernel_sizes, strides):
            padding = kernel // 2  # keep roughly the same length when stride=1
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels
        self.conv_layers = nn.ModuleList(conv_layers)
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Project to transformer dimension if needed
        self.input_projection = (
            nn.Identity() if in_channels == d_model else nn.Linear(in_channels, d_model)
        )

        # Positional encoding + Transformer encoder
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output classifier
        self.fc = nn.Linear(d_model, num_phonemes)
        self.blank_penalty = float(blank_penalty)
        self.blank_id = int(blank_id)

    @staticmethod
    def _update_lengths(lengths, kernel, stride, padding):
        """
        Compute output lengths after Conv1d with given params.
        Using: L_out = floor((L_in + 2*pad - (kernel-1) - 1) / stride + 1)
        """
        return ((lengths + 2 * padding - (kernel - 1) - 1) // stride) + 1

    def forward(self, x, lengths):
        """
        Args:
            x: Tensor of shape (batch, time, input_dim)
            lengths: Tensor of shape (batch,) with original sequence lengths

        Returns:
            logits: (batch, time', num_phonemes)
            out_lengths: (batch,) lengths after convolutional subsampling
        """
        out_lengths = lengths.clone()

        # CNN front-end
        x = x.transpose(1, 2)  # (batch, channels, time)
        for conv, kernel, stride in zip(self.conv_layers, self.kernel_sizes, self.strides):
            x = conv(x)
            padding = kernel // 2
            out_lengths = self._update_lengths(out_lengths, kernel, stride, padding)
        x = x.transpose(1, 2)  # (batch, time', channels)

        # Transformer encoder
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        # Build padding mask with updated lengths
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= out_lengths.to(x.device).unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=mask)
        logits = self.fc(x)
        if self.blank_penalty > 0 and 0 <= self.blank_id < logits.size(-1):
            logits[..., self.blank_id] = logits[..., self.blank_id] - self.blank_penalty
        return logits, out_lengths
