# src/models/conformer.py
import torch
import torch.nn as nn

class ConformerDecoder(nn.Module):
    """
    Conformer: Convolution-augmented Transformer
    Better than pure Transformer for speech/neural signals.
    """
    def __init__(self, input_dim=512, encoder_dim=256, num_phonemes=50,
                 num_layers=12, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, encoder_dim)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.fc = nn.Linear(encoder_dim, num_phonemes)
        
    def forward(self, x, lengths):
        x = self.input_projection(x)
        
        # Create mask
        batch_size, max_len = x.size(0), x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        logits = self.fc(x)
        return logits


class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            encoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Convolution module
        self.conv = ConvolutionModule(encoder_dim)
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim * 4, encoder_dim)
        )
        
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.norm2 = nn.LayerNorm(encoder_dim)
        self.norm3 = nn.LayerNorm(encoder_dim)
        
    def forward(self, x, mask):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=~mask)
        x = residual + x
        
        # Convolution
        residual = x
        x = self.norm2(x)
        x = self.conv(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, encoder_dim, kernel_size=31):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size, 
                     padding=(kernel_size - 1) // 2, groups=encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Conv1d(encoder_dim, encoder_dim, 1)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, time, channels)
        return x