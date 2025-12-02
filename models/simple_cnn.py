# src/models/simple_cnn.py
import torch
import torch.nn as nn

class CNNDecoder(nn.Module):
    """Simple CNN baseline for comparison"""
    def __init__(self, input_dim=512, num_phonemes=50, 
                 num_layers=4, kernel_size=3):
        super().__init__()
        
        channels = [input_dim, 256, 256, 256, 256]
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(channels[i], channels[i+1], kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_phonemes)
        
    def forward(self, x, lengths):
        # x: (batch, time, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, time)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch, time, channels)
        logits = self.fc(x)
        return logits