# src/models/cnn_bilstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTMDecoder(nn.Module):
    """
    CNN + BiLSTM baseline for brain-to-text decoding.
    Convolutions extract local patterns, BiLSTM captures temporal dependencies.
    """
    def __init__(self, input_dim=512, cnn_channels=128, lstm_hidden_dim=256, 
                 num_phonemes=50, lstm_layers=2, dropout=0.2, bidirectional=True):
        """
        Args:
            input_dim: Dimension of input features (512 from neural data)
            cnn_channels: Number of channels in convolutional layer
            lstm_hidden_dim: Hidden state size for LSTM
            num_phonemes: Number of output phoneme classes
            lstm_layers: Number of BiLSTM layers
            dropout: Dropout rate for LSTM
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.bidirectional = bidirectional
        self.lstm_layers = lstm_layers
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # 1D Convolution to extract local temporal features
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=cnn_channels, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        
        # BiLSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        fc_input_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_phonemes)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(fc_input_dim)
    
    def forward(self, x, lengths):
        """
        Forward pass through CNN + BiLSTM decoder.
        
        Args:
            x: (batch, max_seq_len, input_dim) neural features
            lengths: (batch,) actual sequence lengths (before padding)
        
        Returns:
            (batch, max_seq_len, num_phonemes) logits for each phoneme class
        """
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len) for conv1d
        x = x.transpose(1, 2)
        
        # Apply convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Back to (batch, seq_len, channels) for LSTM
        x = x.transpose(1, 2)
        
        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # BiLSTM
        output, _ = self.lstm(packed)
        
        # Unpack to padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        # Project to phoneme logits
        logits = self.fc(output)
        
        return logits
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'CNN-BiLSTM',
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_layers': self.lstm_layers,
            'bidirectional': self.bidirectional,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
