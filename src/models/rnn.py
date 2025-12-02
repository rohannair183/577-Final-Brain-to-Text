# src/models/rnn.py
import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    """
    Simple RNN baseline for brain-to-text decoding.
    Simpler than LSTM - good for quick experimentation and comparison.
    """
    def __init__(self, input_dim=512, hidden_dim=256, num_phonemes=50, 
                 num_layers=2, dropout=0.2, bidirectional=True, nonlinearity='tanh'):
        """
        Args:
            input_dim: Dimension of input features (512 from neural data)
            hidden_dim: Hidden state size
            num_phonemes: Number of output phoneme classes
            num_layers: Number of RNN layers
            dropout: Dropout rate between layers
            bidirectional: Whether to use bidirectional RNN
            nonlinearity: 'tanh' or 'relu'
        """
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Simple RNN layers
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity
        )
        
        # Output projection
        # Account for bidirectional (concatenates forward and backward)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_phonemes)
        
        # Optional: Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(fc_input_dim)
        
    def forward(self, x, lengths):
        """
        Forward pass through RNN decoder.
        
        Args:
            x: (batch, max_seq_len, input_dim) neural features
            lengths: (batch,) actual sequence lengths (before padding)
        
        Returns:
            (batch, max_seq_len, num_phonemes) logits for each phoneme class
        """
        # Pack padded sequence for efficient processing
        # This tells RNN to skip padding, making it faster
        packed = nn.utils.rnn.pack_padded_sequence(
            x, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Run through RNN
        # output: packed sequence of hidden states
        # hidden: final hidden state (not used for CTC)
        output, hidden = self.rnn(packed)
        
        # Unpack back to padded tensor
        # output: (batch, max_seq_len, hidden_dim * num_directions)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Apply layer normalization (helps with training stability)
        output = self.layer_norm(output)
        
        # Project to phoneme vocabulary
        # logits: (batch, max_seq_len, num_phonemes)
        logits = self.fc(output)
        
        return logits
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'RNN',
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


