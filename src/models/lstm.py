# src/models/lstm.py
import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_phonemes=50, 
                 num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Account for bidirectional
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_phonemes)
        
    def forward(self, x, lengths):
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        logits = self.fc(output)
        return logits