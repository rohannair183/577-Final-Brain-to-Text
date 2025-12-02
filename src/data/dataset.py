# src/data/dataset.py
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class BrainToTextDataset(Dataset):
    def __init__(self, hdf5_path):
        """
        Args:
            hdf5_path: Path to HDF5 file (e.g., data_train.hdf5)
        """
        self.hdf5_path = hdf5_path
        
        # Get all trial names efficiently
        with h5py.File(hdf5_path, 'r') as f:
            self.trial_names = [key for key in f.keys() if key.startswith('trial_')]
        
        print(f"Loaded {len(self.trial_names)} trials from {hdf5_path}")
    
    def __len__(self):
        return len(self.trial_names)
    def __getitem__(self, idx):
        trial_name = self.trial_names[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            trial = f[trial_name]
            
            # Load data
            input_features = trial['input_features'][:]
            raw_seq_ids = trial['seq_class_ids'][:]
            raw_transcription = trial['transcription'][:]

        # Process Neural Data
        input_tensor = torch.FloatTensor(input_features)
        
        # Process Targets (Trim padding from seq_class_ids)
        valid_indices = raw_seq_ids != 0
        target_tensor = torch.LongTensor(raw_seq_ids[valid_indices])
        
        # Process Text (Decode for reference)
        valid_chars = raw_transcription[raw_transcription != 0]
        transcription_str = "".join([chr(c) for c in valid_chars])

        return {
            'input_features': input_tensor,
            'target_ids': target_tensor,
            'transcription': transcription_str,
            'trial_name': trial_name,
            # --- RESTORED KEYS BELOW ---
            'input_length': input_tensor.shape[0],
            'target_length': target_tensor.shape[0] 
        }

def collate_fn(batch):
    # 1. Extract data
    input_features_list = [item['input_features'] for item in batch]
    target_ids_list = [item['target_ids'] for item in batch]
    transcriptions = [item['transcription'] for item in batch]
    
    # 2. Get lengths (Now we can just grab them from the dict)
    input_lengths = torch.tensor([item['input_length'] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    
    # 3. Pad Inputs (Batch, Max_Time, 512)
    padded_inputs = pad_sequence(input_features_list, batch_first=True, padding_value=0.0)
    
    # 4. Pad Targets (Batch, Max_Len) - Padding with 0 for CTC blank
    padded_targets = pad_sequence(target_ids_list, batch_first=True, padding_value=0)
    
    return {
        'input_features': padded_inputs,
        'input_lengths': input_lengths,
        'target_ids': padded_targets,
        'target_lengths': target_lengths,
        'transcriptions': transcriptions
    }