# src/data/dataset.py - FINAL FLEXIBLE VERSION

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class BrainToTextDataset(Dataset):
    def __init__(self, hdf5_path, mode='ctc'):
        """
        Args:
            hdf5_path: Path to HDF5 file
            mode: 'ctc' or 'frame_level'
                  - 'ctc': Returns raw phoneme sequence (variable length)
                  - 'frame_level': Returns frame-aligned targets (same length as input)
        """
        self.hdf5_path = hdf5_path
        self.mode = mode
        
        with h5py.File(hdf5_path, 'r') as f:
            self.trial_names = [key for key in f.keys() if key.startswith('trial_')]
        
        print(f"Loaded {len(self.trial_names)} trials from {hdf5_path}")
        print(f"Dataset mode: {self.mode}")
    
    def __len__(self):
        return len(self.trial_names)
    
    def __getitem__(self, idx):
        trial_name = self.trial_names[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            trial = f[trial_name]
            
            input_features = trial['input_features'][:]
            raw_seq_ids = trial['seq_class_ids'][:]
            raw_transcription = trial['transcription'][:]

        # Neural features
        input_tensor = torch.FloatTensor(input_features)
        n_frames = input_tensor.shape[0]
        
        # Clean phoneme sequence (remove padding)
        phoneme_sequence = raw_seq_ids[raw_seq_ids != 0]
        
        # Choose target format based on mode
        if self.mode == 'ctc':
            # CTC: raw phoneme sequence (variable length)
            target_ids = torch.LongTensor(phoneme_sequence)
            target_length = len(phoneme_sequence)
        
        elif self.mode == 'frame_level':
            # Frame-level: align phonemes to input frames
            target_ids = self._align_phonemes_to_frames(phoneme_sequence, n_frames)
            target_length = n_frames
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'ctc' or 'frame_level'")
        
        # Text for reference
        valid_chars = raw_transcription[raw_transcription != 0]
        transcription_str = "".join([chr(c) for c in valid_chars])

        return {
            'input_features': input_tensor,
            'target_ids': target_ids,
            'transcription': transcription_str,
            'trial_name': trial_name,
            'input_length': n_frames,
            'target_length': target_length
        }
    
    def _align_phonemes_to_frames(self, phoneme_ids, n_frames):
        """Uniform alignment for frame-level training"""
        n_phonemes = len(phoneme_ids)
        
        if n_phonemes == 0:
            return torch.full((n_frames,), -1, dtype=torch.long)
        
        aligned = []
        for i in range(n_phonemes):
            start_frame = int(i * n_frames / n_phonemes)
            end_frame = int((i + 1) * n_frames / n_phonemes)
            n_repeat = end_frame - start_frame
            aligned.extend([int(phoneme_ids[i])] * n_repeat)
        
        while len(aligned) < n_frames:
            aligned.append(int(phoneme_ids[-1]))
        
        return torch.LongTensor(aligned[:n_frames])


def collate_fn(batch):
    """Collate function that works for both CTC and frame-level"""
    input_features_list = [item['input_features'] for item in batch]
    target_ids_list = [item['target_ids'] for item in batch]
    transcriptions = [item['transcription'] for item in batch]
    
    input_lengths = torch.tensor([item['input_length'] for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item['target_length'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    padded_inputs = pad_sequence(input_features_list, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(target_ids_list, batch_first=True, padding_value=-1)
    
    return {
        'input_features': padded_inputs,
        'input_lengths': input_lengths,
        'target_ids': padded_targets,
        'target_lengths': target_lengths,
        'transcriptions': transcriptions
    }