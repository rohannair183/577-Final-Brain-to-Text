# # # scripts/test_dataloader.py
# # import sys
# # sys.path.append('.')

# # from torch.utils.data import DataLoader
# # from src.data.dataset import BrainToTextDataset, collate_fn

# # # Test on one session
# # dataset = BrainToTextDataset(
# #     "data/raw/hdf5_data_final/t15.2023.08.11/data_train.hdf5"
# # )

# # print(f"Dataset size: {len(dataset)}")

# # # Test single item
# # sample = dataset[0]
# # print(f"\nSingle sample:")
# # print(f"  Input features shape: {sample['input_features'].shape}")
# # print(f"  Input length: {sample['input_length']}")
# # print(f"  Target shape: {sample['target_ids'].shape}")
# # print(f"  Target length: {sample['target_length']}")
# # print(f"  Trial name: {sample['trial_name']}")

# # # Test DataLoader with batching
# # loader = DataLoader(
# #     dataset, 
# #     batch_size=4, 
# #     shuffle=True, 
# #     collate_fn=collate_fn,
# #     num_workers=0  # Start with 0, increase later
# # )

# # print(f"\n\nTesting DataLoader:")
# # for batch in loader:
# #     print(f"Batch input shape: {batch['input_features'].shape}")
# #     print(f"Batch Target shape: {batch['target_ids'].shape}")
# #     print(f"Input lengths: {batch['input_lengths']}")
# #     print(f"Target lengths: {batch['target_lengths']}")
# #     break  # Just test first batch

# # print("\n✅ DataLoader works!")

# # scripts/check_vocab.py
import h5py
import numpy as np

def check_vocab(hdf5_path):
    all_phonemes = set()
    with h5py.File(hdf5_path, 'r') as f:
       for trial_name in f.keys():
          if trial_name.startswith('trial_'):
            phonemes = f[trial_name]['seq_class_ids'][:]
            all_phonemes.update(phonemes[phonemes != 0])
            phoneme_list = sorted(all_phonemes)
    
    print(f"Unique phonemes found: {len(phoneme_list)}")
    print(f"Phoneme IDs: {phoneme_list}")
    print(f"Min ID: {min(phoneme_list)}, Max ID: {max(phoneme_list)}")        
    return len(phoneme_list)

     

# # Check your training data
vocab_size = check_vocab("data/kaggle_raw/hdf5_data_final/t15.2023.08.11/data_train.hdf5")
print(f"\n👉 Update config to: num_phonemes: {vocab_size + 1}")  # +1 for blank

# scripts/verify_ctc_setup.py
import torch
import torch.nn as nn

# Test CTC loss setup
blank_id = 0

# Create dummy data
batch_size = 2
input_length = 100
target_length = 20

log_probs = torch.randn(input_length, batch_size, vocab_size).log_softmax(2)
targets = torch.randint(1, vocab_size, (batch_size, target_length))
input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
target_lengths = torch.full((batch_size,), target_length, dtype=torch.long)

# Flatten targets
targets_flat = targets.flatten()

# Test CTC loss
ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

try:
    loss = ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)
    print(f"✅ CTC loss works: {loss.item():.4f}")
    print(f"Blank ID: {blank_id}")
    print(f"Vocab size: {vocab_size}")
except Exception as e:
    print(f"❌ CTC loss failed: {e}")