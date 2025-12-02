# # # # scripts/test_dataloader.py
# # # import sys
# # # sys.path.append('.')

# # # from torch.utils.data import DataLoader
# # # from src.data.dataset import BrainToTextDataset, collate_fn

# # # # Test on one session
# # # dataset = BrainToTextDataset(
# # #     "data/raw/hdf5_data_final/t15.2023.08.11/data_train.hdf5"
# # # )

# # # print(f"Dataset size: {len(dataset)}")

# # # # Test single item
# # # sample = dataset[0]
# # # print(f"\nSingle sample:")
# # # print(f"  Input features shape: {sample['input_features'].shape}")
# # # print(f"  Input length: {sample['input_length']}")
# # # print(f"  Target shape: {sample['target_ids'].shape}")
# # # print(f"  Target length: {sample['target_length']}")
# # # print(f"  Trial name: {sample['trial_name']}")

# # # # Test DataLoader with batching
# # # loader = DataLoader(
# # #     dataset, 
# # #     batch_size=4, 
# # #     shuffle=True, 
# # #     collate_fn=collate_fn,
# # #     num_workers=0  # Start with 0, increase later
# # # )

# # # print(f"\n\nTesting DataLoader:")
# # # for batch in loader:
# # #     print(f"Batch input shape: {batch['input_features'].shape}")
# # #     print(f"Batch Target shape: {batch['target_ids'].shape}")
# # #     print(f"Input lengths: {batch['input_lengths']}")
# # #     print(f"Target lengths: {batch['target_lengths']}")
# # #     break  # Just test first batch

# # # print("\n✅ DataLoader works!")

# # # scripts/check_vocab.py
# # import h5py
# # import numpy as np

# # def check_vocab(hdf5_path):
# #     all_phonemes = set()
    
# #     with h5py.File(hdf5_path, 'r') as f:
# #         for trial_name in f.keys():
# #             if trial_name.startswith('trial_'):
# #                 phonemes = f[trial_name]['seq_class_ids'][:]
# #                 all_phonemes.update(phonemes[phonemes != 0])
    
# #     phoneme_list = sorted(all_phonemes)
# #     print(f"Unique phonemes found: {len(phoneme_list)}")
# #     print(f"Phoneme IDs: {phoneme_list}")
# #     print(f"Min ID: {min(phoneme_list)}, Max ID: {max(phoneme_list)}")
    
# #     return len(phoneme_list)

# # # Check your training data
# # vocab_size = check_vocab("data/raw/hdf5_data_final/t15.2023.08.11/data_train.hdf5")
# # print(f"\n👉 Update config to: num_phonemes: {vocab_size + 1}")  # +1 for blank

# # scripts/verify_ctc_setup.py
# import torch
# import torch.nn as nn

# # Test CTC loss setup
# vocab_size = 50  # Update with actual vocab size
# blank_id = 0

# # Create dummy data
# batch_size = 2
# input_length = 100
# target_length = 20

# log_probs = torch.randn(input_length, batch_size, vocab_size).log_softmax(2)
# targets = torch.randint(1, vocab_size, (batch_size, target_length))
# input_lengths = torch.full((batch_size,), input_length, dtype=torch.long)
# target_lengths = torch.full((batch_size,), target_length, dtype=torch.long)

# # Flatten targets
# targets_flat = targets.flatten()

# # Test CTC loss
# ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

# try:
#     loss = ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)
#     print(f"✅ CTC loss works: {loss.item():.4f}")
#     print(f"Blank ID: {blank_id}")
#     print(f"Vocab size: {vocab_size}")
# except Exception as e:
#     print(f"❌ CTC loss failed: {e}")

# scripts/debug_ctc.py
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from src.data.dataloader import create_dataloaders
from src.models import get_model
from src.utils.config import load_config

def debug_ctc():
    # Load config
    config = load_config('config/rnn_baseline_config.yaml')
    config['data']['num_workers'] = 0  # Easier debugging
    config['training']['loss']['type'] = 'ctc'  # Force CTC mode
    
    # Create dataloader
    train_loader, _ = create_dataloaders(config)
    
    # Get first batch
    batch = next(iter(train_loader))
    
    inputs = batch['input_features']
    targets = batch['target_ids']
    input_lengths = batch['input_lengths']
    target_lengths = batch['target_lengths']
    
    print("="*60)
    print("BATCH ANALYSIS")
    print("="*60)
    print(f"Batch size: {inputs.size(0)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"\nInput lengths: {input_lengths}")
    print(f"Target lengths: {target_lengths}")
    
    # CRITICAL CHECK 1: Input length >= Target length for all samples
    print("\n" + "="*60)
    print("CHECK 1: Input lengths >= Target lengths (CTC requirement)")
    print("="*60)
    all_valid = True
    for i in range(len(input_lengths)):
        valid = input_lengths[i] >= target_lengths[i]
        status = "✅" if valid else "❌"
        print(f"  Sample {i}: input={input_lengths[i]}, target={target_lengths[i]} {status}")
        if not valid:
            all_valid = False
    
    if not all_valid:
        print("\n❌ CRITICAL ERROR: Some input lengths < target lengths!")
        print("CTC cannot work if input sequence is shorter than target!")
        return
    
    # CRITICAL CHECK 2: Target values in valid range
    print("\n" + "="*60)
    print("CHECK 2: Target values")
    print("="*60)
    for i in range(min(3, len(targets))):
        target = targets[i][:target_lengths[i]]
        valid_target = target[target >= 0]
        print(f"  Sample {i}:")
        print(f"    Values: {valid_target[:15].numpy()}")
        print(f"    Min: {valid_target.min().item()}, Max: {valid_target.max().item()}")
        
        if valid_target.max() >= 41:
            print(f"    ❌ ERROR: Target values >= 41 (vocab size)")
        if valid_target.min() < 0 and (valid_target != -1).any():
            print(f"    ❌ ERROR: Invalid negative values (not padding)")
    
    # CHECK 3: Create model and test forward pass
    print("\n" + "="*60)
    print("CHECK 3: Model forward pass")
    print("="*60)
    model = get_model(config['model'])
    model.eval()
    
    with torch.no_grad():
        outputs = model(inputs, input_lengths)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
    print(f"Contains NaN: {torch.isnan(outputs).any()}")
    print(f"Contains Inf: {torch.isinf(outputs).any()}")
    
    # CHECK 4: Test CTC loss computation step by step
    print("\n" + "="*60)
    print("CHECK 4: CTC loss computation")
    print("="*60)
    
    # Prepare for CTC
    log_probs = torch.log_softmax(outputs, dim=-1)
    print(f"Log probs range: [{log_probs.min():.2f}, {log_probs.max():.2f}]")
    print(f"Log probs contains NaN: {torch.isnan(log_probs).any()}")
    
    # Transpose: (batch, time, classes) -> (time, batch, classes)
    log_probs = log_probs.transpose(0, 1)
    print(f"After transpose: {log_probs.shape}")
    
    # Flatten targets
    targets_list = []
    for i in range(targets.size(0)):
        target = targets[i, :target_lengths[i]]
        target = target[target >= 0]
        targets_list.append(target)
    
    targets_concat = torch.cat(targets_list)
    print(f"Concatenated targets shape: {targets_concat.shape}")
    print(f"Targets range: [{targets_concat.min()}, {targets_concat.max()}]")
    
    # Convert to correct types
    log_probs = log_probs.float().contiguous()
    targets_concat = targets_concat.int().contiguous()  # int32!
    input_lengths_int = input_lengths.int().contiguous()
    target_lengths_int = target_lengths.int().contiguous()
    
    print(f"\nData types:")
    print(f"  log_probs: {log_probs.dtype}")
    print(f"  targets: {targets_concat.dtype}")
    print(f"  input_lengths: {input_lengths_int.dtype}")
    print(f"  target_lengths: {target_lengths_int.dtype}")
    
    # Compute CTC loss
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    try:
        loss = ctc_loss(log_probs, targets_concat, input_lengths_int, target_lengths_int)
        print(f"\n✅ CTC loss computed: {loss.item():.4f}")
        
        if loss.item() > 50:
            print("⚠️  WARNING: Loss is very high (>50)")
        elif loss.item() > 20:
            print("⚠️  WARNING: Loss is high (>20), but might improve")
        elif torch.isnan(loss) or torch.isinf(loss):
            print("❌ ERROR: Loss is NaN or Inf")
        else:
            print("✅ Loss looks reasonable")
        
        # Test backward pass
        loss.backward()
        print("✅ Backward pass successful")
        
    except Exception as e:
        print(f"❌ CTC loss failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    debug_ctc()
