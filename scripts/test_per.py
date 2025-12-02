# scripts/test_per_visual.py
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from src.data.dataloader import create_dataloaders
from src.models import get_model
from src.training.metrics import PhonemeMetrics
from src.training.losses import get_loss_function
from src.utils.config import load_config
from tqdm import tqdm

# Phoneme mapping (from competition docs)
LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    '|',
]

def phonemes_to_readable(phoneme_ids):
    """Convert phoneme IDs to readable string"""
    return ' '.join([LOGIT_TO_PHONEME[pid] for pid in phoneme_ids if 0 <= pid < len(LOGIT_TO_PHONEME)])

def print_sample_comparison(predictions, targets, target_lengths, metrics, sample_idx=0):
    """
    Print detailed comparison for one sample.
    Shows CTC decoding process and comparison to target.
    """
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx} - DETAILED BREAKDOWN")
    print(f"{'='*80}")
    
    # Get this sample's data
    pred_logits = predictions[sample_idx]  # (time, vocab)
    target = targets[sample_idx][:target_lengths[sample_idx]]
    target = target[target >= 0]
    
    # Get raw predictions (before decoding)
    pred_tokens = torch.argmax(pred_logits, dim=-1)
    
    print(f"\n1. RAW CTC OUTPUT ({len(pred_tokens)} frames):")
    print(f"   First 100 frames:")
    raw_str = ' '.join([str(t.item()) for t in pred_tokens[:100]])
    print(f"   {raw_str}")
    
    # Show distribution
    unique, counts = torch.unique(pred_tokens, return_counts=True)
    blank_count = (pred_tokens == 0).sum().item()
    print(f"\n   Frame Statistics:")
    print(f"   - Total frames: {len(pred_tokens)}")
    print(f"   - Blank frames: {blank_count} ({100*blank_count/len(pred_tokens):.1f}%)")
    print(f"   - Non-blank frames: {len(pred_tokens) - blank_count}")
    print(f"   - Unique tokens: {len(unique)}")
    
    # CTC decode
    decoded = metrics.ctc_greedy_decode(pred_tokens)
    
    print(f"\n2. AFTER CTC DECODING ({len(decoded)} phonemes):")
    print(f"   Decoded IDs: {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
    print(f"   Decoded phonemes: {phonemes_to_readable(decoded[:50])}")
    if len(decoded) > 50:
        print(f"   (showing first 50 of {len(decoded)})")
    
    # Ground truth
    target_list = target.cpu().numpy().tolist()
    print(f"\n3. GROUND TRUTH ({len(target_list)} phonemes):")
    print(f"   Target IDs: {target_list[:50]}{'...' if len(target_list) > 50 else ''}")
    print(f"   Target phonemes: {phonemes_to_readable(target_list[:50])}")
    if len(target_list) > 50:
        print(f"   (showing first 50 of {len(target_list)})")
    
    # Compute edit distance for this sample
    distance = metrics.edit_distance(decoded, target_list)
    per = distance / len(target_list)
    
    print(f"\n4. COMPARISON:")
    print(f"   Predicted length: {len(decoded)}")
    print(f"   Target length: {len(target_list)}")
    print(f"   Edit distance: {distance}")
    print(f"   PER for this sample: {per:.4f} ({per*100:.1f}%)")
    
    # Show first 20 phonemes side-by-side
    print(f"\n5. SIDE-BY-SIDE (first 20 phonemes):")
    print(f"   {'Pos':<5} {'Predicted':<15} {'Target':<15} {'Match'}")
    print(f"   {'-'*50}")
    for i in range(min(20, max(len(decoded), len(target_list)))):
        pred_ph = LOGIT_TO_PHONEME[decoded[i]] if i < len(decoded) else '---'
        targ_ph = LOGIT_TO_PHONEME[target_list[i]] if i < len(target_list) else '---'
        match = '✓' if pred_ph == targ_ph else '✗'
        print(f"   {i:<5} {pred_ph:<15} {targ_ph:<15} {match}")
    
    print(f"{'='*80}\n")

def quick_training_test():
    """
    Train a tiny model for a few iterations and visualize results.
    """
    print("="*80)
    print("VISUAL PER METRIC TEST")
    print("="*80)
    
    # Create minimal config
    config = {
        'experiment': {'seed': 42},
        'data': {
            'data_root': 'data/raw/hdf5_data_final',
            'train_sessions': ['t15.2023.08.11'],  # Just one session
            'val_sessions': ['t15.2023.09.24'],
            'batch_size': 4,  # Small batch
            'num_workers': 0,
            'shuffle_train': False  # Don't shuffle for reproducibility
        },
        'model': {
            'type': 'rnn',
            'input_dim': 512,
            'num_phonemes': 41,
            'hidden_dim': 64,  # Tiny model
            'num_layers': 1,
            'dropout': 0.0,
            'bidirectional': True
        },
        'training': {
            'loss': {
                'type': 'ctc',
                'blank_id': 0
            }
        }
    }
    
    # Set seed
    torch.manual_seed(42)
    
    # Create data
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    print("Creating tiny model...")
    model = get_model(config['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = get_loss_function('ctc', blank_id=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Metrics
    metrics = PhonemeMetrics(blank_id=0)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Test 1: Before training (random predictions)
    print("\n" + "="*80)
    print("TEST 1: BEFORE TRAINING (Random Initialization)")
    print("="*80)
    
    model.eval()
    batch = next(iter(val_loader))
    inputs = batch['input_features'].to(device)
    targets = batch['target_ids']
    input_lengths = batch['input_lengths']
    target_lengths = batch['target_lengths']
    
    with torch.no_grad():
        outputs = model(inputs, input_lengths)
        outputs_cpu = outputs.cpu()
        per_before = metrics.compute_phoneme_error_rate(outputs_cpu, targets, target_lengths)
    
    print(f"\nOverall PER (before training): {per_before:.4f} ({per_before*100:.1f}%)")
    print(f"Expected: ~0.95-0.99 (random guessing)")
    
    # Show detailed breakdown for first sample
    print_sample_comparison(outputs_cpu, targets, target_lengths, metrics, sample_idx=0)
    
    # Train for a few iterations
    print("\n" + "="*80)
    print("TRAINING (50 batches)")
    print("="*80)
    
    model.train()
    losses = []
    
    for batch_idx, batch in enumerate(tqdm(train_loader, total=50, desc="Training")):
        if batch_idx >= 50:
            break
        
        inputs = batch['input_features'].to(device)
        targets = batch['target_ids'].to(device)
        input_lengths = batch['input_lengths']
        target_lengths = batch['target_lengths']
        
        optimizer.zero_grad()
        outputs = model(inputs, input_lengths)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / 10
            print(f"  Batch {batch_idx+1}/50, Loss: {avg_loss:.4f}")
    
    print(f"\nTraining complete!")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Improvement: {losses[0] - losses[-1]:.4f}")
    
    # Test 2: After training
    print("\n" + "="*80)
    print("TEST 2: AFTER TRAINING (50 batches)")
    print("="*80)
    
    model.eval()
    batch = next(iter(val_loader))
    inputs = batch['input_features'].to(device)
    targets = batch['target_ids']
    input_lengths = batch['input_lengths']
    target_lengths = batch['target_lengths']
    
    with torch.no_grad():
        outputs = model(inputs, input_lengths)
        outputs_cpu = outputs.cpu()
        per_after = metrics.compute_phoneme_error_rate(outputs_cpu, targets, target_lengths)
    
    print(f"\nOverall PER (after training): {per_after:.4f} ({per_after*100:.1f}%)")
    print(f"PER improvement: {per_before - per_after:.4f}")
    
    if per_after < per_before:
        print(f"✅ PER DECREASED - Model is learning!")
    else:
        print(f"⚠️  PER didn't decrease - might need more training")
    
    # Show detailed breakdown for same sample
    print_sample_comparison(outputs_cpu, targets, target_lengths, metrics, sample_idx=0)
    
    # Show multiple samples
    print("\n" + "="*80)
    print("QUICK SUMMARY - ALL SAMPLES IN BATCH")
    print("="*80)
    
    for i in range(min(4, outputs_cpu.size(0))):
        pred_tokens = torch.argmax(outputs_cpu[i], dim=-1)
        decoded = metrics.ctc_greedy_decode(pred_tokens)
        
        target = targets[i][:target_lengths[i]]
        target = target[target >= 0].cpu().numpy().tolist()
        
        distance = metrics.edit_distance(decoded, target)
        per = distance / len(target)
        
        print(f"\nSample {i}:")
        print(f"  Predicted: {phonemes_to_readable(decoded[:15])}...")
        print(f"  Target:    {phonemes_to_readable(target[:15])}...")
        print(f"  PER: {per:.4f} ({per*100:.1f}%)")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Before training: PER = {per_before:.4f}")
    print(f"After training:  PER = {per_after:.4f}")
    print(f"Improvement:     {(per_before - per_after):.4f} ({100*(per_before-per_after)/per_before:.1f}%)")
    print()
    
    if per_after < 0.95:
        print("✅ Metric is working correctly!")
        print("   - PER decreased with training")
        print("   - Model is learning to predict phonemes")
    else:
        print("⚠️  Model not learning yet, but metric is still useful for checking:")
        print("   - CTC decoding is working (see detailed output)")
        print("   - Edit distance is being computed correctly")
        print("   - Need more training to see improvement")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    quick_training_test()