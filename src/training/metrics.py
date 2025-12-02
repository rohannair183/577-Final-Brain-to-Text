# src/training/metrics.py
import torch
import numpy as np
from jiwer import wer, cer
from typing import Sequence

class PhonemeMetrics:
    def __init__(self, phoneme_to_char_map=None):
        """
        Args:
            phoneme_to_char_map: Dict mapping phoneme_id -> character(s)
                                Or None to just compute Phoneme Error Rate
        """
        self.p2c_map = phoneme_to_char_map
    
    def phonemes_to_text(self, phoneme_ids):
        """
        Convert phoneme IDs to text.
        
        Args:
            phoneme_ids: numpy array or list of phoneme IDs
        
        Returns:
            text string
        """
        if self.p2c_map is None:
            # If no mapping, just return phoneme IDs as string for debugging
            return " ".join(str(p) for p in phoneme_ids if p > 0)
        
        chars = []
        for pid in phoneme_ids:
            if pid > 0:  # Skip padding/blank
                chars.append(self.p2c_map.get(int(pid), '?'))
        return "".join(chars)
    
    def compute_phoneme_error_rate(self, predictions, targets, target_lengths):
        """
        Compute Phoneme Error Rate (PER) - like WER but for phonemes.
        
        Args:
            predictions: (batch, seq_len, vocab_size) logits
            targets: (batch, seq_len) phoneme IDs (with -1 padding)
            target_lengths: (batch,) actual lengths
        """
        """
        Improved PER computation:
        - Argmax the predictions
        - For each sequence collapse repeated predictions and remove CTC blank (assumed id=0)
        - Remove padding from targets (padding id = -1)
        - Compute Levenshtein edit distance between predicted phoneme sequence and target sequence
        """
        pred_phonemes = torch.argmax(predictions, dim=-1)  # (batch, seq_len)

        def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
            # simple DP implementation
            la, lb = len(a), len(b)
            if la == 0:
                return lb
            if lb == 0:
                return la
            dp = np.zeros((la + 1, lb + 1), dtype=int)
            for i in range(la + 1):
                dp[i, 0] = i
            for j in range(lb + 1):
                dp[0, j] = j
            for i in range(1, la + 1):
                for j in range(1, lb + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i, j] = min(
                        dp[i - 1, j] + 1,      # deletion
                        dp[i, j - 1] + 1,      # insertion
                        dp[i - 1, j - 1] + cost  # substitution
                    )
            return int(dp[la, lb])

        total_errors = 0
        total_phonemes = 0

        BLANK_ID = 0
        for pred, target, length in zip(pred_phonemes, targets, target_lengths):
            # pred: Tensor(seq_len,), target: Tensor(seq_len or target_len,)
            pred_seq = pred[:length].cpu().numpy().tolist()

            # Collapse repeats and remove blanks (CTC-style decoding)
            collapsed = []
            prev = None
            for p in pred_seq:
                if p == prev:
                    prev = p
                    continue
                prev = p
                if int(p) == BLANK_ID:
                    continue
                collapsed.append(int(p))

            # Target: remove padding (-1)
            target_seq = target[:length].cpu().numpy()
            target_seq = [int(x) for x in target_seq if int(x) != -1]

            errors = levenshtein(collapsed, target_seq)
            total_errors += errors
            total_phonemes += len(target_seq)

        per = total_errors / max(total_phonemes, 1)
        return per
    
    def compute_metrics(self, predictions, targets, target_lengths, transcriptions=None):
        """
        Compute all metrics: PER and optionally CER/WER if phoneme mapping exists.
        
        Args:
            predictions: (batch, seq_len, vocab_size)
            targets: (batch, seq_len) phoneme IDs
            target_lengths: (batch,) actual lengths
            transcriptions: List of ground truth text strings (optional)
        
        Returns:
            dict with metrics
        """
        # Always compute Phoneme Error Rate
        per = self.compute_phoneme_error_rate(predictions, targets, target_lengths)
        
        metrics = {'per': per}
        
        # If we have phoneme-to-char mapping and ground truth text, compute WER/CER
        if self.p2c_map is not None and transcriptions is not None:
            pred_phonemes = torch.argmax(predictions, dim=-1)
            
            pred_texts = []
            for pred, length in zip(pred_phonemes, target_lengths):
                pred_seq = pred[:length].cpu().numpy()
                pred_text = self.phonemes_to_text(pred_seq)
                pred_texts.append(pred_text)
            
            # Compute WER/CER against ground truth transcriptions
            try:
                word_error_rate = wer(transcriptions, pred_texts)
                char_error_rate = cer(transcriptions, pred_texts)
                metrics['wer'] = word_error_rate
                metrics['cer'] = char_error_rate
            except:
                # If error in computing, skip
                pass
        
        return metrics


# Create phoneme vocabulary explorer
def explore_phoneme_vocab(hdf5_path):
    """Find all unique phoneme IDs in dataset"""
    import h5py
    
    all_phonemes = set()
    with h5py.File(hdf5_path, 'r') as f:
        for trial_name in list(f.keys())[:100]:  # Sample first 100
            if trial_name.startswith('trial_'):
                phonemes = f[trial_name]['seq_class_ids'][:]
                all_phonemes.update(phonemes[phonemes != 0])
    
    phoneme_list = sorted(all_phonemes)
    print(f"Found {len(phoneme_list)} unique phonemes")
    print(f"Phoneme IDs: {phoneme_list}")
    return phoneme_list


if __name__ == "__main__":
    # Explore vocabulary
    phonemes = explore_phoneme_vocab(
        "data/raw/hdf5_data_final/t15.2023.08.11/data_train.hdf5"
    )
    
    # For now, use PER-only metrics (no phoneme-to-char mapping)
    metrics = PhonemeMetrics(phoneme_to_char_map=None)
    
    # Test
    preds = torch.randn(2, 10, 50)  # (batch=2, seq=10, vocab=50)
    targets = torch.tensor([[1,2,3,4,5,-1,-1,-1,-1,-1],
                           [6,7,8,9,10,11,-1,-1,-1,-1]])
    lengths = torch.tensor([5, 6])
    
    results = metrics.compute_metrics(preds, targets, lengths)
    print(f"Results: {results}")

    
