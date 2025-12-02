# src/training/metrics.py - CORRECTED FOR CTC

import torch
import numpy as np
from jiwer import wer, cer

class PhonemeMetrics:
    def __init__(self, phoneme_to_char_map=None, blank_id=0):
        """
        Args:
            phoneme_to_char_map: Optional mapping for WER/CER
            blank_id: CTC blank token (usually 0)
        """
        self.p2c_map = phoneme_to_char_map
        self.blank_id = blank_id
    
    def ctc_greedy_decode(self, prediction):
        """
        Greedy CTC decoding: collapse blanks and remove consecutive duplicates.
        
        Args:
            prediction: (time_steps,) tensor of predicted token IDs
        
        Returns:
            List of decoded phoneme IDs
        """
        decoded = []
        prev_token = None
        
        for token_id in prediction:
            token_id = token_id.item()
            
            # Skip blanks
            if token_id == self.blank_id:
                prev_token = None
                continue
            
            # Skip consecutive duplicates
            if token_id != prev_token:
                decoded.append(token_id)
                prev_token = token_id
        
        return decoded
    
    def edit_distance(self, seq1, seq2):
        """
        Compute Levenshtein edit distance.
        """
        len1, len2 = len(seq1), len(seq2)
        
        # DP table
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )
        
        return dp[len1][len2]
    
    def compute_phoneme_error_rate(self, predictions, targets, target_lengths):
        """
        Compute Phoneme Error Rate with proper CTC decoding.
        
        Args:
            predictions: (batch, time_steps, vocab_size) logits
            targets: (batch, max_target_len) phoneme IDs
            target_lengths: (batch,) actual target lengths
        """
        # Get most likely tokens
        pred_tokens = torch.argmax(predictions, dim=-1)  # (batch, time_steps)
        
        total_distance = 0
        total_phonemes = 0
        
        for pred, target, length in zip(pred_tokens, targets, target_lengths):
            # STEP 1: CTC decode the FULL prediction sequence
            decoded = self.ctc_greedy_decode(pred)  # Decode all frames!
            
            # STEP 2: Get ground truth (remove padding)
            target_seq = target[:length].cpu().numpy()
            target_seq = target_seq[target_seq != -1]
            target_list = target_seq.tolist()
            
            # STEP 3: Compute edit distance
            distance = self.edit_distance(decoded, target_list)
            
            total_distance += distance
            total_phonemes += len(target_list)
        
        per = total_distance / max(total_phonemes, 1)
        return per
    
    def compute_metrics(self, predictions, targets, target_lengths, transcriptions=None):
        """
        Compute all metrics with proper CTC decoding.
        
        Args:
            predictions: (batch, time_steps, vocab_size)
            targets: (batch, max_target_len)
            target_lengths: (batch,)
            transcriptions: Optional list of text strings
        
        Returns:
            dict with metrics
        """
        # Compute PER with proper CTC decoding
        per = self.compute_phoneme_error_rate(predictions, targets, target_lengths)
        
        metrics = {'per': per}
        
        # TODO: WER/CER if needed
        
        return metrics


# # Test the corrected implementation
# if __name__ == "__main__":
#     print("Testing CTC decoding...")
    
#     metrics = PhonemeMetrics(blank_id=0)
    
#     # Test 1: CTC decoding
#     print("\nTest 1: CTC Decoding")
#     # Simulate CTC output: [blank, 5, 5, blank, 7, blank, 12, 12]
#     prediction = torch.tensor([0, 5, 5, 0, 7, 0, 12, 12])
#     decoded = metrics.ctc_greedy_decode(prediction)
#     print(f"  Input:    {prediction.numpy()}")
#     print(f"  Decoded:  {decoded}")
#     print(f"  Expected: [5, 7, 12]")
#     assert decoded == [5, 7, 12], "❌ CTC decoding failed!"
#     print("  ✅ PASS")
    
#     # Test 2: Perfect predictions (PER should be 0)
#     print("\nTest 2: Perfect Predictions")
#     predictions = torch.zeros(2, 100, 41)
#     targets = torch.tensor([[5, 7, 12, -1, -1],
#                            [3, 8, 15, -1, -1]])
#     target_lengths = torch.tensor([3, 2])
    
#     # Make predictions match targets perfectly
#     # Sample 0: encode [5, 7, 12] in CTC format
#     predictions[0, 0:10, 5] = 10.0   # Phoneme 5
#     predictions[0, 10:20, 7] = 10.0  # Phoneme 7
#     predictions[0, 20:30, 12] = 10.0 # Phoneme 12
#     predictions[0, 30:, 0] = 10.0    # Blanks
    
#     # Sample 1: encode [3, 8] in CTC format
#     predictions[1, 0:10, 3] = 10.0
#     predictions[1, 10:20, 8] = 10.0
#     predictions[1, 20:, 0] = 10.0
    
#     per = metrics.compute_phoneme_error_rate(predictions, targets, target_lengths)
#     print(f"  PER: {per:.4f}")
#     print(f"  Expected: 0.0")
#     assert per < 0.01, "❌ PER should be 0 for perfect predictions!"
#     print("  ✅ PASS")
    
#     # Test 3: Random predictions (PER should be high)
#     print("\nTest 3: Random Predictions")
#     predictions = torch.randn(2, 100, 41)
#     per = metrics.compute_phoneme_error_rate(predictions, targets, target_lengths)
#     print(f"  PER: {per:.4f}")
#     print(f"  Expected: ~0.9-1.0")
#     assert per > 0.7, "❌ PER should be high for random!"
#     print("  ✅ PASS")
    
#     print("\n" + "="*60)
#     print("✅ ALL TESTS PASSED - Metric is now correct!")
#     print("="*60)