<<<<<<< HEAD
# src/training/metrics.py

=======
>>>>>>> 14188cc41c5b4ce968690566eb32871b7aae3790
import torch
import numpy as np
from jiwer import wer, cer

class PhonemeMetrics:
    def __init__(self, phoneme_to_char_map=None, blank_id=0):
        """
        Args:
<<<<<<< HEAD
            phoneme_to_char_map: Dict mapping phoneme_id -> character(s) (optional)
            blank_id: ID used for the CTC blank symbol (default 0)
        """
        self.p2c_map = phoneme_to_char_map
        self.blank_id = blank_id

    # ---------- Helpers ----------

    def _collapse_ctc(self, seq):
        """
        Apply CTC collapsing:
          - remove blanks
          - collapse repeated symbols
        seq: 1D array/list of ints
        """
        collapsed = []
        prev = None
        for s in seq:
            s = int(s)
            if s == self.blank_id:
                prev = s
                continue
            if s == prev:
                continue
            collapsed.append(s)
            prev = s
        return collapsed

    def _edit_distance(self, ref, hyp):
        """
        Levenshtein edit distance between two sequences of ints.
        """
        n, m = len(ref), len(hyp)
        if n == 0:
            return m
        if m == 0:
            return n

        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        for i in range(n + 1):
            dp[i, 0] = i
        for j in range(m + 1):
            dp[0, j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,      # deletion
                    dp[i, j - 1] + 1,      # insertion
                    dp[i - 1, j - 1] + cost  # substitution
                )
        return int(dp[n, m])

    # ---------- Text conversion (optional) ----------

=======
            phoneme_to_char_map: Optional mapping for WER/CER
            blank_id: CTC blank token (usually 0)
        """
        self.p2c_map = phoneme_to_char_map
        self.blank_id = blank_id
    
>>>>>>> 14188cc41c5b4ce968690566eb32871b7aae3790
    def phonemes_to_text(self, phoneme_ids):
            if self.p2c_map is None:
                # If no mapping, just return phoneme IDs as string for debugging
                return " ".join(str(p) for p in phoneme_ids if p > 0)
            
            chars = []
            for pid in phoneme_ids:
                if pid > 0:  # Skip padding/blank
                    chars.append(self.p2c_map.get(int(pid), '?'))
            return "".join(chars)
        
    def ctc_greedy_decode(self, prediction):
        """
<<<<<<< HEAD
        Convert phoneme IDs to text using phoneme_to_char_map.
        If no mapping is provided, join IDs as a space-separated string.
        """
        if self.p2c_map is None:
            return " ".join(str(p) for p in phoneme_ids if p > 0)

        chars = []
        for pid in phoneme_ids:
            if pid > 0:  # skip blank/padding
                chars.append(self.p2c_map.get(int(pid), '?'))
        return "".join(chars)

    # ---------- PER computation ----------

    def compute_phoneme_error_rate(self, predictions, targets, input_lengths, target_lengths):
        """
        Compute Phoneme Error Rate (PER) using CTC-style decoding + edit distance.

        Args:
            predictions: (batch, T, vocab_size) logits
            targets:     (batch, max_target_len) label IDs
            input_lengths:  (batch,) valid time lengths for each prediction
            target_lengths: (batch,) number of valid labels per sample
        """
        # Greedy CTC decoding
        pred_ids = torch.argmax(predictions, dim=-1)  # (B, T)

        total_edits = 0
        total_phonemes = 0

        for i in range(pred_ids.size(0)):
            T = int(input_lengths[i].item())
            L = int(target_lengths[i].item())

            # Predicted sequence (apply CTC collapse)
            pred_seq = pred_ids[i, :T].cpu().numpy()
            pred_seq = self._collapse_ctc(pred_seq)

            # Ground-truth label sequence
            target_seq = targets[i, :L].cpu().numpy().tolist()

            # Edit distance between sequences
            dist = self._edit_distance(target_seq, pred_seq)

            total_edits += dist
            total_phonemes += len(target_seq)

        if total_phonemes == 0:
            return 1.0  # avoid divide-by-zero; treat as terrible

        per = total_edits / total_phonemes
=======
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
>>>>>>> 14188cc41c5b4ce968690566eb32871b7aae3790
        return per

    # ---------- Overall metrics ----------

    def compute_metrics(self, predictions, targets, input_lengths, target_lengths, transcriptions=None):
        """
<<<<<<< HEAD
        Compute PER and optionally WER/CER if phoneme_mapping + transcriptions exist.

        Args:
            predictions:    (batch, T, vocab_size) logits
            targets:        (batch, max_target_len) label IDs
            input_lengths:  (batch,)
            target_lengths: (batch,)
            transcriptions: list of true text strings (optional)

=======
        Compute all metrics with proper CTC decoding.
        
        Args:
            predictions: (batch, time_steps, vocab_size)
            targets: (batch, max_target_len)
            target_lengths: (batch,)
            transcriptions: Optional list of text strings
        
>>>>>>> 14188cc41c5b4ce968690566eb32871b7aae3790
        Returns:
            dict with keys: 'per' (and optionally 'wer', 'cer')
        """
<<<<<<< HEAD
        per = self.compute_phoneme_error_rate(
            predictions, targets, input_lengths, target_lengths
        )

        metrics = {"per": per}

        # Optional: WER/CER if we can map phonemes -> characters
        if self.p2c_map is not None and transcriptions is not None:
            pred_ids = torch.argmax(predictions, dim=-1)

            pred_texts = []
            for i in range(pred_ids.size(0)):
                T = int(input_lengths[i].item())
                seq = pred_ids[i, :T].cpu().numpy()
                seq = self._collapse_ctc(seq)
                pred_texts.append(self.phonemes_to_text(seq))

            try:
                metrics["wer"] = wer(transcriptions, pred_texts)
                metrics["cer"] = cer(transcriptions, pred_texts)
            except Exception:
                # If jiwer complains, just skip WER/CER
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
=======
        # Compute PER with proper CTC decoding
        per = self.compute_phoneme_error_rate(predictions, targets, target_lengths)
        
        metrics = {'per': per}
>>>>>>> 14188cc41c5b4ce968690566eb32871b7aae3790
