"""
Quick utility to inspect phoneme IDs decoded to text vs. stored transcriptions.

Usage:
    python3 scripts/inspect_transcriptions.py --config config/big_rnn_config.yaml --split val --num 3
"""

import argparse
import sys

sys.path.append(".")

from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.training.metrics import PhonemeMetrics


def main(args):
    config = load_config(args.config)

    # Build loaders (mode inferred from loss type)
    train_loader, val_loader = create_dataloaders(config)
    loader = val_loader if args.split == "val" and val_loader is not None else train_loader

    # Phoneme-to-char mapping (None by default; fill with your map if you have one)
    metrics = PhonemeMetrics(phoneme_to_char_map=None)

    print(f"Inspecting split='{args.split}' ({'val' if loader is val_loader else 'train'})")

    batch = next(iter(loader))
    target_ids = batch["target_ids"]
    target_lengths = batch["target_lengths"]
    transcriptions = batch["transcriptions"]

    n_samples = min(args.num, len(transcriptions))
    for i in range(n_samples):
        phonemes = target_ids[i][: target_lengths[i]].cpu().numpy()
        decoded = metrics.phonemes_to_text(phonemes)
        print("\n------ Sample", i)
        print("Decoded phonemes:", decoded)
        print("GT transcription:", transcriptions[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/big_rnn_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Which split to sample from",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=3,
        help="How many samples to print from the batch",
    )
    args = parser.parse_args()
    main(args)

