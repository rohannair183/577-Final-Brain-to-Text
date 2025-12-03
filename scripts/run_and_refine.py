"""
Run a trained RNN model for one batch, decode phoneme logits to text,
optionally send the raw hypotheses to an LLM for refinement, and compute WER/CER.

Usage:
    python3 scripts/run_and_refine.py \
        --config config/big_rnn_config.yaml \
        --checkpoint experiments/big_rnn_deep512/best_model.pt \
        --split val \
        --num 4

Replace `call_llm` with your LLM client call to actually refine text.
"""

import argparse
import os
import sys

import torch
from jiwer import wer, cer

sys.path.append(".")

from src.utils.config import load_config
from src.data.dataloader import create_dataloaders
from src.training.metrics import PhonemeMetrics
from src.data.phoneme_map import PHONEME_TO_CHAR_MAP
from src.models import get_model


def call_llm(raw_pred_text: str, transcript: str, include_reference: bool = True) -> str:
    """
    Refine the raw ASR hypothesis using an LLM.
    Uses the OpenAI client if OPENAI_API_KEY is set; otherwise returns raw_pred_text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return raw_pred_text

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        prompt_lines = [
            "You are cleaning up a noisy ASR transcription.",
            f"Noisy hypothesis: {raw_pred_text}",
        ]
        if include_reference:
            prompt_lines.append(
                f"Reference (for context; do not copy directly): {transcript}"
            )
        prompt_lines.append(
            "Produce the best corrected sentence, concise, lowercased, no extra commentary."
        )
        prompt = "\n".join(prompt_lines)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM refinement failed ({e}); falling back to raw prediction.")
        return raw_pred_text


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ctc_greedy_decode_ids(pred_ids, blank_id: int = 0):
    """
    Collapse repeats and remove blanks for a single sequence of predicted ids.
    """
    collapsed = []
    prev = None
    for pid in pred_ids:
        if pid == prev:
            prev = pid
            continue
        prev = pid
        if int(pid) == blank_id:
            continue
        collapsed.append(int(pid))
    return collapsed


def run(config_path: str, checkpoint_path: str, split: str, num_samples: int):
    config = load_config(config_path)
    train_loader, val_loader = create_dataloaders(config)
    loader = val_loader if split == "val" and val_loader is not None else train_loader

    device = select_device()
    model = get_model(config["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = next(iter(loader))
    inputs = batch["input_features"].to(device)
    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]
    transcriptions = batch["transcriptions"]

    with torch.no_grad():
        outputs = model(inputs, input_lengths)
        if isinstance(outputs, tuple):
            outputs, model_input_lengths = outputs
        else:
            model_input_lengths = input_lengths

    metrics = PhonemeMetrics(phoneme_to_char_map=PHONEME_TO_CHAR_MAP)

    n = min(num_samples, len(transcriptions))
    raw_hyps, refined_hyps = [], []

    include_ref = os.getenv("INCLUDE_REF_IN_LLM_PROMPT", "1") != "0"

    for i in range(n):
        pred_ids = torch.argmax(outputs[i], dim=-1)[: model_input_lengths[i]].cpu().numpy()
        decoded_ids = ctc_greedy_decode_ids(pred_ids, blank_id=0)
        raw_text = metrics.phonemes_to_text(decoded_ids)
        refined_text = call_llm(raw_text, transcriptions[i], include_reference=include_ref)
        raw_hyps.append(raw_text)
        refined_hyps.append(refined_text)

    wer_raw = wer(transcriptions[:n], raw_hyps)
    cer_raw = cer(transcriptions[:n], raw_hyps)
    wer_refined = wer(transcriptions[:n], refined_hyps)
    cer_refined = cer(transcriptions[:n], refined_hyps)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples evaluated: {n}")
    print("\nSample predictions:")
    for j in range(n):
        print(f"[{j}] raw     : {raw_hyps[j]}")
        print(f"    refined : {refined_hyps[j]}")
        print(f"    truth   : {transcriptions[j]}")

    print("\nMetrics:")
    print(f"WER raw     : {wer_raw:.3f}")
    print(f"CER raw     : {cer_raw:.3f}")
    print(f"WER refined : {wer_refined:.3f}")
    print(f"CER refined : {cer_refined:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode, refine, and score WER/CER.")
    parser.add_argument("--config", type=str, default="config/big_rnn_config.yaml",
                        help="Path to config file.")
    parser.add_argument("--checkpoint", type=str, default="experiments/big_rnn_deep512/best_model.pt",
                        help="Path to model checkpoint.")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val",
                        help="Which split to sample from.")
    parser.add_argument("--num", type=int, default=4,
                        help="How many samples from the batch to decode.")
    args = parser.parse_args()

    run(args.config, args.checkpoint, args.split, args.num)
