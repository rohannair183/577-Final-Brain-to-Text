# 577-Final-Brain-to-Text

Brain-to-Text Final Project

### Please note that the dataset required for this project is not included in this file. It must be downloaded externally.

# Brain-to-Text Local Training

Quick start for using the local HDF5 dataloader and training script.

The script uses an LSTM baseline by default and is organized so teammates can easily swap in GRU/RNN/Transformer or add their own models (e.g., CNN front-ends).

## Data layout
- Place session folders under `./data/`.
- Each session folder should contain:
  - `data_train.hdf5`
  - `data_val.hdf5`
  - `data_test.hdf5` (optional for inference/submission)
- Each HDF5 trial group is expected to have:
  - `input_features`: float32 of shape `[T, F]` (T timesteps, F features)
  - `seq_class_ids`: int64 of shape `[T_labels]` (phoneme indices)
- If your key names differ, change `input_key` and `target_key` when constructing `BrainDataset` or update the defaults in `src/dataloder.py`.

## Train from scratch
From the repo root:
```bash
python src/dataloder.py
```
This trains an LSTM+CTC model from scratch and, if `data_test.hdf5` exists, writes `submission_scratch.csv` in the repo root.

## Switching models
- LSTM/GRU/RNN: change `model_type` in `RecurrentModel` when instantiating the model in `main()`.
- Transformer encoder: swap to `TransformerEncModel(...)` and adjust heads/layers/feedforward size.
- Custom (e.g., CNN front-end): create your own `nn.Module` that outputs `[B, T, C]` logits, then plug it into the same training loop and CTCLoss.
- If your feature dimension is not 512, update `data_in` in the model init to match.

## Hyperparameters
Edit the `CFG` class in `src/dataloder.py` to adjust batch size, epochs, learning rate, device, or transformer heads. You can also tweak `temporal_mask` or the `use_augmentation` flags in `load_datasets`.

## Reuse in custom scripts
Import components for custom experiments:
```python
from src.dataloder import (
    BrainDataset,
    ctc_collate,
    RecurrentModel,
    TransformerEncModel,
    CFG,
)
```
Then build your own training loop or plug the datasets into other trainers. The file is structured with big section headers so it’s clear where the dataloader lives and where the baseline models begin.


