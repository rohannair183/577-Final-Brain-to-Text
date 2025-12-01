"""
Train a speech model from scratch using local HDF5 data under ./data.

Expected layout:
  data/
    session_1/
      data_train.hdf5
      data_val.hdf5
      data_test.hdf5
    session_2/
      ...

Each HDF5 file contains trial groups with keys:
  - input_features: float32 [T, F] (T timesteps, F features, e.g., 512)
  - seq_class_ids : int64 [T_labels] (target phoneme indices)

Adjust key names below if yours differ.
"""

import os
import h5py
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm.auto import tqdm


class CFG:
    """Basic configuration."""

    DATA_DIR = os.path.join(os.getcwd(), "data")  # local data folder
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_HEAD = 4  # transformer heads if you switch to TransformerEncModel


print(f"Using device: {CFG.DEVICE}")


def temporal_mask(data, mask_percentage: float = 0.05, mask_value: float = 0.0):
    """
    Randomly mask a percentage of timesteps for robustness.
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)
    seq_len, _ = data.shape
    num_to_mask = int(seq_len * mask_percentage)
    if num_to_mask > 0:
        idx = torch.randperm(seq_len)[:num_to_mask]
        data[idx, :] = mask_value
    return data


class BrainDataset(Dataset):
    """
    One HDF5 split (train/val/test). Each trial group holds:
      - input_key: input_features
      - target_key: seq_class_ids (not present for test; returns empty tensor)
    """

    def __init__(
        self,
        hdf5_file: str,
        input_key: str = "input_features",
        target_key: str = "seq_class_ids",
        is_test: bool = False,
        use_augmentation: bool = False,
    ):
        self.file_path = hdf5_file
        self.input_key = input_key
        self.target_key = target_key
        self.is_test = is_test
        self.use_augmentation = use_augmentation
        self.file = None
        try:
            with h5py.File(self.file_path, "r") as f:
                self.trial_keys = sorted(list(f.keys()))
        except FileNotFoundError:
            print(f"Warning: missing file {self.file_path}; dataset is empty.")
            self.trial_keys = []

    def __len__(self):
        return len(self.trial_keys)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        trial_key = self.trial_keys[idx]
        group = self.file[trial_key]

        x = torch.tensor(group[self.input_key][:], dtype=torch.float32)
        if self.use_augmentation and not self.is_test:
            x = temporal_mask(x, mask_percentage=0.1)

        if self.target_key in group:
            y = torch.tensor(group[self.target_key][:], dtype=torch.long)
        else:
            y = torch.tensor([], dtype=torch.long)

        if self.is_test:
            return x, y, trial_key
        return x, y


def load_datasets(data_dir: str):
    """Combine all session folders into ConcatDatasets."""
    train_sets, val_sets, test_sets = [], [], []
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    print(f"Found {len(subfolders)} session folders in {data_dir}")
    for sub in subfolders:
        train_file = os.path.join(sub, "data_train.hdf5")
        val_file = os.path.join(sub, "data_val.hdf5")
        test_file = os.path.join(sub, "data_test.hdf5")
        train_sets.append(
            BrainDataset(train_file, is_test=False, use_augmentation=True)
        )
        val_sets.append(
            BrainDataset(val_file, is_test=False, use_augmentation=False)
        )
        test_sets.append(
            BrainDataset(test_file, is_test=True, use_augmentation=False)
        )
    return (
        ConcatDataset(train_sets),
        ConcatDataset(val_sets),
        ConcatDataset(test_sets),
    )


train_ds, val_ds, test_ds = load_datasets(CFG.DATA_DIR)
print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")


def ctc_collate(batch):
    """
    Collate for CTC: pad inputs and targets and keep lengths.
    """
    is_test = len(batch[0]) == 3
    if is_test:
        xs, ys, keys = zip(*batch)
    else:
        xs, ys = zip(*batch)
    x_lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    xs_pad = rnn_utils.pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_pad = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=0)
    return (xs_pad, ys_pad, x_lens, y_lens, keys) if is_test else (
        xs_pad,
        ys_pad,
        x_lens,
        y_lens,
    )


train_loader = DataLoader(
    train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, collate_fn=ctc_collate
)
val_loader = DataLoader(
    val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=ctc_collate
)
test_loader = DataLoader(
    test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=ctc_collate
)

# Vocabulary and helpers
VOCAB = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "|",
]
BLANK_ID = 0
TOKEN_MAP = {i + 1: p for i, p in enumerate(VOCAB)}
TOKEN_MAP[BLANK_ID] = ""


def greedy_decode(logits, token_map):
    """
    Collapse repeats/blanks and map to a phoneme string.
    """
    pred_idx = torch.argmax(logits, dim=-1)
    collapsed = torch.unique_consecutive(pred_idx)
    final = [i.item() for i in collapsed if i.item() != BLANK_ID]
    return " ".join(token_map.get(i, "?") for i in final)


class RecurrentModel(nn.Module):
    """
    Adapter -> RNN/GRU/LSTM -> linear -> log_softmax.
    """

    def __init__(
        self,
        model_type: str = "GRU",
        data_in: int = 512,
        adapter_out: int = 256,
        hidden: int = 256,
        output_size: int = len(VOCAB) + 1,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.adapter = nn.Linear(data_in, adapter_out)
        rnn_args = dict(
            input_size=adapter_out,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if model_type == "GRU":
            self.rnn = nn.GRU(**rnn_args)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(**rnn_args)
        else:
            self.rnn = nn.RNN(**rnn_args)
        fc_in = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, output_size)

    def forward(self, x):
        x = self.adapter(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return nn.functional.log_softmax(out, dim=2)


class TransformerEncModel(nn.Module):
    """
    Adapter -> TransformerEncoder -> linear -> log_softmax.
    """

    def __init__(
        self,
        data_in: int = 512,
        adapter_out: int = 256,
        n_head: int = 4,
        num_layers: int = 2,
        dim_ff: int = 512,
        output_size: int = len(VOCAB) + 1,
    ):
        super().__init__()
        self.adapter = nn.Linear(data_in, adapter_out)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=adapter_out,
            nhead=n_head,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(adapter_out, output_size)

    def forward(self, x):
        x = self.adapter(x)
        out = self.encoder(x)
        out = self.fc(out)
        return nn.functional.log_softmax(out, dim=2)


# Choose a model to train from scratch; now using LSTM.
model = RecurrentModel(model_type="LSTM").to(CFG.DEVICE)
criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)


def run_epoch(loader, model, optimizer=None):
    """
    Run one epoch; if optimizer is None, run in eval mode.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    for batch in tqdm(loader, leave=False):
        x, y, x_lens, y_lens = batch
        x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
        x_lens, y_lens = x_lens.to(CFG.DEVICE), y_lens.to(CFG.DEVICE)

        if is_train:
            optimizer.zero_grad()
        y_pred = model(x)  # [B, T, C]
        y_pred = y_pred.permute(1, 0, 2)  # [T, B, C] for CTC
        loss = criterion(y_pred, y, x_lens, y_lens)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


for epoch in range(1, CFG.EPOCHS + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    val_loss = run_epoch(val_loader, model, optimizer=None)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


model.eval()
pred_texts, trial_keys = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        x, y, x_lens, y_lens, keys = batch
        x, x_lens = x.to(CFG.DEVICE), x_lens.to(CFG.DEVICE)
        y_pred = model(x)
        for i in range(x.size(0)):
            logits = y_pred[i, : x_lens[i], :]
            pred_texts.append(greedy_decode(logits, TOKEN_MAP))
            trial_keys.append(keys[i])


if trial_keys:
    submission = pd.DataFrame({"id": trial_keys, "text": pred_texts})
    submission.to_csv("submission_scratch.csv", index=False)
    print("Saved submission_scratch.csv")
else:
    print("No test samples found.")
