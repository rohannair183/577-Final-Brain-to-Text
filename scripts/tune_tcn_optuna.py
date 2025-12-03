# scripts/tune_tcn_optuna.py

import os
import random

import optuna
import torch
import numpy as np

from src.data.dataloader import create_dataloaders
from src.models import get_model
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(trial: optuna.Trial):
    """
    Build a minimal config dict compatible with get_model + Trainer,
    using a SMALL subset of data for fast Optuna tuning.
    """

    # -----------------------------
    # 1) Data: use ONE train + ONE val session for speed
    # -----------------------------
    data_config = {
        "data_root": "data/kaggle_raw/hdf5_data_final",
        "train_sessions": [
            "t15.2023.08.11",
        ],
        "val_sessions": [
            "t15.2023.09.24",
        ],
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle_train": True,
    }

    # -----------------------------
    # 2) Model: TCN hyperparameters (tight search around known-good region)
    # -----------------------------
    model_config = {
        "type": "tcn",
        "input_dim": 512,
        "num_phonemes": 41,  # 0..40 including blank

        "hidden_dim": trial.suggest_categorical(
            "hidden_dim", [256, 384]
        ),
        "num_levels": trial.suggest_int("num_levels", 3, 5),
        "kernel_size": trial.suggest_categorical(
            "kernel_size", [3, 5]
        ),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30, step=0.05),
    }

    # -----------------------------
    # 3) Optimizer & scheduler choice (light, focused)
    # -----------------------------
    optimizer_type = trial.suggest_categorical(
        "optimizer_type", ["adam", "adamw"]
    )

    learning_rate = trial.suggest_float(
        "learning_rate", 3e-4, 3e-3, log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay", 1e-5, 1e-3, log=True
    )

    scheduler_type = "none"
    num_epochs = 4  

    training_config = {
        "num_epochs": num_epochs,

        "loss": {
            "type": "ctc",
            "blank_id": 0,
        },

        "optimizer": {
            "type": optimizer_type,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "momentum": 0.9,
            "nesterov": True,
        },

        "scheduler": {
            "type": scheduler_type,
            # t_max is irrelevant for "none", harmless if present
            "t_max": num_epochs,
            "eta_min": 1e-6,
        },

        "gradient_clip": 1.0,
        "checkpoint_dir": os.path.join(
            "experiments", "optuna_tcn", f"trial_{trial.number}"
        ),
        "save_every_n_epochs": 1000,
        "keep_last_n_checkpoints": 1,
        "early_stopping": {
            "enabled": False,
            "patience": 10,
            "metric": "val_per",
            "mode": "min",
        },
    }

    config = {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "phoneme_to_char_map": None,
    }

    return config


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective:
    - Build config from trial params
    - Train for a few epochs
    - Return validation PER (lower is better)
    """
    set_seed(42 + trial.number)

    config = build_config(trial)

    # Dataloaders
    train_loader, val_loader = create_dataloaders(config["data"])

    # Model
    model = get_model(config["model"])

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config)

    # Train for a few epochs
    num_epochs = config["training"]["num_epochs"]
    trainer.train(num_epochs)

    # Use best validation PER seen during trainer.train
    best_per = trainer.best_val_per

    # Clean up GPU memory between trials
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Trial {trial.number}] best PER = {best_per:.4f}")
    return best_per


def main():
    study = optuna.create_study(
        direction="minimize",  # we want to MINIMIZE PER
        study_name="tcn_optuna_small",
    )

    study.optimize(objective, n_trials=10)

    print("\n=== Optuna finished ===")
    print(f"Best value (PER): {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
