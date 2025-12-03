# optuna_tune.py
import optuna
import yaml
import os
from src.models import get_model
from src.training.trainer import Trainer
from src.data.dataloader import create_dataloaders

def objective(trial):
    # Load base config
    with open("config/cnn_bilstm.yaml") as f:
        config = yaml.safe_load(f)
    
    # Model hyperparameters
    config['model']['lstm_hidden'] = trial.suggest_int("lstm_hidden", 128, 256, step=64)
    config['model']['lstm_layers'] = trial.suggest_int("lstm_layers", 1, 4)
    config['model']['conv_out'] = trial.suggest_int("conv_out", 16, 256, step=16)
    config['model']['conv_kernel_time'] = trial.suggest_int("conv_kernel_time", 3, 51, step=2)

    # Training hyperparameters
    config['training']['optimizer']['learning_rate'] = trial.suggest_float("learning_rate", 0.00001, 0.0005 , log=True)
    config['training']['optimizer']['weight_decay'] = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    config['training']['gradient_clip'] = trial.suggest_float("gradient_clip", 0.1, 3.0)
    config['data']['batch_size'] = trial.suggest_categorical("batch_size", [4, 8, 16])

    # Create model and data
    model = get_model(config['model'])
    train_loader, val_loader = create_dataloaders(config)
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train for a few epochs to save time
    trainer.train(num_epochs=5)
    
    # Return final validation PER (Minimize PER)
    return trainer.val_pers[-1]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Create optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="cnn_bilstm_hyperparam_search",
        storage=None
    )

    study.optimize(objective, timeout=10800, n_trials=50)  

    # Save trial results
    df = study.trials_dataframe()
    df.to_csv("results/cnn_bilstm_optuna_trials.csv", index=False)
    
    print("Best trial parameters:", study.best_trial.params)
    print("Best validation PER:", study.best_value)
