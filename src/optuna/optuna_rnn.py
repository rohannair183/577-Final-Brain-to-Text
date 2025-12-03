# optuna_tune_rnn.py
import optuna
import yaml
import os
from src.models import get_model
from src.training.trainer import Trainer
from src.data.dataloader import create_dataloaders

def objective(trial):
    # Load base config
    with open("config/baseline_rnn.yaml") as f:
        config = yaml.safe_load(f)
    
    # Model hyperparameters
    config['model']['hidden_size'] = trial.suggest_int("hidden_size", 256, 1024, step=128)
    config['model']['num_layers'] = trial.suggest_int("num_layers", 2, 4)
    config['model']['dropout'] = trial.suggest_float("dropout", 0.1, 0.5)
    config['model']['rnn_type'] = trial.suggest_categorical("rnn_type", ['gru', 'lstm'])
    config['model']['bidirectional'] = trial.suggest_categorical("bidirectional", [True, False])
    
    # Training hyperparameters
    optimizer_type = trial.suggest_categorical("optimizer_type", ['adam', 'adamw', 'sgd'])
    config['training']['optimizer']['type'] = optimizer_type
    config['training']['optimizer']['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    config['training']['optimizer']['weight_decay'] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    config['training']['gradient_clip'] = trial.suggest_float("gradient_clip", 0.5, 5.0)
    
    # CTC loss hyperparameters
    config['training']['loss']['blank_penalty_weight'] = trial.suggest_float("blank_penalty_weight", 0.0, 0.7, step=0.1)
    
    # Data hyperparameters
    config['data']['batch_size'] = trial.suggest_categorical("batch_size", [4, 8, 16, 32])

    # Create model and data
    model = get_model(config['model'])
    train_loader, val_loader = create_dataloaders(config)
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train for a few epochs to save time
    trainer.train(num_epochs=10)
    
    # Return final validation PER (Minimize PER)
    return trainer.val_pers[-1]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Create optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="rnn_hyperparam_search",
        storage=None, 
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()   
    )

    study.optimize(objective, timeout=10800, n_trials=50)  

    # Save trial results
    df = study.trials_dataframe()
    df.to_csv("results/rnn_optuna_trials.csv", index=False)
    
    print("Best trial parameters:", study.best_trial.params)
    print("Best validation PER:", study.best_value)