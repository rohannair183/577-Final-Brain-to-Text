# src/utils/config.py
import yaml

def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """Save config to YAML"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)