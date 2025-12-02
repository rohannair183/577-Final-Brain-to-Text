# src/models/__init__.py
from .lstm import LSTMDecoder
from .transformer import TransformerDecoder
from .conformer import ConformerDecoder
from .simple_cnn import CNNDecoder
from .tcn import TCNDecoder

def get_model(model_config):
    """
    Factory function to create model based on config.
    
    Args:
        model_config: Dict with 'type' and model-specific parameters
    
    Returns:
        Model instance
    """
    model_type = model_config['type']
    
    if model_type == 'lstm':
        return LSTMDecoder(
            input_dim=model_config.get('input_dim', 512),
            hidden_dim=model_config['hidden_dim'],
            num_phonemes=model_config['num_phonemes'],
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.2),
            bidirectional=model_config.get('bidirectional', True)
        )
    
    elif model_type == 'transformer':
        return TransformerDecoder(
            input_dim=model_config.get('input_dim', 512),
            d_model=model_config['d_model'],
            num_phonemes=model_config['num_phonemes'],
            num_layers=model_config.get('num_layers', 6),
            num_heads=model_config.get('num_heads', 8),
            dropout=model_config.get('dropout', 0.1)
        )
    
    elif model_type == 'conformer':
        return ConformerDecoder(
            input_dim=model_config.get('input_dim', 512),
            encoder_dim=model_config['encoder_dim'],
            num_phonemes=model_config['num_phonemes'],
            num_layers=model_config.get('num_layers', 12),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.1)
        )
    
    elif model_type == 'cnn':
        return CNNDecoder(
            input_dim=model_config.get('input_dim', 512),
            num_phonemes=model_config['num_phonemes'],
            num_layers=model_config.get('num_layers', 4),
            kernel_size=model_config.get('kernel_size', 3)
        )
    
    elif model_type == 'tcn':
        return TCNDecoder(
            input_dim=model_config.get('input_dim', 512),
            num_phonemes=model_config['num_phonemes'],
            num_levels=model_config.get('num_levels', 5),
            hidden_dim=model_config.get('hidden_dim', 256),
            kernel_size=model_config.get('kernel_size', 3),
            dropout=model_config.get('dropout', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")