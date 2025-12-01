"""
Neural Architecture Search Hyperparameter Configuration

Defines parameters used in the architecture search process,
based on Zoph & Le (2017) paper.
"""

from typing import Dict, Any


def get_nas_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Returns NAS hyperparameter configuration.
    
    Args:
        config_name: Configuration name ('default', 'fast', 'thorough')
    
    Returns:
        Dictionary with hyperparameters
    """
    
    configs = {
        'default': {
            # Controller parameters
            'max_layers': 3,
            'components_per_layer': 4,  # [kernel, filters, stride, pool]
            'lstm_hidden_size': 100,
            'controller_lr': 0.99,
            'lr_decay': 0.96,
            'lr_decay_steps': 500,
            'beta': 1e-4,  # L2 regularization
            'baseline_ema_alpha': 0.95,
            
            # Child Network parameters
            'child_lr': 3e-5,
            'child_epochs': 100,
            'child_batch_size': 20,
            'child_dropout': 0.2,
            
            # NAS Search parameters
            'max_episodes': 2000,
            'children_per_episode': 10,
            'save_every': 50,  # Save checkpoint every N episodes
            
            # Paths
            'checkpoint_dir': 'checkpoints/nas',
            'log_dir': 'logs/nas',
        },
        
        'fast': {
            # Fast configuration for testing
            'max_layers': 2,
            'components_per_layer': 4,
            'lstm_hidden_size': 50,
            'controller_lr': 0.99,
            'lr_decay': 0.96,
            'lr_decay_steps': 100,
            'beta': 1e-4,
            'baseline_ema_alpha': 0.95,
            
            'child_lr': 3e-5,
            'child_epochs': 20,
            'child_batch_size': 32,
            'child_dropout': 0.2,
            
            'max_episodes': 100,
            'children_per_episode': 5,
            'save_every': 10,
            
            'checkpoint_dir': 'checkpoints/nas_fast',
            'log_dir': 'logs/nas_fast',
        },
        
        'thorough': {
            # Exhaustive search
            'max_layers': 5,
            'components_per_layer': 4,
            'lstm_hidden_size': 150,
            'controller_lr': 0.99,
            'lr_decay': 0.96,
            'lr_decay_steps': 1000,
            'beta': 1e-4,
            'baseline_ema_alpha': 0.95,
            
            'child_lr': 3e-5,
            'child_epochs': 150,
            'child_batch_size': 16,
            'child_dropout': 0.2,
            
            'max_episodes': 5000,
            'children_per_episode': 15,
            'save_every': 100,
            
            'checkpoint_dir': 'checkpoints/nas_thorough',
            'log_dir': 'logs/nas_thorough',
        },
        
        'nascnn15': {
            # Configuration based on NAS paper (Zoph & Le 2017)
            # Replica los hiperparámetros del paper para CIFAR-10
            
            # Controller parameters (según paper: 2-layer LSTM, 35 hidden units, ADAM lr=0.0006)
            'max_layers': 6,  # Empieza en 6, aumentará progresivamente
            'layer_schedule': True,  # Activar schedule de capas progresivo
            'layer_increment': 2,  # Aumentar 2 capas cada vez
            'increment_every': 1600,  # Cada 1,600 arquitecturas
            'max_layer_limit': 15,  # Límite máximo (NASCNN15)
            
            'components_per_layer': 4,
            'lstm_hidden_size': 35,  # Según paper
            'controller_lr': 0.0006,  # Según paper (ADAM)
            'lr_decay': 0.96,
            'lr_decay_steps': 1000,
            'beta': 1e-4,
            'baseline_ema_alpha': 0.95,
            
            # Child Network parameters (según paper)
            'child_optimizer': 'SGD',  # SGD + Momentum + Nesterov
            'child_lr': 0.1,  # Según paper
            'child_momentum': 0.9,  # Según paper
            'child_nesterov': True,  # Según paper
            'child_weight_decay': 1e-4,  # Según paper
            'child_epochs': 50,  # Según paper
            'child_batch_size': 128,  # Batch size estándar para CIFAR-10
            'child_dropout': 0.0,  # Sin dropout en las capas
            'reward_top_k': 5,  # Max accuracy de últimas 5 épocas
            'reward_power': 3,  # Elevar al cubo
            
            # NAS Search parameters (según paper: 12,800 total architectures)
            'max_episodes': 1280,  # 12,800 / 10 = 1,280 episodes
            'children_per_episode': 10,
            'save_every': 50,
            
            'checkpoint_dir': 'checkpoints/nas_nascnn15',
            'log_dir': 'logs/nas_nascnn15',
        },
        
        'demo': {
            # Configuration para DEMO/PRUEBA del proceso NAS
            # Simula el paper pero con parámetros reducidos para viabilidad computacional
            
            # Controller parameters (igual que paper)
            'max_layers': 6,  # Empieza en 6
            'layer_schedule': True,  # Activar schedule progresivo
            'layer_increment': 2,  # +2 capas cada vez
            'increment_every': 12,  # Cada 12 arquitecturas (en lugar de 1,600)
            'max_layer_limit': 12,  # Límite: 12 capas (en lugar de 15)
            
            'components_per_layer': 4,
            'lstm_hidden_size': 35,  # Según paper
            'controller_lr': 0.0006,  # Según paper
            'lr_decay': 0.96,
            'lr_decay_steps': 100,
            'beta': 1e-4,
            'baseline_ema_alpha': 0.95,
            
            # Child Network parameters (según paper pero con menos épocas)
            'child_optimizer': 'SGD',
            'child_lr': 0.1,
            'child_momentum': 0.9,
            'child_nesterov': True,
            'child_weight_decay': 1e-4,
            'child_epochs': 10,  # 10 épocas (en lugar de 50) para demo
            'child_batch_size': 128,
            'child_dropout': 0.0,
            'reward_top_k': 3,  # Max de últimas 3 épocas (en lugar de 5)
            'reward_power': 3,  # Al cubo
            
            # NAS Search parameters (reducido para demo)
            'max_episodes': 10,  # 10 episodes
            'children_per_episode': 3,  # 3 arquitecturas por episode = 30 total
            'save_every': 5,
            
            'checkpoint_dir': 'checkpoints/nas_demo',
            'log_dir': 'logs/nas_demo',
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' does not exist. Options: {list(configs.keys())}")
    
    return configs[config_name]


# DNA component limits (según paper Zoph & Le 2017)
DNA_LIMITS = {
    'kernel_size': (1, 7),      # [1, 3, 5, 7] - filter height/width
    'num_filters': (24, 64),    # [24, 36, 48, 64] - según paper
    'stride': (1, 1),           # stride=1 fijo (experimento sin stride variable)
    'pool_size': (1, 1),        # sin pooling en este experimento
}


def get_dna_limits() -> Dict[str, tuple]:
    """Returns DNA validation limits."""
    return DNA_LIMITS.copy()
