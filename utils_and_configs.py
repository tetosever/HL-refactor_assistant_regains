#!/usr/bin/env python3
"""
Utility functions and configuration management for Graph Refactoring RL
"""

import os
import json
import time

import yaml
import torch
import numpy as np
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from torch_geometric.data import Data
import networkx as nx


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    data_path: str = "data_builder/dataset/graph_features"
    max_episode_steps: int = 20
    reward_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_weights is None:
            # FIXED: Reward weights ottimizzati per stabilità
            self.reward_weights = {
                'hub_score': 0.5,
                'step_valid': 0.05,
                'step_invalid': -0.01,
                'cycle_penalty': -0.05,
                'duplicate_penalty': -0.02,
                'adversarial_weight': 0.2,
                'terminal_thresh': 0.02,
                'terminal_bonus': 2.0
            }


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    node_dim: int = 7
    hidden_dim: int = 128
    num_layers: int = 3
    num_actions: int = 7
    global_features_dim: int = 10
    dropout: float = 0.2
    shared_encoder: bool = True


@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    lr_actor: float = 1e-3
    lr_critic: float = 2e-3
    lr_discriminator: float = 5e-4
    gamma: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-5


@dataclass
class TrainingScheduleConfig:
    """Training schedule configuration"""
    num_episodes: int = 2000
    warmup_episodes: int = 400
    adversarial_start_episode: int = 800
    batch_size: int = 16
    update_every: int = 10
    discriminator_update_every: int = 200


@dataclass
class LoggingConfig:
    """Logging and evaluation configuration"""
    log_every: int = 50
    eval_every: int = 100
    save_every: int = 100
    num_eval_episodes: int = 50
    early_stopping_patience: int = 50
    min_improvement: float = 0.01


class ConfigManager:
    """Configuration manager for the entire pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.env_config = EnvironmentConfig()
        self.model_config = ModelConfig()
        self.opt_config = OptimizationConfig()
        self.schedule_config = TrainingScheduleConfig()
        self.logging_config = LoggingConfig()
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        config_path = Path(config_file)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Update configurations
        if 'environment' in config_dict:
            self.env_config = EnvironmentConfig(**config_dict['environment'])
        if 'model' in config_dict:
            self.model_config = ModelConfig(**config_dict['model'])
        if 'optimization' in config_dict:
            self.opt_config = OptimizationConfig(**config_dict['optimization'])
        if 'schedule' in config_dict:
            self.schedule_config = TrainingScheduleConfig(**config_dict['schedule'])
        if 'logging' in config_dict:
            self.logging_config = LoggingConfig(**config_dict['logging'])
    
    def save_config(self, config_file: str):
        """Save current configuration to file"""
        config_dict = {
            'environment': asdict(self.env_config),
            'model': asdict(self.model_config),
            'optimization': asdict(self.opt_config),
            'schedule': asdict(self.schedule_config),
            'logging': asdict(self.logging_config)
        }
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary"""
        return {
            'environment': asdict(self.env_config),
            'model': asdict(self.model_config),
            'optimization': asdict(self.opt_config),
            'schedule': asdict(self.schedule_config),
            'logging': asdict(self.logging_config)
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_reproducibility(seed: int = 42):
    """Setup reproducible random number generation"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_device(preferred_device: str = 'auto') -> torch.device:
    """Setup computation device"""
    if preferred_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(preferred_device)
    
    if device.type == 'cuda':
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("💻 Using CPU")
    
    return device


def setup_directories(base_dir: str = "results") -> Dict[str, Path]:
    """Setup project directory structure"""
    directories = {
        'base': Path(base_dir),
        'discriminator': Path(base_dir) / 'discriminator_pretraining',
        'rl_training': Path(base_dir) / 'rl_training',
        'evaluation': Path(base_dir) / 'evaluation',
        'ablation': Path(base_dir) / 'ablation',
        'logs': Path('logs'),
        'data': Path('data_builder/dataset/graph_features'),
        'configs': Path('configs'),
        'checkpoints': Path(base_dir) / 'checkpoints'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {path}")
    
    return directories


def setup_logging(log_dir: Path, experiment_name: str = "graph_refactor") -> logging.Logger:
    """Setup comprehensive logging"""
    log_file = log_dir / f"{experiment_name}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# DATA UTILITIES
# =============================================================================

def validate_graph_data(data: Data) -> bool:
    """Validate PyG Data object"""
    try:
        # Check node features
        if not hasattr(data, 'x') or data.x is None:
            return False
        
        if data.x.dim() != 2 or data.x.size(1) != 7:
            return False
        
        # Check edge index
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return False
        
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            return False
        
        # Check connectivity
        num_nodes = data.x.size(0)
        if data.edge_index.max() >= num_nodes:
            return False
        
        return True
        
    except Exception:
        return False


def analyze_dataset(data_path: str) -> Dict[str, Any]:
    """Analyze dataset statistics"""
    data_dir = Path(data_path)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    pt_files = list(data_dir.glob("*.pt"))
    
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_path}")
    
    stats = {
        'total_files': len(pt_files),
        'valid_graphs': 0,
        'invalid_graphs': 0,
        'num_nodes': [],
        'num_edges': [],
        'node_features_stats': [],
        'smelly_count': 0,
        'clean_count': 0,
        'connectivity_stats': []
    }
    
    for file_path in pt_files[:100]:  # Sample first 100 files for speed
        try:
            data = torch.load(file_path, map_location='cpu')
            
            if isinstance(data, dict):
                if 'data' in data:
                    graph_data = data['data']
                    label = data.get('label', data.get('is_smelly', 0))
                else:
                    graph_data = Data(x=data['x'], edge_index=data['edge_index'])
                    label = data.get('is_smelly', data.get('y', 0))
            else:
                graph_data = data
                label = getattr(data, 'is_smelly', getattr(data, 'y', 0))
            
            if not validate_graph_data(graph_data):
                stats['invalid_graphs'] += 1
                continue
            
            stats['valid_graphs'] += 1
            stats['num_nodes'].append(graph_data.num_nodes)
            stats['num_edges'].append(graph_data.edge_index.size(1))
            
            # Node features statistics
            node_features = graph_data.x.numpy()
            stats['node_features_stats'].append({
                'mean': node_features.mean(axis=0).tolist(),
                'std': node_features.std(axis=0).tolist(),
                'min': node_features.min(axis=0).tolist(),
                'max': node_features.max(axis=0).tolist()
            })
            
            # Label statistics
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if label == 1:
                stats['smelly_count'] += 1
            else:
                stats['clean_count'] += 1
            
            # Connectivity analysis
            try:
                G = nx.Graph()
                G.add_nodes_from(range(graph_data.num_nodes))
                edges = graph_data.edge_index.t().numpy()
                G.add_edges_from(edges)
                
                stats['connectivity_stats'].append({
                    'is_connected': nx.is_connected(G),
                    'num_components': nx.number_connected_components(G),
                    'density': nx.density(G),
                    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
                })
            except:
                pass
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            stats['invalid_graphs'] += 1
    
    # Compute summary statistics
    if stats['num_nodes']:
        stats['nodes_summary'] = {
            'mean': float(np.mean(stats['num_nodes'])),
            'std': float(np.std(stats['num_nodes'])),
            'min': int(np.min(stats['num_nodes'])),
            'max': int(np.max(stats['num_nodes']))
        }
    
    if stats['num_edges']:
        stats['edges_summary'] = {
            'mean': float(np.mean(stats['num_edges'])),
            'std': float(np.std(stats['num_edges'])),
            'min': int(np.min(stats['num_edges'])),
            'max': int(np.max(stats['num_edges']))
        }
    
    # Class balance
    total_labeled = stats['smelly_count'] + stats['clean_count']
    if total_labeled > 0:
        stats['class_balance'] = {
            'smelly_ratio': stats['smelly_count'] / total_labeled,
            'clean_ratio': stats['clean_count'] / total_labeled,
            'imbalance_ratio': max(stats['smelly_count'], stats['clean_count']) / max(min(stats['smelly_count'], stats['clean_count']), 1)
        }
    
    return stats


def load_sample_graphs(data_path: str, num_samples: int = 10) -> List[Data]:
    """Load sample graphs for testing"""
    data_dir = Path(data_path)
    pt_files = list(data_dir.glob("*.pt"))
    
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_path}")
    
    # Randomly sample files
    sample_files = np.random.choice(pt_files, min(num_samples, len(pt_files)), replace=False)
    
    sample_graphs = []
    for file_path in sample_files:
        try:
            data = torch.load(file_path, map_location='cpu')
            
            # Extract graph data
            if isinstance(data, dict):
                if 'data' in data:
                    graph_data = data['data']
                else:
                    graph_data = Data(x=data['x'], edge_index=data['edge_index'])
            else:
                graph_data = data
            
            if validate_graph_data(graph_data):
                sample_graphs.append(graph_data)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return sample_graphs


# =============================================================================
# MODEL UTILITIES
# =============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_model_summary(model: torch.nn.Module, save_path: str):
    """Save model architecture summary"""
    summary = {
        'model_class': model.__class__.__name__,
        'parameters': count_parameters(model),
        'architecture': str(model)
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None, 
                   device: torch.device = None) -> Dict[str, Any]:
    """Load model checkpoint"""
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, save_path: str, 
                   additional_info: Dict[str, Any] = None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': torch.tensor(time.time())
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

class MetricsTracker:
    """Track and compute metrics during training"""
    
    def __init__(self, moving_average_window: int = 100):
        self.window = moving_average_window
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []
            
            self.metrics[key].append(value)
            self.history[key].append(value)
            
            # Keep only recent values for moving average
            if len(self.metrics[key]) > self.window:
                self.metrics[key] = self.metrics[key][-self.window:]
    
    def get_average(self, key: str) -> float:
        """Get moving average of a metric"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return np.mean(self.metrics[key])
    
    def get_std(self, key: str) -> float:
        """Get standard deviation of a metric"""
        if key not in self.metrics or len(self.metrics[key]) < 2:
            return 0.0
        return np.std(self.metrics[key])
    
    def get_recent(self, key: str, n: int = 10) -> List[float]:
        """Get recent n values of a metric"""
        if key not in self.metrics:
            return []
        return self.metrics[key][-n:]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        summary = {}
        for key in self.metrics:
            if self.metrics[key]:
                summary[key] = {
                    'mean': self.get_average(key),
                    'std': self.get_std(key),
                    'latest': self.metrics[key][-1],
                    'count': len(self.history[key])
                }
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.history.clear()


def compute_graph_metrics(data: Data) -> Dict[str, float]:
    """Compute comprehensive graph metrics"""
    try:
        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        edges = data.edge_index.t().numpy()
        G.add_edges_from(edges)
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Connectivity
        metrics['is_connected'] = nx.is_connected(G)
        metrics['num_components'] = nx.number_connected_components(G)
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
        metrics['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
        
        # Clustering
        metrics['avg_clustering'] = nx.average_clustering(G)
        
        # Path lengths (only for connected graphs)
        if nx.is_connected(G):
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
            metrics['radius'] = nx.radius(G)
        else:
            metrics['avg_shortest_path'] = float('inf')
            metrics['diameter'] = float('inf')
            metrics['radius'] = float('inf')
        
        # Hub score (degree of highest degree node)
        degrees = dict(G.degree())
        metrics['hub_score'] = max(degrees.values()) if degrees else 0
        
        # Modularity (using greedy community detection)
        try:
            communities = nx.community.greedy_modularity_communities(G)
            metrics['modularity'] = nx.community.modularity(G, communities)
        except:
            metrics['modularity'] = 0.0
        
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G)
            metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
            metrics['max_betweenness'] = max(betweenness.values()) if betweenness else 0
        except:
            metrics['avg_betweenness'] = 0.0
            metrics['max_betweenness'] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"Error computing graph metrics: {e}")
        return {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1),
            'density': 0.0,
            'is_connected': 0.0,
            'hub_score': 0.0,
            'modularity': 0.0,
            'avg_shortest_path': float('inf'),
            'avg_clustering': 0.0,
            'avg_degree_centrality': 0.0,
            'avg_betweenness': 0.0
        }


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def create_training_dashboard(metrics_tracker: MetricsTracker, save_path: str):
    """Create comprehensive training dashboard"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        summary = metrics_tracker.get_summary()
        
        # Plot configurations
        plot_configs = [
            ('episode_rewards', 'Episode Rewards', 'Episodes', 'Reward'),
            ('hub_score_improvements', 'Hub Score Improvements', 'Episodes', 'Hub Score Δ'),
            ('actor_losses', 'Actor Loss', 'Updates', 'Loss'),
            ('critic_losses', 'Critic Loss', 'Updates', 'Loss'),
            ('episode_lengths', 'Episode Lengths', 'Episodes', 'Steps'),
            ('discriminator_scores', 'Discriminator Scores', 'Episodes', 'Score')
        ]
        
        for i, (metric_key, title, xlabel, ylabel) in enumerate(plot_configs, 1):
            if metric_key in metrics_tracker.history:
                plt.subplot(2, 3, i)
                values = metrics_tracker.history[metric_key]
                
                # Plot raw values
                plt.plot(values, alpha=0.3, linewidth=0.5)
                
                # Plot moving average
                if len(values) > 10:
                    window = min(50, len(values) // 10)
                    moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(values)), moving_avg, linewidth=2)
                
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Failed to create training dashboard: {e}")


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def test_environment(env_class, data_path: str, num_episodes: int = 5):
    """Test environment functionality"""
    print("🧪 Testing environment...")
    
    try:
        # Initialize environment
        env = env_class(data_path=data_path, max_steps=10)
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}:")
            
            # Reset environment
            state = env.reset()
            print(f"  Initial state shape: {state.shape}")
            
            episode_reward = 0
            step = 0
            done = False
            
            while not done and step < 10:
                # Random action
                action = np.random.randint(0, env.action_space.n)
                
                # Step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1
                
                print(f"  Step {step}: action={action}, reward={reward:.3f}, done={done}")
            
            print(f"  Total reward: {episode_reward:.3f}")
            print(f"  Episode length: {step}")
        
        print("✅ Environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_models(actor_critic_class, discriminator_class, sample_data: Data):
    """Test model functionality"""
    print("🧪 Testing models...")
    
    try:
        # Test Actor-Critic
        print("Testing Actor-Critic...")
        ac_config = {
            'node_dim': 7,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_actions': 5,
            'global_features_dim': 10,
            'dropout': 0.2,
            'shared_encoder': False
        }
        
        actor_critic = actor_critic_class(**ac_config)
        global_features = torch.randn(1, 10)
        
        # Forward pass
        output = actor_critic(sample_data, global_features)
        print(f"  Actor-Critic output keys: {list(output.keys())}")
        print(f"  Action logits shape: {output['action_logits'].shape}")
        print(f"  State value shape: {output['state_value'].shape}")
        
        # Test action sampling
        action, log_prob, value = actor_critic.get_action_and_value(sample_data, global_features)
        print(f"  Sampled action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
        
        # Test Discriminator
        print("Testing Discriminator...")
        disc_config = {
            'node_dim': 7,
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'add_node_scoring': True
        }
        
        discriminator = discriminator_class(**disc_config)
        
        # Forward pass
        disc_output = discriminator(sample_data)
        print(f"  Discriminator output keys: {list(disc_output.keys())}")
        print(f"  Logits shape: {disc_output['logits'].shape}")
        
        print("✅ Model tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


# =============================================================================
# MAIN TESTING FUNCTION
# =============================================================================

def run_system_tests(data_path: str = "data_builder/dataset/graph_features"):
    """Run comprehensive system tests"""
    print("🚀 Running system tests...")
    
    # Test data availability
    print("\n1. Testing data availability...")
    if not check_data_availability(data_path):
        print("❌ Data not available. Please prepare dataset first.")
        return False
    
    # Test configuration management
    print("\n2. Testing configuration management...")
    try:
        config_manager = ConfigManager()
        config_dict = config_manager.get_all_configs()
        print(f"✅ Configuration loaded with {len(config_dict)} sections")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # Test data loading and validation
    print("\n3. Testing data loading...")
    try:
        sample_graphs = load_sample_graphs(data_path, 5)
        print(f"✅ Loaded {len(sample_graphs)} sample graphs")
        
        if sample_graphs:
            metrics = compute_graph_metrics(sample_graphs[0])
            print(f"✅ Graph metrics computed: {len(metrics)} metrics")
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False
    
    # Test environment
    print("\n4. Testing environment...")
    try:
        from rl_gym import RefactorEnv
        success = test_environment(RefactorEnv, data_path, 2)
        if not success:
            return False
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False
    
    # Test models
    print("\n5. Testing models...")
    try:
        from actor_critic_models import ActorCritic
        from discriminator import HubDiscriminator
        
        if sample_graphs:
            success = test_models(ActorCritic, HubDiscriminator, sample_graphs[0])
            if not success:
                return False
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    print("\n✅ All system tests passed!")
    return True


def check_data_availability(data_path: str) -> bool:
    """Check if training data is available"""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_path}")
        return False
    
    pt_files = list(data_dir.glob("*.pt"))
    if not pt_files:
        print(f"❌ No .pt files found in {data_path}")
        return False
    
    print(f"✅ Found {len(pt_files)} data files")
    return True


if __name__ == "__main__":
    # Run system tests
    run_system_tests()
