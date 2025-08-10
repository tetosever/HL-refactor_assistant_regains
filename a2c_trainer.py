#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) Trainer for Graph Refactoring
Implements the complete training pipeline with adversarial discriminator updates.
FIXED VERSION - Compatibile con RefactorEnv aggiornato
"""

import os
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import time
from dataclasses import dataclass, asdict


# Import custom modules (assume questi esistano)
# from rl_gym import RefactorEnv
# from actor_critic_models import ActorCritic, create_actor_critic
# from discriminator import create_discriminator


# =============================================================================
# HYPERPARAMETERS AND CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration parameters"""

    # General
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: int = 42
    experiment_name: str = "graph_refactor_a2c"

    # Data paths
    data_path: str = "data_builder/dataset/graph_features"
    discriminator_path: str = "results/discriminator_pretraining/pretrained_discriminator.pt"
    results_dir: str = "results/rl_training"

    # Environment
    max_episode_steps: int = 20
    num_actions: int = 7
    reward_weights: Optional[Dict[str, float]] = None

    # Training phases - EXTENDED for plateau breakthrough
    num_episodes: int = 3500  # 🔧 INCREASED from 2200
    warmup_episodes: int = 600  # Maintained
    adversarial_start_episode: int = 600

    # A2C hyperparameters
    gamma: float = 0.95
    lr_actor: float = 1e-4
    lr_critic: float = 5e-5
    lr_discriminator: float = 1e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 0.25
    use_cyclic_lr: bool = True

    # 🚀 NEW: Epsilon-Greedy Exploration Parameters
    use_epsilon_greedy: bool = True
    epsilon_start: float = 0.3  # Initial exploration rate
    epsilon_end: float = 0.05  # Minimum exploration rate
    epsilon_decay_episodes: int = 1000  # Episodes to decay from start to end

    # 🎯 NEW: Enhanced Entropy Regularization
    entropy_coef_base: float = 0.05  # Base entropy coefficient
    entropy_coef_adversarial: float = 0.15  # Higher entropy during adversarial
    entropy_ramp_episodes: int = 200  # Episodes to transition between values

    # 📈 NEW: Curriculum Adversarial Weight
    use_curriculum_adversarial: bool = True
    adversarial_weight_start: float = 0.5  # Start with lower adversarial pressure
    adversarial_weight_end: float = 2.0  # Ramp up to full pressure
    adversarial_ramp_episodes: int = 800  # Episodes to ramp from start to end

    # 🔍 NEW: Exploration Monitoring
    track_action_entropy: bool = True
    track_policy_variance: bool = True
    exploration_log_every: int = 25  # Log exploration metrics frequency

    # Actor scheduler parameters
    base_lr_actor: float = 1e-4
    max_lr_actor: float = 5e-4
    step_size_up_actor: int = 100
    step_size_down_actor: Optional[int] = None
    cyclic_mode_actor: str = "triangular2"
    gamma_actor: float = 0.99

    # Critic scheduler parameters
    base_lr_critic: float = 5e-5
    max_lr_critic: float = 2e-4
    step_size_up_critic: int = 80
    step_size_down_critic: Optional[int] = None
    cyclic_mode_critic: str = "triangular2"
    gamma_critic: float = 0.99

    # Training details
    batch_size: int = 16
    update_every: int = 5
    discriminator_update_every: int = 200

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 5.0
    advantage_clip: float = 1.0

    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    shared_encoder: bool = False

    # Logging and evaluation
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500
    num_eval_episodes: int = 50

    # Early stopping
    early_stopping_patience: int = 300  # 🔧 INCREASED patience for longer exploration
    min_improvement: float = 0.01

    def __post_init__(self):
        """Post-initialization validation and setup"""
        if self.adversarial_start_episode < self.warmup_episodes:
            self.adversarial_start_episode = self.warmup_episodes

        if self.discriminator_path and not Path(self.discriminator_path).exists():
            logging.warning(f"Discriminator path does not exist: {self.discriminator_path}")

        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        if self.use_cyclic_lr:
            if self.base_lr_actor > self.max_lr_actor:
                logging.warning("base_lr_actor > max_lr_actor, swapping values")
                self.base_lr_actor, self.max_lr_actor = self.max_lr_actor, self.base_lr_actor

            if self.base_lr_critic > self.max_lr_critic:
                logging.warning("base_lr_critic > max_lr_critic, swapping values")
                self.base_lr_critic, self.max_lr_critic = self.max_lr_critic, self.base_lr_critic

            if self.step_size_down_actor is None:
                self.step_size_down_actor = self.step_size_up_actor
            if self.step_size_down_critic is None:
                self.step_size_down_critic = self.step_size_up_critic

        # Initialize reward_weights if None
        if self.reward_weights is None:
            self.reward_weights = {
                'step_valid': 0.01,
                'step_invalid': -0.02,
                'time_penalty': -0.005,
                'cycle_penalty': -0.05,
                'duplicate_penalty': -0.02,
                'adversarial_weight': self.adversarial_weight_start,  # 🔧 Use curriculum start value
                'patience': 8
            }

        # 🔧 NEW: Validate exploration parameters
        if self.use_epsilon_greedy:
            assert 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0, \
                "Epsilon values must be: 0 ≤ epsilon_end ≤ epsilon_start ≤ 1"

        if self.use_curriculum_adversarial:
            assert self.adversarial_weight_start > 0 and self.adversarial_weight_end > 0, \
                "Adversarial weights must be positive"

    def get_epsilon(self, episode_idx: int) -> float:
        """Calculate current epsilon for epsilon-greedy exploration"""
        if not self.use_epsilon_greedy:
            return 0.0

        # Linear decay starting from episode 1
        if episode_idx >= self.epsilon_decay_episodes:
            return self.epsilon_end

        progress = (episode_idx - 1) / (self.epsilon_decay_episodes - 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def get_entropy_coef(self, episode_idx: int) -> float:
        """Calculate current entropy coefficient"""
        if episode_idx < self.adversarial_start_episode:
            return self.entropy_coef_base

        # Episodes since adversarial started
        adv_episode = episode_idx - self.adversarial_start_episode

        if adv_episode >= self.entropy_ramp_episodes:
            return self.entropy_coef_adversarial

        # Linear ramp from base to adversarial
        progress = adv_episode / self.entropy_ramp_episodes
        return self.entropy_coef_base + (self.entropy_coef_adversarial - self.entropy_coef_base) * progress

    def get_adversarial_weight(self, episode_idx: int) -> float:
        """Calculate current adversarial weight for curriculum learning"""
        if not self.use_curriculum_adversarial:
            return self.reward_weights.get('adversarial_weight', 0.15)

        if episode_idx < self.adversarial_start_episode:
            return 0.0  # No adversarial during warmup

        # Episodes since adversarial started
        adv_episode = episode_idx - self.adversarial_start_episode

        if adv_episode >= self.adversarial_ramp_episodes:
            return self.adversarial_weight_end

        # Smooth ramp using sigmoid-like function for more natural progression
        progress = adv_episode / self.adversarial_ramp_episodes
        # Sigmoid-like smooth ramp
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)

        return self.adversarial_weight_start + (
                    self.adversarial_weight_end - self.adversarial_weight_start) * smooth_progress

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary, filtering unknown parameters"""
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# =============================================================================
# SETUP AND UTILITIES
# =============================================================================

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_discriminator(discriminator_path: str, device: str):
    """Load pre-trained discriminator con fallback"""
    if not os.path.exists(discriminator_path):
        logging.warning(f"Discriminator not found at {discriminator_path}. Training without discriminator.")
        return None

    try:
        checkpoint = torch.load(discriminator_path, map_location=device)
        model_config = checkpoint['model_config']

        # 🔧 FIX: Gestione import con fallback
        try:
            from discriminator import create_discriminator
            discriminator = create_discriminator(**model_config)
        except (ImportError, NameError) as e:
            logging.error(f"Cannot import create_discriminator: {e}")
            logging.warning("Training will continue without discriminator")
            return None

        discriminator.load_state_dict(checkpoint['model_state_dict'])
        discriminator.to(device)
        discriminator.eval()

        logging.info(
            f"✅ Loaded pre-trained discriminator with F1: {checkpoint.get('cv_results', {}).get('mean_f1', 'N/A')}")
        return discriminator

    except Exception as e:
        logging.error(f"Failed to load discriminator: {e}")
        return None


def ensure_clean_data(data):
    """Assicura che Data object sia pulito"""
    if data is None:
        return None
    return Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        num_nodes=data.x.size(0)
    )


# =============================================================================
# EXPERIENCE BUFFER
# =============================================================================

class ExperienceBuffer:
    """Buffer for storing and managing RL experiences with exploration tracking"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rewards = deque(maxlen=1000)

        # 🚀 NEW: Exploration tracking
        self.action_entropies = deque(maxlen=500)
        self.episode_lengths = deque(maxlen=200)
        self.action_frequencies = np.zeros(7)  # Track action usage
        self.total_actions = 0

        # 📊 NEW: Policy variance tracking
        self.policy_variances = deque(maxlen=500)
        self.exploration_scores = deque(maxlen=200)

    def add(self, state_data: Data, global_features: torch.Tensor,
            action: int, reward: float, next_state_data: Optional[Data],
            next_global_features: Optional[torch.Tensor], done: bool,
            log_prob: float, value: float, action_entropy: float = 0.0,
            policy_variance: float = 0.0):  # 🔧 NEW parameters
        """Add experience to buffer with exploration metrics"""

        def clean_data_copy(data):
            if data is None:
                return None
            return Data(
                x=data.x.clone(),
                edge_index=data.edge_index.clone(),
                num_nodes=data.x.size(0)
            )

        experience = {
            'state_data': clean_data_copy(state_data),
            'global_features': global_features.clone(),
            'action': action,
            'reward': reward,
            'next_state_data': clean_data_copy(next_state_data) if next_state_data is not None else None,
            'next_global_features': next_global_features.clone() if next_global_features is not None else None,
            'done': done,
            'log_prob': log_prob,
            'value': value,
            'action_entropy': action_entropy,  # 🔧 NEW
            'policy_variance': policy_variance  # 🔧 NEW
        }

        self.buffer.append(experience)

        # 🚀 NEW: Track exploration metrics
        self.action_entropies.append(action_entropy)
        self.policy_variances.append(policy_variance)

        # Update action frequency tracking
        if 0 <= action < len(self.action_frequencies):
            self.action_frequencies[action] += 1
            self.total_actions += 1

        if done:
            self.rewards.append(reward)

    def get_normalized_reward(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        if len(self.rewards) < 10:
            return reward

        mean_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards) + 1e-8

        return (reward - mean_reward) / std_reward

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        return random.sample(list(self.buffer), batch_size)

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    # 🚀 NEW: Exploration analysis methods
    def get_exploration_stats(self) -> Dict[str, float]:
        """Get comprehensive exploration statistics"""
        stats = {}

        # Action entropy statistics
        if self.action_entropies:
            stats['action_entropy_mean'] = float(np.mean(self.action_entropies))
            stats['action_entropy_std'] = float(np.std(self.action_entropies))
            stats['action_entropy_recent'] = float(
                np.mean(list(self.action_entropies)[-50:]) if len(self.action_entropies) >= 50 else np.mean(
                    self.action_entropies))
        else:
            stats['action_entropy_mean'] = 0.0
            stats['action_entropy_std'] = 0.0
            stats['action_entropy_recent'] = 0.0

        # Policy variance statistics
        if self.policy_variances:
            stats['policy_variance_mean'] = float(np.mean(self.policy_variances))
            stats['policy_variance_std'] = float(np.std(self.policy_variances))
            stats['policy_variance_recent'] = float(
                np.mean(list(self.policy_variances)[-50:]) if len(self.policy_variances) >= 50 else np.mean(
                    self.policy_variances))
        else:
            stats['policy_variance_mean'] = 0.0
            stats['policy_variance_std'] = 0.0
            stats['policy_variance_recent'] = 0.0

        # Action diversity (entropy of action frequencies)
        if self.total_actions > 0:
            action_probs = self.action_frequencies / self.total_actions
            action_probs = action_probs[action_probs > 0]  # Remove zero probabilities
            if len(action_probs) > 1:
                stats['action_diversity'] = float(-np.sum(action_probs * np.log(action_probs)))
            else:
                stats['action_diversity'] = 0.0
        else:
            stats['action_diversity'] = 0.0

        # Most/least used actions
        if self.total_actions > 0:
            stats['most_used_action'] = int(np.argmax(self.action_frequencies))
            stats['least_used_action'] = int(np.argmin(self.action_frequencies[self.action_frequencies > 0])) if np.any(
                self.action_frequencies > 0) else 0
            stats['action_usage_variance'] = float(np.var(self.action_frequencies))
        else:
            stats['most_used_action'] = 0
            stats['least_used_action'] = 0
            stats['action_usage_variance'] = 0.0

        return stats

    def reset_exploration_tracking(self):
        """Reset exploration tracking for new phase"""
        self.action_frequencies.fill(0)
        self.total_actions = 0
        self.action_entropies.clear()
        self.policy_variances.clear()
        self.exploration_scores.clear()

    def get_action_frequency_distribution(self) -> Dict[str, float]:
        """Get normalized action frequency distribution"""
        if self.total_actions == 0:
            return {f'action_{i}': 0.0 for i in range(7)}

        return {
            f'action_{i}': float(self.action_frequencies[i] / self.total_actions)
            for i in range(7)
        }


# =============================================================================
# A2C TRAINER
# =============================================================================

class A2CTrainer:
    """A2C trainer with adversarial discriminator updates"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = config.device

        # Setup directories
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.results_dir)

        # Set random seeds
        set_random_seeds(config.random_seed)

        # ✅ FIX: Initialize environment con reward_weights corretto
        from rl_gym import RefactorEnv  # Import here to avoid circular imports
        self.env = RefactorEnv(
            data_path=config.data_path,
            max_steps=config.max_episode_steps,
            device=config.device,
            reward_weights=config.reward_weights
        )

        # Load discriminator
        self.discriminator = load_discriminator(config.discriminator_path, config.device)
        if self.discriminator is not None:
            self.env.discriminator = self.discriminator

        # Initialize models
        self._init_models()

        # Initialize optimizers
        self._init_optimizers()

        # Experience buffer
        self.experience_buffer = ExperienceBuffer()

        # TensorBoard writer
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'discriminator_losses': [],
            'hub_score_improvements': [],
            'discriminator_scores': []
        }

        self._stop_action_count = 0
        self._episode_lengths_recent = []

        # Best model tracking
        self.best_reward = float('-inf')
        self.early_stopping_counter = 0

        self.current_epsilon = 0.0
        self.current_entropy_coef = config.entropy_coef_base
        self.current_adversarial_weight = 0.0

        self.exploration_stats = {
            'epsilon_values': [],
            'entropy_coef_values': [],
            'adversarial_weight_values': [],
            'action_entropies': [],
            'policy_variances': [],
            'exploration_scores': []
        }

        self.logger.info("🚀 A2C Trainer initialized successfully!")

    def _init_models(self):
        """Initialize actor-critic and discriminator models"""

        # Actor-Critic model
        ac_config = {
            'node_dim': 7,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'num_actions': 7,
            'global_features_dim': 4,  # ✅ FIX: Solo 4 global features ora
            'dropout': self.config.dropout,
            'shared_encoder': False
        }

        from actor_critic_models import create_actor_critic  # Import here
        self.actor_critic = create_actor_critic(ac_config).to(self.device)

        self.logger.info(f"Actor-Critic model parameters: {sum(p.numel() for p in self.actor_critic.parameters()):,}")
        self.logger.info(f"Using separate encoders for Actor and Critic")

    def _init_optimizers(self):
        """Initialize optimizers and schedulers for all models"""

        if self.config.use_cyclic_lr:
            self.logger.info("🔄 Setting up CyclicLR schedulers...")

            # Identifica parametri actor e critic
            if hasattr(self.actor_critic, 'actor_encoder') and hasattr(self.actor_critic, 'critic_encoder'):
                actor_params = list(self.actor_critic.actor_encoder.parameters()) + \
                               list(self.actor_critic.actor_head.parameters())
                critic_params = list(self.actor_critic.critic_encoder.parameters()) + \
                                list(self.actor_critic.critic_head.parameters())
            else:
                # Fallback: cerca per nome dei parametri
                actor_params = []
                critic_params = []

                for name, param in self.actor_critic.named_parameters():
                    if any(keyword in name.lower() for keyword in ['actor', 'policy']):
                        actor_params.append(param)
                    elif any(keyword in name.lower() for keyword in ['critic', 'value']):
                        critic_params.append(param)
                    else:
                        actor_params.append(param)
                        critic_params.append(param)

            # Crea optimizer separati
            self.actor_optimizer = optim.Adam(actor_params, lr=self.config.lr_actor, eps=1e-5)
            self.critic_optimizer = optim.Adam(critic_params, lr=self.config.lr_critic, eps=1e-5)

            # Calcola step_size
            actor_step_size_up = max(1, self.config.step_size_up_actor // self.config.update_every)
            actor_step_size_down = max(1, self.config.step_size_down_actor // self.config.update_every)
            critic_step_size_up = max(1, self.config.step_size_up_critic // self.config.update_every)
            critic_step_size_down = max(1, self.config.step_size_down_critic // self.config.update_every)

            # Scheduler separati
            self.actor_scheduler = optim.lr_scheduler.CyclicLR(
                self.actor_optimizer,
                base_lr=self.config.base_lr_actor,
                max_lr=self.config.max_lr_actor,
                step_size_up=actor_step_size_up,
                step_size_down=actor_step_size_down,
                mode=self.config.cyclic_mode_actor,
                gamma=self.config.gamma_actor,
                cycle_momentum=False
            )

            self.critic_scheduler = optim.lr_scheduler.CyclicLR(
                self.critic_optimizer,
                base_lr=self.config.base_lr_critic,
                max_lr=self.config.max_lr_critic,
                step_size_up=critic_step_size_up,
                step_size_down=critic_step_size_down,
                mode=self.config.cyclic_mode_critic,
                gamma=self.config.gamma_critic,
                cycle_momentum=False
            )

            self.ac_optimizer = None

        else:
            # Optimizer combinato tradizionale
            self.ac_optimizer = optim.Adam(
                self.actor_critic.parameters(),
                lr=self.config.lr_actor,
                eps=1e-5
            )

            self.ac_scheduler = optim.lr_scheduler.StepLR(
                self.ac_optimizer, step_size=500, gamma=0.8
            )

        # Discriminator optimizer
        if self.discriminator is not None:
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.lr_discriminator
            )

            self.discriminator_scheduler = optim.lr_scheduler.StepLR(
                self.discriminator_optimizer, step_size=300, gamma=0.9
            )

    def _get_action_with_epsilon_greedy(self, data: Data, global_features: torch.Tensor,
                                        epsilon: float) -> Tuple[int, float, float, float, float]:
        """
        🚀 NEW: Get action with epsilon-greedy exploration

        Returns:
            Tuple of (action, log_prob, value, action_entropy, policy_variance)
        """
        with torch.no_grad():
            output = self.actor_critic(data, global_features)
            action_probs = F.softmax(output['action_logits'], dim=1)
            value = output['state_value']

            # Calculate action entropy for exploration tracking
            action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1)

            # Calculate policy variance (measure of determinism)
            policy_variance = torch.var(action_probs, dim=1)

            if np.random.random() < epsilon:
                # 🎲 Epsilon-greedy: Random action
                # Use action mask to avoid invalid actions
                if hasattr(self.env, 'get_action_mask'):
                    action_mask = self.env.get_action_mask()
                    valid_actions = np.where(action_mask)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        action = np.random.randint(0, self.config.num_actions)
                else:
                    action = np.random.randint(0, self.config.num_actions)

                # Calculate log_prob for the randomly selected action
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action, device=self.device))
            else:
                # 🎯 Greedy: Use policy action
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                action = action.item()

            return (action, log_prob.item(), value.item(),
                    action_entropy.item(), policy_variance.item())

    def _update_dynamic_parameters(self, episode_idx: int):
        """
        🔧 NEW: Update dynamic training parameters based on curriculum
        """
        # Update epsilon for epsilon-greedy
        self.current_epsilon = self.config.get_epsilon(episode_idx)

        # Update entropy coefficient
        self.current_entropy_coef = self.config.get_entropy_coef(episode_idx)

        # Update adversarial weight in environment
        self.current_adversarial_weight = self.config.get_adversarial_weight(episode_idx)
        if hasattr(self.env, 'reward_weights'):
            self.env.reward_weights['adversarial_weight'] = self.current_adversarial_weight

        # Track parameter evolution
        self.exploration_stats['epsilon_values'].append(self.current_epsilon)
        self.exploration_stats['entropy_coef_values'].append(self.current_entropy_coef)
        self.exploration_stats['adversarial_weight_values'].append(self.current_adversarial_weight)

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """✅ FIX: Extract global features usando il metodo corretto dall'ambiente"""

        metrics = self.env._calculate_metrics(data)

        # 🔧 ASSICURATI CHE SIANO SEMPRE 4 FEATURES
        global_features = torch.tensor([
            metrics['hub_score'],
            metrics['num_nodes'],
            metrics['num_edges'],
            metrics['connected']
        ], dtype=torch.float32, device=self.device)

        # Assicurati che abbia dimensione batch
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        # 🔧 VERIFICA: Deve essere sempre [1, 4]
        assert global_features.shape[1] == 4, f"Global features shape mismatch: {global_features.shape}"

        return global_features

    def _collect_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        🔧 ENHANCED: Collect episode with epsilon-greedy exploration and enhanced tracking
        """
        # Update dynamic parameters for this episode
        self._update_dynamic_parameters(episode_idx)

        # Reset environment
        state = self.env.reset()
        current_data = ensure_clean_data(self.env.current_data)

        episode_reward = 0.0
        episode_length = 0
        episode_action_entropies = []
        episode_policy_variances = []

        initial_hub_score = self.env.initial_metrics['hub_score']

        done = False
        while not done:
            # Get action with epsilon-greedy exploration
            global_features = self._extract_global_features(current_data)

            action, log_prob, value, action_entropy, policy_variance = self._get_action_with_epsilon_greedy(
                current_data, global_features, self.current_epsilon
            )

            # Track exploration metrics
            episode_action_entropies.append(action_entropy)
            episode_policy_variances.append(policy_variance)

            # Track STOP actions
            if action == 6:
                self._stop_action_count += 1

            # Take action
            next_state, reward, done, info = self.env.step(action)
            next_data = ensure_clean_data(self.env.current_data)

            # Save experience with exploration metrics
            next_global_features = None
            if not done:
                next_global_features = self._extract_global_features(next_data)

            # Add experience to buffer with exploration tracking
            self.experience_buffer.add(
                state_data=current_data,
                global_features=global_features,
                action=action,
                reward=reward,
                next_state_data=next_data if not done else None,
                next_global_features=next_global_features,
                done=done,
                log_prob=log_prob,
                value=value,
                action_entropy=action_entropy,  # 🔧 NEW
                policy_variance=policy_variance  # 🔧 NEW
            )

            # Update for next iteration
            current_data = next_data
            episode_reward += reward
            episode_length += 1

        final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
        hub_score_improvement = initial_hub_score - final_hub_score
        best_hub_improvement = initial_hub_score - self.env.best_hub_score

        # 📊 NEW: Calculate episode exploration score
        avg_action_entropy = np.mean(episode_action_entropies) if episode_action_entropies else 0.0
        avg_policy_variance = np.mean(episode_policy_variances) if episode_policy_variances else 0.0
        exploration_score = avg_action_entropy + 0.1 * avg_policy_variance  # Combined exploration metric

        # Track exploration stats
        self.exploration_stats['action_entropies'].append(avg_action_entropy)
        self.exploration_stats['policy_variances'].append(avg_policy_variance)
        self.exploration_stats['exploration_scores'].append(exploration_score)

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'hub_score_improvement': hub_score_improvement,
            'best_hub_improvement': best_hub_improvement,
            'initial_hub_score': initial_hub_score,
            'final_hub_score': final_hub_score,
            'best_hub_score': self.env.best_hub_score,
            'no_improve_steps': self.env.no_improve_steps,

            # 🚀 NEW: Exploration metrics
            'avg_action_entropy': avg_action_entropy,
            'avg_policy_variance': avg_policy_variance,
            'exploration_score': exploration_score,
            'epsilon_used': self.current_epsilon,
            'entropy_coef_used': self.current_entropy_coef,
            'adversarial_weight_used': self.current_adversarial_weight
        }

    def _modify_reward(self, reward: float, episode_idx: int, info: Dict) -> float:
        """Modify reward based on training phase"""

        # Warm-up phase
        if episode_idx < self.config.warmup_episodes:
            hub_improvement = info.get('metrics', {}).get('hub_score', 0) - \
                              self.env.initial_metrics.get('hub_score', 0)

            if hub_improvement > 2.0:
                scaled_reward = hub_improvement * 3.0 + 5.0
            elif hub_improvement > 0.5:
                scaled_reward = hub_improvement * 2.0 + 2.0
            elif hub_improvement < -1.0:
                scaled_reward = hub_improvement * 2.0 - 3.0
            else:
                scaled_reward = hub_improvement

            return np.clip(scaled_reward, -5.0, 10.0)

        # Normal training
        modified_reward = reward

        # Normalizzazione
        if self.config.normalize_rewards and len(self.experience_buffer.rewards) > 50:
            recent_rewards = list(self.experience_buffer.rewards)[-100:]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards) + 1e-6

            modified_reward = (modified_reward - mean_reward) / (std_reward * 2.0)

        modified_reward = np.clip(modified_reward, -3.0, 5.0)

        return modified_reward

    def _update_actor_critic(self):
        """
        🔧 ENHANCED: Update actor-critic with dynamic entropy coefficient
        """
        if len(self.experience_buffer) < self.config.batch_size:
            return {}

        # Sample batch of experiences
        batch = self.experience_buffer.sample_batch(self.config.batch_size)

        # Prepare batch data
        states_data = []
        global_features_list = []
        actions = []
        returns = []
        old_log_probs = []

        for exp in batch:
            state_data = exp['state_data']

            if not hasattr(state_data, 'num_nodes') or state_data.num_nodes is None:
                state_data.num_nodes = state_data.x.size(0)

            states_data.append(state_data)
            global_features_list.append(exp['global_features'])
            actions.append(exp['action'])
            returns.append(exp['reward'])
            old_log_probs.append(exp['log_prob'])

        # Create batched data
        batch_data = Batch.from_data_list(states_data).to(self.device)
        batch_global_features = torch.cat(global_features_list, dim=0)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # Forward pass through actor-critic
        log_probs, values, entropy = self.actor_critic.evaluate_actions(
            batch_data, batch_global_features, actions_tensor
        )

        # Compute advantages
        advantages = returns_tensor - values.detach()

        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns_tensor)

        # 🔧 ENHANCED: Use dynamic entropy coefficient
        entropy_loss = -self.current_entropy_coef * entropy.mean()

        # Total loss
        total_loss = (actor_loss +
                      self.config.value_loss_coef * critic_loss +
                      entropy_loss)

        # Optimization step with proper gradient handling
        if self.config.use_cyclic_lr and hasattr(self, 'actor_optimizer'):
            # Use separate optimizers
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_total_loss = actor_loss + entropy_loss
            critic_total_loss = self.config.value_loss_coef * critic_loss

            actor_total_loss.backward(retain_graph=True)
            critic_total_loss.backward()

            # Gradient clipping
            if hasattr(self.actor_optimizer, 'param_groups'):
                actor_params = []
                for group in self.actor_optimizer.param_groups:
                    actor_params.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(actor_params, self.config.max_grad_norm)

            if hasattr(self.critic_optimizer, 'param_groups'):
                critic_params = []
                for group in self.critic_optimizer.param_groups:
                    critic_params.extend(group['params'])
                torch.nn.utils.clip_grad_norm_(critic_params, self.config.max_grad_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            current_lr_actor = self.actor_optimizer.param_groups[0]['lr']
            current_lr_critic = self.critic_optimizer.param_groups[0]['lr']

            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'total_loss': total_loss.item(),
                'entropy': entropy.mean().item(),
                'entropy_loss': entropy_loss.item(),
                'advantages_mean': advantages.mean().item(),
                'advantages_std': advantages.std().item(),
                'current_lr_actor': current_lr_actor,  # ✅ CORRETTO
                'current_lr_critic': current_lr_critic,  # ✅ CORRETTO
                'current_entropy_coef': self.current_entropy_coef,
                'current_epsilon': self.current_epsilon,
                'current_adversarial_weight': self.current_adversarial_weight
            }

        else:
            # Combined optimizer
            if self.ac_optimizer is None:
                raise ValueError("ac_optimizer is None but cyclic_lr is disabled.")

            self.ac_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
            self.ac_optimizer.step()

            current_lr = self.ac_optimizer.param_groups[0]['lr']

            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'total_loss': total_loss.item(),
                'entropy': entropy.mean().item(),
                'entropy_loss': entropy_loss.item(),
                'advantages_mean': advantages.mean().item(),
                'advantages_std': advantages.std().item(),
                'current_lr': current_lr,
                'current_entropy_coef': self.current_entropy_coef,
                'current_epsilon': self.current_epsilon,
                'current_adversarial_weight': self.current_adversarial_weight
            }

    def _update_discriminator(self):
        """Update discriminator on newly generated graphs"""

        if self.discriminator is None:
            return {}

        # Collect recent graphs from experience buffer
        recent_experiences = list(self.experience_buffer.buffer)[-self.config.batch_size:]

        if len(recent_experiences) < self.config.batch_size // 2:
            return {}

        # Prepare discriminator training data
        positive_graphs = []  # Original smelly graphs
        negative_graphs = []  # RL-modified graphs

        for exp in recent_experiences:
            state_data = exp['state_data']
            if not hasattr(state_data, 'num_nodes') or state_data.num_nodes is None:
                state_data.num_nodes = state_data.x.size(0)
            clean_state_data = Data(
                x=state_data.x.detach().clone(),
                edge_index=state_data.edge_index.detach().clone(),
                num_nodes=state_data.num_nodes
            )
            positive_graphs.append(clean_state_data)

            if exp['next_state_data'] is not None:
                next_state_data = exp['next_state_data']
                if not hasattr(next_state_data, 'num_nodes') or next_state_data.num_nodes is None:
                    next_state_data.num_nodes = next_state_data.x.size(0)
                clean_next_state_data = Data(
                    x=next_state_data.x.detach().clone(),
                    edge_index=next_state_data.edge_index.detach().clone(),
                    num_nodes=next_state_data.num_nodes
                )
                negative_graphs.append(clean_next_state_data)

        if not negative_graphs:
            return {}

        # Create balanced dataset
        min_size = min(len(positive_graphs), len(negative_graphs))
        positive_graphs = positive_graphs[:min_size]
        negative_graphs = negative_graphs[:min_size]

        # Combine and create labels
        all_graphs = positive_graphs + negative_graphs
        labels = [1] * len(positive_graphs) + [1] * len(negative_graphs)

        # Shuffle
        combined = list(zip(all_graphs, labels))
        random.shuffle(combined)
        all_graphs, labels = zip(*combined)

        # Create batch
        batch_data = Batch.from_data_list(list(all_graphs)).to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)

        # Forward pass through discriminator
        self.discriminator.train()
        if hasattr(self.discriminator, 'forward') and 'logits' in self.discriminator(batch_data):
            output = self.discriminator(batch_data)
            logits = output['logits']
        else:
            logits = self.discriminator(batch_data)

        # Discriminator loss
        discriminator_loss = F.cross_entropy(logits, labels_tensor)

        # Optimization step
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
        self.discriminator_optimizer.step()

        self.discriminator.eval()

        # Calculate accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels_tensor).float().mean()

        return {
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_accuracy': accuracy.item()
        }

    def _evaluate_model(self) -> Dict[str, float]:
        """
        🔧 ENHANCED: Evaluation with exploration-aware metrics
        """
        self.actor_critic.eval()

        eval_rewards = []
        eval_hub_improvements = []
        eval_discriminator_scores = []
        eval_episode_lengths = []
        eval_action_entropies = []  # 🔧 NEW

        for eval_episode in range(self.config.num_eval_episodes):
            # Reset environment
            self.env.reset()
            current_data = ensure_clean_data(self.env.current_data)

            episode_reward = 0.0
            episode_length = 0
            episode_entropies = []
            initial_hub_score = self.env.initial_metrics['hub_score']

            done = False
            while not done:
                global_features = self._extract_global_features(current_data)

                # Use greedy action selection for evaluation (no epsilon)
                with torch.no_grad():
                    output = self.actor_critic(current_data, global_features)
                    action_probs = F.softmax(output['action_logits'], dim=1)
                    action = torch.argmax(action_probs, dim=1).item()

                    # Track action entropy even in evaluation
                    action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).item()
                    episode_entropies.append(action_entropy)

                _, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                current_data = ensure_clean_data(self.env.current_data)

            eval_rewards.append(episode_reward)
            eval_episode_lengths.append(episode_length)
            eval_action_entropies.append(np.mean(episode_entropies) if episode_entropies else 0.0)

            # Calculate hub score improvement
            final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
            hub_improvement = initial_hub_score - final_hub_score
            eval_hub_improvements.append(hub_improvement)

            # Get discriminator score if available
            if self.discriminator is not None:
                with torch.no_grad():
                    disc_output = self.discriminator(current_data)
                    if isinstance(disc_output, dict):
                        disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 0].item()  # Clean probability
                    else:
                        disc_score = torch.softmax(disc_output, dim=1)[0, 0].item()
                    eval_discriminator_scores.append(disc_score)

        self.actor_critic.train()

        return {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_hub_improvement_mean': np.mean(eval_hub_improvements),
            'eval_hub_improvement_std': np.std(eval_hub_improvements),
            'eval_discriminator_score_mean': np.mean(eval_discriminator_scores) if eval_discriminator_scores else 0.0,
            'eval_discriminator_score_std': np.std(eval_discriminator_scores) if eval_discriminator_scores else 0.0,

            # 🚀 NEW: Evaluation exploration metrics
            'eval_episode_length_mean': np.mean(eval_episode_lengths),
            'eval_episode_length_std': np.std(eval_episode_lengths),
            'eval_action_entropy_mean': np.mean(eval_action_entropies),
            'eval_action_entropy_std': np.std(eval_action_entropies),

            # Success metrics
            'eval_success_rate': sum(1 for imp in eval_hub_improvements if imp > 0.01) / len(eval_hub_improvements),
            'eval_significant_improvement_rate': sum(1 for imp in eval_hub_improvements if imp > 0.1) / len(
                eval_hub_improvements)
        }

    def _log_metrics(self, episode_idx: int, episode_info: Dict, update_info: Dict = None):
        """
        🔧 ENHANCED: Logging with comprehensive exploration telemetry
        """
        # Update training statistics
        self.training_stats['episode_rewards'].append(episode_info['episode_reward'])
        self.training_stats['episode_lengths'].append(episode_info['episode_length'])

        final_improvement = episode_info.get('hub_score_improvement', 0)
        self.training_stats['hub_score_improvements'].append(final_improvement)

        if update_info:
            if 'actor_loss' in update_info:
                self.training_stats['actor_losses'].append(update_info['actor_loss'])
            if 'critic_loss' in update_info:
                self.training_stats['critic_losses'].append(update_info['critic_loss'])
            if 'discriminator_loss' in update_info:
                self.training_stats['discriminator_losses'].append(update_info['discriminator_loss'])

        # TensorBoard logging - EXISTING
        self.writer.add_scalar('Episode/Reward', episode_info['episode_reward'], episode_idx)
        self.writer.add_scalar('Episode/Length', episode_info['episode_length'], episode_idx)
        self.writer.add_scalar('Episode/HubImprovement', final_improvement, episode_idx)
        self.writer.add_scalar('Episode/BestHubImprovement', episode_info.get('best_hub_improvement', 0), episode_idx)

        # 🚀 NEW: Enhanced exploration logging
        if 'avg_action_entropy' in episode_info:
            self.writer.add_scalar('Exploration/ActionEntropy', episode_info['avg_action_entropy'], episode_idx)
        if 'avg_policy_variance' in episode_info:
            self.writer.add_scalar('Exploration/PolicyVariance', episode_info['avg_policy_variance'], episode_idx)
        if 'exploration_score' in episode_info:
            self.writer.add_scalar('Exploration/ExplorationScore', episode_info['exploration_score'], episode_idx)

        # Dynamic parameters
        epsilon_effective = self.current_epsilon
        self.writer.add_scalar('Parameters/EpsilonEffective', epsilon_effective, episode_idx)
        self.writer.add_scalar('Parameters/EntropyCoef', self.current_entropy_coef, episode_idx)
        self.writer.add_scalar('Parameters/AdversarialWeight', self.current_adversarial_weight, episode_idx)

        # Log epsilon_effective to standard logger
        self.logger.info(f"Episode {episode_idx} | epsilon_effective={epsilon_effective:.3f}")

        # Training phase indicators
        is_warmup = episode_idx < self.config.adversarial_start_episode
        is_adversarial = episode_idx >= self.config.adversarial_start_episode
        self.writer.add_scalar('Phase/IsWarmup', float(is_warmup), episode_idx)
        self.writer.add_scalar('Phase/IsAdversarial', float(is_adversarial), episode_idx)

        if update_info:
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, episode_idx)

        # 🔧 ENHANCED: Anti-collapse telemetry with exploration focus
        if episode_idx % self.config.log_every == 0:
            recent_rewards = self.training_stats['episode_rewards'][-self.config.log_every:]
            recent_improvements = self.training_stats['hub_score_improvements'][-self.config.log_every:]
            recent_lengths = self.training_stats['episode_lengths'][-self.config.log_every:]

            success_rate = sum(1 for imp in recent_improvements if imp > 0.01) / len(recent_improvements)

            # Standard telemetry
            early_stop_rate = sum(1 for length in recent_lengths if length <= 2) / len(recent_lengths)
            reward_variance = np.var(recent_rewards) if len(recent_rewards) > 1 else 0
            stop_action_rate = self._stop_action_count / self.config.log_every

            self.logger.info(
                f"Episode {episode_idx:4d} | "
                f"Phase: {'WarmUp' if is_warmup else 'Adversarial'} | "
                f"Reward: {np.mean(recent_rewards):6.3f}±{np.std(recent_rewards):5.3f} | "
                f"Hub Δ (final): {np.mean(recent_improvements):6.3f}±{np.std(recent_improvements):5.3f} | "
                f"Success: {success_rate:.1%} | "
                f"Buffer: {len(self.experience_buffer)}"
            )

            # 🚀 NEW: Detailed exploration telemetry
            exploration_stats = self.experience_buffer.get_exploration_stats()

            self.logger.info(
                f"🎯 EXPLORATION | "
                f"ε={self.current_epsilon:.3f} | "
                f"Entropy={exploration_stats.get('action_entropy_recent', 0):.3f} | "
                f"PolicyVar={exploration_stats.get('policy_variance_recent', 0):.3f} | "
                f"ActionDiv={exploration_stats.get('action_diversity', 0):.3f} | "
                f"EntropyCoef={self.current_entropy_coef:.3f}"
            )

            # Phase-specific information
            if is_adversarial:
                adv_progress = (episode_idx - self.config.adversarial_start_episode) / max(1,
                                                                                           self.config.adversarial_ramp_episodes)
                self.logger.info(
                    f"🔥 ADVERSARIAL | "
                    f"Progress: {min(adv_progress, 1.0):.1%} | "
                    f"AdvWeight: {self.current_adversarial_weight:.3f} | "
                    f"Plateau: {self.early_stopping_counter}/{self.config.early_stopping_patience}"
                )

            # Log exploration statistics to TensorBoard
            for stat_name, stat_value in exploration_stats.items():
                if isinstance(stat_value, (int, float)):
                    self.writer.add_scalar(f'ExplorationStats/{stat_name}', stat_value, episode_idx)

            # Action frequency distribution
            action_freqs = self.experience_buffer.get_action_frequency_distribution()
            for action_name, freq in action_freqs.items():
                self.writer.add_scalar(f'ActionFrequency/{action_name}', freq, episode_idx)

            # Legacy anti-collapse telemetry
            self.logger.info(
                f"🔍 TELEMETRIA | "
                f"Early STOP: {early_stop_rate:.1%} | "
                f"STOP rate: {stop_action_rate:.1%} | "
                f"Reward var: {reward_variance:.4f}"
            )

            # TensorBoard anti-collapse telemetry
            self.writer.add_scalar('Debug/EarlyStopRate', early_stop_rate, episode_idx)
            self.writer.add_scalar('Debug/RewardVariance', reward_variance, episode_idx)
            self.writer.add_scalar('Debug/StopActionRate', stop_action_rate, episode_idx)

            # Reset counters
            self._stop_action_count = 0

    def _save_checkpoint(self, episode_idx: int, is_best: bool = False):
        """Save model checkpoint"""

        checkpoint = {
            'episode': episode_idx,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'model_config': {
                'node_dim': 7,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_actions': 7,
                'global_features_dim': 4,  # ✅ FIX: Aggiornato a 4
                'dropout': self.config.dropout,
                'shared_encoder': False
            },
            'training_stats': self.training_stats,
            'config': self.config,
            'best_reward': self.best_reward
        }

        # Salva optimizer states corretti
        if self.config.use_cyclic_lr and hasattr(self, 'actor_optimizer'):
            checkpoint['actor_optimizer_state_dict'] = self.actor_optimizer.state_dict()
            checkpoint['critic_optimizer_state_dict'] = self.critic_optimizer.state_dict()
            checkpoint['actor_scheduler_state_dict'] = self.actor_scheduler.state_dict()
            checkpoint['critic_scheduler_state_dict'] = self.critic_scheduler.state_dict()
        elif hasattr(self, 'ac_optimizer') and self.ac_optimizer is not None:
            checkpoint['ac_optimizer_state_dict'] = self.ac_optimizer.state_dict()
            if hasattr(self, 'ac_scheduler'):
                checkpoint['ac_scheduler_state_dict'] = self.ac_scheduler.state_dict()

        if self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
            if hasattr(self, 'discriminator_scheduler'):
                checkpoint['discriminator_scheduler_state_dict'] = self.discriminator_scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.results_dir / f'checkpoint_episode_{episode_idx}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.results_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Saved new best model at episode {episode_idx}")

    def train(self):
        """🚀 ENHANCED: Main training loop with exploration focus and plateau detection"""

        self.logger.info("🚀 Starting Enhanced A2C training with exploration...")
        self.logger.info(f"Training for {self.config.num_episodes} episodes")
        self.logger.info(f"Warm-up episodes: {self.config.warmup_episodes}")
        self.logger.info(f"Adversarial training starts at episode: {self.config.adversarial_start_episode}")
        self.logger.info(f"🎲 Epsilon-greedy enabled: {self.config.use_epsilon_greedy}")
        self.logger.info(f"🎯 Curriculum adversarial: {self.config.use_curriculum_adversarial}")

        start_time = time.time()
        update_count = 0

        # 🚀 NEW: Enhanced tracking
        plateau_detections = 0
        exploration_boosts = 0
        best_exploration_score = 0.0

        for episode_idx in range(1, self.config.num_episodes + 1):

            # 🔧 ENHANCED: Set environment exploration mode based on phase
            if episode_idx == self.config.adversarial_start_episode:
                self.logger.info("🎯 Switching to adversarial phase - enabling exploration mode")
                if hasattr(self.env, 'set_exploration_mode'):
                    exploration_factor = 1.5  # More forgiving during early adversarial
                    self.env.set_exploration_mode(exploration_factor)

            # Collect episode with enhanced exploration
            episode_info = self._collect_episode(episode_idx)

            # Track best exploration score
            current_exploration = episode_info.get('exploration_score', 0.0)
            if current_exploration > best_exploration_score:
                best_exploration_score = current_exploration
                self.logger.info(f"🎲 New best exploration score: {best_exploration_score:.4f}")

            # Update actor-critic every few episodes
            update_info = {}
            if episode_idx % self.config.update_every == 0:
                update_info.update(self._update_actor_critic())
                update_count += 1

                # Step per CyclicLR schedulers
                if self.config.use_cyclic_lr:
                    if hasattr(self, 'actor_scheduler') and hasattr(self, 'critic_scheduler'):
                        self.actor_scheduler.step()
                        self.critic_scheduler.step()

                        # Log learning rates periodically
                        if update_count % 50 == 0:
                            current_lr_actor = self.actor_scheduler.get_last_lr()[0]
                            current_lr_critic = self.critic_scheduler.get_last_lr()[0]

                            self.logger.info(
                                f"📊 Update {update_count}: "
                                f"Actor LR = {current_lr_actor:.2e}, "
                                f"Critic LR = {current_lr_critic:.2e}"
                            )

                            self.writer.add_scalar('Learning_Rate/actor', current_lr_actor, update_count)
                            self.writer.add_scalar('Learning_Rate/critic', current_lr_critic, update_count)
                else:
                    # Traditional scheduler
                    if episode_idx % 100 == 0 and hasattr(self, 'ac_scheduler'):
                        self.ac_scheduler.step()

            # Update discriminator in adversarial phase
            if (episode_idx >= self.config.adversarial_start_episode and
                    episode_idx % self.config.discriminator_update_every == 0):
                discriminator_info = self._update_discriminator()
                update_info.update(discriminator_info)

                if hasattr(self, 'discriminator_scheduler'):
                    self.discriminator_scheduler.step()

            # Enhanced logging
            self._log_metrics(episode_idx, episode_info, update_info)

            # Evaluation with plateau detection
            if episode_idx % self.config.eval_every == 0:
                eval_metrics = self._evaluate_model()

                # 🚀 NEW: Enhanced evaluation logging
                self.logger.info(
                    f"📊 Evaluation at episode {episode_idx}: "
                    f"Reward: {eval_metrics['eval_reward_mean']:.3f}±{eval_metrics['eval_reward_std']:.3f} | "
                    f"Hub Δ: {eval_metrics['eval_hub_improvement_mean']:.3f}±{eval_metrics['eval_hub_improvement_std']:.3f} | "
                    f"Success Rate: {eval_metrics.get('eval_success_rate', 0):.1%} | "
                    f"Entropy: {eval_metrics.get('eval_action_entropy_mean', 0):.3f}"
                )

                # Log all evaluation metrics to TensorBoard
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f'Evaluation/{key}', value, episode_idx)

                # 🚨 NEW: Plateau detection and exploration boost
                if self._detect_plateau_and_adjust(episode_idx, eval_metrics):
                    plateau_detections += 1
                    exploration_boosts += 1

                    # Log plateau detection
                    self.writer.add_scalar('Plateau/DetectionCount', plateau_detections, episode_idx)
                    self.writer.add_scalar('Boost/BoostCount', exploration_boosts, episode_idx)

                # Check for best model
                current_reward = eval_metrics['eval_reward_mean']
                current_hub_improvement = eval_metrics['eval_hub_improvement_mean']

                # 🔧 ENHANCED: Consider both reward and hub improvement for best model
                combined_score = current_reward + 10.0 * current_hub_improvement
                best_combined_score = self.best_reward + 10.0 * 0.5  # Assume previous best hub was 0.5

                if combined_score > best_combined_score + self.config.min_improvement:
                    self.best_reward = current_reward
                    self.early_stopping_counter = 0
                    self._save_checkpoint(episode_idx, is_best=True)

                    self.logger.info(f"🏆 New best model! Combined score: {combined_score:.3f}")
                else:
                    self.early_stopping_counter += 1

                # 🎯 SUCCESS CHECK: Log if we've broken the plateau
                if current_hub_improvement > 0.6:
                    self.logger.info(f"🎉 PLATEAU BREAKTHROUGH! Hub improvement: {current_hub_improvement:.4f}")
                    self.writer.add_scalar('Success/PlateauBreakthrough', 1.0, episode_idx)

            # Save regular checkpoint
            if episode_idx % self.config.save_every == 0:
                self._save_checkpoint(episode_idx)

            # 🔧 ENHANCED: Extended early stopping with exploration consideration
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                # Before stopping, check if we're still exploring effectively
                recent_exploration = self.exploration_stats['exploration_scores'][-50:] if len(
                    self.exploration_stats['exploration_scores']) >= 50 else []

                if recent_exploration and np.mean(recent_exploration) > 0.5:
                    self.logger.info(f"🎲 High exploration detected, extending training...")
                    self.early_stopping_counter = self.config.early_stopping_patience // 2  # Reset partially
                else:
                    self.logger.info(f"🛑 Early stopping at episode {episode_idx}")
                    break

        # Final evaluation and save
        final_eval = self._evaluate_model()
        self._save_checkpoint(episode_idx, is_best=False)

        training_time = time.time() - start_time

        # 🚀 ENHANCED: Final summary with exploration statistics
        self.logger.info(f"✅ Enhanced training completed in {training_time / 3600:.2f} hours")
        self.logger.info(f"🏆 Best reward: {self.best_reward:.3f}")
        self.logger.info(f"🎲 Plateau detections: {plateau_detections}")
        self.logger.info(f"🚀 Exploration boosts triggered: {exploration_boosts}")
        self.logger.info(f"🎯 Best exploration score: {best_exploration_score:.4f}")

        final_hub_improvement = final_eval.get('eval_hub_improvement_mean', 0)
        self.logger.info(f"📊 Final hub improvement: {final_hub_improvement:.4f}")

        if final_hub_improvement > 0.6:
            self.logger.info("🎉 SUCCESS: Plateau breakthrough achieved!")
        elif final_hub_improvement > 0.5:
            self.logger.info("📈 PROGRESS: Improved beyond previous plateau")
        else:
            self.logger.info("⚠️ PLATEAU: Consider further tuning")

        # Save final results with exploration stats
        self._save_final_results_enhanced(final_eval, training_time, {
            'plateau_detections': plateau_detections,
            'exploration_boosts': exploration_boosts,
            'best_exploration_score': best_exploration_score,
            'final_epsilon': self.current_epsilon,
            'final_entropy_coef': self.current_entropy_coef,
            'final_adversarial_weight': self.current_adversarial_weight
        })

        return self.training_stats

    def _save_final_results(self, final_eval: Dict, training_time: float):
        """Save final training results and plots"""

        # Save training statistics
        results = {
            'config': self.config.__dict__,
            'training_stats': self.training_stats,
            'final_evaluation': final_eval,
            'training_time_hours': training_time / 3600,
            'best_reward': self.best_reward,

            # 🚀 NEW: Exploration statistics
            'exploration_summary': exploration_stats,
            'parameter_evolution': {
                'epsilon_values': self.exploration_stats.get('epsilon_values', []),
                'entropy_coef_values': self.exploration_stats.get('entropy_coef_values', []),
                'adversarial_weight_values': self.exploration_stats.get('adversarial_weight_values', [])
            },

            # Experience buffer exploration stats
            'final_exploration_metrics': self.experience_buffer.get_exploration_stats(),
            'action_frequency_distribution': self.experience_buffer.get_action_frequency_distribution()
        }

        with open(self.results_dir / 'enhanced_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Create enhanced plots
        self._create_training_plots()

        self.logger.info(f"📁 Enhanced results saved to {self.results_dir}")

    def _create_training_plots(self):
        """Create training progress plots"""

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Create figure with more subplots for exploration
            fig = plt.figure(figsize=(24, 16))

            # Original plots (top row)
            plt.subplot(3, 4, 1)
            if self.training_stats['episode_rewards']:
                rewards = self.training_stats['episode_rewards']
                plt.plot(rewards, alpha=0.3, color='blue', linewidth=0.5)
                window = min(100, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(rewards)), moving_avg, color='red', linewidth=2)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 2)
            if self.training_stats['hub_score_improvements']:
                hub_improvements = self.training_stats['hub_score_improvements']
                plt.plot(hub_improvements, alpha=0.3, color='green', linewidth=0.5)
                window = min(100, len(hub_improvements) // 10)
                if window > 1:
                    moving_avg = np.convolve(hub_improvements, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(hub_improvements)), moving_avg, color='darkgreen', linewidth=2)
                plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Previous Plateau')
                plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Target Breakthrough')
                plt.title('Hub Score Improvements')
                plt.xlabel('Episode')
                plt.ylabel('Hub Score Δ')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 3)
            if self.training_stats['actor_losses']:
                plt.plot(self.training_stats['actor_losses'], color='orange', alpha=0.7)
                plt.title('Actor Loss')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 4)
            if self.training_stats['episode_lengths']:
                lengths = self.training_stats['episode_lengths']
                plt.plot(lengths, alpha=0.3, color='brown', linewidth=0.5)
                window = min(50, len(lengths) // 10)
                if window > 1:
                    moving_avg = np.convolve(lengths, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(lengths)), moving_avg, color='darkred', linewidth=2)
                plt.title('Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                plt.grid(True, alpha=0.3)

            # 🚀 NEW: Exploration metrics (middle row)
            plt.subplot(3, 4, 5)
            if self.exploration_stats.get('epsilon_values'):
                plt.plot(self.exploration_stats['epsilon_values'], color='purple', linewidth=2)
                plt.title('Epsilon Evolution')
                plt.xlabel('Episode')
                plt.ylabel('Epsilon')
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 6)
            if self.exploration_stats.get('entropy_coef_values'):
                plt.plot(self.exploration_stats['entropy_coef_values'], color='magenta', linewidth=2)
                plt.title('Entropy Coefficient Evolution')
                plt.xlabel('Episode')
                plt.ylabel('Entropy Coef')
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 7)
            if self.exploration_stats.get('action_entropies'):
                entropies = self.exploration_stats['action_entropies']
                plt.plot(entropies, alpha=0.4, color='cyan', linewidth=0.5)
                window = min(50, len(entropies) // 10)
                if window > 1:
                    moving_avg = np.convolve(entropies, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(entropies)), moving_avg, color='blue', linewidth=2)
                plt.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Target Threshold')
                plt.title('Action Entropy')
                plt.xlabel('Episode')
                plt.ylabel('Entropy')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 8)
            if self.exploration_stats.get('adversarial_weight_values'):
                plt.plot(self.exploration_stats['adversarial_weight_values'], color='red', linewidth=2)
                plt.title('Adversarial Weight Curriculum')
                plt.xlabel('Episode')
                plt.ylabel('Adversarial Weight')
                plt.grid(True, alpha=0.3)

            # 🚀 NEW: Additional exploration metrics (bottom row)
            plt.subplot(3, 4, 9)
            if self.exploration_stats.get('exploration_scores'):
                scores = self.exploration_stats['exploration_scores']
                plt.plot(scores, alpha=0.4, color='orange', linewidth=0.5)
                window = min(50, len(scores) // 10)
                if window > 1:
                    moving_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(scores)), moving_avg, color='darkorange', linewidth=2)
                plt.title('Exploration Score')
                plt.xlabel('Episode')
                plt.ylabel('Score')
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 10)
            # Action frequency distribution (final)
            action_freqs = self.experience_buffer.get_action_frequency_distribution()
            if action_freqs:
                actions = list(range(7))
                frequencies = [action_freqs.get(f'action_{i}', 0) for i in actions]
                action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                                'ExtractAbstractUnit', 'ExtractUnit', 'STOP']

                bars = plt.bar(actions, frequencies, color='skyblue', alpha=0.7)
                plt.title('Final Action Distribution')
                plt.xlabel('Action')
                plt.ylabel('Frequency')
                plt.xticks(actions, [name[:8] for name in action_names], rotation=45)
                plt.grid(True, alpha=0.3)

                # Add frequency labels on bars
                for bar, freq in zip(bars, frequencies):
                    if freq > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                 f'{freq:.2f}', ha='center', va='bottom', fontsize=8)

            plt.subplot(3, 4, 11)
            if self.training_stats['critic_losses']:
                plt.plot(self.training_stats['critic_losses'], color='green', alpha=0.7)
                plt.title('Critic Loss')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            plt.subplot(3, 4, 12)
            if self.training_stats['discriminator_losses']:
                plt.plot(self.training_stats['discriminator_losses'], color='red', alpha=0.7)
                plt.title('Discriminator Loss')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.results_dir / 'enhanced_training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 🚀 NEW: Create separate exploration dashboard
            self._create_exploration_dashboard()

        except Exception as e:
            self.logger.warning(f"Failed to create enhanced training plots: {e}")

        def _create_exploration_dashboard(self):
            """🚀 NEW: Create dedicated exploration analysis dashboard"""

            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle('Exploration Analysis Dashboard', fontsize=16, fontweight='bold')

                # Epsilon vs Hub Improvement correlation
                if (self.exploration_stats.get('epsilon_values') and
                        len(self.training_stats['hub_score_improvements']) > 0):
                    eps_vals = self.exploration_stats['epsilon_values']
                    hub_imps = self.training_stats['hub_score_improvements'][-len(eps_vals):]

                    axes[0, 0].scatter(eps_vals, hub_imps, alpha=0.6, c='blue')
                    axes[0, 0].set_xlabel('Epsilon')
                    axes[0, 0].set_ylabel('Hub Improvement')
                    axes[0, 0].set_title('Epsilon vs Hub Improvement')
                    axes[0, 0].grid(True, alpha=0.3)

                # Action Entropy vs Episode Length correlation
                if (self.exploration_stats.get('action_entropies') and
                        len(self.training_stats['episode_lengths']) > 0):
                    entropies = self.exploration_stats['action_entropies']
                    lengths = self.training_stats['episode_lengths'][-len(entropies):]

                    axes[0, 1].scatter(entropies, lengths, alpha=0.6, c='green')
                    axes[0, 1].set_xlabel('Action Entropy')
                    axes[0, 1].set_ylabel('Episode Length')
                    axes[0, 1].set_title('Action Entropy vs Episode Length')
                    axes[0, 1].grid(True, alpha=0.3)

                # Exploration phases visualization
                if self.exploration_stats.get('epsilon_values'):
                    episodes = range(len(self.exploration_stats['epsilon_values']))
                    axes[0, 2].plot(episodes, self.exploration_stats['epsilon_values'],
                                    label='Epsilon', color='purple', linewidth=2)

                    if self.exploration_stats.get('entropy_coef_values'):
                        # Normalize entropy coef for comparison
                        normalized_entropy = np.array(self.exploration_stats['entropy_coef_values']) * 2
                        axes[0, 2].plot(episodes, normalized_entropy,
                                        label='Entropy Coef (x2)', color='orange', linewidth=2)

                    axes[0, 2].axvline(x=self.config.adversarial_start_episode,
                                       color='red', linestyle='--', alpha=0.7, label='Adversarial Start')
                    axes[0, 2].set_xlabel('Episode')
                    axes[0, 2].set_ylabel('Value')
                    axes[0, 2].set_title('Exploration Parameters Evolution')
                    axes[0, 2].legend()
                    axes[0, 2].grid(True, alpha=0.3)

                # Performance vs Exploration correlation
                if (self.exploration_stats.get('exploration_scores') and
                        len(self.training_stats['episode_rewards']) > 0):
                    exp_scores = self.exploration_stats['exploration_scores']
                    rewards = self.training_stats['episode_rewards'][-len(exp_scores):]

                    axes[1, 0].scatter(exp_scores, rewards, alpha=0.6, c='red')
                    axes[1, 0].set_xlabel('Exploration Score')
                    axes[1, 0].set_ylabel('Episode Reward')
                    axes[1, 0].set_title('Exploration vs Performance')
                    axes[1, 0].grid(True, alpha=0.3)

                # Action diversity over time
                exploration_stats = self.experience_buffer.get_exploration_stats()
                if exploration_stats.get('action_diversity', 0) > 0:
                    # Create a simple diversity timeline (approximate)
                    diversity_timeline = [exploration_stats['action_diversity']] * 10
                    axes[1, 1].plot(diversity_timeline, color='cyan', linewidth=3)
                    axes[1, 1].set_xlabel('Time Period')
                    axes[1, 1].set_ylabel('Action Diversity')
                    axes[1, 1].set_title(f'Final Action Diversity: {exploration_stats["action_diversity"]:.3f}')
                    axes[1, 1].grid(True, alpha=0.3)

                # Summary statistics
                summary_text = f"""
    EXPLORATION SUMMARY:
    • Total Episodes: {len(self.training_stats['episode_rewards'])}
    • Adversarial Episodes: {max(0, len(self.training_stats['episode_rewards']) - self.config.adversarial_start_episode)}
    • Final Epsilon: {self.current_epsilon:.3f}
    • Final Entropy Coef: {self.current_entropy_coef:.3f}
    • Best Exploration Score: {max(self.exploration_stats.get('exploration_scores', [0])):.3f}
    • Action Diversity: {exploration_stats.get('action_diversity', 0):.3f}
    • Most Used Action: {exploration_stats.get('most_used_action', 'N/A')}
                """

                axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                                fontsize=10, verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].axis('off')
                axes[1, 2].set_title('Exploration Summary')

                plt.tight_layout()
                plt.savefig(self.results_dir / 'exploration_dashboard.png', dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                self.logger.warning(
                    f"Failed to create exploration dashboard: {e}")  # 🔧 ENHANCED ENVIRONMENT INTEGRATION - Add these methods to RefactorEnv class

    def update_adversarial_weight(self, new_weight: float):
        """
        🚀 NEW: Update adversarial weight dynamically during training

        Args:
            new_weight: New adversarial weight value
        """
        if hasattr(self, 'reward_weights') and self.reward_weights:
            old_weight = self.reward_weights.get('adversarial_weight', 0.0)
            self.reward_weights['adversarial_weight'] = new_weight

            if abs(new_weight - old_weight) > 0.01:  # Only log significant changes
                print(f"🔄 Updated adversarial weight: {old_weight:.3f} → {new_weight:.3f}")

    def get_exploration_friendly_reward_weights(self) -> Dict[str, float]:
        """
        🚀 NEW: Get reward weights optimized for exploration

        Returns:
            Dictionary with exploration-friendly reward weights
        """
        return {
            'hub_weight': 8.0,  # Slightly reduced for less overwhelming signal
            'step_valid': 0.02,  # Small positive reinforcement for valid actions
            'step_invalid': -0.05,  # Moderate penalty to discourage invalid actions
            'time_penalty': -0.005,  # Very small time penalty to encourage exploration
            'early_stop_penalty': -0.3,  # Reduced penalty for early stopping
            'cycle_penalty': -0.1,  # Moderate penalty for cycles
            'duplicate_penalty': -0.05,  # Small penalty for duplicates
            'adversarial_weight': 0.5,  # Will be updated dynamically
            'patience': 12  # Extended patience for exploration
        }

    def set_exploration_mode(self, exploration_factor: float = 1.0):
        """
        🚀 NEW: Set environment to exploration-friendly mode

        Args:
            exploration_factor: Factor to scale exploration-friendly settings (0.5-2.0)
        """
        base_weights = self.get_exploration_friendly_reward_weights()

        # Scale penalties by exploration factor (higher factor = more forgiving)
        self.reward_weights.update({
            'step_valid': base_weights['step_valid'] * exploration_factor,
            'step_invalid': base_weights['step_invalid'] / exploration_factor,
            'time_penalty': base_weights['time_penalty'] / exploration_factor,
            'early_stop_penalty': base_weights['early_stop_penalty'] / exploration_factor,
            'cycle_penalty': base_weights['cycle_penalty'] / exploration_factor,
            'duplicate_penalty': base_weights['duplicate_penalty'] / exploration_factor,
            'patience': int(base_weights['patience'] * exploration_factor)
        })

        print(f"🎲 Environment set to exploration mode (factor: {exploration_factor:.2f})")

    def get_action_difficulty_bonus(self, action: int, success: bool) -> float:
        """
        🚀 NEW: Provide bonus for attempting difficult/rare actions

        Args:
            action: Action attempted
            success: Whether action was successful

        Returns:
            Bonus reward for exploration
        """
        # Define action difficulty (higher = more difficult/rare)
        action_difficulty = {
            0: 0.5,  # RemoveEdge - moderate
            1: 0.3,  # AddEdge - easy
            2: 0.7,  # MoveEdge - difficult
            3: 1.0,  # ExtractMethod - very difficult
            4: 1.2,  # ExtractAbstractUnit - very difficult
            5: 1.0,  # ExtractUnit - very difficult
            6: 0.0  # STOP - no bonus
        }

        difficulty = action_difficulty.get(action, 0.0)

        if success and difficulty > 0:
            # Bonus scales with difficulty and current adversarial weight
            base_bonus = 0.02 * difficulty
            adv_weight = self.reward_weights.get('adversarial_weight', 1.0)

            # Higher adversarial weight = need more exploration bonus
            exploration_bonus = base_bonus * (1.0 + 0.5 * adv_weight)
            return min(exploration_bonus, 0.1)  # Cap bonus

        return 0.0

    def _calculate_exploration_reward(self, action: int, success: bool,
                                      episode_length: int) -> float:
        """
        🚀 NEW: Calculate exploration-specific reward component

        Args:
            action: Action taken
            success: Whether action was successful
            episode_length: Current episode length

        Returns:
            Exploration reward component
        """
        exploration_reward = 0.0

        # 1. Action difficulty bonus
        exploration_reward += self.get_action_difficulty_bonus(action, success)

        # 2. Episode length bonus (encourage longer episodes for exploration)
        if episode_length > 5:  # After minimum exploration period
            length_bonus = min(0.01 * (episode_length - 5), 0.05)  # Max 0.05 bonus
            exploration_reward += length_bonus

        # 3. Anti-repetition bonus (track recent actions)
        if hasattr(self, '_recent_actions'):
            if len(self._recent_actions) >= 3:
                recent_unique = len(set(self._recent_actions[-3:]))
                if recent_unique >= 2:  # Diverse recent actions
                    exploration_reward += 0.01

            # Update recent actions
            self._recent_actions.append(action)
            if len(self._recent_actions) > 5:
                self._recent_actions.pop(0)
        else:
            self._recent_actions = [action]

        return exploration_reward

    def _detect_plateau_and_adjust(self, episode_idx: int, eval_metrics: Dict[str, float]) -> bool:
        """
        🚀 NEW: Detect performance plateau and trigger exploration boost

        Returns:
            True if plateau detected and adjustments made
        """
        if episode_idx < self.config.adversarial_start_episode + 200:
            return False  # Too early to detect plateau

        # Check recent performance
        recent_improvements = self.training_stats['hub_score_improvements'][-100:] if len(
            self.training_stats['hub_score_improvements']) >= 100 else []

        if len(recent_improvements) < 50:
            return False

        # Plateau criteria
        mean_improvement = np.mean(recent_improvements)
        std_improvement = np.std(recent_improvements)
        success_rate = sum(1 for imp in recent_improvements if imp > 0.01) / len(recent_improvements)

        # Detect plateau: low variance and low success rate
        is_plateau = (std_improvement < 0.05 and success_rate < 0.3 and mean_improvement < 0.3)

        if is_plateau and episode_idx % 200 == 0:  # Check every 200 episodes
            self.logger.info(f"🚨 PLATEAU DETECTED at episode {episode_idx}")
            self.logger.info(f"   Mean improvement: {mean_improvement:.4f}")
            self.logger.info(f"   Std improvement: {std_improvement:.4f}")
            self.logger.info(f"   Success rate: {success_rate:.3f}")

            # Trigger exploration boost
            self._trigger_exploration_boost(episode_idx)
            return True

        return False

    def _trigger_exploration_boost(self, episode_idx: int):
        """
        🚀 NEW: Trigger temporary exploration boost to escape plateau
        """
        self.logger.info("🚀 TRIGGERING EXPLORATION BOOST")

        # Temporarily increase epsilon
        if self.config.use_epsilon_greedy:
            self.current_epsilon = min(0.4, self.current_epsilon + 0.2)
            self.logger.info(f"   Boosted epsilon to {self.current_epsilon:.3f}")

        # Temporarily increase entropy coefficient
        self.current_entropy_coef = min(0.25, self.current_entropy_coef + 0.1)
        self.logger.info(f"   Boosted entropy coefficient to {self.current_entropy_coef:.3f}")

        # Reset experience buffer exploration tracking
        self.experience_buffer.reset_exploration_tracking()
        self.logger.info("   Reset exploration tracking")

        # Log boost to TensorBoard
        self.writer.add_scalar('Boost/ExplorationBoostTriggered', 1.0, episode_idx)
        self.writer.add_scalar('Boost/BoostedEpsilon', self.current_epsilon, episode_idx)
        self.writer.add_scalar('Boost/BoostedEntropyCoef', self.current_entropy_coef, episode_idx)

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.mean().item(),
            'entropy_loss': entropy_loss.item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'current_lr_actor': current_lr_actor,
            'current_lr_critic': current_lr_critic,
            'current_entropy_coef': self.current_entropy_coef,
            'current_epsilon': self.current_epsilon,
            'current_adversarial_weight': self.current_adversarial_weight
        }

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function"""

    # Configuration con CyclicLR abilitato
    config = TrainingConfig(
        experiment_name="graph_refactor_a2c_optimized_v3",

        # Episodes ottimizzati
        num_episodes=2200,
        warmup_episodes=400,
        adversarial_start_episode=500,

        # Learning rates ottimizzati
        lr_actor=3e-4,
        lr_critic=1e-4,
        lr_discriminator=1e-4,

        # CyclicLR ottimizzato
        use_cyclic_lr=True,
        base_lr_actor=3e-4,
        max_lr_actor=1e-3,
        step_size_up_actor=150,
        base_lr_critic=1e-4,
        max_lr_critic=3e-4,
        step_size_up_critic=200,

        # Training ottimizzato
        batch_size=32,
        update_every=20,
        discriminator_update_every=40,

        # Regularization ottimizzata
        entropy_coef=0.1,
        max_grad_norm=0.5,
        advantage_clip=0.5,
        reward_clip=5.0,
    )

    print("🚀 Starting Graph Refactoring RL Training with CyclicLR (FIXED)")
    print(f"📊 Configuration: {config.experiment_name}")
    print(f"🎯 Episodes: {config.num_episodes}")
    print(f"🔄 CyclicLR enabled: {config.use_cyclic_lr}")
    if config.use_cyclic_lr:
        print(f"   Actor LR: {config.base_lr_actor:.2e} ↔ {config.max_lr_actor:.2e}")
        print(f"   Critic LR: {config.base_lr_critic:.2e} ↔ {config.max_lr_critic:.2e}")
        print(f"   Step sizes: Actor {config.step_size_up_actor}, Critic {config.step_size_up_critic}")
    print(f"🧠 Device: {config.device}")

    # Initialize trainer
    trainer = A2CTrainer(config)

    # Start training
    try:
        training_stats = trainer.train()
        print("✅ Training completed successfully!")

        # Print final scheduler info
        if config.use_cyclic_lr and hasattr(trainer, 'actor_scheduler'):
            final_lr_actor = trainer.actor_scheduler.get_last_lr()[0]
            final_lr_critic = trainer.critic_scheduler.get_last_lr()[0]
            print(f"📈 Final LRs - Actor: {final_lr_actor:.2e}, Critic: {final_lr_critic:.2e}")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
        trainer._save_checkpoint(0, is_best=False)

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

    finally:
        trainer.writer.close()


if __name__ == "__main__":
    main()