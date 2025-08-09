#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) Trainer for Graph Refactoring
Implements the complete training pipeline with adversarial discriminator updates.
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

# Import custom modules
from rl_gym import RefactorEnv
from actor_critic_models import ActorCritic, create_actor_critic
from discriminator import create_discriminator


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
    reward_weights: Dict[str, float] = None

    # Training phases
    num_episodes: int = 2000
    warmup_episodes: int = 500
    adversarial_start_episode: int = 1000

    # A2C hyperparameters
    gamma: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_discriminator: float = 1e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    use_cyclic_lr: bool = True

    # Actor scheduler parameters
    base_lr_actor: float = 5e-4
    max_lr_actor: float = 2e-3
    step_size_up_actor: int = 100
    step_size_down_actor: int = None
    cyclic_mode_actor: str = "triangular2"  # "triangular", "triangular2", "exp_range"
    gamma_actor: float = 0.99

    # Critic scheduler parameters
    base_lr_critic: float = 1e-3
    max_lr_critic: float = 5e-3
    step_size_up_critic: int = 80
    step_size_down_critic: int = None
    cyclic_mode_critic: str = "triangular2"
    gamma_critic: float = 0.99

    # Training details
    batch_size: int = 16
    update_every: int = 10
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
    early_stopping_patience: int = 200
    min_improvement: float = 0.01

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Ensure adversarial start is after warmup
        if self.adversarial_start_episode < self.warmup_episodes:
            self.adversarial_start_episode = self.warmup_episodes

        # Validate paths
        if self.discriminator_path and not Path(self.discriminator_path).exists():
            logging.warning(f"Discriminator path does not exist: {self.discriminator_path}")

        # Create results directory
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        # --- NUOVO: Validation per CyclicLR parameters ---
        if self.use_cyclic_lr:
            # Assicurati che base_lr <= max_lr
            if self.base_lr_actor > self.max_lr_actor:
                logging.warning("base_lr_actor > max_lr_actor, swapping values")
                self.base_lr_actor, self.max_lr_actor = self.max_lr_actor, self.base_lr_actor

            if self.base_lr_critic > self.max_lr_critic:
                logging.warning("base_lr_critic > max_lr_critic, swapping values")
                self.base_lr_critic, self.max_lr_critic = self.max_lr_critic, self.base_lr_critic

            # Set step_size_down se None
            if self.step_size_down_actor is None:
                self.step_size_down_actor = self.step_size_up_actor
            if self.step_size_down_critic is None:
                self.step_size_down_critic = self.step_size_up_critic

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary, filtering unknown parameters"""
        # Get valid field names
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}

        # Filter config_dict to only include valid fields
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
    """Load pre-trained discriminator"""
    if not os.path.exists(discriminator_path):
        logging.warning(f"Discriminator not found at {discriminator_path}. Training without discriminator.")
        return None

    try:
        checkpoint = torch.load(discriminator_path, map_location=device)
        model_config = checkpoint['model_config']

        discriminator = create_discriminator(**model_config)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        discriminator.to(device)
        discriminator.eval()

        logging.info(
            f"✅ Loaded pre-trained discriminator with F1: {checkpoint.get('cv_results', {}).get('mean_f1', 'N/A')}")
        return discriminator

    except Exception as e:
        logging.error(f"Failed to load discriminator: {e}")
        return None

@staticmethod
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
    """Buffer for storing and managing RL experiences"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.rewards = deque(maxlen=1000)  # For reward normalization

    def add(self, state_data: Data, global_features: torch.Tensor,
            action: int, reward: float, next_state_data: Data,
            next_global_features: torch.Tensor, done: bool,
            log_prob: float, value: float):
        """Add experience to buffer con Data CLEAN"""

        # *** MODIFICA: Crea copie PULITE ***
        def clean_data_copy(data):
            if data is None:
                return None
            return Data(
                x=data.x.clone(),
                edge_index=data.edge_index.clone(),
                num_nodes=data.x.size(0)  # Esplicito
            )

        experience = {
            'state_data': clean_data_copy(state_data),
            'global_features': global_features.clone(),
            'action': action,
            'reward': reward,
            'next_state_data': clean_data_copy(next_state_data),
            'next_global_features': next_global_features.clone() if next_global_features is not None else None,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }

        self.buffer.append(experience)
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

        # Initialize environment
        self.env = RefactorEnv(
            data_path=config.data_path,
            max_steps=config.max_episode_steps,
            device=config.device,
            reward_weights=config.reward_weights  # ✅ PASSA I REWARD WEIGHTS!
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

        # Best model tracking
        self.best_reward = float('-inf')
        self.early_stopping_counter = 0

        self.logger.info("🚀 A2C Trainer initialized successfully!")

    def _init_models(self):
        """Initialize actor-critic and discriminator models"""

        # Actor-Critic model
        ac_config = {
            'node_dim': 7,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'num_actions': 7,  # Aggiornato per 7 azioni
            'global_features_dim': 10,
            'dropout': self.config.dropout,
            'shared_encoder': False  # Disabilitato shared encoder
        }

        self.actor_critic = create_actor_critic(ac_config).to(self.device)

        self.logger.info(f"Actor-Critic model parameters: {sum(p.numel() for p in self.actor_critic.parameters()):,}")
        self.logger.info(f"Using separate encoders for Actor and Critic")

    def _init_optimizers(self):
        """Initialize optimizers and schedulers for all models"""

        # --- NUOVO: CyclicLR Schedulers ---
        if self.config.use_cyclic_lr:
            self.logger.info("🔄 Setting up CyclicLR schedulers...")

            # Separa Actor e Critic optimizer
            # Identifica i parametri dell'actor e del critic correttamente
            if hasattr(self.actor_critic, 'actor_encoder') and hasattr(self.actor_critic, 'critic_encoder'):
                # Caso: encoder separati
                actor_params = list(self.actor_critic.actor_encoder.parameters()) + \
                               list(self.actor_critic.actor_head.parameters())
                critic_params = list(self.actor_critic.critic_encoder.parameters()) + \
                                list(self.actor_critic.critic_head.parameters())
            elif hasattr(self.actor_critic, 'shared_encoder') and self.actor_critic.shared_encoder:
                # Caso: encoder condiviso - separa solo le head
                shared_params = list(self.actor_critic.encoder.parameters())
                actor_head_params = list(self.actor_critic.actor_head.parameters())
                critic_head_params = list(self.actor_critic.critic_head.parameters())

                # Per encoder condiviso, ottimizziamo tutto insieme ma con scheduler diversi per le head
                actor_params = shared_params + actor_head_params
                critic_params = shared_params + critic_head_params
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
                        # Parametri condivisi vanno in entrambi
                        actor_params.append(param)
                        critic_params.append(param)

            # Crea optimizer separati
            self.actor_optimizer = optim.Adam(actor_params, lr=self.config.lr_actor, eps=1e-5)
            self.critic_optimizer = optim.Adam(critic_params, lr=self.config.lr_critic, eps=1e-5)

            # Calcola step_size basato sui update invece che sugli episodi
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

            self.logger.info(
                f"✅ CyclicLR setup complete:\n"
                f"   Actor: {self.config.base_lr_actor:.2e} ↔ {self.config.max_lr_actor:.2e} "
                f"(step_size: {actor_step_size_up}↑/{actor_step_size_down}↓)\n"
                f"   Critic: {self.config.base_lr_critic:.2e} ↔ {self.config.max_lr_critic:.2e} "
                f"(step_size: {critic_step_size_up}↑/{critic_step_size_down}↓)\n"
                f"   Mode: {self.config.cyclic_mode_actor}/{self.config.cyclic_mode_critic}"
            )

            # Non creare ac_optimizer quando usi CyclicLR
            self.ac_optimizer = None

        else:
            # Fallback: Optimizer combinato tradizionale
            self.ac_optimizer = optim.Adam(
                self.actor_critic.parameters(),
                lr=self.config.lr_actor,
                eps=1e-5
            )

            # Learning rate scheduler tradizionale
            self.ac_scheduler = optim.lr_scheduler.StepLR(
                self.ac_optimizer, step_size=500, gamma=0.8
            )
            self.logger.info("📉 Using traditional combined optimizer with StepLR scheduler")

        # Discriminator optimizer (if available)
        if self.discriminator is not None:
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.lr_discriminator
            )

            # Discriminator scheduler (sempre StepLR per semplicità)
            self.discriminator_scheduler = optim.lr_scheduler.StepLR(
                self.discriminator_optimizer, step_size=300, gamma=0.9
            )

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """Extract global features from graph data"""
        global_features = self.env._extract_global_features(data)

        # Assicurati che abbia dimensione batch
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        else:
            print("Le dimensioni di global_features non matchano")

        return global_features

    def _collect_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Collect a single episode of experience con Data CLEAN"""

        # Reset environment
        state = self.env.reset()
        current_data = ensure_clean_data(self.env.current_data)

        episode_reward = 0.0
        episode_length = 0

        initial_hub_score = self.env.initial_metrics['hub_score']

        done = False
        while not done:
            # Get action
            global_features = self._extract_global_features(current_data)

            action, log_prob, value = self.actor_critic.get_action_and_value(
                current_data, global_features
            )

            # Take action
            next_state, reward, done, info = self.env.step(action)
            current_data = ensure_clean_data(self.env.current_data)

            episode_reward += reward
            episode_length += 1

        final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
        hub_score_improvement = initial_hub_score - final_hub_score  # Positivo = miglioramento

        best_hub_improvement = initial_hub_score - self.env.best_hub_score

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'hub_score_improvement': hub_score_improvement,
            'best_hub_improvement': best_hub_improvement,
            'initial_hub_score': initial_hub_score,
            'final_hub_score': final_hub_score,
            'best_hub_score': self.env.best_hub_score,
            'no_improve_steps': self.env.no_improve_steps
        }

    def _modify_reward(self, reward: float, episode_idx: int, info: Dict) -> float:
        """Modify reward based on training phase - VERSIONE OTTIMIZZATA"""

        # Warm-up phase: focus solo su hub score con scaling migliore
        if episode_idx < self.config.warmup_episodes:
            hub_improvement = info.get('metrics', {}).get('hub_score', 0) - \
                              self.env.initial_metrics.get('hub_score', 0)

            # Scala il reward per warm-up con bonus/penalty
            if hub_improvement > 2.0:
                scaled_reward = hub_improvement * 3.0 + 5.0
            elif hub_improvement > 0.5:
                scaled_reward = hub_improvement * 2.0 + 2.0
            elif hub_improvement < -1.0:
                scaled_reward = hub_improvement * 2.0 - 3.0
            else:
                scaled_reward = hub_improvement

            return np.clip(scaled_reward, -5.0, 10.0)

        # Normal training: usa reward completo ma con normalizzazione migliorata
        modified_reward = reward

        # Normalizzazione più conservativa
        if self.config.normalize_rewards and len(self.experience_buffer.rewards) > 50:
            recent_rewards = list(self.experience_buffer.rewards)[-100:]  # Ultimi 100
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards) + 1e-6

            # Normalizzazione più soft
            modified_reward = (modified_reward - mean_reward) / (std_reward * 2.0)

        # Clipping finale più stretto
        modified_reward = np.clip(modified_reward, -3.0, 5.0)

        return modified_reward

    def _compute_advantages(self, experiences: List[Dict]) -> List[float]:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        advantages = []
        returns = []

        # Compute returns and advantages
        R = 0
        for i in reversed(range(len(experiences))):
            exp = experiences[i]

            if exp['done']:
                R = 0

            R = exp['reward'] + self.config.gamma * R
            returns.insert(0, R)

            # Advantage = R - V(s)
            advantage = R - exp['value']
            advantages.insert(0, advantage)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = np.array(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = np.clip(advantages, -self.config.advantage_clip, self.config.advantage_clip)
            advantages = advantages.tolist()

        return advantages, returns

    def _update_actor_critic(self):
        """Update actor-critic networks using collected experiences"""

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

        # Total loss (solo per logging)
        total_loss = (actor_loss +
                      self.config.value_loss_coef * critic_loss -
                      self.config.entropy_coef * entropy.mean())

        # --- CORREZIONE: Optimization step con gestione corretta dei gradienti ---
        if self.config.use_cyclic_lr and hasattr(self, 'actor_optimizer'):
            # Usa optimizer separati

            # Zero gradients
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Compute losses separatamente
            actor_total_loss = actor_loss - self.config.entropy_coef * entropy.mean()
            critic_total_loss = self.config.value_loss_coef * critic_loss

            # Backward pass separati
            actor_total_loss.backward(retain_graph=True)
            critic_total_loss.backward()

            # Gradient clipping corretto per optimizer separati
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

            # Optimization steps
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Learning rates per logging
            current_lr_actor = self.actor_optimizer.param_groups[0]['lr']
            current_lr_critic = self.critic_optimizer.param_groups[0]['lr']

            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'total_loss': total_loss.item(),
                'entropy': entropy.mean().item(),
                'advantages_mean': advantages.mean().item(),
                'advantages_std': advantages.std().item(),
                'current_lr_actor': current_lr_actor,
                'current_lr_critic': current_lr_critic
            }

        else:
            # Fallback: optimizer combinato
            if self.ac_optimizer is None:
                raise ValueError("ac_optimizer is None but cyclic_lr is disabled. Check _init_optimizers().")

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
                'advantages_mean': advantages.mean().item(),
                'advantages_std': advantages.std().item(),
                'current_lr': current_lr
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
            # --- CORREZIONE: Assicurati che num_nodes sia presente ---
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
        """Evaluate model performance"""

        self.actor_critic.eval()

        eval_rewards = []
        eval_hub_improvements = []
        eval_discriminator_scores = []

        for _ in range(self.config.num_eval_episodes):
            # Reset environment
            self.env.reset()
            current_data = self.env.current_data.clone()

            episode_reward = 0.0
            initial_hub_score = self.env.initial_metrics['hub_score']

            done = False
            while not done:
                global_features = self._extract_global_features(current_data)

                # Use greedy action selection for evaluation
                with torch.no_grad():
                    output = self.actor_critic(current_data, global_features)
                    action = torch.argmax(output['action_probs'], dim=1).item()

                _, reward, done, info = self.env.step(action)
                episode_reward += reward
                current_data = self.env.current_data.clone()

            eval_rewards.append(episode_reward)

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
            'eval_discriminator_score_std': np.std(eval_discriminator_scores) if eval_discriminator_scores else 0.0
        }

    def _log_metrics(self, episode_idx: int, episode_info: Dict, update_info: Dict = None):
        """Logging migliorato con metriche final-only (versione corretta)"""

        # Update training statistics
        self.training_stats['episode_rewards'].append(episode_info['episode_reward'])
        self.training_stats['episode_lengths'].append(episode_info['episode_length'])

        final_improvement = episode_info.get('hub_score_improvement', 0)
        self.training_stats['hub_score_improvements'].append(final_improvement)  # ← FIX: Cambia qui

        if update_info:
            if 'actor_loss' in update_info:
                self.training_stats['actor_losses'].append(update_info['actor_loss'])
            if 'critic_loss' in update_info:
                self.training_stats['critic_losses'].append(update_info['critic_loss'])
            if 'discriminator_loss' in update_info:
                self.training_stats['discriminator_losses'].append(update_info['discriminator_loss'])

        # TensorBoard logging
        self.writer.add_scalar('Episode/Reward', episode_info['episode_reward'], episode_idx)
        self.writer.add_scalar('Episode/Length', episode_info['episode_length'], episode_idx)
        self.writer.add_scalar('Episode/HubImprovement', final_improvement, episode_idx)
        self.writer.add_scalar('Episode/BestHubImprovement', episode_info.get('best_hub_improvement', 0), episode_idx)

        if update_info:
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, episode_idx)

        # Console logging
        if episode_idx % self.config.log_every == 0:
            recent_rewards = self.training_stats['episode_rewards'][-self.config.log_every:]
            recent_improvements = self.training_stats['hub_score_improvements'][-self.config.log_every:]

            success_rate = sum(1 for imp in recent_improvements if imp > 0.01) / len(recent_improvements)

            self.logger.info(
                f"Episode {episode_idx:4d} | "
                f"Reward: {np.mean(recent_rewards):6.3f}±{np.std(recent_rewards):5.3f} | "
                f"Hub Δ (final): {np.mean(recent_improvements):6.3f}±{np.std(recent_improvements):5.3f} | "
                f"Success: {success_rate:.1%} | "
                f"Buffer: {len(self.experience_buffer)}"
            )

    def _save_checkpoint(self, episode_idx: int, is_best: bool = False):
        """Save model checkpoint"""

        checkpoint = {
            'episode': episode_idx,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'model_config': {  # ← Aggiungi questo
                'node_dim': 7,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_actions': 7,
                'global_features_dim': 10,
                'dropout': self.config.dropout,
                'shared_encoder': False
            },
            'training_stats': self.training_stats,
            'config': self.config,
            'best_reward': self.best_reward
        }

        # --- CORREZIONE: Salva optimizer states corretti ---
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
        """Main training loop"""

        self.logger.info("🚀 Starting A2C training...")
        self.logger.info(f"Training for {self.config.num_episodes} episodes")
        self.logger.info(f"Warm-up episodes: {self.config.warmup_episodes}")
        self.logger.info(f"Adversarial training starts at episode: {self.config.adversarial_start_episode}")

        start_time = time.time()
        update_count = 0  # Traccia il numero di update per il scheduler

        for episode_idx in range(1, self.config.num_episodes + 1):

            # Collect episode
            episode_info = self._collect_episode(episode_idx)

            # Update actor-critic every few episodes
            update_info = {}
            if episode_idx % self.config.update_every == 0:
                update_info.update(self._update_actor_critic())
                update_count += 1

                # --- CORREZIONE: Step per CyclicLR schedulers ---
                if self.config.use_cyclic_lr:
                    # Usa scheduler separati
                    if hasattr(self, 'actor_scheduler') and hasattr(self, 'critic_scheduler'):
                        self.actor_scheduler.step()
                        self.critic_scheduler.step()

                        # Log learning rates attuali ogni tanto
                        if update_count % 50 == 0:
                            current_lr_actor = self.actor_scheduler.get_last_lr()[0]
                            current_lr_critic = self.critic_scheduler.get_last_lr()[0]

                            self.logger.info(
                                f"📊 Update {update_count}: "
                                f"Actor LR = {current_lr_actor:.2e}, "
                                f"Critic LR = {current_lr_critic:.2e}"
                            )

                            # Aggiungi al TensorBoard
                            self.writer.add_scalar('Learning_Rate/actor', current_lr_actor, update_count)
                            self.writer.add_scalar('Learning_Rate/critic', current_lr_critic, update_count)
                    else:
                        self.logger.warning("⚠️ CyclicLR enabled but schedulers not found!")

                else:
                    # Scheduler tradizionale (ogni 100 episodi invece che ogni update)
                    if episode_idx % 100 == 0 and hasattr(self, 'ac_scheduler'):
                        self.ac_scheduler.step()

            # Update discriminator in adversarial phase
            if (episode_idx >= self.config.adversarial_start_episode and
                    episode_idx % self.config.discriminator_update_every == 0):
                discriminator_info = self._update_discriminator()
                update_info.update(discriminator_info)

                # Step discriminator scheduler
                if hasattr(self, 'discriminator_scheduler'):
                    self.discriminator_scheduler.step()

            # Log metrics
            self._log_metrics(episode_idx, episode_info, update_info)

            # Evaluation
            if episode_idx % self.config.eval_every == 0:
                eval_metrics = self._evaluate_model()

                self.logger.info(
                    f"📊 Evaluation at episode {episode_idx}: "
                    f"Reward: {eval_metrics['eval_reward_mean']:.3f}±{eval_metrics['eval_reward_std']:.3f} | "
                    f"Hub Δ: {eval_metrics['eval_hub_improvement_mean']:.3f}±{eval_metrics['eval_hub_improvement_std']:.3f}"
                )

                # Log evaluation metrics to TensorBoard
                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f'Evaluation/{key}', value, episode_idx)

                # Check for best model
                current_reward = eval_metrics['eval_reward_mean']
                if current_reward > self.best_reward + self.config.min_improvement:
                    self.best_reward = current_reward
                    self.early_stopping_counter = 0
                    self._save_checkpoint(episode_idx, is_best=True)
                else:
                    self.early_stopping_counter += 1

            # Save regular checkpoint
            if episode_idx % self.config.save_every == 0:
                self._save_checkpoint(episode_idx)

            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"🛑 Early stopping at episode {episode_idx}")
                break

        # Final evaluation and save
        final_eval = self._evaluate_model()
        self._save_checkpoint(episode_idx, is_best=False)

        training_time = time.time() - start_time
        self.logger.info(f"✅ Training completed in {training_time / 3600:.2f} hours")
        self.logger.info(f"Best reward: {self.best_reward:.3f}")
        self.logger.info(f"Final evaluation: {final_eval}")

        if self.config.use_cyclic_lr:
            if hasattr(self, 'actor_scheduler') and hasattr(self, 'critic_scheduler'):
                final_lr_actor = self.actor_scheduler.get_last_lr()[0]
                final_lr_critic = self.critic_scheduler.get_last_lr()[0]
                self.logger.info(f"📈 Final learning rates - Actor: {final_lr_actor:.2e}, Critic: {final_lr_critic:.2e}")
            else:
                self.logger.info("📈 CyclicLR enabled but no scheduler info available")
        else:
            if hasattr(self, 'ac_scheduler'):
                final_lr = self.ac_scheduler.get_last_lr()[0]
                self.logger.info(f"📈 Final learning rate: {final_lr:.2e}")

        self.logger.info(f"🔄 Total scheduler updates: {update_count}")

        # Save final results
        self._save_final_results(final_eval, training_time)

        return self.training_stats

    def _save_final_results(self, final_eval: Dict, training_time: float):
        """Save final training results and plots"""

        # Save training statistics
        results = {
            'config': self.config.__dict__,
            'training_stats': self.training_stats,
            'final_evaluation': final_eval,
            'training_time_hours': training_time / 3600,
            'best_reward': self.best_reward
        }

        with open(self.results_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Create plots
        self._create_training_plots()

        self.logger.info(f"📁 Results saved to {self.results_dir}")

    def _create_training_plots(self):
        """Create training progress plots"""

        try:
            # Episode rewards plot
            plt.figure(figsize=(15, 10))

            # Subplot 1: Episode rewards
            plt.subplot(2, 3, 1)
            rewards = self.training_stats['episode_rewards']
            if rewards:
                plt.plot(rewards, alpha=0.3, color='blue')
                # Moving average
                window = min(100, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(rewards)), moving_avg, color='red', linewidth=2)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)

            # Subplot 2: Hub score improvements
            plt.subplot(2, 3, 2)
            hub_improvements = self.training_stats['hub_score_improvements']
            if hub_improvements:
                plt.plot(hub_improvements, alpha=0.3, color='green')
                # Moving average
                if len(hub_improvements) > window:
                    moving_avg = np.convolve(hub_improvements, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(hub_improvements)), moving_avg, color='darkgreen', linewidth=2)
                plt.title('Hub Score Improvements')
                plt.xlabel('Episode')
                plt.ylabel('Hub Score Δ')
                plt.grid(True, alpha=0.3)

            # Subplot 3: Actor losses
            plt.subplot(2, 3, 3)
            if self.training_stats['actor_losses']:
                plt.plot(self.training_stats['actor_losses'], color='orange')
                plt.title('Actor Loss')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            # Subplot 4: Critic losses
            plt.subplot(2, 3, 4)
            if self.training_stats['critic_losses']:
                plt.plot(self.training_stats['critic_losses'], color='purple')
                plt.title('Critic Loss')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            # Subplot 5: Episode lengths
            plt.subplot(2, 3, 5)
            lengths = self.training_stats['episode_lengths']
            if lengths:
                plt.plot(lengths, alpha=0.3, color='brown')
                if len(lengths) > window:
                    moving_avg = np.convolve(lengths, np.ones(window) / window, mode='valid')
                    plt.plot(range(window - 1, len(lengths)), moving_avg, color='darkred', linewidth=2)
                plt.title('Episode Lengths')
                plt.xlabel('Episode')
                plt.ylabel('Steps')
                plt.grid(True, alpha=0.3)

            # Subplot 6: Discriminator losses
            plt.subplot(2, 3, 6)
            if self.training_stats['discriminator_losses']:
                plt.plot(self.training_stats['discriminator_losses'], color='cyan')
                plt.title('Discriminator Loss')
                plt.xlabel('Update')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.warning(f"Failed to create training plots: {e}")


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function"""

    # Configuration con CyclicLR abilitato
    config = TrainingConfig(
        experiment_name="graph_refactor_a2c_optimized_v2",

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

    print("🚀 Starting Graph Refactoring RL Training with CyclicLR")
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