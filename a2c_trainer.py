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
    num_actions: int = 5

    # Training phases
    num_episodes: int = 2000
    warmup_episodes: int = 500  # Warm-up with only Δhub_score reward
    adversarial_start_episode: int = 1000  # When to start adversarial training

    # A2C hyperparameters
    gamma: float = 0.95  # Discount factor
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_discriminator: float = 1e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Training details
    batch_size: int = 32
    update_every: int = 20  # Steps between updates
    discriminator_update_every: int = 100  # Episodes between discriminator updates

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 10.0
    advantage_clip: float = 2.0

    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    shared_encoder: bool = True

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
        """Add experience to buffer"""

        experience = {
            'state_data': state_data.clone(),
            'global_features': global_features.clone(),
            'action': action,
            'reward': reward,
            'next_state_data': next_state_data.clone() if next_state_data is not None else None,
            'next_global_features': next_global_features.clone() if next_global_features is not None else None,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }

        self.buffer.append(experience)
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
            device=config.device
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
            'num_actions': self.config.num_actions,
            'global_features_dim': 10,
            'dropout': self.config.dropout,
            'shared_encoder': self.config.shared_encoder
        }

        self.actor_critic = create_actor_critic(ac_config).to(self.device)

        self.logger.info(f"Actor-Critic model parameters: {sum(p.numel() for p in self.actor_critic.parameters()):,}")

    def _init_optimizers(self):
        """Initialize optimizers for all models"""

        # Actor-Critic optimizer (combined)
        self.ac_optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.config.lr_actor,
            eps=1e-5
        )

        # Discriminator optimizer (if available)
        if self.discriminator is not None:
            self.discriminator_optimizer = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.lr_discriminator
            )

        # Learning rate schedulers
        self.ac_scheduler = optim.lr_scheduler.StepLR(
            self.ac_optimizer, step_size=500, gamma=0.8
        )

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """Extract global features from graph data"""
        # This should match the metrics calculated in the environment
        metrics = self.env._calculate_metrics(data)

        global_features = torch.tensor([
            metrics['hub_score'],
            metrics['density'],
            metrics['modularity'],
            metrics['avg_shortest_path'] if metrics['avg_shortest_path'] != float('inf') else 10.0,
            metrics['clustering'],
            metrics['avg_betweenness'],
            metrics['avg_degree_centrality'],
            metrics['num_nodes'],
            metrics['num_edges'],
            metrics['connected']
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        return global_features

    def _collect_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Collect a single episode of experience"""

        # Reset environment
        state = self.env.reset()
        current_data = self.env.current_data.clone()

        episode_reward = 0.0
        episode_length = 0
        episode_experiences = []

        initial_hub_score = self.env.initial_metrics['hub_score']

        done = False
        while not done:
            # Extract global features
            global_features = self._extract_global_features(current_data)

            # Get action and value from actor-critic
            action, log_prob, value = self.actor_critic.get_action_and_value(
                current_data, global_features
            )

            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            next_data = self.env.current_data.clone()

            # Apply reward modifications based on training phase
            modified_reward = self._modify_reward(reward, episode_idx, info)

            # Extract next global features
            next_global_features = self._extract_global_features(next_data) if not done else None

            # Store experience
            self.experience_buffer.add(
                state_data=current_data,
                global_features=global_features,
                action=action,
                reward=modified_reward,
                next_state_data=next_data if not done else None,
                next_global_features=next_global_features,
                done=done,
                log_prob=log_prob,
                value=value
            )

            episode_experiences.append({
                'reward': modified_reward,
                'value': value,
                'done': done
            })

            episode_reward += modified_reward
            episode_length += 1
            current_data = next_data

        # Calculate final hub score improvement
        final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
        hub_score_improvement = final_hub_score - initial_hub_score

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'hub_score_improvement': hub_score_improvement,
            'experiences': episode_experiences,
            'initial_hub_score': initial_hub_score,
            'final_hub_score': final_hub_score
        }

    def _modify_reward(self, reward: float, episode_idx: int, info: Dict) -> float:
        """Modify reward based on training phase"""

        # Warm-up phase: only hub score improvement
        if episode_idx < self.config.warmup_episodes:
            # Focus only on hub score changes
            hub_improvement = info.get('metrics', {}).get('hub_score', 0) - \
                              self.env.initial_metrics.get('hub_score', 0)
            return hub_improvement

        # Normal training: use full reward
        modified_reward = reward

        # Normalize reward if enabled
        if self.config.normalize_rewards:
            modified_reward = self.experience_buffer.get_normalized_reward(modified_reward)

        # Clip reward
        modified_reward = np.clip(modified_reward, -self.config.reward_clip, self.config.reward_clip)

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
            states_data.append(exp['state_data'])
            global_features_list.append(exp['global_features'])
            actions.append(exp['action'])
            returns.append(exp['reward'])  # Simplified - should use proper returns
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

        # Total loss
        total_loss = (actor_loss +
                      self.config.value_loss_coef * critic_loss -
                      self.config.entropy_coef * entropy.mean())

        # Optimization step
        self.ac_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
        self.ac_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.mean().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
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
            # Use state as positive (original smelly) and modified state as negative
            positive_graphs.append(exp['state_data'])
            if exp['next_state_data'] is not None:
                negative_graphs.append(exp['next_state_data'])

        if not negative_graphs:
            return {}

        # Create balanced dataset
        min_size = min(len(positive_graphs), len(negative_graphs))
        positive_graphs = positive_graphs[:min_size]
        negative_graphs = negative_graphs[:min_size]

        # Combine and create labels
        all_graphs = positive_graphs + negative_graphs
        labels = [1] * len(positive_graphs) + [0] * len(negative_graphs)

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
            hub_improvement = final_hub_score - initial_hub_score
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
        """Log training metrics"""

        # Update training statistics
        self.training_stats['episode_rewards'].append(episode_info['episode_reward'])
        self.training_stats['episode_lengths'].append(episode_info['episode_length'])
        self.training_stats['hub_score_improvements'].append(episode_info['hub_score_improvement'])

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
        self.writer.add_scalar('Episode/HubScoreImprovement', episode_info['hub_score_improvement'], episode_idx)

        if update_info:
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, episode_idx)

        # Console logging
        if episode_idx % self.config.log_every == 0:
            recent_rewards = self.training_stats['episode_rewards'][-self.config.log_every:]
            recent_hub_improvements = self.training_stats['hub_score_improvements'][-self.config.log_every:]

            self.logger.info(
                f"Episode {episode_idx:4d} | "
                f"Reward: {np.mean(recent_rewards):6.3f}±{np.std(recent_rewards):5.3f} | "
                f"Hub Δ: {np.mean(recent_hub_improvements):6.3f}±{np.std(recent_hub_improvements):5.3f} | "
                f"Buffer: {len(self.experience_buffer)}"
            )

    def _save_checkpoint(self, episode_idx: int, is_best: bool = False):
        """Save model checkpoint"""

        checkpoint = {
            'episode': episode_idx,
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'ac_optimizer_state_dict': self.ac_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config,
            'best_reward': self.best_reward
        }

        if self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
            checkpoint['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()

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

        for episode_idx in range(1, self.config.num_episodes + 1):

            # Collect episode
            episode_info = self._collect_episode(episode_idx)

            # Update actor-critic every few episodes
            update_info = {}
            if episode_idx % self.config.update_every == 0:
                update_info.update(self._update_actor_critic())

            # Update discriminator in adversarial phase
            if (episode_idx >= self.config.adversarial_start_episode and
                    episode_idx % self.config.discriminator_update_every == 0):
                discriminator_info = self._update_discriminator()
                update_info.update(discriminator_info)

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

            # Learning rate scheduling
            if episode_idx % 100 == 0:
                self.ac_scheduler.step()

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

    # Configuration
    config = TrainingConfig(
        experiment_name="graph_refactor_a2c_v1",
        num_episodes=2000,
        warmup_episodes=500,
        adversarial_start_episode=1000,
        hidden_dim=128,
        lr_actor=3e-4,
        lr_critic=1e-3,
        batch_size=32,
        update_every=20
    )

    print("🚀 Starting Graph Refactoring RL Training")
    print(f"📊 Configuration: {config.experiment_name}")
    print(f"🎯 Episodes: {config.num_episodes}")
    print(f"🔥 Warm-up: {config.warmup_episodes}")
    print(f"⚔️  Adversarial: {config.adversarial_start_episode}")
    print(f"🧠 Device: {config.device}")

    # Initialize trainer
    trainer = A2CTrainer(config)

    # Start training
    try:
        training_stats = trainer.train()
        print("✅ Training completed successfully!")

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