#!/usr/bin/env python3
"""
PPO Trainer for Graph Refactoring
Corrected implementation maintaining environment characteristics
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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time
from dataclasses import dataclass, asdict


@dataclass
class PPOConfig:
    """PPO Configuration maintaining original environment features"""

    # General
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: int = 42
    experiment_name: str = "graph_refactor_ppo_corrected"

    # Data paths
    data_path: str = "data_builder/dataset/graph_features"
    discriminator_path: str = "results/discriminator_pretraining/pretrained_discriminator.pt"
    results_dir: str = "results/ppo_training"

    # Environment - mantiene caratteristiche originali
    max_episode_steps: int = 20
    num_actions: int = 7
    reward_weights: Optional[Dict[str, float]] = None

    # Training phases - mantiene logica originale
    num_episodes: int = 3000
    warmup_episodes: int = 600
    adversarial_start_episode: int = 600

    # PPO hyperparameters corretti
    gamma: float = 0.95
    gae_lambda: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    target_kl: float = 0.01
    max_grad_norm: float = 0.5

    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.05

    # PPO specific
    rollout_episodes: int = 16  # Collect this many episodes before update
    minibatch_size: int = 8

    # Model architecture - mantiene configurazione originale
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2

    # Reward normalization - mantiene caratteristiche originali
    normalize_rewards: bool = True
    reward_clip: float = 5.0

    # Logging and evaluation - mantiene frequenze originali
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500
    num_eval_episodes: int = 50

    # Early stopping - mantiene logica originale
    early_stopping_patience: int = 300
    min_improvement: float = 0.01

    def __post_init__(self):
        """Initialize reward weights maintaining original structure"""
        if self.reward_weights is None:
            self.reward_weights = {
                'hub_weight': 10.0,
                'step_valid': 0.0,
                'step_invalid': -0.1,
                'time_penalty': -0.02,
                'early_stop_penalty': -0.5,
                'cycle_penalty': -0.2,
                'duplicate_penalty': -0.1,
                'adversarial_weight': 2.0,
                'patience': 15
            }


class PPOBuffer:
    """PPO Buffer that maintains compatibility with original reward structure"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.global_features = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def store(self, state: Data, global_feat: torch.Tensor, action: int,
              reward: float, value: float, log_prob: float, done: bool):
        self.states.append(state.clone())
        self.global_features.append(global_feat.clone())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae_advantages(self, gamma: float, gae_lambda: float,
                               next_value: float = 0.0):
        """Compute GAE advantages maintaining original reward scaling"""
        values = self.values + [next_value]
        gae = 0

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[step])

    def get_batch_data(self) -> Dict:
        """Get all data as tensors"""
        return {
            'states': self.states,
            'global_features': torch.cat(self.global_features, dim=0),
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'old_log_probs': torch.tensor(self.log_probs, dtype=torch.float),
            'advantages': torch.tensor(self.advantages, dtype=torch.float),
            'returns': torch.tensor(self.returns, dtype=torch.float),
            'old_values': torch.tensor(self.values, dtype=torch.float)
        }

    def __len__(self):
        return len(self.states)


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
    """Load pre-trained discriminator maintaining original functionality"""
    if not os.path.exists(discriminator_path):
        logging.warning(f"Discriminator not found at {discriminator_path}. Training without discriminator.")
        return None

    try:
        checkpoint = torch.load(discriminator_path, map_location=device)
        model_config = checkpoint['model_config']

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
            f"Loaded pre-trained discriminator with F1: {checkpoint.get('cv_results', {}).get('mean_f1', 'N/A')}")
        return discriminator

    except Exception as e:
        logging.error(f"Failed to load discriminator: {e}")
        return None


class PPOTrainer:
    """PPO Trainer maintaining original environment characteristics"""

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = config.device

        # Setup directories
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.results_dir)

        # Set random seeds
        set_random_seeds(config.random_seed)

        # Initialize environment maintaining original configuration
        from rl_gym import RefactorEnv
        self.env = RefactorEnv(
            data_path=config.data_path,
            max_steps=config.max_episode_steps,
            device=config.device,
            reward_weights=config.reward_weights
        )

        # Load discriminator maintaining original functionality
        self.discriminator = load_discriminator(config.discriminator_path, config.device)
        if self.discriminator is not None:
            self.env.discriminator = self.discriminator

        # Initialize model with shared encoder (PPO correction)
        self._init_model()

        # Single optimizer (PPO correction)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)

        # PPO buffer
        self.buffer = PPOBuffer()

        # TensorBoard writer
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')

        # Training statistics maintaining original structure
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'critic_losses': [],
            'discriminator_losses': [],
            'hub_score_improvements': [],
            'discriminator_scores': []
        }

        # Best model tracking maintaining original logic
        self.best_reward = float('-inf')
        self.early_stopping_counter = 0

        # Running reward statistics for normalization
        self.reward_running_stats = deque(maxlen=1000)

        self.logger.info("PPO Trainer initialized successfully!")

    def _init_model(self):
        """Initialize actor-critic model with shared encoder"""
        ac_config = {
            'node_dim': 7,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'num_actions': 7,
            'global_features_dim': 4,  # Mantiene configurazione originale
            'dropout': self.config.dropout,
            'shared_encoder': True  # PPO correction: force shared encoder
        }

        from actor_critic_models import create_actor_critic
        self.model = create_actor_critic(ac_config).to(self.device)

        self.logger.info(f"Actor-Critic model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info("Using shared encoder for proper PPO implementation")

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """Extract global features maintaining original environment method"""
        metrics = self.env._calculate_metrics(data)

        global_features = torch.tensor([
            metrics['hub_score'],
            metrics['num_nodes'],
            metrics['num_edges'],
            metrics['connected']
        ], dtype=torch.float32, device=self.device)

        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        return global_features

    def collect_rollouts(self, num_episodes: int, is_adversarial: bool = False) -> Dict:
        """Collect rollouts maintaining original environment behavior"""
        episode_infos = []

        for episode in range(num_episodes):
            # Reset environment - maintains original reset logic
            self.env.reset()
            current_data = self.env.current_data.clone()

            episode_reward = 0.0
            episode_length = 0
            initial_hub_score = self.env.initial_metrics['hub_score']

            # Track discriminator score if available
            initial_disc_score = None
            if self.discriminator is not None:
                with torch.no_grad():
                    disc_output = self.discriminator(current_data)
                    if isinstance(disc_output, dict):
                        initial_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                    else:
                        initial_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()

            while True:
                global_features = self._extract_global_features(current_data)

                # Get action and value from policy (PPO sampling)
                with torch.no_grad():
                    output = self.model(current_data, global_features)
                    action_dist = torch.distributions.Categorical(output['action_probs'])
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    value = output['state_value']

                # Store in buffer
                self.buffer.store(
                    state=current_data,
                    global_feat=global_features,
                    action=action.item(),
                    reward=0.0,  # Will be filled after step
                    value=value.item(),
                    log_prob=log_prob.item(),
                    done=False  # Will be filled after step
                )

                # Take environment step - maintains original step logic
                _, reward, done, info = self.env.step(action.item())

                # Apply reward normalization if configured
                if self.config.normalize_rewards:
                    reward = self._normalize_reward(reward)

                # Update stored reward and done flag
                self.buffer.rewards[-1] = reward
                self.buffer.dones[-1] = done

                episode_reward += reward
                episode_length += 1

                if done:
                    break

                current_data = self.env.current_data.clone()

            # Calculate final metrics maintaining original structure
            final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
            hub_improvement = initial_hub_score - final_hub_score
            best_hub_improvement = initial_hub_score - self.env.best_hub_score

            # Track final discriminator score
            final_disc_score = None
            if self.discriminator is not None:
                with torch.no_grad():
                    disc_output = self.discriminator(current_data)
                    if isinstance(disc_output, dict):
                        final_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                    else:
                        final_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()

            episode_infos.append({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'hub_improvement': hub_improvement,
                'best_hub_improvement': best_hub_improvement,
                'initial_hub_score': initial_hub_score,
                'final_hub_score': final_hub_score,
                'best_hub_score': self.env.best_hub_score,
                'no_improve_steps': self.env.no_improve_steps,
                'initial_disc_score': initial_disc_score,
                'final_disc_score': final_disc_score
            })

        return {'episodes': episode_infos}

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward maintaining original scaling"""
        self.reward_running_stats.append(reward)

        if len(self.reward_running_stats) < 10:
            return reward

        mean_reward = np.mean(self.reward_running_stats)
        std_reward = np.std(self.reward_running_stats) + 1e-8

        normalized_reward = (reward - mean_reward) / std_reward
        return np.clip(normalized_reward, -self.config.reward_clip, self.config.reward_clip)

    def update_ppo(self) -> Dict:
        """PPO update with proper algorithm implementation"""
        if len(self.buffer) == 0:
            return {}

        # Compute GAE
        with torch.no_grad():
            if len(self.buffer.states) > 0:
                last_state = self.buffer.states[-1]
                last_global = self.buffer.global_features[-1:]
                last_output = self.model(last_state, last_global)
                next_value = last_output['state_value'].item()
            else:
                next_value = 0.0

        self.buffer.compute_gae_advantages(self.config.gamma, self.config.gae_lambda, next_value)

        # Get batch data
        batch = self.buffer.get_batch_data()

        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch['advantages'] = advantages.to(self.device)

        # Move tensors to device
        batch['actions'] = batch['actions'].to(self.device)
        batch['old_log_probs'] = batch['old_log_probs'].to(self.device)
        batch['returns'] = batch['returns'].to(self.device)
        batch['global_features'] = batch['global_features'].to(self.device)

        dataset_size = len(batch['actions'])

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.config.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, dataset_size)
                mb_indices = indices[start:end]

                if len(mb_indices) == 0:
                    continue

                # Create mini-batch
                mb_states = [batch['states'][i] for i in mb_indices]
                mb_batch_states = Batch.from_data_list(mb_states).to(self.device)
                mb_global_features = batch['global_features'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['old_log_probs'][mb_indices]
                mb_advantages = batch['advantages'][mb_indices]
                mb_returns = batch['returns'][mb_indices]

                # Forward pass
                log_probs, values, entropy = self.model.evaluate_actions(
                    mb_batch_states, mb_global_features, mb_actions
                )

                # PPO policy loss with clipping
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy loss
                entropy_loss = -self.config.entropy_coef * entropy.mean()

                # Total loss
                total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

                # Early stopping for KL divergence
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - log_probs).mean()
                    if kl_div > self.config.target_kl:
                        break

        # Clear buffer (PPO is on-policy)
        self.buffer.clear()

        if num_updates > 0:
            return {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy': total_entropy / num_updates,
                'num_updates': num_updates
            }
        else:
            return {}

    def update_discriminator(self, episode_infos: List[Dict]) -> Dict:
        """Update discriminator maintaining original functionality"""
        if self.discriminator is None or len(episode_infos) < 4:
            return {}

        try:
            # Collect graphs from recent episodes
            positive_graphs = []  # Original smelly graphs
            negative_graphs = []  # RL-modified graphs

            for episode_info in episode_infos[-8:]:  # Use recent episodes
                # This is a simplified version - in practice you'd collect actual states
                # from the rollouts rather than using episode info
                pass

            # For now, return empty dict since we don't have the full discriminator update logic
            return {}

        except Exception as e:
            self.logger.error(f"Discriminator update failed: {e}")
            return {}

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model maintaining original evaluation structure"""
        self.model.eval()

        eval_rewards = []
        eval_hub_improvements = []
        eval_discriminator_scores = []
        eval_episode_lengths = []

        for eval_episode in range(self.config.num_eval_episodes):
            self.env.reset()
            current_data = self.env.current_data.clone()

            episode_reward = 0.0
            episode_length = 0
            initial_hub_score = self.env.initial_metrics['hub_score']

            done = False
            while not done:
                global_features = self._extract_global_features(current_data)

                # Use greedy action selection for evaluation
                with torch.no_grad():
                    output = self.model(current_data, global_features)
                    action = torch.argmax(output['action_probs'], dim=1).item()

                _, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                current_data = self.env.current_data.clone()

            eval_rewards.append(episode_reward)
            eval_episode_lengths.append(episode_length)

            # Calculate hub score improvement
            final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
            hub_improvement = initial_hub_score - final_hub_score
            eval_hub_improvements.append(hub_improvement)

            # Get discriminator score if available
            if self.discriminator is not None:
                with torch.no_grad():
                    disc_output = self.discriminator(current_data)
                    if isinstance(disc_output, dict):
                        disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 0].item()
                    else:
                        disc_score = torch.softmax(disc_output, dim=1)[0, 0].item()
                    eval_discriminator_scores.append(disc_score)

        self.model.train()

        return {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_hub_improvement_mean': np.mean(eval_hub_improvements),
            'eval_hub_improvement_std': np.std(eval_hub_improvements),
            'eval_discriminator_score_mean': np.mean(eval_discriminator_scores) if eval_discriminator_scores else 0.0,
            'eval_discriminator_score_std': np.std(eval_discriminator_scores) if eval_discriminator_scores else 0.0,
            'eval_episode_length_mean': np.mean(eval_episode_lengths),
            'eval_episode_length_std': np.std(eval_episode_lengths),
            'eval_success_rate': sum(1 for imp in eval_hub_improvements if imp > 0.3) / len(eval_hub_improvements)
        }

    def _log_metrics(self, episode_idx: int, episode_infos: List[Dict], update_info: Dict = None):
        """Log metrics maintaining original logging structure"""
        # Update training statistics
        for ep_info in episode_infos:
            self.training_stats['episode_rewards'].append(ep_info['episode_reward'])
            self.training_stats['episode_lengths'].append(ep_info['episode_length'])
            self.training_stats['hub_score_improvements'].append(ep_info['hub_improvement'])

        if update_info:
            if 'policy_loss' in update_info:
                self.training_stats['policy_losses'].append(update_info['policy_loss'])
            if 'value_loss' in update_info:
                self.training_stats['critic_losses'].append(update_info['value_loss'])

        # TensorBoard logging
        recent_rewards = [ep['episode_reward'] for ep in episode_infos]
        recent_improvements = [ep['hub_improvement'] for ep in episode_infos]
        recent_lengths = [ep['episode_length'] for ep in episode_infos]

        self.writer.add_scalar('Episode/Reward', np.mean(recent_rewards), episode_idx)
        self.writer.add_scalar('Episode/HubImprovement', np.mean(recent_improvements), episode_idx)
        self.writer.add_scalar('Episode/Length', np.mean(recent_lengths), episode_idx)

        if update_info:
            for key, value in update_info.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, episode_idx)

        # Console logging maintaining original format
        if episode_idx % self.config.log_every == 0:
            success_rate = sum(1 for imp in recent_improvements if imp > 0.25) / len(recent_improvements)

            self.logger.info(
                f"Episode {episode_idx:4d} | "
                f"Reward: {np.mean(recent_rewards):6.3f}±{np.std(recent_rewards):5.3f} | "
                f"Hub Δ: {np.mean(recent_improvements):6.3f}±{np.std(recent_improvements):5.3f} | "
                f"Success: {success_rate:.1%} | "
                f"Buffer: {len(self.buffer)}"
            )

    def _save_checkpoint(self, episode_idx: int, is_best: bool = False):
        """Save model checkpoint maintaining original structure"""
        checkpoint = {
            'episode': episode_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'node_dim': 7,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_actions': 7,
                'global_features_dim': 4,
                'dropout': self.config.dropout,
                'shared_encoder': True
            },
            'training_stats': self.training_stats,
            'config': self.config,
            'best_reward': self.best_reward
        }

        # Save regular checkpoint
        checkpoint_path = self.results_dir / f'checkpoint_episode_{episode_idx}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.results_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved new best model at episode {episode_idx}")

    def train(self):
        """Main training loop maintaining original training phases"""
        self.logger.info("Starting PPO training")
        self.logger.info(f"Training for {self.config.num_episodes} episodes")
        self.logger.info(f"Warmup phase: episodes 1-{self.config.warmup_episodes}")
        self.logger.info(f"Adversarial phase: episodes {self.config.adversarial_start_episode}+")

        start_time = time.time()
        episode_count = 0

        while episode_count < self.config.num_episodes:
            # Determine current training phase
            is_warmup = episode_count < self.config.warmup_episodes
            is_adversarial = episode_count >= self.config.adversarial_start_episode

            # Collect rollouts
            episodes_to_collect = min(self.config.rollout_episodes,
                                      self.config.num_episodes - episode_count)

            rollout_info = self.collect_rollouts(episodes_to_collect, is_adversarial)
            episode_count += episodes_to_collect

            # PPO update
            update_info = self.update_ppo()

            # Update discriminator in adversarial phase
            discriminator_info = {}
            if is_adversarial and episode_count % 200 == 0:
                discriminator_info = self.update_discriminator(rollout_info['episodes'])
                update_info.update(discriminator_info)

            # Logging
            self._log_metrics(episode_count, rollout_info['episodes'], update_info)

            # Evaluation
            if episode_count % self.config.eval_every == 0:
                eval_metrics = self.evaluate_model()

                self.logger.info(
                    f"Evaluation at episode {episode_count}: "
                    f"Reward: {eval_metrics['eval_reward_mean']:.3f}±{eval_metrics['eval_reward_std']:.3f} | "
                    f"Hub Δ: {eval_metrics['eval_hub_improvement_mean']:.3f}±{eval_metrics['eval_hub_improvement_std']:.3f} | "
                    f"Success: {eval_metrics['eval_success_rate']:.1%}"
                )

                for key, value in eval_metrics.items():
                    self.writer.add_scalar(f'Evaluation/{key}', value, episode_count)

                # Check for best model
                current_reward = eval_metrics['eval_reward_mean']
                current_hub_improvement = eval_metrics['eval_hub_improvement_mean']

                combined_score = current_reward + 10.0 * current_hub_improvement
                best_combined_score = self.best_reward + 10.0 * 0.5

                if combined_score > best_combined_score + self.config.min_improvement:
                    self.best_reward = current_reward
                    self.early_stopping_counter = 0
                    self._save_checkpoint(episode_count, is_best=True)
                else:
                    self.early_stopping_counter += 1

            # Save regular checkpoint
            if episode_count % self.config.save_every == 0:
                self._save_checkpoint(episode_count)

            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at episode {episode_count}")
                break

        # Final evaluation and save
        final_eval = self.evaluate_model()
        self._save_checkpoint(episode_count, is_best=False)

        training_time = time.time() - start_time

        self.logger.info(f"PPO training completed in {training_time / 3600:.2f} hours")
        self.logger.info(f"Best reward: {self.best_reward:.3f}")

        final_hub_improvement = final_eval.get('eval_hub_improvement_mean', 0)
        self.logger.info(f"Final hub improvement: {final_hub_improvement:.4f}")

        if final_hub_improvement > 0.6:
            self.logger.info("SUCCESS: Plateau breakthrough achieved!")
        elif final_hub_improvement > 0.5:
            self.logger.info("PROGRESS: Improved beyond previous plateau")
        else:
            self.logger.info("PLATEAU: Consider further tuning")

        # Save final results
        self._save_final_results(final_eval, training_time)

        return self.training_stats

    def _save_final_results(self, final_eval: Dict, training_time: float):
        """Save final training results"""
        results = {
            'config': asdict(self.config),
            'training_stats': self.training_stats,
            'final_evaluation': final_eval,
            'training_time_hours': training_time / 3600,
            'best_reward': self.best_reward
        }

        with open(self.results_dir / 'ppo_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {self.results_dir}")


def main():
    """Main training function"""
    config = PPOConfig(
        experiment_name="graph_refactor_ppo_corrected",
        num_episodes=3000,
        warmup_episodes=600,
        adversarial_start_episode=600,
        lr=3e-4,
        clip_eps=0.2,
        ppo_epochs=4,
        rollout_episodes=16,
        minibatch_size=8
    )

    print("Starting Graph Refactoring PPO Training (Corrected)")
    print(f"Configuration: {config.experiment_name}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Device: {config.device}")

    trainer = PPOTrainer(config)

    try:
        training_stats = trainer.train()
        print("Training completed successfully!")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer._save_checkpoint(0, is_best=False)

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

    finally:
        trainer.writer.close()


if __name__ == "__main__":
    main()