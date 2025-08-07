#!/usr/bin/env python3
"""
Enhanced GCPN Training Script with Comprehensive Discriminator Monitoring
VERSIONE AVANZATA con controllo completo del discriminatore durante training adversariale

Key Features:
- Comprehensive discriminator performance monitoring
- Adaptive discriminator training frequency based on performance
- Early stopping and rollback mechanisms for discriminator degradation
- Performance trend analysis and automatic adjustments
- Detailed logging of all discriminator interactions
- Safeguards against discriminator collapse
"""
import datetime
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.data.data
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data

# Import modules
from data_loader import create_unified_loader, validate_dataset_consistency, UnifiedDataLoader
from discriminator import HubDetectionDiscriminator
from hyperparameters_configuration import get_improved_config, print_config_summary
from policy_network import HubRefactoringPolicy
from rl_gym import HubRefactoringEnv
from training_logger import setup_optimized_logging, create_progress_tracker, AdvancedDiscriminatorMonitor


@dataclass
class RefactoringAction:
    """Refactoring action with pattern and parameters"""
    source_node: int
    target_node: int
    pattern: int
    terminate: bool
    confidence: float = 0.0

class EnhancedEnvironmentManager:
    """Manages diverse graph pools for better exploration"""

    def __init__(self, data_loader: UnifiedDataLoader, discriminator: nn.Module,
                 env_config, device: torch.device = torch.device('cpu')):
        self.data_loader = data_loader
        self.discriminator = discriminator
        self.pool_size = env_config.graph_pool_size
        self.device = device
        self.pool_refresh_episodes = env_config.pool_refresh_episodes

        # Graph pools
        self.smelly_pool = []
        self.pool_refresh_counter = 0
        self.total_graphs_seen = 0

        self._initialize_graph_pool()

    def _initialize_graph_pool(self):
        """Initialize diverse graph pool"""
        try:
            # Load larger sample for diversity
            smelly_data, labels = self.data_loader.load_dataset(
                max_samples_per_class=self.pool_size * 3,
                shuffle=True,
                validate_all=False,
                remove_metadata=True
            )

            # Filter smelly graphs only
            smelly_candidates = [data for data, label in zip(smelly_data, labels) if label == 1]

            if len(smelly_candidates) < self.pool_size:
                self.pool_size = len(smelly_candidates)

            # Select diverse subset
            self.smelly_pool = self._select_diverse_graphs(smelly_candidates, self.pool_size)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize graph pool: {e}")

    def _select_diverse_graphs(self, candidates: List[Data], target_size: int) -> List[Data]:
        """Select diverse graphs based on size and complexity"""
        if len(candidates) <= target_size:
            return candidates

        env_config = get_improved_config()['environment']

        # Filter by size range
        size_filtered = [
            g for g in candidates
            if env_config.min_graph_size <= g.x.size(0) <= env_config.max_graph_size
        ]

        if not size_filtered:
            size_filtered = candidates

        # Sort by graph size for diversity
        size_sorted = sorted(size_filtered, key=lambda x: x.x.size(0))

        # Select spread across sizes using diversity bins
        selected = []
        if env_config.size_diversity_bins > 0:
            bin_size = len(size_sorted) // env_config.size_diversity_bins
            for i in range(env_config.size_diversity_bins):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, len(size_sorted))
                if start_idx < len(size_sorted):
                    bin_graphs = size_sorted[start_idx:end_idx]
                    if bin_graphs:
                        selected.append(random.choice(bin_graphs))

        # Fill remaining with random selection
        remaining = [g for g in size_filtered if g not in selected]
        while len(selected) < target_size and remaining:
            selected.append(remaining.pop(random.randint(0, len(remaining) - 1)))

        return selected[:target_size]

    def get_random_graph(self) -> Data:
        """Get random graph from pool"""
        if not self.smelly_pool:
            self._refresh_pool()

        graph = random.choice(self.smelly_pool).clone()
        graph = graph.to(self.device)
        self.total_graphs_seen += 1

        return graph

    def _refresh_pool(self):
        """Refresh the graph pool with new samples"""
        self.pool_refresh_counter += 1
        try:
            self._initialize_graph_pool()
        except Exception as e:
            raise RuntimeError(f"Pool refresh failed: {e}")

    def should_refresh_pool(self, episode: int) -> bool:
        """Check if pool should be refreshed"""
        return episode > 0 and episode % self.pool_refresh_episodes == 0

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'pool_size': len(self.smelly_pool),
            'pool_refreshes': self.pool_refresh_counter,
            'total_graphs_seen': self.total_graphs_seen,
            'graph_sizes': [g.x.size(0) for g in self.smelly_pool] if self.smelly_pool else []
        }


class ExperienceBuffer:
    """Buffer for maintaining rollout examples"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self.pointer = 0

    def add_rollout_data(self, states, rewards, hub_scores):
        """Add rollout data to buffer"""
        rollout_data = {
            'states': states,
            'rewards': rewards,
            'hub_scores': hub_scores,
            'episode': len(self.buffer)
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(rollout_data)
        else:
            self.buffer[self.pointer] = rollout_data
            self.pointer = (self.pointer + 1) % self.max_size

    def get_recent_states(self, num_samples: int = 50):
        """Get recent states for discriminator"""
        all_states = []
        for rollout in self.buffer[-10:]:  # Last 10 rollouts
            all_states.extend(rollout['states'])

        if len(all_states) >= num_samples:
            return random.sample(all_states, num_samples)
        return all_states

    def get_buffer_stats(self):
        """Buffer statistics"""
        return {
            'size': len(self.buffer),
            'total_states': sum(len(r['states']) for r in self.buffer),
            'avg_reward': sum(sum(r['rewards']) for r in self.buffer) / max(len(self.buffer), 1),
            'latest_episode': self.buffer[-1]['episode'] if self.buffer else 0
        }

class TensorBoardLogger:
    """Enhanced TensorBoard logging for RL training"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.log_dir / f"rl_training_{timestamp}"

        self.writer = SummaryWriter(log_dir=str(run_dir))

    def log_episode_metrics(self, episode: int, metrics: Dict[str, float]):
        """Log episode-level metrics"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"Episode/{key}", value, episode)

    def log_training_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training-level metrics"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"Training/{key}", value, step)

    def log_environment_metrics(self, episode: int, env_stats: Dict[str, Any]):
        """Log environment diversity metrics"""
        if 'pool_size' in env_stats:
            self.writer.add_scalar("Environment/pool_size", env_stats['pool_size'], episode)
        if 'total_graphs_seen' in env_stats:
            self.writer.add_scalar("Environment/total_graphs_seen", env_stats['total_graphs_seen'], episode)
        if 'graph_sizes' in env_stats and env_stats['graph_sizes']:
            sizes = env_stats['graph_sizes']
            self.writer.add_histogram("Environment/graph_sizes", np.array(sizes), episode)
            self.writer.add_scalar("Environment/avg_graph_size", np.mean(sizes), episode)

    def log_discriminator_metrics(self, episode: int, metrics: Dict[str, float]):
        """Log discriminator performance"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"Discriminator/{key}", value, episode)

    def log_policy_metrics(self, step: int, metrics: Dict[str, float]):
        """Log policy learning metrics"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"Policy/{key}", value, step)

    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


class PPOTrainer:
    """Unified PPO trainer with both standard and structured training capabilities"""

    def __init__(self, policy: nn.Module, discriminator: nn.Module, discriminator_monitor: AdvancedDiscriminatorMonitor,
                 opt_config, disc_config, lr_config, device: torch.device = torch.device('cpu')):
        self.device = device
        self.opt_config = opt_config
        self.disc_config = disc_config
        self.lr_config = lr_config
        self.discriminator_monitor = discriminator_monitor

        # 🔥 FIX: Aggiungi riferimento al logger
        self.logger = discriminator_monitor.logger

        # Move models to device
        self.policy = policy.to(device)
        self.discriminator = discriminator.to(device)

        # Create optimizers with learning rate configuration
        self.opt_policy = optim.AdamW(
            policy.parameters(),
            lr=lr_config.policy_initial_lr if hasattr(lr_config, 'policy_initial_lr') else lr_config.policy_base_lr,
            eps=opt_config.optimizer_eps,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )

        self.opt_disc = optim.AdamW(
            discriminator.parameters(),
            lr=lr_config.discriminator_lr if hasattr(lr_config,
                                                     'discriminator_lr') else lr_config.discriminator_initial_lr,
            eps=opt_config.optimizer_eps,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )

        # PPO parameters
        self.clip_eps = opt_config.clip_epsilon
        self.gamma = 0.99
        self.lam = 0.95
        self.value_coef = opt_config.value_coefficient
        self.entropy_coef = opt_config.entropy_coefficient
        self.max_grad_norm = opt_config.max_grad_norm

        # 🔥 FIX: Definisci policy_base_lr e discriminator_fixed_lr PRIMA di _setup_schedulers()
        # Store base learning rates for structured training
        self.policy_base_lr = self.opt_policy.param_groups[0]['lr']
        self.discriminator_fixed_lr = self.opt_disc.param_groups[0]['lr']

        # Initialize schedulers based on config type
        self._setup_schedulers()

        # Tracking for both modes
        self.update_count = 0
        self.policy_losses = []
        self.disc_losses = []
        self.disc_metrics = []
        self.policy_metrics = []

        # 🆕 STRUCTURED TRAINING TRACKING
        self.policy_update_count = 0
        self.discriminator_update_count = 0
        self.emergency_resets = 0
        self.last_lr_check = None

        # GPU optimization
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def _setup_schedulers(self):
        """Setup schedulers based on configuration type"""
        lr_config = self.lr_config

        # Check if we have structured config (has discriminator_lr) or advanced config (has policy_base_lr)
        if hasattr(lr_config, 'discriminator_lr'):
            # STRUCTURED MODE: Fixed LRs, no complex scheduling
            self.policy_scheduler = None
            self.disc_scheduler = None
            self.structured_mode = True

            # 🔥 FIX: Usa self.logger e le variabili corrette
            self.logger.log_info(f"🏗️  Structured trainer mode:")
            self.logger.log_info(f"   Policy LR: {self.policy_base_lr:.2e} (with decay)")
            self.logger.log_info(f"   Discriminator LR: {self.discriminator_fixed_lr:.2e} (fixed)")

        else:
            # ADVANCED MODE: Complex scheduling
            self.policy_scheduler = optim.lr_scheduler.CyclicLR(
                self.opt_policy,
                base_lr=lr_config.policy_base_lr,
                max_lr=lr_config.policy_max_lr,
                step_size_up=lr_config.policy_cyclical_step_size,
                mode=lr_config.policy_cyclical_mode,
                gamma=lr_config.policy_gamma,
                cycle_momentum=False
            )

            self.disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_disc,
                mode=lr_config.discriminator_monitor_mode,
                factor=lr_config.discriminator_factor,
                patience=lr_config.discriminator_patience,
                threshold=lr_config.discriminator_threshold,
                threshold_mode=lr_config.discriminator_threshold_mode,
                cooldown=lr_config.discriminator_cooldown,
                min_lr=lr_config.discriminator_min_lr,
                eps=lr_config.discriminator_eps
            )
            self.structured_mode = False
            self.disc_metric_history = []

    def update_discriminator_lr(self, new_lr: float):
        """Update discriminator learning rate with emergency tracking"""
        old_lr = self.opt_disc.param_groups[0]['lr']

        for param_group in self.opt_disc.param_groups:
            param_group['lr'] = new_lr

        # Track emergency resets
        if (self.last_lr_check is not None and
                old_lr < 1e-10 and new_lr > 1e-6):
            self.emergency_resets += 1

        self.last_lr_check = new_lr

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        if not rewards:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        advantages = []
        returns = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

            next_value = values[t]

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(self, states: List[Data], actions: List[Any],
                      old_log_probs: torch.Tensor, advantages: torch.Tensor,
                      returns: torch.Tensor):
        """Standard policy update (for backward compatibility)"""
        epochs = self.opt_config.policy_epochs if hasattr(self.opt_config, 'policy_epochs') else 4
        self._update_policy_internal(states, actions, old_log_probs, advantages, returns, epochs)

        # Update scheduler if in advanced mode
        if not self.structured_mode and self.policy_scheduler:
            self.policy_scheduler.step()

    def update_policy_structured(self, states, actions, old_log_probs, advantages, returns, epochs=4):
        """Structured policy update with specified number of epochs"""
        if not self.structured_mode:
            self.logger.log_warning("update_policy_structured called but not in structured mode")
            return self.update_policy(states, actions, old_log_probs, advantages, returns)

        # 🔥 FIX: Usa self.logger
        self.logger.log_info(f"🎯 Policy update #{self.policy_update_count + 1} - {epochs} epochs")

        # Apply learning rate decay for structured mode
        self._apply_structured_lr_decay()

        # Perform the update
        self._update_policy_internal(states, actions, old_log_probs, advantages, returns, epochs)
        self.policy_update_count += 1

    def _apply_structured_lr_decay(self):
        """Apply learning rate decay in structured mode"""
        if not hasattr(self.lr_config, 'policy_decay_frequency'):
            return

        lr_config = self.lr_config
        if (self.policy_update_count > 0 and
                self.policy_update_count % lr_config.policy_decay_frequency == 0):

            new_lr = max(
                self.policy_base_lr * (lr_config.policy_decay_factor ** (
                            self.policy_update_count // lr_config.policy_decay_frequency)),
                lr_config.policy_min_lr
            )

            for param_group in self.opt_policy.param_groups:
                param_group['lr'] = new_lr

            # 🔥 FIX: Usa self.logger
            self.logger.log_info(f"🔧 Policy LR decay: {self.opt_policy.param_groups[0]['lr']:.2e}")

    def _update_policy_internal(self, states, actions, old_log_probs, advantages, returns, epochs):
        """Internal policy update logic shared by both modes"""
        if len(states) == 0:
            return

        policy_losses, entropy_losses, value_losses, kl_divs = [], [], [], []

        for epoch in range(epochs):
            indices = torch.randperm(len(states), device=self.device)
            batch_size = min(32, len(states))

            epoch_losses = []

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end].cpu().numpy()

                batch_states = [states[i] for i in batch_idx]
                batch_actions = [actions[i] for i in batch_idx]
                batch_old_logp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]

                try:
                    out = self.policy(Batch.from_data_list(batch_states).to(self.device))

                    logp, ent = self._compute_logp_and_entropy(out, batch_actions)
                    ratio = torch.exp(logp - batch_old_logp)
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_preds = out['value'].squeeze()
                    value_loss = F.mse_loss(value_preds, batch_ret)

                    total_loss = (
                            policy_loss
                            + self.value_coef * value_loss
                            - self.entropy_coef * ent.mean()
                    )

                    self.opt_policy.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.opt_policy.step()

                    epoch_losses.append(total_loss.item())
                    policy_losses.append(policy_loss.item())
                    entropy_losses.append(ent.mean().item())
                    value_losses.append(value_loss.item())

                    with torch.no_grad():
                        kl_divs.append((batch_old_logp - logp).mean().item())

                except Exception as e:
                    # 🔥 FIX: Usa self.logger
                    self.logger.log_warning(f"Policy batch failed: {e}")
                    continue

            # Log epoch progress in structured mode
            if self.structured_mode and epoch_losses:
                # 🔥 FIX: Usa self.logger
                self.logger.log_info(f"   Epoch {epoch + 1}/{epochs}: Loss = {np.mean(epoch_losses):.4f}")

        # Update metrics
        if policy_losses:
            self.policy_losses.append(np.mean(policy_losses))
            self.policy_metrics.append({
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': np.mean(entropy_losses),
                'kl_divergence': np.mean(kl_divs)
            })

        self.update_count += 1

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def update_discriminator(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                             generated_graphs: List[Data], episode: int):
        """Standard discriminator update (for backward compatibility)"""
        epochs = self.opt_config.discriminator_epochs if hasattr(self.opt_config, 'discriminator_epochs') else 4
        self._update_discriminator_internal(smelly_graphs, clean_graphs, generated_graphs, episode, epochs)

        # Update scheduler if in advanced mode
        if not self.structured_mode and self.disc_scheduler:
            avg_metrics = self.disc_metrics[-1] if self.disc_metrics else {'accuracy': 0}
            monitored_metric = avg_metrics.get(self.lr_config.discriminator_monitor_metric, 0)
            self.disc_scheduler.step(monitored_metric)

    def update_discriminator_structured(self, smelly_graphs, clean_graphs, generated_graphs, episode, epochs=2):
        """Structured discriminator update with specified number of epochs"""
        if not self.structured_mode:
            self.logger.log_warning("update_discriminator_structured called but not in structured mode")
            return self.update_discriminator(smelly_graphs, clean_graphs, generated_graphs, episode)

        # 🔥 FIX: Usa self.logger
        self.logger.log_info(f"🧠 Discriminator update #{self.discriminator_update_count + 1} - {epochs} epochs")
        self.logger.log_info(
            f"   Data: {len(smelly_graphs)} smelly, {len(clean_graphs)} clean, {len(generated_graphs)} generated")

        # Ensure fixed learning rate in structured mode
        for param_group in self.opt_disc.param_groups:
            param_group['lr'] = self.discriminator_fixed_lr

        # Perform the update
        self._update_discriminator_internal(smelly_graphs, clean_graphs, generated_graphs, episode, epochs)
        self.discriminator_update_count += 1

    def _update_discriminator_internal(self, smelly_graphs, clean_graphs, generated_graphs, episode, epochs):
        """Internal discriminator update logic shared by both modes"""

        # Check LR health before starting
        current_lr = self.opt_disc.param_groups[0]['lr']
        if current_lr < 1e-10:
            self.discriminator_monitor.logger.log_warning(
                f"⚠️  Discriminator LR extremely low ({current_lr:.2e}) - update may be ineffective")

        total_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        epoch_losses = []

        for epoch in range(epochs):
            # Determine batch size
            if self.structured_mode:
                batch_size = min(
                    self.disc_config.max_samples_per_class // 2,  # Smaller batches for stability
                    min(len(smelly_graphs), len(clean_graphs), len(generated_graphs))
                )
            else:
                batch_size = min(
                    self.disc_config.max_samples_per_class,
                    max(self.disc_config.min_samples_per_class,
                        min(len(smelly_graphs), len(clean_graphs), len(generated_graphs)))
                )

            if batch_size <= 2:
                self.discriminator_monitor.logger.log_warning(f"⚠️  Insufficient batch size: {batch_size}")
                continue

            try:
                # Sample data
                smelly_sample = np.random.choice(len(smelly_graphs), batch_size, replace=False).tolist() if len(
                    smelly_graphs) >= batch_size else list(range(len(smelly_graphs)))
                clean_sample = np.random.choice(len(clean_graphs), batch_size, replace=False).tolist() if len(
                    clean_graphs) >= batch_size else list(range(len(clean_graphs)))
                generated_sample = np.random.choice(len(generated_graphs), batch_size, replace=False).tolist() if len(
                    generated_graphs) >= batch_size else list(range(len(generated_graphs)))

                smelly_batch_data = [smelly_graphs[i].to(self.device) for i in smelly_sample]
                clean_batch_data = [clean_graphs[i].to(self.device) for i in clean_sample]
                generated_batch_data = [generated_graphs[i].to(self.device) for i in generated_sample]

                smelly_batch = Batch.from_data_list(smelly_batch_data)
                clean_batch = Batch.from_data_list(clean_batch_data)
                generated_batch = Batch.from_data_list(generated_batch_data)

                # Forward pass
                self.discriminator.train()
                smelly_out = self.discriminator(smelly_batch)
                clean_out = self.discriminator(clean_batch)
                gen_out = self.discriminator(generated_batch)

                if any(out is None or 'logits' not in out for out in [smelly_out, clean_out, gen_out]):
                    self.discriminator_monitor.logger.log_warning(f"⚠️  Invalid discriminator outputs at epoch {epoch}")
                    continue

                # Labels and loss
                smelly_labels = torch.ones(len(smelly_batch_data), dtype=torch.long, device=self.device)
                clean_labels = torch.zeros(len(clean_batch_data), dtype=torch.long, device=self.device)
                gen_labels = torch.zeros(len(generated_batch_data), dtype=torch.long, device=self.device)

                loss_smelly = F.cross_entropy(smelly_out['logits'], smelly_labels,
                                              label_smoothing=self.disc_config.label_smoothing)
                loss_clean = F.cross_entropy(clean_out['logits'], clean_labels,
                                             label_smoothing=self.disc_config.label_smoothing)
                loss_gen = F.cross_entropy(gen_out['logits'], gen_labels,
                                           label_smoothing=self.disc_config.label_smoothing)

                # Gradient penalty (only in advanced mode)
                if not self.structured_mode:
                    gp_loss = self._compute_gradient_penalty(smelly_batch, clean_batch, generated_batch, smelly_out,
                                                             clean_out, gen_out)
                else:
                    gp_loss = 0.0

                disc_loss = loss_smelly + loss_clean + loss_gen + self.disc_config.gradient_penalty_lambda * gp_loss

                # Backward pass
                self.opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.opt_disc.step()

                epoch_losses.append(disc_loss.item())

                # Calculate metrics
                if self.disc_config.calculate_detailed_metrics:
                    with torch.no_grad():
                        self.discriminator.eval()

                        smelly_preds = smelly_out['logits'].argmax(dim=1)
                        clean_preds = clean_out['logits'].argmax(dim=1)
                        gen_preds = gen_out['logits'].argmax(dim=1)

                        all_preds = torch.cat([smelly_preds, clean_preds, gen_preds])
                        all_labels = torch.cat([smelly_labels, clean_labels, gen_labels])

                        correct = (all_preds == all_labels).float()
                        accuracy = correct.mean().item()

                        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
                        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
                        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)

                        total_metrics['accuracy'].append(accuracy)
                        total_metrics['precision'].append(precision.item())
                        total_metrics['recall'].append(recall.item())
                        total_metrics['f1'].append(f1.item())

                # Log epoch progress in structured mode
                if self.structured_mode:
                    # 🔥 FIX: Usa self.logger
                    self.logger.log_info(f"   Epoch {epoch + 1}/{epochs}: Loss = {disc_loss.item():.4f}")

            except Exception as e:
                self.discriminator_monitor.logger.log_warning(f"⚠️  Discriminator training error at epoch {epoch}: {e}")
                continue

        # Update metrics
        if total_metrics['accuracy']:
            avg_metrics = {
                'accuracy': np.mean(total_metrics['accuracy']),
                'precision': np.mean(total_metrics['precision']),
                'recall': np.mean(total_metrics['recall']),
                'f1': np.mean(total_metrics['f1']),
                'loss': np.mean(epoch_losses) if epoch_losses else 0.0
            }

            self.disc_metrics.append(avg_metrics)
            self.disc_losses.append(avg_metrics['loss'])

            current_lr = self.opt_disc.param_groups[0]['lr']
            if self.structured_mode:
                # 🔥 FIX: Usa self.logger
                self.logger.log_info(f"🧠 Discriminator update completed:")
                self.logger.log_info(f"   Acc: {avg_metrics['accuracy']:.3f}, F1: {avg_metrics['f1']:.3f}")
                self.logger.log_info(f"   Loss: {avg_metrics['loss']:.4f}, LR: {current_lr:.2e}")
            else:
                self.discriminator_monitor.logger.log_info(
                    f"🧠 Discriminator training completed - "
                    f"Acc: {avg_metrics['accuracy']:.3f}, F1: {avg_metrics['f1']:.3f}, "
                    f"Loss: {avg_metrics['loss']:.4f}, LR: {current_lr:.2e}"
                )

        # Check for LR degradation
        post_update_lr = self.opt_disc.param_groups[0]['lr']
        if post_update_lr < current_lr * 0.1:
            self.discriminator_monitor.logger.log_warning(
                f"⚠️  Discriminator LR dropped significantly: {current_lr:.2e} -> {post_update_lr:.2e}")

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _compute_gradient_penalty(self, smelly_batch, clean_batch, generated_batch, smelly_out, clean_out, gen_out):
        """Compute gradient penalty (only for advanced mode)"""
        try:
            gp_loss = 0.0
            for batch_data, out in [(smelly_batch, smelly_out), (clean_batch, clean_out), (generated_batch, gen_out)]:
                if batch_data.x.grad is not None:
                    batch_data.x.grad.zero_()

                grads = torch.autograd.grad(
                    outputs=out['logits'].sum(),
                    inputs=batch_data.x,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                gp_loss += (grads.norm(2, dim=1) - 1).pow(2).mean()
            return gp_loss
        except Exception:
            return 0.0

    def _compute_logp_and_entropy(self, policy_out: Dict, actions: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy for actions"""
        log_probs = []
        entropies = []

        num_nodes = policy_out['hub_probs'].size(0)

        for i, action in enumerate(actions):
            try:
                # Hub selection
                if i < policy_out['hub_probs'].size(0):
                    hub_probs = policy_out['hub_probs'][i:i + 1] + 1e-8
                    hub_probs = hub_probs / hub_probs.sum()
                    hub_dist = Categorical(hub_probs)

                    hub_action = min(action.source_node, hub_probs.size(0) - 1)
                    log_prob_hub = hub_dist.log_prob(torch.tensor(hub_action, device=self.device))
                    entropy_hub = hub_dist.entropy()
                else:
                    log_prob_hub = torch.tensor(-5.0, device=self.device)
                    entropy_hub = torch.tensor(0.1, device=self.device)

                # Pattern selection
                if i < policy_out['pattern_probs'].size(0):
                    pattern_probs = policy_out['pattern_probs'][i] + 1e-8
                    pattern_probs = pattern_probs / pattern_probs.sum()
                    pattern_dist = Categorical(pattern_probs)

                    pattern_action = min(action.pattern, pattern_probs.size(0) - 1)
                    log_prob_pattern = pattern_dist.log_prob(torch.tensor(pattern_action, device=self.device))
                    entropy_pattern = pattern_dist.entropy()
                else:
                    log_prob_pattern = torch.tensor(-2.0, device=self.device)
                    entropy_pattern = torch.tensor(0.1, device=self.device)

                # Termination
                if i < policy_out['term_probs'].size(0):
                    term_probs = policy_out['term_probs'][i] + 1e-8
                    term_probs = term_probs / term_probs.sum()
                    term_dist = Categorical(term_probs)

                    term_action = int(action.terminate)
                    log_prob_term = term_dist.log_prob(torch.tensor(term_action, device=self.device))
                    entropy_term = term_dist.entropy()
                else:
                    log_prob_term = torch.tensor(-1.0, device=self.device)
                    entropy_term = torch.tensor(0.1, device=self.device)

                total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
                total_entropy = entropy_hub + entropy_pattern + entropy_term

                log_probs.append(total_log_prob)
                entropies.append(total_entropy)

            except Exception as e:
                log_probs.append(torch.tensor(-5.0, device=self.device))
                entropies.append(torch.tensor(0.1, device=self.device))

        return torch.stack(log_probs), torch.stack(entropies)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest training metrics for logging"""
        metrics = {}

        if self.policy_metrics:
            latest_policy = self.policy_metrics[-1]
            metrics.update({
                'policy_loss': latest_policy['policy_loss'],
                'policy_entropy': latest_policy['entropy'],
                'policy_kl': latest_policy['kl_divergence']
            })

        if self.disc_metrics:
            latest_disc = self.disc_metrics[-1]
            metrics.update({
                'discriminator_accuracy': latest_disc['accuracy'],
                'discriminator_f1': latest_disc['f1'],
                'discriminator_precision': latest_disc['precision'],
                'discriminator_recall': latest_disc['recall'],
                'discriminator_loss': latest_disc.get('loss', 0.0)
            })

        return metrics

    def get_training_mode_info(self) -> Dict[str, Any]:
        """Get information about current training mode"""
        return {
            'structured_mode': self.structured_mode,
            'policy_updates_completed': getattr(self, 'policy_update_count', self.update_count),
            'discriminator_updates_completed': getattr(self, 'discriminator_update_count', 0),
            'emergency_resets': self.emergency_resets,
            'current_policy_lr': self.opt_policy.param_groups[0]['lr'],
            'current_discriminator_lr': self.opt_disc.param_groups[0]['lr']
        }


def safe_torch_load(file_path: Path, device: torch.device):
    """Safely load PyTorch files with PyTorch Geometric compatibility"""
    try:
        with torch.serialization.safe_globals([torch_geometric.data.data.DataEdgeAttr]):
            return torch.load(file_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(file_path, map_location=device, weights_only=False)


def load_pretrained_discriminator(model_path: Path, device: torch.device, logger) -> Tuple[nn.Module, Dict]:
    """Load pretrained discriminator with unified format validation"""
    logger.log_info(f"Loading pretrained discriminator from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained discriminator not found at {model_path}")

    checkpoint = safe_torch_load(model_path, device)

    # Validate unified data pipeline information
    if 'unified_data_pipeline' in checkpoint:
        pipeline_info = checkpoint['unified_data_pipeline']
        if not pipeline_info.get('consistent_with_rl_training', False):
            logger.log_warning("Discriminator may not be fully compatible with RL training")
    else:
        logger.log_warning("Model lacks unified data pipeline information - compatibility uncertain")

    # Get architecture parameters from checkpoint
    model_architecture = checkpoint.get('model_architecture', {})
    node_dim = checkpoint.get('node_dim', 7)
    edge_dim = checkpoint.get('edge_dim', 1)

    # Validate unified format expectations
    if node_dim != 7:
        raise ValueError(f"Discriminator has incompatible feature dimensions: {node_dim} != 7")

    # Use saved architecture parameters or defaults
    hidden_dim = model_architecture.get('hidden_dim', 128)
    num_layers = model_architecture.get('num_layers', 4)
    dropout = model_architecture.get('dropout', 0.15)
    heads = model_architecture.get('heads', 8)

    # Create model with correct architecture
    discriminator = HubDetectionDiscriminator(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads
    ).to(device)

    # Load state dict
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    discriminator.eval()

    # Log performance info
    cv_results = checkpoint.get('cv_results', {})
    if cv_results:
        mean_f1 = cv_results.get('mean_f1', 0)
        mean_acc = cv_results.get('mean_accuracy', 0)
        logger.log_info(f"Loaded discriminator with CV F1: {mean_f1:.4f}, Accuracy: {mean_acc:.4f}")

    return discriminator, checkpoint


def create_diverse_training_environments(env_manager: EnhancedEnvironmentManager,
                                         discriminator: nn.Module, training_config,
                                         device: torch.device = torch.device('cpu')) -> List[HubRefactoringEnv]:
    """Create training environments with diverse graphs using centralized config"""
    num_envs = training_config.num_envs
    max_steps = training_config.max_steps_per_episode

    envs = []
    for i in range(num_envs):
        try:
            # Get random graph from diverse pool
            initial_graph = env_manager.get_random_graph()

            # Create environment with config-based max_steps
            env = HubRefactoringEnv(initial_graph, discriminator, max_steps=max_steps, device=device)
            # Store reference to environment manager for dynamic resets
            env.env_manager = env_manager
            envs.append(env)

        except Exception:
            continue

    return envs


def enhanced_reset_environment(env: HubRefactoringEnv) -> Optional[Data]:
    """Enhanced reset CORRETTO con gestione errori robusta"""
    try:
        if hasattr(env, 'env_manager'):
            # Get new diverse graph
            new_graph = env.env_manager.get_random_graph()

            # Update environment with new graph
            env.initial_data = new_graph.to(env.device)
            env.current_data = None
            env.steps = 0
            env.action_history.clear()
            env._cached_hub_score = None
            env._graph_hash = None

            # Reset with new graph
            reset_state = env.reset()

            # ✅ VALIDAZIONE: verifica che il reset sia valido
            if reset_state is not None and hasattr(reset_state, 'x') and reset_state.x.size(0) > 0:
                return reset_state
            else:
                raise ValueError("Reset produced invalid state")
        else:
            # Fallback to original reset
            reset_state = env.reset()
            if reset_state is not None and hasattr(reset_state, 'x') and reset_state.x.size(0) > 0:
                return reset_state
            else:
                raise ValueError("Fallback reset produced invalid state")

    except Exception as e:
        if hasattr(env, 'logger'):
            env.logger.log_warning(f"Environment reset failed: {e}")

        try:
            return env.reset()
        except:
            return None

def collect_rollouts(envs: List[HubRefactoringEnv], policy: nn.Module,
                     training_config, device: torch.device = torch.device('cpu')) -> Dict[str, List]:
    """Collect rollouts with proper metric transport"""

    steps_per_env = training_config.steps_per_env

    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    # Add containers for step-level metrics
    all_step_infos = []
    all_hub_improvements = []
    all_hub_scores = []
    episode_step_metrics = {
        'total_steps': 0,
        'successful_steps': 0,
        'valid_metrics_steps': 0,
        'total_hub_improvement': 0.0,
        'hub_score_changes': [],
        'step_errors': [],
        'fallback_scores_count': 0
    }

    # Reset environments
    states = []
    valid_envs = []

    for i, env in enumerate(envs):
        try:
            state = env.reset()
            if state.x.device != device:
                state = state.to(device)

            # All states should already have 7 features from unified pipeline
            if state.x.size(1) != 7:
                continue

            states.append(state)
            valid_envs.append(env)
        except Exception:
            continue

    if not states:
        return {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': torch.tensor([], device=device), 'values': [],
            'step_infos': [],
            'episode_metrics': episode_step_metrics
        }

    for step in range(steps_per_env):
        # Ensure all states are on correct device
        states_on_device = []
        for state in states:
            if state.x.device != device:
                state = state.to(device)
            if not hasattr(state, 'batch'):
                state.batch = torch.zeros(state.x.size(0), dtype=torch.long, device=device)
            states_on_device.append(state)

        # Batch states
        try:
            batch_data = Batch.from_data_list(states_on_device)
            if batch_data.x.device != device:
                batch_data = batch_data.to(device)
        except Exception:
            break

        # Policy forward pass
        with torch.no_grad():
            try:
                policy_out = policy(batch_data)

                # FIX: Gestione corretta del value network
                if 'value' in policy_out:
                    values = policy_out['value'].squeeze(-1)
                else:
                    # Fallback usando discriminator
                    values = []
                    for state in states_on_device:
                        pred = None
                        for env in valid_envs:
                            try:
                                with torch.no_grad():
                                    output = env.discriminator(state)
                                    if output and 'logits' in output:
                                        probs = F.softmax(output['logits'], dim=1)
                                        pred = 1.0 - probs[0, 1].item()  # Inverted for value
                                        break
                            except:
                                continue

                        value = pred if pred is not None else 0.5
                        values.append(value)
                    values = torch.tensor(values, device=device)
            except Exception:
                values = torch.zeros(len(states_on_device), device=device)

        # Sample actions
        actions = []
        log_probs = []
        num_graphs = len(states_on_device)
        nodes_per_graph = [state.x.size(0) for state in states_on_device]
        cumsum_nodes = [0] + list(np.cumsum(nodes_per_graph))

        for i in range(num_graphs):
            try:
                start_idx = cumsum_nodes[i]
                end_idx = cumsum_nodes[i + 1]
                num_nodes = nodes_per_graph[i]

                if start_idx >= policy_out['hub_probs'].size(0) or end_idx > policy_out['hub_probs'].size(0):
                    action = RefactoringAction(0, min(1, num_nodes - 1), 0, False)
                    actions.append(action)
                    log_probs.append(torch.tensor(0.0, device=device))
                    continue

                # Sample hub
                hub_probs_i = policy_out['hub_probs'][start_idx:end_idx]
                if hub_probs_i.size(0) == 0:
                    action = RefactoringAction(0, min(1, num_nodes - 1), 0, False)
                    actions.append(action)
                    log_probs.append(torch.tensor(0.0, device=device))
                    continue

                hub_probs_i = hub_probs_i + 1e-8
                hub_dist = Categorical(hub_probs_i)
                hub_local = hub_dist.sample().item()
                hub_local = min(hub_local, num_nodes - 1)

                # Sample pattern
                pattern_idx = start_idx + hub_local
                if pattern_idx < policy_out['pattern_probs'].size(0):
                    pattern_probs_i = policy_out['pattern_probs'][pattern_idx] + 1e-8
                    pattern_dist = Categorical(pattern_probs_i)
                    pattern = pattern_dist.sample().item()
                else:
                    pattern = 0

                # Sample termination
                if i < policy_out['term_probs'].size(0):
                    term_probs_i = policy_out['term_probs'][i] + 1e-8
                else:
                    term_probs_i = torch.tensor([0.7, 0.3], device=device)

                term_dist = Categorical(term_probs_i)
                terminate = term_dist.sample().item() == 1

                # Target selection
                if num_nodes > 1:
                    target_candidates = [j for j in range(num_nodes) if j != hub_local]
                    target_local = random.choice(target_candidates) if target_candidates else hub_local
                else:
                    target_local = hub_local

                action = RefactoringAction(hub_local, target_local, pattern, terminate)
                actions.append(action)

                # Compute log probability
                log_prob = (hub_dist.log_prob(torch.tensor(hub_local, device=device)) +
                            pattern_dist.log_prob(torch.tensor(pattern, device=device)) +
                            term_dist.log_prob(torch.tensor(int(terminate), device=device)))
                log_probs.append(log_prob)

            except Exception:
                action = RefactoringAction(0, 0, 0, False)
                actions.append(action)
                log_probs.append(torch.tensor(0.0, device=device))

        # Execute actions and collect all step metrics
        next_states = []
        rewards = []
        dones = []
        step_infos_batch = []

        for i, (env, action) in enumerate(zip(valid_envs, actions)):
            try:
                state, reward, done, info = env.step(
                    (action.source_node, action.target_node, action.pattern, action.terminate)
                )

                env.current_data = state

                if state.x.device != device:
                    state = state.to(device)
                if not hasattr(state, 'batch'):
                    state.batch = torch.zeros(state.x.size(0), dtype=torch.long, device=device)

                next_states.append(state)
                rewards.append(reward)
                dones.append(done)

                # Store step info for metric transport
                step_infos_batch.append(info)

                # Update episode metrics from step info
                episode_step_metrics['total_steps'] += 1
                if info.get('step_metrics_calculated', False):
                    episode_step_metrics['valid_metrics_steps'] += 1

                if info.get('success', False):
                    episode_step_metrics['successful_steps'] += 1

                if 'hub_improvement' in info:
                    hub_improvement = info['hub_improvement']
                    episode_step_metrics['total_hub_improvement'] += hub_improvement
                    all_hub_improvements.append(hub_improvement)

                if 'new_hub_score' in info:
                    all_hub_scores.append(info['new_hub_score'])
                    episode_step_metrics['hub_score_changes'].append({
                        'env_id': i,
                        'step': step,
                        'old_score': info.get('old_hub_score', 0.5),
                        'new_score': info['new_hub_score'],
                        'improvement': info.get('hub_improvement', 0.0)
                    })

                if info.get('hub_score_is_fallback', False):
                    episode_step_metrics['fallback_scores_count'] += 1

                if 'step_error' in info and info['step_error']:
                    episode_step_metrics['step_errors'].append({
                        'env_id': i,
                        'step': step,
                        'error': info['step_error']
                    })

                # Store experience with step info
                all_states.append(states_on_device[i])
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(done)
                all_log_probs.append(log_probs[i] if i < len(log_probs) else torch.tensor(0.0, device=device))
                all_values.append(values[i].item() if i < len(values) else 0.0)
                all_step_infos.append(info)

                # Reset if done
                if done:
                    try:
                        reset_state = enhanced_reset_environment(env)
                        if reset_state.x.device != device:
                            reset_state = reset_state.to(device)
                        if not hasattr(reset_state, 'batch'):
                            reset_state.batch = torch.zeros(reset_state.x.size(0), dtype=torch.long, device=device)
                        next_states[i] = reset_state
                    except Exception:
                        pass

            except Exception as e:
                # Create fallback info even for errors
                error_info = {
                    'valid': False,
                    'success': False,
                    'hub_improvement': 0.0,
                    'old_hub_score': 0.5,
                    'new_hub_score': 0.5,
                    'step_error': str(e),
                    'step_metrics_calculated': False
                }
                step_infos_batch.append(error_info)
                episode_step_metrics['step_errors'].append({
                    'env_id': i,
                    'step': step,
                    'error': str(e)
                })

                if i < len(states_on_device):
                    next_states.append(states_on_device[i])
                else:
                    fallback_state = states_on_device[0].clone() if states_on_device else None
                    if fallback_state is not None:
                        next_states.append(fallback_state)
                    else:
                        break

                rewards.append(-0.5)
                dones.append(False)

                # Store error experience
                if i < len(states_on_device):
                    all_states.append(states_on_device[i])
                    all_actions.append(action)
                    all_rewards.append(-0.5)
                    all_dones.append(False)
                    all_log_probs.append(log_probs[i] if i < len(log_probs) else torch.tensor(0.0, device=device))
                    all_values.append(0.0)
                    all_step_infos.append(error_info)

        states = next_states

        # Break if all environments are done or we have no valid states
        if not states or all(dones):
            break

        # GPU memory management
        if device.type == 'cuda' and step % 10 == 0:
            torch.cuda.empty_cache()

    if not all_log_probs:
        return {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': torch.tensor([], device=device), 'values': [],
            'step_infos': [],
            'episode_metrics': episode_step_metrics
        }

    # Return with step-level metrics
    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'dones': all_dones,
        'log_probs': torch.stack(all_log_probs),
        'values': all_values,
        'step_infos': all_step_infos,
        'hub_improvements': all_hub_improvements,
        'hub_scores': all_hub_scores,
        'episode_metrics': episode_step_metrics
    }


def setup_gpu_environment(logger):
    """Setup optimal GPU environment"""
    if torch.cuda.is_available():
        # Set memory management
        torch.cuda.empty_cache()

        # Get GPU info
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        logger.log_info(f"GPU Setup: {gpu_count} GPUs, Current: {gpu_name}, Memory: {gpu_memory:.1f} GB")

        # Optimize for memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device = torch.device('cuda')
    else:
        logger.log_warning("CUDA not available, using CPU")
        device = torch.device('cpu')

    return device


def monitor_gpu_usage(logger):
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9

        if allocated > 8.0:  # If using more than 8GB
            torch.cuda.empty_cache()
            logger.log_info("GPU memory cleanup performed")

def get_training_schedule_info(episode: int) -> Dict[str, Any]:
    """Get training schedule information for current episode"""
    config = get_improved_config()
    training_config = config['training']

    warmup = training_config.warmup_episodes
    policy_freq = training_config.policy_update_frequency
    disc_freq = training_config.discriminator_update_frequency

    if episode < warmup:
        phase = "warmup"
        next_policy_update = warmup + policy_freq
        next_disc_update = warmup + (policy_freq * disc_freq)
    else:
        episodes_since_warmup = episode - warmup
        policy_updates_so_far = episodes_since_warmup // policy_freq

        phase = "training"
        next_policy_update = warmup + ((policy_updates_so_far + 1) * policy_freq)
        next_disc_update = warmup + (((policy_updates_so_far // disc_freq) + 1) * policy_freq * disc_freq)

    return {
        'phase': phase,
        'policy_updates_completed': max(0, (episode - warmup) // policy_freq) if episode >= warmup else 0,
        'discriminator_updates_completed': max(0, (episode - warmup) // (policy_freq * disc_freq)) if episode >= warmup else 0,
        'next_policy_update_episode': next_policy_update,
        'next_discriminator_update_episode': next_disc_update,
        'episodes_until_next_policy': max(0, next_policy_update - episode),
        'episodes_until_next_discriminator': max(0, next_disc_update - episode)
    }


def main():
    """Main training loop with STRUCTURED PHASES"""

    # Setup logging and configuration
    logger = setup_optimized_logging()
    config = get_improved_config()

    # Load all configurations
    reward_config = config['rewards']
    training_config = config['training']
    opt_config = config['optimization']
    disc_config = config['discriminator']
    env_config = config['environment']
    lr_config = config['learning_rate']
    adv_config = config['adversarial']

    logger.start_training(config)
    print_config_summary()

    # Setup device and seeds
    device = setup_gpu_environment(logger)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Paths and setup
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'dataset_builder' / 'data' / 'dataset_graph_feature'
    results_dir = base_dir / 'results'
    discriminator_path = results_dir / 'discriminator_pretraining' / 'pretrained_discriminator.pt'

    if not discriminator_path.exists():
        logger.log_error(f"Pretrained discriminator not found at {discriminator_path}")
        return

    # Create data loader and load discriminator
    try:
        data_loader = create_unified_loader(data_dir, device)
        discriminator, discriminator_checkpoint = load_pretrained_discriminator(discriminator_path, device, logger)

        # Initialize discriminator monitor
        discriminator_monitor = AdvancedDiscriminatorMonitor(
            discriminator=discriminator,
            data_loader=data_loader,
            device=device,
            logger=logger
        )

        # Set baseline performance
        baseline_metrics = {
            'accuracy': discriminator_checkpoint.get('cv_results', {}).get('mean_accuracy', 0.0),
            'f1': discriminator_checkpoint.get('cv_results', {}).get('mean_f1', 0.0),
            'precision': discriminator_checkpoint.get('cv_results', {}).get('mean_precision', 0.0),
            'recall': discriminator_checkpoint.get('cv_results', {}).get('mean_recall', 0.0)
        }
        discriminator_monitor.set_baseline_performance(baseline_metrics)

    except Exception as e:
        logger.log_error(f"Failed to setup discriminator: {e}")
        return

    # Create policy and trainer
    hidden_dim = discriminator_checkpoint.get('model_architecture', {}).get('hidden_dim', 128)
    policy = HubRefactoringPolicy(
        node_dim=7, edge_dim=1, hidden_dim=hidden_dim,
        hub_bias_strength=4.0, exploration_rate=0.1
    ).to(device)

    # 🆕 STRUCTURED PPO Trainer with separate learning rates
    trainer = PPOTrainer(
        policy, discriminator, discriminator_monitor,
        opt_config, disc_config, lr_config, device=device
    )

    # Environment setup
    env_manager = EnhancedEnvironmentManager(data_loader, discriminator, env_config, device)
    envs = create_diverse_training_environments(env_manager, discriminator, training_config, device)

    # Update environment rewards
    for env in envs:
        env.REWARD_SUCCESS = reward_config.REWARD_SUCCESS
        env.REWARD_PARTIAL_SUCCESS = reward_config.REWARD_PARTIAL_SUCCESS
        # ... [other reward updates] ...

    # 🆕 STRUCTURED TRAINING COMPONENTS
    experience_buffer = ExperienceBuffer(adv_config.experience_buffer_size)
    metrics_history = create_progress_tracker()

    # Phase tracking
    warmup_episodes = training_config.warmup_episodes
    policy_update_frequency = training_config.policy_update_frequency
    discriminator_update_frequency = training_config.discriminator_update_frequency

    policy_updates_completed = 0
    discriminator_updates_completed = 0
    discriminator_paused = False
    discriminator_pause_remaining = 0

    # TensorBoard setup
    tensorboard_dir = results_dir / 'tensorboard_logs'
    tb_logger = TensorBoardLogger(tensorboard_dir)

    logger.log_info("🚀 STARTING STRUCTURED ADVERSARIAL TRAINING")
    logger.log_info(f"📊 Phase Schedule:")
    logger.log_info(f"   Warm-up: Episodes 0-{warmup_episodes}")
    logger.log_info(f"   Policy updates: Every {policy_update_frequency} episodes")
    logger.log_info(
        f"   Discriminator updates: Every {policy_update_frequency * discriminator_update_frequency} episodes")

    # MAIN TRAINING LOOP with STRUCTURED PHASES
    for episode in range(training_config.num_episodes):
        episode_start_time = time.time()

        # Get current training schedule
        schedule_info = get_training_schedule_info(episode)
        current_phase = schedule_info['phase']

        # 🌅 WARM-UP PHASE: Only rollout collection
        if current_phase == "warmup":
            if episode % 25 == 0:
                logger.log_info(f"🌅 WARM-UP Phase - Episode {episode}/{warmup_episodes}")
                logger.log_info(f"   Collecting experience only - no updates")

        # Update discriminator pause countdown
        if discriminator_paused:
            discriminator_pause_remaining -= 1
            if discriminator_pause_remaining <= 0:
                discriminator_paused = False
                logger.log_info("▶️  Discriminator pause ended - resuming updates")

        # Environment refresh
        if episode % training_config.environment_refresh_frequency == 0 and episode > 0:
            for env in envs:
                try:
                    enhanced_reset_environment(env)
                except Exception:
                    pass
            logger.log_environment_refresh(episode, env_manager.get_stats())

        # COLLECT ROLLOUTS (always done)
        rollout_data = collect_rollouts(envs, policy, training_config, device)

        if len(rollout_data['rewards']) == 0:
            continue

        # Add to experience buffer (always done)
        experience_buffer.add_rollout_data(
            rollout_data['states'],
            rollout_data['rewards'],
            rollout_data.get('hub_scores', [])
        )

        # Calculate episode metrics
        episode_reward = np.mean(rollout_data['rewards'])
        rollout_hub_improvements = rollout_data.get('hub_improvements', [])

        if rollout_hub_improvements:
            successful_improvements = [imp for imp in rollout_hub_improvements if
                                       imp > training_config.hub_improvement_threshold]
            success_rate = len(successful_improvements) / len(rollout_hub_improvements)
            avg_improvement = np.mean(rollout_hub_improvements)
        else:
            success_rate = 0.0
            avg_improvement = 0.0

        # Update metrics history
        metrics_history['episode_rewards'].append(episode_reward)
        metrics_history['success_rates'].append(success_rate)
        metrics_history['hub_improvements'].append(avg_improvement)

        # 🎯 POLICY UPDATE PHASE
        is_policy_update_episode = (
                episode >= warmup_episodes and
                (episode - warmup_episodes) % policy_update_frequency == 0 and
                len(rollout_data['states']) >= 16
        )

        if is_policy_update_episode:
            logger.log_info(f"🎯 POLICY UPDATE #{policy_updates_completed + 1} - Episode {episode}")

            try:
                # Compute GAE
                advantages, returns = trainer.compute_gae(
                    rollout_data['rewards'],
                    rollout_data['values'],
                    rollout_data['dones']
                )

                if advantages.numel() > 0 and returns.numel() > 0:
                    # STRUCTURED Policy Update with specified epochs
                    trainer.update_policy_structured(
                        rollout_data['states'],
                        rollout_data['actions'],
                        rollout_data['log_probs'],
                        advantages,
                        returns,
                        epochs=training_config.policy_epochs_per_update
                    )

                    policy_updates_completed += 1
                    training_metrics = trainer.get_latest_metrics()
                    logger.log_training_update(policy_updates_completed, training_metrics)

                    # TensorBoard logging
                    if training_metrics:
                        for key, value in training_metrics.items():
                            if 'policy' in key:
                                tb_logger.log_policy_metrics(policy_updates_completed,
                                                             {key.replace('policy_', ''): value})

                else:
                    logger.log_warning("⚠️  Skipping policy update - insufficient valid data")

            except Exception as e:
                logger.log_warning(f"Policy update failed: {e}")

        # 🧠 DISCRIMINATOR UPDATE PHASE
        is_discriminator_update_episode = (
                episode >= warmup_episodes and
                policy_updates_completed > 0 and
                policy_updates_completed % discriminator_update_frequency == 0 and
                not discriminator_paused and
                (episode - warmup_episodes) % (policy_update_frequency * discriminator_update_frequency) == 0
        )

        if is_discriminator_update_episode:
            logger.log_info(f"🧠 DISCRIMINATOR UPDATE #{discriminator_updates_completed + 1} - Episode {episode}")
            logger.log_info(f"   After {policy_updates_completed} policy updates")

            try:
                # Load real data
                smelly_samples, labels = data_loader.load_dataset(
                    max_samples_per_class=disc_config.max_samples_per_class,
                    shuffle=True,
                    remove_metadata=True
                )

                smelly_graphs = [data for data, label in zip(smelly_samples, labels) if label == 1]
                clean_graphs = [data for data, label in zip(smelly_samples, labels) if label == 0]

                # Get generated data from buffer
                generated_states = experience_buffer.get_recent_states(disc_config.max_samples_per_class)

                if (len(smelly_graphs) >= disc_config.min_samples_per_class and
                        len(clean_graphs) >= disc_config.min_samples_per_class and
                        len(generated_states) >= disc_config.min_samples_per_class):

                    # STRUCTURED Discriminator Update with specified epochs
                    trainer.update_discriminator_structured(
                        smelly_graphs[:disc_config.max_samples_per_class],
                        clean_graphs[:disc_config.max_samples_per_class],
                        generated_states[:disc_config.max_samples_per_class],
                        episode,
                        epochs=training_config.discriminator_epochs_per_update
                    )

                    discriminator_updates_completed += 1

                    # Post-update validation
                    validation_results = discriminator_monitor.validate_performance(episode, detailed=False)
                    current_accuracy = validation_results.get('accuracy', 0)

                    # Check if discriminator became too strong
                    if current_accuracy > adv_config.max_discriminator_accuracy:
                        discriminator_paused = True
                        discriminator_pause_remaining = adv_config.discriminator_pause_episodes
                        logger.log_warning(f"🛑 Discriminator paused - too strong ({current_accuracy:.3f})")

                    # Log metrics
                    disc_metrics = trainer.get_latest_metrics()
                    if disc_metrics:
                        for key, value in disc_metrics.items():
                            if 'discriminator' in key:
                                tb_logger.log_discriminator_metrics(episode, {key.replace('discriminator_', ''): value})

                else:
                    logger.log_warning("⚠️  Insufficient samples for discriminator update")

            except Exception as e:
                logger.log_warning(f"Discriminator update failed: {e}")

        # EPISODE LOGGING
        episode_metrics = {
            'reward': episode_reward,
            'success_rate': success_rate,
            'hub_improvement': avg_improvement,
            'phase': current_phase,
            'policy_updates': policy_updates_completed,
            'discriminator_updates': discriminator_updates_completed,
            'discriminator_paused': discriminator_paused
        }

        if episode % 25 == 0:
            logger.log_episode_progress(episode, episode_metrics)

        # TensorBoard logging
        tb_logger.log_episode_metrics(episode, episode_metrics)

        # Buffer stats logging
        if episode % 100 == 0:
            buffer_stats = experience_buffer.get_buffer_stats()
            logger.log_info(f"📊 Experience Buffer: {buffer_stats['size']} rollouts, "
                            f"{buffer_stats['total_states']} states, "
                            f"avg reward: {buffer_stats['avg_reward']:.3f}")

        # STRUCTURED PHASE LOGGING
        if episode % 50 == 0:
            next_policy = schedule_info['episodes_until_next_policy']
            next_disc = schedule_info['episodes_until_next_discriminator']

            logger.log_info(f"📅 TRAINING SCHEDULE [Episode {episode}]:")
            logger.log_info(f"   Phase: {current_phase}")
            logger.log_info(f"   Policy updates completed: {policy_updates_completed}")
            logger.log_info(f"   Discriminator updates completed: {discriminator_updates_completed}")
            logger.log_info(f"   Next policy update in: {next_policy} episodes")
            logger.log_info(f"   Next discriminator update in: {next_disc} episodes")
            logger.log_info(f"   Discriminator status: {'PAUSED' if discriminator_paused else 'ACTIVE'}")

        # CHECKPOINT SAVING
        if episode % 500 == 0 and episode > 0:
            try:
                checkpoint = {
                    'episode': episode,
                    'policy_state': policy.state_dict(),
                    'discriminator_state': discriminator.state_dict(),
                    'structured_training_state': {
                        'policy_updates_completed': policy_updates_completed,
                        'discriminator_updates_completed': discriminator_updates_completed,
                        'discriminator_paused': discriminator_paused,
                        'current_phase': current_phase,
                        'experience_buffer_stats': experience_buffer.get_buffer_stats()
                    },
                    'trainer_state': {
                        'update_count': trainer.update_count,
                        'policy_losses': trainer.policy_losses[-50:],
                        'disc_losses': trainer.disc_losses[-50:],
                    },
                    'metrics_history': {
                        'episode_rewards': metrics_history['episode_rewards'][-500:],
                        'success_rates': metrics_history['success_rates'][-500:],
                        'hub_improvements': metrics_history['hub_improvements'][-500:],
                    },
                    'structured_config_used': config
                }

                checkpoint_path = results_dir / f'structured_rl_checkpoint_{episode}.pt'
                torch.save(checkpoint, checkpoint_path)

                logger.log_info(f"💾 Checkpoint saved: {checkpoint_path}")
                logger.log_info(f"   Policy updates: {policy_updates_completed}")
                logger.log_info(f"   Discriminator updates: {discriminator_updates_completed}")
                logger.log_info(f"   Phase: {current_phase}")

            except Exception as e:
                logger.log_warning(f"Failed to save checkpoint: {e}")

        # Memory cleanup
        if episode % 25 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    # === TRAINING COMPLETION ===
    logger.log_info("🏁 STRUCTURED TRAINING COMPLETED!")

    # Final structured summary
    logger.log_info("📊 FINAL STRUCTURED TRAINING SUMMARY:")
    logger.log_info(f"   Total episodes: {training_config.num_episodes}")
    logger.log_info(f"   Warm-up episodes: {warmup_episodes}")
    logger.log_info(f"   Training episodes: {training_config.num_episodes - warmup_episodes}")
    logger.log_info(f"   Policy updates completed: {policy_updates_completed}")
    logger.log_info(f"   Discriminator updates completed: {discriminator_updates_completed}")
    logger.log_info(f"   Update ratio: {policy_updates_completed / max(discriminator_updates_completed, 1):.1f}:1")

    # Final performance metrics
    if metrics_history['episode_rewards']:
        final_100_rewards = metrics_history['episode_rewards'][-100:] if len(
            metrics_history['episode_rewards']) >= 100 else metrics_history['episode_rewards']
        final_100_success = metrics_history['success_rates'][-100:] if len(metrics_history['success_rates']) >= 100 else \
        metrics_history['success_rates']

        final_metrics = {
            'mean_reward_last_100': float(np.mean(final_100_rewards)),
            'mean_success_rate_last_100': float(np.mean(final_100_success)),
            'total_policy_updates': policy_updates_completed,
            'total_discriminator_updates': discriminator_updates_completed,
        }

        logger.log_final_summary(training_config.num_episodes, {'final_performance': final_metrics})

    # Save final structured model
    try:
        final_save_path = results_dir / 'final_structured_rl_model.pt'

        torch.save({
            'policy_state': policy.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'node_dim': 7,
            'edge_dim': 1,
            'structured_training_stats': {
                'total_episodes': training_config.num_episodes,
                'warmup_episodes': warmup_episodes,
                'policy_updates_completed': policy_updates_completed,
                'discriminator_updates_completed': discriminator_updates_completed,
                'final_phase': 'training',
                'update_ratio': policy_updates_completed / max(discriminator_updates_completed, 1)
            },
            'training_history': metrics_history,
            'experience_buffer_final_stats': experience_buffer.get_buffer_stats(),
            'structured_config': config,
            'trainer_final_metrics': trainer.get_latest_metrics()
        }, final_save_path)

        logger.log_info(f"💾 Final structured model saved: {final_save_path}")

    except Exception as e:
        logger.log_error(f"Failed to save final model: {e}")

    finally:
        tb_logger.close()

    # STRUCTURED TRAINING RECOMMENDATIONS
    logger.log_info("\n" + "=" * 70)
    logger.log_info("🎯 STRUCTURED TRAINING ANALYSIS:")
    logger.log_info("=" * 70)

    # Calculate actual ratios and frequencies
    actual_policy_freq = (training_config.num_episodes - warmup_episodes) / max(policy_updates_completed, 1)
    actual_disc_freq = (training_config.num_episodes - warmup_episodes) / max(discriminator_updates_completed, 1)
    actual_ratio = policy_updates_completed / max(discriminator_updates_completed, 1)

    logger.log_info(f"📊 ACTUAL VS PLANNED FREQUENCIES:")
    logger.log_info(f"   Planned policy freq: every {policy_update_frequency} episodes")
    logger.log_info(f"   Actual policy freq: every {actual_policy_freq:.1f} episodes")
    logger.log_info(f"   Planned disc freq: every {policy_update_frequency * discriminator_update_frequency} episodes")
    logger.log_info(f"   Actual disc freq: every {actual_disc_freq:.1f} episodes")
    logger.log_info(f"   Planned ratio: {discriminator_update_frequency}:1")
    logger.log_info(f"   Actual ratio: {actual_ratio:.1f}:1")

    # Recommendations based on actual performance
    if actual_ratio > discriminator_update_frequency * 1.5:
        logger.log_info("✅ Good adversarial balance - discriminator updates were appropriately limited")
    elif actual_ratio < discriminator_update_frequency * 0.5:
        logger.log_info("⚠️  Discriminator updated too frequently - consider increasing interval")
    else:
        logger.log_info("✅ Adversarial balance close to planned ratio")

    if discriminator_paused:
        logger.log_info("⚠️  Training ended with discriminator paused - consider adjusting pause thresholds")
    else:
        logger.log_info("✅ Discriminator remained active throughout training")

    logger.log_info("=" * 70)
    logger.log_info("🏆 STRUCTURED ADVERSARIAL TRAINING COMPLETED SUCCESSFULLY!")
    logger.log_info("=" * 70)


if __name__ == "__main__":
    main()