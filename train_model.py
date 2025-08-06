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
    """PPO trainer with centralized hyperparameter configuration and discriminator monitoring"""

    def __init__(self, policy: nn.Module, discriminator: nn.Module, discriminator_monitor: AdvancedDiscriminatorMonitor,
                 opt_config, disc_config, lr_config, device: torch.device = torch.device('cpu')):
        self.device = device
        self.opt_config = opt_config
        self.disc_config = disc_config
        self.lr_config = lr_config
        self.discriminator_monitor = discriminator_monitor

        # Move models to device
        self.policy = policy.to(device)
        self.discriminator = discriminator.to(device)

        # Create optimizers with new learning rate configuration
        self.opt_policy = optim.AdamW(
            policy.parameters(),
            lr=lr_config.policy_base_lr,  # Start with base LR for CyclicLR
            eps=opt_config.optimizer_eps,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )

        self.opt_disc = optim.AdamW(
            discriminator.parameters(),
            lr=lr_config.discriminator_initial_lr,  # Start with initial LR for ReduceLROnPlateau
            eps=opt_config.optimizer_eps,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )

        # PPO parameters from config
        self.clip_eps = opt_config.clip_epsilon
        self.gamma = 0.99
        self.lam = 0.95
        self.value_coef = opt_config.value_coefficient
        self.entropy_coef = opt_config.entropy_coefficient
        self.max_grad_norm = opt_config.max_grad_norm

        # NEW: Advanced learning rate schedulers
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

        # Track metrics for discriminator scheduler
        self.disc_metric_history = []
        self.current_episode = 0

        # Training statistics
        self.update_count = 0
        self.policy_losses = []
        self.disc_losses = []
        self.disc_metrics = []
        self.policy_metrics = []

        # GPU optimization settings
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

    def update_discriminator_lr(self, new_lr: float):
        """Aggiorna learning rate del discriminatore dal monitor"""
        for param_group in self.opt_disc.param_groups:
            param_group['lr'] = new_lr

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:

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

        # Convert to tensors on device
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages with numerical stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_discriminator(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                             generated_graphs: List[Data], episode: int):
        """Discriminator update with enhanced monitoring and adaptive parameters"""

        # Get adaptive parameters from monitor
        adaptive_params = self.discriminator_monitor.get_adaptive_training_params()

        # Skip update if discriminator is critically degraded
        if adaptive_params['should_skip_update']:
            self.discriminator_monitor.logger.log_warning(
                f"⚠️  Skipping discriminator update due to critical degradation")
            return

        epochs = self.opt_config.discriminator_epochs
        early_stop_threshold = self.disc_config.early_stop_threshold
        label_smoothing = self.disc_config.label_smoothing
        gp_lambda = self.disc_config.gradient_penalty_lambda

        total_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        epoch_losses = []

        self.discriminator_monitor.logger.log_info(
            f"🧠 Starting discriminator update - Episode {episode}")

        for epoch in range(epochs):
            # Sample balanced batches with config-defined sizes
            min_size = min(len(smelly_graphs), len(clean_graphs), len(generated_graphs))
            batch_size = min(
                self.disc_config.max_samples_per_class,
                max(self.disc_config.min_samples_per_class, min_size)
            )

            if batch_size <= 2:
                self.discriminator_monitor.logger.log_warning(f"⚠️  Insufficient batch size: {batch_size}")
                continue

            try:
                # Sample data
                if self.disc_config.use_balanced_sampling:
                    smelly_sample = np.random.choice(len(smelly_graphs), batch_size, replace=False).tolist()
                    clean_sample = np.random.choice(len(clean_graphs), batch_size, replace=False).tolist()
                    generated_sample = np.random.choice(len(generated_graphs), batch_size, replace=False).tolist()
                else:
                    smelly_sample = random.sample(range(len(smelly_graphs)), batch_size)
                    clean_sample = random.sample(range(len(clean_graphs)), batch_size)
                    generated_sample = random.sample(range(len(generated_graphs)), batch_size)

                smelly_batch_data = [smelly_graphs[i].to(self.device) for i in smelly_sample]
                clean_batch_data = [clean_graphs[i].to(self.device) for i in clean_sample]
                generated_batch_data = [generated_graphs[i].to(self.device) for i in generated_sample]

                # Create batches
                smelly_batch = Batch.from_data_list(smelly_batch_data)
                clean_batch = Batch.from_data_list(clean_batch_data)
                generated_batch = Batch.from_data_list(generated_batch_data)

                # Enable gradients for gradient penalty
                smelly_batch.x.requires_grad_(True)
                clean_batch.x.requires_grad_(True)
                generated_batch.x.requires_grad_(True)

                # Forward pass
                self.discriminator.train()
                smelly_out = self.discriminator(smelly_batch)
                clean_out = self.discriminator(clean_batch)
                gen_out = self.discriminator(generated_batch)

                # Validate outputs
                if any(out is None or 'logits' not in out for out in [smelly_out, clean_out, gen_out]):
                    self.discriminator_monitor.logger.log_warning(f"⚠️  Invalid discriminator outputs at epoch {epoch}")
                    continue

                # Labels (smelly=1, clean/generated=0)
                smelly_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
                clean_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                gen_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                # Compute losses with label smoothing
                loss_smelly = F.cross_entropy(smelly_out['logits'], smelly_labels,
                                              label_smoothing=label_smoothing)
                loss_clean = F.cross_entropy(clean_out['logits'], clean_labels,
                                             label_smoothing=label_smoothing)
                loss_gen = F.cross_entropy(gen_out['logits'], gen_labels,
                                           label_smoothing=label_smoothing)

                # Gradient penalty calculation
                gp_loss = 0.0
                try:
                    for batch_data, out in [(smelly_batch, smelly_out),
                                            (clean_batch, clean_out),
                                            (generated_batch, gen_out)]:
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
                except Exception:
                    gp_loss = 0.0

                # Total discriminator loss
                disc_loss = loss_smelly + loss_clean + loss_gen + gp_lambda * gp_loss

                # Backward pass with gradient clipping
                self.opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.opt_disc.step()

                epoch_losses.append(disc_loss.item())

                # Calculate detailed metrics if enabled
                if self.disc_config.calculate_detailed_metrics:
                    with torch.no_grad():
                        self.discriminator.eval()

                        # Get predictions
                        smelly_preds = smelly_out['logits'].argmax(dim=1)
                        clean_preds = clean_out['logits'].argmax(dim=1)
                        gen_preds = gen_out['logits'].argmax(dim=1)

                        # Combine all predictions and labels
                        all_preds = torch.cat([smelly_preds, clean_preds, gen_preds])
                        all_labels = torch.cat([smelly_labels, clean_labels, gen_labels])

                        # Calculate metrics
                        correct = (all_preds == all_labels).float()
                        accuracy = correct.mean().item()

                        # Calculate precision, recall, F1 for positive class (smelly)
                        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
                        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
                        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)

                        # Store metrics
                        total_metrics['accuracy'].append(accuracy)
                        total_metrics['precision'].append(precision.item())
                        total_metrics['recall'].append(recall.item())
                        total_metrics['f1'].append(f1.item())

                self.disc_losses.append(disc_loss.item())

                # Early stopping based on performance
                if total_metrics['accuracy'] and total_metrics['f1']:
                    current_acc = total_metrics['accuracy'][-1]
                    current_f1 = total_metrics['f1'][-1]

                    # Early stopping if performance is very poor
                    if current_acc < early_stop_threshold and current_f1 < early_stop_threshold:
                        self.discriminator_monitor.logger.log_warning(
                            f"⚠️  Early stopping at epoch {epoch} due to poor performance")
                        break

                    # Early stopping if performance degradation is severe
                    if (current_acc < 0.4 and
                            self.discriminator_monitor.perf_state.baseline_accuracy > 0.6):
                        self.discriminator_monitor.logger.log_warning(f"🚨 Early stopping due to severe degradation")
                        break

            except Exception as e:
                self.discriminator_monitor.logger.log_warning(f"⚠️  Discriminator training error at epoch {epoch}: {e}")
                continue

        # Calculate and log average metrics across epochs
        if total_metrics['accuracy']:
            avg_metrics = {
                'accuracy': np.mean(total_metrics['accuracy']),
                'precision': np.mean(total_metrics['precision']),
                'recall': np.mean(total_metrics['recall']),
                'f1': np.mean(total_metrics['f1']),
                'loss': np.mean(epoch_losses) if epoch_losses else 0.0
            }

            self.disc_metrics.append(avg_metrics)

            # NEW: Update ReduceLROnPlateau scheduler with monitored metric
            monitored_metric = avg_metrics[self.lr_config.discriminator_monitor_metric]
            self.disc_scheduler.step(monitored_metric)

            # Log training results
            current_lr = self.opt_disc.param_groups[0]['lr']
            self.discriminator_monitor.logger.log_info(
                f"🧠 Discriminator training completed - "
                f"Acc: {avg_metrics['accuracy']:.3f}, F1: {avg_metrics['f1']:.3f}, "
                f"Loss: {avg_metrics['loss']:.4f}, LR: {current_lr:.2e}"
            )
        else:
            self.discriminator_monitor.logger.log_warning(
                "⚠️  No valid metrics calculated during discriminator training")

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def update_policy(self, states: List[Data], actions: List[Any],
                      old_log_probs: torch.Tensor, advantages: torch.Tensor,
                      returns: torch.Tensor):
        """Policy update with centralized configuration and real critic update"""

        epochs = self.opt_config.policy_epochs
        if len(states) == 0:
            return

        policy_losses, entropy_losses, value_losses, kl_divs = [], [], [], []

        for epoch in range(epochs):
            # shuffle indices
            indices = torch.randperm(len(states), device=self.device)
            batch_size = min(24, len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end].cpu().numpy()

                # batch states/actions
                batch_states = [states[i] for i in batch_idx]
                batch_actions = [actions[i] for i in batch_idx]
                batch_old_logp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]

                # compute losses
                out = self.policy(Batch.from_data_list(batch_states).to(self.device))

                # --- policy loss + entropy ---
                logp, ent = self._compute_logp_and_entropy(out, batch_actions)
                ratio = torch.exp(logp - batch_old_logp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value loss (MSE) ---
                value_preds = out['value'].squeeze()
                value_loss = F.mse_loss(value_preds, batch_ret)

                # --- total loss ---
                total_loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * ent.mean()
                )

                # backward & step
                self.opt_policy.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                # record
                policy_losses.append(policy_loss.item())
                entropy_losses.append(ent.mean().item())
                value_losses.append(value_loss.item())
                with torch.no_grad():
                    kl_divs.append((batch_old_logp - logp).mean().item())

        self.policy_scheduler.step()

        # aggregate metrics
        self.policy_losses.append(np.mean(policy_losses))
        self.policy_metrics.append({
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divs)
        })
        self.update_count += 1

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _compute_logp_and_entropy(self, policy_out: Dict, actions: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy for actions"""
        log_probs = []
        entropies = []

        # Get output dimensions
        num_nodes = policy_out['hub_probs'].size(0)

        for i, action in enumerate(actions):
            try:
                # Hub selection log prob
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

                # Pattern selection log prob
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

                # Termination log prob
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

                # Combine
                total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
                total_entropy = entropy_hub + entropy_pattern + entropy_term

                log_probs.append(total_log_prob)
                entropies.append(total_entropy)

            except Exception as e:
                log_probs.append(torch.tensor(-5.0, device=self.device))
                entropies.append(torch.tensor(0.1, device=self.device))

        return torch.stack(log_probs), torch.stack(entropies)

    def compute_policy_loss(self, states: List[Data], actions: List[Any],
                            old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Policy loss computation with better error handling"""

        # Validate inputs
        if not states or not actions:
            return {
                'policy_loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'entropy': torch.tensor(0.0, device=self.device),
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        # Ensure all states have correct format
        valid_states = []
        valid_actions = []
        valid_indices = []

        for i, state in enumerate(states):
            if state.x.device != self.device:
                state = state.to(self.device)

            # Validate 7-feature format
            if state.x.size(1) != 7:
                continue

            # Ensure batch attribute
            if not hasattr(state, 'batch') or state.batch is None:
                state.batch = torch.zeros(state.x.size(0), dtype=torch.long, device=self.device)

            valid_states.append(state)
            valid_actions.append(actions[i])
            valid_indices.append(i)

        if not valid_states:
            return {
                'policy_loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'entropy': torch.tensor(0.0, device=self.device),
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        # Filter old_log_probs and advantages to match valid states
        if len(valid_indices) < len(old_log_probs):
            old_log_probs = old_log_probs[valid_indices]
            advantages = advantages[valid_indices]

        # Batch states with better error handling
        try:
            batch_data = Batch.from_data_list(valid_states)
            if batch_data.x.device != self.device:
                batch_data = batch_data.to(self.device)
        except Exception:
            return {
                'policy_loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'entropy': torch.tensor(0.0, device=self.device),
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        # Forward pass
        try:
            self.policy.train()
            policy_out = self.policy(batch_data)
        except Exception:
            return {
                'policy_loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'entropy': torch.tensor(0.0, device=self.device),
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        # Compute current log probabilities with improved error handling
        log_probs = []
        entropies = []

        # Handle batched data properly
        num_graphs = len(valid_states)
        nodes_per_graph = [state.x.size(0) for state in valid_states]
        cumsum_nodes = [0] + list(np.cumsum(nodes_per_graph))

        for i, action in enumerate(valid_actions):
            try:
                start_idx = cumsum_nodes[i]
                end_idx = cumsum_nodes[i + 1]

                # Validate indices
                if start_idx >= policy_out['hub_probs'].size(0) or end_idx > policy_out['hub_probs'].size(0):
                    log_probs.append(torch.tensor(-5.0, device=self.device))
                    entropies.append(torch.tensor(0.1, device=self.device))
                    continue

                # Hub selection with better bounds checking
                hub_probs_i = policy_out['hub_probs'][start_idx:end_idx]
                if hub_probs_i.size(0) == 0:
                    log_probs.append(torch.tensor(-5.0, device=self.device))
                    entropies.append(torch.tensor(0.1, device=self.device))
                    continue

                # Add numerical stability
                hub_probs_i = hub_probs_i + 1e-8
                hub_probs_i = hub_probs_i / hub_probs_i.sum()

                hub_dist = Categorical(hub_probs_i)
                hub_action_tensor = torch.tensor(
                    min(action.source_node, hub_probs_i.size(0) - 1),
                    device=self.device
                )

                log_prob_hub = hub_dist.log_prob(hub_action_tensor)
                entropy_hub = hub_dist.entropy()

                # Pattern selection with better error handling
                pattern_idx = start_idx + min(action.source_node, hub_probs_i.size(0) - 1)
                if pattern_idx < policy_out['pattern_probs'].size(0):
                    pattern_probs_i = policy_out['pattern_probs'][pattern_idx] + 1e-8
                    pattern_probs_i = pattern_probs_i / pattern_probs_i.sum()

                    pattern_dist = Categorical(pattern_probs_i)
                    pattern_action_tensor = torch.tensor(
                        min(action.pattern, pattern_probs_i.size(0) - 1),
                        device=self.device
                    )

                    log_prob_pattern = pattern_dist.log_prob(pattern_action_tensor)
                    entropy_pattern = pattern_dist.entropy()
                else:
                    log_prob_pattern = torch.tensor(-2.0, device=self.device)
                    entropy_pattern = torch.tensor(0.1, device=self.device)

                # Termination with better handling
                if i < policy_out['term_probs'].size(0):
                    term_probs_i = policy_out['term_probs'][i] + 1e-8
                    term_probs_i = term_probs_i / term_probs_i.sum()
                else:
                    term_probs_i = torch.tensor([0.7, 0.3], device=self.device)

                term_dist = Categorical(term_probs_i)
                term_action_tensor = torch.tensor(int(action.terminate), device=self.device)
                log_prob_term = term_dist.log_prob(term_action_tensor)
                entropy_term = term_dist.entropy()

                # Combine probabilities
                total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
                total_entropy = entropy_hub + entropy_pattern + entropy_term

                log_probs.append(total_log_prob)
                entropies.append(total_entropy)

            except Exception:
                log_probs.append(torch.tensor(-5.0, device=self.device))
                entropies.append(torch.tensor(0.1, device=self.device))

        if not log_probs:
            return {
                'policy_loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'entropy': torch.tensor(0.0, device=self.device),
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropies).mean()

        # Ensure dimensions match
        old_log_probs = old_log_probs.to(self.device)
        if old_log_probs.size(0) != log_probs.size(0):
            min_size = min(old_log_probs.size(0), log_probs.size(0))
            old_log_probs = old_log_probs[:min_size]
            log_probs = log_probs[:min_size]
            advantages = advantages[:min_size]

        # PPO clipped objective with numerical stability
        ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -10, 10))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute additional metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clipfrac = ((ratio - 1).abs() > self.clip_eps).float().mean()

        return {
            'policy_loss': policy_loss,
            'entropy': entropy,
            'approx_kl': approx_kl,
            'clipfrac': clipfrac
        }

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


def main():
    """Main training loop with comprehensive discriminator monitoring"""
    # Setup optimized logging
    logger = setup_optimized_logging()

    # Load centralized configuration
    config = get_improved_config()
    reward_config = config['rewards']
    training_config = config['training']
    opt_config = config['optimization']
    disc_config = config['discriminator']
    env_config = config['environment']

    # Start training with config
    logger.start_training(config)

    # Print configuration summary (only once)
    print_config_summary()

    # Setup GPU environment
    device = setup_gpu_environment(logger)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'dataset_builder' / 'data' / 'dataset_graph_feature'
    results_dir = base_dir / 'results'
    discriminator_path = results_dir / 'discriminator_pretraining' / 'pretrained_discriminator.pt'

    results_dir.mkdir(exist_ok=True)

    # TensorBoard setup
    tensorboard_dir = results_dir / 'tensorboard_logs'
    tb_logger = TensorBoardLogger(tensorboard_dir)

    # Check if pretrained discriminator exists
    if not discriminator_path.exists():
        logger.log_error(f"Pretrained discriminator not found at {discriminator_path}")
        return

    # Create unified data loader
    try:
        data_loader = create_unified_loader(data_dir, device)
        stats = data_loader.get_dataset_statistics()
        consistency_report = validate_dataset_consistency(data_dir, sample_size=100)

    except Exception as e:
        logger.log_error(f"Failed to create unified data loader: {e}")
        return

    # Load pretrained discriminator
    try:
        discriminator, discriminator_checkpoint = load_pretrained_discriminator(discriminator_path, device, logger)

        # Initialize comprehensive discriminator monitor
        discriminator_monitor = AdvancedDiscriminatorMonitor(
            discriminator=discriminator,
            data_loader=data_loader,
            device=device,
            logger=logger
        )

        # Set baseline from checkpoint
        baseline_metrics = {
            'accuracy': discriminator_checkpoint.get('cv_results', {}).get('mean_accuracy', 0.0),
            'f1': discriminator_checkpoint.get('cv_results', {}).get('mean_f1', 0.0),
            'precision': discriminator_checkpoint.get('cv_results', {}).get('mean_precision', 0.0),
            'recall': discriminator_checkpoint.get('cv_results', {}).get('mean_recall', 0.0)
        }
        discriminator_monitor.set_baseline_performance(baseline_metrics)

        # Initial validation
        logger.log_info("🔍 Performing initial discriminator validation...")
        initial_validation = discriminator_monitor.validate_performance(episode=0, detailed=True)

        if initial_validation['accuracy'] < 0.1:
            logger.log_error("Discriminator performance is critically low!")
            return
        elif initial_validation['failed_predictions'] > initial_validation['total_samples'] // 2:
            logger.log_warning(f"High failure rate: {initial_validation['failed_predictions']} failed predictions")

    except Exception as e:
        logger.log_error(f"Failed to load pretrained discriminator: {e}")
        return

    # Model parameters - FIXED for unified 7-feature format
    hidden_dim = discriminator_checkpoint.get('model_architecture', {}).get('hidden_dim', 128)

    # Create policy network
    policy = HubRefactoringPolicy(
        node_dim=7,
        edge_dim=1,
        hidden_dim=hidden_dim,
        hub_bias_strength=4.0,
        exploration_rate=0.1
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.log_info(f"Policy network - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Setup PPO trainer with discriminator monitor
    trainer = PPOTrainer(policy, discriminator, discriminator_monitor, opt_config, disc_config, config['learning_rate'],
                         device=device)
    # Training parameters from centralized config
    num_episodes = training_config.num_episodes
    num_envs = min(training_config.num_envs if device.type == 'cuda' else 3, stats['label_distribution']['smelly'])
    steps_per_env = training_config.steps_per_env
    update_frequency = training_config.update_frequency
    environment_refresh_frequency = training_config.environment_refresh_frequency

    # Hub score tracking parameters
    hub_score_improvement_threshold = training_config.hub_improvement_threshold
    consecutive_failures_threshold = training_config.consecutive_failures_threshold

    # Create enhanced environment manager
    try:
        env_manager = EnhancedEnvironmentManager(
            data_loader=data_loader,
            discriminator=discriminator,
            env_config=env_config,
            device=device
        )
    except Exception as e:
        logger.log_error(f"Failed to create environment manager: {e}")
        return

    # Create environments
    try:
        envs = create_diverse_training_environments(env_manager, discriminator, training_config, device)
    except Exception as e:
        logger.log_error(f"Failed to create training environments: {e}")
        return

    # Update environment reward structure
    for i, env in enumerate(envs):
        try:
            env.REWARD_SUCCESS = reward_config.REWARD_SUCCESS
            env.REWARD_PARTIAL_SUCCESS = reward_config.REWARD_PARTIAL_SUCCESS
            env.REWARD_FAILURE = reward_config.REWARD_FAILURE
            env.REWARD_STEP = reward_config.REWARD_STEP
            env.REWARD_HUB_REDUCTION = reward_config.REWARD_HUB_REDUCTION
            env.REWARD_INVALID = reward_config.REWARD_INVALID
            env.REWARD_SMALL_IMPROVEMENT = reward_config.REWARD_SMALL_IMPROVEMENT

            env.HUB_SCORE_EXCELLENT = reward_config.HUB_SCORE_EXCELLENT
            env.HUB_SCORE_GOOD = reward_config.HUB_SCORE_GOOD
            env.HUB_SCORE_ACCEPTABLE = reward_config.HUB_SCORE_ACCEPTABLE

            env.improvement_multiplier_boost = reward_config.improvement_multiplier_boost
            env.improvement_multiplier_decay = reward_config.improvement_multiplier_decay
            env.improvement_multiplier_max = reward_config.improvement_multiplier_max
            env.improvement_multiplier_min = reward_config.improvement_multiplier_min
        except Exception as e:
            logger.log_warning(f"Failed to update environment {i} config: {e}")

    # Create progress tracker
    metrics_history = create_progress_tracker()
    consecutive_failures = 0
    best_episode_reward = float('-inf')

    # Get adaptive discriminator parameters
    adaptive_params = discriminator_monitor.get_adaptive_training_params()
    logger.log_info(
        f"🔧 Initial adaptive params - LR: {adaptive_params['discriminator_lr']:.2e}, Freq: {adaptive_params['update_frequency']}")

    # Main training loop with comprehensive discriminator monitoring
    for episode in range(num_episodes):
        episode_start_time = time.time()

        # === DISCRIMINATOR MONITORING PHASE ===

        # Regular discriminator validation
        if discriminator_monitor.should_validate(episode):
            logger.log_info(f"🔍 Validating discriminator at episode {episode}")
            detailed = (episode % discriminator_monitor.detailed_validation_frequency == 0) and episode > 0
            validation_results = discriminator_monitor.validate_performance(episode, detailed=detailed)
            rollback_occurred = discriminator_monitor.check_and_handle_degradation(episode)

            if rollback_occurred:
                logger.log_warning(f"🔄 Discriminator rollback occurred - resetting environments")
                valid_envs = []
                for env in envs:
                    try:
                        reset_state = enhanced_reset_environment(env)
                        if reset_state is not None:
                            valid_envs.append(env)
                    except Exception as e:
                        logger.log_warning(f"Failed to reset environment: {e}")
                envs = valid_envs  # ✅ Usa solo environments validi

            discriminator_monitor.update_best_model(episode)
            tb_logger.log_discriminator_metrics(episode, validation_results)

        # Get updated adaptive parameters
        adaptive_params = discriminator_monitor.get_adaptive_training_params()
        current_disc_update_freq = adaptive_params['update_frequency']

        if adaptive_params['discriminator_lr'] != trainer.opt_disc.param_groups[0]['lr']:
            trainer.update_discriminator_lr(adaptive_params['discriminator_lr'])
            logger.log_info(f"🔧 Updated discriminator LR to {adaptive_params['discriminator_lr']:.2e}")

        # Environment refresh
        if episode % environment_refresh_frequency == 0 and episode > 0:
            reset_count = 0
            for i, env in enumerate(envs):
                try:
                    enhanced_reset_environment(env)
                    reset_count += 1
                except Exception:
                    pass

            logger.log_environment_refresh(episode, env_manager.get_stats())

        # Pool refresh
        if (env_manager.should_refresh_pool(episode) or
                consecutive_failures > consecutive_failures_threshold):
            env_manager._refresh_pool()
            consecutive_failures = 0

        # === ROLLOUT COLLECTION ===
        rollout_data = collect_rollouts(envs, policy, training_config, device)

        if len(rollout_data['rewards']) == 0:
            consecutive_failures += 1
            continue

        # Extract metrics directly from step-level data
        episode_reward = np.mean(rollout_data['rewards'])
        metrics_history['episode_rewards'].append(episode_reward)

        # Get step-level metrics
        step_infos = rollout_data.get('step_infos', [])
        rollout_hub_improvements = rollout_data.get('hub_improvements', [])
        rollout_hub_scores = rollout_data.get('hub_scores', [])
        episode_step_metrics = rollout_data.get('episode_metrics', {})

        # Calculate success rate from actual step improvements
        if rollout_hub_improvements:
            successful_improvements = [imp for imp in rollout_hub_improvements if imp > hub_score_improvement_threshold]
            success_rate = len(successful_improvements) / len(rollout_hub_improvements)
        else:
            success_rate = 0.0
        metrics_history['success_rates'].append(success_rate)

        # Calculate average improvements from step data
        avg_improvement = np.mean(rollout_hub_improvements) if rollout_hub_improvements else 0.0
        metrics_history['hub_improvements'].append(avg_improvement)

        # Calculate average hub scores from step data
        avg_hub_score = np.mean(rollout_hub_scores) if rollout_hub_scores else 0.5
        metrics_history['hub_scores'].append(avg_hub_score)

        # Update consecutive failures counter
        if success_rate > 0.05 or episode_reward > -0.5:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # Track best performance
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward

        # Intelligent episode logging
        episode_metrics = {
            'reward': episode_reward,
            'success_rate': success_rate,
            'hub_improvement': avg_improvement,
            'average_hub_score': avg_hub_score,
            'consecutive_failures': consecutive_failures
        }
        logger.log_episode_progress(episode, episode_metrics)

        # === POLICY UPDATES ===
        if episode % update_frequency == 0 and len(rollout_data['states']) >= 16:
            # ✅ CORRETTO: GAE con tensor conversion
            advantages, returns = trainer.compute_gae(
                rollout_data['rewards'],  # List[float]
                rollout_data['values'],  # List[float]
                rollout_data['dones']  # List[bool]
            )

            # ✅ VALIDAZIONE: controlla che abbiamo dati validi
            if advantages.numel() > 0 and returns.numel() > 0:
                trainer.update_policy(
                    rollout_data['states'],
                    rollout_data['actions'],
                    rollout_data['log_probs'],
                    advantages,
                    returns
                )

                training_metrics = trainer.get_latest_metrics()
                logger.log_training_update(trainer.update_count, training_metrics)

                if training_metrics:
                    for key, value in training_metrics.items():
                        if 'policy' in key:
                            tb_logger.log_policy_metrics(trainer.update_count, {key.replace('policy_', ''): value})
            else:
                logger.log_warning(f"⚠️  Skipping policy update - insufficient valid data")

        # === DISCRIMINATOR UPDATES (with adaptive frequency) ===
        if episode % current_disc_update_freq == 0 and episode > 100:
            if not adaptive_params['should_skip_update']:
                try:
                    logger.log_info(f"🧠 Updating discriminator at episode {episode} (freq: {current_disc_update_freq})")

                    smelly_samples, labels = data_loader.load_dataset(
                        max_samples_per_class=disc_config.max_samples_per_class,
                        shuffle=True,
                        remove_metadata=True
                    )

                    smelly_graphs = [data for data, label in zip(smelly_samples, labels) if label == 1]
                    clean_graphs = [data for data, label in zip(smelly_samples, labels) if label == 0]

                    recent_states = []
                    for env in envs:
                        if hasattr(env, 'current_data') and env.current_data is not None:
                            recent_states.append(env.current_data.clone())

                    min_samples = disc_config.min_samples_per_class
                    if (len(smelly_graphs) >= min_samples and
                            len(clean_graphs) >= min_samples and
                            len(recent_states) >= min_samples):

                        # Perform discriminator update with monitoring
                        trainer.update_discriminator(
                            smelly_graphs[:disc_config.max_samples_per_class],
                            clean_graphs[:disc_config.max_samples_per_class],
                            recent_states[:disc_config.max_samples_per_class],
                            episode
                        )

                        # Immediate post-update validation to check for degradation
                        post_update_validation = discriminator_monitor.validate_performance(
                            episode, detailed=False
                        )

                        # Check if update caused degradation
                        if discriminator_monitor.perf_state.is_critically_degraded():
                            logger.log_warning(f"🚨 Critical degradation detected after discriminator update!")
                            discriminator_monitor.check_and_handle_degradation(episode)

                        # Log to TensorBoard
                        disc_metrics = trainer.get_latest_metrics()
                        if disc_metrics:
                            for key, value in disc_metrics.items():
                                if 'discriminator' in key:
                                    tb_logger.log_discriminator_metrics(episode,
                                                                        {key.replace('discriminator_', ''): value})
                    else:
                        logger.log_warning(f"⚠️  Insufficient samples for discriminator update")

                except Exception as e:
                    logger.log_warning(f"Discriminator update failed at episode {episode}: {e}")

                    # Recovery: reset discriminator a stato precedente se disponibile
                    if hasattr(discriminator_monitor,
                               'best_model_state') and discriminator_monitor.best_model_state is not None:
                        logger.log_info("🔄 Attempting discriminator recovery from best state")
                        try:
                            trainer.discriminator.load_state_dict(discriminator_monitor.best_model_state)
                            discriminator_monitor.emergency_interventions += 1
                            logger.log_info("✅ Discriminator recovery successful")
                        except Exception as recovery_error:
                            logger.log_error(f"❌ Recovery failed: {recovery_error}")
            else:
                logger.log_warning(f"⚠️  Skipping discriminator update due to critical performance")

        # === TENSORBOARD LOGGING ===
        tb_logger.log_episode_metrics(episode, {
            'reward': episode_reward,
            'success_rate': success_rate,
            'hub_improvement': avg_improvement,
            'average_hub_score': avg_hub_score,
            'consecutive_failures': consecutive_failures,
            'best_reward_so_far': best_episode_reward
        })

        # Log discriminator monitoring metrics
        monitoring_summary = discriminator_monitor.get_monitoring_summary()
        if episode % 25 == 0:
            perf_state = monitoring_summary['performance_state']
            tb_logger.log_discriminator_metrics(episode, {
                'adaptive_lr': adaptive_params['discriminator_lr'],
                'adaptive_update_freq': adaptive_params['update_frequency'],
                'degradation_count': perf_state['stability']['consecutive_degradations'],
                'improvement_count': perf_state['stability']['consecutive_improvements'],
                'vs_baseline_acc': perf_state['vs_baseline']['accuracy_diff'],
                'vs_baseline_f1': perf_state['vs_baseline']['f1_diff']
            })

        # === MILESTONE SUMMARIES ===
        logger.log_milestone_summary(episode, metrics_history)

        # === CHECKPOINT SAVING ===
        if episode % 500 == 0 and episode > 0:
            try:
                # ✅ AGGIUNTO: Salva stato discriminator completo
                discriminator_stats = discriminator_monitor.get_monitoring_summary()

                checkpoint = {
                    'episode': episode,
                    'policy_state': policy.state_dict(),
                    'discriminator_state': discriminator.state_dict(),
                    'trainer_state': {
                        'update_count': trainer.update_count,
                        'policy_losses': trainer.policy_losses[-100:],
                        'disc_losses': trainer.disc_losses[-100:],
                        'policy_metrics': trainer.policy_metrics[-50:],
                        'disc_metrics': trainer.disc_metrics[-50:]
                    },
                    'metrics_stats': {
                        'episode_rewards': metrics_history['episode_rewards'][-500:],
                        'success_rates': metrics_history['success_rates'][-500:],
                        'hub_improvements': metrics_history['hub_improvements'][-500:],
                        'hub_scores': metrics_history['hub_scores'][-500:],
                        'best_episode_reward': best_episode_reward,
                        'consecutive_failures': consecutive_failures
                    },
                    'discriminator_monitoring': {
                        'performance_state': discriminator_monitor.perf_state.get_status_summary(),
                        'monitoring_stats': discriminator_stats['monitoring_stats'],
                        'validation_history': discriminator_monitor.validation_history[-20:],
                        'adaptive_params': adaptive_params,
                        'baseline_performance': discriminator_monitor.perf_state.baseline_accuracy,
                        # ✅ AGGIUNTO: stato per recovery
                        'best_model_available': discriminator_monitor.best_model_state is not None,
                        'emergency_interventions': discriminator_monitor.emergency_interventions,
                        'rollback_count': discriminator_monitor.rollback_count
                    },
                    'centralized_config_used': {
                        'rewards': reward_config.__dict__,
                        'training': training_config.__dict__,
                        'optimization': opt_config.__dict__,
                        'discriminator': disc_config.__dict__,
                        'environment': env_config.__dict__
                    }
                }

                checkpoint_path = results_dir / f'rl_checkpoint_{episode}.pt'
                torch.save(checkpoint, checkpoint_path)

                performance = {
                    'best_reward': best_episode_reward,
                    'current_success_rate': success_rate,
                    'discriminator_status': 'stable' if discriminator_monitor.perf_state.is_stable() else 'unstable'
                }
                logger.log_checkpoint_save(episode, checkpoint_path, performance)

                logger.log_info(
                    f"🧠 Discriminator state - Acc: {discriminator_monitor.perf_state.current_accuracy:.3f}, "
                    f"Interventions: {discriminator_monitor.emergency_interventions}, "
                    f"Rollbacks: {discriminator_monitor.rollback_count}")

            except Exception as e:
                logger.log_warning(f"Failed to save checkpoint: {e}")

        # === MEMORY CLEANUP ===
        if episode % 25 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
            monitor_gpu_usage(logger)

        # === EMERGENCY DISCRIMINATOR INTERVENTIONS ===
        # Check for critical discriminator state every 50 episodes
        if episode % 50 == 0 and episode > 200:
            if discriminator_monitor.perf_state.is_critically_degraded():
                logger.log_warning(f"🚨 EMERGENCY: Critical discriminator degradation detected!")

                # Force validation and intervention
                emergency_validation = discriminator_monitor.validate_performance(episode, detailed=True)
                emergency_handled = discriminator_monitor.check_and_handle_degradation(episode)

                if emergency_handled:
                    logger.log_warning(f"🔄 Emergency intervention completed - restarting environments")
                    # Reset all environments after emergency intervention
                    for env in envs:
                        try:
                            enhanced_reset_environment(env)
                        except Exception:
                            pass

    # === TRAINING COMPLETION ===
    logger.log_info("RL Training completed!")

    # Final discriminator analysis
    logger.log_info("🔍 Performing final comprehensive discriminator validation...")
    final_discriminator_validation = discriminator_monitor.validate_performance(
        episode=num_episodes, detailed=True
    )

    # Final performance analysis
    if metrics_history['episode_rewards']:
        final_100_rewards = metrics_history['episode_rewards'][-100:] if len(
            metrics_history['episode_rewards']) >= 100 else metrics_history['episode_rewards']
        final_100_success = metrics_history['success_rates'][-100:] if len(metrics_history['success_rates']) >= 100 else \
            metrics_history['success_rates']
        final_100_improvements = metrics_history['hub_improvements'][-100:] if len(
            metrics_history['hub_improvements']) >= 100 else metrics_history['hub_improvements']
        final_100_hub_scores = metrics_history['hub_scores'][-100:] if len(metrics_history['hub_scores']) >= 100 else \
            metrics_history['hub_scores']

        final_metrics = {
            'mean_reward_last_100': float(np.mean(final_100_rewards)),
            'mean_success_rate_last_100': float(np.mean(final_100_success)),
            'mean_hub_improvement_last_100': float(np.mean(final_100_improvements)),
            'mean_hub_score_last_100': float(np.mean(final_100_hub_scores)),
            'total_updates': trainer.update_count
        }

        # Include discriminator final analysis
        discriminator_final_summary = discriminator_monitor.get_monitoring_summary()

        logger.log_final_summary(num_episodes, {
            'final_performance': final_metrics,
            'discriminator_analysis': discriminator_final_summary
        })

    # === FINAL MODEL SAVING ===
    try:
        final_save_path = results_dir / 'final_rl_model.pt'
        final_stats = {
            'total_episodes': num_episodes,
            'best_episode_reward': best_episode_reward,
            'final_performance': final_metrics if 'final_metrics' in locals() else {},
            'consecutive_failures': consecutive_failures,
            'total_updates': trainer.update_count
        }

        # Comprehensive discriminator statistics for final save
        discriminator_final_stats = {
            'final_validation': final_discriminator_validation,
            'monitoring_summary': discriminator_monitor.get_monitoring_summary(),
            'performance_evolution': {
                'baseline': {
                    'accuracy': discriminator_monitor.perf_state.baseline_accuracy,
                    'f1': discriminator_monitor.perf_state.baseline_f1
                },
                'final': {
                    'accuracy': discriminator_monitor.perf_state.current_accuracy,
                    'f1': discriminator_monitor.perf_state.current_f1
                },
                'best': {
                    'accuracy': discriminator_monitor.perf_state.best_accuracy,
                    'f1': discriminator_monitor.perf_state.best_f1,
                    'episode': discriminator_monitor.perf_state.best_episode
                }
            },
            'interventions': {
                'emergency_interventions': discriminator_monitor.emergency_interventions,
                'rollbacks': discriminator_monitor.rollback_count,
                'total_validations': len(discriminator_monitor.validation_history),
                'failed_validations': discriminator_monitor.perf_state.failed_validations
            },
            'adaptive_learning': {
                'final_lr': adaptive_params['discriminator_lr'],
                'final_update_freq': adaptive_params['update_frequency'],
                'lr_adjustments': discriminator_monitor.current_discriminator_lr != discriminator_monitor.base_discriminator_lr
            }
        }

        torch.save({
            'policy_state': policy.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'node_dim': 7,
            'edge_dim': 1,
            'training_stats': final_stats,
            'training_history': metrics_history,
            'trainer_final_metrics': trainer.get_latest_metrics(),
            'discriminator_comprehensive_analysis': discriminator_final_stats,
            'centralized_config': {
                'rewards': reward_config.__dict__,
                'training': training_config.__dict__,
                'optimization': opt_config.__dict__,
                'discriminator': disc_config.__dict__,
                'environment': env_config.__dict__
            }
        }, final_save_path)

        logger.log_info(f"Final model saved: {final_save_path}")

        # Log final discriminator status
        logger.log_info("🧠 FINAL DISCRIMINATOR ANALYSIS:")
        logger.log_info(f"   Baseline Performance: Acc {discriminator_monitor.perf_state.baseline_accuracy:.3f}")
        logger.log_info(f"   Final Performance: Acc {discriminator_monitor.perf_state.current_accuracy:.3f}")
        logger.log_info(
            f"   Best Performance: Acc {discriminator_monitor.perf_state.best_accuracy:.3f} (Episode {discriminator_monitor.perf_state.best_episode})")
        logger.log_info(f"   Emergency Interventions: {discriminator_monitor.emergency_interventions}")
        logger.log_info(f"   Model Rollbacks: {discriminator_monitor.rollback_count}")
        logger.log_info(f"   Total Validations: {len(discriminator_monitor.validation_history)}")

        # Performance verdict
        acc_change = discriminator_monitor.perf_state.current_accuracy - discriminator_monitor.perf_state.baseline_accuracy
        if acc_change > 0.05:
            verdict = "✅ SIGNIFICANTLY IMPROVED"
        elif acc_change > 0.02:
            verdict = "✅ IMPROVED"
        elif acc_change > -0.02:
            verdict = "➡️  MAINTAINED"
        elif acc_change > -0.05:
            verdict = "⚠️  SLIGHTLY DEGRADED"
        else:
            verdict = "❌ SIGNIFICANTLY DEGRADED"

        logger.log_info(f"   Overall Verdict: {verdict} ({acc_change:+.3f})")

    except Exception as e:
        logger.log_error(f"Failed to save final model: {e}")

    finally:
        tb_logger.close()

    # Final recommendations based on discriminator behavior
    logger.log_info("\n" + "=" * 70)
    logger.log_info("🎯 TRAINING RECOMMENDATIONS BASED ON DISCRIMINATOR ANALYSIS:")
    logger.log_info("=" * 70)

    if discriminator_monitor.emergency_interventions > 3:
        logger.log_info("⚠️  High number of emergency interventions - consider:")
        logger.log_info("   • Lower discriminator learning rate")
        logger.log_info("   • Increase discriminator update frequency")
        logger.log_info("   • Review discriminator architecture complexity")

    if discriminator_monitor.rollback_count > 1:
        logger.log_info("🔄 Multiple rollbacks occurred - consider:")
        logger.log_info("   • More conservative discriminator updates")
        logger.log_info("   • Better discriminator regularization")
        logger.log_info("   • Longer baseline establishment period")

    if discriminator_monitor.perf_state.is_stable():
        logger.log_info("✅ Discriminator remained stable throughout training")
        logger.log_info("   • Current hyperparameters appear well-tuned")
        logger.log_info("   • Consider longer training for better results")
    else:
        logger.log_info("⚠️  Discriminator showed instability - consider:")
        logger.log_info("   • Review adversarial training balance")
        logger.log_info("   • Implement more frequent validation")
        logger.log_info("   • Adjust policy-discriminator update ratio")

    logger.log_info("=" * 70)
    logger.log_info("🏁 COMPREHENSIVE RL TRAINING WITH DISCRIMINATOR MONITORING COMPLETED!")
    logger.log_info("=" * 70)
    logger.log_info(f"📊 View detailed training progress: tensorboard --logdir {tensorboard_dir}")
    logger.log_info(f"⚙️  Edit hyperparameters: hyperparameters_configuration.py")
    logger.log_info(f"📁 Results saved in: {results_dir}")


if __name__ == "__main__":
    main()