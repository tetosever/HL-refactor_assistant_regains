#!/usr/bin/env python3
"""
Enhanced GCPN Training Script with Centralized Hyperparameter Configuration
UPDATED VERSION with centralized config management

Key Features:
- Uses centralized hyperparameter configuration from hyperparameters_configuration.py
- All training parameters are now configurable from a single location
- Enhanced Environment with incremental pattern-based actions
- Discriminator-based adversarial learning with robust validation
- PPO + GAE optimization
- GPU/CUDA optimizations
- UNIFIED 7-feature pipeline compatibility
"""
import datetime
import logging
import random
import json
import os
import time
import warnings
import traceback
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data
import torch_geometric.data.data

# UPDATED IMPORTS - using incremental environment and centralized config
from data_loader import create_unified_loader, validate_dataset_consistency, UnifiedDataLoader
from discriminator import HubDetectionDiscriminator
from policy_network import HubRefactoringPolicy
from rl_gym import HubRefactoringEnv  # UPDATED: using incremental version
from hyperparameters_configuration import get_improved_config, print_config_summary
from dataclasses import dataclass


@dataclass
class RefactoringAction:
    """Refactoring action with pattern and parameters"""
    source_node: int
    target_node: int
    pattern: int
    terminate: bool
    confidence: float = 0.0


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message="Loading .* with weights_only=False")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")


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
        logger.info(f"🔄 Initializing graph pool with {self.pool_size} diverse graphs...")

        try:
            # Load larger sample for diversity
            smelly_data, labels = self.data_loader.load_dataset(
                max_samples_per_class=self.pool_size * 3,  # Get 3x more for selection
                shuffle=True,
                validate_all=False,
                remove_metadata=True
            )

            # Filter smelly graphs only
            smelly_candidates = [data for data, label in zip(smelly_data, labels) if label == 1]

            if len(smelly_candidates) < self.pool_size:
                logger.warning(f"Only {len(smelly_candidates)} smelly graphs available, wanted {self.pool_size}")
                self.pool_size = len(smelly_candidates)

            # Select diverse subset (by graph size, complexity, etc.)
            self.smelly_pool = self._select_diverse_graphs(smelly_candidates, self.pool_size)

            logger.info(f"✅ Initialized pool with {len(self.smelly_pool)} diverse smelly graphs")

        except Exception as e:
            logger.error(f"❌ Failed to initialize graph pool: {e}")
            raise

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
            size_filtered = candidates  # Fallback if no graphs in size range

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
                    # Select random graph from this size bin
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
        logger.info(f"🔄 Refreshing graph pool (refresh #{self.pool_refresh_counter})...")

        try:
            self._initialize_graph_pool()
            logger.info(f"✅ Pool refreshed with {len(self.smelly_pool)} new graphs")
        except Exception as e:
            logger.warning(f"⚠️ Pool refresh failed: {e}, keeping old pool")

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
        logger.info(f"📊 TensorBoard logging to: {run_dir}")
        logger.info(f"🔗 Run: tensorboard --logdir {self.log_dir}")

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
    """PPO trainer with centralized hyperparameter configuration"""

    def __init__(self, policy: nn.Module, discriminator: nn.Module,
                 opt_config, disc_config, device: torch.device = torch.device('cpu')):
        self.device = device
        self.opt_config = opt_config
        self.disc_config = disc_config

        # Move models to device
        self.policy = policy.to(device)
        self.discriminator = discriminator.to(device)

        # Create optimizers with centralized config
        self.opt_policy = optim.AdamW(
            policy.parameters(),
            lr=opt_config.policy_lr,
            eps=opt_config.optimizer_eps,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )
        self.opt_disc = optim.AdamW(
            discriminator.parameters(),
            lr=opt_config.discriminator_lr,
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

        # Learning rate schedulers from config
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.opt_policy,
            step_size=opt_config.policy_lr_decay_step,
            gamma=opt_config.lr_decay_factor
        )
        self.disc_scheduler = optim.lr_scheduler.StepLR(
            self.opt_disc,
            step_size=opt_config.disc_lr_decay_step,
            gamma=opt_config.lr_decay_factor
        )

        # Training statistics
        self.update_count = 0
        self.policy_losses = []
        self.disc_losses = []
        self.disc_metrics = []
        self.policy_metrics = []

        # GPU optimization settings
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        logger.info(f"PPO Trainer initialized with config:")
        logger.info(f"  - Policy LR: {opt_config.policy_lr}")
        logger.info(f"  - Discriminator LR: {opt_config.discriminator_lr}")
        logger.info(f"  - Clip epsilon: {opt_config.clip_epsilon}")
        logger.info(f"  - Entropy coefficient: {opt_config.entropy_coefficient}")

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation - GPU optimized"""
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
                             generated_graphs: List[Data]):
        """Discriminator update with centralized configuration"""

        epochs = self.opt_config.discriminator_epochs
        early_stop_threshold = self.disc_config.early_stop_threshold
        label_smoothing = self.disc_config.label_smoothing
        gp_lambda = self.disc_config.gradient_penalty_lambda

        total_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        for epoch in range(epochs):
            # Sample balanced batches with config-defined sizes
            min_size = min(len(smelly_graphs), len(clean_graphs), len(generated_graphs))
            batch_size = min(
                self.disc_config.max_samples_per_class,
                max(self.disc_config.min_samples_per_class, min_size)
            )

            if batch_size <= 2:
                logger.warning(f"Skipping discriminator update - insufficient samples: {min_size}")
                continue

            try:
                # Sample data with balanced sampling if enabled
                if self.disc_config.use_balanced_sampling:
                    smelly_sample = np.random.choice(len(smelly_graphs), batch_size, replace=False).tolist()
                    clean_sample = np.random.choice(len(clean_graphs), batch_size, replace=False).tolist()
                    generated_sample = np.random.choice(len(generated_graphs), batch_size, replace=False).tolist()
                else:
                    # Simple random sampling
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
                    logger.warning("Discriminator returned invalid output, skipping update")
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
                except Exception as e:
                    logger.debug(f"Gradient penalty calculation failed: {e}")
                    gp_loss = 0.0

                # Total discriminator loss
                disc_loss = loss_smelly + loss_clean + loss_gen + gp_lambda * gp_loss

                # Backward pass with gradient clipping
                self.opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.opt_disc.step()

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
                        tn = ((all_preds == 0) & (all_labels == 0)).sum().float()

                        precision = tp / (tp + fp + 1e-8)
                        recall = tp / (tp + fn + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)

                        # Store metrics
                        total_metrics['accuracy'].append(accuracy)
                        total_metrics['precision'].append(precision.item())
                        total_metrics['recall'].append(recall.item())
                        total_metrics['f1'].append(f1.item())

                        # Log confusion matrix if enabled
                        if self.disc_config.log_confusion_matrix:
                            logger.debug(f"Confusion Matrix - TP: {tp.item():.0f}, FP: {fp.item():.0f}, "
                                         f"TN: {tn.item():.0f}, FN: {fn.item():.0f}")

                self.disc_losses.append(disc_loss.item())

                # Early stopping
                if total_metrics['accuracy'] and total_metrics['f1']:
                    current_acc = total_metrics['accuracy'][-1]
                    current_f1 = total_metrics['f1'][-1]

                    if current_acc < early_stop_threshold and current_f1 < early_stop_threshold:
                        logger.debug(f"Early stopping discriminator at epoch {epoch}: "
                                     f"acc={current_acc:.4f}, F1={current_f1:.4f}")
                        break

                logger.debug(f"Discriminator epoch {epoch}: loss={disc_loss.item():.4f}")

            except Exception as e:
                logger.warning(f"Error in discriminator update epoch {epoch}: {e}")
                continue

        # Calculate average metrics across epochs
        if total_metrics['accuracy']:
            avg_metrics = {
                'accuracy': np.mean(total_metrics['accuracy']),
                'precision': np.mean(total_metrics['precision']),
                'recall': np.mean(total_metrics['recall']),
                'f1': np.mean(total_metrics['f1'])
            }
            self.disc_metrics.append(avg_metrics)
            logger.debug(f"Discriminator update complete - Avg F1: {avg_metrics['f1']:.4f}, "
                         f"Avg Accuracy: {avg_metrics['accuracy']:.4f}")

        # Step learning rate scheduler
        self.disc_scheduler.step()

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def update_policy(self, states: List[Data], actions: List[Any],
                      old_log_probs: torch.Tensor, advantages: torch.Tensor,
                      returns: torch.Tensor):
        """Policy update with centralized configuration"""

        epochs = self.opt_config.policy_epochs

        if len(states) == 0:
            logger.warning("No states provided for policy update")
            return

        policy_losses = []
        entropy_losses = []
        kl_divergences = []

        for epoch in range(epochs):
            # Shuffle data for better training
            indices = torch.randperm(len(states), device=self.device)
            batch_size = min(24, len(states))  # Could also be configurable

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end].cpu().numpy()

                # Create batch
                batch_states = [states[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Compute losses
                loss_info = self.compute_policy_loss(batch_states, batch_actions,
                                                     batch_old_log_probs, batch_advantages)

                # Total loss with config weights
                total_loss = (loss_info['policy_loss'] +
                              self.value_coef * 0 +  # No value loss for now
                              -self.entropy_coef * loss_info['entropy'])

                # Backward pass with gradient clipping
                self.opt_policy.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                # Store metrics
                policy_losses.append(loss_info['policy_loss'].item())
                entropy_losses.append(loss_info['entropy'].item())
                kl_divergences.append(loss_info['approx_kl'].item())

                # Early stopping on large KL divergence
                if loss_info['approx_kl'] > 0.02:
                    logger.debug(f"Early stopping policy update due to large KL: {loss_info['approx_kl']:.4f}")
                    break

        # Store average metrics
        if policy_losses:
            avg_policy_loss = np.mean(policy_losses)
            avg_entropy = np.mean(entropy_losses)
            avg_kl = np.mean(kl_divergences)

            self.policy_losses.append(avg_policy_loss)
            self.policy_metrics.append({
                'policy_loss': avg_policy_loss,
                'entropy': avg_entropy,
                'kl_divergence': avg_kl,
                'update_count': self.update_count
            })

            logger.debug(f"Policy update complete - Loss: {avg_policy_loss:.4f}, "
                         f"Entropy: {avg_entropy:.4f}, KL: {avg_kl:.4f}")

        self.update_count += 1

        # Step learning rate scheduler
        self.policy_scheduler.step()

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

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
                logger.warning(f"State {i} has {state.x.size(1)} features, expected 7")
                continue

            # Ensure batch attribute
            if not hasattr(state, 'batch') or state.batch is None:
                state.batch = torch.zeros(state.x.size(0), dtype=torch.long, device=self.device)

            valid_states.append(state)
            valid_actions.append(actions[i])
            valid_indices.append(i)

        if not valid_states:
            logger.warning("No valid states for policy loss computation")
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
        except Exception as e:
            logger.warning(f"Failed to batch states: {e}")
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
        except Exception as e:
            logger.warning(f"Policy forward pass failed: {e}")
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
                hub_probs_i = hub_probs_i / hub_probs_i.sum()  # Renormalize

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
                    pattern_probs_i = pattern_probs_i / pattern_probs_i.sum()  # Renormalize

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
                    term_probs_i = term_probs_i / term_probs_i.sum()  # Renormalize
                else:
                    term_probs_i = torch.tensor([0.7, 0.3], device=self.device)  # Default probs

                term_dist = Categorical(term_probs_i)
                term_action_tensor = torch.tensor(int(action.terminate), device=self.device)
                log_prob_term = term_dist.log_prob(term_action_tensor)
                entropy_term = term_dist.entropy()

                # Combine probabilities
                total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
                total_entropy = entropy_hub + entropy_pattern + entropy_term

                log_probs.append(total_log_prob)
                entropies.append(total_entropy)

            except Exception as e:
                logger.debug(f"Error computing log prob for action {i}: {e}")
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
        ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -10, 10))  # Clamp for stability
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
                'discriminator_recall': latest_disc['recall']
            })

        return metrics


def safe_torch_load(file_path: Path, device: torch.device):
    """Safely load PyTorch files with PyTorch Geometric compatibility"""
    try:
        with torch.serialization.safe_globals([torch_geometric.data.data.DataEdgeAttr]):
            return torch.load(file_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(file_path, map_location=device, weights_only=False)


def load_pretrained_discriminator(model_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load pretrained discriminator with unified format validation"""
    logger.info(f"Loading pretrained discriminator from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained discriminator not found at {model_path}")

    checkpoint = safe_torch_load(model_path, device)

    # Validate unified data pipeline information
    if 'unified_data_pipeline' in checkpoint:
        pipeline_info = checkpoint['unified_data_pipeline']
        logger.info(f"✅ Model was trained with unified data pipeline:")
        logger.info(f"  - Method: {pipeline_info.get('method', 'unknown')}")
        logger.info(f"  - Feature computer: {pipeline_info.get('feature_computer', 'unknown')}")
        logger.info(f"  - Normalization applied: {not pipeline_info.get('no_normalization_applied', True)}")
        logger.info(f"  - Consistent with RL: {pipeline_info.get('consistent_with_rl_training', False)}")

        if not pipeline_info.get('consistent_with_rl_training', False):
            logger.warning("⚠️ Discriminator may not be fully compatible with RL training!")
    else:
        logger.warning("⚠️ Model lacks unified data pipeline information - compatibility uncertain")

    # Get architecture parameters from checkpoint
    model_architecture = checkpoint.get('model_architecture', {})
    node_dim = checkpoint.get('node_dim', 7)  # Should always be 7 for unified format
    edge_dim = checkpoint.get('edge_dim', 1)

    # Validate unified format expectations
    if node_dim != 7:
        logger.error(f"❌ Expected 7 node features for unified format, got {node_dim}")
        raise ValueError(f"Discriminator has incompatible feature dimensions: {node_dim} != 7")

    # Use saved architecture parameters or defaults
    hidden_dim = model_architecture.get('hidden_dim', 128)
    num_layers = model_architecture.get('num_layers', 4)
    dropout = model_architecture.get('dropout', 0.15)
    heads = model_architecture.get('heads', 8)

    logger.info(f"Creating discriminator with unified format: node_dim={node_dim}, hidden_dim={hidden_dim}")

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
        logger.info(f"Loaded discriminator with CV F1: {mean_f1:.4f}, Accuracy: {mean_acc:.4f}")

    return discriminator, checkpoint


def get_discriminator_prediction(discriminator: nn.Module, data: Data, device: torch.device) -> Optional[float]:
    """Get discriminator prediction for unified format data"""
    try:
        discriminator.eval()
        data = data.clone().to(device)

        # Validate unified format
        if data.x.size(1) != 7:
            logger.warning(f"Data has {data.x.size(1)} features, expected 7")
            return None

        # Ensure batch attribute
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # Ensure edge_attr
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = discriminator(data)
            if output and 'logits' in output:
                probs = F.softmax(output['logits'], dim=1)
                return probs[0, 1].item()

        return None

    except Exception as e:
        logger.warning(f"Discriminator prediction failed: {e}")
        return None


def validate_discriminator_performance(discriminator: nn.Module, data_loader: UnifiedDataLoader,
                                       device: torch.device, num_samples: int = 40) -> Dict[str, float]:
    """Enhanced validation with unified data loader"""
    logger.info("🔍 Validating discriminator with unified data loader...")

    try:
        # Load validation data using unified loader with metadata removal
        val_data, val_labels = data_loader.load_dataset(
            max_samples_per_class=num_samples // 2,
            shuffle=True,
            validate_all=True,
            remove_metadata=True  # Critical for batching
        )

        logger.info(f"Loaded {len(val_data)} samples for validation")

        correct_predictions = 0
        total_predictions = 0
        smelly_correct = 0
        clean_correct = 0
        failed_predictions = 0

        # Test on samples with detailed logging
        for i, (data, label) in enumerate(zip(val_data, val_labels)):
            logger.debug(f"\n--- Testing sample {i} (label: {label}) ---")

            # Get hash for consistency tracking
            data_hash = data_loader.get_data_hash(data)
            logger.debug(f"Data hash: {data_hash[:12]}...")

            prob_smelly = get_discriminator_prediction(discriminator, data, device)

            if prob_smelly is not None:
                prediction = 1 if prob_smelly > 0.5 else 0
                is_correct = prediction == label

                if is_correct:
                    correct_predictions += 1
                    if label == 1:
                        smelly_correct += 1
                    else:
                        clean_correct += 1

                total_predictions += 1
                logger.debug(f"{'✅' if is_correct else '❌'} Sample {i}: pred={prediction}, "
                             f"label={label}, prob={prob_smelly:.4f}")
            else:
                failed_predictions += 1
                logger.debug(f"💥 Sample {i}: prediction failed")

        # Calculate results
        results = {
            'accuracy': correct_predictions / max(total_predictions, 1),
            'smelly_accuracy': smelly_correct / max(sum(val_labels), 1),
            'clean_accuracy': clean_correct / max(len(val_labels) - sum(val_labels), 1),
            'failed_predictions': failed_predictions,
            'total_attempts': len(val_data),
            'success_rate': total_predictions / len(val_data)
        }

        logger.info(f"🎯 Unified validation results:")
        logger.info(f"  - Overall accuracy: {results['accuracy']:.3f}")
        logger.info(f"  - Smelly accuracy: {results['smelly_accuracy']:.3f}")
        logger.info(f"  - Clean accuracy: {results['clean_accuracy']:.3f}")
        logger.info(f"  - Success rate: {results['success_rate']:.3f}")
        logger.info(f"  - Failed predictions: {failed_predictions}")

        return results

    except Exception as e:
        logger.error(f"Unified validation failed: {e}")
        return {'accuracy': 0.0, 'smelly_accuracy': 0.0, 'clean_accuracy': 0.0,
                'failed_predictions': -1, 'success_rate': 0.0}


def create_diverse_training_environments(env_manager: EnhancedEnvironmentManager,
                                         discriminator: nn.Module, training_config,
                                         device: torch.device = torch.device('cpu')) -> List[HubRefactoringEnv]:
    """Create training environments with diverse graphs using centralized config"""
    num_envs = training_config.num_envs
    max_steps = training_config.max_steps_per_episode

    logger.info(f"🌍 Creating {num_envs} environments with diverse graphs...")

    envs = []
    for i in range(num_envs):
        try:
            # Get random graph from diverse pool
            initial_graph = env_manager.get_random_graph()

            # Test discriminator compatibility
            test_pred = get_discriminator_prediction(discriminator, initial_graph, device)
            if test_pred is None:
                logger.warning(f"Environment {i}: Discriminator incompatible, trying another graph...")
                initial_graph = env_manager.get_random_graph()
                test_pred = get_discriminator_prediction(discriminator, initial_graph, device)

            if test_pred is not None:
                logger.info(f"Environment {i}: Initial hub score = {test_pred:.4f}, nodes = {initial_graph.x.size(0)}")

                # Create environment with config-based max_steps
                env = HubRefactoringEnv(initial_graph, discriminator, max_steps=max_steps, device=device)
                # Store reference to environment manager for dynamic resets
                env.env_manager = env_manager
                envs.append(env)
            else:
                logger.warning(f"Environment {i}: Could not create compatible environment")

        except Exception as e:
            logger.warning(f"Failed to create environment {i}: {e}")
            continue

    logger.info(f"✅ Created {len(envs)} diverse training environments")
    return envs


def enhanced_reset_environment(env: HubRefactoringEnv) -> Data:
    """Enhanced reset that picks new graph from pool"""
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
            return env.reset()
        else:
            # Fallback to original reset
            return env.reset()

    except Exception as e:
        logger.warning(f"Enhanced reset failed: {e}, using standard reset")
        return env.reset()


def collect_rollouts(envs: List[HubRefactoringEnv], policy: nn.Module,
                           training_config, device: torch.device = torch.device('cpu')) -> Dict[str, List]:
    """FIXED: Collect rollouts with proper metric transport"""

    steps_per_env = training_config.steps_per_env

    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    # FIXED: Add containers for step-level metrics
    all_step_infos = []  # NEW: Store all step info dictionaries
    all_hub_improvements = []  # NEW: Store hub improvements from each step
    all_hub_scores = []  # NEW: Store hub scores from each step
    episode_step_metrics = {  # NEW: Aggregate metrics per episode
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
                logger.warning(f"Environment {i}: State has {state.x.size(1)} features, expected 7")

            states.append(state)
            valid_envs.append(env)
        except Exception as e:
            logger.warning(f"Error resetting environment {i}: {e}")
            continue

    if not states:
        logger.warning("No valid environments after reset")
        return {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': torch.tensor([], device=device), 'values': [],
            'step_infos': [],  # NEW
            'episode_metrics': episode_step_metrics  # NEW
        }

    logger.debug(f"Starting rollout with {len(states)} valid environments")

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
        except Exception as e:
            logger.warning(f"Failed to batch data at step {step}: {e}")
            break

        # Policy forward pass
        with torch.no_grad():
            try:
                policy_out = policy(batch_data)
                values = []
                for state in states_on_device:
                    pred = get_discriminator_prediction(valid_envs[0].discriminator, state, device)
                    value = 1.0 - pred if pred is not None else 0.5
                    values.append(value)
                values = torch.tensor(values, device=device)
            except Exception as e:
                logger.warning(f"Error in forward pass at step {step}: {e}")
                break

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

            except Exception as e:
                logger.debug(f"Error sampling action for environment {i}: {e}")
                action = RefactoringAction(0, 0, 0, False)
                actions.append(action)
                log_probs.append(torch.tensor(0.0, device=device))

        # FIXED: Execute actions and COLLECT ALL STEP METRICS
        next_states = []
        rewards = []
        dones = []
        step_infos_batch = []  # NEW: Collect info from each step

        for i, (env, action) in enumerate(zip(valid_envs, actions)):
            try:
                state, reward, done, info = env.step(
                    (action.source_node, action.target_node, action.pattern, action.terminate)
                )

                if state.x.device != device:
                    state = state.to(device)
                if not hasattr(state, 'batch'):
                    state.batch = torch.zeros(state.x.size(0), dtype=torch.long, device=device)

                next_states.append(state)
                rewards.append(reward)
                dones.append(done)

                # FIXED: Store step info for metric transport
                step_infos_batch.append(info)

                # FIXED: Update episode metrics from step info
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
                all_step_infos.append(info)  # NEW: Store step info

                # Reset if done
                if done:
                    try:
                        reset_state = env.reset()
                        if reset_state.x.device != device:
                            reset_state = reset_state.to(device)
                        if not hasattr(reset_state, 'batch'):
                            reset_state.batch = torch.zeros(reset_state.x.size(0), dtype=torch.long, device=device)
                        next_states[i] = reset_state
                    except Exception as reset_e:
                        logger.warning(f"Error resetting environment {i}: {reset_e}")

            except Exception as e:
                logger.warning(f"Error in environment {i} step: {e}")

                # FIXED: Create fallback info even for errors
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
                    all_step_infos.append(error_info)  # NEW: Store error info

        states = next_states

        # Break if all environments are done or we have no valid states
        if not states or all(dones):
            break

        # GPU memory management
        if device.type == 'cuda' and step % 10 == 0:
            torch.cuda.empty_cache()

    # FIXED: Log collected metrics for debugging
    logger.debug(f"Rollout metrics collected:")
    logger.debug(f"  Total steps: {episode_step_metrics['total_steps']}")
    logger.debug(f"  Successful steps: {episode_step_metrics['successful_steps']}")
    logger.debug(f"  Valid metrics steps: {episode_step_metrics['valid_metrics_steps']}")
    logger.debug(f"  Total hub improvements collected: {len(all_hub_improvements)}")
    logger.debug(f"  Hub score changes: {len(episode_step_metrics['hub_score_changes'])}")
    logger.debug(f"  Fallback scores: {episode_step_metrics['fallback_scores_count']}")
    logger.debug(f"  Step errors: {len(episode_step_metrics['step_errors'])}")

    if not all_log_probs:
        logger.warning("No valid rollouts collected")
        return {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': torch.tensor([], device=device), 'values': [],
            'step_infos': [],  # NEW
            'episode_metrics': episode_step_metrics  # NEW
        }

    # FIXED: Return with step-level metrics
    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'dones': all_dones,
        'log_probs': torch.stack(all_log_probs),
        'values': all_values,
        'step_infos': all_step_infos,  # NEW: All step info dictionaries
        'hub_improvements': all_hub_improvements,  # NEW: Direct access to improvements
        'hub_scores': all_hub_scores,  # NEW: Direct access to scores
        'episode_metrics': episode_step_metrics  # NEW: Aggregated episode metrics
    }

def setup_gpu_environment():
    """Setup optimal GPU environment"""
    if torch.cuda.is_available():
        # Set memory management
        torch.cuda.empty_cache()

        # Get GPU info
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1e9

        logger.info(f"🚀 GPU Setup:")
        logger.info(f"  Available GPUs: {gpu_count}")
        logger.info(f"  Current GPU: {gpu_name}")
        logger.info(f"  GPU Memory: {gpu_memory:.1f} GB")

        # Optimize for memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device = torch.device('cuda')
    else:
        logger.warning("⚠️  CUDA not available, using CPU")
        device = torch.device('cpu')

    return device


def monitor_gpu_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9

        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_memory:.2f}GB")

        # Clean up if using too much memory
        if allocated > 8.0:  # If using more than 8GB
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleanup performed")


def main():
    """FIXED main training loop with proper step-level metric collection"""

    # Load centralized configuration
    config = get_improved_config()
    reward_config = config['rewards']
    training_config = config['training']
    opt_config = config['optimization']
    disc_config = config['discriminator']
    env_config = config['environment']

    # Print configuration summary
    print_config_summary()

    # Setup GPU environment
    device = setup_gpu_environment()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    logger.info(f"🎯 Starting RL training with CENTRALIZED CONFIG + FIXED METRICS on device: {device}")

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
        logger.error(f"❌ Pretrained discriminator not found at {discriminator_path}")
        logger.error("Please run the unified discriminator pre-training script first")
        return

    # Create unified data loader
    logger.info("📊 Creating unified data loader...")
    try:
        data_loader = create_unified_loader(data_dir, device)
        stats = data_loader.get_dataset_statistics()
        logger.info(f"Dataset loaded with unified loader:")
        logger.info(f"  - Total files: {stats['total_files']}")
        logger.info(f"  - Valid files: {stats['validation_summary']['valid']}")
        logger.info(f"  - Smelly samples: {stats['label_distribution']['smelly']}")
        logger.info(f"  - Clean samples: {stats['label_distribution']['clean']}")

        consistency_report = validate_dataset_consistency(data_dir, sample_size=100)
        if consistency_report['invalid_files'] > 0:
            logger.warning(f"⚠️ Found {consistency_report['invalid_files']} invalid files")
        else:
            logger.info("✅ All sampled files passed consistency validation")

    except Exception as e:
        logger.error(f"❌ Failed to create unified data loader: {e}")
        return

    # Load pretrained discriminator
    try:
        logger.info("🔄 Loading pretrained discriminator...")
        discriminator, discriminator_checkpoint = load_pretrained_discriminator(discriminator_path, device)

        validation_results = validate_discriminator_performance(discriminator, data_loader, device, num_samples=40)

        if validation_results['accuracy'] < 0.1:
            logger.error("❌ Discriminator performance is critically low!")
            return
        elif validation_results['failed_predictions'] > validation_results['total_attempts'] // 2:
            logger.warning(f"⚠️ High failure rate: {validation_results['failed_predictions']} failed predictions")
        else:
            logger.info("✅ Discriminator validation passed!")

    except Exception as e:
        logger.error(f"❌ Failed to load pretrained discriminator: {e}")
        return

    # Model parameters - FIXED for unified 7-feature format
    node_dim = 7
    edge_dim = 1
    hidden_dim = discriminator_checkpoint.get('model_architecture', {}).get('hidden_dim', 128)

    logger.info(f"🔧 Model dimensions - Node: {node_dim} (UNIFIED), Edge: {edge_dim}, Hidden: {hidden_dim}")

    # Create policy network
    logger.info("🧠 Creating policy network...")
    policy = HubRefactoringPolicy(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy network - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Setup PPO trainer
    logger.info("🎯 Setting up PPO trainer...")
    trainer = PPOTrainer(policy, discriminator, opt_config, disc_config, device=device)

    # Training parameters from centralized config
    num_episodes = training_config.num_episodes
    num_envs = min(training_config.num_envs if device.type == 'cuda' else 3, stats['label_distribution']['smelly'])
    steps_per_env = training_config.steps_per_env
    update_frequency = training_config.update_frequency
    environment_refresh_frequency = training_config.environment_refresh_frequency

    # Hub score tracking parameters
    hub_score_improvement_threshold = training_config.hub_improvement_threshold
    consecutive_failures_threshold = training_config.consecutive_failures_threshold

    logger.info("⚙️ CENTRALIZED + FIXED METRICS training configuration:")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Environments: {num_envs}")
    logger.info(f"  Steps per env: {steps_per_env}")
    logger.info(f"  Update frequency: {update_frequency}")
    logger.info(f"  Hub improvement threshold: {hub_score_improvement_threshold}")

    # Create enhanced environment manager
    logger.info("🌍 Creating enhanced environment manager...")
    try:
        env_manager = EnhancedEnvironmentManager(
            data_loader=data_loader,
            discriminator=discriminator,
            env_config=env_config,
            device=device
        )
        logger.info(f"✅ Environment manager created with {env_manager.pool_size} diverse graphs")
    except Exception as e:
        logger.error(f"❌ Failed to create environment manager: {e}")
        return

    # Create environments
    logger.info("🌍 Creating training environments...")
    try:
        envs = create_diverse_training_environments(env_manager, discriminator, training_config, device)
        logger.info(f"✅ Successfully created {len(envs)} diverse training environments")
    except Exception as e:
        logger.error(f"❌ Failed to create training environments: {e}")
        return

    # Update environment reward structure
    logger.info("🎯 Updating environment reward structures...")
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

            logger.debug(f"Environment {i}: Updated with centralized reward config")
        except Exception as e:
            logger.warning(f"Failed to update environment {i} config: {e}")

    logger.info("🚀 Starting FIXED METRICS RL training...")

    # Training variables
    episode_rewards = []
    success_rates = []
    hub_improvements = []
    hub_scores = []
    consecutive_failures = 0
    best_episode_reward = float('-inf')

    # FIXED: Additional metric tracking
    valid_metrics_ratios = []
    fallback_score_ratios = []
    pattern_success_rates = []
    step_error_counts = []

    for episode in range(num_episodes):
        episode_start_time = time.time()

        # Environment refresh
        if episode % environment_refresh_frequency == 0 and episode > 0:
            logger.info(f"🔄 Environment refresh at episode {episode}")

            reset_count = 0
            initial_hub_scores = []

            for i, env in enumerate(envs):
                try:
                    new_state = enhanced_reset_environment(env)
                    initial_score = env.initial_hub_score if hasattr(env, 'initial_hub_score') else None
                    if initial_score is not None:
                        initial_hub_scores.append(initial_score)
                    reset_count += 1
                except Exception as e:
                    logger.warning(f"Failed to reset environment {i}: {e}")

            logger.info(f"✅ Refreshed {reset_count} environments")
            if initial_hub_scores:
                avg_initial_score = np.mean(initial_hub_scores)
                logger.info(f"📊 New environments avg initial hub score: {avg_initial_score:.4f}")

        # Pool refresh
        if (env_manager.should_refresh_pool(episode) or
                consecutive_failures > consecutive_failures_threshold):
            env_manager._refresh_pool()
            consecutive_failures = 0
            logger.info("🔄 Graph pool refreshed")

        # FIXED: Collect rollouts with step-level metrics
        rollout_data = collect_rollouts(envs, policy, training_config, device)

        if len(rollout_data['rewards']) == 0:
            logger.warning(f"No valid rollouts in episode {episode}")
            consecutive_failures += 1
            continue

        # FIXED: Extract metrics directly from step-level data
        episode_reward = np.mean(rollout_data['rewards'])
        episode_rewards.append(episode_reward)

        # FIXED: Get step-level metrics
        step_infos = rollout_data.get('step_infos', [])
        rollout_hub_improvements = rollout_data.get('hub_improvements', [])
        rollout_hub_scores = rollout_data.get('hub_scores', [])
        episode_step_metrics = rollout_data.get('episode_metrics', {})

        # FIXED: Calculate success rate from actual step improvements
        if rollout_hub_improvements:
            successful_improvements = [imp for imp in rollout_hub_improvements if imp > hub_score_improvement_threshold]
            success_rate = len(successful_improvements) / len(rollout_hub_improvements)
        else:
            success_rate = 0.0
        success_rates.append(success_rate)

        # FIXED: Calculate average improvements from step data
        avg_improvement = np.mean(rollout_hub_improvements) if rollout_hub_improvements else 0.0
        hub_improvements.append(avg_improvement)

        # FIXED: Calculate average hub scores from step data
        avg_hub_score = np.mean(rollout_hub_scores) if rollout_hub_scores else 0.5
        hub_scores.append(avg_hub_score)

        # FIXED: Additional detailed metrics
        total_steps = episode_step_metrics.get('total_steps', 0)
        valid_metrics_steps = episode_step_metrics.get('valid_metrics_steps', 0)
        fallback_scores_count = episode_step_metrics.get('fallback_scores_count', 0)
        successful_steps = episode_step_metrics.get('successful_steps', 0)
        step_errors = episode_step_metrics.get('step_errors', [])

        # Calculate ratios
        valid_metrics_ratio = valid_metrics_steps / max(total_steps, 1)
        fallback_ratio = fallback_scores_count / max(total_steps, 1)
        pattern_success_rate = successful_steps / max(total_steps, 1)

        valid_metrics_ratios.append(valid_metrics_ratio)
        fallback_score_ratios.append(fallback_ratio)
        pattern_success_rates.append(pattern_success_rate)
        step_error_counts.append(len(step_errors))

        # Update consecutive failures counter
        if success_rate > 0.05 or episode_reward > -0.5 or pattern_success_rate > 0.1:
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # Track best performance
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward

        # FIXED: Enhanced logging with detailed step metrics
        if episode % 25 == 0:  # More frequent logging for debugging
            logger.info(f"Episode {episode:5d} DETAILED METRICS:")
            logger.info(f"  Reward: {episode_reward:7.3f}")
            logger.info(f"  Success Rate: {success_rate:5.3f} (threshold: {hub_score_improvement_threshold})")
            logger.info(f"  Hub Improvement: {avg_improvement:6.4f}")
            logger.info(f"  Hub Score: {avg_hub_score:5.3f}")
            logger.info(f"  Pattern Success Rate: {pattern_success_rate:5.3f}")
            logger.info(f"  Valid Metrics Ratio: {valid_metrics_ratio:5.3f}")
            logger.info(f"  Fallback Score Ratio: {fallback_ratio:5.3f}")
            logger.info(f"  Step Errors: {len(step_errors)}")
            logger.info(f"  Total Steps: {total_steps}")
            logger.info(f"  Consecutive Failures: {consecutive_failures}")

            # FIXED: Log actual hub improvement values for debugging
            if rollout_hub_improvements:
                imp_stats = {
                    'min': min(rollout_hub_improvements),
                    'max': max(rollout_hub_improvements),
                    'mean': np.mean(rollout_hub_improvements),
                    'positive_count': sum(1 for x in rollout_hub_improvements if x > 0),
                    'above_threshold': sum(1 for x in rollout_hub_improvements if x > hub_score_improvement_threshold)
                }
                logger.info(f"  Hub Improvement Stats: {imp_stats}")

            # FIXED: Log hub score distribution
            if rollout_hub_scores:
                score_stats = {
                    'min': min(rollout_hub_scores),
                    'max': max(rollout_hub_scores),
                    'mean': np.mean(rollout_hub_scores),
                    'fallback_count': sum(1 for x in rollout_hub_scores if abs(x - 0.5) < 1e-6)
                }
                logger.info(f"  Hub Score Stats: {score_stats}")

            # FIXED: Log sample step errors for debugging
            if step_errors and len(step_errors) > 0:
                logger.warning(f"  Sample step errors:")
                for i, error in enumerate(step_errors[:3]):  # Show first 3 errors
                    logger.warning(
                        f"    {i + 1}. Env {error.get('env_id', '?')}, Step {error.get('step', '?')}: {error.get('error', 'unknown')[:100]}")

        # Policy updates
        if episode % update_frequency == 0 and len(rollout_data['states']) >= 16:

            # Compute GAE advantages
            advantages, returns = trainer.compute_gae(
                rollout_data['rewards'],
                rollout_data['values'],
                rollout_data['dones']
            )

            # Update policy
            trainer.update_policy(
                rollout_data['states'],
                rollout_data['actions'],
                rollout_data['log_probs'],
                advantages,
                returns
            )

            # Log policy metrics
            policy_metrics = trainer.get_latest_metrics()
            if policy_metrics:
                for key, value in policy_metrics.items():
                    if 'policy' in key:
                        tb_logger.log_policy_metrics(trainer.update_count, {key.replace('policy_', ''): value})

            # Discriminator fine-tuning
            disc_update_freq = update_frequency * disc_config.update_frequency_multiplier
            if episode % disc_update_freq == 0 and episode > 100:
                try:
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

                        trainer.update_discriminator(
                            smelly_graphs[:disc_config.max_samples_per_class],
                            clean_graphs[:disc_config.max_samples_per_class],
                            recent_states[:disc_config.max_samples_per_class]
                        )

                        disc_metrics = trainer.get_latest_metrics()
                        if disc_metrics:
                            for key, value in disc_metrics.items():
                                if 'discriminator' in key:
                                    tb_logger.log_discriminator_metrics(episode,
                                                                        {key.replace('discriminator_', ''): value})

                except Exception as e:
                    logger.warning(f"Discriminator update failed at episode {episode}: {e}")

        # FIXED: Enhanced TensorBoard logging with step-level metrics
        tb_logger.log_episode_metrics(episode, {
            'reward': episode_reward,
            'success_rate': success_rate,
            'hub_improvement': avg_improvement,
            'average_hub_score': avg_hub_score,
            'consecutive_failures': consecutive_failures,
            'best_reward_so_far': best_episode_reward,
            'valid_metrics_ratio': valid_metrics_ratio,
            'fallback_scores_ratio': fallback_ratio,
            'pattern_success_rate': pattern_success_rate,
            'step_errors_count': len(step_errors),
            'total_steps': total_steps,
            'hub_improvements_count': len(rollout_hub_improvements),
            'positive_improvements': sum(
                1 for x in rollout_hub_improvements if x > 0) if rollout_hub_improvements else 0
        })

        # Enhanced logging every 50 episodes
        if episode % 50 == 0:
            recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            recent_success_rates = success_rates[-50:] if len(success_rates) >= 50 else success_rates
            recent_improvements = hub_improvements[-50:] if len(hub_improvements) >= 50 else hub_improvements
            recent_hub_scores = hub_scores[-50:] if len(hub_scores) >= 50 else hub_scores
            recent_valid_ratios = valid_metrics_ratios[-50:] if len(
                valid_metrics_ratios) >= 50 else valid_metrics_ratios
            recent_pattern_success = pattern_success_rates[-50:] if len(
                pattern_success_rates) >= 50 else pattern_success_rates

            avg_reward_50 = np.mean(recent_rewards) if recent_rewards else 0
            avg_success_50 = np.mean(recent_success_rates) if recent_success_rates else 0
            avg_improvement_50 = np.mean(recent_improvements) if recent_improvements else 0
            avg_hub_score_50 = np.mean(recent_hub_scores) if recent_hub_scores else 0.5
            avg_valid_ratio_50 = np.mean(recent_valid_ratios) if recent_valid_ratios else 0
            avg_pattern_success_50 = np.mean(recent_pattern_success) if recent_pattern_success else 0

            latest_metrics = trainer.get_latest_metrics()
            policy_loss = latest_metrics.get('policy_loss', 0)
            disc_f1 = latest_metrics.get('discriminator_f1', 0)

            logger.info(f"=== EPISODE {episode:5d} SUMMARY (Last 50) ===")
            logger.info(f"Reward: {avg_reward_50:7.3f} | "
                        f"Success: {avg_success_50:5.3f} | "
                        f"Hub Improv: {avg_improvement_50:6.4f} | "
                        f"Hub Score: {avg_hub_score_50:5.3f}")
            logger.info(f"Valid Metrics: {avg_valid_ratio_50:5.3f} | "
                        f"Pattern Success: {avg_pattern_success_50:5.3f} | "
                        f"Policy Loss: {policy_loss:7.4f} | "
                        f"Disc F1: {disc_f1:5.3f}")

            # FIXED: Detailed diagnostic logging
            if avg_success_50 == 0.0:
                logger.warning("🚨 ZERO SUCCESS RATE - Potential issues:")
                logger.warning(f"  - Average valid metrics ratio: {avg_valid_ratio_50:.3f}")
                logger.warning(f"  - Average pattern success: {avg_pattern_success_50:.3f}")
                logger.warning(f"  - Hub improvement threshold: {hub_score_improvement_threshold}")
                logger.warning(f"  - Recent avg improvement: {avg_improvement_50:.6f}")

                if avg_valid_ratio_50 < 0.5:
                    logger.warning("  → Most steps are not calculating metrics properly")
                if avg_pattern_success_50 < 0.1:
                    logger.warning("  → Most patterns are failing to apply")
                if abs(avg_improvement_50) < 1e-6:
                    logger.warning("  → Hub improvements are exactly zero (likely fallback scores)")

            # Early stopping conditions
            if episode > 1000:
                if consecutive_failures > consecutive_failures_threshold * 2:
                    logger.warning(f"Too many consecutive failures ({consecutive_failures})")

                if avg_reward_50 > 5.0 and avg_success_50 > 0.7:
                    logger.info(f"🎉 Excellent performance achieved!")

        # Save checkpoints
        if episode % 500 == 0 and episode > 0:
            try:
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
                    'fixed_metrics_stats': {
                        'episode_rewards': episode_rewards[-500:],
                        'success_rates': success_rates[-500:],
                        'hub_improvements': hub_improvements[-500:],
                        'hub_scores': hub_scores[-500:],
                        'valid_metrics_ratios': valid_metrics_ratios[-500:],
                        'fallback_score_ratios': fallback_score_ratios[-500:],
                        'pattern_success_rates': pattern_success_rates[-500:],
                        'step_error_counts': step_error_counts[-500:],
                        'best_episode_reward': best_episode_reward,
                        'consecutive_failures': consecutive_failures
                    },
                    'centralized_config_used': {
                        'rewards': reward_config.__dict__,
                        'training': training_config.__dict__,
                        'optimization': opt_config.__dict__,
                        'discriminator': disc_config.__dict__,
                        'environment': env_config.__dict__
                    },
                    'metrics_fix_applied': True,
                    'step_level_metric_collection': True
                }

                checkpoint_path = results_dir / f'fixed_metrics_rl_checkpoint_{episode}.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"💾 Saved FIXED METRICS checkpoint: {checkpoint_path}")

            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        # Memory cleanup
        if episode % 25 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    # Training completed - Enhanced summary
    logger.info("🏁 FIXED METRICS RL Training completed!")

    # Final performance analysis
    if episode_rewards:
        final_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        final_100_success = success_rates[-100:] if len(success_rates) >= 100 else success_rates
        final_100_improvements = hub_improvements[-100:] if len(hub_improvements) >= 100 else hub_improvements
        final_100_hub_scores = hub_scores[-100:] if len(hub_scores) >= 100 else hub_scores
        final_100_valid_ratios = valid_metrics_ratios[-100:] if len(
            valid_metrics_ratios) >= 100 else valid_metrics_ratios
        final_100_pattern_success = pattern_success_rates[-100:] if len(
            pattern_success_rates) >= 100 else pattern_success_rates

        logger.info("\n" + "=" * 80)
        logger.info("🎯 FINAL FIXED METRICS TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total episodes: {num_episodes}")
        logger.info(f"Best episode reward: {best_episode_reward:.4f}")
        logger.info(f"Final 100 episodes performance:")
        logger.info(f"  Average reward: {np.mean(final_100_rewards):.4f}")
        logger.info(f"  Average success rate: {np.mean(final_100_success):.4f}")
        logger.info(f"  Average hub improvement: {np.mean(final_100_improvements):.4f}")
        logger.info(f"  Average hub score: {np.mean(final_100_hub_scores):.4f}")
        logger.info(f"  Average valid metrics ratio: {np.mean(final_100_valid_ratios):.4f}")
        logger.info(f"  Average pattern success rate: {np.mean(final_100_pattern_success):.4f}")
        logger.info(f"  Final consecutive failures: {consecutive_failures}")

        # FIXED: Diagnostic summary
        logger.info(f"\nFIXED METRICS DIAGNOSTIC SUMMARY:")
        logger.info(f"  Hub improvement threshold used: {hub_score_improvement_threshold}")
        logger.info(
            f"  Total step-level metrics collected: {sum(len(rollout_data.get('step_infos', [])) for _ in range(10))}")  # Approximate
        logger.info(f"  Average fallback score usage: {np.mean(fallback_score_ratios):.3f}")

        final_success = np.mean(final_100_success)
        final_valid_ratio = np.mean(final_100_valid_ratios)

        if final_success > 0.0:
            logger.info("✅ SUCCESS: Non-zero success rate achieved with fixed metrics!")
        else:
            logger.warning("⚠️ Still zero success rate - possible issues:")
            if final_valid_ratio < 0.5:
                logger.warning("  - Low valid metrics ratio suggests discriminator/pattern issues")
            else:
                logger.warning("  - Valid metrics collected but no improvements above threshold")
                logger.warning(f"  - Consider lowering threshold from {hub_score_improvement_threshold}")

    # Save final model
    try:
        final_save_path = results_dir / 'final_fixed_metrics_rl_model.pt'
        final_stats = {
            'total_episodes': num_episodes,
            'best_episode_reward': best_episode_reward,
            'final_performance': {
                'mean_reward_last_100': float(np.mean(episode_rewards[-100:])) if len(
                    episode_rewards) >= 100 else float(np.mean(episode_rewards)) if episode_rewards else 0,
                'mean_success_rate_last_100': float(np.mean(success_rates[-100:])) if len(
                    success_rates) >= 100 else float(np.mean(success_rates)) if success_rates else 0,
                'mean_hub_improvement_last_100': float(np.mean(hub_improvements[-100:])) if len(
                    hub_improvements) >= 100 else float(np.mean(hub_improvements)) if hub_improvements else 0,
                'mean_hub_score_last_100': float(np.mean(hub_scores[-100:])) if len(hub_scores) >= 100 else float(
                    np.mean(hub_scores)) if hub_scores else 0.5,
                'mean_valid_metrics_ratio_last_100': float(np.mean(valid_metrics_ratios[-100:])) if len(
                    valid_metrics_ratios) >= 100 else float(
                    np.mean(valid_metrics_ratios)) if valid_metrics_ratios else 0,
                'mean_pattern_success_rate_last_100': float(np.mean(pattern_success_rates[-100:])) if len(
                    pattern_success_rates) >= 100 else float(
                    np.mean(pattern_success_rates)) if pattern_success_rates else 0,
                'consecutive_failures': consecutive_failures,
                'total_updates': trainer.update_count
            },
            'metrics_fix_features': {
                'step_level_metric_collection': True,
                'fixed_metric_transport': True,
                'detailed_step_info_tracking': True,
                'fallback_score_detection': True,
                'pattern_success_tracking': True,
                'enhanced_debugging_logs': True
            }
        }

        torch.save({
            'policy_state': policy.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'fixed_metrics_training_stats': final_stats,
            'training_history': {
                'episode_rewards': episode_rewards,
                'success_rates': success_rates,
                'hub_improvements': hub_improvements,
                'hub_scores': hub_scores,
                'valid_metrics_ratios': valid_metrics_ratios,
                'fallback_score_ratios': fallback_score_ratios,
                'pattern_success_rates': pattern_success_rates,
                'step_error_counts': step_error_counts
            },
            'trainer_final_metrics': trainer.get_latest_metrics(),
            'complete_centralized_config': {
                'rewards': reward_config.__dict__,
                'training': training_config.__dict__,
                'optimization': opt_config.__dict__,
                'discriminator': disc_config.__dict__,
                'environment': env_config.__dict__
            },
            'metrics_fixes_applied': [
                'Step-level metric collection in collect_rollouts_fixed()',
                'Direct hub_improvement extraction from step infos',
                'Enhanced success rate calculation from actual improvements',
                'Detailed step-by-step debugging information',
                'Fallback score detection and tracking',
                'Pattern success rate monitoring',
                'Comprehensive error tracking and reporting',
                'Real-time metric validation and consistency checks'
            ],
            'debugging_capabilities': {
                'step_info_collection': True,
                'hub_score_validation': True,
                'pattern_success_tracking': True,
                'error_categorization': True,
                'metric_consistency_checks': True,
                'detailed_logging': True,
                'fallback_detection': True
            }
        }, final_save_path)

        logger.info(f"📁 Final FIXED METRICS model saved: {final_save_path}")

    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    finally:
        tb_logger.close()

    logger.info("=" * 80)
    logger.info("🎉 FIXED METRICS RL TRAINING COMPLETED!")
    logger.info("🔧 Key fixes applied:")
    logger.info("  ✅ Step-level metric collection and transport")
    logger.info("  ✅ Direct hub_improvement extraction from step infos")
    logger.info("  ✅ Success rate calculation from actual step improvements")
    logger.info("  ✅ Enhanced debugging with detailed step information")
    logger.info("  ✅ Fallback score detection and monitoring")
    logger.info("  ✅ Pattern success rate tracking")
    logger.info("  ✅ Comprehensive error tracking and categorization")
    logger.info("  ✅ Real-time metric validation")
    logger.info(f"📊 View training progress: tensorboard --logdir {tensorboard_dir}")
    logger.info(f"⚙️  Edit hyperparameters in: hyperparameters_configuration.py")

    # FIXED: Final diagnostic recommendations
    logger.info("🔍 DIAGNOSTIC RECOMMENDATIONS:")
    if np.mean(success_rates[-100:] if success_rates else [0]) == 0:
        logger.info("  📌 If success rate is still 0, check:")
        logger.info("     - Are hub_improvements being calculated correctly?")
        logger.info("     - Is the improvement threshold too high?")
        logger.info("     - Are patterns applying successfully?")
        logger.info("     - Is the discriminator returning valid scores?")

    if np.mean(valid_metrics_ratios[-100:] if valid_metrics_ratios else [0]) < 0.5:
        logger.info("  📌 Low valid metrics ratio suggests:")
        logger.info("     - Discriminator compatibility issues")
        logger.info("     - Graph structure problems")
        logger.info("     - Pattern application failures")

    if np.mean(fallback_score_ratios[-100:] if fallback_score_ratios else [0]) > 0.8:
        logger.info("  📌 High fallback score usage suggests:")
        logger.info("     - Discriminator is returning invalid outputs")
        logger.info("     - Graph features may be corrupted")
        logger.info("     - Batch processing issues")

    logger.info("=" * 80)

if __name__ == "__main__":
    main()