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


@dataclass
class AdaptiveDifficultyConfig:
    """🎯 Configuration for adaptive difficulty curriculum"""

    # Target performance metrics
    target_success_rate: float = 0.65  # Optimal success rate (65%)
    success_rate_window: int = 100  # Episodes to measure success

    # Difficulty adjustment thresholds
    difficulty_increase_threshold: float = 0.80  # Increase difficulty if success > 80%
    difficulty_decrease_threshold: float = 0.45  # Decrease if success < 45%

    # Adjustment parameters
    adjustment_strength: float = 0.08  # How much to adjust difficulty
    min_difficulty: float = 0.0  # Minimum difficulty (easiest graphs)
    max_difficulty: float = 1.0  # Maximum difficulty (hardest graphs)

    # Graph sharing for stability
    same_graph_policy_updates: int = 2  # Same graph for 2 policy updates

    # Complexity factors for difficulty calculation
    size_weight: float = 0.4  # Weight for graph size
    connectivity_weight: float = 0.4  # Weight for connectivity density
    hub_strength_weight: float = 0.2  # Weight for hub strength

    # Stability features
    difficulty_smoothing: float = 0.7  # Exponential smoothing for difficulty changes
    min_episodes_before_adjustment: int = 50  # Minimum episodes before first adjustment


class AdaptiveDifficultyEnvironmentManager:
    """🎯 Enhanced Environment Manager with Adaptive Difficulty Curriculum"""

    def __init__(self, data_loader: UnifiedDataLoader, discriminator: nn.Module,
                 adaptive_config: AdaptiveDifficultyConfig, device: torch.device = torch.device('cpu')):
        self.data_loader = data_loader
        self.discriminator = discriminator
        self.adaptive_config = adaptive_config
        self.device = device

        # Adaptive difficulty state
        self.current_difficulty = 0.3  # Start with moderate difficulty
        self.success_rate_history = []
        self.difficulty_history = []
        self.difficulty_adjustments = []

        # Graph pools organized by difficulty levels
        self.difficulty_pools = {}  # {difficulty_level: [graphs]}
        self.graph_complexity_scores = {}  # {graph_id: complexity_score}

        # Shared graph management for stability
        self.current_shared_graph = None
        self.policy_updates_with_current_graph = 0
        self.shared_graph_history = []

        # Statistics
        self.total_graphs_seen = 0
        self.pool_refresh_counter = 0
        self.complexity_distribution_stats = {}

        # Logger (will be set externally)
        self.logger = None

        # Initialize difficulty-based pools
        self._initialize_difficulty_pools()

    def set_logger(self, logger):
        """Set logger for curriculum tracking"""
        self.logger = logger

    def _initialize_difficulty_pools(self):
        """Initialize graph pools organized by difficulty levels"""
        try:
            # Load comprehensive dataset for analysis
            smelly_data, labels = self.data_loader.load_dataset(
                max_samples_per_class=1000,  # Large sample for good distribution
                shuffle=True,
                validate_all=False,
                remove_metadata=True
            )

            # Filter smelly graphs only
            smelly_candidates = [data for data, label in zip(smelly_data, labels) if label == 1]

            if self.logger:
                self.logger.log_info(f"🎯 ADAPTIVE DIFFICULTY: Analyzing {len(smelly_candidates)} smelly graphs")

            # Compute complexity scores for all graphs
            complexity_scores = []
            for i, graph in enumerate(smelly_candidates):
                complexity_score = self._compute_complexity_score(graph)
                complexity_scores.append((i, graph, complexity_score))
                self.graph_complexity_scores[i] = complexity_score

            # Sort by complexity
            complexity_scores.sort(key=lambda x: x[2])

            # Create 10 difficulty bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
            num_bins = 10
            graphs_per_bin = len(complexity_scores) // num_bins

            for bin_idx in range(num_bins):
                difficulty_level = bin_idx / (num_bins - 1)  # 0.0 to 1.0

                start_idx = bin_idx * graphs_per_bin
                end_idx = start_idx + graphs_per_bin if bin_idx < num_bins - 1 else len(complexity_scores)

                # Extract graphs for this difficulty level
                bin_graphs = [complexity_scores[i][1] for i in range(start_idx, end_idx)]
                self.difficulty_pools[difficulty_level] = bin_graphs

                # Statistics
                bin_complexities = [complexity_scores[i][2] for i in range(start_idx, end_idx)]
                self.complexity_distribution_stats[difficulty_level] = {
                    'count': len(bin_graphs),
                    'avg_complexity': np.mean(bin_complexities),
                    'min_complexity': min(bin_complexities),
                    'max_complexity': max(bin_complexities),
                    'avg_size': np.mean([g.x.size(0) for g in bin_graphs]),
                    'size_range': [min([g.x.size(0) for g in bin_graphs]), max([g.x.size(0) for g in bin_graphs])]
                }

            if self.logger:
                self.logger.log_info(f"✅ Created {num_bins} difficulty pools:")
                for difficulty, stats in self.complexity_distribution_stats.items():
                    self.logger.log_info(f"   Difficulty {difficulty:.1f}: {stats['count']} graphs, "
                                         f"avg complexity {stats['avg_complexity']:.3f}, "
                                         f"avg size {stats['avg_size']:.1f}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize adaptive difficulty pools: {e}")

    def _compute_complexity_score(self, graph: Data) -> float:
        """Compute normalized complexity score for a graph (0.0 = easiest, 1.0 = hardest)"""
        try:
            num_nodes = graph.x.size(0)
            num_edges = graph.edge_index.size(1)

            # Size component (normalized to 0-1)
            size_score = min(num_nodes / 50.0, 1.0)  # Assume max reasonable size is 50

            # Connectivity component (edges per node, normalized)
            connectivity_score = min((num_edges / max(num_nodes, 1)) / 10.0, 1.0) if num_nodes > 0 else 0

            # Hub strength component (max degree normalized)
            if num_edges > 0:
                # Count degrees
                degrees = torch.bincount(graph.edge_index[0], minlength=num_nodes)
                max_degree = degrees.max().item()
                hub_score = min(max_degree / 20.0, 1.0)  # Assume max reasonable degree is 20
            else:
                hub_score = 0.0

            # Weighted combination
            total_score = (
                    self.adaptive_config.size_weight * size_score +
                    self.adaptive_config.connectivity_weight * connectivity_score +
                    self.adaptive_config.hub_strength_weight * hub_score
            )

            # Clamp to [0, 1]
            return max(0.0, min(1.0, total_score))

        except Exception as e:
            if self.logger:
                self.logger.log_warning(f"Complexity computation failed: {e}")
            return 0.5  # Fallback to medium complexity

    def update_difficulty_based_on_performance(self, episode: int, success_rate: float) -> bool:
        """Update difficulty based on recent performance. Returns True if difficulty changed."""

        # Add to success rate history
        self.success_rate_history.append(success_rate)
        if len(self.success_rate_history) > self.adaptive_config.success_rate_window:
            self.success_rate_history.pop(0)

        # Don't adjust too early
        if (episode < self.adaptive_config.min_episodes_before_adjustment or
                len(self.success_rate_history) < 20):  # Need at least 20 samples
            return False

        # Calculate recent average success rate
        recent_success_rate = np.mean(self.success_rate_history[-50:])  # Last 50 episodes

        old_difficulty = self.current_difficulty
        new_difficulty = old_difficulty

        # Determine if adjustment needed
        if recent_success_rate > self.adaptive_config.difficulty_increase_threshold:
            # Too easy - increase difficulty
            adjustment = self.adaptive_config.adjustment_strength
            new_difficulty = min(old_difficulty + adjustment, self.adaptive_config.max_difficulty)

        elif recent_success_rate < self.adaptive_config.difficulty_decrease_threshold:
            # Too hard - decrease difficulty
            adjustment = -self.adaptive_config.adjustment_strength
            new_difficulty = max(old_difficulty + adjustment, self.adaptive_config.min_difficulty)

        # Apply smoothing to prevent oscillation
        if abs(new_difficulty - old_difficulty) > 0.01:
            smoothed_difficulty = (self.adaptive_config.difficulty_smoothing * old_difficulty +
                                   (1 - self.adaptive_config.difficulty_smoothing) * new_difficulty)

            self.current_difficulty = smoothed_difficulty

            # Log the adjustment
            adjustment_info = {
                'episode': episode,
                'old_difficulty': old_difficulty,
                'new_difficulty': smoothed_difficulty,
                'recent_success_rate': recent_success_rate,
                'target_success_rate': self.adaptive_config.target_success_rate,
                'reason': 'too_easy' if recent_success_rate > self.adaptive_config.difficulty_increase_threshold else 'too_hard'
            }
            self.difficulty_adjustments.append(adjustment_info)

            if self.logger:
                self.logger.log_info(f"🎯 DIFFICULTY ADJUSTED [Episode {episode}]:")
                self.logger.log_info(f"   {old_difficulty:.3f} → {smoothed_difficulty:.3f}")
                self.logger.log_info(f"   Recent success rate: {recent_success_rate:.2%}")
                self.logger.log_info(f"   Target: {self.adaptive_config.target_success_rate:.2%}")
                self.logger.log_info(f"   Reason: Policy is {adjustment_info['reason']}")

            # Add to difficulty history
            self.difficulty_history.append({
                'episode': episode,
                'difficulty': smoothed_difficulty,
                'success_rate': recent_success_rate
            })

            # Reset shared graph to use new difficulty
            self.current_shared_graph = None
            self.policy_updates_with_current_graph = 0

            return True

        return False

    def should_change_shared_graph(self, policy_updates_completed: int) -> bool:
        """Check if should change the shared graph across environments"""
        return (self.current_shared_graph is None or
                self.policy_updates_with_current_graph >= self.adaptive_config.same_graph_policy_updates)

    def get_shared_graph_for_environments(self, num_envs: int, policy_updates_completed: int) -> List[Data]:
        """Get the same graph for all environments based on current difficulty"""

        # Check if we need a new shared graph
        if self.should_change_shared_graph(policy_updates_completed):
            # Find the closest difficulty pool
            available_difficulties = sorted(self.difficulty_pools.keys())
            closest_difficulty = min(available_difficulties,
                                     key=lambda x: abs(x - self.current_difficulty))

            # Select graph from appropriate difficulty pool
            if (closest_difficulty in self.difficulty_pools and
                    self.difficulty_pools[closest_difficulty]):

                selected_graphs = self.difficulty_pools[closest_difficulty]
                self.current_shared_graph = random.choice(selected_graphs)
                self.policy_updates_with_current_graph = 0

                # Track graph selection
                complexity = self._compute_complexity_score(self.current_shared_graph)
                self.shared_graph_history.append({
                    'policy_update': policy_updates_completed,
                    'difficulty_level': closest_difficulty,
                    'target_difficulty': self.current_difficulty,
                    'graph_complexity': complexity,
                    'graph_size': self.current_shared_graph.x.size(0),
                    'graph_edges': self.current_shared_graph.edge_index.size(1)
                })

                if self.logger:
                    self.logger.log_info(f"🔄 NEW SHARED GRAPH [Policy Update {policy_updates_completed}]:")
                    self.logger.log_info(f"   Target difficulty: {self.current_difficulty:.3f}")
                    self.logger.log_info(f"   Selected from pool: {closest_difficulty:.3f}")
                    self.logger.log_info(f"   Graph: {self.current_shared_graph.x.size(0)} nodes, "
                                         f"{self.current_shared_graph.edge_index.size(1)} edges")
                    self.logger.log_info(f"   Complexity score: {complexity:.3f}")

            else:
                # Fallback to any available graph
                for difficulty, graphs in self.difficulty_pools.items():
                    if graphs:
                        self.current_shared_graph = random.choice(graphs)
                        break

        # Return clones for all environments
        if self.current_shared_graph is not None:
            return [self.current_shared_graph.clone().to(self.device) for _ in range(num_envs)]
        else:
            # Ultimate fallback
            raise RuntimeError("No graphs available for shared environment creation")

    def on_policy_update_completed(self, policy_updates_completed: int):
        """Called when a policy update is completed"""
        self.policy_updates_with_current_graph += 1

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptive difficulty statistics"""
        recent_success = self.success_rate_history[-20:] if len(
            self.success_rate_history) >= 20 else self.success_rate_history

        stats = {
            # Current state
            'current_difficulty': self.current_difficulty,
            'target_success_rate': self.adaptive_config.target_success_rate,
            'recent_success_rate': np.mean(recent_success) if recent_success else 0.0,

            # Shared graph info
            'current_shared_graph_size': self.current_shared_graph.x.size(0) if self.current_shared_graph else None,
            'current_shared_graph_complexity': self._compute_complexity_score(
                self.current_shared_graph) if self.current_shared_graph else None,
            'policy_updates_with_current_graph': self.policy_updates_with_current_graph,
            'max_updates_per_graph': self.adaptive_config.same_graph_policy_updates,

            # Adjustment history
            'total_difficulty_adjustments': len(self.difficulty_adjustments),
            'recent_adjustments': self.difficulty_adjustments[-5:] if self.difficulty_adjustments else [],

            # Performance tracking
            'success_rate_history_length': len(self.success_rate_history),
            'difficulty_trend': [entry['difficulty'] for entry in self.difficulty_history[-10:]],
            'success_rate_trend': [entry['success_rate'] for entry in self.difficulty_history[-10:]],

            # Pool information
            'available_difficulty_levels': list(self.difficulty_pools.keys()),
            'graphs_per_difficulty': {k: len(v) for k, v in self.difficulty_pools.items()},
            'complexity_distribution': self.complexity_distribution_stats,

            # Usage stats
            'unique_graphs_used': len(self.shared_graph_history),
            'total_graphs_seen': self.total_graphs_seen
        }

        return stats

    def log_adaptive_progress(self, episode: int, policy_updates: int):
        """Log adaptive curriculum progress"""
        if not self.logger or episode % 100 != 0:
            return

        stats = self.get_adaptive_stats()

        self.logger.log_info(f"🎯 ADAPTIVE DIFFICULTY PROGRESS [Episode {episode}]:")
        self.logger.log_info(f"   Current difficulty: {stats['current_difficulty']:.3f}")
        self.logger.log_info(f"   Recent success rate: {stats['recent_success_rate']:.2%}")
        self.logger.log_info(f"   Target success rate: {stats['target_success_rate']:.2%}")

        if stats['current_shared_graph_size']:
            self.logger.log_info(f"   Current shared graph: {stats['current_shared_graph_size']} nodes "
                                 f"(complexity: {stats['current_shared_graph_complexity']:.3f})")

        self.logger.log_info(
            f"   Updates with current graph: {stats['policy_updates_with_current_graph']}/{stats['max_updates_per_graph']}")
        self.logger.log_info(f"   Total difficulty adjustments: {stats['total_difficulty_adjustments']}")

        if stats['recent_adjustments']:
            last_adj = stats['recent_adjustments'][-1]
            episodes_since = episode - last_adj['episode']
            self.logger.log_info(f"   Last adjustment: {episodes_since} episodes ago "
                                 f"({last_adj['reason']})")

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

class PPOTrainer:
    """🆕 Complete Enhanced PPO trainer with adversarial capabilities"""

    def __init__(self, policy: nn.Module, discriminator: nn.Module, discriminator_monitor,
                 opt_config, disc_config, lr_config, device: torch.device = torch.device('cpu')):
        """Initialize enhanced PPO trainer with all configurations"""

        self.device = device
        self.opt_config = opt_config
        self.disc_config = disc_config
        self.lr_config = lr_config
        self.discriminator_monitor = discriminator_monitor
        self.logger = discriminator_monitor.logger

        # Move models to device
        self.policy = policy.to(device)
        self.discriminator = discriminator.to(device)

        # Create optimizers with learning rate configuration
        self.opt_policy = optim.AdamW(
            policy.parameters(),
            lr=lr_config.policy_initial_lr if hasattr(lr_config, 'policy_initial_lr') else getattr(lr_config,
                                                                                                   'policy_base_lr',
                                                                                                   3e-4),
            eps=opt_config.optimizer_eps,
            weight_decay=opt_config.weight_decay,
            betas=opt_config.betas
        )

        self.opt_disc = optim.AdamW(
            discriminator.parameters(),
            lr=lr_config.discriminator_lr if hasattr(lr_config, 'discriminator_lr') else getattr(lr_config,
                                                                                                 'discriminator_initial_lr',
                                                                                                 2e-4),
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

        # Store base learning rates for structured training
        self.policy_base_lr = self.opt_policy.param_groups[0]['lr']
        self.discriminator_fixed_lr = self.opt_disc.param_groups[0]['lr']

        # Initialize schedulers based on config type
        self._setup_schedulers()

        # Core tracking metrics
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

        # 🆕 FORCE UPDATE TRACKING
        self.force_update_count = 0
        self.consecutive_low_accuracy_episodes = 0
        self.last_force_update_episode = -1
        self.force_update_attempts = 0
        self.force_update_success_count = 0
        self.discriminator_accuracy_history = []
        self.force_update_cooldown_remaining = 0

        # 🆕 ADVERSARIAL TRAINING METRICS
        self.adversarial_update_count = 0
        self.discriminator_collapse_incidents = 0
        self.emergency_interventions_log = []
        self.lr_adjustment_history = []

        # GPU optimization
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self.logger.log_info(f"🔧 PPOTrainer initialized with enhanced adversarial features")

    def _setup_schedulers(self):
        """Setup schedulers based on configuration type"""
        lr_config = self.lr_config

        # Check if we have structured config (discriminator_lr exists)
        if hasattr(lr_config, 'discriminator_lr'):
            # STRUCTURED MODE: Fixed LRs, simple decay
            self.policy_scheduler = None
            self.disc_scheduler = None
            self.structured_mode = True

            self.logger.log_info(f"🏗️  Structured trainer mode activated:")
            self.logger.log_info(f"   Policy LR: {self.policy_base_lr:.2e} (with decay)")
            self.logger.log_info(f"   Discriminator LR: {self.discriminator_fixed_lr:.2e} (fixed)")

        else:
            # ADVANCED MODE: Complex scheduling (legacy support)
            try:
                self.policy_scheduler = optim.lr_scheduler.CyclicLR(
                    self.opt_policy,
                    base_lr=getattr(lr_config, 'policy_base_lr', 1e-4),
                    max_lr=getattr(lr_config, 'policy_max_lr', 3e-4),
                    step_size_up=getattr(lr_config, 'policy_cyclical_step_size', 100),
                    mode=getattr(lr_config, 'policy_cyclical_mode', 'triangular'),
                    gamma=getattr(lr_config, 'policy_gamma', 0.99),
                    cycle_momentum=False
                )

                self.disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt_disc,
                    mode=getattr(lr_config, 'discriminator_monitor_mode', 'max'),
                    factor=getattr(lr_config, 'discriminator_factor', 0.8),
                    patience=getattr(lr_config, 'discriminator_patience', 10),
                    threshold=getattr(lr_config, 'discriminator_threshold', 0.01),
                    threshold_mode=getattr(lr_config, 'discriminator_threshold_mode', 'rel'),
                    cooldown=getattr(lr_config, 'discriminator_cooldown', 5),
                    min_lr=getattr(lr_config, 'discriminator_min_lr', 1e-6),
                    eps=getattr(lr_config, 'discriminator_eps', 1e-8)
                )
            except Exception as e:
                self.logger.log_warning(f"Advanced scheduler setup failed, using structured mode: {e}")
                self.policy_scheduler = None
                self.disc_scheduler = None
                self.structured_mode = True
            else:
                self.structured_mode = False
                self.disc_metric_history = []

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

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy_structured(self, states: List[Data], actions: List[Any],
                                 old_log_probs: torch.Tensor, advantages: torch.Tensor,
                                 returns: torch.Tensor, epochs: int = 4):
        """🆕 Enhanced structured policy update with comprehensive tracking"""

        if not self.structured_mode:
            self.logger.log_warning("update_policy_structured called but not in structured mode")
            return self._update_policy_fallback(states, actions, old_log_probs, advantages, returns)

        self.logger.log_info(f"🎯 Policy update #{self.policy_update_count + 1} - {epochs} epochs")

        # Apply learning rate decay for structured mode
        self._apply_structured_lr_decay()

        # Perform the update with enhanced tracking
        success = self._update_policy_internal(states, actions, old_log_probs, advantages, returns, epochs)

        if success:
            self.policy_update_count += 1
            self.logger.log_info(f"✅ Policy update #{self.policy_update_count} completed successfully")
        else:
            self.logger.log_warning(f"⚠️  Policy update #{self.policy_update_count + 1} had issues")

        return success

    def _apply_structured_lr_decay(self):
        """Apply learning rate decay in structured mode"""
        if not hasattr(self.lr_config, 'policy_decay_frequency'):
            return

        lr_config = self.lr_config
        if (self.policy_update_count > 0 and
                self.policy_update_count % lr_config.policy_decay_frequency == 0):

            old_lr = self.opt_policy.param_groups[0]['lr']
            new_lr = max(
                self.policy_base_lr * (lr_config.policy_decay_factor ** (
                        self.policy_update_count // lr_config.policy_decay_frequency)),
                lr_config.policy_min_lr
            )

            for param_group in self.opt_policy.param_groups:
                param_group['lr'] = new_lr

            # Track LR changes
            self.lr_adjustment_history.append({
                'episode_type': 'policy_decay',
                'old_lr': old_lr,
                'new_lr': new_lr,
                'update_count': self.policy_update_count
            })

            self.logger.log_info(f"🔧 Policy LR decay: {old_lr:.2e} → {new_lr:.2e}")

    def _update_policy_internal(self, states: List[Data], actions: List[Any],
                                old_log_probs: torch.Tensor, advantages: torch.Tensor,
                                returns: torch.Tensor, epochs: int) -> bool:
        """🆕 Enhanced internal policy update with comprehensive error handling"""

        if len(states) == 0:
            self.logger.log_warning("Empty states provided to policy update")
            return False

        policy_losses, entropy_losses, value_losses, kl_divs = [], [], [], []
        total_batches_processed = 0
        failed_batches = 0

        try:
            for epoch in range(epochs):
                indices = torch.randperm(len(states), device=self.device)
                batch_size = min(32, len(states))
                epoch_losses = []
                epoch_batches = 0

                for start in range(0, len(states), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end].cpu().numpy()

                    batch_states = [states[i] for i in batch_idx]
                    batch_actions = [actions[i] for i in batch_idx]
                    batch_old_logp = old_log_probs[batch_idx]
                    batch_adv = advantages[batch_idx]
                    batch_ret = returns[batch_idx]

                    try:
                        # Forward pass through policy
                        batch_data = Batch.from_data_list(batch_states).to(self.device)
                        out = self.policy(batch_data)

                        # Compute log probabilities and entropy
                        logp, ent = self._compute_logp_and_entropy(out, batch_actions)

                        # PPO loss computation
                        ratio = torch.exp(logp - batch_old_logp)
                        surr1 = ratio * batch_adv
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value function loss
                        if 'value' in out:
                            value_preds = out['value'].squeeze()
                            value_loss = F.mse_loss(value_preds, batch_ret)
                        else:
                            # Fallback if no value network
                            value_loss = torch.tensor(0.0, device=self.device)

                        # Total loss
                        total_loss = (
                                policy_loss
                                + self.value_coef * value_loss
                                - self.entropy_coef * ent.mean()
                        )

                        # Backward pass
                        self.opt_policy.zero_grad()
                        total_loss.backward()

                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.max_grad_norm
                        )

                        self.opt_policy.step()

                        # Track metrics
                        epoch_losses.append(total_loss.item())
                        policy_losses.append(policy_loss.item())
                        entropy_losses.append(ent.mean().item())
                        value_losses.append(value_loss.item())

                        with torch.no_grad():
                            kl_divs.append((batch_old_logp - logp).mean().item())

                        epoch_batches += 1
                        total_batches_processed += 1

                    except Exception as e:
                        failed_batches += 1
                        if failed_batches < 5:  # Only log first few failures
                            self.logger.log_warning(f"Policy batch failed: {e}")
                        continue

                # Log epoch progress in structured mode
                if self.structured_mode and epoch_losses:
                    avg_loss = np.mean(epoch_losses)
                    self.logger.log_info(
                        f"   Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f} ({epoch_batches} batches)")

        except Exception as e:
            self.logger.log_error(f"Critical policy update error: {e}")
            return False

        # Update metrics if we have any successful updates
        if policy_losses:
            self.policy_losses.append(np.mean(policy_losses))
            self.policy_metrics.append({
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': np.mean(entropy_losses),
                'kl_divergence': np.mean(kl_divs),
                'batches_processed': total_batches_processed,
                'failed_batches': failed_batches
            })

            # Log summary
            if failed_batches > 0:
                self.logger.log_warning(
                    f"   Policy update: {total_batches_processed} successful, {failed_batches} failed batches")

        else:
            self.logger.log_error("No successful policy batches processed")
            return False

        self.update_count += 1

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return True

    def _update_policy_fallback(self, states: List[Data], actions: List[Any],
                                old_log_probs: torch.Tensor, advantages: torch.Tensor,
                                returns: torch.Tensor):
        """Fallback policy update for non-structured mode"""
        epochs = getattr(self.opt_config, 'policy_epochs', 4)
        success = self._update_policy_internal(states, actions, old_log_probs, advantages, returns, epochs)

        # Update scheduler if in advanced mode
        if not self.structured_mode and self.policy_scheduler:
            self.policy_scheduler.step()

        return success

    def check_discriminator_collapse(self, current_accuracy: float, episode: int) -> bool:
        """🆕 Enhanced discriminator collapse detection with comprehensive tracking"""

        # Add current accuracy to history
        self.discriminator_accuracy_history.append({
            'episode': episode,
            'accuracy': current_accuracy,
            'timestamp': episode
        })

        # Get force update configuration with safe defaults
        min_accuracy = getattr(self.disc_config, 'force_update_accuracy_threshold', 0.58)
        consecutive_threshold = getattr(self.disc_config, 'force_update_consecutive_episodes', 3)
        cooldown_episodes = getattr(self.lr_config, 'force_update_cooldown_episodes', 10)

        # Check if in cooldown period
        if self.force_update_cooldown_remaining > 0:
            self.force_update_cooldown_remaining -= 1
            return False

        # Update consecutive low accuracy counter
        if current_accuracy < min_accuracy:
            self.consecutive_low_accuracy_episodes += 1
        else:
            self.consecutive_low_accuracy_episodes = 0

        # Determine if force update needed
        needs_force_update = (
                self.consecutive_low_accuracy_episodes >= consecutive_threshold and
                episode - self.last_force_update_episode > cooldown_episodes
        )

        if needs_force_update:
            self.discriminator_collapse_incidents += 1

            self.logger.log_warning(f"🚨 DISCRIMINATOR COLLAPSE DETECTED!")
            self.logger.log_warning(f"   Current accuracy: {current_accuracy:.3f}")
            self.logger.log_warning(f"   Consecutive low episodes: {self.consecutive_low_accuracy_episodes}")
            self.logger.log_warning(f"   Threshold: {min_accuracy:.3f}")
            self.logger.log_warning(f"   Episode: {episode}")
            self.logger.log_warning(f"   Collapse incident #{self.discriminator_collapse_incidents}")

        return needs_force_update

    def perform_force_update(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                             generated_graphs: List[Data], episode: int) -> bool:
        """🆕 Enhanced aggressive discriminator force update with comprehensive recovery"""

        self.force_update_count += 1
        self.force_update_attempts += 1
        self.last_force_update_episode = episode
        self.consecutive_low_accuracy_episodes = 0

        # Get force update configuration with safe defaults
        max_epochs = getattr(self.disc_config, 'force_update_max_epochs', 5)
        lr_boost = getattr(self.disc_config, 'force_update_lr_boost', 1.5)
        recovery_threshold = getattr(self.lr_config, 'force_update_recovery_threshold', 0.65)
        cooldown_episodes = getattr(self.lr_config, 'force_update_cooldown_episodes', 10)

        self.logger.log_warning(f"🔄 FORCE UPDATE #{self.force_update_count} - Episode {episode}")
        self.logger.log_info(f"   Max epochs: {max_epochs}")
        self.logger.log_info(f"   LR boost: {lr_boost}x")
        self.logger.log_info(f"   Recovery target: {recovery_threshold:.3f}")

        # Store original learning rate
        original_lr = self.opt_disc.param_groups[0]['lr']
        boosted_lr = min(
            original_lr * lr_boost,
            getattr(self.lr_config, 'force_update_lr_max', 1e-3)
        )

        # Apply LR boost
        for param_group in self.opt_disc.param_groups:
            param_group['lr'] = boosted_lr

        # Track LR adjustment
        self.lr_adjustment_history.append({
            'episode_type': 'force_update_boost',
            'episode': episode,
            'old_lr': original_lr,
            'new_lr': boosted_lr,
            'force_update_count': self.force_update_count
        })

        self.logger.log_info(f"   LR boosted: {original_lr:.2e} → {boosted_lr:.2e}")

        success = False
        recovery_accuracy = 0.0

        try:
            # Log data stats
            self.logger.log_info(
                f"   Training data: {len(smelly_graphs)} smelly, {len(clean_graphs)} clean, {len(generated_graphs)} generated")

            # Perform aggressive training with more epochs
            self._update_discriminator_internal(
                smelly_graphs, clean_graphs, generated_graphs,
                episode, epochs=max_epochs, is_force_update=True
            )

            # Validate recovery
            validation_results = self.discriminator_monitor.validate_performance(episode, detailed=True)
            recovery_accuracy = validation_results.get('accuracy', 0)

            if recovery_accuracy >= recovery_threshold:
                self.force_update_success_count += 1
                self.logger.log_info(f"✅ FORCE UPDATE SUCCESSFUL!")
                self.logger.log_info(f"   Recovery: {recovery_accuracy:.3f} >= {recovery_threshold:.3f}")
                success = True
            else:
                self.logger.log_warning(f"⚠️  FORCE UPDATE PARTIAL")
                self.logger.log_warning(f"   Recovery: {recovery_accuracy:.3f} < {recovery_threshold:.3f}")
                success = False

            # Set cooldown period
            self.force_update_cooldown_remaining = cooldown_episodes

        except Exception as e:
            self.logger.log_error(f"❌ FORCE UPDATE FAILED: {e}")
            success = False

        finally:
            # Restore original learning rate
            for param_group in self.opt_disc.param_groups:
                param_group['lr'] = original_lr

            # Track LR restoration
            self.lr_adjustment_history.append({
                'episode_type': 'force_update_restore',
                'episode': episode,
                'old_lr': boosted_lr,
                'new_lr': original_lr,
                'success': success,
                'recovery_accuracy': recovery_accuracy
            })

            self.logger.log_info(f"   LR restored: {boosted_lr:.2e} → {original_lr:.2e}")

        # Log emergency intervention
        self.emergency_interventions_log.append({
            'episode': episode,
            'force_update_count': self.force_update_count,
            'success': success,
            'recovery_accuracy': recovery_accuracy,
            'original_lr': original_lr,
            'boosted_lr': boosted_lr,
            'epochs_used': max_epochs
        })

        return success

    def update_discriminator_structured(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                                        generated_graphs: List[Data], episode: int, epochs: int = 2):
        """🆕 Enhanced structured discriminator update"""

        if not self.structured_mode:
            self.logger.log_warning("update_discriminator_structured called but not in structured mode")
            return self._update_discriminator_fallback(smelly_graphs, clean_graphs, generated_graphs, episode)

        self.logger.log_info(f"🧠 Discriminator update #{self.discriminator_update_count + 1} - {epochs} epochs")
        self.logger.log_info(
            f"   Data: {len(smelly_graphs)} smelly, {len(clean_graphs)} clean, {len(generated_graphs)} generated")

        # Ensure fixed learning rate in structured mode
        current_lr = self.opt_disc.param_groups[0]['lr']
        if abs(current_lr - self.discriminator_fixed_lr) > 1e-8:
            for param_group in self.opt_disc.param_groups:
                param_group['lr'] = self.discriminator_fixed_lr
            self.logger.log_info(f"   LR reset to fixed: {current_lr:.2e} → {self.discriminator_fixed_lr:.2e}")

        # Perform the update
        success = self._update_discriminator_internal(smelly_graphs, clean_graphs, generated_graphs, episode, epochs)

        if success:
            self.discriminator_update_count += 1
            self.adversarial_update_count += 1

        return success

    def _update_discriminator_internal(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                                       generated_graphs: List[Data], episode: int, epochs: int,
                                       is_force_update: bool = False) -> bool:
        """🆕 Enhanced internal discriminator update with comprehensive tracking"""

        # Pre-update health check
        current_lr = self.opt_disc.param_groups[0]['lr']
        if current_lr < 1e-10:
            self.discriminator_monitor.logger.log_warning(
                f"⚠️  Discriminator LR extremely low ({current_lr:.2e}) - update may be ineffective")

        total_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        epoch_losses = []
        total_batches_processed = 0
        failed_batches = 0

        update_type = "FORCE" if is_force_update else "REGULAR"
        self.logger.log_info(f"🧠 Starting {update_type} discriminator update - {epochs} epochs")

        try:
            for epoch in range(epochs):
                # Determine batch size based on mode and data availability
                available_data_size = min(len(smelly_graphs), len(clean_graphs), len(generated_graphs))

                if self.structured_mode:
                    batch_size = min(
                        self.disc_config.max_samples_per_class // 2,
                        available_data_size
                    )
                else:
                    batch_size = min(
                        self.disc_config.max_samples_per_class,
                        max(self.disc_config.min_samples_per_class, available_data_size)
                    )

                # Force update uses larger batches if possible
                if is_force_update:
                    batch_size = min(self.disc_config.max_samples_per_class, available_data_size)

                if batch_size <= 2:
                    self.discriminator_monitor.logger.log_warning(f"⚠️  Insufficient batch size: {batch_size}")
                    continue

                epoch_batch_losses = []

                try:
                    # Sample data with proper replacement handling
                    def safe_sample(data_list, size):
                        if len(data_list) >= size:
                            return np.random.choice(len(data_list), size, replace=False).tolist()
                        else:
                            return list(range(len(data_list)))

                    smelly_sample = safe_sample(smelly_graphs, batch_size)
                    clean_sample = safe_sample(clean_graphs, batch_size)
                    generated_sample = safe_sample(generated_graphs, batch_size)

                    # Create batches
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

                    # Validate outputs
                    if any(out is None or 'logits' not in out for out in [smelly_out, clean_out, gen_out]):
                        self.discriminator_monitor.logger.log_warning(
                            f"⚠️  Invalid discriminator outputs at epoch {epoch}")
                        failed_batches += 1
                        continue

                    # Prepare labels
                    smelly_labels = torch.ones(len(smelly_batch_data), dtype=torch.long, device=self.device)
                    clean_labels = torch.zeros(len(clean_batch_data), dtype=torch.long, device=self.device)
                    gen_labels = torch.zeros(len(generated_batch_data), dtype=torch.long, device=self.device)

                    # Compute losses
                    loss_smelly = F.cross_entropy(smelly_out['logits'], smelly_labels,
                                                  label_smoothing=self.disc_config.label_smoothing)
                    loss_clean = F.cross_entropy(clean_out['logits'], clean_labels,
                                                 label_smoothing=self.disc_config.label_smoothing)
                    loss_gen = F.cross_entropy(gen_out['logits'], gen_labels,
                                               label_smoothing=self.disc_config.label_smoothing)

                    # Gradient penalty (only in advanced mode and not during force updates)
                    gp_loss = 0.0
                    if not self.structured_mode and not is_force_update:
                        try:
                            gp_loss = self._compute_gradient_penalty(
                                smelly_batch, clean_batch, generated_batch,
                                smelly_out, clean_out, gen_out
                            )
                        except Exception as e:
                            self.logger.log_warning(f"Gradient penalty computation failed: {e}")
                            gp_loss = 0.0

                    # Total discriminator loss
                    disc_loss = (loss_smelly + loss_clean + loss_gen +
                                 self.disc_config.gradient_penalty_lambda * gp_loss)

                    # Backward pass
                    self.opt_disc.zero_grad()
                    disc_loss.backward()

                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.max_grad_norm
                    )

                    self.opt_disc.step()

                    epoch_batch_losses.append(disc_loss.item())
                    total_batches_processed += 1

                    # Calculate metrics if enabled
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

                            # Compute precision, recall, F1
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

                except Exception as e:
                    self.discriminator_monitor.logger.log_warning(
                        f"⚠️  Discriminator training error at epoch {epoch}: {e}")
                    failed_batches += 1
                    continue

                # Track epoch losses
                if epoch_batch_losses:
                    epoch_losses.extend(epoch_batch_losses)
                    avg_epoch_loss = np.mean(epoch_batch_losses)

                    # Log epoch progress
                    if self.structured_mode or is_force_update:
                        loss_str = f"{avg_epoch_loss:.4f}"
                        if is_force_update:
                            loss_str += " [FORCE]"
                        self.logger.log_info(f"   Epoch {epoch + 1}/{epochs}: Loss = {loss_str}")

        except Exception as e:
            self.discriminator_monitor.logger.log_error(f"Critical discriminator update error: {e}")
            return False

        # Process and store metrics
        success = self._process_discriminator_metrics(
            total_metrics, epoch_losses, episode, is_force_update,
            total_batches_processed, failed_batches
        )

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return success

    def _process_discriminator_metrics(self, total_metrics: Dict, epoch_losses: List,
                                       episode: int, is_force_update: bool,
                                       total_batches: int, failed_batches: int) -> bool:
        """Process and log discriminator metrics"""

        if not total_metrics['accuracy']:
            self.logger.log_warning("No valid discriminator metrics collected")
            return False

        # Calculate average metrics
        avg_metrics = {
            'accuracy': np.mean(total_metrics['accuracy']),
            'precision': np.mean(total_metrics['precision']),
            'recall': np.mean(total_metrics['recall']),
            'f1': np.mean(total_metrics['f1']),
            'loss': np.mean(epoch_losses) if epoch_losses else 0.0,
            'batches_processed': total_batches,
            'failed_batches': failed_batches,
            'is_force_update': is_force_update
        }

        # Store metrics
        self.disc_metrics.append(avg_metrics)
        self.disc_losses.append(avg_metrics['loss'])

        # Log results
        current_lr = self.opt_disc.param_groups[0]['lr']
        update_type = "FORCE UPDATE" if is_force_update else "REGULAR UPDATE"

        if self.structured_mode:
            self.logger.log_info(f"🧠 Discriminator {update_type.lower()} completed:")
            self.logger.log_info(f"   Acc: {avg_metrics['accuracy']:.3f}, F1: {avg_metrics['f1']:.3f}")
            self.logger.log_info(f"   Loss: {avg_metrics['loss']:.4f}, LR: {current_lr:.2e}")
            if failed_batches > 0:
                self.logger.log_info(f"   Batches: {total_batches} ok, {failed_batches} failed")
        else:
            self.discriminator_monitor.logger.log_info(
                f"🧠 Discriminator {update_type.lower()} - "
                f"Acc: {avg_metrics['accuracy']:.3f}, F1: {avg_metrics['f1']:.3f}, "
                f"Loss: {avg_metrics['loss']:.4f}, LR: {current_lr:.2e}"
            )

        return True

    def _update_discriminator_fallback(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                                       generated_graphs: List[Data], episode: int):
        """Fallback discriminator update for non-structured mode"""
        epochs = getattr(self.opt_config, 'discriminator_epochs', 4)
        success = self._update_discriminator_internal(smelly_graphs, clean_graphs, generated_graphs, episode, epochs)

        # Update scheduler if in advanced mode
        if not self.structured_mode and self.disc_scheduler:
            avg_metrics = self.disc_metrics[-1] if self.disc_metrics else {'accuracy': 0}
            monitored_metric = avg_metrics.get(getattr(self.lr_config, 'discriminator_monitor_metric', 'accuracy'), 0)
            self.disc_scheduler.step(monitored_metric)

        return success

    def _compute_gradient_penalty(self, smelly_batch: Batch, clean_batch: Batch,
                                  generated_batch: Batch, smelly_out: Dict,
                                  clean_out: Dict, gen_out: Dict) -> float:
        """Compute gradient penalty for WGAN-GP style training"""
        try:
            gp_loss = 0.0
            for batch_data, out in [(smelly_batch, smelly_out), (clean_batch, clean_out), (generated_batch, gen_out)]:
                if batch_data.x.grad is not None:
                    batch_data.x.grad.zero_()

                # Compute gradients
                grads = torch.autograd.grad(
                    outputs=out['logits'].sum(),
                    inputs=batch_data.x,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                # Gradient penalty: (||grad||_2 - 1)^2
                gp_loss += (grads.norm(2, dim=1) - 1).pow(2).mean()

            return gp_loss
        except Exception as e:
            self.logger.log_warning(f"Gradient penalty computation failed: {e}")
            return 0.0

    def _compute_logp_and_entropy(self, policy_out: Dict, actions: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """🆕 Enhanced log probability and entropy computation with robust error handling"""

        log_probs = []
        entropies = []

        try:
            for i, action in enumerate(actions):
                try:
                    # Hub selection probabilities
                    if 'hub_probs' in policy_out and i < policy_out['hub_probs'].size(0):
                        hub_probs = policy_out['hub_probs'][i:i + 1] + 1e-8
                        hub_probs = hub_probs / hub_probs.sum()
                        hub_dist = Categorical(hub_probs)

                        hub_action = min(getattr(action, 'source_node', 0), hub_probs.size(0) - 1)
                        log_prob_hub = hub_dist.log_prob(torch.tensor(hub_action, device=self.device))
                        entropy_hub = hub_dist.entropy()
                    else:
                        log_prob_hub = torch.tensor(-5.0, device=self.device)
                        entropy_hub = torch.tensor(0.1, device=self.device)

                    # Pattern selection probabilities
                    if 'pattern_probs' in policy_out and i < policy_out['pattern_probs'].size(0):
                        pattern_probs = policy_out['pattern_probs'][i] + 1e-8
                        pattern_probs = pattern_probs / pattern_probs.sum()
                        pattern_dist = Categorical(pattern_probs)

                        pattern_action = min(getattr(action, 'pattern', 0), pattern_probs.size(0) - 1)
                        log_prob_pattern = pattern_dist.log_prob(torch.tensor(pattern_action, device=self.device))
                        entropy_pattern = pattern_dist.entropy()
                    else:
                        log_prob_pattern = torch.tensor(-2.0, device=self.device)
                        entropy_pattern = torch.tensor(0.1, device=self.device)

                    # Termination probabilities
                    if 'term_probs' in policy_out and i < policy_out['term_probs'].size(0):
                        term_probs = policy_out['term_probs'][i] + 1e-8
                        term_probs = term_probs / term_probs.sum()
                        term_dist = Categorical(term_probs)

                        term_action = int(getattr(action, 'terminate', False))
                        log_prob_term = term_dist.log_prob(torch.tensor(term_action, device=self.device))
                        entropy_term = term_dist.entropy()
                    else:
                        log_prob_term = torch.tensor(-1.0, device=self.device)
                        entropy_term = torch.tensor(0.1, device=self.device)

                    # Combine probabilities
                    total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
                    total_entropy = entropy_hub + entropy_pattern + entropy_term

                    log_probs.append(total_log_prob)
                    entropies.append(total_entropy)

                except Exception as e:
                    # Fallback for individual action failures
                    log_probs.append(torch.tensor(-5.0, device=self.device))
                    entropies.append(torch.tensor(0.1, device=self.device))

        except Exception as e:
            self.logger.log_error(f"Critical error in log probability computation: {e}")
            # Return fallback values
            fallback_logp = torch.tensor([-5.0] * len(actions), device=self.device)
            fallback_entropy = torch.tensor([0.1] * len(actions), device=self.device)
            return fallback_logp, fallback_entropy

        if log_probs:
            return torch.stack(log_probs), torch.stack(entropies)
        else:
            # Complete fallback
            fallback_logp = torch.tensor([-5.0], device=self.device)
            fallback_entropy = torch.tensor([0.1], device=self.device)
            return fallback_logp, fallback_entropy

    def get_latest_metrics(self) -> Dict[str, Any]:
        """🆕 Enhanced metrics retrieval with comprehensive information"""
        metrics = {}

        # Policy metrics
        if self.policy_metrics:
            latest_policy = self.policy_metrics[-1]
            metrics.update({
                'policy_loss': latest_policy.get('policy_loss', 0.0),
                'policy_entropy': latest_policy.get('entropy', 0.0),
                'policy_kl': latest_policy.get('kl_divergence', 0.0),
                'policy_value_loss': latest_policy.get('value_loss', 0.0),
                'policy_batches_processed': latest_policy.get('batches_processed', 0),
                'policy_failed_batches': latest_policy.get('failed_batches', 0)
            })

        # Discriminator metrics
        if self.disc_metrics:
            latest_disc = self.disc_metrics[-1]
            metrics.update({
                'discriminator_accuracy': latest_disc.get('accuracy', 0.0),
                'discriminator_f1': latest_disc.get('f1', 0.0),
                'discriminator_precision': latest_disc.get('precision', 0.0),
                'discriminator_recall': latest_disc.get('recall', 0.0),
                'discriminator_loss': latest_disc.get('loss', 0.0),
                'discriminator_batches_processed': latest_disc.get('batches_processed', 0),
                'discriminator_failed_batches': latest_disc.get('failed_batches', 0),
                'discriminator_is_force_update': latest_disc.get('is_force_update', False)
            })

        return metrics

    def get_force_update_stats(self) -> Dict[str, Any]:
        """🆕 Comprehensive force update statistics"""
        return {
            'total_force_updates': self.force_update_count,
            'successful_force_updates': self.force_update_success_count,
            'force_update_success_rate': self.force_update_success_count / max(self.force_update_count, 1),
            'consecutive_low_accuracy_episodes': self.consecutive_low_accuracy_episodes,
            'last_force_update_episode': self.last_force_update_episode,
            'force_update_cooldown_remaining': self.force_update_cooldown_remaining,
            'discriminator_collapse_incidents': self.discriminator_collapse_incidents,
            'current_attempts': self.force_update_attempts,
            # Recent accuracy history (last 10)
            'recent_accuracy_history': [entry['accuracy'] for entry in self.discriminator_accuracy_history[-10:]],
            'recent_episodes': [entry['episode'] for entry in self.discriminator_accuracy_history[-10:]],
            # Emergency interventions summary
            'emergency_interventions_count': len(self.emergency_interventions_log),
            'lr_adjustments_count': len(self.lr_adjustment_history)
        }

    def get_training_mode_info(self) -> Dict[str, Any]:
        """🆕 Enhanced training mode information with comprehensive stats"""
        base_info = {
            'structured_mode': self.structured_mode,
            'policy_updates_completed': self.policy_update_count,
            'discriminator_updates_completed': self.discriminator_update_count,
            'adversarial_updates_completed': self.adversarial_update_count,
            'total_updates_completed': self.update_count,
            'emergency_resets': self.emergency_resets,
            'current_policy_lr': self.opt_policy.param_groups[0]['lr'],
            'current_discriminator_lr': self.opt_disc.param_groups[0]['lr']
        }

        # Add force update info
        base_info.update(self.get_force_update_stats())

        # Add performance summaries
        if self.policy_metrics:
            recent_policy_metrics = self.policy_metrics[-5:] if len(self.policy_metrics) >= 5 else self.policy_metrics
            base_info.update({
                'avg_recent_policy_loss': np.mean([m.get('policy_loss', 0) for m in recent_policy_metrics]),
                'avg_recent_policy_entropy': np.mean([m.get('entropy', 0) for m in recent_policy_metrics])
            })

        if self.disc_metrics:
            recent_disc_metrics = self.disc_metrics[-5:] if len(self.disc_metrics) >= 5 else self.disc_metrics
            base_info.update({
                'avg_recent_discriminator_accuracy': np.mean([m.get('accuracy', 0) for m in recent_disc_metrics]),
                'avg_recent_discriminator_f1': np.mean([m.get('f1', 0) for m in recent_disc_metrics])
            })

        return base_info

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """🆕 Get comprehensive training statistics for analysis"""
        return {
            'training_mode_info': self.get_training_mode_info(),
            'latest_metrics': self.get_latest_metrics(),
            'force_update_stats': self.get_force_update_stats(),
            'lr_adjustment_history': self.lr_adjustment_history.copy(),
            'emergency_interventions': self.emergency_interventions_log.copy(),
            'performance_trends': {
                'policy_loss_trend': self.policy_losses[-20:] if len(self.policy_losses) >= 20 else self.policy_losses,
                'discriminator_loss_trend': self.disc_losses[-20:] if len(self.disc_losses) >= 20 else self.disc_losses,
                'discriminator_accuracy_trend': [entry['accuracy'] for entry in
                                                 self.discriminator_accuracy_history[-20:]]
            },
            'system_health': {
                'structured_mode_active': self.structured_mode,
                'policy_lr_in_range': 1e-6 < self.opt_policy.param_groups[0]['lr'] < 1e-2,
                'discriminator_lr_in_range': 1e-6 < self.opt_disc.param_groups[0]['lr'] < 1e-2,
                'recent_failures_low': sum([m.get('failed_batches', 0) for m in self.policy_metrics[-5:]]) < 10,
                'force_updates_controlled': self.force_update_count < 20
            }
        }

    def reset_training_state(self):
        """🆕 Reset training state for fresh start or recovery"""
        self.logger.log_info("🔄 Resetting PPOTrainer state")

        # Reset counters
        self.policy_update_count = 0
        self.discriminator_update_count = 0
        self.adversarial_update_count = 0
        self.force_update_count = 0
        self.consecutive_low_accuracy_episodes = 0
        self.force_update_cooldown_remaining = 0

        # Clear histories but keep some for analysis
        self.policy_losses = []
        self.disc_losses = []
        self.disc_metrics = []
        self.policy_metrics = []

        # Keep emergency logs for analysis
        # self.emergency_interventions_log = []
        # self.lr_adjustment_history = []

        self.logger.log_info("✅ PPOTrainer state reset completed")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, 'device') and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except:
                pass


def create_adaptive_training_environments(env_manager: AdaptiveDifficultyEnvironmentManager,
                                          discriminator: nn.Module, training_config, device,
                                          policy_updates_completed: int) -> List[HubRefactoringEnv]:
    """Create environments with adaptive difficulty curriculum"""

    # Get shared graphs for all environments based on current difficulty
    shared_graphs = env_manager.get_shared_graph_for_environments(
        training_config.num_envs, policy_updates_completed
    )

    # Create environments with the same graph
    envs = []
    for i, graph in enumerate(shared_graphs):
        try:
            env = HubRefactoringEnv(
                initial_data=graph,
                discriminator=discriminator,
                max_steps=training_config.max_steps_per_episode,
                device=device
            )

            # Add adaptive manager reference
            env.adaptive_manager = env_manager
            envs.append(env)

        except Exception as e:
            if env_manager.logger:
                env_manager.logger.log_warning(f"Failed to create adaptive env {i}: {e}")
            continue

    return envs



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


def create_adaptive_training_environments(env_manager: AdaptiveDifficultyEnvironmentManager,
                                          discriminator: nn.Module, training_config, device,
                                          policy_updates_completed: int) -> List[HubRefactoringEnv]:
    """Create environments with adaptive difficulty curriculum"""

    # Get shared graphs for all environments based on current difficulty
    shared_graphs = env_manager.get_shared_graph_for_environments(
        training_config.num_envs, policy_updates_completed
    )

    # Create environments with the same graph
    envs = []
    for i, graph in enumerate(shared_graphs):
        try:
            env = HubRefactoringEnv(
                initial_data=graph,
                discriminator=discriminator,
                max_steps=training_config.max_steps_per_episode,
                device=device
            )

            # Add adaptive manager reference
            env.adaptive_manager = env_manager
            envs.append(env)

        except Exception as e:
            if env_manager.logger:
                env_manager.logger.log_warning(f"Failed to create adaptive env {i}: {e}")
            continue

    return envs


def adaptive_enhanced_reset_environment(env) -> Optional[Data]:
    """Enhanced reset that respects adaptive difficulty curriculum"""
    try:
        if hasattr(env, 'adaptive_manager'):
            # Don't change graph during adaptive phase - just reset environment state
            env.current_data = None
            env.steps = 0
            env.action_history.clear()
            env._cached_hub_score = None
            env._graph_hash = None

            reset_state = env.reset()

            if reset_state is not None and hasattr(reset_state, 'x') and reset_state.x.size(0) > 0:
                return reset_state
            else:
                raise ValueError("Adaptive reset produced invalid state")
        else:
            # Fallback to original reset
            return env.reset()

    except Exception as e:
        if hasattr(env, 'logger'):
            env.logger.log_warning(f"Adaptive environment reset failed: {e}")

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
                        reset_state = adaptive_enhanced_reset_environment(env)
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
    """🎯 Enhanced main training loop with Adaptive Difficulty Curriculum"""

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

    # 🎯 ADAPTIVE DIFFICULTY CONFIGURATION
    adaptive_config = AdaptiveDifficultyConfig(
        target_success_rate=0.65,  # Optimal learning zone
        difficulty_increase_threshold=0.80,  # Too easy if > 80% success
        difficulty_decrease_threshold=0.45,  # Too hard if < 45% success
        same_graph_policy_updates=2,  # Stability through shared graphs
        adjustment_strength=0.08,  # Moderate adjustment speed
        difficulty_smoothing=0.7  # Prevent oscillation
    )

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

    # Enhanced PPO Trainer with force update
    trainer = PPOTrainer(
        policy, discriminator, discriminator_monitor,
        opt_config, disc_config, lr_config, device=device
    )

    # 🎯 ADAPTIVE DIFFICULTY ENVIRONMENT MANAGER
    adaptive_manager = AdaptiveDifficultyEnvironmentManager(
        data_loader, discriminator, adaptive_config, device
    )
    adaptive_manager.set_logger(logger)

    # Create initial environments with adaptive difficulty
    envs = create_adaptive_training_environments(
        adaptive_manager, discriminator, training_config, device, 0
    )

    # Update environment rewards with adversarial config
    for env in envs:
        env.REWARD_SUCCESS = reward_config.REWARD_SUCCESS
        env.REWARD_PARTIAL_SUCCESS = reward_config.REWARD_PARTIAL_SUCCESS
        env.REWARD_FAILURE = reward_config.REWARD_FAILURE
        env.REWARD_STEP = reward_config.REWARD_STEP
        env.REWARD_HUB_REDUCTION = reward_config.REWARD_HUB_REDUCTION
        env.REWARD_INVALID = reward_config.REWARD_INVALID
        env.REWARD_SMALL_IMPROVEMENT = reward_config.REWARD_SMALL_IMPROVEMENT

        # Adversarial reward parameters
        env.adversarial_reward_scale = reward_config.adversarial_reward_scale
        env.adversarial_reward_enabled = reward_config.adversarial_reward_enabled
        env.adversarial_threshold = reward_config.adversarial_threshold
        env.potential_discount = reward_config.potential_discount

    # Training components
    experience_buffer = ExperienceBuffer(adv_config.experience_buffer_size)
    metrics_history = create_progress_tracker()

    # Enhanced tracking for adversarial training
    adversarial_metrics_history = {
        'adversarial_rewards': [],
        'discriminator_probabilities': [],
        'force_updates': [],
        'adversarial_success_rates': []
    }

    # Phase tracking
    warmup_episodes = training_config.warmup_episodes
    policy_update_frequency = training_config.policy_update_frequency
    discriminator_update_frequency = training_config.discriminator_update_frequency

    policy_updates_completed = 0
    discriminator_updates_completed = 0
    discriminator_paused = False
    discriminator_pause_remaining = 0

    logger.log_info("🚀 STARTING ADAPTIVE DIFFICULTY ADVERSARIAL TRAINING")
    logger.log_info(f"📊 Phase Schedule:")
    logger.log_info(f"   Warm-up: Episodes 0-{warmup_episodes}")
    logger.log_info(f"   Policy updates: Every {policy_update_frequency} episodes")
    logger.log_info(
        f"   Discriminator updates: Every {policy_update_frequency * discriminator_update_frequency} episodes")

    logger.log_info(f"🎯 Adaptive Difficulty Features:")
    logger.log_info(f"   Target success rate: {adaptive_config.target_success_rate:.1%}")
    logger.log_info(f"   Difficulty range: {adaptive_config.min_difficulty:.2f} - {adaptive_config.max_difficulty:.2f}")
    logger.log_info(f"   Same graph for {adaptive_config.same_graph_policy_updates} policy updates")
    logger.log_info(f"   Adjustment strength: {adaptive_config.adjustment_strength:.3f}")

    logger.log_info(f"🎁 Adversarial Features:")
    logger.log_info(f"   Reward shaping: {'✅ ENABLED' if reward_config.adversarial_reward_enabled else '❌ DISABLED'}")
    logger.log_info(f"   Force updates: {'✅ ENABLED' if adv_config.enable_force_updates else '❌ DISABLED'}")

    # MAIN TRAINING LOOP with ADAPTIVE DIFFICULTY
    for episode in range(training_config.num_episodes):
        episode_start_time = time.time()

        # Get current training schedule
        schedule_info = get_training_schedule_info(episode)
        current_phase = schedule_info['phase']

        # WARM-UP PHASE: Only rollout collection
        if current_phase == "warmup":
            if episode % 25 == 0:
                logger.log_info(f"🌅 WARM-UP Phase - Episode {episode}/{warmup_episodes}")
                logger.log_info(f"   Collecting experience + building discriminator baseline")
                logger.log_info(f"   Current difficulty: {adaptive_manager.current_difficulty:.3f}")

        # Update discriminator pause countdown
        if discriminator_paused:
            discriminator_pause_remaining -= 1
            if discriminator_pause_remaining <= 0:
                discriminator_paused = False
                logger.log_info("▶️  Discriminator pause ended - resuming updates")

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

        # 🆕 COLLECT ADVERSARIAL METRICS from rollout
        adversarial_rewards_episode = []
        discriminator_probs_episode = []
        adversarial_successes = 0

        for info in rollout_data.get('step_infos', []):
            if 'adversarial_reward' in info:
                adversarial_rewards_episode.append(info['adversarial_reward'])

            if 'adversarial_info' in info:
                adv_info = info['adversarial_info']
                if 'new_prob' in adv_info:
                    discriminator_probs_episode.append(adv_info['new_prob'])
                if adv_info.get('threshold_met', False) and adv_info.get('prob_change', 0) > 0:
                    adversarial_successes += 1

        # Store adversarial metrics
        if adversarial_rewards_episode:
            adversarial_metrics_history['adversarial_rewards'].append(np.mean(adversarial_rewards_episode))
        if discriminator_probs_episode:
            adversarial_metrics_history['discriminator_probabilities'].append(np.mean(discriminator_probs_episode))

        adversarial_success_rate = adversarial_successes / max(len(rollout_data.get('step_infos', [])), 1)
        adversarial_metrics_history['adversarial_success_rates'].append(adversarial_success_rate)

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

        # 🎯 ADAPTIVE DIFFICULTY UPDATE
        if episode > warmup_episodes:
            difficulty_changed = adaptive_manager.update_difficulty_based_on_performance(
                episode, success_rate
            )

            if difficulty_changed:
                # Recreate environments with new difficulty level
                logger.log_info("🔄 Recreating environments for new difficulty level...")
                envs = create_adaptive_training_environments(
                    adaptive_manager, discriminator, training_config, device, policy_updates_completed
                )

                # Update environment rewards (they might have been reset)
                for env in envs:
                    env.REWARD_SUCCESS = reward_config.REWARD_SUCCESS
                    env.REWARD_PARTIAL_SUCCESS = reward_config.REWARD_PARTIAL_SUCCESS
                    env.REWARD_FAILURE = reward_config.REWARD_FAILURE
                    env.REWARD_STEP = reward_config.REWARD_STEP
                    env.REWARD_HUB_REDUCTION = reward_config.REWARD_HUB_REDUCTION
                    env.REWARD_INVALID = reward_config.REWARD_INVALID
                    env.REWARD_SMALL_IMPROVEMENT = reward_config.REWARD_SMALL_IMPROVEMENT
                    env.adversarial_reward_scale = reward_config.adversarial_reward_scale
                    env.adversarial_reward_enabled = reward_config.adversarial_reward_enabled
                    env.adversarial_threshold = reward_config.adversarial_threshold
                    env.potential_discount = reward_config.potential_discount

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

                    # 🎯 NOTIFY ADAPTIVE MANAGER OF POLICY UPDATE
                    adaptive_manager.on_policy_update_completed(policy_updates_completed)

                    # Check if we need to change shared graphs
                    if adaptive_manager.should_change_shared_graph(policy_updates_completed):
                        logger.log_info("🔄 Time to change shared graphs - will update on next rollout")

                else:
                    logger.log_warning("⚠️  Skipping policy update - insufficient valid data")

            except Exception as e:
                logger.log_warning(f"Policy update failed: {e}")

        # 🧠 DISCRIMINATOR UPDATE PHASE (Enhanced with force update logic)
        is_discriminator_update_episode = (
                episode >= warmup_episodes and
                policy_updates_completed > 0 and
                policy_updates_completed % discriminator_update_frequency == 0 and
                not discriminator_paused
        )

        # 🆕 FORCE UPDATE CHECK (independent of regular schedule)
        needs_force_update = False
        if (episode >= warmup_episodes and
                adv_config.enable_force_updates and
                not discriminator_paused):
            # Get current discriminator accuracy from latest validation
            latest_validation = discriminator_monitor.validation_history[
                -1] if discriminator_monitor.validation_history else None
            current_accuracy = latest_validation['metrics']['accuracy'] if latest_validation else baseline_metrics[
                'accuracy']

            needs_force_update = trainer.check_discriminator_collapse(current_accuracy, episode)

        if is_discriminator_update_episode or needs_force_update:
            if needs_force_update and not is_discriminator_update_episode:
                logger.log_warning(f"🚨 EMERGENCY FORCE UPDATE - Episode {episode}")
            else:
                logger.log_info(
                    f"🧠 REGULAR DISCRIMINATOR UPDATE #{discriminator_updates_completed + 1} - Episode {episode}")

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

                    if needs_force_update:
                        # 🆕 PERFORM FORCE UPDATE
                        force_success = trainer.perform_force_update(
                            smelly_graphs[:disc_config.max_samples_per_class],
                            clean_graphs[:disc_config.max_samples_per_class],
                            generated_states[:disc_config.max_samples_per_class],
                            episode
                        )

                        # Track force update
                        adversarial_metrics_history['force_updates'].append({
                            'episode': episode,
                            'success': force_success,
                            'trigger': 'accuracy_collapse'
                        })

                    else:
                        # Regular structured update
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

                else:
                    logger.log_warning("⚠️  Insufficient samples for discriminator update")

            except Exception as e:
                logger.log_warning(f"Discriminator update failed: {e}")

        # 🆕 ENHANCED EPISODE LOGGING with ADAPTIVE DIFFICULTY
        episode_metrics = {
            'reward': episode_reward,
            'success_rate': success_rate,
            'hub_improvement': avg_improvement,
            'phase': current_phase,
            'policy_updates': policy_updates_completed,
            'discriminator_updates': discriminator_updates_completed,
            'discriminator_paused': discriminator_paused,
            # Adversarial metrics
            'adversarial_reward': np.mean(adversarial_rewards_episode) if adversarial_rewards_episode else 0.0,
            'adversarial_success_rate': adversarial_success_rate,
            'avg_discriminator_prob': np.mean(discriminator_probs_episode) if discriminator_probs_episode else 0.5,
            # 🎯 ADAPTIVE DIFFICULTY METRICS
            'current_difficulty': adaptive_manager.current_difficulty,
            'target_success_rate': adaptive_manager.adaptive_config.target_success_rate
        }

        if episode % 25 == 0:
            logger.log_episode_progress(episode, episode_metrics)

            # 🎯 LOG ADAPTIVE DIFFICULTY METRICS
            logger.log_info(f"🎯 Adaptive Difficulty:")
            logger.log_info(f"   Current difficulty: {adaptive_manager.current_difficulty:.3f}")
            logger.log_info(
                f"   Success rate: {success_rate:.2%} (target: {adaptive_manager.adaptive_config.target_success_rate:.2%})")

            # Show recent difficulty trend
            if len(adaptive_manager.difficulty_history) >= 2:
                recent_trend = adaptive_manager.difficulty_history[-2:]
                trend_direction = "↗️" if recent_trend[-1]['difficulty'] > recent_trend[-2]['difficulty'] else "↘️" if \
                recent_trend[-1]['difficulty'] < recent_trend[-2]['difficulty'] else "➡️"
                logger.log_info(
                    f"   Trend: {trend_direction} ({len(adaptive_manager.difficulty_adjustments)} total adjustments)")

            # 🆕 LOG ADVERSARIAL METRICS
            if reward_config.adversarial_reward_enabled and episode > warmup_episodes:
                logger.log_info(f"🎁 Adversarial Metrics:")
                logger.log_info(f"   Avg Adversarial Reward: {episode_metrics['adversarial_reward']:+.3f}")
                logger.log_info(f"   Adversarial Success Rate: {adversarial_success_rate:.1%}")
                logger.log_info(f"   Avg Discriminator Prob: {episode_metrics['avg_discriminator_prob']:.3f}")

        # 🎯 ADAPTIVE DIFFICULTY LOGGING
        if episode % 100 == 0 and episode > warmup_episodes:
            adaptive_manager.log_adaptive_progress(episode, policy_updates_completed)

        # Buffer stats logging
        if episode % 100 == 0:
            buffer_stats = experience_buffer.get_buffer_stats()
            logger.log_info(f"📊 Experience Buffer: {buffer_stats['size']} rollouts, "
                            f"{buffer_stats['total_states']} states, "
                            f"avg reward: {buffer_stats['avg_reward']:.3f}")

        # 🆕 FORCE UPDATE STATS LOGGING
        if episode % 100 == 0 and episode > warmup_episodes:
            force_stats = trainer.get_force_update_stats()
            if force_stats['total_force_updates'] > 0:
                logger.log_info(f"🔄 Force Update Stats:")
                logger.log_info(f"   Total Force Updates: {force_stats['total_force_updates']}")
                logger.log_info(f"   Success Rate: {force_stats['force_update_success_rate']:.1%}")
                logger.log_info(f"   Cooldown Remaining: {force_stats['force_update_cooldown_remaining']}")

        # STRUCTURED PHASE LOGGING with ADAPTIVE INFO
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

            # 🎯 ADAPTIVE DIFFICULTY STATUS
            adaptive_stats = adaptive_manager.get_adaptive_stats()
            logger.log_info(f"🎯 Adaptive Curriculum Status:")
            logger.log_info(f"   Current difficulty: {adaptive_stats['current_difficulty']:.3f}")
            logger.log_info(f"   Recent success: {adaptive_stats['recent_success_rate']:.2%}")
            if adaptive_stats['current_shared_graph_size']:
                logger.log_info(f"   Shared graph: {adaptive_stats['current_shared_graph_size']} nodes "
                                f"(complexity: {adaptive_stats['current_shared_graph_complexity']:.3f})")

        # 🆕 ENHANCED CHECKPOINT SAVING with ADAPTIVE DATA
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
                    # 🆕 ADVERSARIAL METRICS
                    'adversarial_metrics_history': {
                        'adversarial_rewards': adversarial_metrics_history['adversarial_rewards'][-500:],
                        'discriminator_probabilities': adversarial_metrics_history['discriminator_probabilities'][
                                                       -500:],
                        'adversarial_success_rates': adversarial_metrics_history['adversarial_success_rates'][-500:],
                        'force_updates': adversarial_metrics_history['force_updates']
                    },
                    # 🎯 ADAPTIVE DIFFICULTY DATA
                    'adaptive_difficulty_state': {
                        'current_difficulty': adaptive_manager.current_difficulty,
                        'success_rate_history': adaptive_manager.success_rate_history[-200:],  # Last 200 episodes
                        'difficulty_history': adaptive_manager.difficulty_history,
                        'difficulty_adjustments': adaptive_manager.difficulty_adjustments,
                        'shared_graph_history': adaptive_manager.shared_graph_history[-50:],  # Last 50 graphs
                        'adaptive_config': adaptive_config,
                        'complexity_distribution_stats': adaptive_manager.complexity_distribution_stats
                    },
                    'force_update_stats': trainer.get_force_update_stats(),
                    'adaptive_config_used': adaptive_config,
                    'structured_config_used': config
                }

                checkpoint_path = results_dir / f'adaptive_rl_checkpoint_{episode}.pt'
                torch.save(checkpoint, checkpoint_path)

                logger.log_info(f"💾 Adaptive checkpoint saved: {checkpoint_path}")
                logger.log_info(f"   Policy updates: {policy_updates_completed}")
                logger.log_info(f"   Discriminator updates: {discriminator_updates_completed}")
                logger.log_info(f"   Current difficulty: {adaptive_manager.current_difficulty:.3f}")
                logger.log_info(f"   Difficulty adjustments: {len(adaptive_manager.difficulty_adjustments)}")
                logger.log_info(f"   Force updates: {trainer.get_force_update_stats()['total_force_updates']}")

            except Exception as e:
                logger.log_warning(f"Failed to save checkpoint: {e}")

        # Memory cleanup
        if episode % 25 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    # === ENHANCED TRAINING COMPLETION with ADAPTIVE ANALYSIS ===
    logger.log_info("🏁 ADAPTIVE DIFFICULTY ADVERSARIAL TRAINING COMPLETED!")

    # Final adaptive summary
    adaptive_final_stats = adaptive_manager.get_adaptive_stats()

    logger.log_info("📊 FINAL ADAPTIVE TRAINING SUMMARY:")
    logger.log_info(f"   Total episodes: {training_config.num_episodes}")
    logger.log_info(f"   Warm-up episodes: {warmup_episodes}")
    logger.log_info(f"   Training episodes: {training_config.num_episodes - warmup_episodes}")
    logger.log_info(f"   Policy updates completed: {policy_updates_completed}")
    logger.log_info(f"   Discriminator updates completed: {discriminator_updates_completed}")

    # 🎯 ADAPTIVE DIFFICULTY FINAL ANALYSIS
    logger.log_info(f"🎯 Adaptive Difficulty Analysis:")
    logger.log_info(f"   Final difficulty level: {adaptive_final_stats['current_difficulty']:.3f}")
    logger.log_info(f"   Total difficulty adjustments: {adaptive_final_stats['total_difficulty_adjustments']}")
    logger.log_info(f"   Final success rate: {adaptive_final_stats['recent_success_rate']:.2%}")
    logger.log_info(f"   Target success rate: {adaptive_final_stats['target_success_rate']:.2%}")

    if adaptive_final_stats['difficulty_trend']:
        difficulty_change = adaptive_final_stats['difficulty_trend'][-1] - adaptive_final_stats['difficulty_trend'][
            0] if len(adaptive_final_stats['difficulty_trend']) > 1 else 0
        logger.log_info(f"   Overall difficulty progression: {difficulty_change:+.3f}")

    logger.log_info(f"   Unique graphs experienced: {adaptive_final_stats['unique_graphs_used']}")

    # 🆕 ADVERSARIAL SUMMARY (existing code)
    force_stats = trainer.get_force_update_stats()
    logger.log_info(f"🔄 Force Update Summary:")
    logger.log_info(f"   Total force updates: {force_stats['total_force_updates']}")
    logger.log_info(f"   Successful force updates: {force_stats['successful_force_updates']}")
    logger.log_info(f"   Force update success rate: {force_stats['force_update_success_rate']:.1%}")

    if reward_config.adversarial_reward_enabled:
        logger.log_info(f"🎁 Adversarial Reward Summary:")
        if adversarial_metrics_history['adversarial_rewards']:
            avg_adv_reward = np.mean(adversarial_metrics_history['adversarial_rewards'])
            logger.log_info(f"   Average adversarial reward: {avg_adv_reward:+.3f}")

        if adversarial_metrics_history['adversarial_success_rates']:
            avg_adv_success = np.mean(adversarial_metrics_history['adversarial_success_rates'])
            logger.log_info(f"   Average adversarial success rate: {avg_adv_success:.1%}")

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
            # 🆕 ADVERSARIAL FINAL METRICS
            'total_force_updates': force_stats['total_force_updates'],
            'adversarial_reward_enabled': reward_config.adversarial_reward_enabled,
            'final_adversarial_success_rate': adversarial_metrics_history['adversarial_success_rates'][-1] if
            adversarial_metrics_history['adversarial_success_rates'] else 0.0,
            # 🎯 ADAPTIVE FINAL METRICS
            'final_difficulty': adaptive_final_stats['current_difficulty'],
            'total_difficulty_adjustments': adaptive_final_stats['total_difficulty_adjustments'],
            'adaptive_success_rate_achieved': abs(
                adaptive_final_stats['recent_success_rate'] - adaptive_final_stats['target_success_rate']) < 0.1
        }

        logger.log_final_summary(training_config.num_episodes, {'final_performance': final_metrics})

    # Save final enhanced model with adaptive data
    try:
        final_save_path = results_dir / 'final_adaptive_rl_model.pt'

        torch.save({
            'policy_state': policy.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'node_dim': 7,
            'edge_dim': 1,
            'enhanced_training_stats': {
                'total_episodes': training_config.num_episodes,
                'warmup_episodes': warmup_episodes,
                'policy_updates_completed': policy_updates_completed,
                'discriminator_updates_completed': discriminator_updates_completed,
                'final_phase': 'training',
                'update_ratio': policy_updates_completed / max(discriminator_updates_completed, 1),
                # 🆕 ADVERSARIAL STATS
                'adversarial_reward_enabled': reward_config.adversarial_reward_enabled,
                'force_update_stats': force_stats,
                'adversarial_final_metrics': {
                    'mean_adversarial_reward': np.mean(adversarial_metrics_history['adversarial_rewards']) if
                    adversarial_metrics_history['adversarial_rewards'] else 0.0,
                    'mean_adversarial_success_rate': np.mean(
                        adversarial_metrics_history['adversarial_success_rates']) if adversarial_metrics_history[
                        'adversarial_success_rates'] else 0.0
                },
                # 🎯 ADAPTIVE DIFFICULTY FINAL STATS
                'adaptive_difficulty_final_stats': adaptive_final_stats
            },
            'training_history': metrics_history,
            'adversarial_history': adversarial_metrics_history,
            'adaptive_difficulty_complete_history': {
                'difficulty_history': adaptive_manager.difficulty_history,
                'difficulty_adjustments': adaptive_manager.difficulty_adjustments,
                'success_rate_history': adaptive_manager.success_rate_history,
                'shared_graph_history': adaptive_manager.shared_graph_history
            },
            'experience_buffer_final_stats': experience_buffer.get_buffer_stats(),
            'adaptive_config': adaptive_config,
            'enhanced_config': config,
            'trainer_final_metrics': trainer.get_latest_metrics()
        }, final_save_path)

        logger.log_info(f"💾 Final adaptive model saved: {final_save_path}")

    except Exception as e:
        logger.log_error(f"Failed to save final model: {e}")

    # ADAPTIVE TRAINING RECOMMENDATIONS
    logger.log_info("\n" + "=" * 80)
    logger.log_info("🎯 ADAPTIVE DIFFICULTY TRAINING ANALYSIS:")
    logger.log_info("=" * 80)

    # Adaptive-specific analysis
    if adaptive_final_stats['total_difficulty_adjustments'] > 0:
        logger.log_info(f"🎯 ADAPTIVE DIFFICULTY EFFECTIVENESS:")

        # Analyze adjustment pattern
        adjustment_reasons = {}
        for adj in adaptive_manager.difficulty_adjustments:
            reason = adj.get('reason', 'unknown')
            adjustment_reasons[reason] = adjustment_reasons.get(reason, 0) + 1

        logger.log_info(f"   Total adjustments made: {adaptive_final_stats['total_difficulty_adjustments']}")
        for reason, count in adjustment_reasons.items():
            logger.log_info(f"   {reason.replace('_', ' ').title()}: {count} times")

        # Success rate convergence analysis
        target_success = adaptive_final_stats['target_success_rate']
        final_success = adaptive_final_stats['recent_success_rate']
        convergence_quality = 1.0 - abs(target_success - final_success) / target_success

        logger.log_info(f"   Success rate convergence: {convergence_quality:.1%}")
        if convergence_quality > 0.9:
            logger.log_info("   ✅ Excellent convergence to target success rate")
        elif convergence_quality > 0.8:
            logger.log_info("   ✅ Good convergence to target success rate")
        else:
            logger.log_info("   ⚠️  Could improve convergence - consider adjusting parameters")

    else:
        logger.log_info("🎯 ADAPTIVE DIFFICULTY ANALYSIS:")
        logger.log_info("   No difficulty adjustments were needed - initial difficulty was appropriate")

    # Final recommendations
    logger.log_info("🎯 ADAPTIVE TRAINING RECOMMENDATIONS:")

    if adaptive_final_stats['total_difficulty_adjustments'] < 3:
        logger.log_info("   Consider starting with lower initial difficulty for more adaptation")
    elif adaptive_final_stats['total_difficulty_adjustments'] > 20:
        logger.log_info("   Consider larger adjustment_strength for faster convergence")

    if adaptive_final_stats['recent_success_rate'] > 0.8:
        logger.log_info("   Policy learned well - could handle higher difficulty graphs")
    elif adaptive_final_stats['recent_success_rate'] < 0.5:
        logger.log_info("   Policy struggled - consider longer warm-up or easier initial graphs")

    logger.log_info("=" * 80)
    logger.log_info("🏆 ADAPTIVE DIFFICULTY ADVERSARIAL TRAINING COMPLETED SUCCESSFULLY!")
    logger.log_info("=" * 80)


if __name__ == "__main__":
    main()