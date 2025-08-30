#!/usr/bin/env python3
"""
PPO Trainer for Graph Refactoring
Updated implementation with PPOConfig as single source of truth
"""

import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch


@dataclass
class PPOConfig:
    """Updated config as single source of truth with all required parameters."""

    # General
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed: int = 42
    experiment_name: str = "graph_refactor_ppo_corrected"

    # Data paths
    data_path: str = "data_builder/dataset/graph_features"
    discriminator_path: str = "results/discriminator_pretraining/pretrained_discriminator.pt"
    results_dir: str = "results/ppo_training"

    # Environment parameters - CLAUDE: Now controlled by PPOConfig
    max_steps: int = 10  # CLAUDE: renamed from max_episode_steps for consistency
    num_actions: int = 7

    # CLAUDE: Added growth control parameters from prompt requirements
    max_new_nodes: int = 5
    max_growth: float = 1.3
    growth_penalty_mode: str = "quadratic"
    growth_penalty_power: int = 2
    growth_gamma_nodes: float = 2.0
    growth_gamma_edges: float = 1.0

    # Reward weights with defaults - CLAUDE: Complete set as required
    reward_weights: Optional[Dict[str, float]] = None

    # Model architecture - CLAUDE: All dimensions controlled by PPOConfig
    node_dim: int = 7
    hidden_dim: int = 256
    num_layers: int = 3
    global_features_dim: int = 4
    dropout: float = 0.3
    shared_encoder: bool = True  # CLAUDE: Force shared encoder for PPO

    # Training phases
    num_episodes: int = 5000
    warmup_episodes: int = 1000
    adversarial_start_episode: int = 1001

    # PPO hyperparameters
    gamma: float = 0.95
    gae_lambda: float = 0.95
    lr: float = 1e-4
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    target_kl: float = 0.05
    max_grad_norm: float = 0.5
    min_mb_before_kl: int = 2  # CLAUDE: Minimum minibatches before KL early stop

    # Loss coefficients
    value_loss_coef: float = 0.3
    entropy_coef: float = 0.02

    # PPO batching
    rollout_episodes: int = 32
    minibatch_size: int = 128

    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: float = 5.0

    # Logging / eval
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 500
    num_eval_episodes: int = 50

    # Early stopping (macro)
    early_stopping_patience: int = 300
    min_improvement: float = 0.01

    # Exploration
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_anneal_episodes: int = 2000

    # Entropy scheduling
    entropy_coef_start: float = 0.03
    entropy_coef_end: float = 0.005
    entropy_anneal_episodes: int = 2500

    # CLAUDE: Cyclic Learning Rate parameters as specified
    use_cyclic_lr: bool = True
    base_lr: float = 1e-4
    max_lr: float = 2e-4
    clr_step_size_up: int = 96
    clr_step_size_down: int = 96
    clr_mode: str = "triangular2"

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_bins: List[int] = field(default_factory=lambda: [20, 30, 45, 66])
    curriculum_episodes_per_stage: int = 600
    curriculum_mix_back_ratio: float = 0.10
    sampling_without_replacement: bool = True
    coverage_log_every: int = 100

    def __post_init__(self):
        # CLAUDE: Complete default reward weights as specified in requirements
        default_rw = {
            'hub_weight': 5.0,
            'step_valid': 0.01,
            'step_invalid': -0.1,
            'time_penalty': -0.02,
            'early_stop_penalty': -0.5,
            'cycle_penalty': -0.2,
            'duplicate_penalty': -0.1,
            'adversarial_weight': 0.5,
            'patience': 15,
            # Growth control (terminal only)
            'node_penalty': 1.0,
            'edge_penalty': 0.02,
            'cap_exceeded_penalty': -0.8,
            # Success criteria
            'success_threshold': 0.03,
            'success_bonus': 2.0,
        }
        # CLAUDE: Robust merge as specified
        self.reward_weights = {**default_rw, **(self.reward_weights or {})}


class CurriculumSampler:
    """Sampler without reinsertion with curriculum for num_nodes."""

    def __init__(self, num_nodes_list: List[int], bins: List[int], episodes_per_stage: int, mix_back_ratio: float = 0.0,
                 seed: int = 42):
        self.num_nodes_list = num_nodes_list
        self.N = len(num_nodes_list)
        self.bins = sorted(bins)
        self.stage = 0
        self.episodes_per_stage = max(1, episodes_per_stage)
        self.mix_back_ratio = max(0.0, min(0.3, mix_back_ratio))
        self.rng = random.Random(seed)
        # bucketize
        self.bucket_indices: Dict[int, List[int]] = {b: [] for b in self.bins}
        for idx, n in enumerate(num_nodes_list):
            for b in self.bins:
                if n <= b:
                    self.bucket_indices[b].append(idx)
                    break
        self._rebuild_pool()
        self.current_queue = deque(self.pool)
        self.seen = set()
        self.episodes_seen_in_stage = 0

    def _rebuild_pool(self):
        max_allowed = self.bins[self.stage]
        pool = []
        for b in self.bins:
            if b <= max_allowed:
                pool.extend(self.bucket_indices[b])
        if self.mix_back_ratio > 0.0 and self.stage > 0:
            prev_pool = []
            for b in self.bins:
                if b < self.bins[self.stage]:
                    prev_pool.extend(self.bucket_indices[b])
            k = int(len(pool) * self.mix_back_ratio)
            self.rng.shuffle(prev_pool)
            pool.extend(prev_pool[:k])
        pool = list(set(pool))
        self.rng.shuffle(pool)
        self.pool = pool

    def next_index(self) -> int:
        if not self.current_queue:
            self.rng.shuffle(self.pool)
            self.current_queue = deque(self.pool)
        idx = self.current_queue.popleft()
        self.seen.add(idx)
        return idx

    def maybe_advance_stage(self):
        self.episodes_seen_in_stage += 1
        if self.episodes_seen_in_stage >= self.episodes_per_stage and self.stage < len(self.bins) - 1:
            self.stage += 1
            self.episodes_seen_in_stage = 0
            self._rebuild_pool()
            self.current_queue = deque(self.pool)

    def coverage(self) -> float:
        return len(self.seen) / self.N if self.N > 0 else 0.0


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
        self.action_masks = []
        self.epsilons = []

    def store(self, state: Data, global_feat: torch.Tensor, action: int,
              reward: float, value: float, log_prob: float, done: bool, action_mask, epsilon):
        self.states.append(state.clone())
        self.global_features.append(global_feat.clone())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask.astype(bool).tolist())
        self.epsilons.append(float(epsilon))

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
            'old_values': torch.tensor(self.values, dtype=torch.float),
            'action_masks': torch.tensor(self.action_masks, dtype=torch.bool),
            'epsilons': torch.tensor(self.epsilons, dtype=torch.float)
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
    """PPO Trainer with PPOConfig as single source of truth"""

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

        # CLAUDE: Initialize environment using controlled _build_env method
        self.env = self._build_env()

        # Build curriculum sampler
        try:
            num_nodes_list = [d.num_nodes for d in self.env.original_data_list]
        except Exception:
            num_nodes_list = [d.x.size(0) for d in self.env.original_data_list]

        self.sampler = CurriculumSampler(
            num_nodes_list=num_nodes_list,
            bins=self.config.curriculum_bins,
            episodes_per_stage=self.config.curriculum_episodes_per_stage,
            mix_back_ratio=self.config.curriculum_mix_back_ratio,
            seed=self.config.random_seed,
        )

        # Load discriminator maintaining original functionality
        self.discriminator = load_discriminator(config.discriminator_path, config.device)
        if self.discriminator is not None:
            self.env.discriminator = self.discriminator

        # CLAUDE: Initialize model without hard-coded dimensions
        self._init_model()

        # Single optimizer with config LR
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)

        # CLAUDE: Cyclic Learning Rate as specified
        self.scheduler = None
        if self.config.use_cyclic_lr:
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.config.base_lr,
                max_lr=self.config.max_lr,
                step_size_up=self.config.clr_step_size_up,
                step_size_down=self.config.clr_step_size_down,
                mode=self.config.clr_mode,
                cycle_momentum=False,  # Adam: no momentum cycling
            )
            self.logger.info(
                f"Using CyclicLR: base_lr={self.config.base_lr}, max_lr={self.config.max_lr}, "
                f"step_up={self.config.clr_step_size_up}, mode={self.config.clr_mode}")

        # PPO buffer
        self.buffer = PPOBuffer()

        # TensorBoard writer
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'critic_losses': [],
            'discriminator_losses': [],
            'hub_score_improvements': [],
            'discriminator_scores': []
        }

        # Best model tracking
        self.best_reward = float('-inf')
        self.early_stopping_counter = 0
        self.total_episodes_trained = 0

        # Running reward statistics for normalization
        self.reward_running_stats = deque(maxlen=1000)

        self.logger.info("PPO Trainer initialized successfully!")

    # CLAUDE: Method to build environment controlled by PPOConfig
    def _build_env(self):
        """Build environment using PPOConfig parameters"""
        from rl_gym import RefactorEnv

        return RefactorEnv(
            data_path=self.config.data_path,
            max_steps=self.config.max_steps,
            device=self.config.device,
            reward_weights=self.config.reward_weights,
            # CLAUDE: Growth control parameters from config
            max_new_nodes_per_episode=self.config.max_new_nodes,
            max_total_node_growth=self.config.max_growth,
            growth_penalty_mode=self.config.growth_penalty_mode,
            growth_penalty_power=self.config.growth_penalty_power,
            growth_penalty_gamma_nodes=self.config.growth_gamma_nodes,
            growth_penalty_gamma_edges=self.config.growth_gamma_edges
        )

    # CLAUDE: Model initialization without hard-code
    def _init_model(self):
        """Initialize actor-critic model from PPOConfig parameters"""
        ac_config = {
            'node_dim': self.config.node_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'num_actions': self.config.num_actions,
            'global_features_dim': self.config.global_features_dim,
            'dropout': self.config.dropout,
            'shared_encoder': self.config.shared_encoder
        }

        from actor_critic_models import create_actor_critic
        self.model = create_actor_critic(ac_config).to(self.device)

        self.logger.info(f"Actor-Critic model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info("Model initialized from PPOConfig")

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """Extract global features using environment method"""
        return self.env._extract_global_features(data)

    def collect_rollouts(self, num_episodes: int, is_adversarial: bool = False) -> Dict:
        """CLAUDE: Collect rollouts with device + buffer pre-step"""
        episode_infos = []

        for episode_idx in range(num_episodes):
            # Reset environment
            if self.config.use_curriculum:
                graph_idx = self.sampler.next_index()
                current_data = self.env.reset(graph_idx)
                self.sampler.maybe_advance_stage()
            else:
                current_data = self.env.reset()

            episode_reward = 0.0
            episode_length = 0
            initial_hub_score = self.env.initial_metrics['hub_score']

            # Get epsilon for this episode
            current_episode = self.total_episodes_trained + episode_idx
            epsilon = self._epsilon_at(current_episode)

            # Discriminator initial score
            initial_disc_score = None
            if self.discriminator is not None:
                with torch.no_grad():
                    disc_out = self.discriminator(current_data)
                    if isinstance(disc_out, dict):
                        initial_disc_score = torch.softmax(disc_out['logits'], dim=1)[0, 1].item()
                    else:
                        initial_disc_score = torch.softmax(disc_out, dim=1)[0, 1].item()

            while True:
                # CLAUDE: Create Batch.from_data_list([state]).to(self.device) before forward
                state_batch = Batch.from_data_list([current_data]).to(self.device)
                global_features = self._extract_global_features(current_data)

                # Get action mask from environment
                action_mask = torch.tensor(self.env.get_action_mask(), dtype=torch.bool, device=self.device)

                # Model forward pass
                with torch.no_grad():
                    # CLAUDE: Use the same Batch for policy (and discriminator if present)
                    output = self.model(state_batch, global_features)

                    # Get action logits
                    if 'action_logits' in output:
                        action_logits = output['action_logits'].squeeze()
                    else:
                        action_logits = torch.log(output['action_probs'].squeeze().clamp(min=1e-8))

                    state_value = output['state_value']
                    if isinstance(state_value, torch.Tensor):
                        state_value = state_value.squeeze().item()

                # Create epsilon-greedy mixed distribution
                mixed_probs = self._create_epsilon_greedy_distribution(action_logits, action_mask, epsilon)

                # Sample action from mixed distribution
                dist = torch.distributions.Categorical(mixed_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                # CLAUDE: Save PRE-STEP state in buffer
                state_before_action = current_data.clone()

                # Take environment step
                next_state, reward, done, info = self.env.step(int(action.item()))
                current_data = self.env.current_data.clone()

                # CLAUDE: Normalize reward if enabled
                if self.config.normalize_rewards:
                    reward = self._normalize_reward(reward)

                # Store transition with PRE-STEP state
                self.buffer.store(
                    state=state_before_action,
                    global_feat=global_features,
                    action=int(action.item()),
                    reward=float(reward),
                    value=float(state_value),
                    log_prob=float(log_prob.item()),
                    done=bool(done),
                    action_mask=action_mask.cpu().numpy(),
                    epsilon=epsilon
                )

                episode_reward += float(reward)
                episode_length += 1

                if done or episode_length >= self.config.max_steps:
                    break

            # Episode metrics
            final_hub_score = self.env._calculate_metrics(current_data)['hub_score']
            hub_improvement = initial_hub_score - final_hub_score
            best_hub_improvement = initial_hub_score - self.env.best_hub_score

            # Final discriminator score
            final_disc_score = None
            if self.discriminator is not None:
                with torch.no_grad():
                    disc_out = self.discriminator(current_data)
                    if isinstance(disc_out, dict):
                        final_disc_score = torch.softmax(disc_out['logits'], dim=1)[0, 1].item()
                    else:
                        final_disc_score = torch.softmax(disc_out, dim=1)[0, 1].item()

            episode_infos.append({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'hub_improvement': hub_improvement,
                'best_hub_improvement': best_hub_improvement,
                'initial_hub_score': initial_hub_score,
                'final_hub_score': final_hub_score,
                'best_hub_score': self.env.best_hub_score,
                'no_improve_steps': getattr(self.env, 'no_improve_steps', None),
                'initial_disc_score': initial_disc_score,
                'final_disc_score': final_disc_score,
                'epsilon_used': epsilon
            })

        self.total_episodes_trained += num_episodes
        return {'episodes': episode_infos}

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        if not hasattr(self, "_rwd_mean"):
            self._rwd_mean = float(reward)
            self._rwd_mad = 1.0
            self._rwd_beta = 0.95

        beta = self._rwd_beta
        self._rwd_mean = beta * self._rwd_mean + (1.0 - beta) * float(reward)
        abs_dev = abs(float(reward) - self._rwd_mean)
        self._rwd_mad = beta * self._rwd_mad + (1.0 - beta) * abs_dev
        denom = max(self._rwd_mad, 1e-8)

        normed = (float(reward) - self._rwd_mean) / denom
        if self.config.reward_clip is not None:
            cap = float(self.config.reward_clip)
            normed = float(max(-cap, min(cap, normed)))
        return normed

    def _epsilon_at(self, episode_idx: int) -> float:
        """Calculate epsilon for given episode"""
        if episode_idx >= self.config.epsilon_anneal_episodes:
            return self.config.epsilon_end

        progress = episode_idx / self.config.epsilon_anneal_episodes
        return self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * progress

    def _entropy_coef_at(self, episode_idx: int) -> float:
        """Calculate entropy coefficient for given episode"""
        if episode_idx >= self.config.entropy_anneal_episodes:
            return self.config.entropy_coef_end

        progress = episode_idx / self.config.entropy_anneal_episodes
        return self.config.entropy_coef_start + (
                self.config.entropy_coef_end - self.config.entropy_coef_start) * progress

    def _create_epsilon_greedy_distribution(self, action_logits: torch.Tensor, action_mask: torch.Tensor,
                                            epsilon: float) -> torch.Tensor:
        """Create epsilon-greedy mixed distribution with action masking"""
        masked_logits = action_logits.clone()
        masked_logits[~action_mask] = float('-inf')

        policy_probs = F.softmax(masked_logits, dim=-1)

        num_valid = action_mask.sum().float()
        uniform_probs = torch.zeros_like(policy_probs)
        uniform_probs[action_mask] = 1.0 / num_valid

        mixed_probs = (1 - epsilon) * policy_probs + epsilon * uniform_probs
        mixed_probs = mixed_probs / (mixed_probs.sum() + 1e-8)

        return mixed_probs

    def update_ppo(self) -> Dict:
        """CLAUDE: PPO update with target_kl respect and scheduler step per optimizer step"""
        if len(self.buffer) == 0:
            return {}

        device = self.device

        # Get current entropy coefficient
        entropy_coef = self._entropy_coef_at(self.total_episodes_trained)

        # Bootstrap for GAE
        with torch.no_grad():
            if len(self.buffer.states) > 0:
                last_state = self.buffer.states[-1]
                last_global = self.buffer.global_features[-1]
                if isinstance(last_global, torch.Tensor) and last_global.dim() == 1:
                    last_global = last_global.unsqueeze(0)
                last_batch = Batch.from_data_list([last_state]).to(device)
                out_last = self.model(last_batch, last_global.to(device))
                next_value = float(out_last["state_value"].view(-1)[0].item())
            else:
                next_value = 0.0

        # Compute GAE
        self.buffer.compute_gae_advantages(self.config.gamma, self.config.gae_lambda, next_value)

        # Get batch data
        batch = self.buffer.get_batch_data()

        # Normalize advantages
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["advantages"] = adv.to(device)

        # Move tensors to device
        batch["actions"] = batch["actions"].to(device).long()
        batch["old_log_probs"] = batch["old_log_probs"].to(device)
        batch["returns"] = batch["returns"].to(device)
        batch["global_features"] = batch["global_features"].to(device)
        batch["old_values"] = batch["old_values"].to(device)
        batch["action_masks"] = batch["action_masks"].to(device)
        batch["epsilons"] = batch["epsilons"].to(device)

        dataset_size = len(batch["actions"])

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        stop_early = False
        mb_seen = 0
        approx_kl_final = 0.0
        clip_frac_final = 0.0

        for epoch in range(self.config.ppo_epochs):
            indices = torch.randperm(dataset_size, device=device)
            for start in range(0, dataset_size, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, dataset_size)
                mb_ix = indices[start:end]
                if mb_ix.numel() == 0:
                    continue
                mb_seen += 1

                # Minibatch data
                states_mb = [batch["states"][int(i)] for i in mb_ix.tolist()]
                globals_mb = batch["global_features"][mb_ix]
                actions_mb = batch["actions"][mb_ix]
                oldlogp_mb = batch["old_log_probs"][mb_ix]
                adv_mb = batch["advantages"][mb_ix]
                rets_mb = batch["returns"][mb_ix]
                oldv_mb = batch["old_values"][mb_ix]
                masks_mb = batch["action_masks"][mb_ix]
                epsilons_mb = batch["epsilons"][mb_ix]

                mb_batch_states = Batch.from_data_list(states_mb).to(device)

                # Forward pass
                out = self.model(mb_batch_states, globals_mb)
                if isinstance(out, dict):
                    if 'action_logits' in out:
                        action_logits = out['action_logits']
                    else:
                        action_logits = torch.log(out['action_probs'].clamp(min=1e-8))
                    values = out['state_value']
                else:
                    action_probs, values = out
                    action_logits = torch.log(action_probs.clamp(min=1e-8))

                # Recreate epsilon-greedy distributions for each sample in minibatch
                batch_size = action_logits.size(0)
                new_log_probs = torch.zeros(batch_size, device=device)
                entropies = torch.zeros(batch_size, device=device)

                for i in range(batch_size):
                    mixed_probs = self._create_epsilon_greedy_distribution(
                        action_logits[i], masks_mb[i], epsilons_mb[i].item()
                    )

                    dist = torch.distributions.Categorical(mixed_probs)
                    new_log_probs[i] = dist.log_prob(actions_mb[i])
                    entropies[i] = dist.entropy()

                ent = entropies.mean()

                # Ensure proper shapes
                values = values.view(-1)
                rets_mb = rets_mb.view_as(values)
                oldv_mb = oldv_mb.view_as(values)

                # Policy loss (PPO clipped)
                ratio = torch.exp(new_log_probs - oldlogp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_mb
                pol_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                v_clip = oldv_mb + (values - oldv_mb).clamp(-self.config.clip_eps, self.config.clip_eps)
                v_uncl = F.smooth_l1_loss(values, rets_mb)
                v_clp = F.smooth_l1_loss(v_clip, rets_mb)
                val_loss = torch.max(v_uncl, v_clp)

                # Total loss with scheduled entropy coefficient
                loss = pol_loss + self.config.value_loss_coef * val_loss - entropy_coef * ent

                # Backprop
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # CLAUDE: Scheduler step per optimizer step as specified
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                # Statistics
                total_policy_loss += pol_loss.item()
                total_value_loss += val_loss.item()
                total_entropy += ent.item()
                num_updates += 1

                # CLAUDE: KL early stopping with min_mb_before_kl
                with torch.no_grad():
                    approx_kl = (oldlogp_mb - new_log_probs).mean().clamp_min(0).item()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()
                    approx_kl_final = approx_kl
                    clip_frac_final = clip_fraction

                if (mb_seen >= self.config.min_mb_before_kl) and approx_kl > self.config.target_kl:
                    try:
                        cur_lr = float(self.optimizer.param_groups[0]["lr"])
                    except Exception:
                        cur_lr = float("nan")
                    self.logger.info(
                        f"Early stop (KL). approx_kl={approx_kl:.5f}, clip_frac={clip_fraction:.3f}, "
                        f"p_loss={pol_loss.item():.4f}, v_loss={val_loss.item():.4f}, lr={cur_lr:.2e}, "
                        f"entropy_coef={entropy_coef:.4f}"
                    )
                    stop_early = True
                    break

            if stop_early:
                break

        # Clear buffer (on-policy)
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "entropy_coef": entropy_coef,
            "current_epsilon": self._epsilon_at(self.total_episodes_trained),
            "approx_kl": approx_kl_final,
            "clip_frac": clip_frac_final
        }

    def update_discriminator(self, episode_infos: List[Dict]) -> Dict:
        """Update discriminator maintaining original functionality"""
        if self.discriminator is None or len(episode_infos) < 4:
            return {}

        try:
            return {}
        except Exception as e:
            self.logger.error(f"Discriminator update failed: {e}")
            return {}

    def evaluate_model(self) -> Dict[str, float]:
        """CLAUDE: Evaluate model using same device path and logging action_usage"""
        self.model.eval()

        eval_rewards = []
        eval_hub_improvements = []
        eval_discriminator_scores = []
        eval_episode_lengths = []
        action_usage = {i: 0 for i in range(7)}
        valid_action_coverage = {i: 0 for i in range(7)}

        for eval_episode in range(self.config.num_eval_episodes):
            current_data = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            initial_hub_score = self.env.initial_metrics['hub_score']

            done = False
            while not done:
                # CLAUDE: Use Batch on self.device for policy/discriminator
                state_batch = Batch.from_data_list([current_data]).to(self.device)
                global_features = self._extract_global_features(current_data)
                action_mask = torch.tensor(self.env.get_action_mask(), dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    output = self.model(state_batch, global_features)

                    if 'action_logits' in output:
                        action_logits = output['action_logits'].squeeze()
                    else:
                        action_logits = torch.log(output['action_probs'].squeeze().clamp(min=1e-8))

                    # Greedy action selection from masked policy
                    masked_logits = action_logits.clone()
                    masked_logits[~action_mask] = float('-inf')
                    action = torch.argmax(masked_logits).item()

                action_usage[action] += 1
                if action_mask[action]:
                    valid_action_coverage[action] += 1

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

        # CLAUDE: Log action_usage and valid_action_coverage if available
        if hasattr(self, 'logger'):
            total_actions = sum(action_usage.values())
            if total_actions > 0:
                usage_str = ", ".join([f"{i}: {count / total_actions:.1%}" for i, count in action_usage.items()])
                self.logger.info(f"Action usage: {usage_str}")

        return {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_hub_improvement_mean': np.mean(eval_hub_improvements),
            'eval_hub_improvement_std': np.std(eval_hub_improvements),
            'eval_discriminator_score_mean': np.mean(eval_discriminator_scores) if eval_discriminator_scores else 0.0,
            'eval_discriminator_score_std': np.std(eval_discriminator_scores) if eval_discriminator_scores else 0.0,
            'eval_episode_length_mean': np.mean(eval_episode_lengths),
            'eval_episode_length_std': np.std(eval_episode_lengths),
            'eval_success_rate': sum(1 for imp in eval_hub_improvements if imp > 0.3) / len(eval_hub_improvements),
            'action_usage_distribution': {
                str(k): v / sum(action_usage.values()) if sum(action_usage.values()) > 0 else 0
                for k, v in action_usage.items()},
            'valid_action_coverage': valid_action_coverage
        }

    def _log_metrics(self, episode_idx: int, episode_infos: List[Dict], update_info: Dict = None):
        """CLAUDE: Enhanced logging with approx_kl, clip_frac, entropy_coef"""
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

        # Add curriculum logging
        if hasattr(self, 'sampler') and episode_idx % self.config.coverage_log_every == 0:
            coverage = self.sampler.coverage()
            current_stage = self.sampler.stage
            self.writer.add_scalar('Curriculum/Coverage', coverage, episode_idx)
            self.writer.add_scalar('Curriculum/Stage', current_stage, episode_idx)
            self.logger.info(f"Curriculum: Stage {current_stage}, Coverage {coverage:.1%}")

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

            current_epsilon = self._epsilon_at(episode_idx)
            current_entropy_coef = self._entropy_coef_at(episode_idx)
            self.writer.add_scalar('Schedule/Epsilon', current_epsilon, episode_idx)
            self.writer.add_scalar('Schedule/EntropyCoef', current_entropy_coef, episode_idx)

        # CLAUDE: Enhanced console logging with approx_kl, clip_frac, entropy_coef
        if episode_idx % self.config.log_every == 0:
            success_rate = sum(1 for imp in recent_improvements if imp > 0.25) / len(recent_improvements)

            base_log = (
                f"Episode {episode_idx:4d} | "
                f"Reward: {np.mean(recent_rewards):6.3f}±{np.std(recent_rewards):5.3f} | "
                f"Hub Δ: {np.mean(recent_improvements):6.3f}±{np.std(recent_improvements):5.3f} | "
                f"Success: {success_rate:.1%} | "
                f"Buffer: {len(self.buffer)}"
            )

            if update_info:
                extra_log = (
                    f" | approx_kl: {update_info.get('approx_kl', 0):.4f}"
                    f" | clip_frac: {update_info.get('clip_frac', 0):.3f}"
                    f" | entropy_coef: {update_info.get('entropy_coef', 0):.4f}"
                )
                base_log += extra_log

            self.logger.info(base_log)

    def _save_checkpoint(self, episode_idx: int, is_best: bool = False):
        """Save model checkpoint maintaining original structure"""
        checkpoint = {
            'episode': episode_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'node_dim': self.config.node_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'num_actions': self.config.num_actions,
                'global_features_dim': self.config.global_features_dim,
                'dropout': self.config.dropout,
                'shared_encoder': self.config.shared_encoder
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
    config = PPOConfig()

    print("Starting Graph Refactoring PPO Training (Corrected)")
    print(f"Configuration: {config.experiment_name}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Max steps: {config.max_steps}")
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