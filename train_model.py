#!/usr/bin/env python3
"""
Enhanced GCPN Training Script for Hub-like Dependency Refactoring

Key Features:
- Enhanced Policy Network with hub-aware attention
- Advanced Environment with pattern-based actions
- Discriminator-based adversarial learning
- PPO + GAE optimization
- Curriculum learning with sophisticated progression
- Comprehensive monitoring and debugging
"""

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.preprocessing import RobustScaler
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

# Import your components with fallbacks
try:
    from rl_gym import EnhancedDependencyRefactorEnv
except ImportError:
    print("Warning: EnhancedDependencyRefactorEnv not found. Please ensure the module is available.")
    exit(1)

try:
    from discriminator import MultiScaleGCNDiscriminator, HubAwareLoss
except ImportError:
    print("Warning: Discriminator modules not found. Please ensure the module is available.")
    exit(1)

try:
    from policy_network import EnhancedGCPNPolicyNetwork
except ImportError:
    print("Warning: EnhancedGCPNPolicyNetwork not found. Please ensure the module is available.")
    exit(1)

try:
    from normalizer import NormalizeAttrs
except ImportError:
    print("Warning: NormalizeAttrs not found. Will use fallback normalizer.")
    NormalizeAttrs = None


# =============================
# Enhanced Configuration
# =============================

class EnhancedTrainingConfig:
    """Enhanced training configuration with your latest components"""

    # Device and reproducibility
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PPO hyperparameters (optimized for your environment)
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    LR_ACTOR = 2e-4
    LR_CRITIC = 2e-4
    LR_DISCRIMINATOR = 1e-4

    # Training parameters
    UPDATE_EPOCHS = 4
    MINIBATCH_SIZE = 32 if torch.cuda.is_available() else 16  # Larger batches on GPU
    ROLLOUT_STEPS = 2048 if torch.cuda.is_available() else 1024  # More steps on GPU
    TOTAL_UPDATES = 3000
    BETA_ENTROPY = 0.01
    VALUE_LOSS_COEF = 0.5

    # Environment settings (using your enhanced environment)
    NUM_ENVS = 16 if torch.cuda.is_available() else 8  # More envs on GPU
    ENV_MAX_STEPS = 50

    # Adversarial learning parameters
    DISC_UPDATE_FREQ = 5
    ADVERSARIAL_WEIGHT = 0.1  # Weight for adversarial loss
    HUB_LOSS_WEIGHT = 2.0

    # Curriculum learning (enhanced)
    CURRICULUM_PHASES = [
        {"max_nodes": 12, "success_threshold": 65.0},
        {"max_nodes": 16, "success_threshold": 70.0},
        {"max_nodes": 20, "success_threshold": 75.0},
        {"max_nodes": 30, "success_threshold": 80.0},
        {"max_nodes": 50, "success_threshold": 75.0},
        {"max_nodes": 100, "success_threshold": 75.0},
    ]
    MIN_CONSECUTIVE_SUCCESS = 10
    MIN_PHASE_UPDATES = 50

    # Checkpointing and logging
    CHECKPOINT_FREQ = 100
    LOG_FREQ = 10
    DETAILED_LOG_FREQ = 50

    # Gradient clipping
    MAX_GRAD_NORM = 0.5

    # GPU optimization settings
    PIN_MEMORY = True if torch.cuda.is_available() else False
    NON_BLOCKING = True if torch.cuda.is_available() else False


def setup_logging():
    """Setup enhanced logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / 'results' / 'enhanced_rl'
    results_dir.mkdir(parents=True, exist_ok=True)

    tb_logdir = results_dir / 'runs' / f'run_{int(time.time())}'
    tb_logdir.mkdir(parents=True, exist_ok=True)

    return base_dir, results_dir, tb_logdir


def setup_gpu_optimization():
    """Setup GPU optimization settings"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) on Ampere GPUs for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger = logging.getLogger(__name__)
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")

        # Clear GPU cache
        torch.cuda.empty_cache()

    return torch.cuda.is_available()


def manage_gpu_memory():
    """Manage GPU memory during training"""
    if torch.cuda.is_available():
        # Clear cache periodically
        torch.cuda.empty_cache()

        # Get memory usage
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        cached = torch.cuda.memory_reserved() / 1024 ** 3

        return allocated, cached
    return 0.0, 0.0

def setup_directories():
    """Create necessary directories"""
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / 'results' / 'enhanced_rl'
    results_dir.mkdir(parents=True, exist_ok=True)

    tb_logdir = results_dir / 'runs' / f'run_{int(time.time())}'
    tb_logdir.mkdir(parents=True, exist_ok=True)

    return base_dir, results_dir, tb_logdir


# =============================
# Enhanced Data Loading
# =============================

def load_and_preprocess_data(base_dir: Path, device: torch.device):
    """Load and preprocess data with enhanced features"""
    logger = logging.getLogger(__name__)

    # Load configuration with fallback
    config_path = base_dir / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        edge_feats = cfg.get('edge_features', [])
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using default edge features")
        edge_feats = ['dependency_type', 'coupling_strength', 'usage_frequency']  # Default edge features

    # Load dataset
    data_dir = base_dir / 'data' / 'dataset_graph_feature'
    files = list(data_dir.glob('*.pt'))

    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    logger.info(f"Loading {len(files)} data files...")
    dataset = []
    for fp in files:
        try:
            data = torch.load(fp, map_location='cpu')
            dataset.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {fp}: {e}")

    # Filter smelly data
    smelly_data_raw = [d for d in dataset if getattr(d, 'is_smelly', 0) == 1]
    clean_data_raw = [d for d in dataset if getattr(d, 'is_smelly', 0) == 0]

    if not smelly_data_raw:
        raise ValueError("No smelly data found in dataset")

    logger.info(f"Found {len(smelly_data_raw)} smelly graphs, {len(clean_data_raw)} clean graphs")

    # Enhanced normalization
    all_node_features = np.vstack([d.x.cpu().numpy() for d in dataset])
    all_edge_features = np.vstack([d.edge_attr.cpu().numpy() for d in dataset])

    # Robust scaling with outlier handling
    scaler_node = RobustScaler().fit(all_node_features)
    scaler_edge = RobustScaler().fit(all_edge_features)

    # Create a simple normalizer class if NormalizeAttrs is not available
    class SimpleNormalizer:
        def __init__(self, node_scaler, edge_scaler):
            self.node_scaler = node_scaler
            self.edge_scaler = edge_scaler

        def __call__(self, data):
            data_copy = data.clone()
            data_copy.x = torch.tensor(
                self.node_scaler.transform(data.x.cpu().numpy()),
                dtype=torch.float32
            )
            data_copy.edge_attr = torch.tensor(
                self.edge_scaler.transform(data.edge_attr.cpu().numpy()),
                dtype=torch.float32
            )
            return data_copy

    try:
        normalizer = NormalizeAttrs(scaler_node, scaler_edge)
    except NameError:
        logger.warning("NormalizeAttrs not found, using simple normalizer")
        normalizer = SimpleNormalizer(scaler_node, scaler_edge)

    # Apply normalization and move to device efficiently
    logger.info("Moving data to device...")
    smelly_data = []
    clean_data = []

    for d in [normalizer(d) for d in smelly_data_raw]:
        if EnhancedTrainingConfig.PIN_MEMORY and device.type == 'cuda':
            d = d.pin_memory()
        smelly_data.append(d.to(device, non_blocking=EnhancedTrainingConfig.NON_BLOCKING))

    for d in [normalizer(d) for d in clean_data_raw]:
        if EnhancedTrainingConfig.PIN_MEMORY and device.type == 'cuda':
            d = d.pin_memory()
        clean_data.append(d.to(device, non_blocking=EnhancedTrainingConfig.NON_BLOCKING))

    logger.info(f"Data preprocessing complete. Node dim: {smelly_data[0].x.size(1)}, "
                f"Edge dim: {smelly_data[0].edge_attr.size(1)}")

    return smelly_data, clean_data, edge_feats


# =============================
# Enhanced Model Setup
# =============================

def safe_load_model_weights(model, checkpoint_path, model_name, logger):
    """Safely load model weights, handling architecture mismatches"""
    try:
        if not checkpoint_path.exists():
            logger.info(f"No pretrained {model_name} found at {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # If checkpoint is a dict with state_dict key
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and any(
                k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            logger.warning(f"Unrecognized checkpoint format for {model_name}")
            return False

        # Try to load compatible weights only
        model_dict = model.state_dict()
        compatible_dict = {}

        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                logger.debug(f"Skipping incompatible weight: {k}")

        if compatible_dict:
            model.load_state_dict(compatible_dict, strict=False)
            logger.info(f"Loaded {len(compatible_dict)}/{len(model_dict)} compatible weights for {model_name}")
            return True
        else:
            logger.warning(f"No compatible weights found for {model_name}")
            return False

    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}")
        return False


def setup_enhanced_discriminator(node_dim: int, edge_dim: int, device: torch.device):
    """Setup the enhanced discriminator with safe loading"""
    logger = logging.getLogger(__name__)

    discriminator = MultiScaleGCNDiscriminator(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=128,
        num_layers=4,
        dropout=0.3,
        use_attention=True,
        use_gat=True
    ).to(device)

    # Try to load pretrained weights safely
    base_dir = Path(__file__).resolve().parent
    disc_ckpt = base_dir.parent / 'results' / 'best_discriminator.pt'

    if safe_load_model_weights(discriminator, disc_ckpt, "discriminator", logger):
        logger.info("Successfully loaded compatible discriminator weights")
    else:
        logger.info("Training discriminator from scratch")

    return discriminator


def setup_enhanced_policy_and_critic(node_dim: int, edge_dim: int, device: torch.device):
    """Setup enhanced policy network and critic with safe loading"""
    logger = logging.getLogger(__name__)

    # Enhanced policy network
    policy = EnhancedGCPNPolicyNetwork(
        node_in_dim=node_dim,
        edge_in_dim=edge_dim,
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        device=device,
        enable_memory=True,
        max_sequence_length=10
    ).to(device)

    # Enhanced critic network
    class EnhancedCritic(torch.nn.Module):
        def __init__(self, policy_net, hidden_dim=64):
            super().__init__()
            import copy

            # Share encoder with policy
            self.node_embed = copy.deepcopy(policy_net.node_embed)
            self.edge_embed = copy.deepcopy(policy_net.edge_embed)
            self.convs = torch.nn.ModuleList([copy.deepcopy(c) for c in policy_net.convs])
            self.norms = torch.nn.ModuleList([copy.deepcopy(n) for n in policy_net.norms])
            self.hub_attention = copy.deepcopy(policy_net.hub_attention)
            self.hierarchical_pool = copy.deepcopy(policy_net.hierarchical_pool)

            # Value prediction head
            self.value_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(hidden_dim // 2, 1)
            )

            # Initialize value head
            for m in self.value_head.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch if data.batch is not None else torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )

            # Compute hub indicators
            hub_scores = policy.compute_hub_indicators(x)

            # Shared encoder
            x_emb = self.node_embed(x)
            edge_attr_emb = self.edge_embed(edge_attr)

            # GNN layers with hub-aware attention
            for conv, norm in zip(self.convs, self.norms):
                out = conv(x_emb, edge_index, edge_attr_emb)
                out = norm(out)
                out = F.relu(out)
                out = F.dropout(out, p=0.3, training=self.training)

                # Hub-aware attention
                out, _ = self.hub_attention(out, hub_scores)
                x_emb = x_emb + out

            # Hierarchical pooling
            graph_emb = self.hierarchical_pool(x_emb, hub_scores, batch)

            # Value prediction
            return self.value_head(graph_emb)

    critic = EnhancedCritic(policy).to(device)

    # Try to load pretrained policy weights safely
    base_dir = Path(__file__).resolve().parent
    policy_ckpt = base_dir.parent / 'results' / 'policy_pretrained.pt'

    if safe_load_model_weights(policy, policy_ckpt, "policy", logger):
        logger.info("Successfully loaded compatible policy weights")
    else:
        logger.info("Training policy from scratch")

    return policy, critic


# =============================
# Enhanced Curriculum Manager
# =============================

class EnhancedCurriculumManager:
    """Enhanced curriculum learning with phase management"""

    def __init__(self, phases: List[Dict], min_consecutive: int, min_phase_updates: int):
        self.phases = phases
        self.min_consecutive = min_consecutive
        self.min_phase_updates = min_phase_updates

        self.current_phase = 0
        self.phase_updates = 0
        self.success_buffer = deque(maxlen=min_consecutive)
        self.phase_start_time = time.time()

    def get_current_phase(self) -> Dict[str, Any]:
        return self.phases[self.current_phase]

    def should_advance(self, success_rate: float, hub_improvement: float) -> bool:
        """Check if should advance to next phase"""
        self.success_buffer.append(success_rate)
        self.phase_updates += 1

        if (self.phase_updates < self.min_phase_updates or
                len(self.success_buffer) < self.min_consecutive):
            return False

        current_phase = self.get_current_phase()
        avg_success = np.mean(list(self.success_buffer))

        # Enhanced criteria: success rate + hub improvement
        meets_criteria = (
                avg_success >= current_phase['success_threshold'] and
                hub_improvement > 0.1 and  # Meaningful improvement
                self.current_phase < len(self.phases) - 1
        )

        if meets_criteria:
            self.current_phase += 1
            self.phase_updates = 0
            self.success_buffer.clear()
            self.phase_start_time = time.time()
            return True

        return False

    def get_phase_info(self) -> Dict[str, Any]:
        current = self.get_current_phase()
        return {
            'phase': self.current_phase + 1,
            'total_phases': len(self.phases),
            'max_nodes': current['max_nodes'],
            'success_threshold': current['success_threshold'],
            'phase_updates': self.phase_updates,
            'phase_duration': time.time() - self.phase_start_time,
            'progress': (self.current_phase + 1) / len(self.phases)
        }


# =============================
# Enhanced Environment Creation
# =============================

def create_enhanced_environments(candidate_data: List, discriminator,
                                 config: EnhancedTrainingConfig) -> List[EnhancedDependencyRefactorEnv]:
    """Create enhanced training environments"""
    envs = []
    for _ in range(config.NUM_ENVS):
        initial_graph = random.choice(candidate_data)
        env = EnhancedDependencyRefactorEnv(
            initial_data=initial_graph,
            discriminator=discriminator,
            max_steps=config.ENV_MAX_STEPS,
            device=config.DEVICE,

            # Enhanced reward parameters
            pos_terminal=3.0,
            neg_terminal=-1.5,
            step_penalty=-0.02,
            successful_pattern_bonus=1.5,
            failed_pattern_penalty=-0.5,
            pattern_appropriateness_weight=0.3,
            hub_reduction_bonus=2.0,
            centrality_improvement_weight=0.4,
            coupling_reduction_weight=0.3,
            invalid_action_penalty=-0.3,
            architectural_violation_penalty=-0.8,
            diversity_bonus_weight=0.1,

            # Advanced features
            enable_progress_tracking=True,
            enable_pattern_memory=True
        )
        envs.append(env)

    return envs


# =============================
# Enhanced Training Functions
# =============================

def enhanced_collect_rollouts(envs: List[EnhancedDependencyRefactorEnv],
                              candidate_data: List,
                              policy, critic,
                              config: EnhancedTrainingConfig) -> Dict[str, Any]:
    """Enhanced rollout collection with action memory"""

    logger = logging.getLogger(__name__)  # Add logger here
    print("DEBUG: Starting rollout collection function")  # DEBUG

    # Initialize or reset environments
    states = []
    action_histories = []
    for i, env in enumerate(envs):
        if random.random() < 0.1:  # 10% chance to reset with new graph
            env.initial_data = random.choice(candidate_data)
        state = env.reset()
        states.append(state)
        action_histories.append([])

    print(f"DEBUG: Initialized {len(envs)} environments")  # DEBUG

    # Rollout buffers
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
    episode_rewards, episode_successes, episode_steps = [], [], []
    episode_infos = []
    hub_improvements = []

    # Environment tracking
    cumulative_rewards = [0.0] * config.NUM_ENVS
    steps_taken = [0] * config.NUM_ENVS

    for step in range(config.ROLLOUT_STEPS // config.NUM_ENVS):
        if step % 10 == 0:  # DEBUG ogni 10 step
            print(f"DEBUG: Rollout step {step}/{config.ROLLOUT_STEPS // config.NUM_ENVS}")

        # Batch processing with error handling
        try:
            batch = Batch.from_data_list(states).to(config.DEVICE)
        except Exception as e:
            logger.warning(f"Failed to create batch at step {step}: {e}")
            # Skip this step and continue
            continue

        with torch.no_grad():
            # Policy forward pass with action memory
            policy_outputs = []
            try:
                for i, action_history in enumerate(action_histories):
                    if i >= len(states):
                        continue
                    env_data = states[i].to(config.DEVICE)
                    policy_out = policy(env_data, action_history)
                    policy_outputs.append(policy_out)

                # Critic forward pass
                values = critic(batch).squeeze()
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                elif len(values) != config.NUM_ENVS:
                    # Pad or truncate to match number of environments
                    if len(values) < config.NUM_ENVS:
                        padding = torch.zeros(config.NUM_ENVS - len(values), device=config.DEVICE)
                        values = torch.cat([values, padding])
                    else:
                        values = values[:config.NUM_ENVS]

            except Exception as e:
                logger.warning(f"Policy/Critic forward failed at step {step}: {e}")
                continue

        # Process each environment
        next_states = []
        ptr = batch.ptr.cpu().tolist()

        for i, env in enumerate(envs):
            if i >= len(policy_outputs):
                continue

            policy_out = policy_outputs[i]
            env_start, env_end = ptr[i], ptr[i + 1] if i + 1 < len(ptr) else len(batch.x)

            # Sample actions with enhanced logic
            if env_start < env_end:
                a1 = Categorical(policy_out.pi1).sample().item()
                a2 = Categorical(policy_out.pi2).sample().item()
            else:
                a1 = a2 = 0

            a3 = Categorical(policy_out.pi3).sample().item()
            a4 = Categorical(policy_out.pi4).sample().item()

            action = (a1, a2, a3, a4)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Calculate log probability
            logp = (
                    Categorical(policy_out.pi1).log_prob(torch.tensor(a1, device=config.DEVICE)) +
                    Categorical(policy_out.pi2).log_prob(torch.tensor(a2, device=config.DEVICE)) +
                    Categorical(policy_out.pi3).log_prob(torch.tensor(a3, device=config.DEVICE)) +
                    Categorical(policy_out.pi4).log_prob(torch.tensor(a4, device=config.DEVICE))
            )

            # Store experience
            obs_buf.append(states[i])
            act_buf.append(torch.tensor(action, device=config.DEVICE))
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(values[i].item() if i < len(values) else 0.0)

            # Update tracking
            cumulative_rewards[i] += reward
            steps_taken[i] += 1

            # Update action history
            action_histories[i].append(a3)  # Track pattern actions
            if len(action_histories[i]) > 10:  # Keep last 10 actions
                action_histories[i].pop(0)

            # Handle episode completion
            if done:
                episode_rewards.append(cumulative_rewards[i])
                episode_steps.append(steps_taken[i])
                episode_successes.append(1 if info.get('success', False) else 0)
                episode_infos.append(info)

                # Track hub improvement
                if 'hub_improvement' in info:
                    hub_improvements.append(info['hub_improvement'])

                # Reset environment
                env.initial_data = random.choice(candidate_data)
                next_state = env.reset()
                action_histories[i] = []

                cumulative_rewards[i] = 0.0
                steps_taken[i] = 0

            next_states.append(next_state)

        states = next_states

        # Early stopping if we have enough data
        if len(obs_buf) >= config.ROLLOUT_STEPS:
            break

    return {
        'obs_buf': obs_buf,
        'act_buf': act_buf,
        'logp_buf': logp_buf,
        'rew_buf': rew_buf,
        'val_buf': val_buf,
        'episode_rewards': episode_rewards,
        'episode_successes': episode_successes,
        'episode_steps': episode_steps,
        'episode_infos': episode_infos,
        'hub_improvements': hub_improvements
    }


def enhanced_compute_gae(rewards: List[float], values: List[float],
                         gamma: float, lam: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced GAE computation with value clipping"""
    if len(rewards) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    advantages = []
    gae = 0

    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]

        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = advantages + torch.tensor(values, dtype=torch.float32, device=device)

    # Enhanced normalization with outlier handling
    if len(advantages) > 1:
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = torch.clamp(
            (advantages - adv_mean) / (adv_std + 1e-8),
            -10.0, 10.0  # Clip extreme values
        )

    return advantages, returns


def enhanced_ppo_update(rollout_data: Dict, policy, critic, discriminator,
                        opt_actor, opt_critic,
                        config: EnhancedTrainingConfig) -> Tuple[float, float, float]:
    """Enhanced PPO update with adversarial learning"""

    obs_buf = rollout_data['obs_buf']
    act_buf = rollout_data['act_buf']
    logp_buf = rollout_data['logp_buf']
    rew_buf = rollout_data['rew_buf']
    val_buf = rollout_data['val_buf']

    if len(obs_buf) < config.MINIBATCH_SIZE:
        return 0.0, 0.0, 0.0

    # Compute advantages
    advantages, returns = enhanced_compute_gae(
        rew_buf, val_buf, config.GAMMA, config.GAE_LAMBDA, config.DEVICE
    )

    total_actor_loss = total_critic_loss = total_adversarial_loss = 0.0
    n_updates = 0

    # Create dataset
    n_samples = len(obs_buf)

    for epoch in range(config.UPDATE_EPOCHS):
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, config.MINIBATCH_SIZE):
            end = min(start + config.MINIBATCH_SIZE, n_samples)
            batch_indices = indices[start:end]

            if len(batch_indices) < 8:  # Skip very small batches
                continue

            # Create batch
            batch_obs = [obs_buf[i] for i in batch_indices]
            batch_acts = torch.stack([act_buf[i] for i in batch_indices])
            batch_old_logp = torch.stack([logp_buf[i] for i in batch_indices])
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]

            # Forward pass
            batch_data = Batch.from_data_list(batch_obs).to(config.DEVICE)

            # Policy forward with action histories (simplified for batch)
            policy_out = policy(batch_data)

            # Compute new log probabilities
            ptr = batch_data.ptr.cpu().tolist()
            new_logprobs = []

            for i, (a1, a2, a3, a4) in enumerate(batch_acts):
                env_start = ptr[i] if i < len(ptr) else 0
                env_end = ptr[i + 1] if i + 1 < len(ptr) else len(batch_data.x)

                if env_start < env_end:
                    logp1 = Categorical(policy_out.pi1[env_start:env_end]).log_prob(a1)
                    logp2 = Categorical(policy_out.pi2[env_start:env_end]).log_prob(a2)
                else:
                    logp1 = logp2 = torch.tensor(0.0, device=config.DEVICE)

                logp3 = Categorical(policy_out.pi3[i]).log_prob(a3)
                logp4 = Categorical(policy_out.pi4[i]).log_prob(a4)

                total_logp = logp1 + logp2 + logp3 + logp4
                new_logprobs.append(total_logp)

            new_logprobs = torch.stack(new_logprobs)

            # PPO loss computation
            ratio = torch.exp(new_logprobs - batch_old_logp)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Enhanced entropy bonus
            entropy = (
                    Categorical(policy_out.pi1).entropy().mean() +
                    Categorical(policy_out.pi2).entropy().mean() +
                    Categorical(policy_out.pi3).entropy().mean() +
                    Categorical(policy_out.pi4).entropy().mean()
            )
            actor_loss -= config.BETA_ENTROPY * entropy

            # Adversarial loss (policy tries to fool discriminator)
            with torch.no_grad():
                disc_outputs = discriminator(batch_data)
                adversarial_target = torch.zeros_like(disc_outputs['logits'][:, 1])  # Try to look clean

            adversarial_loss = config.ADVERSARIAL_WEIGHT * F.binary_cross_entropy_with_logits(
                disc_outputs['logits'][:, 1], adversarial_target
            )

            total_actor_loss_batch = actor_loss + adversarial_loss

            # Critic loss with clipping
            values = critic(batch_data).squeeze()
            critic_loss = F.mse_loss(values, batch_returns)

            # Backward pass
            opt_actor.zero_grad()
            opt_critic.zero_grad()

            total_loss = total_actor_loss_batch + config.VALUE_LOSS_COEF * critic_loss
            total_loss.backward()

            # Enhanced gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), config.MAX_GRAD_NORM)

            opt_actor.step()
            opt_critic.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_adversarial_loss += adversarial_loss.item()
            n_updates += 1

    return (total_actor_loss / max(n_updates, 1),
            total_critic_loss / max(n_updates, 1),
            total_adversarial_loss / max(n_updates, 1))


def enhanced_update_discriminator(obs_buffer: List, clean_data: List, smelly_data: List,
                                  discriminator, opt_disc,
                                  config: EnhancedTrainingConfig) -> Tuple[float, float, float]:
    """Enhanced discriminator update with hub-aware loss"""

    if len(obs_buffer) < config.MINIBATCH_SIZE:
        return 0.0, 0.0, 0.0

    discriminator.train()
    hub_loss_fn = HubAwareLoss(alpha=0.25, gamma=2.0, hub_weight=config.HUB_LOSS_WEIGHT)

    # Create batches
    batch_size = config.MINIBATCH_SIZE // 3

    # Real smelly data
    real_smelly = Batch.from_data_list(
        random.sample(smelly_data, min(batch_size, len(smelly_data)))
    ).to(config.DEVICE)

    # Real clean data
    real_clean = Batch.from_data_list(
        random.sample(clean_data, min(batch_size, len(clean_data)))
    ).to(config.DEVICE)

    # Generated/modified data from policy
    fake_data = Batch.from_data_list(
        random.sample(obs_buffer, min(batch_size, len(obs_buffer)))
    ).to(config.DEVICE)

    # Forward pass
    real_smelly_outputs = discriminator(real_smelly)
    real_clean_outputs = discriminator(real_clean)
    fake_outputs = discriminator(fake_data)

    # Enhanced loss computation with hub importance
    smelly_targets = torch.ones(real_smelly_outputs['logits'].size(0), device=config.DEVICE)
    clean_targets = torch.zeros(real_clean_outputs['logits'].size(0), device=config.DEVICE)
    fake_targets = torch.zeros(fake_outputs['logits'].size(0), device=config.DEVICE)

    # Hub-aware loss for better detection
    loss_smelly = hub_loss_fn(
        real_smelly_outputs['logits'],
        smelly_targets.long(),
        real_smelly_outputs.get('hub_importance')
    )

    loss_clean = F.cross_entropy(real_clean_outputs['logits'], clean_targets.long())
    loss_fake = F.cross_entropy(fake_outputs['logits'], fake_targets.long())

    total_loss = loss_smelly + loss_clean + loss_fake

    # Backward pass
    opt_disc.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.MAX_GRAD_NORM)
    opt_disc.step()

    # Compute accuracies
    with torch.no_grad():
        smelly_probs = F.softmax(real_smelly_outputs['logits'], dim=1)[:, 1]
        clean_probs = F.softmax(real_clean_outputs['logits'], dim=1)[:, 1]
        fake_probs = F.softmax(fake_outputs['logits'], dim=1)[:, 1]

        acc_smelly = (smelly_probs > 0.5).float().mean().item() * 100
        acc_clean = (clean_probs < 0.5).float().mean().item() * 100
        acc_fake = (fake_probs < 0.5).float().mean().item() * 100

    discriminator.eval()
    return total_loss.item(), (acc_smelly + acc_clean + acc_fake) / 3, acc_fake


# =============================
# Enhanced Monitoring and Logging
# =============================

class EnhancedPerformanceTracker:
    """Enhanced performance tracking with detailed metrics"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'success_rates': deque(maxlen=window_size),
            'episode_steps': deque(maxlen=window_size),
            'episode_rewards': deque(maxlen=window_size),
            'hub_improvements': deque(maxlen=window_size),
            'pattern_success_rates': deque(maxlen=window_size),
            'actor_losses': deque(maxlen=window_size),
            'critic_losses': deque(maxlen=window_size),
            'adversarial_losses': deque(maxlen=window_size)
        }

    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key in self.metrics and value is not None:
                self.metrics[key].append(value)

    def get_trends(self) -> Dict[str, float]:
        """Get performance trends"""
        trends = {}

        for key, values in self.metrics.items():
            if len(values) >= 10:
                # Simple linear trend
                x = np.arange(len(values))
                y = np.array(values)
                slope = np.polyfit(x, y, 1)[0]
                trends[f'{key}_trend'] = slope
                trends[f'{key}_recent'] = np.mean(y[-10:])
                trends[f'{key}_std'] = np.std(y[-10:])

        return trends

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        summary = {}

        for key, values in self.metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_recent'] = np.mean(list(values)[-10:]) if len(values) >= 10 else np.mean(values)

        return summary


def enhanced_log_detailed_metrics(update: int, rollout_data: Dict,
                                  writer: SummaryWriter, logger: logging.Logger):
    """Enhanced detailed logging"""

    episode_infos = rollout_data.get('episode_infos', [])
    if not episode_infos:
        return

    # Analyze episode outcomes
    pattern_usage = {}
    termination_reasons = {}
    hub_metrics = []

    for info in episode_infos:
        # Pattern usage analysis
        patterns_applied = info.get('patterns_applied', 0)
        successful_patterns = info.get('successful_patterns', 0)

        if patterns_applied > 0:
            pattern_success_rate = successful_patterns / patterns_applied
        else:
            pattern_success_rate = 0.0

        # Hub improvement tracking
        hub_improvement = info.get('hub_improvement', 0.0)
        if hub_improvement != 0:
            hub_metrics.append(hub_improvement)

        # Termination reason analysis
        if info.get('success', False):
            termination_reasons['success'] = termination_reasons.get('success', 0) + 1
        else:
            reason = info.get('early_termination_reason', 'timeout')
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    # Log to TensorBoard
    if hub_metrics:
        writer.add_scalar('Detailed/AvgHubImprovement', np.mean(hub_metrics), update)
        writer.add_scalar('Detailed/StdHubImprovement', np.std(hub_metrics), update)

    # Log termination distribution
    total_episodes = len(episode_infos)
    for reason, count in termination_reasons.items():
        percentage = 100 * count / total_episodes
        writer.add_scalar(f'Termination/{reason}_pct', percentage, update)

    # Detailed console logging
    if update % 100 == 0:
        logger.info(f"=== Detailed Analysis at Update {update} ===")
        if hub_metrics:
            logger.info(f"  Hub Improvement: μ={np.mean(hub_metrics):.3f}, σ={np.std(hub_metrics):.3f}")
        if termination_reasons:
            logger.info(f"  Termination Distribution: {termination_reasons}")
        logger.info("=" * 50)


def save_enhanced_checkpoint(update: int, policy, critic, discriminator,
                             curriculum_manager, perf_tracker,
                             results_dir: Path, logger: logging.Logger):
    """Save enhanced checkpoint with additional state"""
    try:
        checkpoint = {
            'update': update,
            'policy_state_dict': policy.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'curriculum_phase': curriculum_manager.current_phase,
            'curriculum_updates': curriculum_manager.phase_updates,
            'performance_summary': perf_tracker.get_summary(),
            'timestamp': time.time()
        }

        torch.save(checkpoint, results_dir / f'enhanced_checkpoint_{update}.pt')

        # Save best model if performance is good
        summary = perf_tracker.get_summary()
        recent_success = summary.get('success_rates_recent', 0)

        if recent_success > 80.0:  # High success rate
            torch.save(checkpoint, results_dir / 'best_enhanced_model.pt')
            logger.info(f"Saved best model at update {update} (success rate: {recent_success:.1f}%)")

        logger.info(f"Saved enhanced checkpoint at update {update}")

    except Exception as e:
        logger.error(f"Failed to save enhanced checkpoint: {e}")


# =============================
# Enhanced Main Function
# =============================

def main():
    """Enhanced main training function"""

    # Setup
    logger = setup_logging()
    base_dir, results_dir, tb_logdir = setup_directories()

    # GPU optimization
    gpu_available = setup_gpu_optimization()

    # Set seeds for reproducibility (inline)
    random.seed(EnhancedTrainingConfig.SEED)
    np.random.seed(EnhancedTrainingConfig.SEED)
    torch.manual_seed(EnhancedTrainingConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(EnhancedTrainingConfig.SEED)
        torch.cuda.manual_seed_all(EnhancedTrainingConfig.SEED)

    logger.info("=" * 70)
    logger.info("ENHANCED GCPN TRAINING FOR HUB-LIKE DEPENDENCY REFACTORING")
    logger.info("=" * 70)
    logger.info(f"Device: {EnhancedTrainingConfig.DEVICE}")
    logger.info(f"GPU Available: {gpu_available}")
    if gpu_available:
        allocated, cached = manage_gpu_memory()
        logger.info(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"TensorBoard logs: {tb_logdir}")
    logger.info(f"Total updates: {EnhancedTrainingConfig.TOTAL_UPDATES}")

    # TensorBoard
    writer = SummaryWriter(str(tb_logdir))

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        smelly_data, clean_data, edge_feats = load_and_preprocess_data(
            base_dir, EnhancedTrainingConfig.DEVICE
        )

        node_dim = smelly_data[0].x.size(1)
        edge_dim = smelly_data[0].edge_attr.size(1)

        logger.info(f"Dataset: {len(smelly_data)} smelly, {len(clean_data)} clean graphs")
        logger.info(f"Features: {node_dim} node, {edge_dim} edge dimensions")

        # Setup enhanced models
        logger.info("Setting up enhanced models...")
        discriminator = setup_enhanced_discriminator(node_dim, edge_dim, EnhancedTrainingConfig.DEVICE)
        policy, critic = setup_enhanced_policy_and_critic(node_dim, edge_dim, EnhancedTrainingConfig.DEVICE)

        # Count parameters
        policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        critic_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
        disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

        logger.info(f"Model parameters: Policy={policy_params:,}, Critic={critic_params:,}, Disc={disc_params:,}")

        # Setup optimizers with learning rate scheduling
        opt_actor = optim.AdamW(policy.parameters(),
                                lr=EnhancedTrainingConfig.LR_ACTOR,
                                weight_decay=1e-4)
        opt_critic = optim.AdamW(critic.parameters(),
                                 lr=EnhancedTrainingConfig.LR_CRITIC,
                                 weight_decay=1e-4)
        opt_disc = optim.AdamW(discriminator.parameters(),
                               lr=EnhancedTrainingConfig.LR_DISCRIMINATOR,
                               weight_decay=1e-4)

        # Learning rate schedulers
        scheduler_actor = optim.lr_scheduler.CosineAnnealingLR(opt_actor, T_max=EnhancedTrainingConfig.TOTAL_UPDATES)
        scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, T_max=EnhancedTrainingConfig.TOTAL_UPDATES)
        scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=EnhancedTrainingConfig.TOTAL_UPDATES)

        # Enhanced training components
        curriculum_manager = EnhancedCurriculumManager(
            EnhancedTrainingConfig.CURRICULUM_PHASES,
            EnhancedTrainingConfig.MIN_CONSECUTIVE_SUCCESS,
            EnhancedTrainingConfig.MIN_PHASE_UPDATES
        )

        perf_tracker = EnhancedPerformanceTracker(window_size=100)

        logger.info(f"Starting curriculum: {curriculum_manager.get_phase_info()}")
        logger.info("=" * 70)

        # Training variables
        envs = None
        best_success_rate = 0.0
        stagnation_counter = 0

        # Main training loop
        for update in range(1, EnhancedTrainingConfig.TOTAL_UPDATES + 1):

            print(f"DEBUG: Starting update {update}")  # DEBUG

            # Get current phase data (with fallback for small datasets)
            current_phase = curriculum_manager.get_current_phase()
            max_nodes = current_phase['max_nodes']
            candidate_data = [d for d in smelly_data if d.x.size(0) <= max_nodes]

            # Fallback if no candidates in current phase
            if not candidate_data:
                candidate_data = smelly_data
                logger.warning(f"No graphs with ≤{max_nodes} nodes, using all {len(smelly_data)} graphs")

            # Ensure we have enough diversity
            if len(candidate_data) < 5:
                # Add some larger graphs if we have too few candidates
                larger_graphs = [d for d in smelly_data if d.x.size(0) <= max_nodes * 2]
                candidate_data.extend(larger_graphs[:10])  # Add up to 10 more

                # Remove duplicates by comparing data identity (memory address)
                seen_ids = set()
                unique_candidate_data = []
                for data in candidate_data:
                    data_id = id(data)
                    if data_id not in seen_ids:
                        seen_ids.add(data_id)
                        unique_candidate_data.append(data)
                candidate_data = unique_candidate_data

                logger.info(f"Expanded candidate set to {len(candidate_data)} graphs for diversity")

            print(f"DEBUG: Using {len(candidate_data)} candidate graphs")  # DEBUG

            # Create/refresh environments periodically
            if update == 1 or update % 200 == 0:
                envs = create_enhanced_environments(
                    candidate_data, discriminator, EnhancedTrainingConfig
                )
                logger.info(f"Refreshed environments with {len(candidate_data)} candidate graphs")

            print("DEBUG: Starting rollout collection...")  # DEBUG

            # Collect rollouts
            rollout_data = enhanced_collect_rollouts(
                envs, candidate_data, policy, critic, EnhancedTrainingConfig
            )

            print(f"DEBUG: Rollout completed - {len(rollout_data['obs_buf'])} samples")  # DEBUG

            # Check if we have enough data
            if len(rollout_data['obs_buf']) < EnhancedTrainingConfig.MINIBATCH_SIZE:
                logger.warning(f"Insufficient data at update {update}, recreating environments...")
                envs = create_enhanced_environments(
                    candidate_data, discriminator, EnhancedTrainingConfig
                )
                continue

            # PPO update
            avg_actor_loss, avg_critic_loss, avg_adversarial_loss = enhanced_ppo_update(
                rollout_data, policy, critic, discriminator,
                opt_actor, opt_critic, EnhancedTrainingConfig
            )

            # Discriminator update
            if update % EnhancedTrainingConfig.DISC_UPDATE_FREQ == 0:
                avg_disc_loss, disc_acc, fake_acc = enhanced_update_discriminator(
                    rollout_data['obs_buf'][-EnhancedTrainingConfig.MINIBATCH_SIZE * 2:],
                    clean_data, smelly_data, discriminator, opt_disc, EnhancedTrainingConfig
                )
            else:
                avg_disc_loss = disc_acc = fake_acc = 0.0

            # Update learning rates AFTER optimizer steps (fix warning)
            scheduler_actor.step()
            scheduler_critic.step()
            scheduler_disc.step()

            # Get current learning rate (fix for variable scope)
            current_lr = scheduler_actor.get_last_lr()[0] if hasattr(scheduler_actor,
                                                                     'get_last_lr') else EnhancedTrainingConfig.LR_ACTOR

            # Compute metrics
            episode_rewards = rollout_data['episode_rewards']
            episode_successes = rollout_data['episode_successes']
            episode_steps = rollout_data['episode_steps']
            hub_improvements = rollout_data['hub_improvements']

            # Always compute basic metrics, even if no episodes completed
            if episode_successes:
                success_rate = 100 * sum(episode_successes) / len(episode_successes)
                avg_reward = np.mean(episode_rewards)
                avg_episode_steps = np.mean(episode_steps)
                avg_hub_improvement = np.mean(hub_improvements) if hub_improvements else 0.0
            else:
                # No episodes completed - compute from rollout data
                success_rate = 0.0
                avg_reward = np.mean(rollout_data['rew_buf']) if rollout_data['rew_buf'] else 0.0
                avg_episode_steps = len(rollout_data['obs_buf']) / max(len(envs), 1)
                avg_hub_improvement = 0.0

            # Force logging even with no completed episodes
            print(
                f"DEBUG: Episodes completed: {len(episode_rewards)}, Total samples: {len(rollout_data['obs_buf'])}")  # DEBUG

            # Update performance tracker
            perf_tracker.update(
                success_rates=success_rate,
                episode_steps=avg_episode_steps,
                episode_rewards=avg_reward,
                hub_improvements=avg_hub_improvement,
                actor_losses=avg_actor_loss,
                critic_losses=avg_critic_loss,
                adversarial_losses=avg_adversarial_loss
            )

            # Curriculum management
            phase_info = curriculum_manager.get_phase_info()
            advanced = curriculum_manager.should_advance(success_rate, avg_hub_improvement)

            if advanced:
                new_phase_info = curriculum_manager.get_phase_info()
                logger.info(f"🎓 CURRICULUM ADVANCED!")
                logger.info(f"   Phase {phase_info['phase']} → {new_phase_info['phase']}")
                logger.info(f"   Max nodes: {phase_info['max_nodes']} → {new_phase_info['max_nodes']}")
                writer.add_scalar('Curriculum/Phase', new_phase_info['phase'], update)
                writer.add_scalar('Curriculum/Advancement', 1, update)

            # Stagnation detection
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Logging (force log every update for debugging)
            if update % EnhancedTrainingConfig.LOG_FREQ == 0 or advanced or len(episode_rewards) == 0:
                logger.info(
                    f"Update {update:04d} | "
                    f"Phase {phase_info['phase']}/{phase_info['total_phases']} (≤{max_nodes}) | "
                    f"SR: {success_rate:5.1f}% | "
                    f"Steps: {avg_episode_steps:5.1f} | "
                    f"HubImp: {avg_hub_improvement:+6.3f} | "
                    f"Eps: {len(episode_rewards):2d} | "
                    f"Samples: {len(rollout_data['obs_buf']):4d} | "
                    f"LR: {current_lr:.2e} | "
                    f"L(A/C/Adv): {avg_actor_loss:.3f}/{avg_critic_loss:.3f}/{avg_adversarial_loss:.3f}"
                )

            # TensorBoard logging
            writer.add_scalar('Train/SuccessRate', success_rate, update)
            writer.add_scalar('Train/AvgEpisodeSteps', avg_episode_steps, update)
            writer.add_scalar('Train/AvgReward', avg_reward, update)
            writer.add_scalar('Train/AvgHubImprovement', avg_hub_improvement, update)
            writer.add_scalar('Train/NumEpisodes', len(episode_rewards), update)
            writer.add_scalar('Loss/Actor', avg_actor_loss, update)
            writer.add_scalar('Loss/Critic', avg_critic_loss, update)
            writer.add_scalar('Loss/Adversarial', avg_adversarial_loss, update)
            writer.add_scalar('Curriculum/CurrentPhase', phase_info['phase'], update)
            writer.add_scalar('Curriculum/MaxNodes', max_nodes, update)
            writer.add_scalar('Curriculum/PhaseUpdates', phase_info['phase_updates'], update)
            writer.add_scalar('Optimization/LearningRate', current_lr, update)
            writer.add_scalar('Optimization/BestSuccessRate', best_success_rate, update)
            writer.add_scalar('Optimization/StagnationCounter', stagnation_counter, update)

            # GPU memory monitoring
            if torch.cuda.is_available():
                allocated, cached = manage_gpu_memory()
                writer.add_scalar('System/GPU_Memory_Allocated_GB', allocated, update)
                writer.add_scalar('System/GPU_Memory_Cached_GB', cached, update)

                # Log GPU memory usage every 50 updates
                if update % 50 == 0:
                    logger.info(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")

            if avg_disc_loss > 0:
                writer.add_scalar('Loss/Discriminator', avg_disc_loss, update)
                writer.add_scalar('Discriminator/Accuracy', disc_acc, update)
                writer.add_scalar('Discriminator/FakeAccuracy', fake_acc, update)

            # Performance trends
            if update % 50 == 0:
                trends = perf_tracker.get_trends()
                for key, value in trends.items():
                    writer.add_scalar(f'Trends/{key}', value, update)

            # Detailed logging
            if update % EnhancedTrainingConfig.DETAILED_LOG_FREQ == 0:
                enhanced_log_detailed_metrics(update, rollout_data, writer, logger)

            # Checkpointing
            if update % EnhancedTrainingConfig.CHECKPOINT_FREQ == 0:
                # Clear GPU cache before checkpointing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                save_enhanced_checkpoint(
                    update, policy, critic, discriminator,
                    curriculum_manager, perf_tracker, results_dir, logger
                )

            # Early stopping conditions
            final_phase = len(EnhancedTrainingConfig.CURRICULUM_PHASES)
            if (success_rate >= 90.0 and
                    avg_hub_improvement > 0.3 and
                    phase_info['phase'] == final_phase):
                logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                logger.info(f"Final metrics: Success={success_rate:.1f}%, HubImp={avg_hub_improvement:.3f}")
                break

            # Stagnation handling
            if stagnation_counter > 300:
                logger.warning(
                    f"Training stagnated for {stagnation_counter} updates. Consider adjusting hyperparameters.")
                # Could implement automatic hyperparameter adjustment here

        # Save final models
        final_checkpoint = {
            'update': update,
            'policy_state_dict': policy.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'final_metrics': perf_tracker.get_summary(),
            'curriculum_info': curriculum_manager.get_phase_info(),
            'training_completed': True
        }

        torch.save(final_checkpoint, results_dir / 'final_enhanced_model.pt')

        # Final summary
        final_summary = perf_tracker.get_summary()
        logger.info("=" * 70)
        logger.info("ENHANCED TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total updates completed: {update}")
        logger.info(f"Final curriculum phase: {curriculum_manager.get_phase_info()['phase']}/{final_phase}")
        logger.info(f"Best success rate achieved: {best_success_rate:.1f}%")

        if final_summary:
            logger.info(f"Final success rate: {final_summary.get('success_rates_recent', 0):.1f}%")
            logger.info(f"Final hub improvement: {final_summary.get('hub_improvements_recent', 0):.3f}")
            logger.info(f"Final episode length: {final_summary.get('episode_steps_recent', 0):.1f}")

        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        writer.close()

    logger.info("Enhanced training script completed!")


if __name__ == "__main__":
    main()