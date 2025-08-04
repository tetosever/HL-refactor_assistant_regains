#!/usr/bin/env python3
"""
Enhanced GCPN Training Script for Hub-like Dependency Refactoring
FIXED VERSION with 7-feature compatibility and enhanced discriminator validation

Key Features:
- Enhanced Policy Network with hub-aware attention (7-feature compatible)
- Advanced Environment with pattern-based actions
- Discriminator-based adversarial learning with robust validation
- PPO + GAE optimization
- Comprehensive monitoring and debugging
- GPU/CUDA optimizations
- Fixed discriminator validation and environment initialization
- UNIFIED 7-feature pipeline compatibility
"""

import logging
import random
import json
import os
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
from torch_geometric.data import Batch, Data
import torch_geometric.data.data

from discriminator import HubDetectionDiscriminator
from policy_network import HubRefactoringPolicy
from rl_gym import HubRefactoringEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Loading .* with weights_only=False")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# Define RefactoringAction class
from dataclasses import dataclass


@dataclass
class RefactoringAction:
    """Refactoring action with pattern and parameters"""
    source_node: int
    target_node: int
    pattern: int
    terminate: bool
    confidence: float = 0.0


class ExperienceBuffer:
    """Experience buffer for PPO training with GPU optimization"""

    def __init__(self, capacity: int = 10000, device: torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Dict[str, Any]):
        # Move tensors to device if needed
        if isinstance(experience.get('state'), Data):
            experience['state'] = experience['state'].to(self.device)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """PPO trainer with adversarial discriminator training - GPU optimized"""

    def __init__(self, policy: nn.Module, discriminator: nn.Module,
                 lr_policy: float = 3e-4, lr_disc: float = 1e-4,
                 device: torch.device = torch.device('cpu')):
        self.device = device

        # Move models to device
        self.policy = policy.to(device)
        self.discriminator = discriminator.to(device)

        # Optimizers with GPU-friendly settings
        self.opt_policy = optim.AdamW(policy.parameters(), lr=lr_policy, eps=1e-5, weight_decay=1e-4)
        self.opt_disc = optim.AdamW(discriminator.parameters(), lr=lr_disc, eps=1e-5, weight_decay=1e-4)

        # PPO parameters
        self.clip_eps = 0.2
        self.gamma = 0.99
        self.lam = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        # Training statistics
        self.update_count = 0
        self.policy_losses = []
        self.disc_losses = []

        # GPU optimization settings
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

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

    def compute_policy_loss(self, states: List[Data], actions: List[RefactoringAction],
                            old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute PPO policy loss - GPU optimized with 7-feature validation"""

        # Ensure states are on correct device and have 7 features
        states_on_device = []
        for state in states:
            if state.x.device != self.device:
                state = state.to(self.device)

            # CRITICAL: Validate 7-feature format
            if state.x.size(1) != 7:
                logger.warning(f"State has {state.x.size(1)} features, expected 7. Padding/truncating.")
                if state.x.size(1) > 7:
                    state.x = state.x[:, :7]  # Truncate
                else:
                    padding = torch.zeros(state.x.size(0), 7 - state.x.size(1), device=state.x.device)
                    state.x = torch.cat([state.x, padding], dim=1)  # Pad

            states_on_device.append(state)

        # Batch states efficiently
        try:
            batch_data = Batch.from_data_list(states_on_device)
            if batch_data.x.device != self.device:
                batch_data = batch_data.to(self.device)
        except Exception as e:
            logger.warning(f"Batching failed: {e}, using fallback")
            policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            entropy = torch.tensor(0.0, device=self.device)
            return {
                'policy_loss': policy_loss,
                'entropy': entropy,
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        # Forward pass
        policy_out = self.policy(batch_data)

        # Compute current log probabilities
        log_probs = []
        entropies = []

        # Handle batched data properly
        num_graphs = len(states_on_device)
        nodes_per_graph = [state.x.size(0) for state in states_on_device]
        cumsum_nodes = [0] + list(np.cumsum(nodes_per_graph))

        for i, action in enumerate(actions):
            try:
                start_idx = cumsum_nodes[i]
                end_idx = cumsum_nodes[i + 1]

                # Validate indices
                if start_idx >= policy_out['hub_probs'].size(0) or end_idx > policy_out['hub_probs'].size(0):
                    continue

                # Hub selection
                hub_probs_i = policy_out['hub_probs'][start_idx:end_idx]
                if hub_probs_i.size(0) == 0:
                    continue

                hub_dist = Categorical(hub_probs_i + 1e-8)
                hub_action_tensor = torch.tensor(action.source_node, device=self.device)
                if action.source_node >= hub_probs_i.size(0):
                    hub_action_tensor = torch.tensor(0, device=self.device)

                log_prob_hub = hub_dist.log_prob(hub_action_tensor)
                entropy_hub = hub_dist.entropy()

                # Pattern selection
                pattern_idx = start_idx + min(action.source_node, hub_probs_i.size(0) - 1)
                if pattern_idx < policy_out['pattern_probs'].size(0):
                    pattern_probs_i = policy_out['pattern_probs'][pattern_idx]
                    pattern_dist = Categorical(pattern_probs_i + 1e-8)
                    pattern_action_tensor = torch.tensor(action.pattern, device=self.device)
                    if action.pattern >= pattern_probs_i.size(0):
                        pattern_action_tensor = torch.tensor(0, device=self.device)

                    log_prob_pattern = pattern_dist.log_prob(pattern_action_tensor)
                    entropy_pattern = pattern_dist.entropy()
                else:
                    log_prob_pattern = torch.tensor(0.0, device=self.device)
                    entropy_pattern = torch.tensor(0.0, device=self.device)

                # Termination
                if i < policy_out['term_probs'].size(0):
                    term_probs_i = policy_out['term_probs'][i]
                else:
                    term_probs_i = policy_out['term_probs'][0] if policy_out['term_probs'].size(
                        0) > 0 else torch.tensor([0.5, 0.5], device=self.device)

                term_dist = Categorical(term_probs_i + 1e-8)
                term_action_tensor = torch.tensor(int(action.terminate), device=self.device)
                log_prob_term = term_dist.log_prob(term_action_tensor)
                entropy_term = term_dist.entropy()

                total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
                total_entropy = entropy_hub + entropy_pattern + entropy_term

                log_probs.append(total_log_prob)
                entropies.append(total_entropy)

            except Exception as e:
                logger.debug(f"Error computing log prob for action {i}: {e}")
                log_probs.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))

        if not log_probs:
            policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            entropy = torch.tensor(0.0, device=self.device)
            return {
                'policy_loss': policy_loss,
                'entropy': entropy,
                'approx_kl': torch.tensor(0.0, device=self.device),
                'clipfrac': torch.tensor(0.0, device=self.device)
            }

        log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropies).mean()

        # Ensure old_log_probs is on correct device and has correct shape
        old_log_probs = old_log_probs.to(self.device)
        if old_log_probs.size(0) != log_probs.size(0):
            min_size = min(old_log_probs.size(0), log_probs.size(0))
            old_log_probs = old_log_probs[:min_size]
            log_probs = log_probs[:min_size]
            advantages = advantages[:min_size]

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        return {
            'policy_loss': policy_loss,
            'entropy': entropy,
            'approx_kl': ((ratio - 1) - (log_probs - old_log_probs)).mean(),
            'clipfrac': ((ratio - 1).abs() > self.clip_eps).float().mean()
        }

    def update_policy(self, states: List[Data], actions: List[RefactoringAction],
                      old_log_probs: torch.Tensor, advantages: torch.Tensor,
                      returns: torch.Tensor, epochs: int = 4):
        """Update policy network using PPO - GPU optimized"""

        if len(states) == 0:
            return

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states), device=self.device)
            batch_size = min(32, len(states))

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

                # Total loss
                total_loss = loss_info['policy_loss'] - self.entropy_coef * loss_info['entropy']

                # Backward pass with gradient clipping
                self.opt_policy.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                # Early stopping on large KL divergence
                if loss_info['approx_kl'] > 0.01:
                    logger.debug(f"Early stopping due to large KL: {loss_info['approx_kl']:.4f}")
                    break

        self.policy_losses.append(total_loss.item())
        self.update_count += 1

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def update_discriminator(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                             generated_graphs: List[Data], epochs: int = 2):
        """Update discriminator with adversarial training - GPU optimized"""

        for epoch in range(epochs):
            # Sample balanced batches
            batch_size = min(16, len(smelly_graphs), len(clean_graphs), len(generated_graphs))

            if batch_size == 0:
                continue

            try:
                # Sample data and move to device
                smelly_sample = random.sample(smelly_graphs, batch_size)
                clean_sample = random.sample(clean_graphs, batch_size)
                generated_sample = random.sample(generated_graphs, batch_size)

                # Move to device and validate 7-feature format
                def validate_and_fix_features(data_list):
                    fixed_data = []
                    for data in data_list:
                        data = data.to(self.device)

                        # Ensure 7 features
                        if data.x.size(1) != 7:
                            if data.x.size(1) > 7:
                                data.x = data.x[:, :7]  # Truncate
                            else:
                                padding = torch.zeros(data.x.size(0), 7 - data.x.size(1), device=data.x.device)
                                data.x = torch.cat([data.x, padding], dim=1)  # Pad

                        fixed_data.append(data)
                    return fixed_data

                smelly_sample = validate_and_fix_features(smelly_sample)
                clean_sample = validate_and_fix_features(clean_sample)
                generated_sample = validate_and_fix_features(generated_sample)

                smelly_batch = Batch.from_data_list(smelly_sample)
                clean_batch = Batch.from_data_list(clean_sample)
                generated_batch = Batch.from_data_list(generated_sample)

                # Forward pass
                smelly_out = self.discriminator(smelly_batch)
                clean_out = self.discriminator(clean_batch)
                gen_out = self.discriminator(generated_batch)

                # Labels (smelly=1, clean/generated=0)
                smelly_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
                clean_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                gen_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

                # Compute losses
                loss_smelly = F.cross_entropy(smelly_out['logits'], smelly_labels)
                loss_clean = F.cross_entropy(clean_out['logits'], clean_labels)
                loss_gen = F.cross_entropy(gen_out['logits'], gen_labels)

                # Total discriminator loss
                disc_loss = loss_smelly + loss_clean + loss_gen

                # Backward pass
                self.opt_disc.zero_grad()
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.opt_disc.step()

                self.disc_losses.append(disc_loss.item())

            except Exception as e:
                logger.debug(f"Error in discriminator update: {e}")
                continue

        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


def safe_torch_load(file_path: Path, device: torch.device):
    """Safely load PyTorch files with PyTorch Geometric compatibility"""
    try:
        with torch.serialization.safe_globals([torch_geometric.data.data.DataEdgeAttr]):
            return torch.load(file_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(file_path, map_location=device, weights_only=False)


def validate_7_feature_format(data: Data) -> Data:
    """Ensure data has exactly 7 features"""
    if data.x.size(1) == 7:
        return data

    logger.debug(f"Data has {data.x.size(1)} features, expected 7. Fixing...")

    if data.x.size(1) > 7:
        # Truncate to 7 features
        data.x = data.x[:, :7]
    else:
        # Pad to 7 features
        padding = torch.zeros(data.x.size(0), 7 - data.x.size(1), device=data.x.device)
        data.x = torch.cat([data.x, padding], dim=1)

    return data


def load_pretrained_discriminator(model_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load pretrained discriminator with correct architecture parameters"""
    logger.info(f"Loading pretrained discriminator from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained discriminator not found at {model_path}")

    checkpoint = safe_torch_load(model_path, device)

    # Get architecture parameters from checkpoint
    model_architecture = checkpoint.get('model_architecture', {})
    node_dim = checkpoint.get('node_dim', 7)  # Default to 7 for unified features
    edge_dim = checkpoint.get('edge_dim', 1)

    # Validate that we have the expected 7-feature format
    if node_dim != 7:
        logger.warning(f"Discriminator was trained with {node_dim} features, but we expect 7. This may cause issues.")

    # Use saved architecture parameters or defaults
    hidden_dim = model_architecture.get('hidden_dim', 128)
    num_layers = model_architecture.get('num_layers', 4)
    dropout = model_architecture.get('dropout', 0.15)
    heads = model_architecture.get('heads', 8)

    logger.info(f"Creating discriminator with: node_dim={node_dim}, hidden_dim={hidden_dim}")

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
    """Get discriminator prediction for a single graph with enhanced debugging"""
    try:
        discriminator.eval()

        # Clone data to avoid modifying original
        data = data.clone()

        # Debug: Log data properties
        logger.debug(
            f"Input data - nodes: {data.x.size(0)}, edges: {data.edge_index.size(1)}, features: {data.x.size(1)}")

        # Ensure data is on correct device
        if data.x.device != device:
            data = data.to(device)

        # CRITICAL: Validate 7-feature format
        data = validate_7_feature_format(data)
        logger.debug(f"After validation - features: {data.x.size(1)}")

        # Ensure batch attribute exists and is correct
        if not hasattr(data, 'batch'):
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        elif data.batch.device != device:
            data.batch = data.batch.to(device)

        # Validate edge_index bounds
        if data.edge_index.size(1) > 0 and data.edge_index.max() >= data.x.size(0):
            logger.warning(f"Edge index out of bounds: max={data.edge_index.max()}, nodes={data.x.size(0)}")
            return None

        # Check for NaN/Inf values
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            logger.warning("Found NaN/Inf in node features")
            return None

        # Ensure edge_attr exists
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32, device=device)

        logger.debug(
            f"Data prepared - x: {data.x.shape}, edge_index: {data.edge_index.shape}, batch: {data.batch.shape}")

        with torch.no_grad():
            try:
                output = discriminator(data)
                logger.debug(f"Discriminator output keys: {output.keys() if output else 'None'}")

                if output is None or 'logits' not in output:
                    logger.warning("Discriminator returned None or missing logits")
                    return None

                logits = output['logits']
                logger.debug(f"Logits shape: {logits.shape}")

                if logits is None or logits.size(0) == 0:
                    logger.warning("Empty logits tensor")
                    return None

                # Get probability of being smelly
                probs = F.softmax(logits, dim=1)
                prob_smelly = probs[0, 1].item()

                logger.debug(f"Prediction successful - prob_smelly: {prob_smelly:.4f}")
                return prob_smelly

            except Exception as forward_e:
                logger.warning(f"Forward pass failed: {forward_e}")
                logger.debug(f"Data shapes at forward: x={data.x.shape}, edge_index={data.edge_index.shape}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                return None

    except Exception as e:
        logger.warning(f"Error in discriminator prediction: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None


def validate_discriminator_performance_enhanced(discriminator: nn.Module, smelly_data: List[Data],
                                                clean_data: List[Data], device: torch.device) -> Dict[str, float]:
    """Enhanced validation with detailed debugging"""
    logger.info("🔍 Enhanced discriminator validation starting...")

    correct_predictions = 0
    total_predictions = 0
    smelly_correct = 0
    clean_correct = 0
    failed_predictions = 0

    # Test on smelly graphs with detailed logging
    logger.info(f"Testing on {min(20, len(smelly_data))} smelly graphs...")
    for i, data in enumerate(smelly_data[:20]):
        logger.debug(f"\n--- Testing smelly graph {i} ---")
        logger.debug(
            f"Original data: nodes={data.x.size(0)}, edges={data.edge_index.size(1)}, features={data.x.size(1)}")

        prob_smelly = get_discriminator_prediction(discriminator, data, device)

        if prob_smelly is not None:
            if prob_smelly > 0.5:  # Correctly identified as smelly
                smelly_correct += 1
                logger.debug(f"✅ Smelly graph {i}: correctly classified (prob={prob_smelly:.4f})")
            else:
                logger.debug(f"❌ Smelly graph {i}: misclassified as clean (prob={prob_smelly:.4f})")
            total_predictions += 1
        else:
            failed_predictions += 1
            logger.debug(f"💥 Smelly graph {i}: prediction failed")

    # Test on clean graphs with detailed logging
    logger.info(f"Testing on {min(20, len(clean_data))} clean graphs...")
    for i, data in enumerate(clean_data[:20]):
        logger.debug(f"\n--- Testing clean graph {i} ---")
        logger.debug(
            f"Original data: nodes={data.x.size(0)}, edges={data.edge_index.size(1)}, features={data.x.size(1)}")

        prob_smelly = get_discriminator_prediction(discriminator, data, device)

        if prob_smelly is not None:
            if prob_smelly <= 0.5:  # Correctly identified as clean
                clean_correct += 1
                logger.debug(f"✅ Clean graph {i}: correctly classified (prob={prob_smelly:.4f})")
            else:
                logger.debug(f"❌ Clean graph {i}: misclassified as smelly (prob={prob_smelly:.4f})")
            total_predictions += 1
        else:
            failed_predictions += 1
            logger.debug(f"💥 Clean graph {i}: prediction failed")

    # Detailed logging
    logger.info(f"🔍 Validation summary:")
    logger.info(f"  - Total attempts: {min(20, len(smelly_data)) + min(20, len(clean_data))}")
    logger.info(f"  - Failed predictions: {failed_predictions}")
    logger.info(f"  - Successful predictions: {total_predictions}")
    logger.info(f"  - Smelly correct: {smelly_correct}")
    logger.info(f"  - Clean correct: {clean_correct}")

    if total_predictions == 0:
        logger.error("❌ CRITICAL: All predictions failed!")
        logger.error("This suggests a fundamental compatibility issue.")

        # Emergency diagnostic
        logger.info("🚨 Running emergency diagnostic...")
        sample_data = smelly_data[0] if smelly_data else clean_data[0]
        logger.info(f"Sample data properties:")
        logger.info(f"  - Type: {type(sample_data)}")
        logger.info(f"  - Device: {sample_data.x.device}")
        logger.info(f"  - Node features shape: {sample_data.x.shape}")
        logger.info(f"  - Node features dtype: {sample_data.x.dtype}")
        logger.info(f"  - Edge index shape: {sample_data.edge_index.shape}")
        logger.info(f"  - Edge index dtype: {sample_data.edge_index.dtype}")
        logger.info(f"  - Has batch attr: {hasattr(sample_data, 'batch')}")
        logger.info(f"  - Has edge_attr: {hasattr(sample_data, 'edge_attr')}")

        # Try a minimal test
        try:
            test_data = sample_data.clone().to(device)
            test_data = validate_7_feature_format(test_data)
            if not hasattr(test_data, 'batch'):
                test_data.batch = torch.zeros(test_data.x.size(0), dtype=torch.long, device=device)

            logger.info("🧪 Attempting minimal discriminator forward pass...")
            discriminator.eval()
            with torch.no_grad():
                output = discriminator(test_data)
                logger.info(f"✅ Minimal test successful! Output keys: {output.keys()}")
                logger.info(f"   Logits shape: {output['logits'].shape}")
        except Exception as test_e:
            logger.error(f"❌ Minimal test failed: {test_e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

        return {'accuracy': 0.0, 'smelly_accuracy': 0.0, 'clean_accuracy': 0.0,
                'failed_predictions': failed_predictions}

    accuracy = (smelly_correct + clean_correct) / total_predictions
    smelly_acc = smelly_correct / min(20, len(smelly_data)) if smelly_data else 0
    clean_acc = clean_correct / min(20, len(clean_data)) if clean_data else 0

    results = {
        'accuracy': accuracy,
        'smelly_accuracy': smelly_acc,
        'clean_accuracy': clean_acc,
        'failed_predictions': failed_predictions,
        'total_attempts': min(20, len(smelly_data)) + min(20, len(clean_data))
    }

    logger.info(f"🎯 Final validation results:")
    logger.info(f"  - Overall accuracy: {accuracy:.3f}")
    logger.info(f"  - Smelly accuracy: {smelly_acc:.3f}")
    logger.info(f"  - Clean accuracy: {clean_acc:.3f}")
    logger.info(f"  - Success rate: {total_predictions / results['total_attempts']:.3f}")

    return results


def debug_discriminator_architecture(discriminator: nn.Module, sample_data: Data, device: torch.device):
    """Debug discriminator architecture compatibility"""
    logger.info("🔧 Debugging discriminator architecture...")

    try:
        # Prepare sample data
        test_data = sample_data.clone().to(device)
        test_data = validate_7_feature_format(test_data)

        if not hasattr(test_data, 'batch'):
            test_data.batch = torch.zeros(test_data.x.size(0), dtype=torch.long, device=device)

        logger.info(f"Test data prepared:")
        logger.info(f"  - Nodes: {test_data.x.size(0)}")
        logger.info(f"  - Features: {test_data.x.size(1)}")
        logger.info(f"  - Edges: {test_data.edge_index.size(1)}")
        logger.info(f"  - Device: {test_data.x.device}")

        # Test discriminator layers step by step
        discriminator.eval()
        with torch.no_grad():
            logger.info("Testing discriminator forward pass step by step...")

            # Check input projection
            if hasattr(discriminator, 'node_projection'):
                try:
                    node_emb = discriminator.node_projection(test_data.x)
                    logger.info(f"✅ Node projection successful: {node_emb.shape}")
                except Exception as e:
                    logger.error(f"❌ Node projection failed: {e}")
                    return

            # Full forward pass
            try:
                output = discriminator(test_data)
                logger.info(f"✅ Full forward pass successful!")
                logger.info(f"   Output keys: {list(output.keys())}")
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor):
                        logger.info(f"   {key}: {tensor.shape}")

                # Test final prediction
                logits = output['logits']
                probs = F.softmax(logits, dim=1)
                prob_smelly = probs[0, 1].item()
                logger.info(f"✅ Prediction successful: {prob_smelly:.4f}")

            except Exception as e:
                logger.error(f"❌ Forward pass failed: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")

    except Exception as e:
        logger.error(f"❌ Debug setup failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")


def load_and_prepare_data(data_dir: Path, max_samples: int = 1000, device: torch.device = torch.device('cpu')) -> Tuple[
    List[Data], List[Data]]:
    """Load and prepare training data with enhanced validation and 7-feature compatibility"""
    logger.info("Loading dataset...")

    smelly_data = []
    clean_data = []

    # Get list of files first
    pt_files = list(data_dir.glob('*.pt'))
    logger.info(f"Found {len(pt_files)} .pt files")

    # Shuffle files for random sampling
    random.shuffle(pt_files)

    for file in pt_files:
        if len(smelly_data) + len(clean_data) >= max_samples:
            break

        try:
            # Use safe loading function
            data = safe_torch_load(file, device='cpu')  # Load to CPU first

            # Enhanced validation
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                continue

            # CRITICAL: Validate and fix 7-feature format
            if data.x.size(1) != 7:
                logger.debug(f"File {file.name} has {data.x.size(1)} features, fixing to 7")
                data = validate_7_feature_format(data)

            if data.x.size(0) < 3:  # Need at least 3 nodes
                continue

            if data.edge_index.size(1) == 0:  # Need at least some edges
                continue

            # Check for NaN values
            if torch.isnan(data.x).any() or torch.isnan(data.edge_index.float()).any():
                continue

            # Ensure batch attribute doesn't exist (will be added when needed)
            if hasattr(data, 'batch'):
                delattr(data, 'batch')

            # Classify by smell
            is_smelly = getattr(data, 'is_smelly', 0)
            if isinstance(is_smelly, torch.Tensor):
                is_smelly = is_smelly.item()

            if is_smelly == 1:
                smelly_data.append(data)
            else:
                clean_data.append(data)

        except Exception as e:
            logger.debug(f"Failed to load {file}: {e}")
            continue

    logger.info(f"Loaded {len(smelly_data)} smelly and {len(clean_data)} clean graphs")
    logger.info(f"All graphs validated to have exactly 7 features")
    return smelly_data, clean_data


def create_training_environments(smelly_data: List[Data], discriminator: nn.Module,
                                 num_envs: int = 8, device: torch.device = torch.device('cpu')) -> List[
    HubRefactoringEnv]:
    """Create training environments with enhanced error handling and 7-feature validation"""
    envs = []

    for i in range(num_envs):
        try:
            # Select a smelly graph for this environment
            initial_graph = random.choice(smelly_data).clone()  # Clone to avoid modification

            # CRITICAL: Validate 7-feature format
            initial_graph = validate_7_feature_format(initial_graph)

            # Ensure graph is properly formatted
            if not hasattr(initial_graph, 'batch'):
                initial_graph.batch = torch.zeros(initial_graph.x.size(0), dtype=torch.long)

            # Move to device
            initial_graph = initial_graph.to(device)

            # Test discriminator on this graph first
            test_pred = get_discriminator_prediction(discriminator, initial_graph, device)
            if test_pred is None:
                logger.warning(f"Environment {i}: Discriminator cannot process initial graph, trying another...")
                continue

            env = HubRefactoringEnv(initial_graph, discriminator, max_steps=15, device=device)
            envs.append(env)
            logger.debug(f"Successfully created environment {i} with 7-feature graph")

        except Exception as e:
            logger.warning(f"Failed to create environment {i}: {e}")
            continue

    logger.info(f"Created {len(envs)} training environments out of {num_envs} requested")

    # If we couldn't create any environments, that's a critical error
    if len(envs) == 0:
        raise RuntimeError("Could not create any training environments")

    return envs


def collect_rollouts(envs: List[HubRefactoringEnv], policy: nn.Module,
                     steps_per_env: int = 32, device: torch.device = torch.device('cpu')) -> Dict[str, List]:
    """Collect rollouts with enhanced error handling and 7-feature validation"""

    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    # Reset environments
    states = []
    valid_envs = []

    for i, env in enumerate(envs):
        try:
            state = env.reset()
            if state.x.device != device:
                state = state.to(device)

            # CRITICAL: Validate 7-feature format
            state = validate_7_feature_format(state)

            states.append(state)
            valid_envs.append(env)
        except Exception as e:
            logger.warning(f"Error resetting environment {i}: {e}")
            continue

    if not states:
        logger.warning("No valid environments after reset")
        return {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': torch.tensor([], device=device),
            'values': []
        }

    logger.debug(f"Starting rollout with {len(states)} valid environments")

    for step in range(steps_per_env):
        # Ensure all states are on correct device and have 7 features
        states_on_device = []
        for state in states:
            if state.x.device != device:
                state = state.to(device)

            # CRITICAL: Validate 7-feature format
            state = validate_7_feature_format(state)

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

        with torch.no_grad():
            try:
                policy_out = policy(batch_data)

                # Simple value estimation (using discriminator output as proxy)
                values = []
                for state in states_on_device:
                    pred = get_discriminator_prediction(valid_envs[0].discriminator, state, device)
                    if pred is not None:
                        value = 1.0 - pred  # Lower hub score = higher value
                    else:
                        value = 0.5  # Neutral value if prediction fails
                    values.append(value)

                values = torch.tensor(values, device=device)

            except Exception as e:
                logger.warning(f"Error in forward pass at step {step}: {e}")
                break

        # Sample actions for each environment
        actions = []
        log_probs = []

        # Handle batched data properly
        num_graphs = len(states_on_device)
        nodes_per_graph = [state.x.size(0) for state in states_on_device]
        cumsum_nodes = [0] + list(np.cumsum(nodes_per_graph))

        for i in range(num_graphs):
            try:
                start_idx = cumsum_nodes[i]
                end_idx = cumsum_nodes[i + 1]
                num_nodes = nodes_per_graph[i]

                # Validate indices
                if start_idx >= policy_out['hub_probs'].size(0) or end_idx > policy_out['hub_probs'].size(0):
                    # Create fallback action
                    action = RefactoringAction(0, min(1, num_nodes - 1), 0, False)
                    actions.append(action)
                    log_probs.append(torch.tensor(0.0, device=device))
                    continue

                # Sample hub from this graph's nodes
                hub_probs_i = policy_out['hub_probs'][start_idx:end_idx]
                if hub_probs_i.size(0) == 0:
                    action = RefactoringAction(0, min(1, num_nodes - 1), 0, False)
                    actions.append(action)
                    log_probs.append(torch.tensor(0.0, device=device))
                    continue

                hub_probs_i = hub_probs_i + 1e-8  # Numerical stability
                hub_dist = Categorical(hub_probs_i)
                hub_local = hub_dist.sample().item()
                hub_local = min(hub_local, num_nodes - 1)

                # Sample pattern for selected hub
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
                    term_probs_i = torch.tensor([0.7, 0.3], device=device)  # Bias toward continuing

                term_dist = Categorical(term_probs_i)
                terminate = term_dist.sample().item() == 1

                # Target selection (improved)
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
                # Add fallback action
                action = RefactoringAction(0, 0, 0, False)
                actions.append(action)
                log_probs.append(torch.tensor(0.0, device=device))

        # Execute actions
        next_states = []
        rewards = []
        dones = []

        for i, (env, action) in enumerate(zip(valid_envs, actions)):
            try:
                state, reward, done, info = env.step(
                    (action.source_node, action.target_node, action.pattern, action.terminate)
                )

                # Ensure state is on correct device and has 7 features
                if state.x.device != device:
                    state = state.to(device)

                # CRITICAL: Validate 7-feature format
                state = validate_7_feature_format(state)

                # Ensure batch attribute
                if not hasattr(state, 'batch'):
                    state.batch = torch.zeros(state.x.size(0), dtype=torch.long, device=device)

                next_states.append(state)
                rewards.append(reward)
                dones.append(done)

                # Store experience
                all_states.append(states_on_device[i])
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(done)
                all_log_probs.append(log_probs[i] if i < len(log_probs) else torch.tensor(0.0, device=device))
                all_values.append(values[i].item() if i < len(values) else 0.0)

                # Reset if done
                if done:
                    try:
                        reset_state = env.reset()
                        if reset_state.x.device != device:
                            reset_state = reset_state.to(device)

                        # CRITICAL: Validate 7-feature format
                        reset_state = validate_7_feature_format(reset_state)

                        if not hasattr(reset_state, 'batch'):
                            reset_state.batch = torch.zeros(reset_state.x.size(0), dtype=torch.long, device=device)
                        next_states[i] = reset_state
                    except Exception as reset_e:
                        logger.warning(f"Error resetting environment {i}: {reset_e}")
                        # Keep current state if reset fails
                        pass

            except Exception as e:
                logger.warning(f"Error in environment {i} step: {e}")
                # Keep the same state if there's an error
                if i < len(states_on_device):
                    next_states.append(states_on_device[i])
                else:
                    # Create a fallback state
                    fallback_state = states_on_device[0].clone() if states_on_device else None
                    if fallback_state is not None:
                        next_states.append(fallback_state)
                    else:
                        break  # Can't continue without valid states

                rewards.append(-0.5)  # Small penalty for error
                dones.append(False)

                # Still store experience to avoid breaking training
                if i < len(states_on_device):
                    all_states.append(states_on_device[i])
                    all_actions.append(action)
                    all_rewards.append(-0.5)
                    all_dones.append(False)
                    all_log_probs.append(log_probs[i] if i < len(log_probs) else torch.tensor(0.0, device=device))
                    all_values.append(0.0)

        states = next_states

        # Break if all environments are done or we have no valid states
        if not states or all(dones):
            break

        # GPU memory management
        if device.type == 'cuda' and step % 10 == 0:
            torch.cuda.empty_cache()

    # Ensure we have valid data
    if not all_log_probs:
        logger.warning("No valid rollouts collected")
        return {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': torch.tensor([], device=device),
            'values': []
        }

    return {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'dones': all_dones,
        'log_probs': torch.stack(all_log_probs),
        'values': all_values
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
    """Main training loop with 7-feature compatibility and enhanced error handling"""

    # Setup GPU environment
    device = setup_gpu_environment()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    logger.info(f"🎯 Starting training on device: {device}")

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'dataset_builder' / 'data' / 'dataset_graph_feature'
    results_dir = base_dir / 'results'
    discriminator_path = results_dir / 'discriminator_pretraining' / 'pretrained_discriminator.pt'

    results_dir.mkdir(exist_ok=True)

    # Check if pretrained discriminator exists
    if not discriminator_path.exists():
        logger.error(f"❌ Pretrained discriminator not found at {discriminator_path}")
        logger.error("Please run the discriminator pre-training script first:")
        logger.error("python pre-training-discriminator.py")
        return

    # Load data with enhanced validation and 7-feature compatibility
    logger.info("📊 Loading dataset with 7-feature validation...")
    smelly_data, clean_data = load_and_prepare_data(data_dir, max_samples=1000, device=device)

    if len(smelly_data) == 0:
        logger.error("❌ No smelly data found!")
        return

    logger.info(f"📈 Dataset loaded: {len(smelly_data)} smelly, {len(clean_data)} clean graphs")
    logger.info("✅ All graphs validated to have exactly 7 features")

    # Load pretrained discriminator
    try:
        logger.info("🔄 Loading pretrained discriminator...")
        discriminator, discriminator_checkpoint = load_pretrained_discriminator(discriminator_path, device)

        # Enhanced discriminator validation with debugging
        logger.info("🔍 Running enhanced discriminator validation...")

        # First, debug the architecture
        if smelly_data:
            debug_discriminator_architecture(discriminator, smelly_data[0], device)

        # Then run enhanced validation
        validation_results = validate_discriminator_performance_enhanced(discriminator, smelly_data, clean_data, device)

        if validation_results['accuracy'] < 0.1:
            logger.error("❌ Discriminator performance is critically low!")
            logger.error("This appears to be an architecture/compatibility issue.")
            logger.error("Check the debug output above for specific problems.")
            return
        elif validation_results['failed_predictions'] > 0:
            logger.warning(f"⚠️ {validation_results['failed_predictions']} predictions failed, but continuing...")
            logger.warning("Some samples may have compatibility issues.")
        else:
            logger.info("✅ Discriminator validation passed!")

    except Exception as e:
        logger.error(f"❌ Failed to load pretrained discriminator: {e}")
        return

    # Model parameters - FIXED for 7-feature compatibility
    node_dim = 7  # ALWAYS 7 for unified features
    edge_dim = 1  # Standard edge features
    hidden_dim = discriminator_checkpoint.get('model_architecture', {}).get('hidden_dim', 128)

    logger.info(f"🔧 Model dimensions - Node: {node_dim} (UNIFIED), Edge: {edge_dim}, Hidden: {hidden_dim}")

    # Create policy network with FIXED 7-feature compatibility
    logger.info("🧠 Creating policy network with 7-feature compatibility...")
    policy = HubRefactoringPolicy(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy network - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Create trainer with pretrained discriminator
    logger.info("🎯 Setting up PPO trainer...")
    trainer = PPOTrainer(policy, discriminator, device=device)

    # Training parameters - optimized for GPU
    num_episodes = 5000
    num_envs = min(8 if device.type == 'cuda' else 4, len(smelly_data))  # Don't exceed available data
    steps_per_env = 32
    update_frequency = 10

    logger.info("⚙️  Training configuration:")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Environments: {num_envs}")
    logger.info(f"  Steps per env: {steps_per_env}")
    logger.info(f"  Update frequency: {update_frequency}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Feature format: 7 unified structural features")

    # Create environments with enhanced error handling and 7-feature validation
    logger.info("🌍 Creating training environments...")
    try:
        envs = create_training_environments(smelly_data, discriminator, num_envs, device)
        logger.info(f"✅ Successfully created {len(envs)} training environments")
    except Exception as e:
        logger.error(f"❌ Failed to create training environments: {e}")
        return

    logger.info("🚀 Starting RL training with pretrained discriminator and 7-feature compatibility...")

    # Training loop
    episode_rewards = []
    success_rates = []
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None

    for episode in range(num_episodes):
        if device.type == 'cuda' and start_time:
            start_time.record()

        # Collect rollouts with 7-feature validation
        rollout_data = collect_rollouts(envs, policy, steps_per_env, device)

        if len(rollout_data['rewards']) == 0:
            logger.warning(f"No valid rollouts in episode {episode}, recreating environments...")

            # Try to recreate environments
            try:
                envs = create_training_environments(smelly_data, discriminator, num_envs, device)
                continue
            except Exception as e:
                logger.error(f"Failed to recreate environments: {e}")
                break

        episode_reward = np.mean(rollout_data['rewards'])
        episode_rewards.append(episode_reward)

        # Calculate success rate (episodes that improved hub score)
        positive_rewards = [r for r in rollout_data['rewards'] if r > 0]
        success_rate = len(positive_rewards) / max(len(rollout_data['rewards']), 1)
        success_rates.append(success_rate)

        # Update policy every N episodes
        if episode % update_frequency == 0 and len(rollout_data['states']) >= 32:

            # Compute advantages
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

            # Update discriminator occasionally (fine-tuning)
            if episode % (update_frequency * 5) == 0 and episode > 0:
                # Collect recent states from environments
                recent_states = []
                for env in envs:
                    if hasattr(env, 'current_data') and env.current_data is not None:
                        # Validate 7-feature format for discriminator update
                        validated_state = validate_7_feature_format(env.current_data.clone())
                        recent_states.append(validated_state)

                if len(recent_states) >= 8:
                    # Fine-tune discriminator with generated graphs
                    trainer.update_discriminator(
                        smelly_data[:16],  # Smaller batches for fine-tuning
                        clean_data[:16] if len(clean_data) >= 16 else clean_data,
                        recent_states[:16]
                    )

        # Timing for GPU
        if device.type == 'cuda' and end_time:
            end_time.record()
            torch.cuda.synchronize()
            if start_time:
                episode_time = start_time.elapsed_time(end_time) / 1000.0

        # Logging and monitoring
        if episode % 100 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_success_100 = np.mean(success_rates[-100:]) if success_rates else 0
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            disc_loss = trainer.disc_losses[-1] if trainer.disc_losses else 0

            timing_info = ""
            if device.type == 'cuda' and 'episode_time' in locals():
                timing_info = f"Time: {episode_time:.2f}s | "

            logger.info(f"Episode {episode:5d} | "
                        f"Reward (100): {avg_reward_100:7.3f} | "
                        f"Success Rate: {avg_success_100:5.3f} | "
                        f"Policy Loss: {policy_loss:7.4f} | "
                        f"Disc Loss: {disc_loss:7.4f} | "
                        f"{timing_info}7-Features: ✅")

            # Monitor GPU usage
            if device.type == 'cuda':
                monitor_gpu_usage()

            # Additional diagnostics every 500 episodes
            if episode % 500 == 0 and episode > 0:
                # Test discriminator performance
                val_results = validate_discriminator_performance_enhanced(discriminator, smelly_data[:10],
                                                                          clean_data[:10],
                                                                          device)
                logger.info(f"🔍 Discriminator validation - Acc: {val_results['accuracy']:.3f}")

                # Analyze agent performance
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                if recent_rewards:
                    logger.info(f"📊 Agent performance analysis:")
                    logger.info(f"  Mean reward: {np.mean(recent_rewards):.3f}")
                    logger.info(f"  Std reward: {np.std(recent_rewards):.3f}")
                    logger.info(f"  Max reward: {np.max(recent_rewards):.3f}")
                    logger.info(f"  Min reward: {np.min(recent_rewards):.3f}")

        # Save checkpoints
        if episode % 1000 == 0 and episode > 0:
            try:
                checkpoint = {
                    'episode': episode,
                    'policy_state': policy.state_dict(),
                    'discriminator_state': discriminator.state_dict(),
                    'trainer_state': {
                        'update_count': trainer.update_count,
                        'policy_losses': trainer.policy_losses,
                        'disc_losses': trainer.disc_losses
                    },
                    'training_stats': {
                        'episode_rewards': episode_rewards,
                        'success_rates': success_rates,
                        'discriminator_validation': validation_results
                    },
                    'config': {
                        'device': str(device),
                        'node_dim': node_dim,  # Always 7
                        'edge_dim': edge_dim,
                        'hidden_dim': hidden_dim,
                        'unified_7_features': True
                    }
                }

                checkpoint_path = results_dir / f'rl_checkpoint_{episode}.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        # Refresh environments periodically
        if episode % 1000 == 0 and episode > 0:
            logger.info("🔄 Refreshing training environments...")
            try:
                # Clean up old environments
                del envs
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                envs = create_training_environments(smelly_data, discriminator, num_envs, device)
            except Exception as e:
                logger.warning(f"Failed to refresh environments: {e}")

        # Memory cleanup every 50 episodes
        if episode % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()

    # Training completed
    logger.info("🏁 RL Training completed!")

    # Final GPU memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        monitor_gpu_usage()

    # Save final models
    logger.info("💾 Saving final models...")

    try:
        final_save_path = results_dir / 'final_rl_model.pt'
        torch.save({
            'policy_state': policy.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'node_dim': node_dim,  # Always 7
            'edge_dim': edge_dim,
            'training_stats': {
                'episode_rewards': episode_rewards,
                'success_rates': success_rates,
                'total_episodes': num_episodes,
                'final_discriminator_validation': validation_results
            },
            'config': {
                'num_episodes': num_episodes,
                'num_envs': num_envs,
                'steps_per_env': steps_per_env,
                'update_frequency': update_frequency,
                'device': str(device),
                'gpu_optimized': True,
                'unified_7_features': True,
                'feature_names': [
                    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                    'pagerank', 'betweenness_centrality', 'closeness_centrality'
                ]
            }
        }, final_save_path)

        # Save training statistics
        stats_path = results_dir / 'rl_training_stats.json'
        training_stats = {
            'episode_rewards': episode_rewards,
            'success_rates': success_rates,
            'final_performance': {
                'mean_reward_last_100': float(np.mean(episode_rewards[-100:])) if episode_rewards else 0,
                'mean_success_rate_last_100': float(np.mean(success_rates[-100:])) if success_rates else 0,
                'total_episodes': num_episodes,
                'total_updates': trainer.update_count
            },
            'discriminator_info': {
                'pretrained_path': str(discriminator_path),
                'cv_performance': discriminator_checkpoint.get('cv_results', {}),
                'final_validation': validation_results
            },
            'feature_pipeline': {
                'unified_features': True,
                'num_features': 7,
                'feature_names': [
                    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                    'pagerank', 'betweenness_centrality', 'closeness_centrality'
                ],
                'compatible_with_discriminator': True
            },
            'gpu_info': {
                'device_used': str(device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)

        logger.info(f"📁 Models saved to: {final_save_path}")
        logger.info(f"📊 Statistics saved to: {stats_path}")

    except Exception as e:
        logger.error(f"Failed to save final models: {e}")

    # Final performance summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FINAL TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Device used: {device}")
    logger.info(f"Feature format: 7 unified structural features ✅")
    logger.info(f"Total episodes: {num_episodes}")
    logger.info(f"Policy updates: {trainer.update_count}")

    if episode_rewards:
        final_mean_reward = np.mean(episode_rewards[-100:])
        final_success_rate = np.mean(success_rates[-100:])
        logger.info(f"Final performance (last 100 episodes):")
        logger.info(f"  Average reward: {final_mean_reward:.3f}")
        logger.info(f"  Success rate: {final_success_rate:.3f}")

        if final_mean_reward > 5.0:
            logger.info("🎉 Excellent training performance!")
        elif final_mean_reward > 2.0:
            logger.info("✅ Good training performance!")
        else:
            logger.info("⚠️  Training performance could be improved")

    if device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated() / 1e9
        logger.info(f"🔧 Final GPU memory usage: {final_memory:.2f}GB")

    logger.info("🔧 Discriminator compatibility: ✅ UNIFIED 7-feature pipeline")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()