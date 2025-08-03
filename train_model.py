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
import json
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data

from discriminator import HubDetectionDiscriminator
from policy_network import HubRefactoringPolicy
from rl_gym import HubRefactoringEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define RefactoringAction class since it's missing
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
    """Experience buffer for PPO training"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Dict[str, Any]):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """PPO trainer with adversarial discriminator training"""

    def __init__(self, policy: nn.Module, discriminator: nn.Module,
                 lr_policy: float = 3e-4, lr_disc: float = 1e-4,
                 device: torch.device = torch.device('cpu')):
        self.device = device
        self.policy = policy.to(device)
        self.discriminator = discriminator.to(device)

        # Optimizers
        self.opt_policy = optim.Adam(policy.parameters(), lr=lr_policy, eps=1e-5)
        self.opt_disc = optim.Adam(discriminator.parameters(), lr=lr_disc, eps=1e-5)

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

    def compute_gae(self, rewards: List[float], values: List[float],
                    dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_policy_loss(self, states: List[Data], actions: List[RefactoringAction],
                            old_log_probs: torch.Tensor, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute PPO policy loss"""

        # Batch states
        batch_data = Batch.from_data_list(states).to(self.device)
        policy_out = self.policy(batch_data)

        # Compute current log probabilities
        log_probs = []
        entropies = []

        # Handle batched data properly
        num_graphs = len(states)
        nodes_per_graph = [state.x.size(0) for state in states]
        cumsum_nodes = [0] + list(np.cumsum(nodes_per_graph))

        for i, action in enumerate(actions):
            start_idx = cumsum_nodes[i]
            end_idx = cumsum_nodes[i + 1]

            # Hub selection
            hub_probs_i = policy_out['hub_probs'][start_idx:end_idx]
            hub_dist = Categorical(hub_probs_i)
            log_prob_hub = hub_dist.log_prob(torch.tensor(action.source_node, device=self.device))
            entropy_hub = hub_dist.entropy()

            # Pattern selection
            pattern_probs_i = policy_out['pattern_probs'][start_idx + action.source_node]
            pattern_dist = Categorical(pattern_probs_i)
            log_prob_pattern = pattern_dist.log_prob(torch.tensor(action.pattern, device=self.device))
            entropy_pattern = pattern_dist.entropy()

            # Termination
            term_probs_i = policy_out['term_probs'][i] if i < policy_out['term_probs'].size(0) else \
                policy_out['term_probs'][0]
            term_dist = Categorical(term_probs_i)
            log_prob_term = term_dist.log_prob(torch.tensor(int(action.terminate), device=self.device))
            entropy_term = term_dist.entropy()

            total_log_prob = log_prob_hub + log_prob_pattern + log_prob_term
            total_entropy = entropy_hub + entropy_pattern + entropy_term

            log_probs.append(total_log_prob)
            entropies.append(total_entropy)

        log_probs = torch.stack(log_probs)
        entropy = torch.stack(entropies).mean()

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
        """Update policy network using PPO"""

        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            batch_size = min(32, len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

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

                # Backward pass
                self.opt_policy.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                # Early stopping on large KL divergence
                if loss_info['approx_kl'] > 0.01:
                    logger.warning(f"Early stopping due to large KL: {loss_info['approx_kl']:.4f}")
                    break

        self.policy_losses.append(total_loss.item())
        self.update_count += 1

    def update_discriminator(self, smelly_graphs: List[Data], clean_graphs: List[Data],
                             generated_graphs: List[Data], epochs: int = 2):
        """Update discriminator with adversarial training"""

        for epoch in range(epochs):
            # Sample balanced batches
            batch_size = min(16, len(smelly_graphs), len(clean_graphs), len(generated_graphs))

            if batch_size == 0:
                continue

            # Sample data
            smelly_batch = Batch.from_data_list(
                random.sample(smelly_graphs, batch_size)
            ).to(self.device)

            clean_batch = Batch.from_data_list(
                random.sample(clean_graphs, batch_size)
            ).to(self.device)

            generated_batch = Batch.from_data_list(
                random.sample(generated_graphs, batch_size)
            ).to(self.device)

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


def load_pretrained_discriminator(model_path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load pretrained discriminator"""
    logger.info(f"Loading pretrained discriminator from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained discriminator not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Create model with saved dimensions
    discriminator = HubDetectionDiscriminator(
        node_dim=checkpoint['node_dim'],
        edge_dim=checkpoint['edge_dim'],
        hidden_dim=128,
        num_layers=3
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


def validate_discriminator_performance(discriminator: nn.Module, smelly_data: List[Data],
                                       clean_data: List[Data], device: torch.device) -> Dict[str, float]:
    """Validate discriminator performance on current dataset"""
    logger.info("Validating discriminator performance...")

    discriminator.eval()
    correct_predictions = 0
    total_predictions = 0
    smelly_correct = 0
    clean_correct = 0

    with torch.no_grad():
        # Test on smelly graphs
        for data in smelly_data[:50]:  # Test on subset
            try:
                data = data.to(device)
                output = discriminator(data)
                prob_smelly = F.softmax(output['logits'], dim=1)[0, 1].item()

                if prob_smelly > 0.5:  # Correctly identified as smelly
                    smelly_correct += 1
                total_predictions += 1
            except:
                continue

        # Test on clean graphs
        for data in clean_data[:50]:  # Test on subset
            try:
                data = data.to(device)
                output = discriminator(data)
                prob_smelly = F.softmax(output['logits'], dim=1)[0, 1].item()

                if prob_smelly <= 0.5:  # Correctly identified as clean
                    clean_correct += 1
                total_predictions += 1
            except:
                continue

    if total_predictions == 0:
        logger.warning("Could not validate discriminator - no valid predictions")
        return {'accuracy': 0.0, 'smelly_accuracy': 0.0, 'clean_accuracy': 0.0}

    accuracy = (smelly_correct + clean_correct) / total_predictions
    smelly_acc = smelly_correct / min(50, len(smelly_data)) if smelly_data else 0
    clean_acc = clean_correct / min(50, len(clean_data)) if clean_data else 0

    results = {
        'accuracy': accuracy,
        'smelly_accuracy': smelly_acc,
        'clean_accuracy': clean_acc
    }

    logger.info(f"Discriminator validation: Overall: {accuracy:.3f}, Smelly: {smelly_acc:.3f}, Clean: {clean_acc:.3f}")

    return results


def load_and_prepare_data(data_dir: Path, max_samples: int = 1000) -> Tuple[List[Data], List[Data]]:
    """Load and prepare training data"""
    logger.info("Loading dataset...")

    smelly_data = []
    clean_data = []

    for file in data_dir.glob('*.pt'):
        if len(smelly_data) + len(clean_data) >= max_samples:
            break

        try:
            data = torch.load(file, map_location='cpu')

            # Validate data structure
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                continue

            # Check for required features
            if data.x.size(1) < 9:  # Ensure enough features
                continue

            # Classify by smell
            is_smelly = getattr(data, 'is_smelly', 0)
            if isinstance(is_smelly, torch.Tensor):
                is_smelly = is_smelly.item()

            if is_smelly == 1:
                smelly_data.append(data)
            else:
                clean_data.append(data)

        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
            continue

    logger.info(f"Loaded {len(smelly_data)} smelly and {len(clean_data)} clean graphs")
    return smelly_data, clean_data


def create_training_environments(smelly_data: List[Data], discriminator: nn.Module,
                                 num_envs: int = 8, device: torch.device = torch.device('cpu')) -> List[
    HubRefactoringEnv]:
    """Create training environments"""
    envs = []
    for i in range(num_envs):
        # Select a smelly graph for this environment
        initial_graph = random.choice(smelly_data)
        env = HubRefactoringEnv(initial_graph, discriminator, max_steps=15, device=device)
        envs.append(env)
    return envs


def collect_rollouts(envs: List[HubRefactoringEnv], policy: nn.Module,
                     steps_per_env: int = 32, device: torch.device = torch.device('cpu')) -> Dict[str, List]:
    """Collect rollouts from environments"""

    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    all_log_probs = []
    all_values = []

    # Reset environments
    states = [env.reset() for env in envs]

    for step in range(steps_per_env):
        # Batch states
        try:
            batch_data = Batch.from_data_list(states).to(device)
        except Exception as e:
            logger.warning(f"Failed to batch data: {e}")
            break

        with torch.no_grad():
            policy_out = policy(batch_data)

            # Simple value estimation (using discriminator)
            disc_out = envs[0].discriminator(batch_data)
            values = 1.0 - F.softmax(disc_out['logits'], dim=1)[:, 1]  # Lower hub score = higher value

        # Sample actions for each environment
        actions = []
        log_probs = []

        # Handle batched data properly
        num_graphs = len(states)
        nodes_per_graph = [state.x.size(0) for state in states]

        # Create proper node indices for each graph
        cumsum_nodes = [0] + list(np.cumsum(nodes_per_graph))

        for i in range(num_graphs):
            start_idx = cumsum_nodes[i]
            end_idx = cumsum_nodes[i + 1]
            num_nodes = nodes_per_graph[i]

            # Sample hub from this graph's nodes
            hub_probs_i = policy_out['hub_probs'][start_idx:end_idx]
            hub_dist = Categorical(hub_probs_i)
            hub_local = hub_dist.sample().item()  # Local index within this graph

            # Sample pattern for selected hub
            pattern_probs_i = policy_out['pattern_probs'][start_idx + hub_local]
            pattern_dist = Categorical(pattern_probs_i)
            pattern = pattern_dist.sample().item()

            # Sample termination
            term_probs_i = policy_out['term_probs'][i] if i < policy_out['term_probs'].size(0) else \
                policy_out['term_probs'][0]
            term_dist = Categorical(term_probs_i)
            terminate = term_dist.sample().item() == 1

            # Target selection (improved)
            if num_nodes > 1:
                # Select different target from hub
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

        # Execute actions
        next_states = []
        rewards = []
        dones = []

        for i, (env, action) in enumerate(zip(envs, actions)):
            try:
                state, reward, done, info = env.step(
                    (action.source_node, action.target_node, action.pattern, action.terminate)
                )

                next_states.append(state)
                rewards.append(reward)
                dones.append(done)

                # Store experience
                all_states.append(states[i])
                all_actions.append(action)
                all_rewards.append(reward)
                all_dones.append(done)
                all_log_probs.append(log_probs[i])
                all_values.append(values[i].item() if i < len(values) else 0.0)

                # Reset if done
                if done:
                    next_states[i] = env.reset()

            except Exception as e:
                logger.warning(f"Error in environment {i}: {e}")
                # Keep the same state if there's an error
                next_states.append(states[i])
                rewards.append(-1.0)  # Small penalty for error
                dones.append(False)

                # Still store experience to avoid breaking the training
                all_states.append(states[i])
                all_actions.append(action)
                all_rewards.append(-1.0)
                all_dones.append(False)
                all_log_probs.append(log_probs[i])
                all_values.append(0.0)

        states = next_states

        # Break if all environments are done
        if all(dones):
            break

    # Ensure we have valid data
    if not all_log_probs:
        logger.warning("No valid rollouts collected")
        return {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': torch.tensor([]),
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


def main():
    """Main training loop with pretrained discriminator"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

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
        logger.error("python pretrain_discriminator.py")
        return

    # Load data
    smelly_data, clean_data = load_and_prepare_data(data_dir, max_samples=1000)

    if len(smelly_data) == 0:
        logger.error("No smelly data found!")
        return

    # Load pretrained discriminator
    try:
        discriminator, discriminator_checkpoint = load_pretrained_discriminator(discriminator_path, device)

        # Validate discriminator on current dataset
        validation_results = validate_discriminator_performance(discriminator, smelly_data, clean_data, device)

        if validation_results['accuracy'] < 0.6:
            logger.warning("⚠️  Discriminator performance is low on current dataset!")
            logger.warning("Consider re-training the discriminator or checking data quality.")
        else:
            logger.info("✅ Discriminator validation passed!")

    except Exception as e:
        logger.error(f"❌ Failed to load pretrained discriminator: {e}")
        return

    # Model parameters from discriminator checkpoint
    node_dim = discriminator_checkpoint['node_dim']
    edge_dim = discriminator_checkpoint['edge_dim']
    hidden_dim = 128

    logger.info(f"Model dimensions - Node: {node_dim}, Edge: {edge_dim}")

    # Create policy network
    logger.info("Creating policy network...")
    policy = HubRefactoringPolicy(node_dim, edge_dim, hidden_dim).to(device)

    # Create trainer with pretrained discriminator
    trainer = PPOTrainer(policy, discriminator, device=device)

    # Training parameters
    num_episodes = 5000
    num_envs = 8
    steps_per_env = 32
    update_frequency = 10

    logger.info("Training configuration:")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Environments: {num_envs}")
    logger.info(f"  Steps per env: {steps_per_env}")
    logger.info(f"  Update frequency: {update_frequency}")

    # Create environments with pretrained discriminator
    envs = create_training_environments(smelly_data, discriminator, num_envs, device)

    logger.info("🚀 Starting RL training with pretrained discriminator...")

    # Training loop
    episode_rewards = []
    success_rates = []

    for episode in range(num_episodes):
        # Collect rollouts
        rollout_data = collect_rollouts(envs, policy, steps_per_env, device)

        if len(rollout_data['rewards']) == 0:
            logger.warning(f"No valid rollouts in episode {episode}")
            continue

        episode_reward = np.mean(rollout_data['rewards'])
        episode_rewards.append(episode_reward)

        # Calculate success rate (episodes that improved hub score)
        positive_rewards = [r for r in rollout_data['rewards'] if r > 0]
        success_rate = len(positive_rewards) / max(len(rollout_data['rewards']), 1)
        success_rates.append(success_rate)

        # Update policy every N episodes
        if episode % update_frequency == 0 and len(rollout_data['states']) >= 64:

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
            if episode % (update_frequency * 3) == 0 and episode > 0:
                # Collect recent states from environments
                recent_states = [env.current_data for env in envs if env.current_data is not None]

                if len(recent_states) >= 8:
                    # Fine-tune discriminator with generated graphs
                    trainer.update_discriminator(
                        smelly_data[:16],  # Smaller batches for fine-tuning
                        clean_data[:16] if len(clean_data) >= 16 else clean_data,
                        recent_states[:16]
                    )

        # Logging and monitoring
        if episode % 100 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_success_100 = np.mean(success_rates[-100:]) if success_rates else 0
            policy_loss = trainer.policy_losses[-1] if trainer.policy_losses else 0
            disc_loss = trainer.disc_losses[-1] if trainer.disc_losses else 0

            logger.info(f"Episode {episode:5d} | "
                        f"Reward (100): {avg_reward_100:7.3f} | "
                        f"Success Rate: {avg_success_100:5.3f} | "
                        f"Policy Loss: {policy_loss:7.4f} | "
                        f"Disc Loss: {disc_loss:7.4f}")

            # Additional diagnostics every 500 episodes
            if episode % 500 == 0 and episode > 0:
                # Test discriminator performance
                val_results = validate_discriminator_performance(discriminator, smelly_data[:20], clean_data[:20],
                                                                 device)
                logger.info(f"Discriminator validation - Acc: {val_results['accuracy']:.3f}")

                # Analyze agent performance
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                if recent_rewards:
                    logger.info(f"Agent performance analysis:")
                    logger.info(f"  Mean reward: {np.mean(recent_rewards):.3f}")
                    logger.info(f"  Std reward: {np.std(recent_rewards):.3f}")
                    logger.info(f"  Max reward: {np.max(recent_rewards):.3f}")
                    logger.info(f"  Min reward: {np.min(recent_rewards):.3f}")

        # Save checkpoints
        if episode % 1000 == 0 and episode > 0:
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
                }
            }

            checkpoint_path = results_dir / f'rl_checkpoint_{episode}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"💾 Saved checkpoint: {checkpoint_path}")

        # Refresh environments periodically
        if episode % 500 == 0 and episode > 0:
            envs = create_training_environments(smelly_data, discriminator, num_envs, device)
            logger.info("🔄 Refreshed training environments")

    # Training completed
    logger.info("🏁 RL Training completed!")

    # Save final models
    logger.info("💾 Saving final models...")

    final_save_path = results_dir / 'final_rl_model.pt'
    torch.save({
        'policy_state': policy.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'node_dim': node_dim,
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
            'update_frequency': update_frequency
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
        }
    }

    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)

    # Final performance summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 FINAL TRAINING SUMMARY")
    logger.info("=" * 60)
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

    logger.info(f"📁 Models saved to: {final_save_path}")
    logger.info(f"📊 Statistics saved to: {stats_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()