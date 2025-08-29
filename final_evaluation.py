#!/usr/bin/env python3
"""
Graph Refactoring PPO - Comprehensive Evaluation Script
=======================================================

Comprehensive evaluation and visualization of the trained PPO-based
Graph Refactoring model, including:
- Performance metrics visualization
- Before/After graph comparisons
- Success rate analysis
- Interactive exploration of results
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Import project modules (PPO-based)
from rl_gym import PPORefactorEnv
from actor_critic_models import create_actor_critic

# Try to import discriminator (may not be available)
try:
    from discriminator import create_discriminator
except ImportError:
    create_discriminator = None
    print("Warning: Discriminator module not available")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Graph Refactoring PPO - Evaluation Dashboard")
print("=" * 60)


# =============================================================================
# CONFIGURATION
# =============================================================================

class EvaluationConfig:
    """Configuration for PPO evaluation - MATCHED TO TRAINING CONFIG"""

    def __init__(self):
        # Paths - Updated for PPO structure
        self.model_path = "results/ppo_training/best_model.pt"
        self.discriminator_path = "results/discriminator_pretraining/pretrained_discriminator.pt"
        self.data_path = "data_builder/dataset/graph_features"
        self.results_dir = "results/ppo_evaluation"

        # Evaluation parameters
        self.num_eval_episodes = 50
        self.num_visualization_episodes = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # PPO-specific parameters - MATCHED TO PPOConfig
        self.max_episode_steps = 10  # Changed from 20 to match training
        self.reward_weights = {
            'hub_weight': 5.0,  # Match PPOConfig default
            'step_valid': 0.01,
            'step_invalid': -0.1,
            'time_penalty': -0.02,
            'early_stop_penalty': -0.5,
            'cycle_penalty': -0.2,
            'duplicate_penalty': -0.1,
            'adversarial_weight': 0.5,  # Match PPOConfig default
            'patience': 15
        }

        # Visualization parameters (unchanged)
        self.figure_size = (15, 10)
        self.node_size_factor = 300
        self.edge_width = 2.0
        self.colors = {
            'original_nodes': '#3498db',
            'original_edges': '#2c3e50',
            'added_nodes': '#e74c3c',
            'added_edges': '#e74c3c',
            'removed_edges': '#95a5a6',
            'hub_node': '#f39c12',
            'modified_nodes': '#9b59b6'
        }


eval_config = EvaluationConfig()


# =============================================================================
# MODEL LOADING UTILITIES (UPDATED FOR PPO)
# =============================================================================

def load_trained_ppo_model(model_path: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained PPO model"""

    print(f"Loading PPO model from {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration (PPO structure)
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Fallback configuration for PPO
        model_config = {
            'node_dim': 7,
            'hidden_dim': 128,
            'num_layers': 3,
            'num_actions': 7,
            'global_features_dim': 4,  # PPO uses 4 global features
            'dropout': 0.2,
            'shared_encoder': True  # PPO requires shared encoder
        }
        print("Using fallback PPO model configuration")

    # Create and load model
    model = create_actor_critic(model_config).to(device)

    # Load state dict with proper key handling
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback for different checkpoint structures
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"PPO model loaded successfully")
    print(f"   Architecture: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Shared encoder: {model_config.get('shared_encoder', 'Unknown')}")

    return model, checkpoint


def load_discriminator(discriminator_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load pre-trained discriminator (same as before)"""

    if not Path(discriminator_path).exists():
        print(f"Discriminator not found: {discriminator_path}")
        return None

    if create_discriminator is None:
        print("Discriminator creation function not available")
        return None

    try:
        checkpoint = torch.load(discriminator_path, map_location=device)
        model_config = checkpoint['model_config']

        discriminator = create_discriminator(**model_config)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        discriminator.to(device)
        discriminator.eval()

        print(f"Discriminator loaded successfully")
        return discriminator

    except Exception as e:
        print(f"Failed to load discriminator: {e}")
        return None


# =============================================================================
# PPO EVALUATION ENGINE
# =============================================================================

class PPOGraphRefactoringEvaluator:
    """Comprehensive evaluator for PPO-based graph refactoring"""

    def __init__(self, model, discriminator, config: EvaluationConfig):
        self.model = model
        self.discriminator = discriminator
        self.config = config
        self.device = config.device

        # Initialize PPO-compatible environment
        self.env = PPORefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=config.max_episode_steps,
            device=config.device,
            reward_weights=config.reward_weights
        )

        # Results storage
        self.evaluation_results = []
        self.episode_trajectories = []

    def run_single_episode(self, episode_id: int, save_trajectory: bool = False) -> Dict:
        """Run a single evaluation episode with PPO model"""

        # Reset environment (returns Data object)
        initial_data = self.env.reset()
        initial_metrics = self.env._calculate_metrics(initial_data)

        # Episode tracking
        episode_data = {
            'episode_id': episode_id,
            'initial_hub_score': initial_metrics['hub_score'],
            'initial_metrics': initial_metrics,
            'initial_graph': initial_data.clone() if save_trajectory else None,
            'actions_taken': [],
            'states': [],
            'rewards': [],
            'step_info': [],
            'action_confidences': []
        }

        if save_trajectory:
            episode_data['states'].append(initial_data.clone())

        # Get initial discriminator score
        initial_disc_score = None
        if self.discriminator is not None:
            with torch.no_grad():
                try:
                    disc_output = self.discriminator(initial_data)
                    if isinstance(disc_output, dict):
                        initial_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                    else:
                        initial_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                except Exception as e:
                    print(f"Discriminator error on initial state: {e}")

        episode_data['initial_disc_score'] = initial_disc_score

        # Run episode
        episode_reward = 0.0
        episode_length = 0
        done = False
        current_data = initial_data

        while not done:
            # Extract global features using PPO method
            global_features = self.env.get_global_features()

            # Get action from PPO model (greedy evaluation)
            with torch.no_grad():
                output = self.model(current_data, global_features)
                action_probs = output['action_probs']
                action = torch.argmax(action_probs, dim=1).item()
                confidence = action_probs[0, action].item()

            # Take action (returns Data object as next_state)
            next_data, reward, done, info = self.env.step(action)

            # Record step
            episode_data['actions_taken'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['action_confidences'].append(confidence)
            episode_data['step_info'].append({
                'action': action,
                'confidence': confidence,
                'reward': reward,
                'done': done,
                'action_success': info.get('action_success', False),
                'hub_score': info.get('current_hub_score', 0.0)
            })

            if save_trajectory:
                episode_data['states'].append(next_data.clone())

            episode_reward += reward
            episode_length += 1
            current_data = next_data

        # Final metrics
        final_data = current_data
        final_metrics = self.env._calculate_metrics(final_data)

        # Get final discriminator score
        final_disc_score = None
        if self.discriminator is not None:
            with torch.no_grad():
                try:
                    disc_output = self.discriminator(final_data)
                    if isinstance(disc_output, dict):
                        final_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                    else:
                        final_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                except Exception as e:
                    print(f"Discriminator error on final state: {e}")

        # Compile episode results
        hub_improvement = initial_metrics['hub_score'] - final_metrics['hub_score']
        disc_improvement = 0.0
        if initial_disc_score is not None and final_disc_score is not None:
            disc_improvement = initial_disc_score - final_disc_score

        episode_data.update({
            'final_hub_score': final_metrics['hub_score'],
            'final_metrics': final_metrics,
            'final_graph': final_data.clone() if save_trajectory else None,
            'final_disc_score': final_disc_score,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'hub_improvement': hub_improvement,
            'disc_improvement': disc_improvement,
            'success': hub_improvement > 0.01,
            'significant_success': hub_improvement > 0.05,
            'num_valid_actions': sum(1 for step in episode_data['step_info'] if step['action_success']),
            'final_action_was_stop': episode_data['actions_taken'][-1] == 6 if episode_data['actions_taken'] else False,
            'avg_confidence': np.mean(episode_data['action_confidences']) if episode_data[
                'action_confidences'] else 0.0,
            'best_hub_score': getattr(self.env, 'best_hub_score', final_metrics['hub_score'])
        })

        return episode_data

    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""

        print(f"Running PPO evaluation on {self.config.num_eval_episodes} episodes...")

        # Run episodes
        for episode_id in range(self.config.num_eval_episodes):
            save_trajectory = episode_id < self.config.num_visualization_episodes

            episode_result = self.run_single_episode(episode_id, save_trajectory)
            self.evaluation_results.append(episode_result)

            if save_trajectory:
                self.episode_trajectories.append(episode_result)

            # Progress update
            if (episode_id + 1) % 10 == 0:
                print(f"   Completed {episode_id + 1}/{self.config.num_eval_episodes} episodes")

        # Compute summary statistics
        summary_stats = self._compute_summary_statistics()

        print("Evaluation completed!")
        self._print_summary_stats(summary_stats)

        return {
            'summary_stats': summary_stats,
            'episode_results': self.evaluation_results,
            'trajectories': self.episode_trajectories
        }

    def _compute_summary_statistics(self) -> Dict:
        """Compute comprehensive summary statistics for PPO"""

        results = self.evaluation_results

        # Basic metrics
        episode_rewards = [r['episode_reward'] for r in results]
        hub_improvements = [r['hub_improvement'] for r in results]
        episode_lengths = [r['episode_length'] for r in results]
        disc_improvements = [r['disc_improvement'] for r in results if r['disc_improvement'] != 0.0]
        action_confidences = [r['avg_confidence'] for r in results]

        # Success metrics
        successes = [r['success'] for r in results]
        significant_successes = [r['significant_success'] for r in results]

        # Action analysis
        all_actions = []
        action_success_rates = {}
        for r in results:
            all_actions.extend(r['actions_taken'])
            for step in r['step_info']:
                action = step['action']
                if action not in action_success_rates:
                    action_success_rates[action] = {'total': 0, 'successful': 0}
                action_success_rates[action]['total'] += 1
                if step['action_success']:
                    action_success_rates[action]['successful'] += 1

        action_distribution = {}
        for action in range(7):
            action_distribution[action] = all_actions.count(action) / len(all_actions) if all_actions else 0

        for action in action_success_rates:
            action_success_rates[action]['rate'] = (
                action_success_rates[action]['successful'] / action_success_rates[action]['total']
                if action_success_rates[action]['total'] > 0 else 0
            )

        summary = {
            # Basic performance
            'num_episodes': len(results),
            'success_rate': np.mean(successes),
            'significant_success_rate': np.mean(significant_successes),

            # Reward statistics
            'mean_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'median_episode_reward': np.median(episode_rewards),

            # Hub improvement statistics
            'mean_hub_improvement': np.mean(hub_improvements),
            'std_hub_improvement': np.std(hub_improvements),
            'median_hub_improvement': np.median(hub_improvements),
            'max_hub_improvement': np.max(hub_improvements),
            'min_hub_improvement': np.min(hub_improvements),

            # Episode length statistics
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths),
            'median_episode_length': np.median(episode_lengths),

            # PPO-specific statistics
            'mean_action_confidence': np.mean(action_confidences),
            'std_action_confidence': np.std(action_confidences),

            # Discriminator statistics
            'discriminator_available': len(disc_improvements) > 0,
            'mean_disc_improvement': np.mean(disc_improvements) if disc_improvements else 0.0,
            'std_disc_improvement': np.std(disc_improvements) if disc_improvements else 0.0,

            # Action analysis
            'action_distribution': action_distribution,
            'action_success_rates': action_success_rates,
            'total_actions_taken': len(all_actions),

            # Performance bins
            'excellent_performance': sum(1 for imp in hub_improvements if imp > 0.1) / len(hub_improvements),
            'good_performance': sum(1 for imp in hub_improvements if 0.05 < imp <= 0.1) / len(hub_improvements),
            'moderate_performance': sum(1 for imp in hub_improvements if 0.01 < imp <= 0.05) / len(hub_improvements),
            'poor_performance': sum(1 for imp in hub_improvements if imp <= 0.01) / len(hub_improvements),
        }

        return summary

    def _print_summary_stats(self, stats: Dict):
        """Print formatted summary statistics"""

        print("\nEVALUATION SUMMARY")
        print("=" * 50)

        print(f"SUCCESS METRICS:")
        print(f"   Overall Success Rate: {stats['success_rate']:.1%}")
        print(f"   Significant Success Rate: {stats['significant_success_rate']:.1%}")
        print(f"   Excellent Performance (>0.1): {stats['excellent_performance']:.1%}")
        print(f"   Good Performance (0.05-0.1): {stats['good_performance']:.1%}")

        print(f"\nHUB IMPROVEMENT:")
        print(f"   Mean: {stats['mean_hub_improvement']:.4f} ± {stats['std_hub_improvement']:.4f}")
        print(f"   Median: {stats['median_hub_improvement']:.4f}")
        print(f"   Best: {stats['max_hub_improvement']:.4f}")
        print(f"   Worst: {stats['min_hub_improvement']:.4f}")

        print(f"\nEPISODE REWARDS:")
        print(f"   Mean: {stats['mean_episode_reward']:.3f} ± {stats['std_episode_reward']:.3f}")
        print(f"   Median: {stats['median_episode_reward']:.3f}")

        print(f"\nPPO METRICS:")
        print(
            f"   Mean Action Confidence: {stats['mean_action_confidence']:.3f} ± {stats['std_action_confidence']:.3f}")
        print(f"   Mean Episode Length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")

        if stats['discriminator_available']:
            print(f"\nDISCRIMINATOR IMPROVEMENT:")
            print(f"   Mean: {stats['mean_disc_improvement']:.4f} ± {stats['std_disc_improvement']:.4f}")

        print(f"\nACTION USAGE:")
        action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                        'ExtractAbstractUnit', 'ExtractUnit', 'STOP']
        for action, name in enumerate(action_names):
            if action in stats['action_distribution']:
                freq = stats['action_distribution'][action]
                success_rate = stats['action_success_rates'].get(action, {}).get('rate', 0)
                print(f"   {name}: {freq:.1%} usage, {success_rate:.1%} success")


# =============================================================================
# VISUALIZATION FUNCTIONS (UPDATED FOR PPO)
# =============================================================================

def create_ppo_performance_dashboard(evaluation_results: Dict) -> None:
    """Create comprehensive performance dashboard for PPO"""

    results = evaluation_results['episode_results']
    stats = evaluation_results['summary_stats']

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))

    # 1. Hub Improvement Distribution
    plt.subplot(3, 4, 1)
    hub_improvements = [r['hub_improvement'] for r in results]
    plt.hist(hub_improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    plt.axvline(np.mean(hub_improvements), color='green', linestyle='-', alpha=0.7, label='Mean')
    plt.xlabel('Hub Score Improvement')
    plt.ylabel('Frequency')
    plt.title('Hub Improvement Distribution (PPO)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Action Confidence Distribution
    plt.subplot(3, 4, 2)
    confidences = [r['avg_confidence'] for r in results]
    plt.hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Average Action Confidence')
    plt.ylabel('Frequency')
    plt.title('PPO Action Confidence Distribution')
    plt.grid(True, alpha=0.3)

    # 3. Episode Rewards Over Time
    plt.subplot(3, 4, 3)
    episode_rewards = [r['episode_reward'] for r in results]
    plt.plot(episode_rewards, alpha=0.7, color='blue', linewidth=1)
    plt.axhline(np.mean(episode_rewards), color='red', linestyle='--', alpha=0.7, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards (PPO)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Action Distribution
    plt.subplot(3, 4, 4)
    action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                    'ExtractAbstractUnit', 'ExtractUnit', 'STOP']
    action_counts = [stats['action_distribution'].get(i, 0) for i in range(7)]

    bars = plt.bar(range(7), action_counts, alpha=0.7, color='orange')
    plt.xlabel('Action')
    plt.ylabel('Usage Frequency')
    plt.title('Action Usage Distribution (PPO)')
    plt.xticks(range(7), [name[:8] for name in action_names], rotation=45)
    plt.grid(True, alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, count) in enumerate(zip(bars, action_counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{count:.1%}', ha='center', va='bottom', fontsize=8)

    # 5. Hub Score: Before vs After
    plt.subplot(3, 4, 5)
    initial_scores = [r['initial_hub_score'] for r in results]
    final_scores = [r['final_hub_score'] for r in results]

    plt.scatter(initial_scores, final_scores, alpha=0.6, color='purple')
    plt.plot([min(initial_scores), max(initial_scores)],
             [min(initial_scores), max(initial_scores)],
             'r--', alpha=0.7, label='No change')
    plt.xlabel('Initial Hub Score')
    plt.ylabel('Final Hub Score')
    plt.title('Hub Score: Before vs After (PPO)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Performance Categories Pie Chart
    plt.subplot(3, 4, 6)
    categories = ['Excellent (>0.1)', 'Good (0.05-0.1)', 'Moderate (0.01-0.05)', 'Poor (≤0.01)']
    values = [stats['excellent_performance'], stats['good_performance'],
              stats['moderate_performance'], stats['poor_performance']]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Performance Categories (PPO)')

    # 7. Confidence vs Success Rate
    plt.subplot(3, 4, 7)
    success_binary = [1 if r['success'] else 0 for r in results]
    confidences = [r['avg_confidence'] for r in results]

    # Create confidence bins and calculate success rate for each
    conf_bins = np.linspace(min(confidences), max(confidences), 10)
    bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    bin_success_rates = []

    for i in range(len(conf_bins) - 1):
        mask = (np.array(confidences) >= conf_bins[i]) & (np.array(confidences) < conf_bins[i + 1])
        if mask.any():
            bin_success_rates.append(np.mean(np.array(success_binary)[mask]))
        else:
            bin_success_rates.append(0)

    plt.bar(bin_centers, bin_success_rates, width=(conf_bins[1] - conf_bins[0]) * 0.8,
            alpha=0.7, color='lightcoral')
    plt.xlabel('Action Confidence')
    plt.ylabel('Success Rate')
    plt.title('Confidence vs Success Rate (PPO)')
    plt.grid(True, alpha=0.3)

    # 8. Cumulative Success Rate
    plt.subplot(3, 4, 8)
    cumulative_successes = np.cumsum([r['success'] for r in results])
    cumulative_rate = cumulative_successes / np.arange(1, len(results) + 1)

    plt.plot(cumulative_rate, color='green', linewidth=2)
    plt.axhline(stats['success_rate'], color='red', linestyle='--', alpha=0.7,
                label=f'Final: {stats["success_rate"]:.1%}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Success Rate')
    plt.title('Cumulative Success Rate (PPO)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 9. Episode Length vs Hub Improvement
    plt.subplot(3, 4, 9)
    lengths = [r['episode_length'] for r in results]
    improvements = [r['hub_improvement'] for r in results]

    plt.scatter(lengths, improvements, alpha=0.6, color='teal')
    plt.xlabel('Episode Length')
    plt.ylabel('Hub Improvement')
    plt.title('Episode Length vs Improvement (PPO)')
    plt.grid(True, alpha=0.3)

    # 10. Action Success Rates
    plt.subplot(3, 4, 10)
    success_rates = [stats['action_success_rates'].get(i, {}).get('rate', 0) for i in range(7)]

    bars = plt.bar(range(7), success_rates, alpha=0.7, color='lightcoral')
    plt.xlabel('Action')
    plt.ylabel('Success Rate')
    plt.title('Action Success Rates (PPO)')
    plt.xticks(range(7), [name[:8] for name in action_names], rotation=45)
    plt.grid(True, alpha=0.3)

    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if rate > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{rate:.1%}', ha='center', va='bottom', fontsize=8)

    # 11. Reward vs Confidence Scatter
    plt.subplot(3, 4, 11)
    rewards = [r['episode_reward'] for r in results]
    plt.scatter(confidences, rewards, alpha=0.6, color='gold')
    plt.xlabel('Average Action Confidence')
    plt.ylabel('Episode Reward')
    plt.title('Confidence vs Reward (PPO)')
    plt.grid(True, alpha=0.3)

    # 12. PPO Training Convergence (if available)
    plt.subplot(3, 4, 12)
    # Plot success rate in sliding window
    window_size = 5
    if len(results) >= window_size:
        windowed_success = []
        for i in range(window_size - 1, len(results)):
            window_successes = [results[j]['success'] for j in range(i - window_size + 1, i + 1)]
            windowed_success.append(np.mean(window_successes))

        plt.plot(range(window_size - 1, len(results)), windowed_success,
                 color='mediumpurple', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel(f'Success Rate (window={window_size})')
        plt.title('PPO Learning Progress')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(eval_config.results_dir) / 'ppo_performance_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def create_ppo_interactive_trajectory_plot(episode_data: Dict) -> go.Figure:
    """Create interactive plot of PPO episode trajectory"""

    steps = range(len(episode_data['step_info']) + 1)
    hub_scores = [episode_data['initial_hub_score']]
    hub_scores.extend([step['hub_score'] for step in episode_data['step_info']])

    actions_taken = ['Initial'] + [f"Action {step['action']}" for step in episode_data['step_info']]
    rewards = [0] + episode_data['rewards']
    confidences = [0] + [step['confidence'] for step in episode_data['step_info']]

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Hub Score Trajectory', 'Step Rewards', 'Action Confidence'),
        vertical_spacing=0.1
    )

    # Hub score trajectory
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=hub_scores,
            mode='lines+markers',
            name='Hub Score',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Step %{x}</b><br>Hub Score: %{y:.4f}<br>Action: %{text}<extra></extra>',
            text=actions_taken
        ),
        row=1, col=1
    )

    # Add success threshold line
    fig.add_hline(y=episode_data['initial_hub_score'], row=1, col=1,
                  line_dash="dash", line_color="red",
                  annotation_text="Initial Hub Score")

    # Step rewards
    fig.add_trace(
        go.Bar(
            x=steps[1:],  # Exclude initial step
            y=rewards[1:],  # Exclude initial reward
            name='Step Reward',
            marker_color=['green' if r > 0 else 'red' for r in rewards[1:]],
            hovertemplate='<b>Step %{x}</b><br>Reward: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Action confidence
    fig.add_trace(
        go.Scatter(
            x=steps[1:],  # Exclude initial step
            y=confidences[1:],  # Exclude initial confidence
            mode='lines+markers',
            name='Confidence',
            line=dict(color='green', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Step %{x}</b><br>Confidence: %{y:.3f}<br>Action: %{text}<extra></extra>',
            text=actions_taken[1:]
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"PPO Episode {episode_data['episode_id']} Trajectory - "
              f"Final Improvement: {episode_data['hub_improvement']:.4f}",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="Step", row=3, col=1)
    fig.update_yaxes(title_text="Hub Score", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=2, col=1)
    fig.update_yaxes(title_text="Confidence", row=3, col=1)

    return fig


def visualize_ppo_graph_comparison(before_graph: Data, after_graph: Data,
                                   episode_info: Dict, save_path: Optional[str] = None) -> None:
    """Visualize before/after graph comparison for PPO results"""

    # Convert to NetworkX
    G_before = to_networkx(before_graph, to_undirected=True)
    G_after = to_networkx(after_graph, to_undirected=True)

    # Find differences
    nodes_before = set(G_before.nodes())
    nodes_after = set(G_after.nodes())
    edges_before = set(G_before.edges())
    edges_after = set(G_after.edges())

    added_nodes = nodes_after - nodes_before
    removed_nodes = nodes_before - nodes_after
    added_edges = edges_after - edges_before
    removed_edges = edges_before - edges_after

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Determine layout (use the same for both graphs)
    all_nodes = nodes_before.union(nodes_after)
    if len(all_nodes) <= 50:
        pos = nx.spring_layout(G_after if len(G_after.nodes()) >= len(G_before.nodes()) else G_before,
                               k=1, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G_after if len(G_after.nodes()) >= len(G_before.nodes()) else G_before,
                               k=3, iterations=30, seed=42)

    # Plot BEFORE graph
    ax1.set_title(f"BEFORE PPO Refactoring\nHub Score: {episode_info['initial_hub_score']:.4f}",
                  fontsize=14, fontweight='bold')

    # Draw edges first (so they appear behind nodes)
    if G_before.edges():
        nx.draw_networkx_edges(G_before, pos, ax=ax1, edge_color=eval_config.colors['original_edges'],
                               width=eval_config.edge_width, alpha=0.6)

    # Draw nodes
    if G_before.nodes():
        node_colors = [eval_config.colors['hub_node'] if node == 0 else eval_config.colors['original_nodes']
                       for node in G_before.nodes()]
        node_sizes = [eval_config.node_size_factor * 1.5 if node == 0 else eval_config.node_size_factor
                      for node in G_before.nodes()]

        nx.draw_networkx_nodes(G_before, pos, ax=ax1, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        # Draw labels
        nx.draw_networkx_labels(G_before, pos, ax=ax1, font_size=8, font_weight='bold')

    ax1.set_aspect('equal')
    ax1.axis('off')

    # Plot AFTER graph
    ax2.set_title(f"AFTER PPO Refactoring\nHub Score: {episode_info['final_hub_score']:.4f}\n"
                  f"Improvement: {episode_info['hub_improvement']:.4f}",
                  fontsize=14, fontweight='bold')

    # Draw original edges
    original_edges = [(u, v) for u, v in G_after.edges() if (u, v) in edges_before or (v, u) in edges_before]
    if original_edges:
        nx.draw_networkx_edges(G_after, pos, edgelist=original_edges, ax=ax2,
                               edge_color=eval_config.colors['original_edges'],
                               width=eval_config.edge_width, alpha=0.6)

    # Draw added edges
    added_edges_list = [(u, v) for u, v in G_after.edges() if (u, v) in added_edges or (v, u) in added_edges]
    if added_edges_list:
        nx.draw_networkx_edges(G_after, pos, edgelist=added_edges_list, ax=ax2,
                               edge_color=eval_config.colors['added_edges'],
                               width=eval_config.edge_width * 1.5, alpha=0.8, style='dashed')

    # Draw nodes
    if G_after.nodes():
        node_colors = []
        node_sizes = []
        for node in G_after.nodes():
            if node == 0:  # Hub node
                node_colors.append(eval_config.colors['hub_node'])
                node_sizes.append(eval_config.node_size_factor * 1.5)
            elif node in added_nodes:  # Added nodes
                node_colors.append(eval_config.colors['added_nodes'])
                node_sizes.append(eval_config.node_size_factor * 1.2)
            else:  # Original nodes
                node_colors.append(eval_config.colors['original_nodes'])
                node_sizes.append(eval_config.node_size_factor)

        nx.draw_networkx_nodes(G_after, pos, ax=ax2, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        # Draw labels
        nx.draw_networkx_labels(G_after, pos, ax=ax2, font_size=8, font_weight='bold')

    ax2.set_aspect('equal')
    ax2.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['hub_node'],
                   markersize=12, label='Hub Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['original_nodes'],
                   markersize=10, label='Original Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['added_nodes'],
                   markersize=10, label='Added Nodes'),
        plt.Line2D([0], [0], color=eval_config.colors['original_edges'], linewidth=2, label='Original Edges'),
        plt.Line2D([0], [0], color=eval_config.colors['added_edges'], linewidth=2,
                   linestyle='--', label='Added Edges')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_elements), fontsize=10)

    # Add episode information
    info_text = f"""
PPO Episode {episode_info['episode_id']} Summary:
• Actions taken: {len(episode_info['actions_taken'])}
• Episode reward: {episode_info['episode_reward']:.3f}
• Success: {'✅' if episode_info['success'] else '❌'}
• Valid actions: {episode_info['num_valid_actions']}/{len(episode_info['actions_taken'])}
• Avg confidence: {episode_info.get('avg_confidence', 0):.3f}
"""

    plt.figtext(0.02, 0.95, info_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    # Add modifications summary
    mod_text = f"""
Graph Modifications:
• Nodes added: {len(added_nodes)}
• Nodes removed: {len(removed_nodes)}
• Edges added: {len(added_edges)}
• Edges removed: {len(removed_edges)}
"""

    plt.figtext(0.98, 0.95, mod_text, fontsize=10, fontfamily='monospace', ha='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_ppo_action_analysis_plot(evaluation_results: Dict) -> go.Figure:
    """Create interactive action analysis plot for PPO"""

    results = evaluation_results['episode_results']
    stats = evaluation_results['summary_stats']

    action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                    'ExtractAbstractUnit', 'ExtractUnit', 'STOP']

    # Prepare data
    action_usage = [stats['action_distribution'].get(i, 0) * 100 for i in range(7)]
    action_success = [stats['action_success_rates'].get(i, {}).get('rate', 0) * 100 for i in range(7)]

    # Calculate average confidence per action
    action_confidences = {i: [] for i in range(7)}
    for r in results:
        for i, action in enumerate(r['actions_taken']):
            if i < len(r['action_confidences']):
                action_confidences[action].append(r['action_confidences'][i])

    avg_confidences = [np.mean(action_confidences[i]) * 100 if action_confidences[i] else 0 for i in range(7)]

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Action Usage Frequency', 'Action Success Rates', 'Action Confidence'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    # Action usage
    fig.add_trace(
        go.Bar(
            x=action_names,
            y=action_usage,
            name='Usage %',
            marker_color='lightblue',
            hovertemplate='<b>%{x}</b><br>Usage: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # Action success rates
    fig.add_trace(
        go.Bar(
            x=action_names,
            y=action_success,
            name='Success %',
            marker_color='lightgreen',
            hovertemplate='<b>%{x}</b><br>Success Rate: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )

    # Action confidence
    fig.add_trace(
        go.Bar(
            x=action_names,
            y=avg_confidences,
            name='Confidence %',
            marker_color='lightyellow',
            hovertemplate='<b>%{x}</b><br>Avg Confidence: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title="PPO Action Analysis Dashboard",
        height=500,
        showlegend=False
    )

    fig.update_xaxes(title_text="Actions", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Actions", row=1, col=2, tickangle=45)
    fig.update_xaxes(title_text="Actions", row=1, col=3, tickangle=45)
    fig.update_yaxes(title_text="Usage Frequency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Avg Confidence (%)", row=1, col=3)

    return fig


# =============================================================================
# PPO ANALYSIS FUNCTIONS
# =============================================================================

def analyze_ppo_failure_cases(evaluation_results: Dict) -> None:
    """Analyze episodes where PPO model failed to improve"""

    results = evaluation_results['episode_results']
    failures = [r for r in results if not r['success']]

    if not failures:
        print("No failure cases found! All episodes were successful.")
        return

    print(f"\nPPO FAILURE CASE ANALYSIS ({len(failures)} episodes)")
    print("=" * 50)

    # PPO-specific failure patterns
    low_confidence_failures = [f for f in failures if f.get('avg_confidence', 1) < 0.5]
    early_stops = [f for f in failures if f['final_action_was_stop'] and f['episode_length'] <= 3]
    long_episodes = [f for f in failures if f['episode_length'] >= 15]
    negative_improvements = [f for f in failures if f['hub_improvement'] < -0.01]

    print(f"Failure Patterns:")
    print(
        f"   Low confidence episodes (<0.5): {len(low_confidence_failures)} ({len(low_confidence_failures) / len(failures):.1%})")
    print(f"   Early stops (≤3 steps): {len(early_stops)} ({len(early_stops) / len(failures):.1%})")
    print(f"   Long episodes (≥15 steps): {len(long_episodes)} ({len(long_episodes) / len(failures):.1%})")
    print(f"   Negative improvements: {len(negative_improvements)} ({len(negative_improvements) / len(failures):.1%})")

    # Confidence analysis
    failure_confidences = [f.get('avg_confidence', 0) for f in failures]
    print(f"\nConfidence Analysis in Failures:")
    print(f"   Average confidence: {np.mean(failure_confidences):.3f}")
    print(f"   Confidence std: {np.std(failure_confidences):.3f}")

    # Compare to successful episodes
    successes = [r for r in results if r['success']]
    if successes:
        success_confidences = [s.get('avg_confidence', 0) for s in successes]
        print(f"\nComparison (Success vs Failure):")
        print(f"   Success avg confidence: {np.mean(success_confidences):.3f}")
        print(f"   Failure avg confidence: {np.mean(failure_confidences):.3f}")


def analyze_ppo_convergence(evaluation_results: Dict) -> None:
    """Analyze PPO learning convergence patterns"""

    results = evaluation_results['episode_results']

    print(f"\nPPO CONVERGENCE ANALYSIS")
    print("=" * 50)

    # Split into early and late episodes
    mid_point = len(results) // 2
    early_episodes = results[:mid_point]
    late_episodes = results[mid_point:]

    early_success_rate = np.mean([r['success'] for r in early_episodes])
    late_success_rate = np.mean([r['success'] for r in late_episodes])

    early_confidence = np.mean([r.get('avg_confidence', 0) for r in early_episodes])
    late_confidence = np.mean([r.get('avg_confidence', 0) for r in late_episodes])

    early_improvement = np.mean([r['hub_improvement'] for r in early_episodes])
    late_improvement = np.mean([r['hub_improvement'] for r in late_episodes])

    print(f"Early Episodes (1-{mid_point}):")
    print(f"   Success rate: {early_success_rate:.1%}")
    print(f"   Avg confidence: {early_confidence:.3f}")
    print(f"   Avg hub improvement: {early_improvement:.4f}")

    print(f"\nLate Episodes ({mid_point + 1}-{len(results)}):")
    print(f"   Success rate: {late_success_rate:.1%}")
    print(f"   Avg confidence: {late_confidence:.3f}")
    print(f"   Avg hub improvement: {late_improvement:.4f}")

    print(f"\nImprovement over time:")
    print(f"   Success rate change: {(late_success_rate - early_success_rate):.1%}")
    print(f"   Confidence change: {(late_confidence - early_confidence):+.3f}")
    print(f"   Hub improvement change: {(late_improvement - early_improvement):+.4f}")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_complete_ppo_evaluation():
    """Run complete PPO evaluation pipeline"""

    print("Starting Complete PPO Evaluation Pipeline")
    print("=" * 60)

    # Create results directory
    Path(eval_config.results_dir).mkdir(parents=True, exist_ok=True)

    # Load PPO model
    print("Loading trained PPO models...")
    model, model_checkpoint = load_trained_ppo_model(eval_config.model_path, eval_config.device)
    discriminator = load_discriminator(eval_config.discriminator_path, eval_config.device)

    # Initialize evaluator
    print("Initializing PPO evaluator...")
    evaluator = PPOGraphRefactoringEvaluator(model, discriminator, eval_config)

    # Run evaluation
    print("Running comprehensive PPO evaluation...")
    evaluation_results = evaluator.run_comprehensive_evaluation()

    # Save results
    results_file = Path(eval_config.results_dir) / 'ppo_evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert for JSON serialization
        serializable_results = {}
        for key, value in evaluation_results.items():
            if key == 'episode_results':
                serializable_episodes = []
                for episode in value:
                    episode_copy = episode.copy()
                    # Remove non-serializable graph objects
                    for graph_key in ['initial_graph', 'final_graph', 'states']:
                        if graph_key in episode_copy:
                            del episode_copy[graph_key]
                    serializable_episodes.append(episode_copy)
                serializable_results[key] = serializable_episodes
            elif key == 'trajectories':
                serializable_trajectories = []
                for traj in value:
                    traj_copy = traj.copy()
                    for graph_key in ['initial_graph', 'final_graph', 'states']:
                        if graph_key in traj_copy:
                            del traj_copy[graph_key]
                    serializable_trajectories.append(traj_copy)
                serializable_results[key] = serializable_trajectories
            else:
                serializable_results[key] = value

        json.dump(serializable_results, f, indent=2, default=str)

    print(f"Results saved to {results_file}")

    # Create visualizations
    print("Creating PPO performance dashboard...")
    create_ppo_performance_dashboard(evaluation_results)

    # Create graph comparisons for trajectory episodes
    print("Creating graph visualizations...")
    for i, episode_data in enumerate(evaluation_results['trajectories']):
        if episode_data['initial_graph'] is not None and episode_data['final_graph'] is not None:
            save_path = Path(eval_config.results_dir) / f'ppo_graph_comparison_episode_{i}.png'
            visualize_ppo_graph_comparison(
                episode_data['initial_graph'],
                episode_data['final_graph'],
                episode_data,
                save_path
            )

    # Create interactive plots
    print("Creating interactive visualizations...")

    # Trajectory plots for first few episodes
    for i, episode_data in enumerate(evaluation_results['trajectories'][:3]):
        trajectory_fig = create_ppo_interactive_trajectory_plot(episode_data)
        trajectory_fig.write_html(Path(eval_config.results_dir) / f'ppo_trajectory_episode_{i}.html')

    # Action analysis plot
    action_fig = create_ppo_action_analysis_plot(evaluation_results)
    action_fig.write_html(Path(eval_config.results_dir) / 'ppo_action_analysis.html')

    # Additional PPO-specific analyses
    analyze_ppo_failure_cases(evaluation_results)
    analyze_ppo_convergence(evaluation_results)

    print("PPO evaluation completed successfully!")
    print(f"All results saved to: {eval_config.results_dir}")

    return evaluation_results


# =============================================================================
# QUICK DIAGNOSTIC FUNCTIONS
# =============================================================================

def quick_ppo_model_check():
    """Quick check if PPO models are available"""
    model_exists = Path(eval_config.model_path).exists()
    disc_exists = Path(eval_config.discriminator_path).exists()
    data_exists = Path(eval_config.data_path).exists()

    print("PPO MODEL AVAILABILITY CHECK:")
    print(f"   PPO model: {'✅' if model_exists else '❌'} {eval_config.model_path}")
    print(f"   Discriminator: {'✅' if disc_exists else '❌'} {eval_config.discriminator_path}")
    print(f"   Data: {'✅' if data_exists else '❌'} {eval_config.data_path}")

    return model_exists and data_exists


def run_quick_ppo_test():
    """Run a quick test with 5 episodes to check if everything works"""
    print("Running quick PPO test (5 episodes)...")

    # Temporarily modify config for quick test
    original_episodes = eval_config.num_eval_episodes
    eval_config.num_eval_episodes = 5

    try:
        model, _ = load_trained_ppo_model(eval_config.model_path, eval_config.device)
        discriminator = load_discriminator(eval_config.discriminator_path, eval_config.device)

        evaluator = PPOGraphRefactoringEvaluator(model, discriminator, eval_config)
        results = evaluator.run_comprehensive_evaluation()

        print("Quick test completed successfully!")
        return results

    except Exception as e:
        print(f"Quick test failed: {e}")
        return None
    finally:
        # Restore original config
        eval_config.num_eval_episodes = original_episodes


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check if PPO model exists
    if not Path(eval_config.model_path).exists():
        print(f"PPO model not found at {eval_config.model_path}")
        print("Please train the PPO model first or update the model path.")
    else:
        # Run complete PPO evaluation
        results = run_complete_ppo_evaluation()

        print(f"\nPPO evaluation finished!")
        print(f"Check {eval_config.results_dir} for all visualizations and results")

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

print("\nPPO EVALUATION SCRIPT READY!")
print("Run the following functions:")
print("1. quick_ppo_model_check() - Check if models are available")
print("2. run_quick_ppo_test() - Quick test with 5 episodes")
print("3. run_complete_ppo_evaluation() - Full evaluation")