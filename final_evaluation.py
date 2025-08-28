#!/usr/bin/env python3
"""
Graph Refactoring RL - Comprehensive Evaluation Notebook
========================================================

This notebook provides comprehensive evaluation and visualization of the trained
Graph Refactoring RL model, including:
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

# Import project modules
from rl_gym import RefactorEnv
from actor_critic_models import create_actor_critic
from discriminator import create_discriminator

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Graph Refactoring RL - Evaluation Dashboard")
print("=" * 60)


# =============================================================================
# CONFIGURATION
# =============================================================================

class EvaluationConfig:
    """Configuration for evaluation"""

    def __init__(self):
        # Paths
        self.model_path = "results/rl_training/best_model.pt"
        self.discriminator_path = "results/discriminator_pretraining/pretrained_discriminator.pt"
        self.data_path = "data_builder/dataset/graph_features"
        self.results_dir = "results/evaluation"

        # Evaluation parameters
        self.num_eval_episodes = 50
        self.num_visualization_episodes = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Visualization parameters
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
# MODEL LOADING UTILITIES
# =============================================================================

def load_trained_model(model_path: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained RL model"""

    print(f"📦 Loading model from {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Fallback configuration
        model_config = {
            'node_dim': 7,
            'hidden_dim': 128,
            'num_layers': 3,
            'num_actions': 7,
            'global_features_dim': 4,
            'dropout': 0.2,
            'shared_encoder': False
        }
        print("⚠️ Using fallback model configuration")

    # Create and load model
    model = create_actor_critic(model_config).to(device)
    model.load_state_dict(checkpoint['actor_critic_state_dict'])
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Architecture: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, checkpoint


def load_discriminator(discriminator_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load pre-trained discriminator"""

    if not Path(discriminator_path).exists():
        print(f"⚠️ Discriminator not found: {discriminator_path}")
        return None

    try:
        checkpoint = torch.load(discriminator_path, map_location=device)
        model_config = checkpoint['model_config']

        discriminator = create_discriminator(**model_config)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        discriminator.to(device)
        discriminator.eval()

        print(f"✅ Discriminator loaded successfully")
        return discriminator

    except Exception as e:
        print(f"⚠️ Failed to load discriminator: {e}")
        return None


# =============================================================================
# EVALUATION ENGINE
# =============================================================================

class GraphRefactoringEvaluator:
    """Comprehensive evaluator for graph refactoring RL"""

    def __init__(self, model, discriminator, config: EvaluationConfig):
        self.model = model
        self.discriminator = discriminator
        self.config = config
        self.device = config.device

        # Initialize environment
        self.env = RefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=20,
            device=config.device
        )

        # Results storage
        self.evaluation_results = []
        self.episode_trajectories = []

    def run_single_episode(self, episode_id: int, save_trajectory: bool = False) -> Dict:
        """Run a single evaluation episode"""

        # Reset environment
        self.env.reset()
        initial_data = self.env.current_data.clone()
        initial_metrics = self.env._calculate_metrics(initial_data)

        # Episode tracking
        episode_data = {
            'episode_id': episode_id,
            'initial_hub_score': initial_metrics['hub_score'],
            'initial_metrics': initial_metrics,
            'initial_graph': initial_data,
            'actions_taken': [],
            'states': [],
            'rewards': [],
            'step_info': []
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
                except:
                    pass

        episode_data['initial_disc_score'] = initial_disc_score

        # Run episode
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            current_data = self.env.current_data

            # DOPO (fix):
            metrics = self.env._calculate_metrics(current_data)
            global_features = torch.tensor([
                metrics['hub_score'],
                metrics['num_nodes'],
                metrics['num_edges'],
                metrics['connected']
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            # Get action from model (greedy evaluation)
            with torch.no_grad():
                output = self.model(current_data, global_features)
                action_probs = torch.softmax(output['action_logits'], dim=1)
                action = torch.argmax(action_probs, dim=1).item()
                confidence = action_probs[0, action].item()

            # Take action
            _, reward, done, info = self.env.step(action)

            # Record step
            episode_data['actions_taken'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['step_info'].append({
                'action': action,
                'confidence': confidence,
                'reward': reward,
                'done': done,
                'action_success': info.get('action_success', False),
                'hub_score': info.get('metrics', {}).get('hub_score', 0.0)
            })

            if save_trajectory:
                episode_data['states'].append(self.env.current_data.clone())

            episode_reward += reward
            episode_length += 1

        # Final metrics
        final_data = self.env.current_data
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
                except:
                    pass

        # Compile episode results
        hub_improvement = initial_metrics['hub_score'] - final_metrics['hub_score']
        disc_improvement = 0.0
        if initial_disc_score is not None and final_disc_score is not None:
            disc_improvement = initial_disc_score - final_disc_score

        episode_data.update({
            'final_hub_score': final_metrics['hub_score'],
            'final_metrics': final_metrics,
            'final_graph': final_data,
            'final_disc_score': final_disc_score,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'hub_improvement': hub_improvement,
            'disc_improvement': disc_improvement,
            'success': hub_improvement > 0.01,  # Success threshold
            'significant_success': hub_improvement > 0.05,
            'num_valid_actions': sum(1 for step in episode_data['step_info'] if step['action_success']),
            'final_action_was_stop': episode_data['actions_taken'][-1] == 6 if episode_data['actions_taken'] else False
        })

        return episode_data

    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""

        print(f"🔄 Running evaluation on {self.config.num_eval_episodes} episodes...")

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

        print("✅ Evaluation completed!")
        self._print_summary_stats(summary_stats)

        return {
            'summary_stats': summary_stats,
            'episode_results': self.evaluation_results,
            'trajectories': self.episode_trajectories
        }

    def _compute_summary_statistics(self) -> Dict:
        """Compute comprehensive summary statistics"""

        results = self.evaluation_results

        # Basic metrics
        episode_rewards = [r['episode_reward'] for r in results]
        hub_improvements = [r['hub_improvement'] for r in results]
        episode_lengths = [r['episode_length'] for r in results]
        disc_improvements = [r['disc_improvement'] for r in results if r['disc_improvement'] is not None]

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

        print("\n📊 EVALUATION SUMMARY")
        print("=" * 50)

        print(f"📈 SUCCESS METRICS:")
        print(f"   Overall Success Rate: {stats['success_rate']:.1%}")
        print(f"   Significant Success Rate: {stats['significant_success_rate']:.1%}")
        print(f"   Excellent Performance (>0.1): {stats['excellent_performance']:.1%}")
        print(f"   Good Performance (0.05-0.1): {stats['good_performance']:.1%}")

        print(f"\n🎯 HUB IMPROVEMENT:")
        print(f"   Mean: {stats['mean_hub_improvement']:.4f} ± {stats['std_hub_improvement']:.4f}")
        print(f"   Median: {stats['median_hub_improvement']:.4f}")
        print(f"   Best: {stats['max_hub_improvement']:.4f}")
        print(f"   Worst: {stats['min_hub_improvement']:.4f}")

        print(f"\n🏆 EPISODE REWARDS:")
        print(f"   Mean: {stats['mean_episode_reward']:.3f} ± {stats['std_episode_reward']:.3f}")
        print(f"   Median: {stats['median_episode_reward']:.3f}")

        print(f"\n📏 EPISODE LENGTH:")
        print(f"   Mean: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"   Median: {stats['median_episode_length']:.1f}")

        if stats['discriminator_available']:
            print(f"\n🎭 DISCRIMINATOR IMPROVEMENT:")
            print(f"   Mean: {stats['mean_disc_improvement']:.4f} ± {stats['std_disc_improvement']:.4f}")

        print(f"\n🎮 ACTION USAGE:")
        action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                        'ExtractAbstractUnit', 'ExtractUnit', 'STOP']
        for action, name in enumerate(action_names):
            if action in stats['action_distribution']:
                freq = stats['action_distribution'][action]
                success_rate = stats['action_success_rates'].get(action, {}).get('rate', 0)
                print(f"   {name}: {freq:.1%} usage, {success_rate:.1%} success")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_performance_dashboard(evaluation_results: Dict) -> None:
    """Create comprehensive performance dashboard"""

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
    plt.title('Hub Improvement Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Success Rate by Episode Length
    plt.subplot(3, 4, 2)
    lengths = [r['episode_length'] for r in results]
    successes = [r['success'] for r in results]

    length_bins = np.arange(1, max(lengths) + 2)
    success_by_length = []
    for length in length_bins[:-1]:
        episodes_at_length = [s for l, s in zip(lengths, successes) if l == length]
        if episodes_at_length:
            success_by_length.append(np.mean(episodes_at_length))
        else:
            success_by_length.append(0)

    plt.bar(length_bins[:-1], success_by_length, alpha=0.7, color='lightgreen')
    plt.xlabel('Episode Length')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Episode Length')
    plt.grid(True, alpha=0.3)

    # 3. Episode Rewards Over Time
    plt.subplot(3, 4, 3)
    episode_rewards = [r['episode_reward'] for r in results]
    plt.plot(episode_rewards, alpha=0.7, color='blue', linewidth=1)
    plt.axhline(np.mean(episode_rewards), color='red', linestyle='--', alpha=0.7, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
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
    plt.title('Action Usage Distribution')
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
    plt.title('Hub Score: Before vs After')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Performance Categories Pie Chart
    plt.subplot(3, 4, 6)
    categories = ['Excellent (>0.1)', 'Good (0.05-0.1)', 'Moderate (0.01-0.05)', 'Poor (≤0.01)']
    values = [stats['excellent_performance'], stats['good_performance'],
              stats['moderate_performance'], stats['poor_performance']]
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

    plt.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Performance Categories')

    # 7. Action Success Rates
    plt.subplot(3, 4, 7)
    success_rates = [stats['action_success_rates'].get(i, {}).get('rate', 0) for i in range(7)]

    bars = plt.bar(range(7), success_rates, alpha=0.7, color='lightcoral')
    plt.xlabel('Action')
    plt.ylabel('Success Rate')
    plt.title('Action Success Rates')
    plt.xticks(range(7), [name[:8] for name in action_names], rotation=45)
    plt.grid(True, alpha=0.3)

    # Add percentage labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if rate > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{rate:.1%}', ha='center', va='bottom', fontsize=8)

    # 8. Cumulative Success Rate
    plt.subplot(3, 4, 8)
    cumulative_successes = np.cumsum([r['success'] for r in results])
    cumulative_rate = cumulative_successes / np.arange(1, len(results) + 1)

    plt.plot(cumulative_rate, color='green', linewidth=2)
    plt.axhline(stats['success_rate'], color='red', linestyle='--', alpha=0.7,
                label=f'Final: {stats["success_rate"]:.1%}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Success Rate')
    plt.title('Cumulative Success Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 9. Discriminator Improvement (if available)
    if stats['discriminator_available']:
        plt.subplot(3, 4, 9)
        disc_improvements = [r['disc_improvement'] for r in results if r['disc_improvement'] is not None]
        plt.hist(disc_improvements, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        plt.axvline(np.mean(disc_improvements), color='green', linestyle='-', alpha=0.7, label='Mean')
        plt.xlabel('Discriminator Score Improvement')
        plt.ylabel('Frequency')
        plt.title('Discriminator Improvement')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 10. Hub Improvement vs Episode Reward
    plt.subplot(3, 4, 10)
    rewards = [r['episode_reward'] for r in results]
    improvements = [r['hub_improvement'] for r in results]

    plt.scatter(improvements, rewards, alpha=0.6, color='teal')
    plt.xlabel('Hub Score Improvement')
    plt.ylabel('Episode Reward')
    plt.title('Hub Improvement vs Reward')
    plt.grid(True, alpha=0.3)

    # 11. Episode Length Distribution
    plt.subplot(3, 4, 11)
    plt.hist(lengths, bins=range(1, max(lengths) + 2), alpha=0.7, color='gold', edgecolor='black')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(lengths):.1f}')
    plt.xlabel('Episode Length')
    plt.ylabel('Frequency')
    plt.title('Episode Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 12. Success Rate by Hub Improvement Range
    plt.subplot(3, 4, 12)
    improvement_ranges = ['<0', '0-0.01', '0.01-0.05', '0.05-0.1', '>0.1']
    range_counts = [
        sum(1 for imp in hub_improvements if imp < 0),
        sum(1 for imp in hub_improvements if 0 <= imp <= 0.01),
        sum(1 for imp in hub_improvements if 0.01 < imp <= 0.05),
        sum(1 for imp in hub_improvements if 0.05 < imp <= 0.1),
        sum(1 for imp in hub_improvements if imp > 0.1)
    ]

    bars = plt.bar(improvement_ranges, range_counts, alpha=0.7, color='mediumpurple')
    plt.xlabel('Hub Improvement Range')
    plt.ylabel('Number of Episodes')
    plt.title('Episodes by Improvement Range')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, range_counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(Path(eval_config.results_dir) / 'performance_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_graph_comparison(before_graph: Data, after_graph: Data,
                               episode_info: Dict, save_path: Optional[str] = None) -> None:
    """Visualize before/after graph comparison with modifications highlighted"""

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
    ax1.set_title(f"BEFORE Refactoring\nHub Score: {episode_info['initial_hub_score']:.4f}",
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
    ax2.set_title(f"AFTER Refactoring\nHub Score: {episode_info['final_hub_score']:.4f}\n"
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
Episode {episode_info['episode_id']} Summary:
• Actions taken: {len(episode_info['actions_taken'])}
• Episode reward: {episode_info['episode_reward']:.3f}
• Success: {'✅' if episode_info['success'] else '❌'}
• Valid actions: {episode_info['num_valid_actions']}/{len(episode_info['actions_taken'])}
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


def create_interactive_trajectory_plot(episode_data: Dict) -> go.Figure:
    """Create interactive plot of episode trajectory"""

    steps = range(len(episode_data['step_info']) + 1)  # +1 for initial state
    hub_scores = [episode_data['initial_hub_score']]
    hub_scores.extend([step['hub_score'] for step in episode_data['step_info']])

    actions_taken = ['Initial'] + [f"Action {step['action']}" for step in episode_data['step_info']]
    rewards = [0] + episode_data['rewards']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Hub Score Trajectory', 'Step Rewards'),
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

    # Update layout
    fig.update_layout(
        title=f"Episode {episode_data['episode_id']} Trajectory - "
              f"Final Improvement: {episode_data['hub_improvement']:.4f}",
        height=600,
        showlegend=True
    )

    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Hub Score", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=2, col=1)

    return fig


def create_action_analysis_plot(evaluation_results: Dict) -> go.Figure:
    """Create interactive action analysis plot"""

    results = evaluation_results['episode_results']
    stats = evaluation_results['summary_stats']

    action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                    'ExtractAbstractUnit', 'ExtractUnit', 'STOP']

    # Prepare data
    action_usage = [stats['action_distribution'].get(i, 0) * 100 for i in range(7)]
    action_success = [stats['action_success_rates'].get(i, {}).get('rate', 0) * 100 for i in range(7)]

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Action Usage Frequency', 'Action Success Rates'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
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

    # Update layout
    fig.update_layout(
        title="Action Analysis Dashboard",
        height=500,
        showlegend=False
    )

    fig.update_xaxes(title_text="Actions", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Actions", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Usage Frequency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=2)

    return fig


# =============================================================================
# MAIN EVALUATION EXECUTION
# =============================================================================

def run_complete_evaluation():
    """Run complete evaluation pipeline"""

    print("🚀 Starting Complete Evaluation Pipeline")
    print("=" * 60)

    # Create results directory
    Path(eval_config.results_dir).mkdir(parents=True, exist_ok=True)

    # Load models
    print("📦 Loading trained models...")
    model, model_checkpoint = load_trained_model(eval_config.model_path, eval_config.device)
    discriminator = load_discriminator(eval_config.discriminator_path, eval_config.device)

    # Initialize evaluator
    print("🔄 Initializing evaluator...")
    evaluator = GraphRefactoringEvaluator(model, discriminator, eval_config)

    # Run evaluation
    print("📊 Running comprehensive evaluation...")
    evaluation_results = evaluator.run_comprehensive_evaluation()

    # Save results
    results_file = Path(eval_config.results_dir) / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in evaluation_results.items():
            if key == 'episode_results':
                # Handle episode results specially
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
                # Handle trajectories specially (similar to episode_results)
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

    print(f"💾 Results saved to {results_file}")

    # Create visualizations
    print("📈 Creating performance dashboard...")
    create_performance_dashboard(evaluation_results)

    # Create graph comparisons for trajectory episodes
    print("🎨 Creating graph visualizations...")
    for i, episode_data in enumerate(evaluation_results['trajectories']):
        save_path = Path(eval_config.results_dir) / f'graph_comparison_episode_{i}.png'
        visualize_graph_comparison(
            episode_data['initial_graph'],
            episode_data['final_graph'],
            episode_data,
            save_path
        )

    # Create interactive plots
    print("🎭 Creating interactive visualizations...")

    # Trajectory plots for first few episodes
    for i, episode_data in enumerate(evaluation_results['trajectories'][:3]):
        trajectory_fig = create_interactive_trajectory_plot(episode_data)
        trajectory_fig.write_html(Path(eval_config.results_dir) / f'trajectory_episode_{i}.html')

    # Action analysis plot
    action_fig = create_action_analysis_plot(evaluation_results)
    action_fig.write_html(Path(eval_config.results_dir) / 'action_analysis.html')

    print("✅ Evaluation completed successfully!")
    print(f"📁 All results saved to: {eval_config.results_dir}")

    return evaluation_results


# =============================================================================
# ADDITIONAL ANALYSIS FUNCTIONS
# =============================================================================

def analyze_failure_cases(evaluation_results: Dict) -> None:
    """Analyze episodes where the model failed to improve"""

    results = evaluation_results['episode_results']
    failures = [r for r in results if not r['success']]

    if not failures:
        print("🎉 No failure cases found! All episodes were successful.")
        return

    print(f"\n🔍 FAILURE CASE ANALYSIS ({len(failures)} episodes)")
    print("=" * 50)

    # Common failure patterns
    early_stops = [f for f in failures if f['final_action_was_stop'] and f['episode_length'] <= 3]
    long_episodes = [f for f in failures if f['episode_length'] >= 15]
    negative_improvements = [f for f in failures if f['hub_improvement'] < -0.01]

    print(f"📊 Failure Patterns:")
    print(f"   Early stops (≤3 steps): {len(early_stops)} ({len(early_stops) / len(failures):.1%})")
    print(f"   Long episodes (≥15 steps): {len(long_episodes)} ({len(long_episodes) / len(failures):.1%})")
    print(f"   Negative improvements: {len(negative_improvements)} ({len(negative_improvements) / len(failures):.1%})")

    # Action patterns in failures
    failure_actions = []
    for f in failures:
        failure_actions.extend(f['actions_taken'])

    action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                    'ExtractAbstractUnit', 'ExtractUnit', 'STOP']

    print(f"\n🎮 Action Usage in Failures:")
    for action in range(7):
        usage = failure_actions.count(action) / len(failure_actions) if failure_actions else 0
        print(f"   {action_names[action]}: {usage:.1%}")

    # Average metrics for failures
    avg_failure_reward = np.mean([f['episode_reward'] for f in failures])
    avg_failure_length = np.mean([f['episode_length'] for f in failures])
    avg_failure_improvement = np.mean([f['hub_improvement'] for f in failures])

    print(f"\n📉 Failure Metrics:")
    print(f"   Average reward: {avg_failure_reward:.3f}")
    print(f"   Average length: {avg_failure_length:.1f}")
    print(f"   Average improvement: {avg_failure_improvement:.4f}")


def analyze_success_patterns(evaluation_results: Dict) -> None:
    """Analyze patterns in successful episodes"""

    results = evaluation_results['episode_results']
    successes = [r for r in results if r['success']]

    if not successes:
        print("❌ No successful episodes found!")
        return

    print(f"\n🎯 SUCCESS PATTERN ANALYSIS ({len(successes)} episodes)")
    print("=" * 50)

    # Success tiers
    excellent = [s for s in successes if s['hub_improvement'] > 0.1]
    good = [s for s in successes if 0.05 < s['hub_improvement'] <= 0.1]
    moderate = [s for s in successes if 0.01 < s['hub_improvement'] <= 0.05]

    print(f"📈 Success Tiers:")
    print(f"   Excellent (>0.1): {len(excellent)} ({len(excellent) / len(successes):.1%})")
    print(f"   Good (0.05-0.1): {len(good)} ({len(good) / len(successes):.1%})")
    print(f"   Moderate (0.01-0.05): {len(moderate)} ({len(moderate) / len(successes):.1%})")

    # Optimal episode lengths
    success_lengths = [s['episode_length'] for s in successes]
    print(f"\n📏 Episode Length Analysis:")
    print(f"   Mean length: {np.mean(success_lengths):.1f}")
    print(f"   Median length: {np.median(success_lengths):.1f}")
    print(f"   Range: {min(success_lengths)}-{max(success_lengths)}")

    # Most effective actions in successful episodes
    success_actions = []
    for s in successes:
        success_actions.extend(s['actions_taken'])

    action_names = ['RemoveEdge', 'AddEdge', 'MoveEdge', 'ExtractMethod',
                    'ExtractAbstractUnit', 'ExtractUnit', 'STOP']

    print(f"\n🎮 Action Usage in Successes:")
    for action in range(7):
        usage = success_actions.count(action) / len(success_actions) if success_actions else 0
        print(f"   {action_names[action]}: {usage:.1%}")


def create_comparative_analysis(evaluation_results: Dict) -> None:
    """Create comparative analysis between successful and failed episodes"""

    results = evaluation_results['episode_results']
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]

    print(f"\n⚖️  COMPARATIVE ANALYSIS")
    print("=" * 50)

    if not successes or not failures:
        print("Cannot perform comparative analysis - need both successes and failures")
        return

    # Create comparison DataFrame
    comparison_data = {
        'Metric': [
            'Average Episode Length',
            'Average Episode Reward',
            'Average Valid Actions',
            'Average Hub Score Initial',
            'Early Stop Rate (%)',
            'Action RemoveEdge (%)',
            'Action AddEdge (%)',
            'Action ExtractMethod (%)',
            'Action STOP (%)'
        ],
        'Successful Episodes': [
            f"{np.mean([s['episode_length'] for s in successes]):.1f}",
            f"{np.mean([s['episode_reward'] for s in successes]):.3f}",
            f"{np.mean([s['num_valid_actions'] for s in successes]):.1f}",
            f"{np.mean([s['initial_hub_score'] for s in successes]):.4f}",
            f"{sum(1 for s in successes if s['final_action_was_stop'] and s['episode_length'] <= 3) / len(successes) * 100:.1f}",
            f"{sum(s['actions_taken'].count(0) for s in successes) / sum(len(s['actions_taken']) for s in successes) * 100:.1f}",
            f"{sum(s['actions_taken'].count(1) for s in successes) / sum(len(s['actions_taken']) for s in successes) * 100:.1f}",
            f"{sum(s['actions_taken'].count(3) for s in successes) / sum(len(s['actions_taken']) for s in successes) * 100:.1f}",
            f"{sum(s['actions_taken'].count(6) for s in successes) / sum(len(s['actions_taken']) for s in successes) * 100:.1f}",
        ],
        'Failed Episodes': [
            f"{np.mean([f['episode_length'] for f in failures]):.1f}",
            f"{np.mean([f['episode_reward'] for f in failures]):.3f}",
            f"{np.mean([f['num_valid_actions'] for f in failures]):.1f}",
            f"{np.mean([f['initial_hub_score'] for f in failures]):.4f}",
            f"{sum(1 for f in failures if f['final_action_was_stop'] and f['episode_length'] <= 3) / len(failures) * 100:.1f}",
            f"{sum(f['actions_taken'].count(0) for f in failures) / sum(len(f['actions_taken']) for f in failures) * 100:.1f}",
            f"{sum(f['actions_taken'].count(1) for f in failures) / sum(len(f['actions_taken']) for f in failures) * 100:.1f}",
            f"{sum(f['actions_taken'].count(3) for f in failures) / sum(len(f['actions_taken']) for f in failures) * 100:.1f}",
            f"{sum(f['actions_taken'].count(6) for f in failures) / sum(len(f['actions_taken']) for f in failures) * 100:.1f}",
        ]
    }

    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check if model exists
    if not Path(eval_config.model_path).exists():
        print(f"❌ Model not found at {eval_config.model_path}")
        print("Please train the model first or update the model path.")
    else:
        # Run complete evaluation
        results = run_complete_evaluation()

        # Additional analyses
        analyze_failure_cases(results)
        analyze_success_patterns(results)
        create_comparative_analysis(results)

        print(f"\n🎉 Complete evaluation finished!")
        print(f"📊 Check {eval_config.results_dir} for all visualizations and results")

# =============================================================================
# NOTEBOOK EXECUTION CELLS
# =============================================================================

# Cell 1: Setup and Configuration
print("📝 Notebook ready! Run the cells below to execute the evaluation:")
print("\n1. First, ensure your model paths are correct in EvaluationConfig")
print("2. Run run_complete_evaluation() to start the evaluation")
print("3. Check the results directory for all generated visualizations")


# Cell 2: Quick Model Check
def quick_model_check():
    """Quick check if models are available"""
    model_exists = Path(eval_config.model_path).exists()
    disc_exists = Path(eval_config.discriminator_path).exists()
    data_exists = Path(eval_config.data_path).exists()

    print("🔍 MODEL AVAILABILITY CHECK:")
    print(f"   Main model: {'✅' if model_exists else '❌'} {eval_config.model_path}")
    print(f"   Discriminator: {'✅' if disc_exists else '❌'} {eval_config.discriminator_path}")
    print(f"   Data: {'✅' if data_exists else '❌'} {eval_config.data_path}")

    return model_exists and data_exists

# Cell 3: Run Evaluation
# Uncomment the line below to run the evaluation
# results = run_complete_evaluation()