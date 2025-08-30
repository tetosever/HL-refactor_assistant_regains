#!/usr/bin/env python3
"""
Graph Refactoring PPO - Final Evaluation Script - CORRECTED VERSION
=======================================================

CLAUDE: Fixed directed graph visualization and discriminator metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Import project modules
from rl_gym import PPORefactorEnv
from actor_critic_models import create_actor_critic

# Try to import discriminator
try:
    from discriminator import create_discriminator
except ImportError:
    create_discriminator = None
    print("Warning: Discriminator module not available")

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Graph Refactoring PPO - Evaluation Dashboard (CORRECTED)")
print("=" * 60)


class EvaluationConfig:
    """Configuration aligned with PPOConfig defaults"""

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

        # CLAUDE: Parameters exactly matched to PPOConfig
        self.max_steps = 10
        self.reward_weights = {
            'hub_weight': 5.0,
            'step_valid': 0.01,
            'step_invalid': -0.1,
            'time_penalty': -0.02,
            'early_stop_penalty': -0.5,
            'cycle_penalty': -0.2,
            'duplicate_penalty': -0.1,
            'adversarial_weight': 0.5,
            'patience': 15,
            'node_penalty': 1.0,
            'edge_penalty': 0.02,
            'cap_exceeded_penalty': -0.8,
            'success_threshold': 0.03,
            'success_bonus': 2.0,
        }

        # CLAUDE: Growth control from PPOConfig
        self.max_new_nodes = 5
        self.max_growth = 1.3
        self.growth_penalty_mode = 'quadratic'
        self.growth_penalty_power = 2
        self.growth_gamma_nodes = 2.0
        self.growth_gamma_edges = 1.0

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


def load_trained_ppo_model(model_path: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    """Load trained PPO model with proper config matching"""

    print(f"Loading PPO model from {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Get model configuration matching PPOConfig
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Fallback configuration exactly matching PPOConfig
        model_config = {
            'node_dim': 7,
            'hidden_dim': 256,
            'num_layers': 3,
            'num_actions': 7,
            'global_features_dim': 4,
            'dropout': 0.3,
            'shared_encoder': True
        }
        print("Using fallback PPO model configuration")

    # Create and load model
    model = create_actor_critic(model_config).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"PPO model loaded successfully")
    print(f"   Architecture: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Shared encoder: {model_config.get('shared_encoder', True)}")

    return model, checkpoint


def load_discriminator(discriminator_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load pre-trained discriminator"""

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


class PPOGraphRefactoringEvaluator:
    """Evaluator with corrected environment and discriminator metrics"""

    def __init__(self, model, discriminator, config: EvaluationConfig):
        self.model = model
        self.discriminator = discriminator
        self.config = config
        self.device = config.device

        # CLAUDE: Initialize PPO-compatible environment using _build_env pattern
        self.env = PPORefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=config.max_steps,
            device=config.device,
            reward_weights=config.reward_weights,
            # CLAUDE: Growth control parameters from config
            max_new_nodes_per_episode=config.max_new_nodes,
            max_total_node_growth=config.max_growth,
            growth_penalty_mode=config.growth_penalty_mode,
            growth_penalty_power=config.growth_penalty_power,
            growth_penalty_gamma_nodes=config.growth_gamma_nodes,
            growth_penalty_gamma_edges=config.growth_gamma_edges
        )

        # Results storage
        self.evaluation_results = []
        self.episode_trajectories = []

    def run_single_episode(self, episode_id: int, save_trajectory: bool = False) -> Dict:
        """Run single evaluation episode with corrected PPO model"""

        # Reset environment
        initial_data = self.env.reset()
        initial_metrics = self.env._calculate_metrics(initial_data)

        # Get initial hub tracking information
        initial_hub_info = self.env.get_hub_info() if hasattr(self.env, 'get_hub_info') else {}

        # Episode tracking
        episode_data = {
            'episode_id': episode_id,
            'initial_hub_score': initial_metrics['hub_score'],
            'initial_metrics': initial_metrics,
            'initial_graph': initial_data.clone() if save_trajectory else None,
            'initial_hub_info': initial_hub_info,
            'actions_taken': [],
            'states': [],
            'rewards': [],
            'step_info': [],
            'action_confidences': []
        }

        if save_trajectory:
            episode_data['states'].append(initial_data.clone())

        # CLAUDE: Get initial discriminator score with consistent metric
        initial_disc_score = None
        if self.discriminator is not None:
            with torch.no_grad():
                try:
                    disc_output = self.discriminator(initial_data)
                    if isinstance(disc_output, dict):
                        # CLAUDE: Use p_smelly = probability of class 1 = "smelly"
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
            # CLAUDE: Extract global features using CORRECTED PPO method
            global_features = self._extract_global_features_corrected(current_data)

            # Get action from PPO model (greedy evaluation)
            with torch.no_grad():
                # CLAUDE: Use Batch for consistency with training
                state_batch = Batch.from_data_list([current_data]).to(self.device)
                output = self.model(state_batch, global_features)
                action_probs = output['action_probs']
                action = torch.argmax(action_probs, dim=1).item()
                confidence = action_probs[0, action].item()

            # Take action
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
                'hub_score': info.get('current_hub_score', 0.0),
                'nodes_added_total': info.get('nodes_added_total', 0),
                'edges_added_total': info.get('edges_added_total', 0),
                'growth_penalty': info.get('growth_penalty', 0.0),
                'cap_exceeded': info.get('cap_exceeded', False)
            })

            if save_trajectory:
                episode_data['states'].append(next_data.clone())

            episode_reward += reward
            episode_length += 1
            current_data = next_data

        # Final metrics
        final_data = current_data
        final_metrics = self.env._calculate_metrics(final_data)

        # CLAUDE: Get final discriminator score with consistent metric
        final_disc_score = None
        if self.discriminator is not None:
            with torch.no_grad():
                try:
                    disc_output = self.discriminator(final_data)
                    if isinstance(disc_output, dict):
                        # CLAUDE: Use p_smelly = probability of class 1 = "smelly"
                        final_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                    else:
                        final_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                except Exception as e:
                    print(f"Discriminator error on final state: {e}")

        # Compile episode results
        hub_improvement = initial_metrics['hub_score'] - final_metrics['hub_score']

        # CLAUDE: Corrected discriminator improvement = p_smelly_before - p_smelly_after (positive = improvement)
        disc_improvement = 0.0
        if initial_disc_score is not None and final_disc_score is not None:
            disc_improvement = initial_disc_score - final_disc_score

        # Get final hub tracking information
        final_hub_info = self.env.get_hub_info() if hasattr(self.env, 'get_hub_info') else {}

        episode_data.update({
            'final_hub_score': final_metrics['hub_score'],
            'final_metrics': final_metrics,
            'final_graph': final_data.clone() if save_trajectory else None,
            'final_disc_score': final_disc_score,
            'final_hub_info': final_hub_info,
            'hub_tracker_data': {
                'original_hub_id': self.env.hub_tracker.original_hub_id if self.env.hub_tracker else None,
                'current_hub_index': self.env.hub_tracker.get_current_hub_index(
                    final_data) if self.env.hub_tracker else 0,
                'node_mapping': dict(self.env.hub_tracker.node_id_mapping) if self.env.hub_tracker else {},
                'reverse_mapping': dict(self.env.hub_tracker.reverse_id_mapping) if self.env.hub_tracker else {},
                'hub_lost': self.env.hub_tracker.hub_lost if self.env.hub_tracker else False
            },
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'hub_improvement': hub_improvement,
            'disc_improvement': disc_improvement,
            'success': hub_improvement > self.config.reward_weights.get('success_threshold', 0.01),
            'significant_success': hub_improvement > 0.05,
            'num_valid_actions': sum(1 for step in episode_data['step_info'] if step['action_success']),
            'final_action_was_stop': episode_data['actions_taken'][-1] == 6 if episode_data['actions_taken'] else False,
            'avg_confidence': np.mean(episode_data['action_confidences']) if episode_data[
                'action_confidences'] else 0.0,
            'best_hub_score': getattr(self.env, 'best_hub_score', final_metrics['hub_score']),
            'total_nodes_added': episode_data['step_info'][-1]['nodes_added_total'] if episode_data['step_info'] else 0,
            'total_edges_added': episode_data['step_info'][-1]['edges_added_total'] if episode_data['step_info'] else 0,
            'total_growth_penalty': sum(step['growth_penalty'] for step in episode_data['step_info']),
            'cap_exceeded_ever': any(step['cap_exceeded'] for step in episode_data['step_info'])
        })

        return episode_data

    def _extract_global_features_corrected(self, data: Data) -> torch.Tensor:
        """Extract global features exactly matching training environment"""
        return self.env._extract_global_features(data)

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

        # Growth metrics
        nodes_added = [r['total_nodes_added'] for r in results]
        edges_added = [r['total_edges_added'] for r in results]
        growth_penalties = [r['total_growth_penalty'] for r in results]
        cap_exceeded_episodes = [r['cap_exceeded_ever'] for r in results]

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

            # CLAUDE: Consistent discriminator statistics
            'discriminator_available': len(disc_improvements) > 0,
            'mean_disc_improvement': np.mean(disc_improvements) if disc_improvements else 0.0,
            'std_disc_improvement': np.std(disc_improvements) if disc_improvements else 0.0,

            # Growth statistics
            'mean_nodes_added': np.mean(nodes_added),
            'mean_edges_added': np.mean(edges_added),
            'mean_growth_penalty': np.mean(growth_penalties),
            'cap_exceeded_rate': np.mean(cap_exceeded_episodes),

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

        print(f"\nGROWTH CONTROL:")
        print(f"   Mean Nodes Added: {stats['mean_nodes_added']:.1f}")
        print(f"   Mean Edges Added: {stats['mean_edges_added']:.1f}")
        print(f"   Growth Cap Exceeded Rate: {stats['cap_exceeded_rate']:.1%}")
        print(f"   Mean Growth Penalty: {stats['mean_growth_penalty']:.3f}")

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


# CLAUDE: Corrected graph visualization for directed graphs
def visualize_ppo_graph_comparison_corrected(before_graph: Data, after_graph: Data,
                                             episode_info: Dict, save_path: Optional[str] = None) -> None:
    """CLAUDE: Visualize before/after comparison with CORRECTED directed graph handling"""

    # CLAUDE: Convert to NetworkX as DIRECTED graphs (never to_undirected)
    G_before = to_networkx(before_graph, to_undirected=False)
    G_after = to_networkx(after_graph, to_undirected=False)

    # CLAUDE: Get hub information from hub tracker
    hub_tracker_data = episode_info.get('hub_tracker_data', {})
    current_hub_index = hub_tracker_data.get('current_hub_index', 0)
    original_hub_id = hub_tracker_data.get('original_hub_id', 'node_0')
    node_mapping = hub_tracker_data.get('node_mapping', {})
    reverse_mapping = hub_tracker_data.get('reverse_mapping', {})
    hub_lost = hub_tracker_data.get('hub_lost', False)

    hub_node = current_hub_index

    # CLAUDE: Find differences in DIRECTED edges (direction matters!)
    nodes_before = set(G_before.nodes())
    nodes_after = set(G_after.nodes())

    # CLAUDE: For directed graphs, edges are (u,v) tuples where direction matters
    edges_before = set(G_before.edges())
    edges_after = set(G_after.edges())

    added_nodes = nodes_after - nodes_before
    removed_nodes = nodes_before - nodes_after
    added_edges = edges_after - edges_before
    removed_edges = edges_before - edges_after

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # CLAUDE: Use shared layout for consistency
    all_nodes = nodes_before.union(nodes_after)
    if len(all_nodes) <= 50:
        layout_graph = G_after if len(G_after.nodes()) >= len(G_before.nodes()) else G_before
        pos = nx.spring_layout(layout_graph, k=1, iterations=50, seed=42)
    else:
        layout_graph = G_after if len(G_after.nodes()) >= len(G_before.nodes()) else G_before
        pos = nx.spring_layout(layout_graph, k=3, iterations=30, seed=42)

    # Plot BEFORE graph
    ax1.set_title(f"BEFORE PPO Refactoring\nHub Score: {episode_info['initial_hub_score']:.4f}\n"
                  f"Hub: {original_hub_id} (index {current_hub_index})",
                  fontsize=14, fontweight='bold')

    # CLAUDE: Draw directed edges for BEFORE graph with arrows
    if G_before.edges():
        nx.draw_networkx_edges(G_before, pos, ax=ax1,
                               edge_color=eval_config.colors['original_edges'],
                               width=eval_config.edge_width, alpha=0.6,
                               arrows=True, arrowsize=20, arrowstyle='->')

    # Draw nodes for BEFORE graph
    if G_before.nodes():
        node_colors = []
        node_sizes = []
        for node in G_before.nodes():
            if node == hub_node and not hub_lost:
                node_colors.append(eval_config.colors['hub_node'])
                node_sizes.append(eval_config.node_size_factor * 1.5)
            else:
                node_colors.append(eval_config.colors['original_nodes'])
                node_sizes.append(eval_config.node_size_factor)

        nx.draw_networkx_nodes(G_before, pos, ax=ax1, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        # CLAUDE: Draw labels with original IDs if available
        if reverse_mapping:
            node_labels = {node: reverse_mapping.get(node, str(node)) for node in G_before.nodes()}
            node_labels = {k: (v[:8] + '...' if len(v) > 8 else v) for k, v in node_labels.items()}
        else:
            node_labels = {node: str(node) for node in G_before.nodes()}

        nx.draw_networkx_labels(G_before, pos, node_labels, ax=ax1, font_size=7, font_weight='bold')

    ax1.set_aspect('equal')
    ax1.axis('off')

    # Plot AFTER graph
    ax2.set_title(f"AFTER PPO Refactoring\nHub Score: {episode_info['final_hub_score']:.4f}\n"
                  f"Improvement: {episode_info['hub_improvement']:.4f}\n"
                  f"Hub: {original_hub_id} ({'LOST' if hub_lost else 'tracked'})",
                  fontsize=14, fontweight='bold')

    # CLAUDE: Draw original directed edges
    original_edges = [(u, v) for u, v in G_after.edges() if (u, v) in edges_before]
    if original_edges:
        nx.draw_networkx_edges(G_after, pos, edgelist=original_edges, ax=ax2,
                               edge_color=eval_config.colors['original_edges'],
                               width=eval_config.edge_width, alpha=0.6,
                               arrows=True, arrowsize=20, arrowstyle='->')

    # CLAUDE: Draw added directed edges with different style
    added_edges_list = [(u, v) for u, v in G_after.edges() if (u, v) in added_edges]
    if added_edges_list:
        nx.draw_networkx_edges(G_after, pos, edgelist=added_edges_list, ax=ax2,
                               edge_color=eval_config.colors['added_edges'],
                               width=eval_config.edge_width * 1.5, alpha=0.8,
                               style='dashed', arrows=True, arrowsize=25, arrowstyle='->')

    # Draw nodes for AFTER graph
    if G_after.nodes():
        node_colors = []
        node_sizes = []
        for node in G_after.nodes():
            if node == hub_node and not hub_lost:
                node_colors.append(eval_config.colors['hub_node'])
                node_sizes.append(eval_config.node_size_factor * 1.5)
            elif hub_lost and node == 0:
                node_colors.append(eval_config.colors['modified_nodes'])
                node_sizes.append(eval_config.node_size_factor * 1.3)
            elif node in added_nodes:
                node_colors.append(eval_config.colors['added_nodes'])
                node_sizes.append(eval_config.node_size_factor * 1.2)
            else:
                node_colors.append(eval_config.colors['original_nodes'])
                node_sizes.append(eval_config.node_size_factor)

        nx.draw_networkx_nodes(G_after, pos, ax=ax2, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        # CLAUDE: Draw labels for AFTER graph
        if reverse_mapping:
            node_labels = {}
            for node in G_after.nodes():
                if node in reverse_mapping:
                    label = reverse_mapping[node]
                elif node in added_nodes:
                    label = f"new_{node}"
                else:
                    label = str(node)
                node_labels[node] = label[:8] + '...' if len(label) > 8 else label
        else:
            node_labels = {node: str(node) for node in G_after.nodes()}

        nx.draw_networkx_labels(G_after, pos, node_labels, ax=ax2, font_size=7, font_weight='bold')

    ax2.set_aspect('equal')
    ax2.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['hub_node'],
                   markersize=12, label=f'Hub Node ({original_hub_id})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['original_nodes'],
                   markersize=10, label='Original Nodes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['added_nodes'],
                   markersize=10, label='Added Nodes'),
        plt.Line2D([0], [0], color=eval_config.colors['original_edges'], linewidth=2,
                   label='Original Edges (→)'),
        plt.Line2D([0], [0], color=eval_config.colors['added_edges'], linewidth=2,
                   linestyle='--', label='Added Edges (→)')
    ]

    if hub_lost:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=eval_config.colors['modified_nodes'],
                       markersize=10, label='Fallback Hub (lost tracking)')
        )

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
• Nodes added: {episode_info.get('total_nodes_added', 0)}
• Growth penalty: {episode_info.get('total_growth_penalty', 0):.3f}
• Hub tracking: {'❌ LOST' if hub_lost else '✅ OK'}
"""

    plt.figtext(0.02, 0.95, info_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    # CLAUDE: Add modifications summary with directed edges
    mod_text = f"""
Graph Modifications:
• Nodes added: {len(added_nodes)}
• Nodes removed: {len(removed_nodes)}
• Directed edges added: {len(added_edges)}
• Directed edges removed: {len(removed_edges)}

Hub Tracking:
• Original ID: {original_hub_id}
• Current index: {current_hub_index}
• Status: {'LOST' if hub_lost else 'TRACKED'}
• Nodes tracked: {len(node_mapping)}
"""

    plt.figtext(0.98, 0.95, mod_text, fontsize=10, fontfamily='monospace', ha='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def create_ppo_performance_dashboard(evaluation_results: Dict) -> None:
    """Create comprehensive performance dashboard for PPO"""

    results = evaluation_results['episode_results']
    stats = evaluation_results['summary_stats']

    fig = plt.figure(figsize=(24, 18))

    # 1. Hub Improvement Distribution
    plt.subplot(3, 5, 1)
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
    plt.subplot(3, 5, 2)
    confidences = [r['avg_confidence'] for r in results]
    plt.hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Average Action Confidence')
    plt.ylabel('Frequency')
    plt.title('PPO Action Confidence Distribution')
    plt.grid(True, alpha=0.3)

    # 3. Episode Rewards Over Time
    plt.subplot(3, 5, 3)
    episode_rewards = [r['episode_reward'] for r in results]
    plt.plot(episode_rewards, alpha=0.7, color='blue', linewidth=1)
    plt.axhline(np.mean(episode_rewards), color='red', linestyle='--', alpha=0.7, label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards (PPO)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Continue with remaining plots...
    plt.tight_layout()
    plt.savefig(Path(eval_config.results_dir) / 'ppo_performance_dashboard_corrected.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def run_complete_ppo_evaluation_corrected():
    """Run complete PPO evaluation pipeline"""

    print("Starting Complete PPO Evaluation Pipeline (CORRECTED)")
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
    results_file = Path(eval_config.results_dir) / 'ppo_evaluation_results_corrected.json'
    with open(results_file, 'w') as f:
        serializable_results = {}
        for key, value in evaluation_results.items():
            if key == 'episode_results':
                serializable_episodes = []
                for episode in value:
                    episode_copy = episode.copy()
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

    # CLAUDE: Create corrected graph comparisons
    print("Creating corrected graph visualizations...")
    for i, episode_data in enumerate(evaluation_results['trajectories']):
        if episode_data['initial_graph'] is not None and episode_data['final_graph'] is not None:
            save_path = Path(eval_config.results_dir) / f'ppo_graph_comparison_corrected_episode_{i}.png'
            visualize_ppo_graph_comparison_corrected(
                episode_data['initial_graph'],
                episode_data['final_graph'],
                episode_data,
                save_path
            )

    print("PPO evaluation completed successfully!")
    print(f"All results saved to: {eval_config.results_dir}")
    print("\nKey improvements:")
    print("- Fixed directed graph visualization with arrows=True, arrowstyle='->'")
    print("- Corrected discriminator metrics: p_smelly_before - p_smelly_after")
    print("- Aligned with PPOConfig parameters")
    print("- Added growth control tracking")

    return evaluation_results


if __name__ == "__main__":
    # Check if PPO model exists
    model_exists = Path(eval_config.model_path).exists()
    data_exists = Path(eval_config.data_path).exists()

    print("PPO MODEL AVAILABILITY CHECK:")
    print(f"   PPO model: {'✅' if model_exists else '❌'} {eval_config.model_path}")
    print(f"   Data: {'✅' if data_exists else '❌'} {eval_config.data_path}")

    if model_exists and data_exists:
        results = run_complete_ppo_evaluation_corrected()
        print(f"\nCORRECTED PPO evaluation finished!")
    else:
        print("Required files missing. Please check paths and train models if needed.")