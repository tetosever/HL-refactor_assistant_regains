#!/usr/bin/env python3
"""
Enhanced Hub Score Computation with Global Structural Metrics
Adds global graph metrics to make hub-score more stable and informative
"""

import math
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GlobalGraphMetrics:
    """Compute global structural metrics for hub detection"""

    @staticmethod
    def compute_gini_coefficient(degrees: list) -> float:
        """
        Compute Gini coefficient for degree distribution
        Returns value in [0,1] where 1 = maximum inequality (hub-like structure)
        """
        try:
            if not degrees or len(degrees) <= 1:
                return 0.0

            # Remove zeros and sort
            degrees = [d for d in degrees if d > 0]
            if len(degrees) <= 1:
                return 0.0

            degrees = sorted(degrees)
            n = len(degrees)

            # Compute Gini coefficient
            cumsum = np.cumsum(degrees)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

            # Clamp to [0,1] and handle edge cases
            return max(0.0, min(1.0, gini))

        except Exception as e:
            logger.debug(f"Gini coefficient computation failed: {e}")
            return 0.0

    @staticmethod
    def compute_degree_centralization(G: nx.Graph) -> float:
        """
        Compute Freeman's degree centralization
        Returns value in [0,1] where 1 = star graph (maximum centralization)
        """
        try:
            if len(G.nodes()) <= 2:
                return 0.0

            # Get degree centrality values
            degree_cent = nx.degree_centrality(G)
            if not degree_cent:
                return 0.0

            centralities = list(degree_cent.values())
            max_centrality = max(centralities)

            # Freeman's centralization formula
            n = len(G.nodes())
            numerator = sum(max_centrality - c for c in centralities)

            # Maximum possible centralization (star graph)
            if n <= 2:
                return 0.0

            if G.is_directed():
                max_possible = (n - 1) * (n - 2)
            else:
                max_possible = (n - 1) * (n - 2) / 2

            if max_possible <= 0:
                return 0.0

            centralization = numerator / max_possible
            return max(0.0, min(1.0, centralization))

        except Exception as e:
            logger.debug(f"Degree centralization computation failed: {e}")
            return 0.0

    @staticmethod
    def compute_betweenness_centralization(G: nx.Graph) -> float:
        """
        Compute betweenness centralization
        Returns value in [0,1] where 1 = maximum centralization (hub structure)
        """
        try:
            n = len(G.nodes())
            if n <= 3:
                return 0.0

            # For large graphs, use sampling approximation
            if n > 100:
                # Sample-based betweenness for efficiency
                k = min(50, n)
                try:
                    bet_cent = nx.betweenness_centrality(G, k=k, normalized=True)
                except:
                    # Fallback to basic computation with timeout
                    bet_cent = GlobalGraphMetrics._compute_betweenness_fallback(G)
            else:
                try:
                    bet_cent = nx.betweenness_centrality(G, normalized=True)
                except:
                    bet_cent = GlobalGraphMetrics._compute_betweenness_fallback(G)

            if not bet_cent:
                return 0.0

            centralities = list(bet_cent.values())
            max_centrality = max(centralities)

            # Centralization formula for betweenness
            numerator = sum(max_centrality - c for c in centralities)

            # Maximum possible betweenness centralization
            if G.is_directed():
                max_possible = (n - 1) * (n - 2)
            else:
                max_possible = (n - 1) * (n - 2) / 2

            if max_possible <= 0:
                return 0.0

            centralization = numerator / max_possible
            return max(0.0, min(1.0, centralization))

        except Exception as e:
            logger.debug(f"Betweenness centralization computation failed: {e}")
            return 0.0

    @staticmethod
    def _compute_betweenness_fallback(G: nx.Graph) -> Dict[int, float]:
        """Fallback betweenness computation for problematic graphs"""
        try:
            # Simple degree-based approximation when betweenness fails
            degree_cent = nx.degree_centrality(G)
            # Normalize to approximate betweenness distribution
            return {node: cent * 0.5 for node, cent in degree_cent.items()}
        except:
            # Ultimate fallback - uniform distribution
            return {node: 0.0 for node in G.nodes()}


class AdaptiveWeightScheduler:
    """Manages adaptive weights for hub-score components during training"""

    def __init__(self):
        # Training phase thresholds
        self.early_threshold = 2000
        self.mid_threshold = 6000

        # Weight configurations for different phases
        self.weight_configs = {
            'early': {
                'gini_weight': 0.35,
                'degree_centralization_weight': 0.30,
                'betweenness_centralization_weight': 0.25,
                'discriminator_weight': 0.10
            },
            'mid': {
                'gini_weight': 0.25,
                'degree_centralization_weight': 0.25,
                'betweenness_centralization_weight': 0.25,
                'discriminator_weight': 0.25
            },
            'late': {
                'gini_weight': 0.15,
                'degree_centralization_weight': 0.15,
                'betweenness_centralization_weight': 0.20,
                'discriminator_weight': 0.50
            }
        }

    def get_weights(self, episode: int) -> Dict[str, float]:
        """Get current weights based on training episode"""
        if episode < self.early_threshold:
            return self.weight_configs['early']
        elif episode < self.mid_threshold:
            # Linear interpolation between early and mid
            progress = (episode - self.early_threshold) / (self.mid_threshold - self.early_threshold)
            early_weights = self.weight_configs['early']
            mid_weights = self.weight_configs['mid']

            interpolated = {}
            for key in early_weights:
                interpolated[key] = (1 - progress) * early_weights[key] + progress * mid_weights[key]
            return interpolated
        else:
            # Linear interpolation between mid and late
            max_episodes = 10000  # Assume maximum training episodes
            progress = min(1.0, (episode - self.mid_threshold) / (max_episodes - self.mid_threshold))
            mid_weights = self.weight_configs['mid']
            late_weights = self.weight_configs['late']

            interpolated = {}
            for key in mid_weights:
                interpolated[key] = (1 - progress) * mid_weights[key] + progress * late_weights[key]
            return interpolated

    def get_phase_name(self, episode: int) -> str:
        """Get current training phase name"""
        if episode < self.early_threshold:
            return "early"
        elif episode < self.mid_threshold:
            return "mid"
        else:
            return "late"


# Enhanced HubRefactoringEnv methods to add to the existing class
class EnhancedHubScoreComputation:
    """Enhanced hub-score computation methods to be integrated into HubRefactoringEnv"""

    def __init__(self):
        # Initialize adaptive weight scheduler
        self.weight_scheduler = AdaptiveWeightScheduler()

        # Track current episode for weight adaptation
        self.current_episode = 0

        # Cache for expensive global computations
        self._cached_global_metrics = None
        self._global_metrics_hash = None

        # Component score tracking for debugging
        self.last_component_scores = {
            'gini': 0.0,
            'degree_centralization': 0.0,
            'betweenness_centralization': 0.0,
            'discriminator': 0.0,
            'composite': 0.0
        }

    def set_current_episode(self, episode: int):
        """Set current episode for adaptive weighting"""
        self.current_episode = episode

    def _compute_global_structural_metrics(self, data) -> Dict[str, float]:
        """
        Compute global structural metrics for hub detection
        Returns normalized metrics where higher values indicate more hub-like structure
        """
        try:
            # Check cache first
            current_hash = self._get_graph_hash(data)
            if (self._cached_global_metrics is not None and
                    self._global_metrics_hash == current_hash):
                return self._cached_global_metrics

            # Convert to NetworkX
            G = to_networkx(data, to_undirected=False)

            if len(G.nodes()) < 3:
                # Not enough nodes for meaningful global metrics
                metrics = {
                    'gini_coefficient': 0.0,
                    'degree_centralization': 0.0,
                    'betweenness_centralization': 0.0
                }
            else:
                # Compute degree sequence
                degrees = [G.degree(node) for node in G.nodes()]

                # 1. Gini coefficient on degree distribution
                gini = GlobalGraphMetrics.compute_gini_coefficient(degrees)

                # 2. Degree centralization
                degree_cent = GlobalGraphMetrics.compute_degree_centralization(G)

                # 3. Betweenness centralization
                betweenness_cent = GlobalGraphMetrics.compute_betweenness_centralization(G)

                metrics = {
                    'gini_coefficient': gini,
                    'degree_centralization': degree_cent,
                    'betweenness_centralization': betweenness_cent
                }

            # Cache the results
            self._cached_global_metrics = metrics
            self._global_metrics_hash = current_hash

            return metrics

        except Exception as e:
            logger.warning(f"Global metrics computation failed: {e}")
            return {
                'gini_coefficient': 0.0,
                'degree_centralization': 0.0,
                'betweenness_centralization': 0.0
            }

    def _get_discriminator_score(self) -> float:
        """
        Get discriminator-based hub score (existing implementation)
        Returns value in [0,1] where 1 = more hub-like/smelly
        """
        try:
            self._ensure_data_consistency(self.current_data)

            with torch.no_grad():
                self.discriminator.eval()
                disc_out = self.discriminator(self.current_data)

                if disc_out is None or 'logits' not in disc_out:
                    return 0.5  # Neutral fallback

                logits = disc_out['logits']
                if logits.dim() == 0 or logits.size(0) == 0:
                    return 0.5

                # Get probability of being "smelly" (hub-like)
                probs = F.softmax(logits, dim=1)
                if probs.size(1) < 2:
                    return 0.5

                score = probs[0, 1].item()

            # Validate score
            if not (0.0 <= score <= 1.0) or math.isnan(score):
                score = 0.5

            return score

        except Exception as e:
            logger.debug(f"Discriminator score computation failed: {e}")
            return 0.5

    def _get_current_hub_score(self) -> float:
        """
        Enhanced hub score computation with global structural metrics
        Combines 4 components with adaptive weights based on training progress
        """
        try:
            # Check if we can use cached composite score
            current_hash = self._get_graph_hash(self.current_data)
            if (self._cached_hub_score is not None and
                    self._graph_hash == current_hash):
                return self._cached_hub_score

            # Get adaptive weights for current training phase
            weights = self.weight_scheduler.get_weights(self.current_episode)
            phase = self.weight_scheduler.get_phase_name(self.current_episode)

            # 1. Compute global structural metrics
            global_metrics = self._compute_global_structural_metrics(self.current_data)

            # 2. Get discriminator score
            discriminator_score = self._get_discriminator_score()

            # 3. Combine all components with adaptive weights
            composite_score = (
                    weights['gini_weight'] * global_metrics['gini_coefficient'] +
                    weights['degree_centralization_weight'] * global_metrics['degree_centralization'] +
                    weights['betweenness_centralization_weight'] * global_metrics['betweenness_centralization'] +
                    weights['discriminator_weight'] * discriminator_score
            )

            # Ensure score is in [0,1] range
            composite_score = max(0.0, min(1.0, composite_score))

            # Store component scores for debugging
            self.last_component_scores = {
                'gini': global_metrics['gini_coefficient'],
                'degree_centralization': global_metrics['degree_centralization'],
                'betweenness_centralization': global_metrics['betweenness_centralization'],
                'discriminator': discriminator_score,
                'composite': composite_score,
                'weights_used': weights,
                'training_phase': phase
            }

            # Cache the result
            self._cached_hub_score = composite_score
            self._graph_hash = current_hash

            # Debug logging (only if debug level enabled)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"🎯 Hub Score Components (Episode {self.current_episode}, Phase: {phase}):\n"
                    f"   Gini: {global_metrics['gini_coefficient']:.3f} (weight: {weights['gini_weight']:.2f})\n"
                    f"   Degree Cent: {global_metrics['degree_centralization']:.3f} (weight: {weights['degree_centralization_weight']:.2f})\n"
                    f"   Between Cent: {global_metrics['betweenness_centralization']:.3f} (weight: {weights['betweenness_centralization_weight']:.2f})\n"
                    f"   Discriminator: {discriminator_score:.3f} (weight: {weights['discriminator_weight']:.2f})\n"
                    f"   💯 COMPOSITE: {composite_score:.3f}"
                )

            return composite_score

        except Exception as e:
            logger.error(f"Enhanced hub score computation failed: {e}")
            # Fallback to discriminator-only score
            return self._get_discriminator_score()

    def get_hub_score_breakdown(self) -> Dict[str, any]:
        """
        Get detailed breakdown of hub score components for analysis
        Useful for debugging and understanding model behavior
        """
        return {
            'component_scores': self.last_component_scores.copy(),
            'current_episode': self.current_episode,
            'training_phase': self.weight_scheduler.get_phase_name(self.current_episode),
            'current_weights': self.weight_scheduler.get_weights(self.current_episode)
        }

    def log_hub_score_analysis(self, detailed: bool = False) -> str:
        """
        Generate formatted logging string for hub score analysis
        """
        breakdown = self.get_hub_score_breakdown()
        components = breakdown['component_scores']

        if not components:
            return "Hub score breakdown not available"

        log_lines = []
        log_lines.append(f"🎯 HUB SCORE ANALYSIS (Episode {self.current_episode}):")
        log_lines.append(f"   Training Phase: {breakdown['training_phase']}")
        log_lines.append(f"   Final Composite Score: {components.get('composite', 0.0):.4f}")

        if detailed:
            weights = components.get('weights_used', {})
            log_lines.append("   📊 Component Breakdown:")
            log_lines.append(
                f"      Gini Coefficient: {components.get('gini', 0.0):.3f} × {weights.get('gini_weight', 0.0):.2f} = {components.get('gini', 0.0) * weights.get('gini_weight', 0.0):.3f}")
            log_lines.append(
                f"      Degree Centralization: {components.get('degree_centralization', 0.0):.3f} × {weights.get('degree_centralization_weight', 0.0):.2f} = {components.get('degree_centralization', 0.0) * weights.get('degree_centralization_weight', 0.0):.3f}")
            log_lines.append(
                f"      Betweenness Centralization: {components.get('betweenness_centralization', 0.0):.3f} × {weights.get('betweenness_centralization_weight', 0.0):.2f} = {components.get('betweenness_centralization', 0.0) * weights.get('betweenness_centralization_weight', 0.0):.3f}")
            log_lines.append(
                f"      Discriminator Score: {components.get('discriminator', 0.0):.3f} × {weights.get('discriminator_weight', 0.0):.2f} = {components.get('discriminator', 0.0) * weights.get('discriminator_weight', 0.0):.3f}")
        else:
            log_lines.append(
                f"   Components: G={components.get('gini', 0.0):.2f}, DC={components.get('degree_centralization', 0.0):.2f}, BC={components.get('betweenness_centralization', 0.0):.2f}, D={components.get('discriminator', 0.0):.2f}")

        return "\n".join(log_lines)

    def invalidate_global_cache(self):
        """Invalidate global metrics cache when graph changes"""
        self._cached_global_metrics = None
        self._global_metrics_hash = None
        # Also invalidate hub score cache
        self._cached_hub_score = None
        self._graph_hash = None


# Integration helper functions for adding to HubRefactoringEnv
def integrate_enhanced_hub_score(env_instance):
    """
    Helper function to integrate enhanced hub score into existing HubRefactoringEnv
    """
    # Add the enhanced computation methods
    enhanced = EnhancedHubScoreComputation()

    # Bind methods to the environment instance
    env_instance.weight_scheduler = enhanced.weight_scheduler
    env_instance.current_episode = enhanced.current_episode
    env_instance._cached_global_metrics = enhanced._cached_global_metrics
    env_instance._global_metrics_hash = enhanced._global_metrics_hash
    env_instance.last_component_scores = enhanced.last_component_scores

    # Replace methods
    env_instance.set_current_episode = enhanced.set_current_episode.__get__(env_instance)
    env_instance._compute_global_structural_metrics = enhanced._compute_global_structural_metrics.__get__(env_instance)
    env_instance._get_discriminator_score = enhanced._get_discriminator_score.__get__(env_instance)
    env_instance._get_current_hub_score = enhanced._get_current_hub_score.__get__(env_instance)
    env_instance.get_hub_score_breakdown = enhanced.get_hub_score_breakdown.__get__(env_instance)
    env_instance.log_hub_score_analysis = enhanced.log_hub_score_analysis.__get__(env_instance)
    env_instance.invalidate_global_cache = enhanced.invalidate_global_cache.__get__(env_instance)

    # Override the step method to update episode tracking
    original_step = env_instance.step

    def enhanced_step(action):
        # Invalidate caches when graph changes
        env_instance.invalidate_global_cache()
        return original_step(action)

    env_instance.step = enhanced_step

    return env_instance


# Usage example and testing
def test_enhanced_hub_score():
    """Test function for enhanced hub score computation"""
    import torch
    from torch_geometric.data import Data

    # Create test graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # 7 structural features per node
    x = torch.tensor([
        [3.0, 2.0, 0.6, 1.5, 0.4, 0.3, 0.5],  # Hub-like node
        [1.0, 1.0, 0.3, 1.0, 0.1, 0.1, 0.3],  # Regular node
        [1.0, 1.0, 0.3, 1.0, 0.1, 0.1, 0.3],  # Regular node
        [1.0, 1.0, 0.3, 1.0, 0.1, 0.1, 0.3],  # Regular node
    ], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    # Test global metrics computation
    enhanced = EnhancedHubScoreComputation()
    enhanced.current_data = data  # Mock current_data

    print("Testing Enhanced Hub Score Computation:")
    print("=" * 50)

    # Test at different training phases
    for episode in [500, 3000, 7000]:
        enhanced.set_current_episode(episode)

        # Mock discriminator score
        enhanced._get_discriminator_score = lambda: 0.7
        enhanced._get_graph_hash = lambda d: f"test_hash_{episode}"
        enhanced._ensure_data_consistency = lambda d: d

        global_metrics = enhanced._compute_global_structural_metrics(data)
        composite_score = enhanced._get_current_hub_score()

        print(f"\nEpisode {episode}:")
        print(f"  Global Metrics: {global_metrics}")
        print(f"  Composite Score: {composite_score:.4f}")
        print(f"  Phase: {enhanced.weight_scheduler.get_phase_name(episode)}")
        print(enhanced.log_hub_score_analysis(detailed=True))


if __name__ == "__main__":
    test_enhanced_hub_score()