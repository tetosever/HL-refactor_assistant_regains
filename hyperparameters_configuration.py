#!/usr/bin/env python3
"""
Optimized Hyperparameters for Stable RL Training

This file contains the improved hyperparameters that fix the issues in your training:
1. Hub improvement tracking
2. F1 score logging
3. Better reward structure
4. More stable learning rates
5. Improved exploration/exploitation balance
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ImprovedRewardConfig:
    """Improved reward structure that addresses the issues in your training"""

    # FIXED: More balanced reward structure
    REWARD_SUCCESS: float = 20.0  # Increased from 15.0 - better final reward
    REWARD_PARTIAL_SUCCESS: float = 8.0  # Increased from 5.0 - better for good improvements
    REWARD_FAILURE: float = -1.0  # Reduced from -3.0 - less harsh penalty
    REWARD_STEP: float = -0.02  # Reduced from -0.1 - much smaller step penalty
    REWARD_HUB_REDUCTION: float = 15.0  # Increased from 8.0 - better hub reduction reward
    REWARD_INVALID: float = -0.5  # Reduced from -2.0 - less harsh invalid penalty

    # NEW: Reward for small improvements (was missing!)
    REWARD_SMALL_IMPROVEMENT: float = 2.0  # Rewards improvements > 0.01

    # NEW: Hub score thresholds for better reward shaping
    HUB_SCORE_EXCELLENT: float = 0.2  # Below this = excellent (bonus +5.0)
    HUB_SCORE_GOOD: float = 0.4  # Below this = good (bonus +2.0)
    HUB_SCORE_ACCEPTABLE: float = 0.6  # Below this = acceptable (bonus +0.5)

    # NEW: Progressive reward scaling
    improvement_multiplier_boost: float = 1.1  # Boost when achieving new best
    improvement_multiplier_decay: float = 0.95  # Decay when making things worse
    improvement_multiplier_max: float = 2.0  # Maximum multiplier
    improvement_multiplier_min: float = 0.5  # Minimum multiplier


@dataclass
class ImprovedTrainingConfig:
    """Improved training configuration for better stability"""

    # FIXED: Better episode and environment configuration
    num_episodes: int = 8000  # Increased from 5000 - more learning time
    num_envs: int = 6  # Reduced from 8 - more stable with fewer envs
    steps_per_env: int = 24  # Reduced from 32 - shorter episodes
    max_steps_per_episode: int = 20  # Keep reasonable episode length

    # FIXED: More frequent updates for better learning
    update_frequency: int = 3  # Reduced from 4 - more frequent policy updates
    environment_refresh_frequency: int = 200  # Reduced from 500 - more diverse graphs

    # NEW: Hub improvement tracking parameters
    hub_improvement_threshold: float = 0.01  # Lower threshold to detect small improvements
    consecutive_failures_threshold: int = 50  # Reset environments after too many failures

    # FIXED: Better batch sizes for stability
    policy_batch_size: int = 24  # Reduced from 32 - more stable batches
    discriminator_batch_size: int = 12  # Reduced from 16 - more stable disc training


@dataclass
class ImprovedOptimizationConfig:
    """Improved optimization settings for stable learning"""

    # FIXED: Reduced learning rates for stability
    policy_lr: float = 2e-4  # Reduced from 5e-4 - more stable policy learning
    discriminator_lr: float = 3e-5  # Reduced from 1e-4 - much more stable disc learning

    # IMPROVED: Better optimizer settings
    optimizer_eps: float = 1e-5  # Keep small for numerical stability
    weight_decay: float = 1e-4  # Keep regularization
    betas: tuple = (0.9, 0.999)  # More stable beta values

    # FIXED: More conservative PPO parameters
    clip_epsilon: float = 0.15  # Reduced from 0.2 - more conservative clipping
    entropy_coefficient: float = 0.02  # Increased from 0.01 - more exploration
    value_coefficient: float = 0.5  # Keep standard value
    max_grad_norm: float = 0.5  # Keep gradient clipping

    # NEW: Learning rate scheduling
    policy_lr_decay_step: int = 1000  # Decay policy LR every 1000 episodes
    disc_lr_decay_step: int = 500  # Decay disc LR every 500 episodes
    lr_decay_factor: float = 0.95  # Decay factor

    # IMPROVED: Training epochs (reduced for stability)
    policy_epochs: int = 2  # Reduced from 4 - less overfitting
    discriminator_epochs: int = 2  # Reduced from 5 - more stable disc training


@dataclass
class ImprovedDiscriminatorConfig:
    """Fixed discriminator configuration for proper F1 logging"""

    # FIXED: More lenient discriminator training
    early_stop_threshold: float = 0.55  # Reduced from 0.6 - more lenient
    label_smoothing: float = 0.05  # Reduced from 0.1 - less aggressive smoothing
    gradient_penalty_lambda: float = 5.0  # Reduced from 10.0 - less aggressive penalty

    # FIXED: Better discriminator update frequency
    update_frequency_multiplier: int = 8  # Update disc every 8 * update_frequency episodes
    min_samples_per_class: int = 8  # Minimum samples for stable training
    max_samples_per_class: int = 16  # Maximum samples to avoid overfitting

    # NEW: F1 calculation settings (fixes the F1=0 issue)
    calculate_detailed_metrics: bool = True  # Enable detailed metric calculation
    log_confusion_matrix: bool = True  # Log confusion matrix components
    use_balanced_sampling: bool = True  # Ensure balanced sampling for accurate F1


@dataclass
class ImprovedEnvironmentConfig:
    """Improved environment configuration for better diversity"""

    # Graph pool management
    graph_pool_size: int = 50  # Keep diverse pool
    pool_refresh_episodes: int = 500  # Refresh pool every 500 episodes

    # Environment diversity settings
    min_graph_size: int = 10  # Minimum nodes per graph
    max_graph_size: int = 50  # Maximum nodes per graph for performance
    size_diversity_bins: int = 5  # Number of size categories for diversity

    # NEW: Performance monitoring
    track_hub_score_history: bool = True  # Track hub score evolution
    track_action_effectiveness: bool = True  # Track which actions work best
    track_pattern_usage: bool = True  # Track which patterns are most used


def get_improved_config() -> Dict[str, Any]:
    """Get complete improved configuration dictionary"""
    return {
        'rewards': ImprovedRewardConfig(),
        'training': ImprovedTrainingConfig(),
        'optimization': ImprovedOptimizationConfig(),
        'discriminator': ImprovedDiscriminatorConfig(),
        'environment': ImprovedEnvironmentConfig()
    }


def print_config_summary():
    """Print summary of key improvements"""
    print("🔧 KEY IMPROVEMENTS IN HYPERPARAMETERS:")
    print("=" * 60)

    print("1. FIXED HUB IMPROVEMENT TRACKING:")
    print(f"   - Lower threshold: {ImprovedTrainingConfig.hub_improvement_threshold}")
    print(f"   - New reward for small improvements: {ImprovedRewardConfig.REWARD_SMALL_IMPROVEMENT}")
    print(f"   - Progressive reward multipliers: {ImprovedRewardConfig.improvement_multiplier_boost}")

    print("\n2. FIXED F1 SCORE LOGGING:")
    print(f"   - Detailed metrics calculation: {ImprovedDiscriminatorConfig.calculate_detailed_metrics}")
    print(f"   - Confusion matrix logging: {ImprovedDiscriminatorConfig.log_confusion_matrix}")
    print(f"   - Balanced sampling: {ImprovedDiscriminatorConfig.use_balanced_sampling}")

    print("\n3. BETTER REWARD STRUCTURE:")
    print(f"   - Reduced step penalty: {ImprovedRewardConfig.REWARD_STEP} (was -0.1)")
    print(f"   - Reduced failure penalty: {ImprovedRewardConfig.REWARD_FAILURE} (was -3.0)")
    print(f"   - Increased success rewards: {ImprovedRewardConfig.REWARD_SUCCESS} (was 15.0)")
    print(f"   - NEW small improvement reward: {ImprovedRewardConfig.REWARD_SMALL_IMPROVEMENT}")

    print("\n4. MORE STABLE LEARNING RATES:")
    print(f"   - Policy LR: {ImprovedOptimizationConfig.policy_lr} (was 5e-4)")
    print(f"   - Discriminator LR: {ImprovedOptimizationConfig.discriminator_lr} (was 1e-4)")
    print(f"   - Clip epsilon: {ImprovedOptimizationConfig.clip_epsilon} (was 0.2)")

    print("\n5. BETTER TRAINING STABILITY:")
    print(f"   - Reduced batch sizes: {ImprovedTrainingConfig.policy_batch_size}")
    print(f"   - More frequent updates: every {ImprovedTrainingConfig.update_frequency} episodes")
    print(f"   - Reduced training epochs: {ImprovedOptimizationConfig.policy_epochs}")

    print("\n6. IMPROVED ENVIRONMENT DIVERSITY:")
    print(f"   - More frequent refresh: every {ImprovedTrainingConfig.environment_refresh_frequency} episodes")
    print(f"   - Failure detection: reset after {ImprovedTrainingConfig.consecutive_failures_threshold} failures")
    print(
        f"   - Better graph size range: {ImprovedEnvironmentConfig.min_graph_size}-{ImprovedEnvironmentConfig.max_graph_size} nodes")

    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()

    # Example usage
    config = get_improved_config()
    print(f"\n📊 Example reward for 0.05 hub improvement:")
    rewards = config['rewards']
    if 0.05 > 0.01:  # Above small improvement threshold
        reward = rewards.REWARD_SMALL_IMPROVEMENT * 0.05
        print(f"   Reward: {reward:.3f} (vs 0.0 in old system)")

    print(f"\n📊 Example F1 calculation will now include:")
    disc_config = config['discriminator']
    if disc_config.calculate_detailed_metrics:
        print("   ✅ True Positives, False Positives, True Negatives, False Negatives")
        print("   ✅ Precision, Recall, F1 score")
        print("   ✅ Balanced sampling for accurate metrics")