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
    REWARD_SUCCESS: float = 20.0
    REWARD_PARTIAL_SUCCESS: float = 8.0
    REWARD_FAILURE: float = -1.0
    REWARD_STEP: float = -0.02
    REWARD_HUB_REDUCTION: float = 15.0
    REWARD_INVALID: float = -0.5

    # NEW: Reward for small improvements (was missing!)
    REWARD_SMALL_IMPROVEMENT: float = 2.0

    # NEW: Hub score thresholds for better reward shaping
    HUB_SCORE_EXCELLENT: float = 0.2
    HUB_SCORE_GOOD: float = 0.4
    HUB_SCORE_ACCEPTABLE: float = 0.6

    # NEW: Progressive reward scaling
    improvement_multiplier_boost: float = 1.1
    improvement_multiplier_decay: float = 0.95
    improvement_multiplier_max: float = 2.0
    improvement_multiplier_min: float = 0.5


@dataclass
class ImprovedTrainingConfig:
    """Improved training configuration for better stability"""

    # FIXED: Better episode and environment configuration
    num_episodes: int = 8000
    num_envs: int = 16
    steps_per_env: int = 24
    max_steps_per_episode: int = 20

    # FIXED: More frequent updates for better learning
    update_frequency: int = 8
    environment_refresh_frequency: int = 300

    # NEW: Hub improvement tracking parameters
    hub_improvement_threshold: float = 0.005
    consecutive_failures_threshold: int = 50

    # FIXED: Better batch sizes for stability
    policy_batch_size: int = 24
    discriminator_batch_size: int = 12


@dataclass
class ImprovedOptimizationConfig:
    """Improved optimization settings for stable learning"""

    # FIXED: Reduced learning rates for stability
    policy_lr: float = 1e-5
    discriminator_lr: float = 5e-6

    # IMPROVED: Better optimizer settings
    optimizer_eps: float = 1e-5
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # FIXED: More conservative PPO parameters
    clip_epsilon: float = 0.15
    entropy_coefficient: float = 0.03
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    # NEW: Learning rate scheduling
    policy_lr_decay_step: int = 2000
    disc_lr_decay_step: int = 2000
    lr_decay_factor: float = 0.98

    # IMPROVED: Training epochs (reduced for stability)
    policy_epochs: int = 4
    discriminator_epochs: int = 4


@dataclass
class ImprovedDiscriminatorConfig:
    """Fixed discriminator configuration for proper F1 logging"""

    # FIXED: More lenient discriminator training
    early_stop_threshold: float = 0.55
    label_smoothing: float = 0.05
    gradient_penalty_lambda: float = 5.0

    # FIXED: Better discriminator update frequency
    update_frequency_multiplier: int = 50
    min_samples_per_class: int = 8
    max_samples_per_class: int = 16

    # NEW: F1 calculation settings (fixes the F1=0 issue)
    calculate_detailed_metrics: bool = True
    log_confusion_matrix: bool = True
    use_balanced_sampling: bool = True


@dataclass
class ImprovedEnvironmentConfig:
    """Improved environment configuration for better diversity"""

    # Graph pool management
    graph_pool_size: int = 200
    pool_refresh_episodes: int = 500

    # Environment diversity settings
    min_graph_size: int = 10
    max_graph_size: int = 50
    size_diversity_bins: int = 5

    # NEW: Performance monitoring
    track_hub_score_history: bool = True
    track_action_effectiveness: bool = True
    track_pattern_usage: bool = True


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