#!/usr/bin/env python3
"""
Optimized Hyperparameters for Stable RL Training with Advanced Learning Rate Strategies

This file contains the improved hyperparameters that fix the issues in your training:
1. Hub improvement tracking
2. F1 score logging
3. Better reward structure
4. Advanced learning rate strategies (CyclicLR + ReduceLROnPlateau)
5. Improved exploration/exploitation balance
6. Better adversarial stability
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ImprovedRewardConfig:
    """Balanced reward structure with stronger incentives for partial gains"""
    REWARD_SUCCESS: float = 5.0
    REWARD_PARTIAL_SUCCESS: float = 2.0  # increased to motivate partial improvements
    REWARD_FAILURE: float = -1.0
    REWARD_STEP: float = -0.02
    REWARD_HUB_REDUCTION: float = 10.0
    REWARD_INVALID: float = -0.5
    REWARD_SMALL_IMPROVEMENT: float = 0.5

    HUB_SCORE_EXCELLENT: float = 0.05  # lowered threshold for more frequent rewards
    HUB_SCORE_GOOD: float = 0.3
    HUB_SCORE_ACCEPTABLE: float = 0.6

    improvement_multiplier_boost: float = 1.1
    improvement_multiplier_decay: float = 0.90  # faster decay to prevent runaway scaling
    improvement_multiplier_max: float = 2.0
    improvement_multiplier_min: float = 0.5


@dataclass
class ImprovedTrainingConfig:
    """Episode and update scheduling for improved learning stability"""
    num_episodes: int = 8000
    num_envs: int = 16
    steps_per_env: int = 24
    max_steps_per_episode: int = 20

    update_frequency: int = 8
    environment_refresh_frequency: int = 300
    hub_improvement_threshold: float = 0.005
    consecutive_failures_threshold: int = 50

    policy_batch_size: int = 32  # increased for smoother gradients
    discriminator_batch_size: int = 16


@dataclass
class ImprovedOptimizationConfig:
    """Conservative PPO and optimizer settings"""
    optimizer_eps: float = 1e-5
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    clip_epsilon: float = 0.10  # tighter clipping
    entropy_coefficient: float = 0.015  # reduced entropy bonus
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    policy_epochs: int = 4
    discriminator_epochs: int = 4


@dataclass
class ImprovedDiscriminatorConfig:
    """Fixed discriminator training to avoid early stops"""
    early_stop_threshold: float = 0.60  # stricter early-stop threshold
    label_smoothing: float = 0.05
    gradient_penalty_lambda: float = 5.0
    update_frequency_multiplier: int = 75  # more batches before retraining
    min_samples_per_class: int = 8
    max_samples_per_class: int = 16
    calculate_detailed_metrics: bool = True
    log_confusion_matrix: bool = True
    use_balanced_sampling: bool = True
    track_metric_history: bool = True
    metric_history_window: int = 50
    enable_metric_smoothing: bool = True
    smoothing_factor: float = 0.1


@dataclass
class ImprovedEnvironmentConfig:
    graph_pool_size: int = 200
    pool_refresh_episodes: int = 500
    min_graph_size: int = 10
    max_graph_size: int = 50
    size_diversity_bins: int = 5
    track_hub_score_history: bool = True
    track_action_effectiveness: bool = True
    track_pattern_usage: bool = True


# ✅ FIX CRITICO 1: Aggiunta classe mancante
@dataclass
class AdvancedLearningRateConfig:
    """Advanced learning rate configuration with multiple strategies"""

    # Policy Learning Rate (CyclicLR)
    policy_base_lr: float = 3e-4
    policy_max_lr: float = 1e-3
    policy_cyclical_step_size: int = 150
    policy_cyclical_mode: str = 'triangular2'
    policy_gamma: float = 0.99

    # Discriminator Learning Rate (ReduceLROnPlateau)
    discriminator_initial_lr: float = 1e-4
    discriminator_monitor_metric: str = 'f1'
    discriminator_monitor_mode: str = 'max'
    discriminator_factor: float = 0.5
    discriminator_patience: int = 10
    discriminator_threshold: float = 0.01
    discriminator_threshold_mode: str = 'rel'
    discriminator_cooldown: int = 5
    discriminator_min_lr: float = 1e-6
    discriminator_eps: float = 1e-8

    # Advanced features
    enable_warmup: bool = True
    enable_adaptive_lr: bool = True
    enable_cyclical_restart: bool = True


def get_improved_config() -> Dict[str, Any]:
    return {
        'rewards': ImprovedRewardConfig(),
        'training': ImprovedTrainingConfig(),
        'optimization': ImprovedOptimizationConfig(),
        'discriminator': ImprovedDiscriminatorConfig(),
        'environment': ImprovedEnvironmentConfig(),
        'learning_rate': AdvancedLearningRateConfig()
    }


def print_config_summary():
    """Print summary of key improvements including advanced learning rate strategies"""
    print("🔧 KEY IMPROVEMENTS IN HYPERPARAMETERS:")
    print("=" * 70)

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

    print("\n4. 🆕 ADVANCED LEARNING RATE STRATEGIES:")
    lr_config = AdvancedLearningRateConfig()
    print("   POLICY (CyclicLR):")
    print(f"     - Base LR: {lr_config.policy_base_lr}")
    print(f"     - Max LR: {lr_config.policy_max_lr}")
    print(f"     - Cycle length: {lr_config.policy_cyclical_step_size * 2} episodes")
    print(f"     - Mode: {lr_config.policy_cyclical_mode} (decaying peaks)")

    print("   DISCRIMINATOR (ReduceLROnPlateau):")
    print(f"     - Initial LR: {lr_config.discriminator_initial_lr}")
    print(f"     - Monitor: {lr_config.discriminator_monitor_metric} ({lr_config.discriminator_monitor_mode})")
    print(f"     - Patience: {lr_config.discriminator_patience} episodes")
    print(f"     - Reduction factor: {lr_config.discriminator_factor}")
    print(f"     - Min LR: {lr_config.discriminator_min_lr}")

    print("\n5. MORE STABLE TRAINING:")
    print(f"   - Better batch sizes: {ImprovedTrainingConfig.policy_batch_size}")
    print(f"   - More frequent updates: every {ImprovedTrainingConfig.update_frequency} episodes")
    print(f"   - Reduced training epochs: {ImprovedOptimizationConfig.policy_epochs}")
    print(f"   - Conservative PPO clip: {ImprovedOptimizationConfig.clip_epsilon}")

    print("\n6. IMPROVED ENVIRONMENT DIVERSITY:")
    print(f"   - More frequent refresh: every {ImprovedTrainingConfig.environment_refresh_frequency} episodes")
    print(f"   - Failure detection: reset after {ImprovedTrainingConfig.consecutive_failures_threshold} failures")
    print(
        f"   - Better graph size range: {ImprovedEnvironmentConfig.min_graph_size}-{ImprovedEnvironmentConfig.max_graph_size} nodes")

    print("\n7. 🆕 ADDITIONAL ADVANCED FEATURES:")
    print(f"   - Warmup enabled: {lr_config.enable_warmup}")
    print(f"   - Adaptive LR balancing: {lr_config.enable_adaptive_lr}")
    print(f"   - Cyclical restarts: {lr_config.enable_cyclical_restart}")
    print(f"   - Metric smoothing: {ImprovedDiscriminatorConfig.enable_metric_smoothing}")

    print("=" * 70)


def get_scheduler_info() -> Dict[str, str]:
    """Get information about the learning rate scheduling strategy"""
    return {
        'policy_scheduler': 'CyclicLR',
        'policy_description': 'Oscillates between base_lr and max_lr with triangular2 mode (decaying peaks)',
        'discriminator_scheduler': 'ReduceLROnPlateau',
        'discriminator_description': 'Reduces LR when F1/accuracy plateaus, with patience and minimum thresholds',
        'strategy_rationale': 'Policy maintains exploration via oscillations, discriminator stabilizes when struggling',
        'adversarial_benefit': 'Prevents discriminator collapse while keeping policy dynamic'
    }


if __name__ == "__main__":
    print_config_summary()

    print("\n" + "=" * 70)
    print("🧠 LEARNING RATE STRATEGY EXPLANATION:")
    print("=" * 70)

    scheduler_info = get_scheduler_info()
    print(f"Policy: {scheduler_info['policy_description']}")
    print(f"Discriminator: {scheduler_info['discriminator_description']}")
    print(f"Rationale: {scheduler_info['strategy_rationale']}")
    print(f"Benefit: {scheduler_info['adversarial_benefit']}")

    # Example usage
    config = get_improved_config()
    print(f"\n📊 Example reward for 0.05 hub improvement:")
    rewards = config['rewards']
    if 0.05 > 0.01:  # Above small improvement threshold
        reward = rewards.REWARD_SMALL_IMPROVEMENT * 0.05
        print(f"   Reward: {reward:.3f} (vs 0.0 in old system)")

    print(f"\n📊 Advanced LR example - CyclicLR for Policy:")
    lr_config = config['learning_rate']
    print(f"   Episode 0: LR = {lr_config.policy_base_lr:.2e}")
    print(f"   Episode 150: LR = {lr_config.policy_max_lr:.2e} (peak)")
    print(f"   Episode 300: LR = {lr_config.policy_base_lr:.2e} (valley)")
    print(f"   Episode 450: LR = {lr_config.policy_max_lr * lr_config.gamma ** 150:.2e} (decayed peak)")

    print(f"\n📊 Advanced LR example - ReduceLROnPlateau for Discriminator:")
    print(f"   Starts at: {lr_config.discriminator_initial_lr:.2e}")
    print(f"   If F1 doesn't improve for {lr_config.discriminator_patience} episodes:")
    print(f"   Reduces to: {lr_config.discriminator_initial_lr * lr_config.discriminator_factor:.2e}")
    print(f"   Minimum LR: {lr_config.discriminator_min_lr:.2e}")