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
class AdvancedLearningRateConfig:
    """NEW: Advanced learning rate strategies for better adversarial stability"""

    # === POLICY LEARNING RATE (CyclicLR) ===
    policy_base_lr: float = 1e-6
    policy_max_lr: float = 3e-5
    policy_cyclical_step_size: int = 150  # Episodes per half cycle
    policy_cyclical_mode: str = 'triangular2'  # Decaying max_lr over time
    policy_gamma: float = 0.99994  # Decay factor for triangular2 mode

    # === DISCRIMINATOR LEARNING RATE (ReduceLROnPlateau) ===
    discriminator_initial_lr: float = 2e-5
    discriminator_patience: int = 20  # Episodes to wait before reducing
    discriminator_factor: float = 0.6  # Reduction factor
    discriminator_min_lr: float = 1e-7
    discriminator_threshold: float = 0.015  # Minimum change to qualify as improvement
    discriminator_threshold_mode: str = 'abs'  # Absolute change
    discriminator_cooldown: int = 10  # Episodes to wait after reduction
    discriminator_eps: float = 1e-8  # Minimum decay

    # === MONITORING METRICS FOR LR SCHEDULING ===
    # For ReduceLROnPlateau - what metric to monitor for discriminator
    discriminator_monitor_metric: str = 'f1'  # Options: 'accuracy', 'f1', 'loss'
    discriminator_monitor_mode: str = 'max'  # 'max' for accuracy/f1, 'min' for loss

    # === ADVANCED OPTIONS ===
    # Enable learning rate warmup for both networks
    enable_warmup: bool = True
    warmup_episodes_policy: int = 100
    warmup_episodes_discriminator: int = 150
    warmup_factor: float = 0.1  # Start at 10% of base LR

    # Enable learning rate restart for CyclicLR
    enable_cyclical_restart: bool = True
    restart_frequency_episodes: int = 1000  # Reset cycle every N episodes

    # Adaptive learning rate based on adversarial balance
    enable_adaptive_lr: bool = True
    adaptive_factor_min: float = 0.5
    adaptive_factor_max: float = 1.5
    balance_threshold: float = 0.1  # When discriminator vs policy balance is off


@dataclass
class ImprovedOptimizationConfig:
    """Improved optimization settings for stable learning - integrated with advanced LR"""

    # BASE OPTIMIZER SETTINGS (the LR will be managed by schedulers)
    optimizer_eps: float = 1e-5
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # PPO PARAMETERS - more conservative for stability
    clip_epsilon: float = 0.15
    entropy_coefficient: float = 0.03
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    # TRAINING EPOCHS - reduced for stability
    policy_epochs: int = 4
    discriminator_epochs: int = 4

    # === DEPRECATED: Old LR parameters (kept for backward compatibility) ===
    # These will be overridden by AdvancedLearningRateConfig
    policy_lr: float = 1e-5  # Will be ignored if using CyclicLR
    discriminator_lr: float = 5e-6  # Will be ignored if using ReduceLROnPlateau
    policy_lr_decay_step: int = 2000  # Will be ignored
    disc_lr_decay_step: int = 2000  # Will be ignored
    lr_decay_factor: float = 0.98  # Will be ignored


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

    # NEW: Enhanced metrics for ReduceLROnPlateau
    track_metric_history: bool = True
    metric_history_window: int = 50  # Episodes to keep in history
    enable_metric_smoothing: bool = True
    smoothing_factor: float = 0.1  # EMA smoothing for noisy metrics


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
    """Get complete improved configuration dictionary with advanced learning rates"""
    return {
        'rewards': ImprovedRewardConfig(),
        'training': ImprovedTrainingConfig(),
        'optimization': ImprovedOptimizationConfig(),
        'discriminator': ImprovedDiscriminatorConfig(),
        'environment': ImprovedEnvironmentConfig(),
        'learning_rate': AdvancedLearningRateConfig()  # NEW: Advanced LR configuration
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