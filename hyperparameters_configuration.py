#!/usr/bin/env python3
"""
STRUCTURED Adversarial Training with Adversarial Reward Shaping
🆕 FEATURE: Potential-based reward shaping parameters
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ImprovedRewardConfig:
    """🆕 Enhanced reward structure with adversarial shaping"""
    # Traditional rewards
    REWARD_SUCCESS: float = 5.0
    REWARD_PARTIAL_SUCCESS: float = 2.0
    REWARD_FAILURE: float = -1.0
    REWARD_STEP: float = -0.02
    REWARD_HUB_REDUCTION: float = 10.0
    REWARD_INVALID: float = -0.5
    REWARD_SMALL_IMPROVEMENT: float = 0.5

    # 🆕 ADVERSARIAL REWARD SHAPING PARAMETERS
    adversarial_reward_enabled: bool = True
    adversarial_reward_scale: float = 5.0  # Scaling factor for adversarial rewards
    adversarial_threshold: float = 0.02  # Minimum probability change to matter
    potential_discount: float = 0.99  # γ for potential-based shaping

    # Adaptive adversarial scaling
    adversarial_scale_adaptive: bool = True
    adversarial_scale_min: float = 1.0
    adversarial_scale_max: float = 10.0
    adversarial_scale_success_boost: float = 1.1
    adversarial_scale_failure_decay: float = 0.95

    # Hub score thresholds
    HUB_SCORE_EXCELLENT: float = 0.05
    HUB_SCORE_GOOD: float = 0.3
    HUB_SCORE_ACCEPTABLE: float = 0.6

    # Progressive reward scaling
    improvement_multiplier_boost: float = 1.1
    improvement_multiplier_decay: float = 0.90
    improvement_multiplier_max: float = 2.0
    improvement_multiplier_min: float = 0.5


@dataclass
class StructuredTrainingConfig:
    """🆕 STRUCTURED training with explicit phases"""
    num_episodes: int = 3000
    num_envs: int = 16
    steps_per_env: int = 24
    max_steps_per_episode: int = 20

    # 🔥 WARM-UP PHASE
    warmup_episodes: int = 100  # Solo roll-out per i primi 100 episodi

    # 🔥 POLICY UPDATE SCHEDULE
    policy_update_frequency: int = 5  # Ogni 5 episodi
    policy_epochs_per_update: int = 4  # 4 epoche PPO
    policy_batch_size: int = 32

    # 🔥 DISCRIMINATOR UPDATE SCHEDULE
    discriminator_update_frequency: int = 10  # Ogni 10 policy updates (= 50 episodi)
    discriminator_epochs_per_update: int = 2  # 2 epoche per update
    discriminator_batch_size: int = 16

    # Environment settings
    environment_refresh_frequency: int = 300
    hub_improvement_threshold: float = 0.005
    consecutive_failures_threshold: int = 50


@dataclass
class ImprovedOptimizationConfig:
    """Conservative PPO settings for structured training"""
    optimizer_eps: float = 1e-5
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # Tighter PPO settings for more stable updates
    clip_epsilon: float = 0.15  # Slightly more permissive
    entropy_coefficient: float = 0.02  # Bit more exploration
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5

    # Note: epochs now controlled by StructuredTrainingConfig


@dataclass
class ImprovedDiscriminatorConfig:
    """🆕 Enhanced discriminator config with force update controls"""
    early_stop_threshold: float = 0.45  # More permissive
    label_smoothing: float = 0.1  # More smoothing for stability
    gradient_penalty_lambda: float = 1.0  # Reduced for gentler training

    min_samples_per_class: int = 12  # Slightly more data per update
    max_samples_per_class: int = 24
    calculate_detailed_metrics: bool = True
    log_confusion_matrix: bool = False  # Reduce log spam
    use_balanced_sampling: bool = True
    track_metric_history: bool = True
    metric_history_window: int = 20
    enable_metric_smoothing: bool = True
    smoothing_factor: float = 0.3

    # 🆕 FORCE UPDATE PARAMETERS
    force_update_enabled: bool = True
    force_update_accuracy_threshold: float = 0.58  # Force update if below this
    force_update_consecutive_episodes: int = 3  # Force after N consecutive low episodes
    force_update_max_epochs: int = 5  # Extra epochs for force update
    force_update_lr_boost: float = 1.5  # Temporarily boost LR for force update


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


@dataclass
class StructuredLearningRateConfig:
    """Learning rates optimized for structured training"""

    # Policy Learning Rate - Più stabile per update frequenti
    policy_initial_lr: float = 3e-4
    policy_min_lr: float = 1e-5
    policy_decay_factor: float = 0.95  # Gentle decay ogni tot policy updates
    policy_decay_frequency: int = 20  # Decay ogni 20 policy updates

    # Discriminator Learning Rate - Stabile e conservativo
    discriminator_lr: float = 2e-4  # Fisso - no scheduling complesso
    discriminator_min_lr: float = 1e-5
    discriminator_max_lr: float = 5e-4

    # 🆕 FORCE UPDATE LR PARAMETERS
    force_update_lr_multiplier: float = 2.0  # Boost LR during force updates
    force_update_lr_max: float = 1e-3  # Maximum LR during force update


@dataclass
class StructuredAdversarialConfig:
    """🆕 Enhanced adversarial config with force update controls"""

    # Buffer Management
    experience_buffer_size: int = 1000  # Mantieni ultimi 1000 roll-out examples
    discriminator_real_data_ratio: float = 0.5  # 50% real, 50% generated

    # Safety checks
    max_discriminator_accuracy: float = 0.92  # Pausa se troppo forte
    min_discriminator_accuracy: float = 0.58  # 🆕 Force update se troppo debole

    # 🆕 FORCE UPDATE CONTROLS
    enable_force_updates: bool = True
    force_update_consecutive_threshold: int = 3  # Force after 3 consecutive episodes below threshold
    force_update_recovery_threshold: float = 0.65  # Recovery target accuracy
    force_update_max_attempts: int = 3  # Max consecutive force attempts
    force_update_cooldown_episodes: int = 10  # Wait episodes after force update

    # Phase tracking
    track_phase_metrics: bool = True
    log_phase_transitions: bool = True

    # Emergency controls
    enable_discriminator_pause: bool = True
    discriminator_pause_episodes: int = 25  # Pausa per 25 episodi se troppo forte

    # 🆕 ADVERSARIAL REWARD MONITORING
    log_adversarial_rewards: bool = True
    adversarial_reward_log_frequency: int = 50  # Log stats every N episodes


def get_improved_config() -> Dict[str, Any]:
    return {
        'rewards': ImprovedRewardConfig(),
        'training': StructuredTrainingConfig(),
        'optimization': ImprovedOptimizationConfig(),
        'discriminator': ImprovedDiscriminatorConfig(),
        'environment': ImprovedEnvironmentConfig(),
        'learning_rate': StructuredLearningRateConfig(),
        'adversarial': StructuredAdversarialConfig()
    }


def print_config_summary():
    """🆕 Enhanced config summary with adversarial features"""
    print("🔧 STRUCTURED ADVERSARIAL TRAINING STRATEGY:")
    print("=" * 70)

    config = get_improved_config()
    training_config = config['training']
    lr_config = config['learning_rate']
    adv_config = config['adversarial']
    reward_config = config['rewards']

    print(f"🌅 1. WARM-UP PHASE (Episodes 0-{training_config.warmup_episodes}):")
    print("   - ONLY rollout collection")
    print("   - NO policy updates")
    print("   - NO discriminator updates")
    print("   - Build experience buffer")
    print("   - 🆕 Collect baseline discriminator probabilities")

    print(f"\n🎯 2. POLICY UPDATE PHASE (Every {training_config.policy_update_frequency} episodes):")
    print(f"   - {training_config.policy_epochs_per_update} epochs of PPO")
    print(f"   - Batch size: {training_config.policy_batch_size}")
    print(f"   - Learning rate: {lr_config.policy_initial_lr:.2e}")
    print(f"   - Tight clipping: {config['optimization'].clip_epsilon}")
    print("   - 🆕 Adversarial reward shaping active")

    print(f"\n🧠 3. DISCRIMINATOR UPDATE PHASE (Every {training_config.discriminator_update_frequency} policy updates):")
    print(
        f"   - Frequency: Every {training_config.policy_update_frequency * training_config.discriminator_update_frequency} episodes")
    print(f"   - {training_config.discriminator_epochs_per_update} epochs per update")
    print(
        f"   - Mix: {adv_config.discriminator_real_data_ratio:.0%} real + {1 - adv_config.discriminator_real_data_ratio:.0%} generated")
    print(f"   - Learning rate: {lr_config.discriminator_lr:.2e} (fixed)")
    print("   - 🆕 Force update if accuracy < 0.58")

    print(f"\n🎁 4. ADVERSARIAL REWARD SHAPING:")
    if reward_config.adversarial_reward_enabled:
        print(f"   - ✅ ENABLED with scale: {reward_config.adversarial_reward_scale}")
        print(f"   - Potential discount γ: {reward_config.potential_discount}")
        print(f"   - Threshold for significance: {reward_config.adversarial_threshold}")
        print("   - Potential-based: F(s,a,s') = γΦ(s') - Φ(s)")
        print("   - Where Φ(s) = -P(smelly|s)")
    else:
        print("   - ❌ DISABLED - only traditional rewards")

    print(f"\n⚖️  5. ADVERSARIAL BALANCE CONTROLS:")
    print(f"   - Experience buffer size: {adv_config.experience_buffer_size}")
    print(f"   - Discriminator pause if accuracy > {adv_config.max_discriminator_accuracy}")
    print(f"   - 🆕 Force update if accuracy < {adv_config.min_discriminator_accuracy}")
    print(f"   - Emergency pause duration: {adv_config.discriminator_pause_episodes} episodes")
    print(f"   - 🆕 Force update recovery target: {adv_config.force_update_recovery_threshold}")

    print("\n" + "=" * 70)
    print("🎯 EXPECTED BENEFITS:")
    print("✅ Stable warm-up prevents early instability")
    print("✅ Regular policy updates with tight control")
    print("✅ Discriminator stays aligned but not overpowering")
    print("✅ 🆕 Adversarial reward drives policy toward discriminator confusion")
    print("✅ 🆕 Force updates prevent discriminator collapse")
    print("✅ Clear phase separation prevents conflicts")
    print("✅ Experience buffer maintains training diversity")
    print("=" * 70)