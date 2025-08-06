#!/usr/bin/env python3
"""
Hub-Centric Hyperparameters for Stable RL Training
Sistema di reward semplificato completamente focalizzato su hub improvement
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HubCentricRewardConfig:
    """
    Sistema di reward completamente basato su hub improvement
    PRINCIPIO: Hub improvement è l'UNICO indicatore di successo
    """

    # === HUB IMPROVEMENT MULTIPLIERS ===
    # Questi sono gli UNICI parametri che contano
    SIGNIFICANT_IMPROVEMENT_MULTIPLIER: float = 10.0  # Per miglioramenti >0.05
    GOOD_IMPROVEMENT_MULTIPLIER: float = 5.0  # Per miglioramenti 0.01-0.05
    SMALL_IMPROVEMENT_MULTIPLIER: float = 2.0  # Per miglioramenti 0.005-0.01
    DEGRADATION_PENALTY_MULTIPLIER: float = -3.0  # Per peggioramenti <-0.01

    # === SOGLIE DI MIGLIORAMENTO ===
    SIGNIFICANT_THRESHOLD: float = 0.02  # Soglia per miglioramento significativo
    GOOD_THRESHOLD: float = 0.005  # Soglia per miglioramento buono
    SMALL_THRESHOLD: float = 0.001  # Soglia per miglioramento minimo
    NEUTRAL_THRESHOLD: float = -0.01  # Soglia per considerare neutrale

    # === UNICA PENALITÀ MANTENUTA ===
    STEP_PENALTY: float = -0.05  # Piccola penalità per ogni step


@dataclass
class ImprovedTrainingConfig:
    """Training config invariato (solo reward sono cambiati)"""
    num_episodes: int = 8000
    num_envs: int = 16
    steps_per_env: int = 24
    max_steps_per_episode: int = 20
    update_frequency: int = 8
    environment_refresh_frequency: int = 50
    hub_improvement_threshold: float = 0.005  # Threshold per success rate logging
    consecutive_failures_threshold: int = 50
    policy_batch_size: int = 24
    discriminator_batch_size: int = 12


@dataclass
class AdvancedLearningRateConfig:
    """Learning rate config invariato"""
    # === POLICY LEARNING RATE (CyclicLR) ===
    policy_base_lr: float = 5e-6
    policy_max_lr: float = 1e-4
    policy_cyclical_step_size: int = 150
    policy_cyclical_mode: str = 'triangular2'
    policy_gamma: float = 0.99994

    # === DISCRIMINATOR LEARNING RATE (ReduceLROnPlateau) ===
    discriminator_initial_lr: float = 2e-5
    discriminator_patience: int = 20
    discriminator_factor: float = 0.6
    discriminator_min_lr: float = 1e-7
    discriminator_threshold: float = 0.015
    discriminator_threshold_mode: str = 'abs'
    discriminator_cooldown: int = 10
    discriminator_eps: float = 1e-8
    discriminator_monitor_metric: str = 'f1'
    discriminator_monitor_mode: str = 'max'

    # === ADVANCED OPTIONS ===
    enable_warmup: bool = True
    warmup_episodes_policy: int = 100
    warmup_episodes_discriminator: int = 150
    warmup_factor: float = 0.1
    enable_cyclical_restart: bool = True
    restart_frequency_episodes: int = 1000
    enable_adaptive_lr: bool = True
    adaptive_factor_min: float = 0.5
    adaptive_factor_max: float = 1.5
    balance_threshold: float = 0.1


@dataclass
class ImprovedOptimizationConfig:
    """Optimization config invariato"""
    optimizer_eps: float = 1e-5
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.1
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    policy_epochs: int = 4
    discriminator_epochs: int = 4

    # Deprecated parameters (mantenuti per backward compatibility)
    policy_lr: float = 1e-5
    discriminator_lr: float = 5e-6
    policy_lr_decay_step: int = 2000
    disc_lr_decay_step: int = 2000
    lr_decay_factor: float = 0.98


@dataclass
class ImprovedDiscriminatorConfig:
    """Discriminator config invariato"""
    early_stop_threshold: float = 0.55
    label_smoothing: float = 0.05
    gradient_penalty_lambda: float = 5.0
    update_frequency_multiplier: int = 50
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
    """Environment config invariato"""
    graph_pool_size: int = 50
    pool_refresh_episodes: int = 25
    min_graph_size: int = 10
    max_graph_size: int = 50
    size_diversity_bins: int = 5
    track_hub_score_history: bool = True
    track_action_effectiveness: bool = True
    track_pattern_usage: bool = True


# === BACKWARD COMPATIBILITY FUNCTIONS ===
def get_improved_config() -> Dict[str, Any]:
    """
    Backward compatibility: returns hub-centric config
    DEPRECATED: Use get_hub_centric_config() instead
    """
    return get_hub_centric_config()


def get_hub_centric_config() -> Dict[str, Any]:
    """Get complete hub-centric configuration"""
    return {
        'rewards': HubCentricRewardConfig(),
        'training': ImprovedTrainingConfig(),
        'optimization': ImprovedOptimizationConfig(),
        'discriminator': ImprovedDiscriminatorConfig(),
        'environment': ImprovedEnvironmentConfig(),
        'learning_rate': AdvancedLearningRateConfig()
    }


def print_config_summary():
    """Print hub-centric configuration summary"""
    print("🎯 HUB-CENTRIC CONFIGURATION:")
    print("=" * 70)

    print("REWARD SYSTEM CHANGES:")
    print("   ❌ REMOVED: All complex reward components")
    print("   ✅ NEW: Simple hub-improvement based rewards")

    config = HubCentricRewardConfig()
    print(f"\n📊 HUB-CENTRIC REWARD FORMULA:")
    print(f"   Significant (>0.05): +{config.SIGNIFICANT_IMPROVEMENT_MULTIPLIER}x × improvement")
    print(f"   Good (0.01-0.05):    +{config.GOOD_IMPROVEMENT_MULTIPLIER}x × improvement")
    print(f"   Small (0.005-0.01):  +{config.SMALL_IMPROVEMENT_MULTIPLIER}x × improvement")
    print(f"   Neutral (-0.01-0.005): 0.0 reward")
    print(f"   Degradation (<-0.01): {config.DEGRADATION_PENALTY_MULTIPLIER}x × |improvement|")
    print(f"   Step penalty:         {config.STEP_PENALTY}")

    training_config = ImprovedTrainingConfig()
    print(f"\n📈 TRAINING PARAMETERS:")
    print(f"   Episodes: {training_config.num_episodes}")
    print(f"   Environments: {training_config.num_envs}")
    print(f"   Update frequency: {training_config.update_frequency}")
    print(f"   Hub improvement threshold: {training_config.hub_improvement_threshold}")

    lr_config = AdvancedLearningRateConfig()
    print(f"\n🧠 ADVANCED LEARNING RATES:")
    print("   POLICY (CyclicLR):")
    print(f"     Base LR: {lr_config.policy_base_lr:.2e}")
    print(f"     Max LR: {lr_config.policy_max_lr:.2e}")
    print(f"     Cycle: {lr_config.policy_cyclical_step_size * 2} episodes")

    print("   DISCRIMINATOR (ReduceLROnPlateau):")
    print(f"     Initial LR: {lr_config.discriminator_initial_lr:.2e}")
    print(f"     Monitor: {lr_config.discriminator_monitor_metric}")
    print(f"     Patience: {lr_config.discriminator_patience} episodes")

    print(f"\n📊 EXPECTED REWARD RANGES:")
    print(f"   Excellent episode (0.1 improvement): +{0.1 * config.SIGNIFICANT_IMPROVEMENT_MULTIPLIER:.1f}")
    print(f"   Good episode (0.03 improvement):     +{0.03 * config.GOOD_IMPROVEMENT_MULTIPLIER:.2f}")
    print(f"   Small progress (0.008 improvement):  +{0.008 * config.SMALL_IMPROVEMENT_MULTIPLIER:.3f}")
    print(f"   Bad episode (-0.02 improvement):     {-0.02 * abs(config.DEGRADATION_PENALTY_MULTIPLIER):.2f}")
    print(f"   Per-episode range: -0.5 to +2.0 (vs previous 14-18)")

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


# Funzione di utilità per calcolare reward hub-centric
def compute_hub_centric_reward(hub_improvement: float, config: HubCentricRewardConfig = None) -> float:
    """
    Calcola reward basato SOLO su hub improvement
    """
    if config is None:
        config = HubCentricRewardConfig()

    if hub_improvement > config.SIGNIFICANT_THRESHOLD:
        return config.SIGNIFICANT_IMPROVEMENT_MULTIPLIER * hub_improvement
    elif hub_improvement > config.GOOD_THRESHOLD:
        return config.GOOD_IMPROVEMENT_MULTIPLIER * hub_improvement
    elif hub_improvement > config.SMALL_THRESHOLD:
        return config.SMALL_IMPROVEMENT_MULTIPLIER * hub_improvement
    elif hub_improvement > config.NEUTRAL_THRESHOLD:
        return 0.0  # Neutrale
    else:
        return config.DEGRADATION_PENALTY_MULTIPLIER * abs(hub_improvement)


if __name__ == "__main__":
    print_config_summary()

    print(f"\n🧪 REWARD EXAMPLES:")
    print("=" * 40)

    # Test examples
    improvements = [0.15, 0.08, 0.03, 0.008, 0.002, -0.005, -0.02, -0.08]

    for improvement in improvements:
        reward = compute_hub_centric_reward(improvement)
        print(f"Hub improvement: {improvement:+.3f} → Reward: {reward:+.3f}")