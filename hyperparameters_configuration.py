#!/usr/bin/env python3
"""
STRUCTURED Adversarial Training with Warm-up and Phased Updates

Strategia:
1. WARM-UP (episodi 0-N): Solo roll-out, no updates
2. POLICY UPDATES: Ogni 5 episodi, 4 epoche PPO
3. DISCRIMINATOR UPDATES: Ogni 10 policy updates (50 episodi), 2 epoche
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ImprovedRewardConfig:
    """Balanced reward structure"""
    REWARD_SUCCESS: float = 5.0
    REWARD_PARTIAL_SUCCESS: float = 2.0
    REWARD_FAILURE: float = -1.0
    REWARD_STEP: float = -0.02
    REWARD_HUB_REDUCTION: float = 10.0
    REWARD_INVALID: float = -0.5
    REWARD_SMALL_IMPROVEMENT: float = 0.5

    HUB_SCORE_EXCELLENT: float = 0.05
    HUB_SCORE_GOOD: float = 0.3
    HUB_SCORE_ACCEPTABLE: float = 0.6

    improvement_multiplier_boost: float = 1.1
    improvement_multiplier_decay: float = 0.90
    improvement_multiplier_max: float = 2.0
    improvement_multiplier_min: float = 0.5


@dataclass
class StructuredTrainingConfig:
    """🆕 STRUCTURED training with explicit phases"""
    num_episodes: int = 8000
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
    """Discriminator config optimized for structured updates"""
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


@dataclass
class StructuredAdversarialConfig:
    """🆕 Configurazione per adversarial training strutturato"""

    # Buffer Management
    experience_buffer_size: int = 1000  # Mantieni ultimi 1000 roll-out examples
    discriminator_real_data_ratio: float = 0.5  # 50% real, 50% generated

    # Safety checks
    max_discriminator_accuracy: float = 0.92  # Pausa se troppo forte
    min_discriminator_accuracy: float = 0.58  # Force update se troppo debole

    # Phase tracking
    track_phase_metrics: bool = True
    log_phase_transitions: bool = True

    # Emergency controls
    enable_discriminator_pause: bool = True
    discriminator_pause_episodes: int = 25  # Pausa per 25 episodi se troppo forte


@dataclass
class ExperienceBuffer:
    """🆕 Buffer per mantenere esempi di roll-out"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = []
        self.pointer = 0

    def add_rollout_data(self, states, rewards, hub_scores):
        """Aggiungi dati di roll-out al buffer"""
        rollout_data = {
            'states': states,
            'rewards': rewards,
            'hub_scores': hub_scores,
            'episode': len(self.buffer)
        }

        if len(self.buffer) < self.max_size:
            self.buffer.append(rollout_data)
        else:
            self.buffer[self.pointer] = rollout_data
            self.pointer = (self.pointer + 1) % self.max_size

    def get_recent_states(self, num_samples: int = 50):
        """Ottieni stati recenti per discriminatore"""
        all_states = []
        for rollout in self.buffer[-10:]:  # Ultimi 10 roll-out
            all_states.extend(rollout['states'])

        if len(all_states) >= num_samples:
            import random
            return random.sample(all_states, num_samples)
        return all_states

    def get_buffer_stats(self):
        """Statistiche del buffer"""
        return {
            'size': len(self.buffer),
            'total_states': sum(len(r['states']) for r in self.buffer),
            'avg_reward': sum(sum(r['rewards']) for r in self.buffer) / max(len(self.buffer), 1),
            'latest_episode': self.buffer[-1]['episode'] if self.buffer else 0
        }


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
    """Print summary of structured training approach"""
    print("🔧 STRUCTURED ADVERSARIAL TRAINING STRATEGY:")
    print("=" * 70)

    config = get_improved_config()
    training_config = config['training']
    lr_config = config['learning_rate']
    adv_config = config['adversarial']

    print(f"🌅 1. WARM-UP PHASE (Episodes 0-{training_config.warmup_episodes}):")
    print("   - ONLY rollout collection")
    print("   - NO policy updates")
    print("   - NO discriminator updates")
    print("   - Build experience buffer")
    print("   - Let discriminator see initial 'real' examples")

    print(f"\n🎯 2. POLICY UPDATE PHASE (Every {training_config.policy_update_frequency} episodes):")
    print(f"   - {training_config.policy_epochs_per_update} epochs of PPO")
    print(f"   - Batch size: {training_config.policy_batch_size}")
    print(f"   - Learning rate: {lr_config.policy_initial_lr:.2e}")
    print(f"   - Tight clipping: {config['optimization'].clip_epsilon}")
    print("   - Small, stable steps")

    print(f"\n🧠 3. DISCRIMINATOR UPDATE PHASE (Every {training_config.discriminator_update_frequency} policy updates):")
    print(
        f"   - Frequency: Every {training_config.policy_update_frequency * training_config.discriminator_update_frequency} episodes")
    print(f"   - {training_config.discriminator_epochs_per_update} epochs per update")
    print(
        f"   - Mix: {adv_config.discriminator_real_data_ratio:.0%} real + {1 - adv_config.discriminator_real_data_ratio:.0%} generated")
    print(f"   - Learning rate: {lr_config.discriminator_lr:.2e} (fixed)")
    print("   - Stays aligned but doesn't become too strong")

    print(f"\n📊 4. TIMELINE EXAMPLE (first 200 episodes):")
    warmup = training_config.warmup_episodes
    policy_freq = training_config.policy_update_frequency
    disc_freq = training_config.discriminator_update_frequency

    print(f"   Episodes 0-{warmup}: Warm-up only")
    print(f"   Episode {warmup + policy_freq}: First policy update")
    print(f"   Episode {warmup + 2 * policy_freq}: Second policy update")
    print(f"   Episode {warmup + policy_freq * disc_freq}: First discriminator update")
    print(f"   Episode {warmup + policy_freq * disc_freq * 2}: Second discriminator update")

    print(f"\n⚖️  5. ADVERSARIAL BALANCE CONTROLS:")
    print(f"   - Experience buffer size: {adv_config.experience_buffer_size}")
    print(f"   - Discriminator pause if accuracy > {adv_config.max_discriminator_accuracy}")
    print(f"   - Force update if accuracy < {adv_config.min_discriminator_accuracy}")
    print(f"   - Emergency pause duration: {adv_config.discriminator_pause_episodes} episodes")

    print("\n" + "=" * 70)
    print("🎯 EXPECTED BENEFITS:")
    print("✅ Stable warm-up prevents early instability")
    print("✅ Regular policy updates with tight control")
    print("✅ Discriminator stays aligned but not overpowering")
    print("✅ Clear phase separation prevents conflicts")
    print("✅ Experience buffer maintains training diversity")
    print("=" * 70)


def get_training_schedule_info(episode: int) -> Dict[str, Any]:
    """Get training schedule information for current episode"""
    config = get_improved_config()
    training_config = config['training']

    warmup = training_config.warmup_episodes
    policy_freq = training_config.policy_update_frequency
    disc_freq = training_config.discriminator_update_frequency

    if episode < warmup:
        phase = "warmup"
        next_policy_update = warmup + policy_freq
        next_disc_update = warmup + (policy_freq * disc_freq)
    else:
        episodes_since_warmup = episode - warmup
        policy_updates_so_far = episodes_since_warmup // policy_freq

        phase = "training"
        next_policy_update = warmup + ((policy_updates_so_far + 1) * policy_freq)
        next_disc_update = warmup + (((policy_updates_so_far // disc_freq) + 1) * policy_freq * disc_freq)

    return {
        'phase': phase,
        'policy_updates_completed': max(0, (episode - warmup) // policy_freq) if episode >= warmup else 0,
        'discriminator_updates_completed': max(0, (episode - warmup) // (
                    policy_freq * disc_freq)) if episode >= warmup else 0,
        'next_policy_update_episode': next_policy_update,
        'next_discriminator_update_episode': next_disc_update,
        'episodes_until_next_policy': max(0, next_policy_update - episode),
        'episodes_until_next_discriminator': max(0, next_disc_update - episode)
    }


if __name__ == "__main__":
    print_config_summary()

    print("\n" + "=" * 70)
    print("📅 TRAINING SCHEDULE EXAMPLES:")
    print("=" * 70)

    # Show schedule for key episodes
    key_episodes = [0, 50, 100, 105, 110, 155, 200, 255]

    for ep in key_episodes:
        schedule = get_training_schedule_info(ep)
        print(f"Episode {ep:3d}: {schedule['phase']:>8} | "
              f"Policy: {schedule['policy_updates_completed']:2d} | "
              f"Disc: {schedule['discriminator_updates_completed']:2d} | "
              f"Next P: {schedule['episodes_until_next_policy']:2d} | "
              f"Next D: {schedule['episodes_until_next_discriminator']:2d}")

    print("\nLegend: P=Policy updates, D=Discriminator updates, Next=episodes until next update")