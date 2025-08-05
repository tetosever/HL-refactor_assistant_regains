#!/usr/bin/env python3
"""
TrainingLogger - Sistema di logging ottimizzato per training RL
Nuovo script separato che gestisce tutto il logging in modo intelligente
"""

import logging
import time
import warnings
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np


class TrainingLogger:
    """Logger ottimizzato per il training RL con focus sui risultati chiave"""

    def __init__(self, log_level: int = logging.INFO, log_file: Optional[Path] = None):
        # Setup del logger principale
        self.logger = logging.getLogger("RL_Training")
        self.logger.setLevel(log_level)

        # Rimuovi handler esistenti per evitare duplicati
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Console handler con formato pulito
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formato semplificato e chiaro
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler opzionale
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # File sempre dettagliato
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Metriche per tracking
        self.training_start_time = time.time()
        self.last_significant_log = 0
        self.best_metrics = {
            'reward': float('-inf'),
            'success_rate': 0.0,
            'hub_improvement': 0.0
        }

        # Contatori per log intelligenti
        self.episode_log_count = 0
        self.last_milestone = 0

    def start_training(self, config: Dict[str, Any]):
        """Log di inizio training con configurazione chiave"""
        self.logger.info("🚀 STARTING RL TRAINING")
        self.logger.info("=" * 50)

        # Solo parametri più importanti
        training = config.get('training', {})
        rewards = config.get('rewards', {})
        opt = config.get('optimization', {})

        self.logger.info(f"📊 TRAINING CONFIG:")
        self.logger.info(f"   Episodes: {getattr(training, 'num_episodes', 'N/A')}")
        self.logger.info(f"   Environments: {getattr(training, 'num_envs', 'N/A')}")
        self.logger.info(f"   Policy LR: {getattr(opt, 'policy_lr', 'N/A')}")
        self.logger.info(f"   Hub threshold: {getattr(training, 'hub_improvement_threshold', 'N/A')}")

        self.logger.info(f"🎯 REWARD STRUCTURE:")
        self.logger.info(f"   Success: +{getattr(rewards, 'REWARD_SUCCESS', 'N/A')}")
        self.logger.info(f"   Small improvement: +{getattr(rewards, 'REWARD_SMALL_IMPROVEMENT', 'N/A')}")
        self.logger.info(f"   Hub reduction: +{getattr(rewards, 'REWARD_HUB_REDUCTION', 'N/A')}")

        self.logger.info("=" * 50)

    def log_discriminator_validation(self, results: Dict[str, float]):
        """Log validazione discriminatore - solo se significativo"""
        accuracy = results.get('accuracy', 0)
        f1 = results.get('f1', 0)
        success_rate = results.get('success_rate', 0)

        if accuracy > 0.75:
            status = "✅ EXCELLENT"
        elif accuracy > 0.65:
            status = "✅ GOOD"
        elif accuracy > 0.5:
            status = "⚠️  FAIR"
        else:
            status = "❌ POOR"

        self.logger.info(f"🔍 DISCRIMINATOR VALIDATION {status}")
        self.logger.info(f"   Accuracy: {accuracy:.3f} | F1: {f1:.3f} | Success Rate: {success_rate:.3f}")

        if accuracy < 0.6:
            self.logger.warning("   ⚠️  Low discriminator performance may affect training")

    def should_log_episode(self, episode: int, improved: bool = False) -> bool:
        """Determina se loggare questo episodio (logica intelligente)"""
        # Sempre per miglioramenti significativi
        if improved:
            return True

        # Log frequenti all'inizio
        if episode < 50:
            return episode % 10 == 0

        # Log moderati per primi 500 episodi
        if episode < 500:
            return episode % 25 == 0

        # Log meno frequenti dopo
        return episode % 50 == 0

    def log_episode_progress(self, episode: int, metrics: Dict[str, float],
                             force_log: bool = False):
        """Log progresso episodio - solo se significativo"""

        reward = metrics.get('reward', 0)
        success_rate = metrics.get('success_rate', 0)
        hub_improvement = metrics.get('hub_improvement', 0)
        hub_score = metrics.get('average_hub_score', 0.5)
        consecutive_failures = metrics.get('consecutive_failures', 0)

        # Determina se c'è stato un miglioramento significativo
        improved = (
                reward > self.best_metrics['reward'] + 0.5 or
                success_rate > self.best_metrics['success_rate'] + 0.05 or
                hub_improvement > self.best_metrics['hub_improvement'] + 0.01
        )

        # Update best metrics
        if reward > self.best_metrics['reward']:
            self.best_metrics['reward'] = reward
        if success_rate > self.best_metrics['success_rate']:
            self.best_metrics['success_rate'] = success_rate
        if hub_improvement > self.best_metrics['hub_improvement']:
            self.best_metrics['hub_improvement'] = hub_improvement

        # Decidi se loggare
        should_log = force_log or self.should_log_episode(episode, improved)

        if not should_log:
            return

        # Formato compatto per episode progress
        status_emoji = "🎯" if improved else "📊"
        if consecutive_failures > 30:
            status_emoji = "⚠️ "
        elif success_rate > 0.15:
            status_emoji = "✅"
        elif success_rate > 0.05:
            status_emoji = "🔄"

        self.logger.info(
            f"{status_emoji} Episode {episode:4d} | "
            f"Reward: {reward:6.2f} | "
            f"Success: {success_rate:5.1%} | "
            f"Hub↓: {hub_improvement:+.3f} | "
            f"Score: {hub_score:.3f}"
        )

        # Warning per problemi significativi
        if consecutive_failures > 50:
            self.logger.warning(f"   🚨 {consecutive_failures} consecutive failures")

        if success_rate == 0 and episode > 200:
            self.logger.warning("   ⚠️  Zero success rate persisting")

        self.episode_log_count += 1

    def log_milestone_summary(self, episode: int, metrics_history: Dict[str, List[float]]):
        """Log riassunto milestone ogni 100 episodi"""
        if episode % 100 != 0 or episode == 0:
            return

        if episode == self.last_milestone:
            return
        self.last_milestone = episode

        self.logger.info("=" * 60)
        self.logger.info(f"📊 MILESTONE {episode:4d} - Last 100 Episodes Summary")
        self.logger.info("-" * 60)

        # Calcola statistiche degli ultimi 100 episodi
        for metric_name, values in metrics_history.items():
            if not values:
                continue

            recent_100 = values[-100:] if len(values) >= 100 else values
            if not recent_100:
                continue

            avg = np.mean(recent_100)
            best = max(recent_100) if metric_name != 'hub_scores' else min(recent_100)

            if metric_name == 'episode_rewards':
                self.logger.info(f"🎯 Reward        | Avg: {avg:6.2f} | Best: {best:6.2f}")
            elif metric_name == 'success_rates':
                self.logger.info(f"✅ Success Rate  | Avg: {avg:5.1%} | Best: {best:5.1%}")
            elif metric_name == 'hub_improvements':
                self.logger.info(f"📉 Hub Improve   | Avg: {avg:+.3f} | Best: {max(recent_100):+.3f}")
            elif metric_name == 'hub_scores':
                self.logger.info(f"🎲 Hub Score     | Avg: {avg:.3f} | Best: {best:.3f}")

        # Calcola tempo trascorso e velocità
        elapsed = time.time() - self.training_start_time
        eps_per_hour = episode / (elapsed / 3600) if elapsed > 0 else 0

        self.logger.info(f"⏱️  Time Elapsed  | {elapsed / 60:.1f}m | {eps_per_hour:.1f} eps/hour")

        # Trend analysis
        if len(metrics_history.get('success_rates', [])) >= 50:
            recent_50 = metrics_history['success_rates'][-50:]
            earlier_50 = metrics_history['success_rates'][-100:-50] if len(
                metrics_history['success_rates']) >= 100 else []

            if earlier_50:
                recent_avg = np.mean(recent_50)
                earlier_avg = np.mean(earlier_50)
                trend = "📈 Improving" if recent_avg > earlier_avg else "📉 Declining" if recent_avg < earlier_avg else "➡️  Stable"
                self.logger.info(f"📈 Trend         | {trend} ({recent_avg:.1%} vs {earlier_avg:.1%})")

        self.logger.info("=" * 60)

    def log_training_update(self, update_count: int, metrics: Dict[str, float]):
        """Log aggiornamenti training - solo se significativi"""
        # Log solo ogni 5 update per ridurre noise
        if update_count % 5 != 0:
            return

        policy_loss = metrics.get('policy_loss', 0)
        disc_f1 = metrics.get('discriminator_f1', 0)
        disc_acc = metrics.get('discriminator_accuracy', 0)
        policy_entropy = metrics.get('policy_entropy', 0)

        # Log solo se abbiamo metriche significative
        if policy_loss > 0 or disc_f1 > 0:
            status = "📈" if policy_loss > 0 else "📉"

            self.logger.info(
                f"{status} Update {update_count:3d} | "
                f"Policy Loss: {policy_loss:.4f} | "
                f"Entropy: {policy_entropy:.3f} | "
                f"Disc F1: {disc_f1:.3f}"
            )

    def log_checkpoint_save(self, episode: int, path: Path, performance: Dict[str, float]):
        """Log salvataggio checkpoint con performance"""
        best_reward = performance.get('best_reward', 0)
        current_success = performance.get('current_success_rate', 0)

        self.logger.info(f"💾 Checkpoint saved at episode {episode}")
        self.logger.info(f"   File: {path.name} | Best Reward: {best_reward:.2f} | Success: {current_success:.1%}")

    def log_environment_refresh(self, episode: int, stats: Dict[str, Any]):
        """Log refresh environment - informazioni essenziali"""
        pool_size = stats.get('pool_size', 0)
        graph_sizes = stats.get('graph_sizes', [])
        avg_size = np.mean(graph_sizes) if graph_sizes else 0

        self.logger.info(f"🔄 Environment refresh at episode {episode}")
        self.logger.info(f"   Pool: {pool_size} graphs | Avg size: {avg_size:.1f} nodes")

    def log_final_summary(self, total_episodes: int, final_metrics: Dict[str, Any]):
        """Log riassunto finale completo"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🏁 TRAINING COMPLETED - FINAL SUMMARY")
        self.logger.info("=" * 70)

        # Metriche finali
        performance = final_metrics.get('final_performance', {})

        self.logger.info(f"📊 FINAL PERFORMANCE (Last 100 episodes):")
        self.logger.info(f"   Average Reward: {performance.get('mean_reward_last_100', 0):.3f}")
        self.logger.info(f"   Success Rate: {performance.get('mean_success_rate_last_100', 0):.1%}")
        self.logger.info(f"   Hub Improvement: {performance.get('mean_hub_improvement_last_100', 0):+.3f}")
        self.logger.info(f"   Average Hub Score: {performance.get('mean_hub_score_last_100', 0.5):.3f}")

        # Best achieved
        self.logger.info(f"🏆 BEST ACHIEVED:")
        self.logger.info(f"   Best Reward: {self.best_metrics['reward']:.3f}")
        self.logger.info(f"   Best Success Rate: {self.best_metrics['success_rate']:.1%}")
        self.logger.info(f"   Best Hub Improvement: {self.best_metrics['hub_improvement']:+.3f}")

        # Training stats
        total_time = time.time() - self.training_start_time
        total_updates = final_metrics.get('total_updates', 0)

        self.logger.info(f"⏱️  TRAINING STATS:")
        self.logger.info(f"   Total Episodes: {total_episodes}")
        self.logger.info(f"   Total Time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)")
        self.logger.info(f"   Policy Updates: {total_updates}")
        self.logger.info(f"   Episodes/Hour: {total_episodes / (total_time / 3600):.1f}")
        self.logger.info(f"   Episode logs generated: {self.episode_log_count}")

        # Analisi qualitativa dei risultati
        self._log_training_analysis(performance)

        self.logger.info("=" * 70)

    def _log_training_analysis(self, performance: Dict[str, Any]):
        """Analisi qualitativa dei risultati del training"""
        self.logger.info(f"🔍 TRAINING ANALYSIS:")

        final_success = performance.get('mean_success_rate_last_100', 0)
        final_improvement = performance.get('mean_hub_improvement_last_100', 0)
        final_reward = performance.get('mean_reward_last_100', 0)

        # Success rate analysis
        if final_success > 0.15:
            self.logger.info("   ✅ Excellent success rate achieved")
        elif final_success > 0.08:
            self.logger.info("   ✅ Good success rate achieved")
        elif final_success > 0.03:
            self.logger.info("   ⚠️  Moderate success rate - consider longer training")
        else:
            self.logger.info("   ❌ Low success rate - check discriminator/thresholds")

        # Hub improvement analysis
        if final_improvement > 0.02:
            self.logger.info("   ✅ Strong hub score improvements")
        elif final_improvement > 0.005:
            self.logger.info("   ✅ Consistent hub score improvements")
        elif final_improvement > 0:
            self.logger.info("   ⚠️  Small improvements - may need longer training")
        else:
            self.logger.info("   ❌ No consistent improvements - check reward structure")

        # Overall performance
        if final_reward > 2.0:
            self.logger.info("   🎉 Outstanding overall performance")
        elif final_reward > 0.5:
            self.logger.info("   ✅ Good overall performance")
        elif final_reward > -1.0:
            self.logger.info("   ⚠️  Fair performance - room for improvement")
        else:
            self.logger.info("   ❌ Poor performance - review hyperparameters")

    def log_error(self, error_msg: str, context: str = ""):
        """Log errori significativi"""
        if context:
            self.logger.error(f"❌ {context}: {error_msg}")
        else:
            self.logger.error(f"❌ {error_msg}")

    def log_warning(self, warning_msg: str, context: str = ""):
        """Log warning importanti"""
        if context:
            self.logger.warning(f"⚠️  {context}: {warning_msg}")
        else:
            self.logger.warning(f"⚠️  {warning_msg}")

    def log_info(self, info_msg: str):
        """Log informazioni generali"""
        self.logger.info(f"ℹ️  {info_msg}")

    def set_debug_mode(self, enabled: bool = True):
        """Abilita/disabilita modalità debug per troubleshooting"""
        if enabled:
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("🔧 Debug mode enabled - detailed logging active")
        else:
            self.logger.setLevel(logging.INFO)
            self.logger.info("📊 Standard logging mode - reduced verbosity")


def setup_optimized_logging(log_level: int = logging.INFO,
                            log_file: Optional[Path] = None) -> TrainingLogger:
    """Setup del logger ottimizzato con configurazione globale"""

    # Silenzia logger troppo verbosi di librerie esterne
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('torch_geometric').setLevel(logging.WARNING)
    logging.getLogger('networkx').setLevel(logging.ERROR)

    # Silenzia i moduli del progetto (riducibili a WARNING o ERROR)
    logging.getLogger('rl_gym').setLevel(logging.WARNING)
    logging.getLogger('policy_network').setLevel(logging.WARNING)
    logging.getLogger('discriminator').setLevel(logging.WARNING)
    logging.getLogger('data_loader').setLevel(logging.WARNING)

    # Riduci warning non necessari
    warnings.filterwarnings("ignore", message=".*weights_only.*")
    warnings.filterwarnings("ignore", message=".*Loading.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

    return TrainingLogger(log_level, log_file)


def create_progress_tracker() -> Dict[str, List[float]]:
    """Crea un tracker per le metriche di progresso"""
    return {
        'episode_rewards': [],
        'success_rates': [],
        'hub_improvements': [],
        'hub_scores': [],
        'training_losses': [],
        'discriminator_metrics': []
    }


# Funzioni di utilità per logging condizionale
def should_log_detailed(episode: int, improved: bool = False) -> bool:
    """Determina quando fare log dettagliato"""
    return (
            episode < 50 or  # Primi episodi sempre
            episode % 100 == 0 or  # Ogni 100 episodi
            improved  # Quando c'è miglioramento significativo
    )


def configure_external_logging():
    """Configura il logging per librerie esterne"""
    # Riduci significativamente il logging di librerie esterne
    external_loggers = [
        'matplotlib', 'PIL', 'torch', 'torch_geometric',
        'networkx', 'numpy', 'scipy', 'sklearn'
    ]

    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


if __name__ == "__main__":
    # Test del logger
    logger = setup_optimized_logging()

    # Test configurazione
    config = {
        'training': type('obj', (object,), {'num_episodes': 5000, 'num_envs': 6}),
        'rewards': type('obj', (object,), {'REWARD_SUCCESS': 20.0, 'REWARD_SMALL_IMPROVEMENT': 2.0}),
        'optimization': type('obj', (object,), {'policy_lr': 1e-4})
    }

    logger.start_training(config)

    # Test episode logging
    for episode in range(0, 250, 25):
        metrics = {
            'reward': -2.0 + episode * 0.02,
            'success_rate': min(0.25, episode * 0.001),
            'hub_improvement': 0.01 + episode * 0.0001,
            'average_hub_score': 0.7 - episode * 0.001,
            'consecutive_failures': max(0, 20 - episode // 10)
        }
        logger.log_episode_progress(episode, metrics)

        if episode == 100:
            logger.log_milestone_summary(episode, {
                'episode_rewards': [metrics['reward']] * 100,
                'success_rates': [metrics['success_rate']] * 100
            })

    print("\n✅ TrainingLogger test completato!")