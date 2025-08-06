#!/usr/bin/env python3
"""
TrainingLogger - Sistema di logging ottimizzato per training RL con tracking discriminatore
Versione completa con monitoraggio continuo delle performance del discriminatore
"""

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from data_loader import UnifiedDataLoader


class TrainingLogger:
    """Logger ottimizzato per il training RL con focus sui risultati chiave e tracking discriminatore"""

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

        # === NUOVO: Tracking del discriminatore ===
        self.discriminator_history = {
            'accuracies': [],
            'f1_scores': [],
            'precisions': [],
            'recalls': [],
            'episodes': [],
            'validation_counts': 0,
            'update_episodes': []  # Episodi in cui è stato aggiornato
        }

        # Baseline performance dal pre-training
        self.baseline_discriminator_performance = {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

        # Tracking delle tendenze del discriminatore
        self.discriminator_trend_window = 10  # Finestra per calcolare trend
        self.discriminator_degradation_threshold = 0.1  # Soglia per degradazione significativa

        # Performance tracking del discriminatore
        self.discriminator_best_performance = {
            'accuracy': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'episode': 0
        }

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
        self.logger.info(f"   Discriminator LR: {getattr(opt, 'discriminator_lr', 'N/A')}")
        self.logger.info(f"   Hub threshold: {getattr(training, 'hub_improvement_threshold', 'N/A')}")

        self.logger.info(f"🎯 REWARD STRUCTURE:")
        self.logger.info(f"   Success: +{getattr(rewards, 'REWARD_SUCCESS', 'N/A')}")
        self.logger.info(f"   Small improvement: +{getattr(rewards, 'REWARD_SMALL_IMPROVEMENT', 'N/A')}")
        self.logger.info(f"   Hub reduction: +{getattr(rewards, 'REWARD_HUB_REDUCTION', 'N/A')}")

        self.logger.info("=" * 50)

    def set_discriminator_baseline(self, baseline_metrics: Dict[str, float]):
        """Imposta le performance baseline dal pre-training del discriminatore"""
        self.baseline_discriminator_performance.update(baseline_metrics)

        # Inizializza anche il best performance
        self.discriminator_best_performance.update(baseline_metrics)
        self.discriminator_best_performance['episode'] = 0

        self.logger.info(f"🎯 DISCRIMINATOR BASELINE SET:")
        self.logger.info(f"   Accuracy: {baseline_metrics.get('accuracy', 0):.3f}")
        self.logger.info(f"   F1: {baseline_metrics.get('f1', 0):.3f}")
        self.logger.info(f"   Precision: {baseline_metrics.get('precision', 0):.3f}")
        self.logger.info(f"   Recall: {baseline_metrics.get('recall', 0):.3f}")

    def log_discriminator_validation(self, results: Dict[str, float], episode: int = 0):
        """Log validazione discriminatore - versione estesa per tracking continuo"""
        accuracy = results.get('accuracy', 0)
        f1 = results.get('f1', 0)
        precision = results.get('precision', 0)
        recall = results.get('recall', 0)
        success_rate = results.get('success_rate', 0)

        # Determina lo status
        if accuracy > 0.75:
            status = "✅ EXCELLENT"
        elif accuracy > 0.65:
            status = "✅ GOOD"
        elif accuracy > 0.5:
            status = "⚠️  FAIR"
        else:
            status = "❌ POOR"

        # Log base
        if episode == 0:  # Validazione iniziale
            self.logger.info(f"🔍 DISCRIMINATOR INITIAL VALIDATION {status}")
        else:  # Validazione durante training
            self.logger.info(f"🔍 DISCRIMINATOR VALIDATION [Episode {episode}] {status}")

        self.logger.info(
            f"   Accuracy: {accuracy:.3f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

        if success_rate > 0:
            self.logger.info(f"   Success Rate: {success_rate:.3f}")

        # Warning per performance basse
        if accuracy < 0.6:
            self.logger.warning("   ⚠️  Low discriminator performance may affect training")

    def log_discriminator_training_update(self, episode: int, validation_results: Dict[str, float],
                                          training_metrics: Optional[Dict[str, float]] = None):
        """Log aggiornamenti del discriminatore durante il training RL"""

        accuracy = validation_results.get('accuracy', 0)
        f1 = validation_results.get('f1', 0)
        precision = validation_results.get('precision', 0)
        recall = validation_results.get('recall', 0)

        # Store history
        self.discriminator_history['accuracies'].append(accuracy)
        self.discriminator_history['f1_scores'].append(f1)
        self.discriminator_history['precisions'].append(precision)
        self.discriminator_history['recalls'].append(recall)
        self.discriminator_history['episodes'].append(episode)
        self.discriminator_history['validation_counts'] += 1
        self.discriminator_history['update_episodes'].append(episode)

        # Update best performance se necessario
        if accuracy > self.discriminator_best_performance['accuracy']:
            self.discriminator_best_performance.update({
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'episode': episode
            })
            improved = True
        else:
            improved = False

        # Calcola trend e cambiamenti rispetto al baseline
        baseline_acc = self.baseline_discriminator_performance.get('accuracy', 0)
        baseline_f1 = self.baseline_discriminator_performance.get('f1', 0)

        acc_change = accuracy - baseline_acc
        f1_change = f1 - baseline_f1

        # Determina status del discriminatore
        if accuracy > baseline_acc + 0.05:
            disc_status = "📈 IMPROVING"
        elif accuracy < baseline_acc - self.discriminator_degradation_threshold:
            disc_status = "📉 DEGRADING"
        elif abs(accuracy - baseline_acc) < 0.02:
            disc_status = "➡️  STABLE"
        else:
            disc_status = "🔄 FLUCTUATING"

        # Log principale
        self.logger.info(f"🧠 DISCRIMINATOR UPDATE [Episode {episode}] {disc_status}")
        self.logger.info(f"   Current:  Acc: {accuracy:.3f} | F1: {f1:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f}")
        self.logger.info(f"   vs Base:  Acc: {acc_change:+.3f} | F1: {f1_change:+.3f}")

        if improved:
            self.logger.info(f"   🎉 NEW BEST PERFORMANCE!")

        # Log metriche di training se disponibili
        if training_metrics:
            disc_loss = training_metrics.get('discriminator_loss', 0)
            if disc_loss > 0:
                self.logger.info(f"   Training Loss: {disc_loss:.4f}")

        # Analisi del trend se abbiamo abbastanza dati
        if len(self.discriminator_history['accuracies']) >= self.discriminator_trend_window:
            self._analyze_discriminator_trend(episode)

        # Warning per degradazione significativa
        if accuracy < baseline_acc - self.discriminator_degradation_threshold:
            self.logger.warning(f"   🚨 SIGNIFICANT DEGRADATION from baseline (-{abs(acc_change):.3f})")
            self.logger.warning(f"   Consider adjusting discriminator learning rate or update frequency")

    def _analyze_discriminator_trend(self, episode: int):
        """Analizza il trend delle performance del discriminatore"""
        window = self.discriminator_trend_window
        recent_accs = self.discriminator_history['accuracies'][-window:]
        recent_f1s = self.discriminator_history['f1_scores'][-window:]

        if len(recent_accs) < window:
            return

        # Calcola trend lineare semplice
        x = np.arange(len(recent_accs))
        acc_slope = np.polyfit(x, recent_accs, 1)[0]
        f1_slope = np.polyfit(x, recent_f1s, 1)[0]

        # Determina significatività del trend
        acc_trend_strength = abs(acc_slope) * window  # Cambiamento totale stimato

        if acc_trend_strength > 0.05:  # Trend significativo
            if acc_slope > 0:
                trend_desc = f"📈 STRONG UPWARD trend (+{acc_slope * window:.3f} over {window} validations)"
            else:
                trend_desc = f"📉 STRONG DOWNWARD trend ({acc_slope * window:.3f} over {window} validations)"

            self.logger.info(f"   {trend_desc}")

            # Raccomandazioni basate sul trend
            if acc_slope < -0.01:  # Forte degradazione
                self.logger.warning(f"   💡 RECOMMENDATION: Consider reducing discriminator learning rate")
            elif acc_slope > 0.01:  # Forte miglioramento
                self.logger.info(f"   💡 GOOD: Discriminator is adapting well to policy improvements")

    def log_discriminator_summary(self, episode: int):
        """Log riassunto delle performance del discriminatore ogni milestone"""
        if not self.discriminator_history['accuracies']:
            return

        # Statistiche generali
        total_updates = len(self.discriminator_history['update_episodes'])
        avg_accuracy = np.mean(self.discriminator_history['accuracies'])
        avg_f1 = np.mean(self.discriminator_history['f1_scores'])

        # Statistiche recenti (ultimi 5 update)
        recent_window = min(5, len(self.discriminator_history['accuracies']))
        if recent_window > 0:
            recent_acc = np.mean(self.discriminator_history['accuracies'][-recent_window:])
            recent_f1 = np.mean(self.discriminator_history['f1_scores'][-recent_window:])
        else:
            recent_acc = avg_accuracy
            recent_f1 = avg_f1

        # Best ever
        best = self.discriminator_best_performance

        self.logger.info(f"🧠 DISCRIMINATOR SUMMARY [Episode {episode}]")
        self.logger.info(f"   Total Updates: {total_updates}")
        self.logger.info(f"   Average Performance: Acc: {avg_accuracy:.3f} | F1: {avg_f1:.3f}")
        self.logger.info(f"   Recent Performance: Acc: {recent_acc:.3f} | F1: {recent_f1:.3f}")
        self.logger.info(
            f"   Best Performance: Acc: {best['accuracy']:.3f} | F1: {best['f1']:.3f} (Episode {best['episode']})")

        # Confronto con baseline
        baseline = self.baseline_discriminator_performance
        acc_vs_baseline = recent_acc - baseline['accuracy']
        f1_vs_baseline = recent_f1 - baseline['f1']

        if acc_vs_baseline > 0.02:
            baseline_status = "📈 ABOVE baseline"
        elif acc_vs_baseline < -0.05:
            baseline_status = "📉 BELOW baseline"
        else:
            baseline_status = "➡️  NEAR baseline"

        self.logger.info(f"   vs Baseline: {baseline_status} (Acc: {acc_vs_baseline:+.3f}, F1: {f1_vs_baseline:+.3f})")

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

        # === NUOVO: Discriminator summary nelle milestone ===
        self.log_discriminator_summary(episode)

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

        # === NUOVO: Final discriminator summary ===
        self._log_final_discriminator_analysis()

        # Analisi qualitativa dei risultati
        self._log_training_analysis(performance)

        self.logger.info("=" * 70)

    def _log_final_discriminator_analysis(self):
        """Analisi finale delle performance del discriminatore"""
        if not self.discriminator_history['accuracies']:
            self.logger.info(f"🧠 DISCRIMINATOR ANALYSIS: No updates recorded")
            return

        self.logger.info(f"🧠 DISCRIMINATOR FINAL ANALYSIS:")

        # Statistiche complessive
        total_updates = len(self.discriminator_history['update_episodes'])
        first_episode = min(self.discriminator_history['episodes']) if self.discriminator_history['episodes'] else 0
        last_episode = max(self.discriminator_history['episodes']) if self.discriminator_history['episodes'] else 0

        # Performance finale vs iniziale vs baseline
        final_acc = self.discriminator_history['accuracies'][-1] if self.discriminator_history['accuracies'] else 0
        final_f1 = self.discriminator_history['f1_scores'][-1] if self.discriminator_history['f1_scores'] else 0

        first_acc = self.discriminator_history['accuracies'][0] if self.discriminator_history['accuracies'] else 0
        baseline_acc = self.baseline_discriminator_performance.get('accuracy', 0)

        # Best performance
        best = self.discriminator_best_performance

        self.logger.info(f"   Total Updates: {total_updates} (Episodes {first_episode}-{last_episode})")
        self.logger.info(f"   Baseline Performance: Acc: {baseline_acc:.3f}")
        self.logger.info(
            f"   First RL Performance: Acc: {first_acc:.3f} | F1: {self.discriminator_history['f1_scores'][0] if self.discriminator_history['f1_scores'] else 0:.3f}")
        self.logger.info(f"   Final Performance: Acc: {final_acc:.3f} | F1: {final_f1:.3f}")
        self.logger.info(
            f"   Best Ever: Acc: {best['accuracy']:.3f} | F1: {best['f1']:.3f} (Episode {best['episode']})")

        # Analisi del cambiamento
        vs_baseline = final_acc - baseline_acc
        vs_first = final_acc - first_acc

        if vs_baseline > 0.05:
            baseline_conclusion = "✅ SIGNIFICANTLY IMPROVED from baseline"
        elif vs_baseline > 0.02:
            baseline_conclusion = "✅ IMPROVED from baseline"
        elif vs_baseline > -0.02:
            baseline_conclusion = "➡️  MAINTAINED baseline performance"
        elif vs_baseline > -0.05:
            baseline_conclusion = "⚠️  SLIGHTLY DEGRADED from baseline"
        else:
            baseline_conclusion = "❌ SIGNIFICANTLY DEGRADED from baseline"

        self.logger.info(f"   vs Baseline: {baseline_conclusion} ({vs_baseline:+.3f})")
        self.logger.info(f"   Training Evolution: {vs_first:+.3f} change during RL training")

        # Stabilità (varianza delle performance)
        if len(self.discriminator_history['accuracies']) > 1:
            acc_std = np.std(self.discriminator_history['accuracies'])
            if acc_std < 0.02:
                stability = "✅ VERY STABLE"
            elif acc_std < 0.05:
                stability = "✅ STABLE"
            elif acc_std < 0.1:
                stability = "⚠️  SOMEWHAT UNSTABLE"
            else:
                stability = "❌ UNSTABLE"

            self.logger.info(f"   Stability: {stability} (std: {acc_std:.3f})")

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

    def get_discriminator_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche complete del discriminatore per salvataggio/analisi"""
        if not self.discriminator_history['accuracies']:
            return {}

        return {
            'total_updates': len(self.discriminator_history['update_episodes']),
            'update_episodes': self.discriminator_history['update_episodes'].copy(),
            'baseline_performance': self.baseline_discriminator_performance.copy(),
            'best_performance': self.discriminator_best_performance.copy(),
            'final_performance': {
                'accuracy': self.discriminator_history['accuracies'][-1],
                'f1': self.discriminator_history['f1_scores'][-1],
                'precision': self.discriminator_history['precisions'][-1],
                'recall': self.discriminator_history['recalls'][-1]
            } if self.discriminator_history['accuracies'] else {},
            'performance_history': {
                'accuracies': self.discriminator_history['accuracies'].copy(),
                'f1_scores': self.discriminator_history['f1_scores'].copy(),
                'precisions': self.discriminator_history['precisions'].copy(),
                'recalls': self.discriminator_history['recalls'].copy(),
                'episodes': self.discriminator_history['episodes'].copy()
            },
            'statistics': {
                'mean_accuracy': np.mean(self.discriminator_history['accuracies']),
                'std_accuracy': np.std(self.discriminator_history['accuracies']),
                'mean_f1': np.mean(self.discriminator_history['f1_scores']),
                'std_f1': np.std(self.discriminator_history['f1_scores']),
                'accuracy_range': (min(self.discriminator_history['accuracies']),
                                   max(self.discriminator_history['accuracies'])),
                'f1_range': (min(self.discriminator_history['f1_scores']),
                             max(self.discriminator_history['f1_scores']))
            } if self.discriminator_history['accuracies'] else {}
        }

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

@dataclass
class DiscriminatorPerformanceState:
    """Stato delle performance del discriminatore per monitoraggio avanzato"""
    baseline_accuracy: float = 0.0
    baseline_f1: float = 0.0
    baseline_precision: float = 0.0
    baseline_recall: float = 0.0

    current_accuracy: float = 0.0
    current_f1: float = 0.0
    current_precision: float = 0.0
    current_recall: float = 0.0

    best_accuracy: float = 0.0
    best_f1: float = 0.0
    best_episode: int = 0

    # Tracking del trend
    accuracy_history: List[float] = None
    f1_history: List[float] = None
    episode_history: List[int] = None

    # Contatori per controllo adaptivo
    consecutive_degradations: int = 0
    consecutive_improvements: int = 0
    total_updates: int = 0
    successful_validations: int = 0
    failed_validations: int = 0

    # Soglie di allarme
    degradation_threshold: float = 0.05  # Soglia di degradazione significativa
    critical_threshold: float = 0.10  # Soglia critica per intervento
    stability_window: int = 10  # Finestra per calcolo stabilità

    def __post_init__(self):
        if self.accuracy_history is None:
            self.accuracy_history = []
        if self.f1_history is None:
            self.f1_history = []
        if self.episode_history is None:
            self.episode_history = []

    def update(self, metrics: Dict[str, float], episode: int):
        """Aggiorna lo stato con nuove metriche"""
        self.current_accuracy = metrics.get('accuracy', 0)
        self.current_f1 = metrics.get('f1', 0)
        self.current_precision = metrics.get('precision', 0)
        self.current_recall = metrics.get('recall', 0)

        # Update history
        self.accuracy_history.append(self.current_accuracy)
        self.f1_history.append(self.current_f1)
        self.episode_history.append(episode)

        # Update best performance
        if self.current_accuracy > self.best_accuracy:
            self.best_accuracy = self.current_accuracy
            self.best_f1 = self.current_f1
            self.best_episode = episode
            self.consecutive_improvements += 1
            self.consecutive_degradations = 0
        else:
            self.consecutive_improvements = 0

        # Check for degradation
        if self.is_degrading():
            self.consecutive_degradations += 1
        else:
            self.consecutive_degradations = 0

        self.total_updates += 1

    def set_baseline(self, metrics: Dict[str, float]):
        """Imposta performance baseline"""
        self.baseline_accuracy = metrics.get('accuracy', 0)
        self.baseline_f1 = metrics.get('f1', 0)
        self.baseline_precision = metrics.get('precision', 0)
        self.baseline_recall = metrics.get('recall', 0)

    def is_degrading(self) -> bool:
        """Controlla se c'è degradazione significativa rispetto al baseline"""
        return (self.baseline_accuracy - self.current_accuracy) > self.degradation_threshold

    def is_critically_degraded(self) -> bool:
        """Controlla se c'è degradazione critica"""
        return (self.baseline_accuracy - self.current_accuracy) > self.critical_threshold

    def is_stable(self) -> bool:
        """Controlla se le performance sono stabili nella finestra recente"""
        if len(self.accuracy_history) < self.stability_window:
            return True

        recent_acc = self.accuracy_history[-self.stability_window:]
        std_dev = np.std(recent_acc)
        return std_dev < 0.03  # Soglia di stabilità

    def get_trend(self) -> str:
        """Calcola trend delle performance recenti"""
        if len(self.accuracy_history) < 5:
            return "insufficient_data"

        recent = self.accuracy_history[-5:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"

    def should_reduce_training_frequency(self) -> bool:
        """Determina se ridurre la frequenza di training del discriminatore"""
        return (self.consecutive_degradations >= 3 or
                self.is_critically_degraded() or
                not self.is_stable())

    def should_increase_training_frequency(self) -> bool:
        """Determina se aumentare la frequenza di training del discriminatore"""
        return (self.consecutive_improvements >= 5 and
                self.is_stable() and
                self.current_accuracy > self.baseline_accuracy + 0.02)

    def get_status_summary(self) -> Dict[str, Any]:
        """Restituisce riassunto dello stato attuale"""
        return {
            'current_performance': {
                'accuracy': self.current_accuracy,
                'f1': self.current_f1,
                'precision': self.current_precision,
                'recall': self.current_recall
            },
            'vs_baseline': {
                'accuracy_diff': self.current_accuracy - self.baseline_accuracy,
                'f1_diff': self.current_f1 - self.baseline_f1,
                'is_degrading': self.is_degrading(),
                'is_critical': self.is_critically_degraded()
            },
            'stability': {
                'is_stable': self.is_stable(),
                'trend': self.get_trend(),
                'consecutive_degradations': self.consecutive_degradations,
                'consecutive_improvements': self.consecutive_improvements
            },
            'recommendations': {
                'reduce_frequency': self.should_reduce_training_frequency(),
                'increase_frequency': self.should_increase_training_frequency()
            }
        }

class AdvancedDiscriminatorMonitor:
    """Monitor avanzato per controllo completo del discriminatore"""

    def __init__(self, discriminator: nn.Module, data_loader: UnifiedDataLoader,
                 device: torch.device, logger):
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.device = device
        self.logger = logger

        # Stato delle performance
        self.perf_state = DiscriminatorPerformanceState()

        # Configurazione monitoring
        self.validation_frequency = 25  # Validazione ogni N episodi
        self.detailed_validation_frequency = 100  # Validazione dettagliata ogni N episodi
        self.emergency_validation_threshold = 5  # Validazione d'emergenza dopo N degradazioni

        # Backup del modello per rollback
        self.best_model_state = None
        self.emergency_backup_state = None
        self.backup_episode = 0

        # Adaptive training parameters
        self.base_discriminator_lr = 5e-5
        self.current_discriminator_lr = self.base_discriminator_lr
        self.base_update_frequency = 50
        self.current_update_frequency = self.base_update_frequency

        # Statistiche dettagliate
        self.validation_history = []
        self.emergency_interventions = 0
        self.rollback_count = 0

        self.logger.log_info("🔍 Advanced Discriminator Monitor initialized")

    def set_baseline_performance(self, baseline_metrics: Dict[str, float]):
        """Imposta performance baseline dal pretraining"""
        self.perf_state.set_baseline(baseline_metrics)

        # Salva stato iniziale come backup d'emergenza
        self.emergency_backup_state = self.discriminator.state_dict().copy()

        # Log baseline
        self.logger.set_discriminator_baseline(baseline_metrics)
        self.logger.log_info(f"📊 Discriminator baseline set - Accuracy: {baseline_metrics.get('accuracy', 0):.3f}")

    def should_validate(self, episode: int) -> bool:
        """Determina se eseguire validazione in questo episodio"""
        # Validazione regolare
        if episode % self.validation_frequency == 0:
            return True

        # Validazione d'emergenza se degradazione consecutiva
        if self.perf_state.consecutive_degradations >= self.emergency_validation_threshold:
            self.logger.log_warning(
                f"🚨 Emergency validation triggered - {self.perf_state.consecutive_degradations} consecutive degradations")
            return True

        return False

    def validate_performance(self, episode: int, detailed: bool = False) -> Dict[str, float]:
        """Validazione completa delle performance del discriminatore"""
        try:
            # Determine sample size based on validation type
            sample_size = 100 if detailed else 50

            # Load validation data
            val_data, val_labels = self.data_loader.load_dataset(
                max_samples_per_class=sample_size // 2,
                shuffle=True,
                validate_all=True,
                remove_metadata=True
            )

            if len(val_data) < 10:
                self.logger.log_warning("⚠️  Insufficient validation data")
                return self._get_fallback_metrics()

            # Detailed metrics collection
            predictions = []
            true_labels = []
            prediction_confidences = []
            failed_predictions = 0

            self.discriminator.eval()
            with torch.no_grad():
                for data, label in zip(val_data, val_labels):
                    try:
                        # Get prediction
                        prob_smelly = self._get_safe_prediction(data)

                        if prob_smelly is not None:
                            prediction = 1 if prob_smelly > 0.5 else 0
                            predictions.append(prediction)
                            true_labels.append(label)
                            prediction_confidences.append(abs(prob_smelly - 0.5))
                        else:
                            failed_predictions += 1

                    except Exception as e:
                        failed_predictions += 1
                        continue

            if not predictions:
                self.logger.log_error("❌ All validation predictions failed")
                return self._get_fallback_metrics()

            # Calculate comprehensive metrics
            predictions = np.array(predictions)
            true_labels = np.array(true_labels)

            # Basic metrics
            accuracy = np.mean(predictions == true_labels)

            # Per-class metrics
            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))
            tn = np.sum((predictions == 0) & (true_labels == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            avg_confidence = np.mean(prediction_confidences) if prediction_confidences else 0

            results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'specificity': float(specificity),
                'avg_confidence': float(avg_confidence),
                'failed_predictions': failed_predictions,
                'total_samples': len(val_data),
                'success_rate': len(predictions) / len(val_data)
            }

            # Update performance state
            self.perf_state.update(results, episode)

            # Store validation history
            self.validation_history.append({
                'episode': episode,
                'metrics': results.copy(),
                'detailed': detailed
            })

            # Log results with appropriate level of detail
            self._log_validation_results(episode, results, detailed)

            return results

        except Exception as e:
            self.logger.log_error(f"Validation failed: {e}")
            self.perf_state.failed_validations += 1
            return self._get_fallback_metrics()

    def _get_safe_prediction(self, data: Data) -> Optional[float]:
        """Ottiene predizione sicura dal discriminatore"""
        try:
            data = data.clone().to(self.device)

            # Validate unified format
            if data.x.size(1) != 7:
                return None

            # Ensure required attributes
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)

            if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                num_edges = data.edge_index.size(1)
                data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32, device=self.device)

            output = self.discriminator(data)
            if output and 'logits' in output:
                probs = F.softmax(output['logits'], dim=1)
                return probs[0, 1].item()

            return None

        except Exception:
            return None

    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Metriche di fallback in caso di errore"""
        return {
            'accuracy': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'specificity': 0.5,
            'avg_confidence': 0.1,
            'failed_predictions': -1,
            'success_rate': 0.0
        }

    def _log_validation_results(self, episode: int, results: Dict[str, float], detailed: bool):
        """Log risultati validazione con livello appropriato"""
        # Always log to discriminator training update for tracking
        self.logger.log_discriminator_training_update(episode, results)

        if detailed or self.perf_state.is_degrading():
            self.logger.log_info(f"🔍 DETAILED DISCRIMINATOR VALIDATION [Episode {episode}]")
            self.logger.log_info(f"   Accuracy: {results['accuracy']:.3f} | F1: {results['f1']:.3f}")
            self.logger.log_info(f"   Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f}")
            self.logger.log_info(
                f"   Confidence: {results['avg_confidence']:.3f} | Success Rate: {results['success_rate']:.3f}")

            # Status analysis
            status = self.perf_state.get_status_summary()
            vs_baseline = status['vs_baseline']
            stability = status['stability']

            self.logger.log_info(
                f"   vs Baseline: Acc {vs_baseline['accuracy_diff']:+.3f} | F1 {vs_baseline['f1_diff']:+.3f}")
            self.logger.log_info(f"   Trend: {stability['trend']} | Stable: {stability['is_stable']}")

            if vs_baseline['is_degrading']:
                self.logger.log_warning(f"   ⚠️  Performance degradation detected")
            if vs_baseline['is_critical']:
                self.logger.log_warning(f"   🚨 CRITICAL degradation - intervention needed")

    def check_and_handle_degradation(self, episode: int) -> bool:
        """Controlla e gestisce degradazione del discriminatore"""
        if not self.perf_state.is_degrading():
            return False

        self.logger.log_warning(f"🚨 Discriminator degradation detected at episode {episode}")

        # Analisi della degradazione
        status = self.perf_state.get_status_summary()

        if self.perf_state.is_critically_degraded():
            self.logger.log_warning(f"🚨 CRITICAL DEGRADATION - Emergency measures activated")
            return self._handle_critical_degradation(episode)
        else:
            self.logger.log_warning(f"⚠️  Moderate degradation - Adjusting training parameters")
            return self._handle_moderate_degradation(episode)

    def _handle_critical_degradation(self, episode: int) -> bool:
        """Gestisce degradazione critica con misure d'emergenza"""
        self.emergency_interventions += 1

        # Opzione 1: Rollback al miglior modello
        if self.best_model_state is not None and episode - self.backup_episode > 100:
            self.logger.log_warning(
                f"🔄 ROLLBACK: Restoring best discriminator state from episode {self.backup_episode}")
            self.discriminator.load_state_dict(self.best_model_state)
            self.rollback_count += 1

            # Reset learning rate
            self.current_discriminator_lr = self.base_discriminator_lr * 0.5
            self.current_update_frequency = self.base_update_frequency * 2

            return True

        # Opzione 2: Riduzione drastica learning rate
        else:
            self.logger.log_warning(f"🔧 EMERGENCY: Reducing discriminator learning rate")
            self.current_discriminator_lr *= 0.1
            self.current_update_frequency *= 3

            return False

    def _handle_moderate_degradation(self, episode: int) -> bool:
        """Gestisce degradazione moderata con aggiustamenti graduali"""
        # Riduce learning rate del discriminatore
        self.current_discriminator_lr *= 0.8

        # Aumenta frequenza di update
        self.current_update_frequency = max(25, int(self.current_update_frequency * 0.8))

        self.logger.log_info(
            f"🔧 Adjusted: LR={self.current_discriminator_lr:.2e}, Freq={self.current_update_frequency}")

        return False

    def update_best_model(self, episode: int):
        """Aggiorna il backup del miglior modello"""
        if (self.perf_state.current_accuracy > self.perf_state.best_accuracy and
                self.perf_state.is_stable()):
            self.best_model_state = self.discriminator.state_dict().copy()
            self.backup_episode = episode

            self.logger.log_info(
                f"💾 New best discriminator model saved (Episode {episode}, Acc: {self.perf_state.current_accuracy:.3f})")

    def get_adaptive_training_params(self) -> Dict[str, Any]:
        """Restituisce parametri di training adattivi"""
        return {
            'discriminator_lr': self.current_discriminator_lr,
            'update_frequency': self.current_update_frequency,
            'should_skip_update': self.perf_state.is_critically_degraded(),
            'validation_frequency': max(10,
                                        self.validation_frequency // 2) if self.perf_state.is_degrading() else self.validation_frequency
        }

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Riassunto completo del monitoraggio"""
        return {
            'performance_state': self.perf_state.get_status_summary(),
            'monitoring_stats': {
                'total_validations': len(self.validation_history),
                'successful_validations': self.perf_state.successful_validations,
                'failed_validations': self.perf_state.failed_validations,
                'emergency_interventions': self.emergency_interventions,
                'rollback_count': self.rollback_count
            },
            'adaptive_params': self.get_adaptive_training_params(),
            'best_performance': {
                'accuracy': self.perf_state.best_accuracy,
                'f1': self.perf_state.best_f1,
                'episode': self.perf_state.best_episode
            }
        }

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
    # Test del logger con tracking discriminatore
    logger = setup_optimized_logging()

    # Test configurazione
    config = {
        'training': type('obj', (object,), {'num_episodes': 5000, 'num_envs': 6}),
        'rewards': type('obj', (object,), {'REWARD_SUCCESS': 20.0, 'REWARD_SMALL_IMPROVEMENT': 2.0}),
        'optimization': type('obj', (object,), {'policy_lr': 1e-4, 'discriminator_lr': 5e-5})
    }

    logger.start_training(config)

    # Test baseline discriminatore
    logger.set_discriminator_baseline({
        'accuracy': 0.742,
        'f1': 0.738,
        'precision': 0.745,
        'recall': 0.731
    })

    # Simula validazione iniziale
    logger.log_discriminator_validation({
        'accuracy': 0.742,
        'f1': 0.738,
        'precision': 0.745,
        'recall': 0.731,
        'success_rate': 0.85
    })

    # Test episode logging
    for episode in range(0, 350, 25):
        metrics = {
            'reward': -2.0 + episode * 0.02,
            'success_rate': min(0.25, episode * 0.001),
            'hub_improvement': 0.01 + episode * 0.0001,
            'average_hub_score': 0.7 - episode * 0.001,
            'consecutive_failures': max(0, 20 - episode // 10)
        }
        logger.log_episode_progress(episode, metrics)

        # Simula aggiornamenti discriminatore ogni 50 episodi
        if episode > 0 and episode % 50 == 0:
            # Simula performance che fluttuano leggermente
            base_acc = 0.742
            variation = np.random.normal(0, 0.03)  # Piccola variazione
            new_acc = max(0.5, min(0.95, base_acc + variation))

            # F1 correlato ma con sua variazione
            f1_variation = np.random.normal(0, 0.025)
            new_f1 = max(0.5, min(0.95, 0.738 + variation * 0.8 + f1_variation))

            logger.log_discriminator_training_update(episode, {
                'accuracy': new_acc,
                'f1': new_f1,
                'precision': new_acc + 0.01,
                'recall': new_f1 - 0.005
            }, {
                                                         'discriminator_loss': 0.3 + np.random.normal(0, 0.1)
                                                     })

        # Test milestone con discriminator summary
        if episode == 100:
            logger.log_milestone_summary(episode, {
                'episode_rewards': [metrics['reward']] * 100,
                'success_rates': [metrics['success_rate']] * 100
            })

    # Test statistiche finali discriminatore
    disc_stats = logger.get_discriminator_statistics()
    print(f"\n🧠 DISCRIMINATOR STATS:")
    print(f"Total updates: {disc_stats.get('total_updates', 0)}")
    print(f"Mean accuracy: {disc_stats.get('statistics', {}).get('mean_accuracy', 0):.3f}")
    print(f"Accuracy range: {disc_stats.get('statistics', {}).get('accuracy_range', (0, 0))}")

    print("\n✅ Enhanced TrainingLogger with Discriminator Tracking test completato!")