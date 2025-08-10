#!/usr/bin/env python3
"""
Main Training Script for Graph Refactoring RL
Complete pipeline: Pre-training discriminator → Warm-up RL → Adversarial fine-tuning → Evaluation
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, Any
import json
import time

# Import all necessary modules
from discriminator import create_discriminator
from pre_training_discriminator import main as pretrain_discriminator
from a2c_trainer import A2CTrainer, TrainingConfig
from rl_gym import RefactorEnv
from actor_critic_models import create_actor_critic
from utils_and_configs import ConfigManager


def setup_project_structure():
    """Create necessary project directories"""

    directories = [
        "data_builder/dataset/graph_features",
        "results/discriminator_pretraining",
        "results/rl_training",
        "results/evaluation",
        "logs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("📁 Project structure created")


def setup_logging(log_file: str = "logs/main_training.log"):
    """Setup comprehensive logging"""

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def check_data_availability(data_path: str) -> bool:
    """Check if training data is available"""

    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_path}")
        return False

    pt_files = list(data_dir.glob("*.pt"))
    if not pt_files:
        print(f"❌ No .pt files found in {data_path}")
        return False

    print(f"✅ Found {len(pt_files)} data files in {data_path}")
    return True


def run_discriminator_pretraining(config: Dict[str, Any], force_retrain: bool = False) -> str:
    """Run discriminator pre-training phase"""

    logger = logging.getLogger(__name__)
    discriminator_path = config['discriminator_path']

    # Check if pre-trained discriminator already exists
    if Path(discriminator_path).exists() and not force_retrain:
        logger.info(f"✅ Pre-trained discriminator found at {discriminator_path}")
        return discriminator_path

    logger.info("🔧 Starting discriminator pre-training...")

    try:
        # Run discriminator pre-training
        # Note: This assumes the pre-training script is properly configured
        pretrain_discriminator()

        if Path(discriminator_path).exists():
            logger.info("✅ Discriminator pre-training completed successfully")
            return discriminator_path
        else:
            raise FileNotFoundError(f"Discriminator not found after training: {discriminator_path}")

    except Exception as e:
        logger.error(f"❌ Discriminator pre-training failed: {e}")
        raise


def run_warmup_training(config: TrainingConfig) -> Dict[str, Any]:
    """Run warm-up RL training (only hub score reward)"""

    logger = logging.getLogger(__name__)
    logger.info("🔥 Starting warm-up RL training...")

    # Create warm-up configuration
    # Filter out conflicting parameters to avoid duplicate keyword arguments
    filtered_config = {k: v for k, v in config.__dict__.items()
                       if k not in ['experiment_name', 'num_episodes', 'results_dir',
                                    'warmup_episodes', 'adversarial_start_episode']}

    warmup_config = TrainingConfig(
        experiment_name=f"{config.experiment_name}_warmup",
        num_episodes=config.warmup_episodes,
        warmup_episodes=config.warmup_episodes,  # All episodes are warm-up
        adversarial_start_episode=config.warmup_episodes + 1,  # No adversarial training
        results_dir=f"{config.results_dir}_warmup",
        **filtered_config
    )

    # Run warm-up training
    trainer = A2CTrainer(warmup_config)
    warmup_stats = trainer.train()

    logger.info("✅ Warm-up training completed")

    return {
        'warmup_stats': warmup_stats,
        'best_warmup_model': f"{warmup_config.results_dir}/best_model.pt"
    }


def run_adversarial_training(config: TrainingConfig, warmup_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run adversarial fine-tuning (RL + discriminator updates)"""

    logger = logging.getLogger(__name__)
    logger.info("⚔️ Starting adversarial fine-tuning...")

    # Create adversarial configuration
    # Filter out conflicting parameters
    filtered_config = {k: v for k, v in config.__dict__.items()
                       if k not in ['experiment_name', 'num_episodes', 'results_dir',
                                    'warmup_episodes', 'adversarial_start_episode']}

    adversarial_config = TrainingConfig(
        experiment_name=f"{config.experiment_name}_adversarial",
        num_episodes=config.num_episodes - config.warmup_episodes,
        warmup_episodes=0,  # No warm-up in this phase
        adversarial_start_episode=1,  # Start adversarial immediately
        results_dir=f"{config.results_dir}_adversarial",
        **filtered_config
    )

    # Initialize trainer
    trainer = A2CTrainer(adversarial_config)

    # --- CORREZIONE: Load warm-up model con gestione corretta degli optimizer ---
    warmup_model_path = warmup_results.get('best_warmup_model')
    if warmup_model_path and Path(warmup_model_path).exists():
        try:
            checkpoint = torch.load(warmup_model_path, map_location=trainer.device)

            # Load model state
            trainer.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

            # Load optimizer states basato sulla configurazione
            if trainer.config.use_cyclic_lr:
                # CyclicLR usa optimizer separati
                if hasattr(trainer, 'actor_optimizer') and 'actor_optimizer_state_dict' in checkpoint:
                    trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                if hasattr(trainer, 'critic_optimizer') and 'critic_optimizer_state_dict' in checkpoint:
                    trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

                # Load scheduler states
                if hasattr(trainer, 'actor_scheduler') and 'actor_scheduler_state_dict' in checkpoint:
                    trainer.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
                if hasattr(trainer, 'critic_scheduler') and 'critic_scheduler_state_dict' in checkpoint:
                    trainer.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])

                logger.info("✅ Loaded warm-up model with CyclicLR optimizers")
            else:
                # Optimizer combinato tradizionale
                if hasattr(trainer,
                           'ac_optimizer') and trainer.ac_optimizer is not None and 'ac_optimizer_state_dict' in checkpoint:
                    trainer.ac_optimizer.load_state_dict(checkpoint['ac_optimizer_state_dict'])

                # Load scheduler state
                if hasattr(trainer, 'ac_scheduler') and 'ac_scheduler_state_dict' in checkpoint:
                    trainer.ac_scheduler.load_state_dict(checkpoint['ac_scheduler_state_dict'])

                logger.info("✅ Loaded warm-up model with combined optimizer")

            logger.info(f"✅ Loaded warm-up model from {warmup_model_path}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load warm-up model: {e}. Starting from scratch.")

    # Run adversarial training
    adversarial_stats = trainer.train()

    logger.info("✅ Adversarial training completed")

    return {
        'adversarial_stats': adversarial_stats,
        'best_adversarial_model': f"{adversarial_config.results_dir}/best_model.pt"
    }


def run_evaluation(config: TrainingConfig, training_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run comprehensive evaluation of trained models"""

    logger = logging.getLogger(__name__)
    logger.info("📊 Starting comprehensive evaluation...")

    eval_results = {}

    # Load best model
    best_model_path = training_results.get('best_adversarial_model')
    if not best_model_path or not Path(best_model_path).exists():
        logger.warning("⚠️ Best model not found, using latest checkpoint")
        best_model_path = f"{config.results_dir}/best_model.pt"

    if not Path(best_model_path).exists():
        logger.error("❌ No trained model found for evaluation")
        return eval_results

    try:
        # Load model
        checkpoint = torch.load(best_model_path, map_location=config.device)

        # 🔧 FIX: Usa la configurazione SALVATA nel checkpoint invece di ricrearla
        if 'model_config' in checkpoint:
            # Usa la config del modello salvata
            ac_config = checkpoint['model_config']
            logger.info(f"🔧 Using saved model config: {ac_config}")
        else:
            # Fallback: prova a inferire dalle dimensioni del checkpoint
            state_dict = checkpoint['actor_critic_state_dict']

            # Trova il primo layer per inferire input_dim
            actor_head_key = None
            for key in state_dict.keys():
                if 'actor_head' in key and 'weight' in key:
                    actor_head_key = key
                    break

            if actor_head_key:
                input_dim = state_dict[actor_head_key].shape[1]
                logger.info(f"🔍 Inferred input dimension from checkpoint: {input_dim}")

                # Ricostruisci config basandoti sulle dimensioni
                # input_dim = max_nodes * 7 + max_nodes^2 + global_features
                # Prova diverse combinazioni per trovare quella corretta

                possible_configs = []
                for max_nodes in [30, 40, 50, 60, 66, 70]:
                    for global_feat in [4, 10]:
                        expected_dim = max_nodes * 7 + max_nodes * max_nodes + global_feat
                        if expected_dim == input_dim:
                            possible_configs.append((max_nodes, global_feat))

                if possible_configs:
                    max_nodes_inferred, global_feat_inferred = possible_configs[0]
                    logger.info(f"🎯 Inferred: max_nodes={max_nodes_inferred}, global_features={global_feat_inferred}")

                    ac_config = {
                        'node_dim': 7,
                        'hidden_dim': config.hidden_dim,
                        'num_layers': config.num_layers,
                        'num_actions': 7,
                        'global_features_dim': global_feat_inferred,
                        'dropout': config.dropout,
                        'shared_encoder': False
                    }
                else:
                    logger.error(f"❌ Cannot infer model config from input_dim={input_dim}")
                    return eval_results
            else:
                logger.error("❌ Cannot find actor_head in checkpoint")
                return eval_results

        # Create model con la config corretta
        from actor_critic_models import create_actor_critic
        model = create_actor_critic(ac_config).to(config.device)

        # Load state dict
        model.load_state_dict(checkpoint['actor_critic_state_dict'])
        model.eval()

        # Load discriminator if available
        discriminator = None
        if Path(config.discriminator_path).exists():
            try:
                disc_checkpoint = torch.load(config.discriminator_path, map_location=config.device)
                from discriminator import create_discriminator
                discriminator = create_discriminator(**disc_checkpoint['model_config'])
                discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
                discriminator.to(config.device)
                discriminator.eval()
            except Exception as e:
                logger.warning(f"⚠️ Could not load discriminator: {e}")

        # 🔧 FIX: Initialize environment con config compatibile
        # Usa le stesse global_features del modello

        # Crea reward_weights compatibili
        eval_reward_weights = {
            'hub_weight': 10.0,
            'step_valid': 0.05,
            'step_invalid': -0.1,
            'time_penalty': -0.01,
            'early_stop_penalty': -0.5,
            'cycle_penalty': -0.2,
            'duplicate_penalty': -0.1,
            'adversarial_weight': 2.0,
            'patience': 15
        }

        env = RefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=config.max_episode_steps,
            device=config.device,
            reward_weights=eval_reward_weights  # 🔧 Aggiungi reward_weights
        )

        # Run evaluation episodes
        eval_episodes = 100
        eval_metrics = {
            'episode_rewards': [],
            'hub_score_improvements': [],
            'initial_hub_scores': [],
            'final_hub_scores': [],
            'episode_lengths': [],
            'discriminator_scores_before': [],
            'discriminator_scores_after': [],
            'success_rate': 0.0
        }

        successful_episodes = 0

        for episode in range(eval_episodes):
            # Reset environment
            env.reset()
            initial_data = env.current_data.clone()
            initial_hub_score = env.initial_metrics['hub_score']

            # Get initial discriminator score
            if discriminator is not None:
                with torch.no_grad():
                    try:
                        disc_output = discriminator(initial_data)
                        if isinstance(disc_output, dict):
                            initial_disc_score = torch.softmax(disc_output['logits'], dim=1)[
                                0, 1].item()  # Smelly probability
                        else:
                            initial_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                        eval_metrics['discriminator_scores_before'].append(initial_disc_score)
                    except Exception as e:
                        logger.warning(f"⚠️ Discriminator error on initial: {e}")

            # Run episode
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Extract global features COMPATIBILI con il modello
                current_data = env.current_data
                global_features = env._extract_global_features(current_data).unsqueeze(0)

                # 🔧 VERIFICA DIMENSIONI
                expected_obs_dim = ac_config.get('global_features_dim', 4)
                if global_features.shape[1] != expected_obs_dim:
                    logger.warning(
                        f"⚠️ Global features mismatch: got {global_features.shape[1]}, expected {expected_obs_dim}")
                    # Adatta le dimensioni
                    if global_features.shape[1] > expected_obs_dim:
                        global_features = global_features[:, :expected_obs_dim]
                    else:
                        # Pad con zeri
                        padding = torch.zeros(1, expected_obs_dim - global_features.shape[1],
                                              device=global_features.device)
                        global_features = torch.cat([global_features, padding], dim=1)

                # Get action from model (greedy)
                with torch.no_grad():
                    output = model(current_data, global_features)
                    action = torch.argmax(output['action_probs'], dim=1).item()

                # Take action
                _, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            # Get final metrics
            final_data = env.current_data
            final_hub_score = env._calculate_metrics(final_data)['hub_score']
            hub_improvement = final_hub_score - initial_hub_score

            # Get final discriminator score
            if discriminator is not None:
                with torch.no_grad():
                    try:
                        disc_output = discriminator(final_data)
                        if isinstance(disc_output, dict):
                            final_disc_score = torch.softmax(disc_output['logits'], dim=1)[
                                0, 1].item()  # Smelly probability
                        else:
                            final_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                        eval_metrics['discriminator_scores_after'].append(final_disc_score)
                    except Exception as e:
                        logger.warning(f"⚠️ Discriminator error on final: {e}")

            # Record metrics
            eval_metrics['episode_rewards'].append(episode_reward)
            eval_metrics['hub_score_improvements'].append(hub_improvement)
            eval_metrics['initial_hub_scores'].append(initial_hub_score)
            eval_metrics['final_hub_scores'].append(final_hub_score)
            eval_metrics['episode_lengths'].append(episode_length)

            # Check if episode was successful (positive hub score improvement)
            if hub_improvement > 0:
                successful_episodes += 1

        # Calculate summary statistics
        eval_metrics['success_rate'] = successful_episodes / eval_episodes

        eval_summary = {
            'num_episodes': eval_episodes,
            'success_rate': eval_metrics['success_rate'],
            'avg_episode_reward': float(np.mean(eval_metrics['episode_rewards'])),
            'std_episode_reward': float(np.std(eval_metrics['episode_rewards'])),
            'avg_hub_improvement': float(np.mean(eval_metrics['hub_score_improvements'])),
            'std_hub_improvement': float(np.std(eval_metrics['hub_score_improvements'])),
            'avg_episode_length': float(np.mean(eval_metrics['episode_lengths'])),
            'discriminator_improvement': 0.0
        }

        if eval_metrics['discriminator_scores_before'] and eval_metrics['discriminator_scores_after']:
            avg_disc_before = float(np.mean(eval_metrics['discriminator_scores_before']))
            avg_disc_after = float(np.mean(eval_metrics['discriminator_scores_after']))
            eval_summary['discriminator_improvement'] = avg_disc_before - avg_disc_after
            eval_summary['avg_discriminator_score_before'] = avg_disc_before
            eval_summary['avg_discriminator_score_after'] = avg_disc_after

        # Log evaluation results
        logger.info("📊 Evaluation Results:")
        logger.info(f"  Success Rate: {eval_summary['success_rate']:.3f}")
        logger.info(
            f"  Avg Reward: {eval_summary['avg_episode_reward']:.3f} ± {eval_summary['std_episode_reward']:.3f}")
        logger.info(
            f"  Avg Hub Improvement: {eval_summary['avg_hub_improvement']:.3f} ± {eval_summary['std_hub_improvement']:.3f}")
        logger.info(f"  Avg Episode Length: {eval_summary['avg_episode_length']:.1f}")

        if 'discriminator_improvement' in eval_summary:
            logger.info(f"  Discriminator Improvement: {eval_summary['discriminator_improvement']:.3f}")

        eval_results = {
            'summary': eval_summary,
            'detailed_metrics': eval_metrics
        }

        # Save evaluation results
        eval_dir = Path("results/evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)

        with open(eval_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)

        logger.info(f"✅ Evaluation completed and saved to {eval_dir}")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

    return eval_results

def create_training_config_from_manager(config_manager: ConfigManager,
                                        device: str,
                                        experiment_name: str,
                                        discriminator_path: str,
                                        results_dir: str) -> TrainingConfig:
    """Convert ConfigManager to TrainingConfig"""

    env_config = config_manager.env_config
    model_config = config_manager.model_config
    opt_config = config_manager.opt_config
    schedule_config = config_manager.schedule_config
    logging_config = config_manager.logging_config

    return TrainingConfig(
        # General
        device=device,
        experiment_name=experiment_name,

        # Environment
        data_path=env_config.data_path,
        discriminator_path=discriminator_path,
        results_dir=results_dir,
        max_episode_steps=env_config.max_episode_steps,
        reward_weights=env_config.reward_weights,

        # Model
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        num_actions=model_config.num_actions,
        dropout=model_config.dropout,
        shared_encoder=model_config.shared_encoder,

        # Optimization
        lr_actor=opt_config.lr_actor,
        lr_critic=opt_config.lr_critic,
        lr_discriminator=opt_config.lr_discriminator,
        gamma=opt_config.gamma,
        value_loss_coef=opt_config.value_loss_coef,
        entropy_coef=opt_config.entropy_coef,
        max_grad_norm=opt_config.max_grad_norm,

        # Schedule
        num_episodes=schedule_config.num_episodes,
        warmup_episodes=schedule_config.warmup_episodes,
        adversarial_start_episode=schedule_config.adversarial_start_episode,
        batch_size=schedule_config.batch_size,
        update_every=schedule_config.update_every,
        discriminator_update_every=schedule_config.discriminator_update_every,

        # Logging
        log_every=logging_config.log_every,
        eval_every=logging_config.eval_every,
        save_every=logging_config.save_every,
        num_eval_episodes=logging_config.num_eval_episodes,
        early_stopping_patience=logging_config.early_stopping_patience,
        min_improvement=logging_config.min_improvement
    )

def run_ablation_study(config: TrainingConfig) -> Dict[str, Any]:
    """Run ablation study to analyze component contributions"""

    logger = logging.getLogger(__name__)
    logger.info("🔬 Starting ablation study...")

    ablation_results = {}

    # Test different configurations
    ablation_configs = [
        {
            'name': 'no_discriminator',
            'description': 'Training without discriminator feedback',
            'changes': {'discriminator_path': None}
        },
        {
            'name': 'no_adversarial',
            'description': 'Training without adversarial updates',
            'changes': {'adversarial_start_episode': config.num_episodes + 1}
        },
        {
            'name': 'reduced_architecture',
            'description': 'Smaller network architecture',
            'changes': {'hidden_dim': 64, 'num_layers': 2}
        },
        {
            'name': 'different_lr',
            'description': 'Different learning rates',
            'changes': {'lr_actor': 1e-4, 'lr_critic': 5e-4}
        }
    ]

    for ablation_config in ablation_configs:
        try:
            logger.info(f"🧪 Running ablation: {ablation_config['name']}")

            # Create modified config
            modified_config = TrainingConfig(**{
                **config.__dict__,
                'experiment_name': f"{config.experiment_name}_ablation_{ablation_config['name']}",
                'num_episodes': config.num_episodes // 4,  # Shorter training for ablation
                'results_dir': f"{config.results_dir}_ablation_{ablation_config['name']}",
                **ablation_config['changes']
            })

            # Run training
            trainer = A2CTrainer(modified_config)
            stats = trainer.train()

            # Quick evaluation
            eval_metrics = trainer._evaluate_model()

            ablation_results[ablation_config['name']] = {
                'description': ablation_config['description'],
                'final_performance': eval_metrics,
                'training_stats': {
                    'final_reward': stats['episode_rewards'][-10:] if stats['episode_rewards'] else [0],
                    'final_hub_improvements': stats['hub_score_improvements'][-10:] if stats[
                        'hub_score_improvements'] else [0]
                }
            }

            logger.info(f"✅ Ablation {ablation_config['name']} completed")

        except Exception as e:
            logger.error(f"❌ Ablation {ablation_config['name']} failed: {e}")
            ablation_results[ablation_config['name']] = {'error': str(e)}

    # Save ablation results
    ablation_dir = Path("results/ablation")
    ablation_dir.mkdir(parents=True, exist_ok=True)

    with open(ablation_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)

    logger.info(f"🔬 Ablation study completed and saved to {ablation_dir}")

    return ablation_results


def main():
    """Enhanced main training function with exploration-focused configuration"""

    # 🔧 Enhanced configuration with exploration parameters
    config = TrainingConfig(
        experiment_name="graph_refactor_a2c_exploration_v4",

        # 🚀 EXTENDED training for plateau breakthrough
        num_episodes=3500,  # Increased from 2200
        warmup_episodes=600,  # Maintained
        adversarial_start_episode=600,

        # 🎯 Base learning rates (will be enhanced by curriculum)
        lr_actor=3e-4,
        lr_critic=1e-4,
        lr_discriminator=1e-4,

        # 🔥 EXPLORATION PARAMETERS
        use_epsilon_greedy=True,
        epsilon_start=0.3,  # High initial exploration
        epsilon_end=0.05,  # Minimum exploration
        epsilon_decay_episodes=1000,  # Long decay for sustained exploration

        # 📈 ENHANCED ENTROPY REGULARIZATION
        entropy_coef_base=0.05,  # Warmup phase
        entropy_coef_adversarial=0.15,  # Higher entropy during adversarial
        entropy_ramp_episodes=200,  # Smooth transition

        # 🎯 CURRICULUM ADVERSARIAL LEARNING
        use_curriculum_adversarial=True,
        adversarial_weight_start=0.5,  # Start gentle
        adversarial_weight_end=2.0,  # Ramp to full strength
        adversarial_ramp_episodes=800,  # Long curriculum

        # 🔍 EXPLORATION MONITORING
        track_action_entropy=True,
        track_policy_variance=True,
        exploration_log_every=25,

        # CyclicLR maintained but with longer cycles
        use_cyclic_lr=True,
        base_lr_actor=3e-4,
        max_lr_actor=1e-3,
        step_size_up_actor=200,  # Longer cycles for stability
        base_lr_critic=1e-4,
        max_lr_critic=3e-4,
        step_size_up_critic=250,  # Longer cycles

        # Training parameters optimized for exploration
        batch_size=32,
        update_every=15,  # More frequent updates
        discriminator_update_every=40,  # More frequent discriminator updates

        # Enhanced regularization
        entropy_coef=0.1,  # Will be overridden by dynamic scheduling
        max_grad_norm=0.5,
        advantage_clip=0.5,
        reward_clip=5.0,

        # Extended patience for exploration
        early_stopping_patience=400,  # Much longer patience
        min_improvement=0.005,  # More sensitive improvement detection

        # Enhanced logging
        log_every=25,
        eval_every=150,  # More frequent evaluation
        save_every=250,
        num_eval_episodes=30,  # More evaluation episodes
    )

    print("🚀 Starting Enhanced Graph Refactoring RL Training")
    print("=" * 60)
    print(f"🎯 Experiment: {config.experiment_name}")
    print(f"📊 Total Episodes: {config.num_episodes}")
    print(f"🧠 Device: {config.device}")
    print("")

    # 🔥 EXPLORATION CONFIGURATION SUMMARY
    print("🎲 EXPLORATION CONFIGURATION:")
    print(f"   Epsilon-Greedy: {config.use_epsilon_greedy}")
    if config.use_epsilon_greedy:
        print(
            f"   Epsilon: {config.epsilon_start:.3f} → {config.epsilon_end:.3f} over {config.epsilon_decay_episodes} episodes")
    print(f"   Entropy Regularization: {config.entropy_coef_base:.3f} → {config.entropy_coef_adversarial:.3f}")
    print(f"   Curriculum Adversarial: {config.use_curriculum_adversarial}")
    if config.use_curriculum_adversarial:
        print(f"   Adversarial Weight: {config.adversarial_weight_start:.3f} → {config.adversarial_weight_end:.3f}")
    print("")

    # 📊 TRAINING PHASES
    print("📅 TRAINING PHASES:")
    print(f"   Phase 1 - Warmup: Episodes 1-{config.warmup_episodes}")
    print(f"   Phase 2 - Adversarial: Episodes {config.adversarial_start_episode}-{config.num_episodes}")
    print(f"   Exploration Boost: Auto-triggered on plateau detection")
    print("")

    # 🔄 CYCLICRL CONFIGURATION
    if config.use_cyclic_lr:
        print("🔄 CYCLICRL CONFIGURATION:")
        print(
            f"   Actor LR: {config.base_lr_actor:.2e} ↔ {config.max_lr_actor:.2e} (cycle: {config.step_size_up_actor})")
        print(
            f"   Critic LR: {config.base_lr_critic:.2e} ↔ {config.max_lr_critic:.2e} (cycle: {config.step_size_up_critic})")
    print("")
    print("=" * 60)

    # Initialize trainer with enhanced configuration
    trainer = A2CTrainer(config)

    # Start training with comprehensive error handling
    try:
        print("🚀 Starting enhanced training with exploration focus...")
        print(f"🎯 Target: Break plateau at hub_improvement=0.5, reach 0.7+")
        print(f"📈 Expected: Higher action entropy, increased episode variance")
        print("")

        training_stats = trainer.train()

        print("")
        print("✅ Training completed successfully!")
        print("=" * 60)

        # Print final exploration statistics
        if hasattr(trainer, 'exploration_stats') and trainer.exploration_stats:
            final_entropy = trainer.exploration_stats['action_entropies'][-50:] if trainer.exploration_stats[
                'action_entropies'] else [0]
            final_epsilon = trainer.exploration_stats['epsilon_values'][-1] if trainer.exploration_stats[
                'epsilon_values'] else 0
            final_adversarial = trainer.exploration_stats['adversarial_weight_values'][-1] if trainer.exploration_stats[
                'adversarial_weight_values'] else 0

            print("📊 FINAL EXPLORATION STATS:")
            print(f"   Final Action Entropy: {np.mean(final_entropy):.3f}")
            print(f"   Final Epsilon: {final_epsilon:.3f}")
            print(f"   Final Adversarial Weight: {final_adversarial:.3f}")
            print("")

        # Print performance summary
        if training_stats and 'hub_score_improvements' in training_stats:
            recent_improvements = training_stats['hub_score_improvements'][-100:]
            best_improvement = max(training_stats['hub_score_improvements']) if training_stats[
                'hub_score_improvements'] else 0
            mean_recent = np.mean(recent_improvements) if recent_improvements else 0
            success_rate = sum(1 for imp in recent_improvements if imp > 0.01) / len(
                recent_improvements) if recent_improvements else 0

            print("🎯 PERFORMANCE SUMMARY:")
            print(f"   Best Hub Improvement: {best_improvement:.4f}")
            print(f"   Recent Mean Improvement: {mean_recent:.4f}")
            print(f"   Success Rate (recent): {success_rate:.1%}")

            # Check if plateau was broken
            if best_improvement > 0.6:
                print("🎉 SUCCESS: Plateau breakthrough achieved!")
            elif best_improvement > 0.5:
                print("📈 PROGRESS: Improvement beyond previous plateau")
            else:
                print("⚠️  PLATEAU: Consider further exploration tuning")

        # Print final scheduler info
        if config.use_cyclic_lr and hasattr(trainer, 'actor_scheduler'):
            final_lr_actor = trainer.actor_scheduler.get_last_lr()[0]
            final_lr_critic = trainer.critic_scheduler.get_last_lr()[0]
            print(f"📈 Final LRs - Actor: {final_lr_actor:.2e}, Critic: {final_lr_critic:.2e}")

    except KeyboardInterrupt:
        print("⏹️  Training interrupted by user")
        trainer._save_checkpoint(0, is_best=False)
        print("💾 Emergency checkpoint saved")

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        # Save emergency checkpoint
        try:
            trainer._save_checkpoint(0, is_best=False)
            print("💾 Emergency checkpoint saved")
        except:
            print("❌ Could not save emergency checkpoint")
        raise

    finally:
        if hasattr(trainer, 'writer'):
            trainer.writer.close()
        print("🏁 Training session ended")


if __name__ == "__main__":
    main()