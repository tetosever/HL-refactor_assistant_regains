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

        state_dict_keys = list(checkpoint['actor_critic_state_dict'].keys())

        # Controlla se usa encoder separati
        has_actor_encoder = any('actor_encoder' in key for key in state_dict_keys)
        has_shared_encoder = any('encoder.input_proj' in key for key in state_dict_keys)

        # Determina shared_encoder
        if has_actor_encoder:
            actual_shared_encoder = False
        elif has_shared_encoder:
            actual_shared_encoder = True
        else:
            logger.error("Cannot determine encoder architecture from checkpoint")
            return eval_results

        # Determina numero di azioni dal peso dell'actor_head
        actor_head_keys = [k for k in state_dict_keys if 'actor_head' in k and 'weight' in k]
        if actor_head_keys:
            # Prendi l'ultimo layer (quello finale)
            final_layer_key = max(actor_head_keys, key=lambda x: int(x.split('.')[1]))
            final_layer_shape = checkpoint['actor_critic_state_dict'][final_layer_key].shape
            actual_num_actions = final_layer_shape[0]
        else:
            logger.error("Cannot determine number of actions from checkpoint")
            return eval_results

        logger.info(
            f"🔍 Detected architecture: shared_encoder={actual_shared_encoder}, num_actions={actual_num_actions}")

        # Create model
        ac_config = {
            'node_dim': 7,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'num_actions': actual_num_actions,
            'global_features_dim': 10,
            'dropout': config.dropout,
            'shared_encoder': actual_shared_encoder
        }

        model = create_actor_critic(ac_config).to(config.device)
        model.load_state_dict(checkpoint['actor_critic_state_dict'])
        model.eval()

        # Load discriminator if available
        discriminator = None
        if Path(config.discriminator_path).exists():
            disc_checkpoint = torch.load(config.discriminator_path, map_location=config.device)
            discriminator = create_discriminator(**disc_checkpoint['model_config'])
            discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
            discriminator.to(config.device)
            discriminator.eval()

        # Initialize environment for evaluation
        env = RefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=config.max_episode_steps,
            device=config.device
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
                    disc_output = discriminator(initial_data)
                    if isinstance(disc_output, dict):
                        initial_disc_score = torch.softmax(disc_output['logits'], dim=1)[
                            0, 1].item()  # Smelly probability
                    else:
                        initial_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                    eval_metrics['discriminator_scores_before'].append(initial_disc_score)

            # Run episode
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Extract global features
                current_data = env.current_data
                global_features = env._extract_global_features(current_data).unsqueeze(0)

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
                    disc_output = discriminator(final_data)
                    if isinstance(disc_output, dict):
                        final_disc_score = torch.softmax(disc_output['logits'], dim=1)[
                            0, 1].item()  # Smelly probability
                    else:
                        final_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                    eval_metrics['discriminator_scores_after'].append(final_disc_score)

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
            eval_summary[
                'discriminator_improvement'] = avg_disc_before - avg_disc_after  # Reduction in smelly probability
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
    """Main training pipeline"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Graph Refactoring RL Training Pipeline')
    parser.add_argument('--config-file', type=str, default=None,
                        help='Path to configuration file (JSON/YAML)')
    parser.add_argument('--data-path', type=str, default='data_builder/dataset/graph_features',
                        help='Path to training data')
    parser.add_argument('--discriminator-path', type=str,
                        default='results/discriminator_pretraining/pretrained_discriminator.pt',
                        help='Path to pre-trained discriminator')
    parser.add_argument('--results-dir', type=str, default='results/rl_training',
                        help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default='graph_refactor_rl_v1',
                        help='Experiment name')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--skip-pretraining', action='store_true',
                        help='Skip discriminator pre-training')
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip warm-up phase')
    parser.add_argument('--skip-adversarial', action='store_true',
                        help='Skip adversarial phase')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation phase')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retrain discriminator even if exists')

    args = parser.parse_args()

    # Setup
    setup_project_structure()
    logger = setup_logging()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info("🚀 Starting Graph Refactoring RL Training Pipeline")
    logger.info(f"📁 Data path: {args.data_path}")
    logger.info(f"🧠 Device: {device}")
    logger.info(f"🎯 Experiment: {args.experiment_name}")

    # Check data availability
    if not check_data_availability(args.data_path):
        logger.error("❌ Training data not available. Please prepare the dataset first.")
        sys.exit(1)

    # ✅ NUOVO: Load configuration from utils_and_configs
    try:
        # Initialize config manager
        config_manager = ConfigManager(config_file=args.config_file)

        # Override with command line arguments if provided
        if args.data_path != 'data_builder/dataset/graph_features':
            config_manager.env_config.data_path = args.data_path

        # Create TrainingConfig from the config manager
        config = create_training_config_from_manager(
            config_manager=config_manager,
            device=device,
            experiment_name=args.experiment_name,
            discriminator_path=args.discriminator_path,
            results_dir=args.results_dir
        )

        logger.info("✅ Configuration loaded from utils_and_configs")

    except Exception as e:
        logger.error(f"❌ Failed to create training configuration: {e}")
        sys.exit(1)

    training_results = {}

    try:
        # Phase 1: Discriminator Pre-training
        if not args.skip_pretraining:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 1: DISCRIMINATOR PRE-TRAINING")
            logger.info("=" * 60)

            discriminator_path = run_discriminator_pretraining({
                'discriminator_path': args.discriminator_path
            }, force_retrain=args.force_retrain)

            training_results['discriminator_path'] = discriminator_path

        # Phase 2: Warm-up RL Training
        if not args.skip_warmup:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 2: WARM-UP RL TRAINING")
            logger.info("=" * 60)

            warmup_results = run_warmup_training(config)
            training_results.update(warmup_results)

        # Phase 3: Adversarial Fine-tuning
        if not args.skip_adversarial:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 3: ADVERSARIAL FINE-TUNING")
            logger.info("=" * 60)

            adversarial_results = run_adversarial_training(config, training_results)
            training_results.update(adversarial_results)

        # Phase 4: Evaluation
        if not args.skip_evaluation:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 4: COMPREHENSIVE EVALUATION")
            logger.info("=" * 60)

            eval_results = run_evaluation(config, training_results)
            training_results['evaluation'] = eval_results

        # Phase 5: Ablation Study (Optional)
        if args.run_ablation:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 5: ABLATION STUDY")
            logger.info("=" * 60)

            ablation_results = run_ablation_study(config)
            training_results['ablation'] = ablation_results

        # Save complete results
        final_results_path = Path(args.results_dir) / 'complete_training_results.json'
        with open(final_results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)

        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Complete results saved to: {final_results_path}")

        # Print summary
        if 'evaluation' in training_results:
            eval_summary = training_results['evaluation']['summary']
            logger.info("\n📈 Final Performance Summary:")
            logger.info(f"  Success Rate: {eval_summary.get('success_rate', 0):.1%}")
            logger.info(f"  Avg Hub Improvement: {eval_summary.get('avg_hub_improvement', 0):.3f}")
            logger.info(f"  Avg Episode Reward: {eval_summary.get('avg_episode_reward', 0):.3f}")

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")

    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

    finally:
        logger.info("🏁 Training pipeline finished")


if __name__ == "__main__":
    import numpy as np  # Import needed for evaluation

    main()