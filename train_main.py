#!/usr/bin/env python3
"""
Main Training Script for Graph Refactoring PPO - CORRECTED VERSION
Maintains original training pipeline structure but fixes PPO implementation
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
import numpy as np

# CORRECTED: Import PPO trainer instead of A2C
from ppo_trainer import PPOTrainer, PPOConfig
from rl_gym import RefactorEnv, PPORefactorEnv
from actor_critic_models import create_actor_critic


def setup_project_structure():
    """UNCHANGED: Create necessary project directories"""
    directories = [
        "data_builder/dataset/graph_features",
        "results/discriminator_pretraining",
        "results/ppo_training",  # CORRECTED: ppo_training instead of rl_training
        "results/evaluation",
        "logs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("Project structure created")


def setup_logging(log_file: str = "logs/main_ppo_training.log"):  # CORRECTED: filename
    """UNCHANGED: Setup comprehensive logging"""
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
    """UNCHANGED: Check if training data is available"""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"Data directory not found: {data_path}")
        return False

    pt_files = list(data_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {data_path}")
        return False

    print(f"Found {len(pt_files)} data files in {data_path}")
    return True


def run_discriminator_pretraining(discriminator_path: str, force_retrain: bool = False) -> str:
    """MAINTAINED: Run discriminator pre-training if needed"""
    logger = logging.getLogger(__name__)

    if Path(discriminator_path).exists() and not force_retrain:
        logger.info(f"Pre-trained discriminator found at {discriminator_path}")
        return discriminator_path

    logger.info("Starting discriminator pre-training...")

    try:
        # Note: Import and run discriminator pretraining
        # from pre_training_discriminator import main as pretrain_discriminator
        # pretrain_discriminator()

        if Path(discriminator_path).exists():
            logger.info("Discriminator pre-training completed successfully")
            return discriminator_path
        else:
            logger.warning("Discriminator not found after training, continuing without it")
            return ""

    except Exception as e:
        logger.warning(f"Discriminator pre-training failed: {e}, continuing without it")
        return ""


def run_ppo_training(config: PPOConfig) -> Dict[str, Any]:
    """
    CORRECTED: Run complete PPO training (replaces separate warmup/adversarial phases)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting PPO training...")

    # CORRECTED: Single PPO trainer handles all phases
    trainer = PPOTrainer(config)
    training_stats = trainer.train()

    logger.info("PPO training completed")

    return {
        'training_stats': training_stats,
        'best_model': f"{config.results_dir}/best_model.pt"
    }


def run_evaluation(config: PPOConfig, training_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    CORRECTED: Comprehensive evaluation of trained PPO model
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive evaluation...")

    eval_results = {}
    best_model_path = training_results.get('best_model')

    if not best_model_path or not Path(best_model_path).exists():
        logger.warning("Best model not found, using latest checkpoint")
        best_model_path = f"{config.results_dir}/latest_checkpoint.pt"

    if not Path(best_model_path).exists():
        logger.error("No trained model found for evaluation")
        return eval_results

    try:
        # Load model checkpoint
        checkpoint = torch.load(best_model_path, map_location=config.device)

        # CORRECTED: Use model config from checkpoint
        if 'model_config' in checkpoint:
            ac_config = checkpoint['model_config']
            logger.info(f"Using saved model config: {ac_config}")
        else:
            # Fallback config
            ac_config = {
                'node_dim': 7,
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'num_actions': 7,
                'global_features_dim': 4,  # CORRECTED: PPO uses 4 features
                'dropout': config.dropout,
                'shared_encoder': True  # CORRECTED: PPO requires shared encoder
            }

        # Create and load model
        model = create_actor_critic(ac_config).to(config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load discriminator if available
        discriminator = None
        if config.discriminator_path and Path(config.discriminator_path).exists():
            try:
                disc_checkpoint = torch.load(config.discriminator_path, map_location=config.device)
                from discriminator import create_discriminator
                discriminator = create_discriminator(**disc_checkpoint['model_config'])
                discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
                discriminator.to(config.device).eval()
            except Exception as e:
                logger.warning(f"Could not load discriminator: {e}")

        # CORRECTED: Initialize PPO-compatible environment
        env = PPORefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=config.max_episode_steps,
            device=config.device,
            reward_weights=config.reward_weights
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
            # CORRECTED: Reset returns Data object
            current_data = env.reset()
            initial_hub_score = env.initial_metrics['hub_score']

            # Get initial discriminator score
            if discriminator is not None:
                with torch.no_grad():
                    try:
                        disc_output = discriminator(current_data)
                        if isinstance(disc_output, dict):
                            initial_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                        else:
                            initial_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                        eval_metrics['discriminator_scores_before'].append(initial_disc_score)
                    except Exception as e:
                        logger.warning(f"Discriminator error on initial: {e}")

            # Run episode
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # CORRECTED: Extract global features using PPO method
                global_features = env.get_global_features()

                # Get greedy action from model
                with torch.no_grad():
                    output = model(current_data, global_features)
                    action = torch.argmax(output['action_probs'], dim=1).item()

                # CORRECTED: Step returns Data object
                next_data, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                current_data = next_data

            # Calculate final metrics
            final_hub_score = env._calculate_metrics(current_data)['hub_score']
            hub_improvement = initial_hub_score - final_hub_score

            # Get final discriminator score
            if discriminator is not None:
                with torch.no_grad():
                    try:
                        disc_output = discriminator(current_data)
                        if isinstance(disc_output, dict):
                            final_disc_score = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                        else:
                            final_disc_score = torch.softmax(disc_output, dim=1)[0, 1].item()
                        eval_metrics['discriminator_scores_after'].append(final_disc_score)
                    except Exception as e:
                        logger.warning(f"Discriminator error on final: {e}")

            # Record metrics
            eval_metrics['episode_rewards'].append(episode_reward)
            eval_metrics['hub_score_improvements'].append(hub_improvement)
            eval_metrics['initial_hub_scores'].append(initial_hub_score)
            eval_metrics['final_hub_scores'].append(final_hub_score)
            eval_metrics['episode_lengths'].append(episode_length)

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
        logger.info("Evaluation Results:")
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

        with open(eval_dir / 'ppo_evaluation_results.json', 'w') as f:  # CORRECTED: filename
            json.dump(eval_results, f, indent=2, default=str)

        logger.info(f"Evaluation completed and saved to {eval_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    return eval_results


def create_ppo_config(args) -> PPOConfig:
    """
    CORRECTED: Create PPO configuration from arguments
    """
    return PPOConfig(
        # General
        device=args.device,
        experiment_name=args.experiment_name,
        random_seed=args.seed,

        # Data paths
        data_path=args.data_path,
        discriminator_path=args.discriminator_path,
        results_dir=args.results_dir,

        # Environment - maintains original structure
        max_episode_steps=args.max_steps,
        reward_weights={
            'hub_weight': 10.0,
            'step_valid': 0.0,
            'step_invalid': -0.1,
            'time_penalty': -0.02,
            'early_stop_penalty': -0.5,
            'cycle_penalty': -0.2,
            'duplicate_penalty': -0.1,
            'adversarial_weight': 2.0,
            'patience': 15
        },

        # Training phases - maintains original logic
        num_episodes=args.num_episodes,
        warmup_episodes=args.warmup_episodes,
        adversarial_start_episode=args.adversarial_start_episode,

        # PPO hyperparameters
        lr=args.lr,
        gamma=0.95,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=4,
        rollout_episodes=16,
        minibatch_size=8,
        value_loss_coef=0.5,
        entropy_coef=0.05,
        max_grad_norm=0.5,

        # Model architecture - maintains original
        hidden_dim=args.hidden_dim,
        num_layers=3,
        dropout=0.2,

        # Logging and evaluation - maintains original frequencies
        log_every=50,
        eval_every=200,
        save_every=500,
        num_eval_episodes=50,
        early_stopping_patience=300,
        min_improvement=0.01
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Graph Refactoring PPO Training')

    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default='data_builder/dataset/graph_features',
                        help='Path to training data')
    parser.add_argument('--discriminator_path', type=str,
                        default='results/discriminator_pretraining/pretrained_discriminator.pt',
                        help='Path to pre-trained discriminator')

    # Training arguments
    parser.add_argument('--num_episodes', type=int, default=3000,
                        help='Total number of training episodes')
    parser.add_argument('--warmup_episodes', type=int, default=600,
                        help='Number of warmup episodes')
    parser.add_argument('--adversarial_start_episode', type=int, default=600,
                        help='Episode to start adversarial training')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum steps per episode')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for GNN layers')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output arguments
    parser.add_argument('--experiment_name', type=str,
                        default='graph_refactor_ppo_corrected',
                        help='Experiment name for results')
    parser.add_argument('--results_dir', type=str,
                        default='results/ppo_training',
                        help='Directory to save results')

    # Mode arguments
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'both'],
                        default='both', help='Training mode')
    parser.add_argument('--skip_pretrain_discriminator', action='store_true',
                        help='Skip discriminator pretraining')

    args = parser.parse_args()

    # Handle auto device selection
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


def main():
    """
    CORRECTED: Main function with proper PPO pipeline
    """
    args = parse_args()

    print("=== Graph Refactoring PPO Training (Corrected) ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Data path: {args.data_path}")

    # Setup
    setup_project_structure()
    logger = setup_logging()

    # Check data availability
    if not check_data_availability(args.data_path):
        logger.error("Training data not available. Please check data_path.")
        return

    # Create PPO configuration
    config = create_ppo_config(args)

    try:
        # Step 1: Discriminator pretraining (if needed)
        discriminator_path = ""
        if not args.skip_pretrain_discriminator:
            logger.info("=== Step 1: Discriminator Pretraining ===")
            discriminator_path = run_discriminator_pretraining(
                args.discriminator_path, force_retrain=False
            )
            config.discriminator_path = discriminator_path

        # Step 2: PPO Training
        training_results = {}
        if args.mode in ['train', 'both']:
            logger.info("=== Step 2: PPO Training ===")
            training_results = run_ppo_training(config)
            logger.info("PPO training phase completed successfully")

        # Step 3: Evaluation
        if args.mode in ['eval', 'both']:
            logger.info("=== Step 3: Evaluation ===")
            if not training_results and args.mode == 'eval':
                # Load from existing checkpoint for eval-only mode
                training_results = {'best_model': f"{config.results_dir}/best_model.pt"}

            eval_results = run_evaluation(config, training_results)
            logger.info("Evaluation completed successfully")

            # Print final results
            if eval_results and 'summary' in eval_results:
                summary = eval_results['summary']
                print("\n=== FINAL RESULTS ===")
                print(f"Success Rate: {summary['success_rate']:.1%}")
                print(f"Average Hub Improvement: {summary['avg_hub_improvement']:.4f}")
                print(f"Average Reward: {summary['avg_episode_reward']:.3f}")

                if summary['avg_hub_improvement'] > 0.6:
                    print("🎉 SUCCESS: Significant improvement achieved!")
                elif summary['avg_hub_improvement'] > 0.3:
                    print("📈 PROGRESS: Moderate improvement achieved")
                else:
                    print("⚠️ LIMITED: Consider tuning hyperparameters")

        logger.info("=== Training Pipeline Completed Successfully ===")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()