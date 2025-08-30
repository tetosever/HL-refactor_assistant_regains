#!/usr/bin/env python3
"""
Main Training Script for Graph Refactoring PPO - UPDATED VERSION
Uses PPOConfig as single source of truth with robust reward weight handling
"""

import os
import sys
import logging
import torch
from pathlib import Path
from typing import Dict, Any
import json
import time
import numpy as np

# Import PPO components
from ppo_trainer import PPOTrainer, PPOConfig
from rl_gym import PPORefactorEnv
from actor_critic_models import create_actor_critic


def setup_project_structure():
    """Create necessary project directories"""
    directories = [
        "data_builder/dataset/graph_features",
        "results/discriminator_pretraining",
        "results/ppo_training",
        "results/evaluation",
        "logs"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("Project structure created")


def setup_logging(log_file: str = "logs/main_ppo_training.log"):
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
        print(f"Data directory not found: {data_path}")
        return False

    pt_files = list(data_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {data_path}")
        return False

    print(f"Found {len(pt_files)} data files in {data_path}")
    return True


def run_discriminator_pretraining(discriminator_path: str, force_retrain: bool = False) -> str:
    """Run discriminator pre-training if needed"""
    logger = logging.getLogger(__name__)

    if Path(discriminator_path).exists() and not force_retrain:
        logger.info(f"Pre-trained discriminator found at {discriminator_path}")
        return discriminator_path

    logger.info("Starting discriminator pre-training...")

    try:
        # Note: Uncomment and modify this when discriminator training is available
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
    """CLAUDE: Run complete PPO training using PPOConfig as single source of truth"""
    logger = logging.getLogger(__name__)
    logger.info("Starting PPO training...")

    trainer = PPOTrainer(config)
    training_stats = trainer.train()

    logger.info("PPO training completed")

    return {
        'training_stats': training_stats,
        'best_model': f"{config.results_dir}/best_model.pt"
    }


def run_evaluation(config: PPOConfig, training_results: Dict[str, Any]) -> Dict[str, Any]:
    """CLAUDE: Comprehensive evaluation using _build_env pattern from trainer"""
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

        # CLAUDE: Use model config from checkpoint or PPOConfig defaults
        if 'model_config' in checkpoint:
            ac_config = checkpoint['model_config']
            logger.info(f"Using saved model config: {ac_config}")
        else:
            # CLAUDE: Fallback config from PPOConfig
            ac_config = {
                'node_dim': config.node_dim,
                'hidden_dim': config.hidden_dim,
                'num_layers': config.num_layers,
                'num_actions': config.num_actions,
                'global_features_dim': config.global_features_dim,
                'dropout': config.dropout,
                'shared_encoder': config.shared_encoder
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
                try:
                    from discriminator import create_discriminator
                    discriminator = create_discriminator(**disc_checkpoint['model_config'])
                    discriminator.load_state_dict(disc_checkpoint['model_state_dict'])
                    discriminator.to(config.device).eval()
                except ImportError:
                    logger.warning("Discriminator module not available")
            except Exception as e:
                logger.warning(f"Could not load discriminator: {e}")

        # CLAUDE: Initialize environment using same pattern as trainer (_build_env)
        env = PPORefactorEnv(
            data_path=config.data_path,
            discriminator=discriminator,
            max_steps=config.max_steps,
            device=config.device,
            reward_weights=config.reward_weights,
            # CLAUDE: Growth control parameters from config
            max_new_nodes_per_episode=config.max_new_nodes,
            max_total_node_growth=config.max_growth,
            growth_penalty_mode=config.growth_penalty_mode,
            growth_penalty_power=config.growth_penalty_power,
            growth_penalty_gamma_nodes=config.growth_gamma_nodes,
            growth_penalty_gamma_edges=config.growth_gamma_edges
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
            # Reset returns Data object
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
                # Extract global features using PPO method
                global_features = env.get_global_features()

                # Get greedy action from model
                with torch.no_grad():
                    from torch_geometric.data import Batch
                    state_batch = Batch.from_data_list([current_data]).to(config.device)
                    output = model(state_batch, global_features)
                    action = torch.argmax(output['action_probs'], dim=1).item()

                # Step returns Data object
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

        with open(eval_dir / 'ppo_evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)

        logger.info(f"Evaluation completed and saved to {eval_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    return eval_results


def print_reward_weights_safely(reward_weights: Dict[str, float]):
    """CLAUDE: Print reward weights using .get() to avoid KeyError"""
    print("Reward Configuration:")

    # Main reward components
    print(f"  Hub Weight: {reward_weights.get('hub_weight', 'N/A')}")
    print(f"  Adversarial Weight: {reward_weights.get('adversarial_weight', 'N/A')}")
    print(f"  Success Threshold: {reward_weights.get('success_threshold', 'N/A')}")
    print(f"  Success Bonus: {reward_weights.get('success_bonus', 'N/A')}")

    # Step rewards
    step_valid = reward_weights.get('step_valid', 'N/A')
    step_invalid = reward_weights.get('step_invalid', 'N/A')
    time_penalty = reward_weights.get('time_penalty', 'N/A')
    print(f"  Step Valid: {step_valid}, Invalid: {step_invalid}, Time: {time_penalty}")

    # Growth control
    node_penalty = reward_weights.get('node_penalty', 'N/A')
    edge_penalty = reward_weights.get('edge_penalty', 'N/A')
    cap_exceeded = reward_weights.get('cap_exceeded_penalty', 'N/A')
    print(f"  Growth - Node: {node_penalty}, Edge: {edge_penalty}, Cap Exceeded: {cap_exceeded}")

    # Structural penalties
    cycle_penalty = reward_weights.get('cycle_penalty', 'N/A')
    duplicate_penalty = reward_weights.get('duplicate_penalty', 'N/A')
    early_stop_penalty = reward_weights.get('early_stop_penalty', 'N/A')
    print(f"  Structural - Cycle: {cycle_penalty}, Duplicate: {duplicate_penalty}, Early Stop: {early_stop_penalty}")

    # Patience
    patience = reward_weights.get('patience', 'N/A')
    print(f"  Patience: {patience}")


def main():
    """
    CLAUDE: Main function using PPOConfig as single source of truth
    """

    # Create PPOConfig with all defaults - no arguments, no overrides
    config = PPOConfig()

    print("=== Graph Refactoring PPO Training (PPOConfig-Based) ===")
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Data path: {config.data_path}")
    print(f"Learning rate: {config.lr}")
    print(f"Max episode steps: {config.max_steps}")
    print(f"Warmup episodes: {config.warmup_episodes}")
    print()

    # CLAUDE: Print reward weights safely using .get()
    print_reward_weights_safely(config.reward_weights)
    print()

    # CLAUDE: Print growth control parameters
    print("Growth Control Configuration:")
    print(f"  Max New Nodes: {config.max_new_nodes}")
    print(f"  Max Growth Ratio: {config.max_growth}")
    print(f"  Growth Penalty Mode: {config.growth_penalty_mode}")
    print(f"  Growth Gamma Nodes: {config.growth_gamma_nodes}")
    print(f"  Growth Gamma Edges: {config.growth_gamma_edges}")
    print()

    # Setup
    setup_project_structure()
    logger = setup_logging()

    # Check data availability
    if not check_data_availability(config.data_path):
        logger.error("Training data not available. Please check data_path in PPOConfig.")
        return

    try:
        # Step 1: Discriminator pretraining (if needed)
        logger.info("=== Step 1: Discriminator Pretraining ===")
        discriminator_path = run_discriminator_pretraining(
            config.discriminator_path, force_retrain=False
        )
        # Update config with actual discriminator path
        if discriminator_path:
            config.discriminator_path = discriminator_path

        # Step 2: PPO Training
        logger.info("=== Step 2: PPO Training ===")
        training_results = run_ppo_training(config)
        logger.info("PPO training phase completed successfully")

        # Step 3: Evaluation
        logger.info("=== Step 3: Evaluation ===")
        eval_results = run_evaluation(config, training_results)
        logger.info("Evaluation completed successfully")

        # Print final results
        if eval_results and 'summary' in eval_results:
            summary = eval_results['summary']
            print("\n=== FINAL RESULTS ===")
            print(f"Success Rate: {summary['success_rate']:.1%}")
            print(f"Average Hub Improvement: {summary['avg_hub_improvement']:.4f}")
            print(f"Average Reward: {summary['avg_episode_reward']:.3f}")
            print(f"Average Episode Length: {summary['avg_episode_length']:.1f}")

            # CLAUDE: Check acceptance criteria from prompt
            success_rate = summary['success_rate']
            avg_reward = summary['avg_episode_reward']
            avg_hub_delta = summary['avg_hub_improvement']
            avg_length = summary['avg_episode_length']

            print(f"\n=== ACCEPTANCE CRITERIA CHECK ===")
            print(f"✓ Success Rate ≥ 90%: {'✅' if success_rate >= 0.9 else '❌'} ({success_rate:.1%})")
            print(f"✓ Avg Reward ≥ 3.2: {'✅' if avg_reward >= 3.2 else '❌'} ({avg_reward:.3f})")
            print(f"✓ Avg Hub Δ ≥ 0.40: {'✅' if avg_hub_delta >= 0.40 else '❌'} ({avg_hub_delta:.3f})")
            print(f"✓ Avg Episode Length ≤ 3: {'✅' if avg_length <= 3.0 else '❌'} ({avg_length:.1f})")

            if (success_rate >= 0.9 and avg_reward >= 3.2 and
                    avg_hub_delta >= 0.40 and avg_length <= 3.0):
                print("🎉 ALL ACCEPTANCE CRITERIA MET!")
            elif avg_hub_delta > 0.6:
                print("SUCCESS: Significant improvement achieved!")
            elif avg_hub_delta > 0.3:
                print("PROGRESS: Moderate improvement achieved")
            else:
                print("LIMITED: Consider tuning hyperparameters in PPOConfig")

        logger.info("=== Training Pipeline Completed Successfully ===")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()