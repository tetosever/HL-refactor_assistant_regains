#!/usr/bin/env python3
"""
Discriminator Pre-training Script with K-Fold Cross Validation

This script pre-trains the discriminator to distinguish between smelly and clean graphs
before starting the RL training. This is ESSENTIAL for the RL agent to learn effectively.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from discriminator import HubDetectionDiscriminator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_dir: Path, max_samples_per_class: int = 1000) -> Tuple[List[Data], List[int]]:
    """Load and prepare data with balanced classes"""
    logger.info("Loading dataset for discriminator pre-training...")

    smelly_data = []
    clean_data = []

    # Load all data
    for file in data_dir.glob('*.pt'):
        try:
            data = torch.load(file, map_location='cpu')

            # Validate data structure
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                continue

            # Check for required features
            if data.x.size(1) < 9:
                continue

            # Check graph size (reasonable for discriminator)
            if data.x.size(0) < 3 or data.x.size(0) > 100:
                continue

            # Classify by smell
            is_smelly = getattr(data, 'is_smelly', 0)
            if isinstance(is_smelly, torch.Tensor):
                is_smelly = is_smelly.item()

            if is_smelly == 1 and len(smelly_data) < max_samples_per_class:
                smelly_data.append(data)
            elif is_smelly == 0 and len(clean_data) < max_samples_per_class:
                clean_data.append(data)

            # Stop if we have enough samples
            if len(smelly_data) >= max_samples_per_class and len(clean_data) >= max_samples_per_class:
                break

        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
            continue

    # Balance the dataset
    min_samples = min(len(smelly_data), len(clean_data))
    if min_samples == 0:
        raise ValueError("No valid data found!")

    smelly_data = smelly_data[:min_samples]
    clean_data = clean_data[:min_samples]

    # Combine data and labels
    all_data = smelly_data + clean_data
    all_labels = [1] * len(smelly_data) + [0] * len(clean_data)

    logger.info(f"Loaded {len(smelly_data)} smelly and {len(clean_data)} clean graphs")
    logger.info(f"Total dataset size: {len(all_data)} graphs")

    return all_data, all_labels


def evaluate_model(model: nn.Module, data_loader: List[Tuple[Data, int]],
                   device: torch.device, batch_size: int = 32) -> Dict[str, float]:
    """Evaluate model and return comprehensive metrics"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(0, len(data_loader), batch_size):
            batch_data = []
            batch_labels = []

            for j in range(i, min(i + batch_size, len(data_loader))):
                data, label = data_loader[j]
                batch_data.append(data)
                batch_labels.append(label)

            try:
                # Create batch
                batch = Batch.from_data_list(batch_data).to(device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

                # Forward pass
                output = model(batch)
                loss = criterion(output['logits'], labels_tensor)

                # Get predictions
                probs = F.softmax(output['logits'], dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_tensor.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being smelly

                total_loss += loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Evaluation batch error: {e}")
                continue

    if num_batches == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.5, 'loss': float('inf')}

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    avg_loss = total_loss / num_batches

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train_single_fold(model: nn.Module, train_data: List[Tuple[Data, int]],
                      val_data: List[Tuple[Data, int]], device: torch.device,
                      epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                      patience: int = 15) -> Dict[str, Any]:
    """Train model on a single fold"""

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        num_batches = 0

        # Shuffle training data
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch_data = []
            batch_labels = []

            for j in range(i, min(i + batch_size, len(train_data))):
                data, label = train_data[j]
                batch_data.append(data)
                batch_labels.append(label)

            try:
                # Create batch
                batch = Batch.from_data_list(batch_data).to(device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

                # Forward pass
                output = model(batch)
                loss = criterion(output['logits'], labels_tensor)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Statistics
                epoch_train_loss += loss.item()
                preds = torch.argmax(output['logits'], dim=1)
                epoch_train_correct += (preds == labels_tensor).sum().item()
                epoch_train_total += len(labels_tensor)
                num_batches += 1

            except Exception as e:
                logger.warning(f"Training batch error: {e}")
                continue

        if num_batches == 0:
            logger.warning(f"No valid batches in epoch {epoch}")
            continue

        train_loss = epoch_train_loss / num_batches
        train_acc = epoch_train_correct / max(epoch_train_total, 1)

        # Validation phase
        val_metrics = evaluate_model(model, val_data, device, batch_size)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Learning rate scheduler
        scheduler.step(val_metrics['f1'])

        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Log progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                        f"Val F1: {val_metrics['f1']:.4f}")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final validation
    final_metrics = evaluate_model(model, val_data, device, batch_size)

    return {
        'history': history,
        'final_metrics': final_metrics,
        'best_val_f1': best_val_f1,
        'epochs_trained': epoch + 1
    }


def k_fold_cross_validation(data: List[Data], labels: List[int],
                            node_dim: int, edge_dim: int, device: torch.device,
                            k_folds: int = 5, epochs: int = 100,
                            batch_size: int = 32, lr: float = 1e-3) -> Dict[str, Any]:
    """Perform k-fold cross validation"""
    logger.info(f"Starting {k_folds}-fold cross validation...")

    # Stratified K-Fold to ensure balanced splits
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = []
    all_true_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        logger.info(f"\n=== Fold {fold + 1}/{k_folds} ===")

        # Create fold data
        train_data = [(data[i], labels[i]) for i in train_idx]
        val_data = [(data[i], labels[i]) for i in val_idx]

        logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

        # Create fresh model for this fold
        model = HubDetectionDiscriminator(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=128,
            num_layers=3
        ).to(device)

        # Train on this fold
        fold_result = train_single_fold(
            model, train_data, val_data, device,
            epochs=epochs, batch_size=batch_size, lr=lr
        )

        fold_results.append(fold_result)
        all_predictions.extend(fold_result['final_metrics']['predictions'])
        all_true_labels.extend(fold_result['final_metrics']['labels'])

        # Log fold results
        metrics = fold_result['final_metrics']
        logger.info(f"Fold {fold + 1} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  AUC: {metrics['auc']:.4f}")

    # Aggregate results
    accuracies = [r['final_metrics']['accuracy'] for r in fold_results]
    precisions = [r['final_metrics']['precision'] for r in fold_results]
    recalls = [r['final_metrics']['recall'] for r in fold_results]
    f1s = [r['final_metrics']['f1'] for r in fold_results]
    aucs = [r['final_metrics']['auc'] for r in fold_results]

    cv_results = {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_precision': np.mean(precisions),
        'std_precision': np.std(precisions),
        'mean_recall': np.mean(recalls),
        'std_recall': np.std(recalls),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels
    }

    return cv_results


def train_final_model(data: List[Data], labels: List[int],
                      node_dim: int, edge_dim: int, device: torch.device,
                      epochs: int = 150, batch_size: int = 32,
                      lr: float = 1e-3) -> nn.Module:
    """Train final model on all data"""
    logger.info("Training final model on all data...")

    # Create model
    model = HubDetectionDiscriminator(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        num_layers=3
    ).to(device)

    # Prepare data
    all_data = [(data[i], labels[i]) for i in range(len(data))]

    # Split for final validation (80/20)
    split_idx = int(0.8 * len(all_data))
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    train_data = [all_data[i] for i in indices[:split_idx]]
    val_data = [all_data[i] for i in indices[split_idx:]]

    # Train
    final_result = train_single_fold(
        model, train_data, val_data, device,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=20
    )

    logger.info("Final model training completed!")
    logger.info(f"Final validation metrics:")
    for key, value in final_result['final_metrics'].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    return model


def plot_confusion_matrix(y_true: List[int], y_pred: List[int],
                          save_path: Path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Smelly'],
                yticklabels=['Clean', 'Smelly'])
    plt.title('Discriminator Confusion Matrix (K-Fold CV)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results(cv_results: Dict[str, Any], save_dir: Path):
    """Save cross-validation results"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_summary = {
        'cross_validation_summary': {
            'accuracy': f"{cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}",
            'precision': f"{cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}",
            'recall': f"{cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}",
            'f1_score': f"{cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}",
            'auc_roc': f"{cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}"
        },
        'individual_folds': []
    }

    for i, fold_result in enumerate(cv_results['fold_results']):
        fold_metrics = fold_result['final_metrics']
        metrics_summary['individual_folds'].append({
            f'fold_{i + 1}': {
                'accuracy': fold_metrics['accuracy'],
                'precision': fold_metrics['precision'],
                'recall': fold_metrics['recall'],
                'f1_score': fold_metrics['f1'],
                'auc_roc': fold_metrics['auc']
            }
        })

    # Save to JSON
    with open(save_dir / 'cv_results.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # Plot confusion matrix
    plot_confusion_matrix(
        cv_results['all_true_labels'],
        cv_results['all_predictions'],
        save_dir / 'confusion_matrix.png'
    )

    logger.info(f"Results saved to {save_dir}")


def main():
    """Main training function"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'dataset_builder' / 'data' / 'dataset_graph_feature'
    results_dir = base_dir / 'results' / 'discriminator_pretraining'

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Load data
    try:
        data, labels = load_and_prepare_data(data_dir, max_samples_per_class=800)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    if len(data) == 0:
        logger.error("No valid data found!")
        return

    # Model parameters
    node_dim = data[0].x.size(1)
    edge_dim = 1  # Basic edge features

    logger.info(f"Dataset: {len(data)} graphs")
    logger.info(f"Node features: {node_dim}")
    logger.info(f"Edge features: {edge_dim}")
    logger.info(f"Class distribution: {sum(labels)} smelly, {len(labels) - sum(labels)} clean")

    # Hyperparameters
    config = {
        'k_folds': 5,
        'epochs': 80,
        'batch_size': 24,
        'learning_rate': 1e-3,
        'final_epochs': 120
    }

    logger.info(f"Training configuration: {config}")

    try:
        # K-fold cross validation
        cv_results = k_fold_cross_validation(
            data, labels, node_dim, edge_dim, device,
            k_folds=config['k_folds'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate']
        )

        # Print final results
        logger.info("\n=== CROSS-VALIDATION RESULTS ===")
        logger.info(f"Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        logger.info(f"Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
        logger.info(f"Recall:    {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        logger.info(f"F1 Score:  {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        logger.info(f"AUC-ROC:   {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")

        # Check if performance is acceptable
        if cv_results['mean_f1'] < 0.7:
            logger.warning("F1 score is below 0.7. Consider tuning hyperparameters.")
        elif cv_results['mean_f1'] > 0.85:
            logger.info("Excellent discriminator performance! ✅")
        else:
            logger.info("Good discriminator performance. ✅")

        # Train final model on all data
        final_model = train_final_model(
            data, labels, node_dim, edge_dim, device,
            epochs=config['final_epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate']
        )

        # Save everything
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save final model
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'cv_results': cv_results,
            'config': config
        }, results_dir / 'pretrained_discriminator.pt')

        # Save results
        save_results(cv_results, results_dir)

        logger.info(f"\n✅ Discriminator pre-training completed successfully!")
        logger.info(f"📁 Pretrained model saved to: {results_dir / 'pretrained_discriminator.pt'}")
        logger.info(f"📊 Cross-validation results saved to: {results_dir}")
        logger.info(f"\nYou can now use this pretrained discriminator in your RL training!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()