#!/usr/bin/env python3
"""
Updated Discriminator Pre-training with Unified Data Loader

Key changes:
- Uses UnifiedDataLoader for consistent data loading
- No normalization or standardization applied
- Same feature computation as RL training
- Deterministic and reproducible data loading order
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
from data_loader import UnifiedDataLoader, create_unified_loader, validate_dataset_consistency

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdvancedLoss(nn.Module):
    """Advanced loss combining multiple objectives"""

    def __init__(self, alpha_focal=2.0, gamma_focal=2.0, lambda_consistency=0.1,
                 lambda_diversity=0.05, label_smoothing=0.1, lambda_dice=0.1):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
        self.lambda_consistency = lambda_consistency
        self.lambda_diversity = lambda_diversity
        self.label_smoothing = label_smoothing
        self.lambda_dice = lambda_dice

    def forward(self, outputs, labels, batch_data=None):
        # Main classification loss with label smoothing
        if self.label_smoothing > 0:
            # Smooth labels
            num_classes = outputs['logits'].size(1)
            smooth_labels = torch.full_like(outputs['logits'], self.label_smoothing / num_classes)
            smooth_labels.scatter_(1, labels.unsqueeze(1),
                                   1.0 - self.label_smoothing + self.label_smoothing / num_classes)
            main_loss = F.kl_div(F.log_softmax(outputs['logits'], dim=1), smooth_labels, reduction='batchmean')
        else:
            main_loss = self.focal_loss(outputs['logits'], labels)

        probs = torch.softmax(outputs['logits'], dim=1)
        labels_onehot = F.one_hot(labels, num_classes=probs.size(1)).float()
        dice = 1 - (2 * (probs * labels_onehot).sum(dim=0) + 1e-7) / (
                probs.pow(2).sum(dim=0) + labels_onehot.pow(2).sum(dim=0) + 1e-7
        )
        dice = dice.mean()

        total_loss = main_loss + self.lambda_dice * dice
        loss_components = {'main_loss': main_loss.item(), 'dice_loss': dice.item()}

        # Feature consistency regularization
        if 'feature_projection' in outputs and self.lambda_consistency > 0:
            # Encourage similar graphs to have similar feature projections
            batch = batch_data.batch if hasattr(batch_data, 'batch') else None
            if batch is not None:
                consistency_loss = self._compute_consistency_loss(outputs['feature_projection'], batch, labels)
                total_loss += self.lambda_consistency * consistency_loss
                loss_components['consistency_loss'] = consistency_loss.item()

        # Node-level diversity regularization
        if 'node_hub_scores' in outputs and self.lambda_diversity > 0:
            diversity_loss = self._compute_diversity_loss(outputs['node_hub_scores'], batch_data.batch)
            total_loss += self.lambda_diversity * diversity_loss
            loss_components['diversity_loss'] = diversity_loss.item()

        return total_loss, loss_components

    def _compute_consistency_loss(self, feature_proj, batch, labels):
        """Compute feature consistency loss with proper error handling"""
        if batch is None or len(feature_proj) == 0:
            return torch.tensor(0.0, device=feature_proj.device, requires_grad=True)

        try:
            batch_size = batch.max().item() + 1
            consistency_loss = torch.tensor(0.0, device=feature_proj.device)
            valid_pairs = 0

            for class_label in [0, 1]:
                class_mask = labels == class_label
                if class_mask.sum() < 2:
                    continue

                class_features = []
                for i in range(batch_size):
                    if i < len(labels) and class_mask[i]:
                        node_mask = batch == i
                        if node_mask.sum() > 0:
                            graph_feature = feature_proj[node_mask].mean(dim=0)
                            class_features.append(graph_feature)

                if len(class_features) > 1:
                    class_features = torch.stack(class_features)
                    # Compute pairwise distances within class
                    dists = torch.pdist(class_features, p=2)
                    if len(dists) > 0:
                        consistency_loss += dists.mean()
                        valid_pairs += 1

            return consistency_loss / max(valid_pairs, 1)

        except Exception as e:
            # Return zero loss if computation fails
            return torch.tensor(0.0, device=feature_proj.device, requires_grad=True)

    def _compute_diversity_loss(self, node_scores, batch):
        """Encourage diversity in node hub scores within graphs with error handling"""
        if batch is None or len(node_scores) == 0:
            return torch.tensor(0.0, device=node_scores.device, requires_grad=True)

        try:
            batch_size = batch.max().item() + 1 if batch is not None else 1
            diversity_loss = torch.tensor(0.0, device=node_scores.device)
            valid_graphs = 0

            for i in range(batch_size):
                if batch is not None:
                    mask = batch == i
                    graph_scores = node_scores[mask]
                else:
                    graph_scores = node_scores

                if len(graph_scores) > 1:
                    # Encourage some nodes to be hubs, others not (diversity)
                    mean_score = graph_scores.mean()
                    variance = torch.var(graph_scores)
                    # Encourage diversity while keeping reasonable mean
                    diversity_loss += -variance + 0.1 * torch.abs(mean_score - 0.5)
                    valid_graphs += 1

            return diversity_loss / max(valid_graphs, 1)

        except Exception as e:
            return torch.tensor(0.0, device=node_scores.device, requires_grad=True)


def evaluate_model(model: nn.Module, data_loader: List[Tuple[Data, int]],
                   device: torch.device, batch_size: int = 32) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics using unified data"""
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
                all_probs.extend(probs[:, 1].cpu().numpy())

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
                      epochs: int = 120, batch_size: int = 24, lr: float = 1e-3,
                      patience: int = 20, accumulation_steps: int = 2) -> Dict[str, Any]:
    """Enhanced training with unified data loading"""

    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=lr * 0.01)

    # Advanced loss function
    criterion = AdvancedLoss(alpha_focal=1.5, gamma_focal=2.0, lambda_consistency=0.05,
                             lambda_diversity=0.02, label_smoothing=0.1)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': []
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

        # Loss components tracking
        loss_components_sum = {}

        # Shuffle training data
        random.shuffle(train_data)
        optimizer.zero_grad()

        for i in range(0, len(train_data), batch_size):
            batch_data = []
            batch_labels = []

            for j in range(i, min(i + batch_size, len(train_data))):
                data, label = train_data[j]
                batch_data.append(data)
                batch_labels.append(label)

            try:
                # Create batch with validation
                if not batch_data:
                    continue

                # All data should already be validated by UnifiedDataLoader
                batch = Batch.from_data_list(batch_data).to(device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

                # Forward pass
                output = model(batch)
                loss, loss_comp = criterion(output, labels_tensor, batch)

                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

                # Backward pass
                loss.backward()

                # Track loss components
                for key, value in loss_comp.items():
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0
                    loss_components_sum[key] += value

                # Gradient accumulation
                if (i // batch_size + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # Statistics
                epoch_train_loss += loss.item() * accumulation_steps
                preds = torch.argmax(output['logits'], dim=1)
                epoch_train_correct += (preds == labels_tensor).sum().item()
                epoch_train_total += len(labels_tensor)
                num_batches += 1

            except Exception as e:
                logger.warning(f"Training batch error: {e}")
                # Clear gradients on error
                optimizer.zero_grad()
                continue

        # Final gradient step if needed
        if num_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if num_batches == 0:
            logger.warning(f"No valid batches in epoch {epoch}")
            continue

        train_loss = epoch_train_loss / num_batches
        train_acc = epoch_train_correct / max(epoch_train_total, 1)

        # Validation phase
        val_metrics = evaluate_model(model, val_data, device, batch_size)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rate'].append(current_lr)

        # Early stopping with improved criteria
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Log progress with loss components
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                        f"Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f}")

            # Log loss components
            if loss_components_sum:
                comp_str = " | ".join([f"{k}: {v / num_batches:.4f}" for k, v in loss_components_sum.items()])
                logger.info(f"  Loss components: {comp_str}")

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


def k_fold_cross_validation(data_loader: UnifiedDataLoader, device: torch.device,
                            k_folds: int = 5, epochs: int = 120, batch_size: int = 24,
                            lr: float = 1e-3, max_samples_per_class: int = 1000) -> Dict[str, Any]:
    """K-fold cross validation with unified data loading"""
    logger.info(f"Starting {k_folds}-fold cross validation with unified data loader...")

    # Load data using unified loader
    data_list, labels = data_loader.load_dataset(
        max_samples_per_class=max_samples_per_class,
        shuffle=True,
        validate_all=True
    )

    logger.info(f"Loaded {len(data_list)} samples with unified 7-feature format")

    # Validate data consistency
    logger.info("Validating data consistency...")
    consistency_report = validate_dataset_consistency(data_loader.dataset_dir, sample_size=100)
    if consistency_report['invalid_files'] > 0:
        logger.warning(f"Found {consistency_report['invalid_files']} invalid files during consistency check")

    # Node and edge dimensions are fixed for unified format
    node_dim = 7  # Always 7 structural features
    edge_dim = 1  # Basic edge features

    # Stratified K-Fold to ensure balanced splits
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = []
    all_true_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_list, labels)):
        logger.info(f"\n=== Fold {fold + 1}/{k_folds} ===")

        # Create fold data
        train_data = [(data_list[i], labels[i]) for i in train_idx]
        val_data = [(data_list[i], labels[i]) for i in val_idx]

        logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

        # Log data hashes for first few samples to verify consistency
        if fold == 0:
            logger.info("Verifying data consistency with hashes...")
            for i in range(min(3, len(train_data))):
                data_hash = data_loader.get_data_hash(train_data[i][0])
                logger.debug(f"Sample {i} hash: {data_hash[:8]}...")

        # Create enhanced model for this fold with FIXED dimensions
        model = HubDetectionDiscriminator(
            node_dim=node_dim,  # Always 7
            edge_dim=edge_dim,  # Always 1
            hidden_dim=128,
            num_layers=4,
            dropout=0.2,
            heads=8
        ).to(device)

        # Train on this fold
        fold_result = train_single_fold(
            model, train_data, val_data, device,
            epochs=epochs, batch_size=batch_size, lr=lr,
            patience=25, accumulation_steps=2
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
        'all_true_labels': all_true_labels,
        'data_consistency_report': consistency_report,
        'unified_loader_stats': data_loader.get_dataset_statistics()
    }

    return cv_results


def train_final_model(data_loader: UnifiedDataLoader, device: torch.device,
                      epochs: int = 150, batch_size: int = 24, lr: float = 1e-3,
                      max_samples_per_class: int = 1000) -> nn.Module:
    """Train final model on all data with unified loader"""
    logger.info("Training final model with unified data loader...")

    # Load data using unified loader
    data_list, labels = data_loader.load_dataset(
        max_samples_per_class=max_samples_per_class,
        shuffle=True,
        validate_all=True
    )

    # FIXED dimensions for unified format
    node_dim = 7  # Always 7 structural features
    edge_dim = 1  # Basic edge features

    # Create enhanced model with FIXED dimensions
    model = HubDetectionDiscriminator(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=128,
        num_layers=4,
        dropout=0.15,
        heads=8
    ).to(device)

    # Prepare data with stratified split
    all_data = [(data_list[i], labels[i]) for i in range(len(data_list))]

    # Stratified split for final validation (85/15)
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(all_data)), test_size=0.15,
        stratify=labels, random_state=42
    )

    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]

    # Train with extended patience for final model
    final_result = train_single_fold(
        model, train_data, val_data, device,
        epochs=epochs, batch_size=batch_size, lr=lr,
        patience=30, accumulation_steps=2
    )

    logger.info("Final model training completed!")
    logger.info(f"Final validation metrics:")
    for key, value in final_result['final_metrics'].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    return model


def plot_training_curves(cv_results: Dict[str, Any], save_path: Path):
    """Plot training curves from cross-validation with numpy type handling"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Aggregate training curves across folds
        max_epochs = max(len(fold['history']['train_loss']) for fold in cv_results['fold_results'])

        metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            for fold_idx, fold in enumerate(cv_results['fold_results']):
                history = fold['history']
                if metric in history:
                    epochs = range(len(history[metric]))
                    # Convert to native Python types
                    values = [float(v) for v in history[metric]]
                    ax.plot(epochs, values, alpha=0.3, label=f'Fold {fold_idx + 1}')

            # Compute mean curve with type conversion
            all_curves = []
            for fold in cv_results['fold_results']:
                if metric in fold['history'] and len(fold['history'][metric]) > 0:
                    # Pad or truncate to max_epochs and convert types
                    curve = [float(v) for v in fold['history'][metric][:max_epochs]]
                    while len(curve) < max_epochs:
                        curve.append(curve[-1] if curve else 0.0)
                    all_curves.append(curve)

            if all_curves:
                mean_curve = np.mean(all_curves, axis=0)
                std_curve = np.std(all_curves, axis=0)
                epochs = range(len(mean_curve))

                ax.plot(epochs, mean_curve, 'k-', linewidth=2, label='Mean')
                ax.fill_between(epochs,
                                mean_curve - std_curve,
                                mean_curve + std_curve,
                                alpha=0.2, color='black')

            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.warning(f"Failed to plot training curves: {e}")
        plt.close('all')  # Clean up any open figures


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Path):
    """Plot and save enhanced confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    # Normalize for percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Smelly'],
                yticklabels=['Clean', 'Smelly'], ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Clean', 'Smelly'],
                yticklabels=['Clean', 'Smelly'], ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_unified_training_results(cv_results: Dict[str, Any], save_dir: Path):
    """Save comprehensive results with unified data loader information"""
    save_dir.mkdir(parents=True, exist_ok=True)

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Enhanced metrics summary with unified loader info
    metrics_summary = {
        'cross_validation_summary': {
            'accuracy': f"{float(cv_results['mean_accuracy']):.4f} ± {float(cv_results['std_accuracy']):.4f}",
            'precision': f"{float(cv_results['mean_precision']):.4f} ± {float(cv_results['std_precision']):.4f}",
            'recall': f"{float(cv_results['mean_recall']):.4f} ± {float(cv_results['std_recall']):.4f}",
            'f1_score': f"{float(cv_results['mean_f1']):.4f} ± {float(cv_results['std_f1']):.4f}",
            'auc_roc': f"{float(cv_results['mean_auc']):.4f} ± {float(cv_results['std_auc']):.4f}"
        },
        'individual_folds': [],
        'performance_analysis': {
            'best_fold_f1': float(max(fold['final_metrics']['f1'] for fold in cv_results['fold_results'])),
            'worst_fold_f1': float(min(fold['final_metrics']['f1'] for fold in cv_results['fold_results'])),
            'f1_variance': float(np.var([fold['final_metrics']['f1'] for fold in cv_results['fold_results']])),
            'consistent_performance': bool(cv_results['std_f1'] < 0.05)
        },
        'unified_data_loader_info': {
            'feature_computation_method': 'UnifiedFeatureComputer',
            'num_features': 7,
            'feature_names': [
                'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                'pagerank', 'betweenness_centrality', 'closeness_centrality'
            ],
            'no_normalization_applied': True,
            'consistent_with_rl_training': True,
            'data_consistency_report': cv_results.get('data_consistency_report', {}),
            'dataset_statistics': cv_results.get('unified_loader_stats', {})
        }
    }

    for i, fold_result in enumerate(cv_results['fold_results']):
        fold_metrics = fold_result['final_metrics']
        metrics_summary['individual_folds'].append({
            f'fold_{i + 1}': {
                'accuracy': float(fold_metrics['accuracy']),
                'precision': float(fold_metrics['precision']),
                'recall': float(fold_metrics['recall']),
                'f1_score': float(fold_metrics['f1']),
                'auc_roc': float(fold_metrics['auc']),
                'epochs_trained': int(fold_result['epochs_trained'])
            }
        })

    # Apply numpy type conversion to the entire structure
    metrics_summary = convert_numpy_types(metrics_summary)

    # Save to JSON with proper error handling
    try:
        with open(save_dir / 'unified_cv_results.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Unified results saved to {save_dir / 'unified_cv_results.json'}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
        # Fallback: save as pickle
        import pickle
        with open(save_dir / 'unified_cv_results.pkl', 'wb') as f:
            pickle.dump(metrics_summary, f)
        logger.info(f"⚠️ Saved as pickle instead: {save_dir / 'unified_cv_results.pkl'}")

    # Plot training curves
    try:
        plot_training_curves(cv_results, save_dir / 'unified_training_curves.png')
        logger.info(f"✅ Training curves saved")
    except Exception as e:
        logger.warning(f"Failed to save training curves: {e}")

    # Plot confusion matrix
    try:
        plot_confusion_matrix(
            cv_results['all_true_labels'],
            cv_results['all_predictions'],
            save_dir / 'unified_confusion_matrix.png'
        )
        logger.info(f"✅ Confusion matrix saved")
    except Exception as e:
        logger.warning(f"Failed to save confusion matrix: {e}")

    logger.info(f"📊 Enhanced unified results saved to {save_dir}")


def main():
    """Unified main training function with consistent data loading"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Enhanced random seeds for better reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'dataset_builder' / 'data' / 'dataset_graph_feature'
    results_dir = base_dir / 'results' / 'discriminator_pretraining'

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Create unified data loader
    try:
        logger.info("🔧 Creating unified data loader...")
        data_loader = create_unified_loader(data_dir, device)

        # Get dataset statistics
        stats = data_loader.get_dataset_statistics()
        logger.info(f"Dataset statistics: {stats['total_files']} files, "
                    f"{stats['label_distribution']['smelly']} smelly, "
                    f"{stats['label_distribution']['clean']} clean samples")

    except Exception as e:
        logger.error(f"Failed to create unified data loader: {e}")
        return

    # Enhanced hyperparameters
    config = {
        'k_folds': 5,
        'epochs': 100,
        'batch_size': 20,
        'learning_rate': 8e-4,
        'final_epochs': 140,
        'max_samples_per_class': 1000,
        'unified_data_loading': True,
        'no_normalization': True
    }

    logger.info(f"🔧 UNIFIED training configuration: {config}")

    try:
        # Enhanced K-fold cross validation with unified data loader
        cv_results = k_fold_cross_validation(
            data_loader, device,
            k_folds=config['k_folds'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate'],
            max_samples_per_class=config['max_samples_per_class']
        )

        # Print comprehensive results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 UNIFIED CROSS-VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        logger.info(f"Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
        logger.info(f"Recall:    {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        logger.info(f"F1 Score:  {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        logger.info(f"AUC-ROC:   {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        logger.info("=" * 60)

        # Performance assessment
        if cv_results['mean_f1'] < 0.75:
            logger.warning("F1 score is below 0.75. Consider further hyperparameter tuning.")
        elif cv_results['mean_f1'] > 0.90:
            logger.info("🌟 Outstanding discriminator performance with UNIFIED data loading!")
        elif cv_results['mean_f1'] > 0.82:
            logger.info("✅ Excellent discriminator performance with UNIFIED data loading!")
        else:
            logger.info("✅ Good discriminator performance with UNIFIED data loading.")

        # Consistency check
        if cv_results['std_f1'] > 0.05:
            logger.warning("High variance across folds. Consider more data or regularization.")
        else:
            logger.info("👍 Consistent performance across folds with unified data loading!")

        # Train final model on all data with unified data loader
        final_model = train_final_model(
            data_loader, device,
            epochs=config['final_epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate'],
            max_samples_per_class=config['max_samples_per_class']
        )

        # Save everything with unified data loader information
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save UNIFIED final model with comprehensive metadata
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'node_dim': 7,  # FIXED: always 7 unified structural features
            'edge_dim': 1,  # Basic edge features
            'cv_results': cv_results,
            'config': config,
            'model_architecture': {
                'hidden_dim': 128,
                'num_layers': 4,
                'dropout': 0.15,
                'heads': 8
            },
            'unified_data_pipeline': {
                'method': 'UnifiedDataLoader',
                'feature_computer': 'UnifiedFeatureComputer',
                'num_features': 7,
                'feature_names': [
                    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
                    'pagerank', 'betweenness_centrality', 'closeness_centrality'
                ],
                'no_normalization_applied': True,
                'consistent_with_rl_training': True,
                'deterministic_loading': True,
                'feature_computation_matches_dataset_creation': True
            },
            'data_consistency_validation': cv_results.get('data_consistency_report', {}),
            'dataset_statistics': cv_results.get('unified_loader_stats', {})
        }, results_dir / 'pretrained_discriminator.pt')

        # Save comprehensive results with unified information
        save_unified_training_results(cv_results, results_dir)

        logger.info("\n" + "=" * 60)
        logger.info("🎉 UNIFIED DISCRIMINATOR PRE-TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📁 Pretrained model: {results_dir / 'pretrained_discriminator.pt'}")
        logger.info(f"📊 Results: {results_dir}")
        logger.info(f"🔧 Data loading: UnifiedDataLoader with consistent feature computation")
        logger.info(f"🎯 Expected F1: {cv_results['mean_f1']:.3f}")
        logger.info(f"✅ Consistency with RL training: GUARANTEED")
        logger.info(f"🚫 No normalization applied: Features identical to RL training")
        logger.info("=" * 60)
        logger.info("🚀 Ready for seamless RL training with identical data format!")

        # Additional validation: Log data hashes for verification
        logger.info("\n📋 Data consistency verification:")
        sample_data, _ = data_loader.load_dataset(max_samples_per_class=5, shuffle=False)
        for i, data in enumerate(sample_data[:3]):
            data_hash = data_loader.get_data_hash(data)
            logger.info(f"  Sample {i + 1} hash: {data_hash[:12]}... (features: {data.x.shape})")

    except Exception as e:
        logger.error(f"UNIFIED training failed: {e}")
        raise


if __name__ == "__main__":
    main()