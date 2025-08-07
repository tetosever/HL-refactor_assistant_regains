#!/usr/bin/env python3
"""
Clean Discriminator Pre-training
Simplified and consistent implementation for hub detection discriminator.
"""

import logging
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Import your discriminator
from discriminator import HubDiscriminator, SimpleHubDiscriminator, create_discriminator

# =============================================================================
# HYPERPARAMETERS - Easy to modify
# =============================================================================
# Training Configuration
K_FOLDS = 5
EPOCHS = 100
FINAL_EPOCHS = 120
BATCH_SIZE = 24
LEARNING_RATE = 1e-3
PATIENCE = 20
ACCUMULATION_STEPS = 2
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# Model Configuration
NODE_DIM = 7  # Number of node features
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.3
MODEL_VERSION = "clean"  # "clean" or "simple"
ADD_NODE_SCORING = True

# Data Configuration
MAX_SAMPLES_PER_CLASS = 1000
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15  # Train=70%, Val=15%, Test=15%
USE_NORMALIZATION = True
USE_BALANCED_SAMPLING = False

# Loss Configuration
USE_FOCAL_LOSS = True  # Set to False for standard CrossEntropyLoss
USE_LABEL_SMOOTHING = False  # Set to False for hard labels
FOCAL_ALPHA = 1.5
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

# Scheduler Configuration
USE_CYCLICAL_LR = True
CYCLICAL_BASE_LR = 1e-4
CYCLICAL_MAX_LR = 1e-3
CYCLICAL_STEP_SIZE_UP = 20
CYCLICAL_MODE = 'triangular2'
CYCLICAL_GAMMA = 1.0
USE_ONECYCLE_LR = False
T_0 = 20  # Cosine annealing period
ETA_MIN_FACTOR = 0.01  # Minimum LR as factor of initial LR

# Random Seeds
RANDOM_SEED = 42

# =============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize node and edge features using StandardScaler"""

    def __init__(self):
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.fitted = False

    def fit(self, data_list: List[Data]):
        """Fit scalers on the entire dataset"""
        # Collect all node features
        all_node_features = []
        all_edge_features = []

        for data in data_list:
            all_node_features.append(data.x.numpy())
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_features.append(data.edge_attr.numpy())

        # Fit node scaler
        node_matrix = np.vstack(all_node_features)
        self.node_scaler.fit(node_matrix)

        # Fit edge scaler if edge features exist
        if all_edge_features:
            edge_matrix = np.vstack(all_edge_features)
            self.edge_scaler.fit(edge_matrix)

        self.fitted = True
        logger.info(f"✅ Fitted normalizers on {len(data_list)} graphs")
        logger.info(f"   Node features: mean={self.node_scaler.mean_[:3]}, std={self.node_scaler.scale_[:3]}")

    def transform(self, data: Data) -> Data:
        """Transform a single data object"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")

        # Clone the data
        normalized_data = data.clone()

        # Normalize node features
        normalized_data.x = torch.tensor(
            self.node_scaler.transform(data.x.numpy()),
            dtype=torch.float32
        )

        # Normalize edge features if they exist
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            normalized_data.edge_attr = torch.tensor(
                self.edge_scaler.transform(data.edge_attr.numpy()),
                dtype=torch.float32
            )

        return normalized_data


def create_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create a balanced sampler for addressing class imbalance"""
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = class_weights[labels_tensor]

    logger.info(f"📊 Class distribution: {class_counts.tolist()}")
    logger.info(f"📊 Class weights: {class_weights.tolist()}")

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""

    def __init__(self, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_graph_data(data_dir: Path, max_samples_per_class: int = MAX_SAMPLES_PER_CLASS) -> Tuple[List[Data], List[int]]:
    """
    Load graph data from single directory where each .pt file contains both data and label.
    Expected structure:
    data_dir/
    ├── graph1.pt  (contains data with y attribute for label)
    ├── graph2.pt
    └── graph3.pt

    Each .pt file should contain either:
    - PyG Data object with .y attribute for label
    - Dict with 'data' and 'label' keys
    - Dict with 'x', 'edge_index', and 'y' keys
    """
    logger.info(f"Loading data from {data_dir}")

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Get all .pt files
    all_files = list(data_dir.glob("*.pt"))
    if not all_files:
        raise ValueError(f"No .pt files found in {data_dir}")

    logger.info(f"Found {len(all_files)} .pt files")

    # Debug: Check first file structure
    if all_files:
        try:
            sample_file = all_files[0]
            sample_data = torch.load(sample_file, map_location='cpu')
            logger.info(f"📋 Sample file structure from {sample_file.name}:")
            if isinstance(sample_data, Data):
                attrs = [attr for attr in dir(sample_data) if not attr.startswith('_')]
                logger.info(f"  Data attributes: {attrs}")
                if hasattr(sample_data, 'is_smelly'):
                    logger.info(f"  ✅ Found is_smelly: {sample_data.is_smelly}")
                if hasattr(sample_data, 'x'):
                    logger.info(f"  Node features shape: {sample_data.x.shape}")
            elif isinstance(sample_data, dict):
                logger.info(f"  Dict keys: {list(sample_data.keys())}")
        except Exception as e:
            logger.warning(f"Could not inspect sample file: {e}")

    data_list = []
    labels = []
    smelly_count = 0
    clean_count = 0

    for file_path in all_files:
        try:
            # Load the file
            loaded = torch.load(file_path, map_location='cpu')

            # Extract data and label based on file format
            if isinstance(loaded, Data):
                # PyG Data object with .is_smelly attribute
                data = loaded
                if hasattr(data, 'is_smelly') and data.is_smelly is not None:
                    label = int(data.is_smelly.item() if torch.is_tensor(data.is_smelly) else data.is_smelly)
                elif hasattr(data, 'y') and data.y is not None:
                    # Fallback to .y attribute
                    label = int(data.y.item() if torch.is_tensor(data.y) else data.y)
                else:
                    logger.warning(f"Skipping {file_path}: no label found in Data object (expected .is_smelly or .y)")
                    continue

            elif isinstance(loaded, dict):
                if 'data' in loaded and 'label' in loaded:
                    # Dict format with separate data and label
                    data = loaded['data']
                    label = int(loaded['label'])
                    if not isinstance(data, Data):
                        # Convert dict to Data object
                        data = Data(x=data['x'], edge_index=data['edge_index'])

                elif 'x' in loaded and 'edge_index' in loaded:
                    # Dict format with x, edge_index, and label info
                    data = Data(x=loaded['x'], edge_index=loaded['edge_index'])
                    if 'is_smelly' in loaded:
                        label = int(
                            loaded['is_smelly'].item() if torch.is_tensor(loaded['is_smelly']) else loaded['is_smelly'])
                    elif 'y' in loaded:
                        label = int(loaded['y'].item() if torch.is_tensor(loaded['y']) else loaded['y'])
                    else:
                        logger.warning(f"Skipping {file_path}: no label found in dict (expected 'is_smelly' or 'y')")
                        continue
                else:
                    logger.warning(f"Skipping {file_path}: unrecognized dict format")
                    continue
            else:
                logger.warning(f"Skipping {file_path}: unrecognized file format")
                continue

            # Validate node features
            if not hasattr(data, 'x') or data.x is None:
                logger.warning(f"Skipping {file_path}: no node features found")
                continue

            if data.x.size(1) != NODE_DIM:
                logger.warning(f"Skipping {file_path}: expected {NODE_DIM} features, got {data.x.size(1)}")
                continue

            # Validate label
            if label not in [0, 1]:
                logger.warning(f"Skipping {file_path}: invalid label {label}, expected 0 or 1")
                continue

            # Check if we've reached the limit for this class
            if label == 1 and smelly_count >= max_samples_per_class:
                continue
            if label == 0 and clean_count >= max_samples_per_class:
                continue

            # Add to dataset
            data_list.append(data)
            labels.append(label)

            if label == 1:
                smelly_count += 1
            else:
                clean_count += 1

        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue

    logger.info(f"Loaded {len(data_list)} graphs total: {smelly_count} smelly, {clean_count} clean")

    if len(data_list) == 0:
        raise ValueError("No valid data loaded. Check data directory structure and file formats.")

    # Check for class imbalance
    imbalance_ratio = max(smelly_count, clean_count) / max(min(smelly_count, clean_count), 1)
    if imbalance_ratio > 3:
        logger.warning(f"⚠️ Class imbalance detected: ratio {imbalance_ratio:.2f}")
        logger.warning("Consider using FocalLoss or class weighting")
    else:
        logger.info(f"✅ Balanced dataset: ratio {imbalance_ratio:.2f}")

    return data_list, labels


def evaluate_model(model: nn.Module, data_list: List[Tuple[Data, int]],
                   device: torch.device) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i in range(0, len(data_list), BATCH_SIZE):
            batch_data = []
            batch_labels = []

            for j in range(i, min(i + BATCH_SIZE, len(data_list))):
                data, label = data_list[j]
                batch_data.append(data)
                batch_labels.append(label)

            try:
                # Create batch
                batch = Batch.from_data_list(batch_data).to(device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

                # Forward pass
                if MODEL_VERSION == "simple":
                    logits = model(batch)
                    output = {'logits': logits}
                else:
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
                      val_data: List[Tuple[Data, int]], device: torch.device) -> Dict[str, Any]:
    """Train model for single fold with CyclicalLR and improved configuration"""

    # Optimizer with improved settings
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # CyclicalLR scheduler as requested
    if USE_CYCLICAL_LR:
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=CYCLICAL_BASE_LR,
            max_lr=CYCLICAL_MAX_LR,
            step_size_up=CYCLICAL_STEP_SIZE_UP,
            mode=CYCLICAL_MODE,
            gamma=CYCLICAL_GAMMA,
            cycle_momentum=False  # Disable momentum cycling for stability
        )
        logger.info(f"📈 Using CyclicalLR: base_lr={CYCLICAL_BASE_LR}, max_lr={CYCLICAL_MAX_LR}")
    else:
        # Fallback to simple scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        logger.info("📈 Using ReduceLROnPlateau")

    # Loss function - simplified for balanced dataset
    if USE_LABEL_SMOOTHING and LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingLoss(LABEL_SMOOTHING)
        logger.info(f"Using Label Smoothing Loss (ε={LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss")

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

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        num_batches = 0

        # Shuffle training data
        random.shuffle(train_data)

        for i in range(0, len(train_data), BATCH_SIZE):
            batch_data = []
            batch_labels = []

            for j in range(i, min(i + BATCH_SIZE, len(train_data))):
                data, label = train_data[j]
                batch_data.append(data)
                batch_labels.append(label)

            try:
                if not batch_data:
                    continue

                # Create batch
                batch = Batch.from_data_list(batch_data).to(device)
                labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)

                optimizer.zero_grad()

                # Forward pass
                if MODEL_VERSION == "simple":
                    logits = model(batch)
                    output = {'logits': logits}
                else:
                    output = model(batch)

                loss = criterion(output['logits'], labels_tensor)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                # Update CyclicalLR every step
                if USE_CYCLICAL_LR:
                    scheduler.step()

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
        train_acc = epoch_train_correct / epoch_train_total

        # Validation phase
        val_metrics = evaluate_model(model, val_data, device)

        # Update ReduceLROnPlateau if not using CyclicalLR
        if not USE_CYCLICAL_LR:
            scheduler.step(val_metrics['f1'])

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rate'].append(current_lr)

        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Log progress
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                        f"Val F1: {val_metrics['f1']:.4f} | LR: {current_lr:.6f}")

        # Early stopping check
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final validation
    final_metrics = evaluate_model(model, val_data, device)

    return {
        'history': history,
        'final_metrics': final_metrics,
        'best_val_f1': best_val_f1,
        'epochs_trained': epoch + 1
    }


def k_fold_cross_validation(data_list: List[Data], labels: List[int], device: torch.device) -> Dict[str, Any]:
    """Perform K-fold cross validation"""
    logger.info(f"Starting {K_FOLDS}-fold cross validation...")

    # Stratified K-Fold to ensure balanced splits
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_predictions = []
    all_true_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_list, labels)):
        logger.info(f"\n=== Fold {fold + 1}/{K_FOLDS} ===")

        # Create fold data
        train_data = [(data_list[i], labels[i]) for i in train_idx]
        val_data = [(data_list[i], labels[i]) for i in val_idx]

        logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

        # Create model for this fold
        model = create_discriminator(
            version=MODEL_VERSION,
            node_dim=NODE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            add_node_scoring=ADD_NODE_SCORING
        ).to(device)

        # Train on this fold
        fold_result = train_single_fold(model, train_data, val_data, device)

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


def train_final_model(data_list: List[Data], labels: List[int], device: torch.device) -> Tuple[
    nn.Module, Dict[str, float]]:
    """Train final model with proper train/val/test split"""
    logger.info("Training final model with train/val/test split...")

    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        range(len(data_list)),
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )

    # Second split: separate train and validation from remaining data
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),  # Adjust for remaining data
        stratify=train_val_labels,
        random_state=RANDOM_SEED
    )

    # Create datasets
    train_data = [(data_list[i], labels[i]) for i in train_indices]
    val_data = [(data_list[i], labels[i]) for i in val_indices]
    test_data = [(data_list[i], labels[i]) for i in test_indices]

    logger.info(f"Final split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create model
    model = create_discriminator(
        version=MODEL_VERSION,
        node_dim=NODE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        add_node_scoring=ADD_NODE_SCORING
    ).to(device)

    # Train with extended epochs for final model
    global EPOCHS
    original_epochs = EPOCHS
    EPOCHS = FINAL_EPOCHS

    final_result = train_single_fold(model, train_data, val_data, device)

    # Restore original epochs
    EPOCHS = original_epochs

    # Final test evaluation
    test_metrics = evaluate_model(model, test_data, device)

    logger.info("Final model training completed!")
    logger.info(f"Final validation metrics:")
    for key, value in final_result['final_metrics'].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    logger.info(f"Final TEST metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    return model, test_metrics


def plot_training_curves(cv_results: Dict[str, Any], save_path: Path):
    """Plot training curves from cross-validation"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        max_epochs = max(len(fold['history']['train_loss']) for fold in cv_results['fold_results'])

        metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            for fold_idx, fold in enumerate(cv_results['fold_results']):
                history = fold['history']
                if metric in history:
                    epochs = range(len(history[metric]))
                    values = [float(v) for v in history[metric]]
                    ax.plot(epochs, values, alpha=0.3, label=f'Fold {fold_idx + 1}')

            # Compute mean curve
            all_curves = []
            for fold in cv_results['fold_results']:
                if metric in fold['history'] and len(fold['history'][metric]) > 0:
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
        plt.close('all')


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Path):
    """Plot and save confusion matrix"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.warning(f"Failed to plot confusion matrix: {e}")
        plt.close('all')


def save_results(cv_results: Dict[str, Any], save_dir: Path):
    """Save comprehensive training results"""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
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

    # Create metrics summary
    metrics_summary = {
        'hyperparameters': {
            'k_folds': K_FOLDS,
            'epochs': EPOCHS,
            'final_epochs': FINAL_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'node_dim': NODE_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'model_version': MODEL_VERSION,
            'add_node_scoring': ADD_NODE_SCORING,
            'focal_alpha': FOCAL_ALPHA,
            'focal_gamma': FOCAL_GAMMA,
            'label_smoothing': LABEL_SMOOTHING,
            'weight_decay': WEIGHT_DECAY,
            'patience': PATIENCE
        },
        'cross_validation_summary': {
            'accuracy': f"{float(cv_results['mean_accuracy']):.4f} ± {float(cv_results['std_accuracy']):.4f}",
            'precision': f"{float(cv_results['mean_precision']):.4f} ± {float(cv_results['std_precision']):.4f}",
            'recall': f"{float(cv_results['mean_recall']):.4f} ± {float(cv_results['std_recall']):.4f}",
            'f1_score': f"{float(cv_results['mean_f1']):.4f} ± {float(cv_results['std_f1']):.4f}",
            'auc_roc': f"{float(cv_results['mean_auc']):.4f} ± {float(cv_results['std_auc']):.4f}"
        },
        'individual_folds': []
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

    # Apply numpy type conversion
    metrics_summary = convert_numpy_types(metrics_summary)

    # Save results
    try:
        with open(save_dir / 'cv_results.json', 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"✅ Results saved to {save_dir / 'cv_results.json'}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")

    # Plot training curves
    try:
        plot_training_curves(cv_results, save_dir / 'training_curves.png')
        logger.info(f"✅ Training curves saved")
    except Exception as e:
        logger.warning(f"Failed to save training curves: {e}")

    # Plot confusion matrix
    try:
        plot_confusion_matrix(
            cv_results['all_true_labels'],
            cv_results['all_predictions'],
            save_dir / 'confusion_matrix.png'
        )
        logger.info(f"✅ Confusion matrix saved")
    except Exception as e:
        logger.warning(f"Failed to save confusion matrix: {e}")


def main():
    """Main training function"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seeds
    set_random_seeds(RANDOM_SEED)

    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data_builder' / 'dataset' / 'graph_features'
    results_dir = base_dir / 'results' / 'discriminator_pretraining'

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Load data
    try:
        data_list, labels = load_graph_data(data_dir, MAX_SAMPLES_PER_CLASS)

        # Apply normalization if enabled
        if USE_NORMALIZATION:
            logger.info("🔧 Applying data normalization...")
            normalizer = DataNormalizer()
            normalizer.fit(data_list)
            data_list = [normalizer.transform(data) for data in data_list]
            logger.info("✅ Data normalization completed")
        else:
            logger.info("⚠️ Data normalization disabled")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Print configuration
    logger.info("\n" + "=" * 60)
    logger.info("🔧 IMPROVED DISCRIMINATOR PRE-TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model version: {MODEL_VERSION}")
    logger.info(f"Node features: {NODE_DIM}")
    logger.info(f"Hidden dim: {HIDDEN_DIM} (increased)")
    logger.info(f"Num layers: {NUM_LAYERS} (increased)")
    logger.info(f"Dropout: {DROPOUT} (reduced)")
    logger.info(f"Add node scoring: {ADD_NODE_SCORING}")
    logger.info(f"K-folds: {K_FOLDS}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Final epochs: {FINAL_EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Learning rate: {LEARNING_RATE} (reduced)")
    logger.info(f"Weight decay: {WEIGHT_DECAY} (reduced)")
    logger.info(f"CyclicalLR: base={CYCLICAL_BASE_LR}, max={CYCLICAL_MAX_LR}")
    logger.info(f"Label smoothing: {LABEL_SMOOTHING} (reduced)")
    logger.info(f"Data split: {TRAIN_SPLIT:.0%}/{VALIDATION_SPLIT:.0%}/{TEST_SPLIT:.0%} (80/10/10)")
    logger.info(f"Normalization: {USE_NORMALIZATION} (essential)")
    logger.info(f"Balanced sampling: {USE_BALANCED_SAMPLING} (disabled - already balanced)")
    logger.info(f"Max samples per class: {MAX_SAMPLES_PER_CLASS}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    try:
        # K-fold cross validation
        cv_results = k_fold_cross_validation(data_list, labels, device)

        # Print comprehensive results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 CROSS-VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        logger.info(f"Precision: {cv_results['mean_precision']:.4f} ± {cv_results['std_precision']:.4f}")
        logger.info(f"Recall:    {cv_results['mean_recall']:.4f} ± {cv_results['std_recall']:.4f}")
        logger.info(f"F1 Score:  {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        logger.info(f"AUC-ROC:   {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        logger.info("=" * 60)

        # Performance assessment
        if cv_results['mean_f1'] < 0.75:
            logger.warning("⚠️ F1 score is below 0.75. Consider hyperparameter tuning.")
        elif cv_results['mean_f1'] > 0.90:
            logger.info("🌟 Outstanding discriminator performance!")
        elif cv_results['mean_f1'] > 0.82:
            logger.info("✅ Excellent discriminator performance!")
        else:
            logger.info("✅ Good discriminator performance.")

        # Consistency check
        if cv_results['std_f1'] > 0.05:
            logger.warning("⚠️ High variance across folds. Consider more data or regularization.")
        else:
            logger.info("👍 Consistent performance across folds!")

        # Train final model on all data with proper test split
        final_model, test_metrics = train_final_model(data_list, labels, device)

        # Save everything
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save final model with metadata
        model_save_path = results_dir / 'pretrained_discriminator.pt'
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'model_config': {
                'version': MODEL_VERSION,
                'node_dim': NODE_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'add_node_scoring': ADD_NODE_SCORING
            },
            'training_config': {
                'k_folds': K_FOLDS,
                'epochs': EPOCHS,
                'final_epochs': FINAL_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'cyclical_base_lr': CYCLICAL_BASE_LR,
                'cyclical_max_lr': CYCLICAL_MAX_LR,
                'label_smoothing': LABEL_SMOOTHING,
                'use_cyclical_lr': USE_CYCLICAL_LR
            },
            'cv_results': {
                'mean_f1': float(cv_results['mean_f1']),
                'std_f1': float(cv_results['std_f1']),
                'mean_accuracy': float(cv_results['mean_accuracy']),
                'mean_auc': float(cv_results['mean_auc'])
            },
            'test_results': {
                'test_f1': float(test_metrics['f1']),
                'test_accuracy': float(test_metrics['accuracy']),
                'test_precision': float(test_metrics['precision']),
                'test_recall': float(test_metrics['recall']),
                'test_auc': float(test_metrics['auc'])
            },
            'data_info': {
                'total_samples': len(data_list),
                'smelly_samples': sum(labels),
                'clean_samples': len(labels) - sum(labels),
                'node_features': NODE_DIM
            }
        }, model_save_path)

        logger.info(f"💾 Model saved to: {model_save_path}")

        # Save comprehensive results
        save_results(cv_results, results_dir)

        logger.info("\n" + "=" * 60)
        logger.info("🎉 DISCRIMINATOR PRE-TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📁 Pretrained model: {model_save_path}")
        logger.info(f"📊 Results directory: {results_dir}")
        logger.info(f"🎯 Expected F1 score: {cv_results['mean_f1']:.3f}")
        logger.info(f"🔧 Model version: {MODEL_VERSION}")
        logger.info(f"⚙️ Node features: {NODE_DIM}")
        logger.info("=" * 60)
        logger.info("🚀 Ready for RL training!")

        # Quick usage example
        logger.info("\n📝 Usage example for RL training:")
        logger.info("```python")
        logger.info("# Load pretrained discriminator")
        logger.info("checkpoint = torch.load('pretrained_discriminator.pt')")
        logger.info("model_config = checkpoint['model_config']")
        logger.info("discriminator = create_discriminator(**model_config)")
        logger.info("discriminator.load_state_dict(checkpoint['model_state_dict'])")
        logger.info("```")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()