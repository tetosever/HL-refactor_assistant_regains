# Graph Refactoring with PPO - Replication Package

This repository contains the implementation of a Proximal Policy Optimization (PPO) system for automatic graph refactoring using Graph Neural Networks and Reinforcement Learning.

## System Overview

The system consists of three main components:
1. **Data Pipeline**: Creates 1-hop ego graphs from dependency graphs for training
2. **Discriminator Pre-training**: Trains a GNN to detect hub smell patterns
3. **PPO Training**: Uses reinforcement learning to learn graph refactoring actions

## Directory Structure

```
├── data_builder/
│   └── dataset/
│       └── graph_features/        # Training data (*.pt files)
├── results/
│   ├── discriminator_pretraining/  # Pre-trained discriminator models
│   ├── ppo_training/              # PPO training outputs
│   └── evaluation/                # Evaluation results
├── logs/                          # Training logs
└── configs/                       # Configuration files
```

## Prerequisites

```bash
pip install torch torch-geometric networkx scikit-learn matplotlib seaborn
pip install pandas numpy plotly yaml
```

## Quick Start

### 1. Data Preparation

First, prepare your graph dataset using the data collection pipeline:

```bash
python data_collect.py --config config.yml --projects project1 project2
```

**Configuration**: Edit `config.yml` to specify:
- `paths.arcan_output`: Path to Arcan analysis results
- `paths.dataset_output`: Where to save processed graphs
- `projects`: List of projects to process

**Expected Input**: 
- `smell-characteristics.csv`: Component smell information
- `smell-affects.csv`: Smell relationships
- `dependency-graph-*.graphml`: Dependency graphs

**Output**: Creates `*.pt` files containing PyG Data objects with:
- Node features (7-dimensional): fan_in, fan_out, degree_centrality, in_out_ratio, pagerank, betweenness_centrality, closeness_centrality
- Edge connectivity information
- Smell labels (`is_smelly` attribute)

### 2. Discriminator Pre-training

Train the hub smell discriminator:

```bash
python pre_training_discriminator.py
```

**Key Parameters** (modify in script):
- `NODE_DIM = 7`: Number of node features
- `HIDDEN_DIM = 64`: GNN hidden dimensions
- `NUM_LAYERS = 3`: Number of GCN layers
- `K_FOLDS = 5`: Cross-validation folds
- `EPOCHS = 100`: Training epochs per fold

**Output**: Saves `pretrained_discriminator.pt` with:
- Model state dict
- Training configuration
- Cross-validation results
- Test performance metrics

### 3. PPO Training

Run the complete PPO training pipeline:

```bash
python train_main.py
```

Or use the PPO trainer directly:

```bash
python ppo_trainer.py
```

**Key Configuration** (in `PPOConfig` class):
- `num_episodes = 5000`: Total training episodes
- `max_steps = 10`: Maximum steps per episode  
- `lr = 1e-4`: Learning rate
- `reward_weights`: Dictionary controlling reward components

## Core Components

### Data Collection (`data_collect.py`)

**SmellMapBuilder**: Processes Arcan CSV files to create component→smell mappings
```python
builder = SmellMapBuilder(project_dir)
smell_map, stats = builder.build_smell_map()
```

**SubgraphExtractor**: Extracts 1-hop ego graphs around components
```python
extractor = SubgraphExtractor(config)
ego_graph = extractor.extract_ego_graph(G, center_node)
data = extractor.create_pyg_data(ego_graph, center_node, is_smelly)
```

### Environment (`rl_gym.py`)

**RefactorEnv**: Main RL environment supporting 7 actions:
- 0: RemoveEdge - Remove edge from hub
- 1: AddEdge - Add edge from hub  
- 2: MoveEdge - Combination of remove + add
- 3: ExtractMethod - Create method extraction pattern
- 4: ExtractAbstractUnit - Create abstraction layer
- 5: ExtractUnit - Split hub responsibilities
- 6: STOP - Terminate episode

**PPORefactorEnv**: PPO-compatible wrapper with growth constraints

**Key Methods**:
```python
env = PPORefactorEnv(data_path="data/", max_steps=10)
state = env.reset()  # Returns Data object
next_state, reward, done, info = env.step(action)
```

### Actor-Critic Models (`actor_critic_models.py`)

**ActorCritic**: Combined model with shared encoder for PPO
```python
config = {
    'node_dim': 7,
    'hidden_dim': 256, 
    'num_layers': 3,
    'num_actions': 7,
    'global_features_dim': 4,
    'shared_encoder': True
}
model = create_actor_critic(config)
```

**Key Methods**:
```python
output = model(data, global_features)
action, log_prob, value = model.get_action_and_value(data, global_features)
log_probs, values, entropy = model.evaluate_actions(data, global_features, actions)
```

### Discriminator (`discriminator.py`)

**HubDiscriminator**: GNN for detecting hub smell patterns
```python
discriminator = create_discriminator(
    version="clean",
    node_dim=7,
    hidden_dim=64,
    num_layers=3,
    add_node_scoring=True
)
```

**Usage**:
```python
output = discriminator(data)
classification = discriminator.classify_graph(data)
node_analysis = discriminator.get_node_hub_analysis(data)
```

### PPO Trainer (`ppo_trainer.py`)

**PPOConfig**: Single source of truth for all parameters
```python
config = PPOConfig()
config.num_episodes = 5000
config.lr = 1e-4
config.max_steps = 20
```

**PPOTrainer**: Main training loop
```python
trainer = PPOTrainer(config)
training_stats = trainer.train()
```

## Training Pipeline

The complete pipeline (`train_main.py`) executes:

1. **Setup**: Creates directories, loads configuration
2. **Data Validation**: Checks dataset availability  
3. **Discriminator Pre-training**: Trains hub smell detector
4. **PPO Training**: Learns refactoring policy
5. **Evaluation**: Tests trained model performance

## Reward System

The reward function combines multiple components:

```python
reward_weights = {
    'hub_weight': 5.0,           # Main objective: hub score improvement
    'step_valid': 0.01,          # Valid action bonus
    'step_invalid': -0.1,        # Invalid action penalty
    'time_penalty': -0.02,       # Time step penalty
    'adversarial_weight': 0.5,   # Discriminator-based reward
    'success_threshold': 0.03,   # Success detection threshold
    'success_bonus': 2.0,        # Bonus for achieving success
    'node_penalty': 1.0,         # Growth control penalty
    'edge_penalty': 0.02,        # Edge growth penalty
    'cap_exceeded_penalty': -0.8 # Hard limit violation
}
```

## Growth Control

The system implements growth constraints to prevent unrealistic graph expansion:

- `max_new_nodes`: Maximum nodes added per episode (default: 5)
- `max_growth`: Maximum growth ratio (default: 1.3x original size)
- Terminal penalties for excessive growth
- Action masking when at capacity

## Evaluation

Run comprehensive evaluation:

```bash
python final_evaluation.py
```

**Outputs**:
- Performance metrics (success rate, rewards, improvements)
- Graph visualizations (before/after comparisons)
- Hub tracking analysis
- Discriminator score changes

**Key Metrics**:
- Success Rate: Episodes with significant hub improvement
- Hub Score Improvement: Reduction in hub smell intensity
- Discriminator Improvement: Reduction in smell probability
- Episode Efficiency: Steps to achieve improvements

## Model Checkpoints

**Discriminator**: `results/discriminator_pretraining/pretrained_discriminator.pt`
- Contains model weights, configuration, and validation metrics

**PPO Model**: `results/ppo_training/best_model.pt`  
- Contains actor-critic weights, training history, and performance stats

## Configuration

Key configuration files:

**`config.yml`**: Data processing configuration
```yaml
paths:
  arcan_output: "./arcan_out/intervalDays2"
  dataset_output: "./dataset/graph_features"

extraction:
  max_graph_size: 10000
  min_subgraph_size: 3
  max_subgraph_size: 200
```

**`PPOConfig`**: All training parameters in single dataclass
```python
@dataclass
class PPOConfig:
    num_episodes: int = 5000
    max_steps: int = 10
    lr: float = 1e-4
    # ... many more parameters
```

## Troubleshooting

**Data Issues**:
- Ensure `*.pt` files contain PyG Data objects with 7-dimensional node features
- Check that `is_smelly` attribute exists for labels
- Verify graph connectivity and valid edge indices

**Training Issues**:
- Monitor for NaN values in loss functions
- Check GPU memory usage for large graphs
- Verify discriminator loading if using adversarial training

**Evaluation Issues**:
- Ensure model checkpoints exist before evaluation
- Check data path consistency across components
- Verify hub tracking doesn't lose target nodes

## Paper Replication

To replicate paper results:

1. Use the same Arcan output structure with CSV files
2. Run data collection with provided configuration
3. Pre-train discriminator with 5-fold cross-validation  
4. Train PPO with default PPOConfig parameters
5. Evaluate using provided metrics and visualization scripts

Expected performance:
- Success Rate: >90%
- Average Hub Improvement: >0.40
- Average Reward: >3.2
- Average Episode Length: ≤3 steps
