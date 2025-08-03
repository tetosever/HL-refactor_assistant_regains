#!/usr/bin/env python3
"""
Simplified Data Pipeline with Smart Skipping

Maintains the original data/ folder structure with .pt files
but adds intelligent skipping when files haven't changed.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from comp2smelly import extract_smell_maps
from extract_subgraphs import load_feature_config, collect_dataset_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration paths (same as before)
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
ARCAN_DIR = ROOT / "arcan_out" / "intervalDays7"
OUTPUT_DIR = ROOT / "dataset_builder" / "data"
OUTPUT_DIR_SUBGRAPH = OUTPUT_DIR / "dataset_graph_feature"  # Your .pt files go here
PIPELINE_STATS_FILE = OUTPUT_DIR / "pipeline_stats.json"


def get_project_source_hash(project_dir: Path) -> str:
    """Generate hash of all source files for change detection"""

    source_patterns = [
        "dependency-graph-*.graphml",
        "smell-characteristics.csv",
        "smell-affects.csv"
    ]

    file_info = []
    for pattern in source_patterns:
        for file_path in sorted(project_dir.glob(pattern)):
            if file_path.exists():
                stat = file_path.stat()
                file_info.append(f"{file_path.name}:{stat.st_mtime}:{stat.st_size}")

    combined = "|".join(file_info)
    return hashlib.md5(combined.encode()).hexdigest()


def load_processing_history() -> Dict:
    """Load history of processed projects"""
    history_file = OUTPUT_DIR / "processing_history.json"

    if history_file.exists():
        try:
            return json.loads(history_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to load processing history: {e}")

    return {}


def save_processing_history(history: Dict):
    """Save history of processed projects"""
    history_file = OUTPUT_DIR / "processing_history.json"

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save processing history: {e}")


def should_skip_project(project: str, arcan_dir: Path, output_dir: Path,
                        force_refresh: bool = False) -> Tuple[bool, str]:
    """Determine if project should be skipped based on existing files and changes"""

    if force_refresh:
        return False, "Force refresh requested"

    project_dir = arcan_dir / project

    # Check if source directory exists
    if not project_dir.exists():
        return True, f"Project directory not found: {project_dir}"

    # Check if required files exist
    required_files = [
        "smell-characteristics.csv",
        "smell-affects.csv"
    ]

    for req_file in required_files:
        if not (project_dir / req_file).exists():
            return True, f"Missing required file: {req_file}"

    # Check if any graphml files exist
    graphml_files = list(project_dir.glob("dependency-graph-*.graphml"))
    if not graphml_files:
        return True, "No graphml files found"

    # Check if .pt files exist for this project
    existing_pt_files = list(output_dir.glob(f"{project}_*.pt"))
    if not existing_pt_files:
        return False, "No existing .pt files found, need to process"

    # Check if source files have changed since last processing
    history = load_processing_history()
    current_hash = get_project_source_hash(project_dir)

    if project in history:
        last_hash = history[project].get('source_hash', '')
        if current_hash == last_hash:
            return True, f"Files unchanged, {len(existing_pt_files)} .pt files exist"

    return False, "Source files changed, need to reprocess"


def compute_graph_features_optimized(G: nx.Graph, center_node: str,
                                     is_smelly: bool, config: dict) -> Optional[Data]:
    """Compute features directly without depending on Arcan original features"""

    try:
        # Validate center node exists
        if center_node not in G:
            return None

        # Extract ego graph
        ego_graph = nx.ego_graph(G, center_node, radius=config.get('radius', 1), undirected=False)

        # Size checks
        if len(ego_graph) < config.get('min_subgraph_size', 3):
            return None

        if ego_graph.number_of_edges() < config.get('min_edges', 1):
            return None

        # Remove isolated nodes if configured
        if config.get('remove_isolated_nodes', True):
            isolated = list(nx.isolates(ego_graph))
            if isolated:
                ego_graph.remove_nodes_from(isolated)
                if len(ego_graph) < config.get('min_subgraph_size', 3):
                    return None

        # Compute all topological metrics
        in_degrees = dict(ego_graph.in_degree())
        out_degrees = dict(ego_graph.out_degree())

        # Compute centrality metrics efficiently
        if len(ego_graph) <= 50:
            try:
                pagerank = nx.pagerank(ego_graph, alpha=0.85, max_iter=100)
                betweenness = nx.betweenness_centrality(ego_graph)
                closeness = nx.closeness_centrality(ego_graph)
            except:
                # Fallback for problematic graphs
                pagerank = {n: 1.0 / len(ego_graph) for n in ego_graph.nodes()}
                betweenness = {n: 0.0 for n in ego_graph.nodes()}
                closeness = {n: 1.0 for n in ego_graph.nodes()}
        else:
            # Simplified metrics for large graphs
            pagerank = {n: 1.0 / len(ego_graph) for n in ego_graph.nodes()}
            betweenness = {n: 0.0 for n in ego_graph.nodes()}
            closeness = {n: 1.0 for n in ego_graph.nodes()}

        # Prepare node features (clear original and add computed ones)
        for node, attrs in ego_graph.nodes(data=True):
            attrs.clear()  # Remove all original Arcan features

            # Add our computed features
            in_deg = float(in_degrees.get(node, 0))
            out_deg = float(out_degrees.get(node, 0))
            total_deg = in_deg + out_deg

            attrs.update({
                'in_degree': in_deg,
                'out_degree': out_deg,
                'total_degree': total_deg,
                'in_out_ratio': in_deg / (out_deg + 1e-8),
                'degree_centrality': total_deg / (len(ego_graph) - 1 + 1e-8),
                'pagerank': float(pagerank.get(node, 0)),
                'betweenness': float(betweenness.get(node, 0)),
                'closeness': float(closeness.get(node, 0)),
                'hub_score': float(pagerank.get(node, 0)) * 0.6 + float(betweenness.get(node, 0)) * 0.4,
                # Add some additional useful features
                'clustering': 0.0,  # Can be computed if needed
                'eigenvector_centrality': float(pagerank.get(node, 0))  # Approximation
            })

        # Prepare edge features (simplified)
        for u, v, attrs in ego_graph.edges(data=True):
            attrs.clear()  # Remove original features
            attrs['weight'] = 1.0  # Simple uniform weight

        # Convert to PyG Data
        node_attrs = ['in_degree', 'out_degree', 'total_degree', 'in_out_ratio',
                      'degree_centrality', 'pagerank', 'betweenness', 'closeness',
                      'hub_score', 'clustering', 'eigenvector_centrality']
        edge_attrs = ['weight']

        data = from_networkx(ego_graph, group_node_attrs=node_attrs, group_edge_attrs=edge_attrs)

        # Add metadata
        data.is_smelly = torch.tensor([int(is_smelly)], dtype=torch.long)
        data.center_node = center_node
        data.subgraph_size = len(ego_graph)
        data.num_edges_sg = ego_graph.number_of_edges()

        return data

    except Exception as e:
        logger.warning(f"Failed to compute features for {center_node}: {e}")
        return None


def process_project_optimized(project: str, arcan_dir: Path, output_dir: Path,
                              feature_config: dict, force_refresh: bool = False) -> Dict:
    """Process project with optimized feature computation"""

    start_time = time.time()
    project_stats = {
        'project': project,
        'start_time': time.time(),
        'success': False,
        'error': None,
        'subgraph_stats': {},
        'processing_time': 0,
        'files_processed': 0,
        'skipped': False
    }

    try:
        logger.info(f"=== Processing project: {project} ===")

        # Check if we should skip this project
        should_skip, reason = should_skip_project(project, arcan_dir, output_dir, force_refresh)

        if should_skip:
            project_stats['skipped'] = True
            project_stats['success'] = True
            logger.info(f"⚡ Skipping {project}: {reason}")
            return project_stats

        logger.info(f"🔄 Processing {project}: {reason}")

        # Extract smell maps
        logger.info(f"Extracting smell maps for {project}")
        smell_map = extract_smell_maps(project, arcan_dir)

        if not smell_map:
            project_stats['error'] = "No smell map generated"
            return project_stats

        # Process graphml files
        project_dir = arcan_dir / project
        graphml_files = list(project_dir.glob("dependency-graph-*.graphml"))

        processed_count = 0

        for graphml_file in graphml_files:
            try:
                # Load graph
                G = nx.read_graphml(graphml_file)

                # Extract version info from filename
                import re
                match = re.search(r"dependency-graph-(\d+)_([0-9a-fA-F]+)\.graphml", graphml_file.name)
                if not match:
                    continue

                version_idx, version_hash = match.groups()

                # Check graph size
                if len(G) > feature_config.get('max_graph_size', 10000):
                    logger.warning(f"Skipping {graphml_file.name}: graph too large ({len(G)} nodes)")
                    continue

                # Process each component in the smell map
                for comp_id, smelly_versions in smell_map.items():
                    is_smelly = version_hash in smelly_versions

                    # Skip non-smelly if not configured to include them
                    if not is_smelly and not feature_config.get('include_non_smelly', False):
                        continue

                    # Compute features directly
                    data = compute_graph_features_optimized(G, comp_id, is_smelly, feature_config)

                    if data is None:
                        continue

                    # Add additional metadata
                    data.project_name = project
                    data.component_id = comp_id
                    data.version_hash = version_hash
                    data.version_idx = int(version_idx)
                    data.extraction_timestamp = torch.tensor([int(time.time())], dtype=torch.long)

                    # Generate filename
                    filename = f"{project}_{comp_id}_{processed_count:04d}_{version_idx}_{version_hash}.pt"
                    output_path = output_dir / filename

                    # Save .pt file
                    torch.save(data, output_path)
                    processed_count += 1

                logger.debug(f"Processed {graphml_file.name}: extracted features for components")

            except Exception as e:
                logger.warning(f"Failed to process {graphml_file}: {e}")
                continue

        # Update processing history
        history = load_processing_history()
        history[project] = {
            'source_hash': get_project_source_hash(project_dir),
            'last_processed': time.time(),
            'files_processed': processed_count,
            'graphml_files': len(graphml_files)
        }
        save_processing_history(history)

        project_stats['files_processed'] = processed_count
        project_stats['subgraph_stats'] = {
            'files_generated': processed_count,
            'graphml_processed': len(graphml_files)
        }
        project_stats['success'] = True

        logger.info(f"✅ Successfully processed {project}: {processed_count} .pt files generated")

    except Exception as e:
        project_stats['error'] = str(e)
        logger.error(f"❌ Error processing project {project}: {e}")

    finally:
        project_stats['processing_time'] = time.time() - start_time
        logger.info(f"Project {project} processed in {project_stats['processing_time']:.2f} seconds")

    return project_stats


def run_simplified_pipeline(force_refresh_all: bool = False,
                            force_refresh_projects: List[str] = None):
    """Run simplified pipeline that saves .pt files in data/ folder"""

    pipeline_start = time.time()
    pipeline_stats = {
        'pipeline_start': time.time(),
        'total_projects': 0,
        'successful_projects': 0,
        'skipped_projects': 0,
        'failed_projects': 0,
        'project_details': [],
        'dataset_stats': {},
        'total_time': 0,
        'errors': []
    }

    try:
        # Load configuration
        logger.info("Loading configuration...")
        projects = yaml.safe_load(CONFIG_PATH.read_text())['projects']
        feature_config = load_feature_config(CONFIG_PATH)

        pipeline_stats['total_projects'] = len(projects)
        logger.info(f"Found {len(projects)} projects to process")

        # Create output directory
        OUTPUT_DIR_SUBGRAPH.mkdir(parents=True, exist_ok=True)

        # Process each project
        for project in projects:
            force_refresh = force_refresh_all or (force_refresh_projects and project in force_refresh_projects)

            project_stats = process_project_optimized(
                project, ARCAN_DIR, OUTPUT_DIR_SUBGRAPH, feature_config, force_refresh
            )

            pipeline_stats['project_details'].append(project_stats)

            if project_stats['success']:
                pipeline_stats['successful_projects'] += 1
                if project_stats['skipped']:
                    pipeline_stats['skipped_projects'] += 1
            else:
                pipeline_stats['failed_projects'] += 1
                pipeline_stats['errors'].append({
                    'project': project,
                    'error': project_stats['error']
                })

        logger.info("✅ Processing completed.")

        # Collect final dataset statistics
        logger.info("Collecting final dataset statistics...")
        pipeline_stats['dataset_stats'] = collect_dataset_stats(OUTPUT_DIR_SUBGRAPH)

        logger.info("📊 Final Dataset Statistics:")
        logger.info(f"  - Total .pt files: {pipeline_stats['dataset_stats']['total_subgraphs']}")
        logger.info(f"  - Projects: {len(pipeline_stats['dataset_stats']['projects'])}")
        logger.info(f"  - Unique components: {len(pipeline_stats['dataset_stats']['components'])}")
        logger.info(f"  - Smelly samples: {pipeline_stats['dataset_stats'].get('smelly_count', 'N/A')}")
        logger.info(f"  - Clean samples: {pipeline_stats['dataset_stats'].get('non_smelly_count', 'N/A')}")

    except Exception as e:
        error_msg = f"Pipeline critical failure: {e}"
        logger.error(f"❌ {error_msg}")
        pipeline_stats['errors'].append({
            'component': 'pipeline',
            'error': error_msg
        })
        raise

    finally:
        # Final statistics
        pipeline_stats['total_time'] = time.time() - pipeline_start

        # Save statistics
        with open(PIPELINE_STATS_FILE, 'w') as f:
            json.dump(pipeline_stats, f, indent=2, default=str)

        # Log summary
        logger.info("🏁 Pipeline finished.")
        logger.info(
            f"Results: {pipeline_stats['successful_projects']}/{pipeline_stats['total_projects']} projects successful")
        logger.info(f"Skipped: {pipeline_stats['skipped_projects']} (unchanged)")
        logger.info(f"Total time: {pipeline_stats['total_time']:.2f} seconds")

        if pipeline_stats['errors']:
            logger.warning(f"Encountered {len(pipeline_stats['errors'])} errors during processing")


def clean_project_data(project: str):
    """Remove all .pt files for a specific project"""
    files_removed = 0
    for pt_file in OUTPUT_DIR_SUBGRAPH.glob(f"{project}_*.pt"):
        pt_file.unlink()
        files_removed += 1

    # Remove from processing history
    history = load_processing_history()
    if project in history:
        del history[project]
        save_processing_history(history)

    logger.info(f"🗑️  Removed {files_removed} .pt files for {project}")


def show_data_info():
    """Show information about existing data"""
    if not OUTPUT_DIR_SUBGRAPH.exists():
        logger.info("No data directory found")
        return

    stats = collect_dataset_stats(OUTPUT_DIR_SUBGRAPH)
    history = load_processing_history()

    logger.info("📊 Current Data Status:")
    logger.info(f"  Total .pt files: {stats['total_subgraphs']}")
    logger.info(f"  Projects: {stats['projects']}")
    logger.info(f"  Smelly samples: {stats.get('smelly_count', 0)}")
    logger.info(f"  Clean samples: {stats.get('non_smelly_count', 0)}")

    logger.info("\n📂 Processing History:")
    for project, info in history.items():
        last_time = pd.Timestamp(info['last_processed'], unit='s').strftime('%Y-%m-%d %H:%M')
        logger.info(f"  {project}: {info['files_processed']} files (last: {last_time})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run simplified data pipeline")
    parser.add_argument("--force-refresh-all", action="store_true",
                        help="Force refresh all projects (ignore cache)")
    parser.add_argument("--force-refresh", nargs='+',
                        help="Force refresh specific projects")
    parser.add_argument("--clean", help="Remove all .pt files for specific project")
    parser.add_argument("--info", action="store_true",
                        help="Show information about existing data")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        if args.clean:
            clean_project_data(args.clean)
        elif args.info:
            show_data_info()
        else:
            run_simplified_pipeline(
                force_refresh_all=args.force_refresh_all,
                force_refresh_projects=args.force_refresh
            )

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)