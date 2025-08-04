#!/usr/bin/env python3
"""
LOGICA CORRETTA: Dynamic Component Detection

Il problema era che il codice:
1. Predeterminava i center_nodes dai characteristics CSV
2. Li cercava nei GraphML
3. Ma dovrebbe fare l'OPPOSTO:
   - Per ogni GraphML, trova TUTTE le componenti
   - Per ogni componente, verifica se versionId è in smell_map[component]
   - Etichetta di conseguenza
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx
import pandas as pd
import torch
import json

from graph_feature_extraction import create_hub_focused_data, validate_extracted_features

logger = logging.getLogger(__name__)


def extract_features_with_correct_logic(
        graphml_path: Path,
        smell_map: Dict[str, List[str]],
        config: dict
) -> List:
    """
    🚨 LOGICA CORRETTA:
    1. Carica GraphML e estrae versionId dal nome file
    2. Per OGNI nodo nel grafo, verifica se è nella smell_map
    3. Se è nella smell_map, controlla se versionId è nelle sue versioni smelly
    4. Etichetta di conseguenza
    """
    extracted_data = []

    # Estrai versionId dal nome file
    match = re.search(r"dependency-graph-(\d+)_([0-9a-fA-F]+)\.graphml", graphml_path.name)
    if not match:
        logger.warning(f"Cannot parse version from {graphml_path.name}")
        return extracted_data

    version_index, version_id = match.groups()

    logger.debug(f"Processing {graphml_path.name}: versionId = {version_id}")

    try:
        # Carica il grafo
        G = nx.read_graphml(graphml_path)
        logger.debug(f"Loaded graph: {len(G)} nodes, {G.number_of_edges()} edges")

        # Check graph size limits
        max_graph_size = config.get('max_graph_size', 10000)
        if len(G) > max_graph_size:
            logger.warning(f"Graph too large ({len(G)} nodes), skipping")
            return extracted_data

        # 🚨 LOGICA CORRETTA: Per ogni nodo nel grafo
        all_nodes = [str(node_id) for node_id in G.nodes()]
        logger.debug(f"Graph contains {len(all_nodes)} nodes")

        smelly_processed = 0
        clean_processed = 0
        components_in_smell_map = 0

        # Statistiche per debug
        nodes_checked = 0
        nodes_in_smell_map = 0

        for node_id in all_nodes:
            nodes_checked += 1

            # 🚨 STEP 1: Verifica se questo nodo è nella smell_map
            if node_id not in smell_map:
                # Questo nodo non è mai stato rilevato come smell in nessuna versione
                # Potremmo volerlo includere come campione clean, ma per ora skippiamo
                # per ridurre il rumore (troppi nodi non-hub)
                continue

            nodes_in_smell_map += 1
            components_in_smell_map += 1

            # 🚨 STEP 2: Questo nodo è nella smell_map, verifica se è smelly in questa versione
            smelly_versions = smell_map[node_id]
            is_smelly = version_id in smelly_versions

            logger.debug(
                f"  Node {node_id}: {len(smelly_versions)} smelly versions, current={version_id}, is_smelly={is_smelly}")

            # 🚨 STEP 3: Decidi se processare questo nodo
            should_process = False

            if is_smelly:
                # SEMPRE processa nodi smelly
                should_process = True
                logger.debug(f"    → Processing SMELLY node {node_id}")
            elif config.get('include_non_smelly', False):
                # Processa nodi clean solo se configurato
                should_process = True
                logger.debug(f"    → Processing CLEAN node {node_id}")
            else:
                logger.debug(f"    → Skipping CLEAN node {node_id} (not configured)")

            if not should_process:
                continue

            # 🚨 STEP 4: Estrai features per questo nodo come centro
            data = create_hub_focused_data(G, node_id, is_smelly, config)

            if data is not None and validate_extracted_features(data):
                # Add metadata
                data.version_hash = version_id
                data.version_index = int(version_index)
                data.graphml_path = str(graphml_path)
                data.smelly_version_count = len(smelly_versions)

                extracted_data.append(data)

                if is_smelly:
                    smelly_processed += 1
                    logger.debug(f"    ✅ Extracted SMELLY subgraph for {node_id}")
                else:
                    clean_processed += 1
                    logger.debug(f"    ✅ Extracted CLEAN subgraph for {node_id}")
            else:
                logger.debug(f"    ❌ Failed to extract/validate subgraph for {node_id}")

        # Log statistiche per questo GraphML
        logger.info(f"GraphML {graphml_path.name} processed:")
        logger.info(f"  - Total nodes in graph: {len(all_nodes)}")
        logger.info(f"  - Nodes in smell_map: {nodes_in_smell_map}")
        logger.info(f"  - Smelly extracted: {smelly_processed}")
        logger.info(f"  - Clean extracted: {clean_processed}")
        logger.info(f"  - Total extracted: {len(extracted_data)}")

    except Exception as e:
        logger.error(f"Failed to process {graphml_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return extracted_data


def process_project_with_correct_logic(project: str, arcan_dir: Path, output_dir: Path,
                                       config: dict, force_refresh: bool = False) -> Dict:
    """
    🚨 VERSIONE CORRETTA che usa la logica giusta
    """
    start_time = time.time()
    project_stats = {
        'project': project,
        'start_time': time.time(),
        'success': False,
        'error': None,
        'processing_time': 0,
        'files_processed': 0,
        'smelly_count': 0,
        'clean_count': 0,
        'total_extracted': 0,
        'graphml_processed': 0
    }

    try:
        logger.info(f"=== Processing project: {project} (CORRECT LOGIC) ===")

        project_dir = arcan_dir / project

        # 1. Carica smell_map
        map_path = project_dir / "smell_map.json"
        if not map_path.exists():
            project_stats['error'] = f"smell_map.json not found: {map_path}"
            return project_stats

        try:
            with open(map_path, 'r') as f:
                smell_map = json.load(f)
        except Exception as e:
            project_stats['error'] = f"Failed to load smell_map.json: {e}"
            return project_stats

        logger.info(f"Loaded smell_map: {len(smell_map)} components")

        # Debug smell_map
        if smell_map:
            sample_comp = next(iter(smell_map.keys()))
            sample_versions = smell_map[sample_comp]
            logger.info(f"Sample component {sample_comp}: {len(sample_versions)} smelly versions")
            logger.debug(f"  First 5 versions: {sample_versions[:5]}")

        # 2. Trova tutti i GraphML files
        graphml_files = list(project_dir.glob("dependency-graph-*.graphml"))
        if not graphml_files:
            project_stats['error'] = "No GraphML files found"
            return project_stats

        logger.info(f"Found {len(graphml_files)} GraphML files to process")

        # 3. Processa ogni GraphML con la logica corretta
        total_smelly = 0
        total_clean = 0
        files_processed = 0

        for graphml_file in graphml_files:
            try:
                logger.debug(f"\n--- Processing {graphml_file.name} ---")

                # 🚨 USA LA LOGICA CORRETTA
                extracted_data = extract_features_with_correct_logic(
                    graphml_path=graphml_file,
                    smell_map=smell_map,
                    config=config
                )

                # Salva i dati estratti
                for i, data in enumerate(extracted_data):
                    # Extract version info for filename
                    match = re.search(r"dependency-graph-(\d+)_([0-9a-fA-F]+)\.graphml", graphml_file.name)
                    if match:
                        version_idx, version_id = match.groups()
                    else:
                        continue

                    # Generate filename
                    filename = f"{project}_{data.center_node}_{files_processed:04d}_{version_idx}_{version_id[:8]}.pt"
                    output_path = output_dir / filename

                    # Add additional metadata
                    data.project_name = project
                    data.component_id = data.center_node
                    data.extraction_timestamp = torch.tensor([int(time.time())], dtype=torch.long)

                    # Count and save
                    if hasattr(data, 'is_smelly') and data.is_smelly.item():
                        total_smelly += 1
                    else:
                        total_clean += 1

                    torch.save(data, output_path)
                    files_processed += 1

                project_stats['graphml_processed'] += 1
                logger.debug(f"Completed {graphml_file.name}: {len(extracted_data)} subgraphs extracted")

            except Exception as e:
                logger.warning(f"Failed to process {graphml_file}: {e}")
                continue

        # Final statistics
        project_stats.update({
            'success': True,
            'files_processed': files_processed,
            'smelly_count': total_smelly,
            'clean_count': total_clean,
            'total_extracted': files_processed,
            'processing_time': time.time() - start_time
        })

        logger.info(f"✅ Project {project} completed:")
        logger.info(f"   - GraphML files processed: {project_stats['graphml_processed']}")
        logger.info(f"   - Total subgraphs: {files_processed}")
        logger.info(f"   - Smelly samples: {total_smelly}")
        logger.info(f"   - Clean samples: {total_clean}")
        logger.info(f"   - Processing time: {project_stats['processing_time']:.2f}s")

        # 🚨 VERIFICA CRITICA
        if total_smelly == 0:
            logger.warning(f"⚠️  Still no smelly samples for {project}!")
            logger.warning("This suggests a deeper issue in the smell_map or logic.")
        else:
            logger.info(f"🎉 SUCCESS: Found {total_smelly} smelly samples!")

    except Exception as e:
        project_stats['error'] = str(e)
        logger.error(f"❌ Error processing project {project}: {e}")

    finally:
        project_stats['processing_time'] = time.time() - start_time

    return project_stats


def debug_smell_map_consistency(project: str, arcan_dir: Path):
    """
    🔍 Debug per verificare consistenza della smell_map
    """
    logger.info(f"🔍 Debugging smell_map consistency for {project}")

    project_dir = arcan_dir / project

    # 1. Load smell_map
    map_path = project_dir / "smell_map.json"
    if not map_path.exists():
        logger.error(f"❌ smell_map.json not found: {map_path}")
        return

    smell_map = json.loads(map_path.read_text())
    logger.info(f"✅ Smell map loaded: {len(smell_map)} components")

    # 2. Analizza GraphML files
    graphml_files = list(project_dir.glob("dependency-graph-*.graphml"))
    logger.info(f"✅ Found {len(graphml_files)} GraphML files")

    if not graphml_files:
        logger.error("❌ No GraphML files found!")
        return

    # 3. Test con un GraphML file
    test_file = graphml_files[0]
    logger.info(f"🧪 Testing with {test_file.name}")

    # Estrai versionId
    match = re.search(r"dependency-graph-(\d+)_([0-9a-fA-F]+)\.graphml", test_file.name)
    if not match:
        logger.error(f"❌ Cannot parse version from {test_file.name}")
        return

    version_index, version_id = match.groups()
    logger.info(f"   Version: index={version_index}, id={version_id}")

    # 4. Carica grafo e controlla nodi
    try:
        G = nx.read_graphml(test_file)
        graph_nodes = [str(n) for n in G.nodes()]
        logger.info(f"   Graph nodes: {len(graph_nodes)}")

        # 5. Verifica overlap con smell_map
        smell_map_components = set(smell_map.keys())
        graph_node_set = set(graph_nodes)
        overlap = smell_map_components.intersection(graph_node_set)

        logger.info(f"   Smell map components: {len(smell_map_components)}")
        logger.info(f"   Node overlap: {len(overlap)}")

        if len(overlap) == 0:
            logger.error("❌ NO OVERLAP between graph nodes and smell_map components!")
            logger.error("   This means the component IDs don't match.")
            logger.error(f"   Sample graph nodes: {graph_nodes[:5]}")
            logger.error(f"   Sample smell components: {list(smell_map_components)[:5]}")
        else:
            logger.info(f"✅ Found overlap: {len(overlap)} components")

            # 6. Test specifico per version matching
            smelly_in_this_version = 0
            clean_in_this_version = 0

            for comp_id in overlap:
                smelly_versions = smell_map[comp_id]
                is_smelly = version_id in smelly_versions

                if is_smelly:
                    smelly_in_this_version += 1
                else:
                    clean_in_this_version += 1

            logger.info(f"   In version {version_id}:")
            logger.info(f"     Smelly components: {smelly_in_this_version}")
            logger.info(f"     Clean components: {clean_in_this_version}")

            if smelly_in_this_version == 0:
                logger.warning("⚠️  No smelly components in this version")
                logger.warning("   Try with a different GraphML file or check smell_map")
            else:
                logger.info("🎉 Found smelly components! Logic should work.")

    except Exception as e:
        logger.error(f"❌ Failed to load/analyze graph: {e}")


def run_correct_logic_pipeline(projects: List[str], arcan_dir: Path, output_dir: Path, config: dict):
    """
    🚨 PIPELINE PRINCIPALE con logica corretta
    """
    pipeline_stats = {
        'total_projects': len(projects),
        'successful_projects': 0,
        'failed_projects': 0,
        'total_smelly_samples': 0,
        'total_clean_samples': 0,
        'project_details': []
    }

    logger.info(f"🚀 Starting CORRECT LOGIC pipeline for {len(projects)} projects")

    for project in projects:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing project: {project}")
        logger.info(f"{'=' * 50}")

        project_stats = process_project_with_correct_logic(
            project, arcan_dir, output_dir, config
        )

        pipeline_stats['project_details'].append(project_stats)

        if project_stats['success']:
            pipeline_stats['successful_projects'] += 1
            pipeline_stats['total_smelly_samples'] += project_stats['smelly_count']
            pipeline_stats['total_clean_samples'] += project_stats['clean_count']
        else:
            pipeline_stats['failed_projects'] += 1
            logger.error(f"❌ Project {project} failed: {project_stats['error']}")

    # Final summary
    logger.info(f"\n🏁 PIPELINE COMPLETED")
    logger.info(f"   Successful projects: {pipeline_stats['successful_projects']}/{pipeline_stats['total_projects']}")
    logger.info(f"   Total smelly samples: {pipeline_stats['total_smelly_samples']}")
    logger.info(f"   Total clean samples: {pipeline_stats['total_clean_samples']}")
    logger.info(f"   Total samples: {pipeline_stats['total_smelly_samples'] + pipeline_stats['total_clean_samples']}")

    if pipeline_stats['total_smelly_samples'] == 0:
        logger.error("❌ CRITICAL: Still no smelly samples found!")
        logger.error("   Run debug_smell_map_consistency() on individual projects")
    else:
        logger.info("🎉 SUCCESS: Found smelly samples with correct logic!")

    return pipeline_stats


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run CORRECT LOGIC pipeline")
    parser.add_argument("--debug-project", help="Debug specific project")
    parser.add_argument("--test-project", help="Test single project with correct logic")
    parser.add_argument("--run-all", action="store_true", help="Run all projects")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Paths
    ROOT = Path(__file__).resolve().parent.parent
    ARCAN_DIR = ROOT / "arcan_out" / "intervalDays2"
    OUTPUT_DIR = ROOT / "dataset_builder" / "data" / "dataset_graph_feature"
    CONFIG_PATH = ROOT / "config.yaml"

    # Config
    config = {
        'min_subgraph_size': 3,
        'max_subgraph_size': 200,
        'max_graph_size': 10000,
        'radius': 1,
        'remove_isolated_nodes': True,
        'min_edges': 1,
        'include_non_smelly': True,  # 🚨 IMPORTANTE: True per avere campioni bilanciati
        'validate_features': True
    }

    try:
        if args.debug_project:
            debug_smell_map_consistency(args.debug_project, ARCAN_DIR)

        elif args.test_project:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stats = process_project_with_correct_logic(
                args.test_project, ARCAN_DIR, OUTPUT_DIR, config
            )
            print(f"\nTest completed: {stats}")

        elif args.run_all:
            # Load projects from config
            projects = yaml.safe_load(CONFIG_PATH.read_text())['projects']
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            pipeline_stats = run_correct_logic_pipeline(
                projects, ARCAN_DIR, OUTPUT_DIR, config
            )

            # Save stats
            stats_file = OUTPUT_DIR.parent / "correct_logic_pipeline_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(pipeline_stats, f, indent=2)
            print(f"Pipeline stats saved to: {stats_file}")
        else:
            print("Use --debug-project, --test-project, or --run-all")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        logger.error(traceback.format_exc())