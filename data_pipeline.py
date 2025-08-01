#!/usr/bin/env python3
"""
Pipeline ottimizzata per la generazione del dataset con:
- Gestione errori robusta
- Statistiche dettagliate
- Bilanciamento del dataset
- Validazione della qualità
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import yaml

from comp2smelly import extract_smell_maps
from extract_sequence_of_actions import build_expert_sequences
from extract_subgraphs import extract_graph_features, load_feature_config, collect_dataset_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurazione percorsi
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "dataset_builder" / "config.yaml"
ARCAN_DIR = ROOT / "arcan_out" / "intervalDays2"
OUTPUT_DIR = ROOT / "dataset_builder" / "data"
OUTPUT_DIR_SUBGRAPH = OUTPUT_DIR / "dataset_graph_feature"
SEQS_OUT = OUTPUT_DIR / "expert_sequences.pt"
PIPELINE_STATS_FILE = OUTPUT_DIR / "pipeline_stats.json"


def load_projects_name(path: Path, key: str) -> List[str]:
    """Carica lista progetti con validazione"""
    try:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        val = cfg.get(key, [])
        projects = val if isinstance(val, list) else [val]

        # Filtra progetti vuoti
        projects = [p for p in projects if p and isinstance(p, str)]

        if not projects:
            raise ValueError(f"No valid projects found in config key '{key}'")

        return projects

    except Exception as e:
        logger.error(f"Failed to load projects from {path}: {e}")
        raise


def validate_project_structure(project: str, arcan_dir: Path) -> tuple[bool, List[str]]:
    """Valida la struttura del progetto prima del processing"""
    project_dir = arcan_dir / project
    errors = []

    if not project_dir.exists():
        errors.append(f"Project directory not found: {project_dir}")
        return False, errors

    # Controlla file essenziali
    required_patterns = [
        "dependency-graph-*.graphml",
        "smell-characteristics.csv",
        "smell-affects.csv"
    ]

    for pattern in required_patterns:
        files = list(project_dir.glob(pattern))
        if not files:
            errors.append(f"No files matching pattern: {pattern}")

    return len(errors) == 0, errors


def check_dataset_balance(output_dir: Path) -> Dict:
    """Analizza il bilanciamento del dataset"""
    stats = collect_dataset_stats(output_dir)

    balance_info = {
        'total_samples': stats['total_subgraphs'],
        'smelly_samples': stats.get('smelly_count', 0),
        'non_smelly_samples': stats.get('non_smelly_count', 0),
        'balance_ratio': 0.0,
        'is_balanced': False,
        'recommendation': ""
    }

    if balance_info['total_samples'] > 0:
        if balance_info['smelly_samples'] > 0:
            balance_info['balance_ratio'] = balance_info['non_smelly_samples'] / balance_info['smelly_samples']

        # Considera bilanciato se il rapporto è tra 0.5 e 2.0
        balance_info['is_balanced'] = 0.5 <= balance_info['balance_ratio'] <= 2.0

        if balance_info['balance_ratio'] < 0.5:
            balance_info['recommendation'] = "Consider including more non-smelly samples"
        elif balance_info['balance_ratio'] > 2.0:
            balance_info['recommendation'] = "Consider including more smelly samples"
        else:
            balance_info['recommendation'] = "Dataset appears reasonably balanced"

    return balance_info


def save_pipeline_statistics(stats: Dict, output_file: Path):
    """Salva statistiche complete della pipeline"""
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Pipeline statistics saved to {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save pipeline statistics: {e}")


def process_single_project(
        project: str,
        arcan_dir: Path,
        output_dir: Path,
        feature_config: Dict
) -> Dict:
    """Processa un singolo progetto con statistiche dettagliate"""
    start_time = time.time()
    project_stats = {
        'project': project,
        'start_time': time.time(),
        'success': False,
        'error': None,
        'smell_map_stats': {},
        'subgraph_stats': {},
        'processing_time': 0
    }

    try:
        logger.info(f"=== Processing project: {project} ===")

        # Validazione struttura progetto
        is_valid, errors = validate_project_structure(project, arcan_dir)
        if not is_valid:
            project_stats['error'] = f"Validation failed: {'; '.join(errors)}"
            logger.error(project_stats['error'])
            return project_stats

        # Estrazione smell maps
        logger.info(f"Extracting smell maps for {project}")
        smell_map = extract_smell_maps(project, arcan_dir)

        # Carica statistiche smell map se disponibili
        stats_file = arcan_dir / project / "smell_map_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    project_stats['smell_map_stats'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load smell map stats: {e}")

        # Estrazione subgrafi
        logger.info(f"Extracting subgraphs for {project}")
        extract_graph_features(
            project_name=project,
            arcan_dir=arcan_dir,
            output_dir=output_dir,
            feature_config=feature_config,
        )

        # Raccogliere statistiche subgrafi per questo progetto
        all_stats = collect_dataset_stats(output_dir)
        project_files = [f for f in output_dir.glob("*.pt") if f.name.startswith(f"{project}_")]

        project_stats['subgraph_stats'] = {
            'files_generated': len(project_files),
            'project_specific_stats': f"Generated {len(project_files)} subgraphs"
        }

        project_stats['success'] = True
        logger.info(f"✅ Successfully processed {project}")

    except Exception as e:
        project_stats['error'] = str(e)
        logger.error(f"❌ Error processing project {project}: {e}")

    finally:
        project_stats['processing_time'] = time.time() - start_time
        logger.info(f"Project {project} processed in {project_stats['processing_time']:.2f} seconds")

    return project_stats


def run_pipeline():
    """Pipeline principale con gestione errori e statistiche complete"""
    pipeline_start = time.time()
    pipeline_stats = {
        'pipeline_start': time.time(),
        'total_projects': 0,
        'successful_projects': 0,
        'failed_projects': 0,
        'project_details': [],
        'dataset_stats': {},
        'balance_analysis': {},
        'sequence_extraction': {},
        'total_time': 0,
        'errors': []
    }

    try:
        # Carica configurazione
        logger.info("Loading configuration...")
        projects = load_projects_name(CONFIG_PATH, 'projects')
        feature_config = load_feature_config(CONFIG_PATH)

        pipeline_stats['total_projects'] = len(projects)
        logger.info(f"Found {len(projects)} projects to process")

        # Crea directory di output
        OUTPUT_DIR_SUBGRAPH.mkdir(parents=True, exist_ok=True)

        # Processa ogni progetto
        for project in projects:
            project_stats = process_single_project(
                project, ARCAN_DIR, OUTPUT_DIR_SUBGRAPH, feature_config
            )

            pipeline_stats['project_details'].append(project_stats)

            if project_stats['success']:
                pipeline_stats['successful_projects'] += 1
            else:
                pipeline_stats['failed_projects'] += 1
                pipeline_stats['errors'].append({
                    'project': project,
                    'error': project_stats['error']
                })

        logger.info("✅ Subgraph 1‑hop extraction completed.")

        # Statistiche del dataset completo
        logger.info("Collecting final dataset statistics...")
        pipeline_stats['dataset_stats'] = collect_dataset_stats(OUTPUT_DIR_SUBGRAPH)

        # Analisi bilanciamento
        pipeline_stats['balance_analysis'] = check_dataset_balance(OUTPUT_DIR_SUBGRAPH)

        logger.info("Dataset Statistics:")
        logger.info(f"  - Total subgraphs: {pipeline_stats['dataset_stats']['total_subgraphs']}")
        logger.info(f"  - Projects: {len(pipeline_stats['dataset_stats']['projects'])}")
        logger.info(f"  - Unique components: {len(pipeline_stats['dataset_stats']['components'])}")
        logger.info(f"  - Smelly samples: {pipeline_stats['dataset_stats'].get('smelly_count', 'N/A')}")
        logger.info(f"  - Non-smelly samples: {pipeline_stats['dataset_stats'].get('non_smelly_count', 'N/A')}")

        balance = pipeline_stats['balance_analysis']
        logger.info(f"Balance Analysis:")
        logger.info(f"  - Balance ratio: {balance['balance_ratio']:.2f}")
        logger.info(f"  - Is balanced: {balance['is_balanced']}")
        logger.info(f"  - Recommendation: {balance['recommendation']}")

        # Estrazione sequenze expert (se richiesta)
        if feature_config.get('extract_sequences', True):
            try:
                logger.info("Starting expert sequence collection...")
                seq_start = time.time()

                seqs = build_expert_sequences(str(OUTPUT_DIR_SUBGRAPH))
                SEQS_OUT.parent.mkdir(parents=True, exist_ok=True)
                torch.save(seqs, SEQS_OUT)

                seq_time = time.time() - seq_start
                pipeline_stats['sequence_extraction'] = {
                    'success': True,
                    'output_file': str(SEQS_OUT),
                    'processing_time': seq_time,
                    'sequence_count': len(seqs) if hasattr(seqs, '__len__') else 'Unknown'
                }

                logger.info(f"✅ Expert sequences saved in {SEQS_OUT} ({seq_time:.2f}s)")

            except Exception as e:
                error_msg = f"Failed to build expert sequences: {e}"
                logger.error(f"❌ {error_msg}")
                pipeline_stats['sequence_extraction'] = {
                    'success': False,
                    'error': error_msg
                }
                pipeline_stats['errors'].append({
                    'component': 'sequence_extraction',
                    'error': error_msg
                })

    except Exception as e:
        error_msg = f"Pipeline critical failure: {e}"
        logger.error(f"❌ {error_msg}")
        pipeline_stats['errors'].append({
            'component': 'pipeline',
            'error': error_msg
        })
        raise

    finally:
        # Statistiche finali
        pipeline_stats['total_time'] = time.time() - pipeline_start

        # Salva statistiche complete
        save_pipeline_statistics(pipeline_stats, PIPELINE_STATS_FILE)

        # Log finale
        logger.info("🏁 Pipeline finished.")
        logger.info(
            f"Results: {pipeline_stats['successful_projects']}/{pipeline_stats['total_projects']} projects successful")
        logger.info(f"Total time: {pipeline_stats['total_time']:.2f} seconds")

        if pipeline_stats['errors']:
            logger.warning(f"Encountered {len(pipeline_stats['errors'])} errors during processing")


def generate_dataset_report(stats_file: Path, output_file: Optional[Path] = None):
    """Genera un report dettagliato del dataset"""
    if not stats_file.exists():
        logger.error(f"Stats file not found: {stats_file}")
        return

    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        report_lines = [
            "# Dataset Generation Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Pipeline Summary",
            f"- Total projects processed: {stats.get('total_projects', 0)}",
            f"- Successful projects: {stats.get('successful_projects', 0)}",
            f"- Failed projects: {stats.get('failed_projects', 0)}",
            f"- Total processing time: {stats.get('total_time', 0):.2f} seconds",
            "",
            "## Dataset Statistics",
        ]

        dataset_stats = stats.get('dataset_stats', {})
        report_lines.extend([
            f"- Total subgraphs: {dataset_stats.get('total_subgraphs', 0)}",
            f"- Unique projects: {len(dataset_stats.get('projects', []))}",
            f"- Unique components: {len(dataset_stats.get('components', []))}",
            f"- Smelly samples: {dataset_stats.get('smelly_count', 0)}",
            f"- Non-smelly samples: {dataset_stats.get('non_smelly_count', 0)}",
            ""
        ])

        # Analisi bilanciamento
        balance = stats.get('balance_analysis', {})
        report_lines.extend([
            "## Balance Analysis",
            f"- Balance ratio: {balance.get('balance_ratio', 0):.2f}",
            f"- Is balanced: {balance.get('is_balanced', False)}",
            f"- Recommendation: {balance.get('recommendation', 'N/A')}",
            ""
        ])

        # Errori se presenti
        errors = stats.get('errors', [])
        if errors:
            report_lines.extend([
                "## Errors Encountered",
                ""
            ])
            for error in errors:
                report_lines.append(
                    f"- {error.get('project', error.get('component', 'Unknown'))}: {error.get('error', 'Unknown error')}")

        report_content = "\n".join(report_lines)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_content, encoding='utf-8')
            logger.info(f"Dataset report saved to {output_file}")
        else:
            print(report_content)

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run the complete dataset generation pipeline")
    parser.add_argument("--generate-report", action="store_true", help="Generate dataset report after pipeline")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing stats")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Configura logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        if args.report_only:
            # Solo generazione report
            report_file = OUTPUT_DIR / "dataset_report.md"
            generate_dataset_report(PIPELINE_STATS_FILE, report_file)
        else:
            # Esegui pipeline completa
            run_pipeline()

            # Genera report se richiesto
            if args.generate_report:
                report_file = OUTPUT_DIR / "dataset_report.md"
                generate_dataset_report(PIPELINE_STATS_FILE, report_file)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)