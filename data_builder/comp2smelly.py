#!/usr/bin/env python3
"""
Script corretto per build_smell_map_arcan.py con logica originale mantenuta

Il problema nel codice attuale era:
1. Join scorretto: usava smellId (vertexId) invece di fromId per il join con affects
2. Filtro troppo restrittivo sui tipi di costrutto
3. Logica di mapping invertita - mappava hub invece che componenti affette

Questa versione ripristina la logica originale corretta.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

# Tipi di costrutti da escludere (come nel codice originale)
EXCLUDED_TYPES = {"PACKAGE"}


def validate_csv_files(project_dir: Path) -> Tuple[bool, List[str]]:
    """Valida l'esistenza e la struttura dei file CSV richiesti"""
    required_files = {
        "smell-characteristics.csv": ["vertexId", "versionId", "AffectedConstructType"],
        "smell-affects.csv": ["fromId", "toId", "versionId"]
    }

    errors = []

    for filename, required_cols in required_files.items():
        filepath = project_dir / filename

        if not filepath.exists():
            errors.append(f"Missing required file: {filename}")
            continue

        try:
            # Leggi solo le prime righe per validazione
            df_sample = pd.read_csv(filepath, nrows=5)
            missing_cols = set(required_cols) - set(df_sample.columns)

            if missing_cols:
                errors.append(f"File {filename} missing columns: {missing_cols}")

        except Exception as e:
            errors.append(f"Cannot read {filename}: {e}")

    return len(errors) == 0, errors


def analyze_construct_types(project_dir: Path) -> Dict:
    """Analizza i tipi di costrutti presenti nel progetto per debugging"""
    chars_path = project_dir / "smell-characteristics.csv"

    if not chars_path.exists():
        return {}

    try:
        df_chars = pd.read_csv(chars_path)

        # Statistiche sui tipi di costrutti
        type_counts = df_chars['AffectedConstructType'].value_counts().to_dict()

        # Identifica quali saranno esclusi
        unique_types = set(df_chars['AffectedConstructType'].unique())
        excluded_types = unique_types.intersection(EXCLUDED_TYPES)
        included_types = unique_types - EXCLUDED_TYPES

        analysis = {
            'total_constructs': len(df_chars),
            'unique_types': list(unique_types),
            'type_counts': type_counts,
            'included_types': list(included_types),
            'excluded_types_found': list(excluded_types),
            'potential_smells': sum(count for type_name, count in type_counts.items()
                                    if type_name not in EXCLUDED_TYPES)
        }

        return analysis

    except Exception as e:
        logger.error(f"Failed to analyze construct types: {e}")
        return {}


def get_cache_key(project_dir: Path) -> str:
    """Genera una chiave di cache basata sui timestamp dei file CSV"""
    files_to_check = ["smell-characteristics.csv", "smell-affects.csv"]

    hash_input = []
    for filename in files_to_check:
        filepath = project_dir / filename
        if filepath.exists():
            stat = filepath.stat()
            hash_input.append(f"{filename}:{stat.st_mtime}:{stat.st_size}")

    return hashlib.md5(":".join(hash_input).encode()).hexdigest()


def load_cached_result(project_dir: Path, cache_key: str) -> Optional[Dict]:
    """Carica risultato dalla cache se disponibile e valido"""
    cache_file = project_dir / ".smell_map.json"

    if not cache_file.exists():
        return None

    try:
        cache_data = json.loads(cache_file.read_text(encoding='utf-8'))
        if cache_data.get('cache_key') == cache_key:
            logger.info(f"Using cached smell map for {project_dir.name}")
            return cache_data.get('mapping')
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")

    return None


def save_to_cache(project_dir: Path, cache_key: str, mapping: Dict, stats: Dict):
    """Salva risultato nella cache"""
    cache_file = project_dir / ".smell_map.json"

    cache_data = {
        'cache_key': cache_key,
        'mapping': mapping,
        'stats': stats,
        'generated_at': pd.Timestamp.now().isoformat()
    }

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        logger.debug(f"Cached results in {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def build_smell_map(project_dir: Path) -> Tuple[Dict, Dict]:
    """
    Costruisce un dizionario compId -> [versionId, ...]
    Ripristina la logica originale corretta del codice vecchio
    """
    # Valida file di input
    is_valid, errors = validate_csv_files(project_dir)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    # Controlla cache
    cache_key = get_cache_key(project_dir)
    cached_result = load_cached_result(project_dir, cache_key)
    if cached_result is not None:
        return cached_result, {}

    # Carica i file CSV
    chars_path = project_dir / "smell-characteristics.csv"
    affects_path = project_dir / "smell-affects.csv"

    try:
        df_chars = pd.read_csv(chars_path)
        df_affects = pd.read_csv(affects_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV files: {e}")

    # Analizza tipi di costrutti prima del filtraggio
    construct_analysis = analyze_construct_types(project_dir)

    # Statistiche pre-processing
    stats = {
        'total_characteristics': len(df_chars),
        'total_affects': len(df_affects),
        'unique_versions_chars': df_chars['versionId'].nunique(),
        'unique_versions_affects': df_affects['versionId'].nunique(),
        'construct_analysis': construct_analysis
    }

    # LOGICA ORIGINALE CORRETTA:
    # 1. Filtra characteristics escludendo solo PACKAGE (come nel codice originale)
    hub_rows = df_chars[~df_chars["AffectedConstructType"].isin(EXCLUDED_TYPES)].copy()

    if hub_rows.empty:
        logger.warning("No valid smell components found after filtering")
        logger.info(f"Available construct types: {construct_analysis.get('type_counts', {})}")
        return {}, stats

    # 2. Rinomina vertexId -> smellId in characteristics
    hub_rows = hub_rows.rename(columns={"vertexId": "smellId"})

    # 3. Rinomina fromId -> smellId in affects
    affects_ren = df_affects.rename(columns={"fromId": "smellId"})

    # Log delle statistiche di filtraggio
    logger.info(f"Filtering results:")
    logger.info(f"  - Total constructs: {len(df_chars)}")
    logger.info(f"  - Valid smell constructs: {len(hub_rows)}")
    logger.info(f"  - Excluded constructs: {len(df_chars) - len(hub_rows)}")

    # Aggiorna statistiche
    stats.update({
        'valid_smell_constructs': len(hub_rows),
        'excluded_constructs': len(df_chars) - len(hub_rows),
        'included_construct_types': hub_rows['AffectedConstructType'].value_counts().to_dict()
    })

    # Controllo consistenza versioni
    common_versions = set(hub_rows['versionId']) & set(affects_ren['versionId'])
    if not common_versions:
        logger.warning("No common versions between characteristics and affects")
        return {}, stats

    stats['common_versions'] = len(common_versions)

    # 4. Join su smellId e versionId (QUESTA È LA CHIAVE!)
    # smellId in hub_rows è il vertexId originale (ID dello smell)
    # smellId in affects_ren è il fromId originale (ID del nodo sorgente, che dovrebbe essere lo smell)
    try:
        aff_join = affects_ren.merge(
            hub_rows,
            on=["smellId", "versionId"],
            how="inner"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to join DataFrames: {e}")

    if aff_join.empty:
        logger.warning("No records after join - check data consistency")
        logger.info("Sample characteristics smellIds: " + str(hub_rows['smellId'].head().tolist()))
        logger.info("Sample affects smellIds: " + str(affects_ren['smellId'].head().tolist()))
        return {}, stats

    # Statistiche post-join
    stats.update({
        'joined_records': len(aff_join),
        'unique_components_affected': aff_join['toId'].nunique() if 'toId' in aff_join.columns else 0,
        'unique_smell_hubs': aff_join['smellId'].nunique() if 'smellId' in aff_join.columns else 0
    })

    logger.info(f"Successfully joined data: {len(aff_join)} records for {stats['unique_smell_hubs']} smell hubs")

    # 5. Costruisci comp2smelly (LOGICA ORIGINALE)
    # Mappa componente affetta (toId) -> versioni in cui è affetta
    comp2smelly = defaultdict(set)

    for _, row in aff_join.iterrows():
        comp_id = str(row["toId"])  # Componente affetta dallo smell
        ver_id = str(row["versionId"])  # Versione in cui è affetta
        comp2smelly[comp_id].add(ver_id)

    # Converte in dizionario con liste ordinate (come nel codice originale)
    comp2smelly_sorted = {comp: sorted(versions) for comp, versions in comp2smelly.items()}

    # Statistiche finali
    stats.update({
        'total_affected_components': len(comp2smelly_sorted),
        'avg_versions_per_component': sum(len(v) for v in comp2smelly_sorted.values()) / len(
            comp2smelly_sorted) if comp2smelly_sorted else 0,
        'max_versions_per_component': max(len(v) for v in comp2smelly_sorted.values()) if comp2smelly_sorted else 0,
        'min_versions_per_component': min(len(v) for v in comp2smelly_sorted.values()) if comp2smelly_sorted else 0
    })

    # Salva in cache
    save_to_cache(project_dir, cache_key, comp2smelly_sorted, stats)

    return comp2smelly_sorted, stats


def extract_smell_maps(project: str, arcan_dir: Path) -> Dict:
    """
    Funzione principale per estrarre le mappature "smelly" con logica originale corretta
    """
    project_dir = arcan_dir / project

    if not project_dir.exists():
        error_msg = f"Directory non trovata: {project_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Processing smell maps for project: {project}")

    try:
        mapping, stats = build_smell_map(project_dir)

        # Salva mapping principale
        output_json = project_dir / "smell_map.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        # Salva statistiche dettagliate
        stats_json = project_dir / "smell_map_stats.json"
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ Smell map saved: {output_json}")
        logger.info(f"   - Affected components: {stats.get('total_affected_components', 0)}")
        logger.info(f"   - Smell hubs: {stats.get('unique_smell_hubs', 0)}")
        logger.info(f"   - Avg versions per component: {stats.get('avg_versions_per_component', 0):.2f}")
        logger.info(f"✅ Statistics saved: {stats_json}")

        return mapping

    except Exception as e:
        logger.error(f"Failed to process {project}: {e}")
        raise


def validate_smell_map(mapping: Dict, project_dir: Path) -> List[str]:
    """Valida la qualità della smell map generata"""
    warnings = []

    if not mapping:
        warnings.append("Empty smell map generated - no affected components found")
        return warnings

    # Controlla componenti con singola versione (potenziali false positive)
    single_version_comps = [comp for comp, versions in mapping.items() if len(versions) == 1]
    if len(single_version_comps) > len(mapping) * 0.5:
        warnings.append(f"High number of single-version components: {len(single_version_comps)}/{len(mapping)}")

    # Controlla consistenza degli ID versione
    all_versions = set()
    for versions in mapping.values():
        all_versions.update(versions)

    if len(all_versions) < 2:
        warnings.append("Very few unique versions found - check data quality")

    # Controlla pattern negli ID
    version_lengths = [len(v) for v in all_versions]
    if len(set(version_lengths)) > 3:
        warnings.append("Inconsistent version ID formats detected")

    return warnings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract smell maps from Arcan output (corrected logic)")
    parser.add_argument("project", help="Project name")
    parser.add_argument("arcan_dir", type=Path, help="Path to arcan_out directory")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    parser.add_argument("--analyze-types", action="store_true", help="Show detailed construct type analysis")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    try:
        if args.analyze_types:
            # Analisi dettagliata dei tipi di costrutti
            project_dir = args.arcan_dir / args.project
            analysis = analyze_construct_types(project_dir)

            print(f"\n=== Construct Type Analysis for {args.project} ===")
            print(f"Total constructs: {analysis.get('total_constructs', 0)}")
            print(f"Valid smell constructs: {analysis.get('potential_smells', 0)}")
            print(f"\nAll construct types found:")
            for type_name, count in analysis.get('type_counts', {}).items():
                status = "EXCLUDED" if type_name in EXCLUDED_TYPES else "INCLUDED"
                print(f"  {type_name}: {count} ({status})")

            print(f"\nIncluded types: {analysis.get('included_types', [])}")
            print(f"Excluded types found: {analysis.get('excluded_types_found', [])}")

        mapping = extract_smell_maps(args.project, args.arcan_dir)

        if args.validate:
            project_dir = args.arcan_dir / args.project
            warnings = validate_smell_map(mapping, project_dir)
            if warnings:
                logger.warning("Validation warnings:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")

        print(f"Successfully processed {args.project}")
        print(f"Found {len(mapping)} components affected by smells")

    except Exception as e:
        logger.error(f"Failed: {e}")
        exit(1)