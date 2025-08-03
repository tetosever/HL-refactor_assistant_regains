#!/usr/bin/env python3
"""
Estrazione ottimizzata di subgraph 1‑hop + metriche topologiche migliorata,
con validazione robusta, configurazione flessibile, e tracciamento degli ID.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

logger = logging.getLogger(__name__)
GRAPH_RE = re.compile(r"dependency-graph-(\d+)_([0-9a-fA-F]+)\.graphml")


def validate_and_clean_features(G: nx.Graph, node_feats: List[str], edge_feats: List[str]) -> Tuple[
    List[str], List[str]]:
    """Valida e pulisce le feature richieste contro quelle effettivamente disponibili nel grafo"""

    # Raccogli tutti gli attributi disponibili nei nodi
    available_node_attrs = set()
    for n, attrs in G.nodes(data=True):
        available_node_attrs.update(attrs.keys())

    # Raccogli tutti gli attributi disponibili negli archi
    available_edge_attrs = set()
    for u, v, attrs in G.edges(data=True):
        available_edge_attrs.update(attrs.keys())

    # Filtra node_feats mantenendo solo quelli disponibili (eccetto metriche topologiche)
    topological_metrics = {'deg_in', 'deg_out', 'betweenness', 'closeness'}
    valid_node_feats = []

    for feat in node_feats:
        if feat in topological_metrics:
            # Le metriche topologiche vengono calcolate, quindi sono sempre valide
            valid_node_feats.append(feat)
        elif feat in available_node_attrs:
            valid_node_feats.append(feat)
        else:
            logger.warning(f"Node feature '{feat}' not found in graph, skipping")

    # Filtra edge_feats mantenendo solo quelli disponibili
    valid_edge_feats = []
    for feat in edge_feats:
        if feat in available_edge_attrs:
            valid_edge_feats.append(feat)
        else:
            logger.warning(f"Edge feature '{feat}' not found in graph, skipping")

    if len(valid_node_feats) != len(node_feats):
        logger.info(f"Node features reduced from {len(node_feats)} to {len(valid_node_feats)}")

    if len(valid_edge_feats) != len(edge_feats):
        logger.info(f"Edge features reduced from {len(edge_feats)} to {len(valid_edge_feats)}")

    return valid_node_feats, valid_edge_feats


def load_feature_config(path: Path) -> dict:
    """Carica configurazione con validazione e valori di default migliorati"""
    try:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        raise

    # Configurazione base features
    feats = {
        'node_feats': cfg.get('node_features', []).copy(),
        'edge_feats': cfg.get('edge_features', []).copy(),
        'smell_feats': cfg.get('smell_features', []).copy()
    }

    # Configurazione extraction con defaults
    extraction_cfg = cfg.get('extraction', {})
    feats.update({
        'max_graph_size': extraction_cfg.get('max_graph_size', 10000),
        'min_subgraph_size': extraction_cfg.get('min_subgraph_size', 3),
        'radius': extraction_cfg.get('radius', 1),
        'include_non_smelly': extraction_cfg.get('include_non_smelly', False),
        'parallel_processing': extraction_cfg.get('parallel_processing', True)
    })

    # Configurazione quality filters
    quality_cfg = cfg.get('quality_filters', {})
    feats.update({
        'remove_isolated_nodes': quality_cfg.get('remove_isolated_nodes', True),
        'min_edges': quality_cfg.get('min_edges', 1)
    })

    # Aggiungiamo le metriche topologiche ai node features
    feats['node_feats'].extend(['deg_in', 'deg_out', 'betweenness', 'closeness'])

    return feats


def debug_graph_attributes(sg: nx.Graph, node_feats: List[str], edge_feats: List[str]) -> Dict:
    """Debug function per analizzare gli attributi del grafo prima della conversione"""
    debug_info = {
        'node_attr_types': {},
        'edge_attr_types': {},
        'problematic_nodes': [],
        'problematic_edges': []
    }

    # Analizza attributi nodi
    for n, attrs in sg.nodes(data=True):
        for k, v in attrs.items():
            if k in node_feats:
                attr_type = type(v).__name__
                if k not in debug_info['node_attr_types']:
                    debug_info['node_attr_types'][k] = set()
                debug_info['node_attr_types'][k].add(attr_type)

                # Identifica attributi problematici
                if not isinstance(v, (int, float, bool)) and v is not None:
                    debug_info['problematic_nodes'].append({
                        'node': n,
                        'attribute': k,
                        'type': attr_type,
                        'value': str(v)[:50]  # Primi 50 caratteri
                    })

    # Analizza attributi archi
    for u, v, attrs in sg.edges(data=True):
        for k, val in attrs.items():
            if k in edge_feats:
                attr_type = type(val).__name__
                if k not in debug_info['edge_attr_types']:
                    debug_info['edge_attr_types'][k] = set()
                debug_info['edge_attr_types'][k].add(attr_type)

                # Identifica attributi problematici
                if not isinstance(val, (int, float, bool)) and val is not None:
                    debug_info['problematic_edges'].append({
                        'edge': f"{u}->{v}",
                        'attribute': k,
                        'type': attr_type,
                        'value': str(val)[:50]
                    })

    # Converti set in liste per serializzazione
    for k, v in debug_info['node_attr_types'].items():
        debug_info['node_attr_types'][k] = list(v)
    for k, v in debug_info['edge_attr_types'].items():
        debug_info['edge_attr_types'][k] = list(v)

    return debug_info


def validate_component_exists(G: nx.Graph, center: str) -> bool:
    """Valida che il componente esista e sia raggiungibile"""
    if center not in G:
        return False

    # Controlla che abbia almeno un vicino (altrimenti l'ego-graph è triviale)
    neighbors = list(G.neighbors(center))
    return len(neighbors) > 0


def compute_metrics_efficiently(sg: nx.Graph) -> Tuple[Dict, Dict]:
    """Calcola metriche topologiche in modo efficiente"""
    if len(sg) < 10:  # Per grafi piccoli, usa metriche semplificate
        betw = {n: 0.0 for n in sg.nodes()}
        clos = {n: 1.0 for n in sg.nodes()}
    else:
        try:
            betw = nx.betweenness_centrality(sg)
            clos = nx.closeness_centrality(sg)
        except Exception as e:
            logger.warning(f"Failed to compute centrality metrics: {e}")
            betw = {n: 0.0 for n in sg.nodes()}
            clos = {n: 1.0 for n in sg.nodes()}

    return betw, clos


def create_stable_edge_fingerprint(edge_index: torch.Tensor) -> Tuple:
    """Crea un fingerprint stabile per gli edge ordinandoli"""
    if edge_index.numel() == 0:
        return tuple()

    # Converti in lista di tuple e ordina per avere fingerprint stabile
    edges = edge_index.t().tolist()
    edges.sort(key=lambda x: (x[0], x[1]))
    return tuple(map(tuple, edges))


def ego_1hop_to_data(
        G: nx.Graph,
        center: str,
        is_smelly: bool,
        node_feats: List[str],
        edge_feats: List[str],
        config: dict
) -> Optional[Data]:
    """
    Estrae il sottografo ego-1hop con validazione robusta e tracciamento ID
    """
    # Validazione input
    if not validate_component_exists(G, center):
        return None

    # Controllo dimensione grafo
    if len(G) > config.get('max_graph_size', 10000):
        logger.warning(f"Graph too large ({len(G)} nodes), skipping component {center}")
        return None

    # Estrazione ego-graph
    sg = nx.ego_graph(G, center, radius=config.get('radius', 1), undirected=False)

    # Controllo dimensione minima subgrafo
    if len(sg) < config.get('min_subgraph_size', 3):
        return None

    # Rimozione nodi isolati se configurato
    if config.get('remove_isolated_nodes', True):
        isolated = list(nx.isolates(sg))
        if isolated:
            sg.remove_nodes_from(isolated)
            if len(sg) < config.get('min_subgraph_size', 3):
                return None

    # Controllo numero minimo di archi
    if sg.number_of_edges() < config.get('min_edges', 1):
        return None

    # Calcolo metriche topologiche
    indeg = dict(sg.in_degree())
    outdeg = dict(sg.out_degree())
    betw, clos = compute_metrics_efficiently(sg)

    # Prepara mapping originale ID -> nuovo ID per tracciamento
    node_id_mapping = {}
    original_node_ids = []

    # Filtra e aggiorna attributi nodi
    for i, (n, attrs) in enumerate(sg.nodes(data=True)):
        node_id_mapping[n] = i
        original_node_ids.append(str(n))  # Mantieni ID originale come stringa

        # Rimuovi chiavi non richieste
        for k in list(attrs):
            if k not in node_feats:
                attrs.pop(k, None)

        # Aggiungi metriche topologiche
        attrs.update({
            'deg_in': float(indeg.get(n, 0)),
            'deg_out': float(outdeg.get(n, 0)),
            'betweenness': float(betw.get(n, 0)),
            'closeness': float(clos.get(n, 0)),
        })

    # Prepara informazioni sugli archi
    all_edge_keys = set().union(*(d.keys() for _, _, d in sg.edges(data=True)))
    original_edge_ids = []
    edge_sources = []
    edge_targets = []

    for u, v, d in sg.edges(data=True):
        # Traccia gli ID originali degli archi
        edge_id = d.get('id', f"{u}->{v}")  # Usa ID se presente, altrimenti crea uno
        original_edge_ids.append(str(edge_id))
        edge_sources.append(str(u))
        edge_targets.append(str(v))

        # Assicura tutte le chiavi edge
        for key in all_edge_keys:
            d.setdefault(key, 0.0)

        # Rimuovi chiavi non richieste
        for k in list(d):
            if k not in edge_feats:
                d.pop(k, None)

        # Cast a float
        for k in edge_feats:
            try:
                d[k] = float(d.get(k, 0.0))
            except (TypeError, ValueError):
                d[k] = 0.0

    for n, attrs in sg.nodes(data=True):
        for feat in node_feats:
            v = attrs.get(feat)
            if isinstance(v, list):
                attrs[feat] = torch.tensor(v, dtype=torch.float)

    for u, v, attrs in sg.edges(data=True):
        for feat in edge_feats:
            val = attrs.get(feat)
            if isinstance(val, list):
                attrs[feat] = torch.tensor(val, dtype=torch.float)

    # Conversione in Data PyG
    try:
        data = from_networkx(
            sg,
            group_node_attrs=node_feats,
            group_edge_attrs=edge_feats
        )
    except Exception as e:
        logger.warning(f"Failed to convert subgraph to PyG data: {e}")
        return None

    # Aggiungi metadati personalizzati
    data.is_smelly = torch.tensor([int(is_smelly)], dtype=torch.long)
    data.center_node = center
    data.original_node_ids = original_node_ids
    data.original_edge_ids = original_edge_ids
    data.edge_sources = edge_sources
    data.edge_targets = edge_targets
    data.subgraph_size = len(sg)
    data.num_edges_sg = sg.number_of_edges()

    return data


def process_single_component(args: Tuple) -> Tuple[str, int, Optional[Data]]:
    """Processa un singolo componente - per parallelizzazione"""
    G, comp_id, smelly_versions, version, feature_config = args

    is_smelly = version in smelly_versions

    # Salta componenti non-smelly se non richiesti
    if not is_smelly and not feature_config.get('include_non_smelly', False):
        return comp_id, 0, None

    data = ego_1hop_to_data(
        G=G,
        center=comp_id,
        is_smelly=is_smelly,
        node_feats=feature_config['node_feats'],
        edge_feats=feature_config['edge_feats'],
        config=feature_config
    )

    return comp_id, 1 if data is not None else 0, data


def collect_dataset_stats(output_dir: Path) -> Dict:
    """Raccoglie statistiche sul dataset generato"""
    pt_files = list(output_dir.glob("*.pt"))
    stats = {
        'total_subgraphs': len(pt_files),
        'projects': set(),
        'components': set(),
        'smelly_count': 0,
        'non_smelly_count': 0,
        'size_distribution': [],
        'edge_distribution': []
    }

    for pt_file in pt_files:
        try:
            parts = pt_file.stem.split('_')
            if len(parts) >= 2:
                stats['projects'].add(parts[0])
                stats['components'].add(parts[1])

            data = torch.load(pt_file, map_location='cpu')
            stats['size_distribution'].append(data.num_nodes)
            stats['edge_distribution'].append(data.num_edges)

            if hasattr(data, 'is_smelly'):
                if data.is_smelly.item():
                    stats['smelly_count'] += 1
                else:
                    stats['non_smelly_count'] += 1

        except Exception as e:
            logger.warning(f"Failed to load {pt_file} for stats: {e}")
            continue

    # Converti set in liste per JSON serialization
    stats['projects'] = list(stats['projects'])
    stats['components'] = list(stats['components'])

    return stats


def extract_graph_features(
        project_name: str,
        arcan_dir: Path,
        output_dir: Path,
        feature_config: dict
) -> None:
    """
    Estrazione migliorata con parallelizzazione opzionale e statistiche
    """
    project_dir = arcan_dir / project_name
    map_path = project_dir / "smell_map.json"

    if not project_dir.exists() or not map_path.exists():
        raise FileNotFoundError(f"{project_name}: arcan_out o smell_map.json mancanti")

    # Carica mappature smell
    try:
        comp2smelly = json.loads(map_path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"Failed to load smell map for {project_name}: {e}")
        raise

    # Prepara lista di graphml da processare
    records: List[Tuple[int, str, Path]] = []
    for fp in project_dir.glob("dependency-graph-*.graphml"):
        m = GRAPH_RE.match(fp.name)
        if m:
            records.append((int(m.group(1)), m.group(2), fp))

    records.sort(key=lambda t: t[0])
    logger.info(f"Processing {project_name}: {len(records)} graphml files")

    # Inizializza contatori
    comp_index = {comp: 0 for comp in comp2smelly}
    last_fp_map = {comp: None for comp in comp2smelly}
    duplicates = 0
    processed_count = 0
    error_count = 0

    for idx, version, gpath in records:
        # Carica il grafo
        try:
            G = nx.read_graphml(gpath)
            logger.debug(f"Loaded graph {gpath.name}: {len(G)} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"Skipping {gpath.name}: read_graphml failed: {e}")
            error_count += 1
            continue

        # Controlla dimensione grafo
        if len(G) > feature_config.get('max_graph_size', 10000):
            logger.warning(f"Skipping {gpath.name}: graph too large ({len(G)} nodes)")
            continue

        # Processa ogni componente
        components_processed = 0

        for comp_id, smelly_versions in comp2smelly.items():
            is_smelly = version in smelly_versions

            # Salta componenti non-smelly se non richiesti
            if not is_smelly and not feature_config.get('include_non_smelly', False):
                continue

            # Estrai subgrafo
            data = ego_1hop_to_data(
                G=G,
                center=comp_id,
                is_smelly=is_smelly,
                node_feats=feature_config['node_feats'],
                edge_feats=feature_config['edge_feats'],
                config=feature_config
            )

            if data is None:
                continue

            # Controlla duplicati con fingerprint stabile
            fp_tuple = create_stable_edge_fingerprint(data.edge_index)
            if fp_tuple == last_fp_map[comp_id]:
                duplicates += 1
                continue

            last_fp_map[comp_id] = fp_tuple

            # Salva il .pt con metadati aggiuntivi
            fname = f"{project_name}_{comp_id}_{comp_index[comp_id]:04d}_{idx:04d}_{version}.pt"
            outfp = output_dir / fname

            try:
                # Aggiungi metadati per tracciamento
                data.project_name = project_name
                data.graph_version = version
                data.graph_index = idx
                data.extraction_timestamp = torch.tensor([torch.get_rng_state().sum().item()], dtype=torch.long)

                torch.save(data, outfp)
                comp_index[comp_id] += 1
                components_processed += 1
                processed_count += 1

            except Exception as e:
                logger.warning(f"torch.save failed for {outfp}: {e}")
                error_count += 1
                continue

        logger.debug(f"Graph {gpath.name}: processed {components_processed} components")

    # Statistiche finali
    logger.info(f"Completed {project_name}:")
    logger.info(f"  - Processed: {processed_count} subgraphs")
    logger.info(f"  - Duplicates eliminated: {duplicates}")
    logger.info(f"  - Errors: {error_count}")

    # Salva statistiche del progetto
    stats = collect_dataset_stats(output_dir)
    project_stats = {k: v for k, v in stats.items() if project_name in str(v)}

    stats_file = output_dir / f"{project_name}_stats.json"
    try:
        with open(stats_file, 'w') as f:
            json.dump(project_stats, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save stats for {project_name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract optimized 1-hop subgraphs with enhanced features")
    parser.add_argument("project_name", help="Nome della cartella arcan_out/<project>")
    parser.add_argument("arcan_dir", type=Path, help="Path a arcan_out")
    parser.add_argument("output_dir", type=Path, help="Directory di destinazione dei .pt")
    parser.add_argument("config_yaml", type=Path, help="Path al config.yaml")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    try:
        feature_cfg = load_feature_config(args.config_yaml)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        extract_graph_features(
            project_name=args.project_name,
            arcan_dir=args.arcan_dir,
            output_dir=args.output_dir,
            feature_config=feature_cfg
        )

        # Genera statistiche finali
        final_stats = collect_dataset_stats(args.output_dir)
        logger.info("Final dataset statistics:")
        logger.info(f"  - Total subgraphs: {final_stats['total_subgraphs']}")
        logger.info(f"  - Projects: {len(final_stats['projects'])}")
        logger.info(f"  - Unique components: {len(final_stats['components'])}")
        logger.info(f"  - Smelly: {final_stats['smelly_count']}")
        logger.info(f"  - Non-smelly: {final_stats['non_smelly_count']}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise