#!/usr/bin/env python3
"""
Estrazione di sequenze expert per pre‑training (soluzione pulita),
includendo sia gli stati (Data) che le azioni atomiche:
- Skippa i .pt che non si riescono a caricare
- Per ogni flip colleziona paralleli:
    * 'states': lista di Data (prima di ciascuna action)
    * 'actions': tensor [N×5] con (u,v,op,term,step_id)
      op: 0=RemoveEdge, 1=AddEdge, 2=ExtractMethod, 3=Terminate
"""
import glob
import logging
import os
from collections import defaultdict

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_filename(fp: str):
    name = os.path.basename(fp).replace('.pt','')
    parts = name.split('_')
    if len(parts) < 5:
        raise ValueError(f"Filename '{name}' non conforme al pattern atteso.")
    project, comp = parts[0], parts[1]
    counter, idx  = int(parts[2]), int(parts[3])
    return project, comp, counter, idx

def build_expert_sequences(data_dir: str):
    files = glob.glob(os.path.join(data_dir, '*.pt'))
    logger.info(f"Trovati {len(files)} file .pt in '{data_dir}'")

    # Raggruppa per progetto e componente
    groups = defaultdict(list)
    for fp in files:
        try:
            project, comp, counter, idx = parse_filename(fp)
            groups[(project, comp)].append((counter, idx, fp))
        except Exception as e:
            logger.warning(f"Skipping file nome non valido '{fp}': {e}")

    sequences = []
    logger.info(f"Processo {len(groups)} gruppi (project, component)")

    for (project, comp), file_list in groups.items():
        logger.info(f"--- Inizio gruppo {project}/{comp}: {len(file_list)} versioni")

        # Carica e ordina snapshot, skippa .pt non caricarli
        graphs_meta = []
        for counter, idx, fp in sorted(file_list, key=lambda x: x[0]):
            try:
                g = torch.load(fp, map_location='cpu')
            except Exception as e:
                logger.warning(f"  → Skipping {os.path.basename(fp)}: load failed: {e}")
                continue
            if not hasattr(g, 'edge_index') or not hasattr(g, 'is_smelly'):
                logger.warning(f"  → Skipping {os.path.basename(fp)}: attributi mancanti")
                continue
            graphs_meta.append({
                'graph':      g,
                'step_id':    counter,
                'is_smelly':  int(g.is_smelly.item()),
                'comp_index': counter,
                'version_idx': idx
            })

        if len(graphs_meta) < 2:
            logger.info(f"  → Non abbastanza snapshot ({len(graphs_meta)}), skip.")
            continue

        last_flip = 0
        flip_count = 0

        # Estrai sequenze per ogni flip
        for i in range(1, len(graphs_meta)):
            prev_sm = graphs_meta[i-1]['is_smelly']
            curr_sm = graphs_meta[i]['is_smelly']
            if curr_sm != prev_sm:
                flip_count += 1
                is_positive = (prev_sm==1 and curr_sm==0)
                logger.info(
                    f" Flip #{flip_count}: {project}/{comp} "
                    f"{graphs_meta[last_flip]['step_id']}({prev_sm})->"
                    f"{graphs_meta[i]['step_id']}({curr_sm}) "
                    f"[{'POS' if is_positive else 'NEG'}]"
                )

                states_list = []
                actions_list = []

                # per ogni step j->j+1 fino al flip
                for j in range(last_flip, i):
                    gA = graphs_meta[j]['graph']
                    gB = graphs_meta[j+1]['graph']
                    sid = graphs_meta[j]['step_id']

                    E0 = set(map(tuple, gA.edge_index.t().tolist()))
                    E1 = set(map(tuple, gB.edge_index.t().tolist()))

                    # RemoveEdge (op=0)
                    for u,v in E0 - E1:
                        states_list.append(gA)
                        actions_list.append([u, v, 0, 0, sid])

                    # AddEdge (op=1)
                    for u,v in E1 - E0:
                        states_list.append(gA)
                        actions_list.append([u, v, 1, 0, sid])

                    # ExtractMethod (nodo aggiunto, op=2)
                    if gB.num_nodes > gA.num_nodes:
                        states_list.append(gA)
                        actions_list.append([0, 0, 2, 0, sid])

                # Termination action sulla versione flip (op=3)
                flip_sid = graphs_meta[i]['step_id']
                states_list.append(graphs_meta[i]['graph'])
                actions_list.append([0, 0, 3, 1, flip_sid])

                # Salva sequenza
                sequences.append({
                    'project':     project,
                    'component':   comp,
                    'comp_index':  graphs_meta[i]['comp_index'],
                    'start_idx':   graphs_meta[last_flip]['version_idx'],
                    'end_idx':     graphs_meta[i]['version_idx'],
                    'start_state': prev_sm,
                    'end_state':   curr_sm,
                    'is_positive': is_positive,
                    'states':      states_list,                      # lista di Data
                    'actions':     torch.tensor(actions_list, dtype=torch.long)
                })

                last_flip = i

        logger.info(f" Fine gruppo {project}/{comp}: {flip_count} flip, tot seq={len(sequences)}")

    pos = sum(1 for s in sequences if s['is_positive'])
    neg = len(sequences) - pos
    logger.info(f"Generated {len(sequences)} sequences (pos={pos}, neg={neg})")
    return sequences

if __name__ == '__main__':
    DATA_DIR = 'data/dataset_graph_feature'
    seqs = build_expert_sequences(DATA_DIR)
    os.makedirs('data', exist_ok=True)
    torch.save(seqs, 'data/expert_sequences.pt')
    logger.info("Salvato data/expert_sequences.pt con stati e azioni")
