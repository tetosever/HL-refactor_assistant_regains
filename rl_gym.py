"""
Ambiente di reinforcement learning per la rifattorizzazione automatica
di sub-graph 1-hop di dependency graph usando PyTorch Geometric.

REFACTORED VERSION - Focus su hub-centric metrics e performance ottimizzate.

PRINCIPALI MIGLIORAMENTI:
- Hub score basato su metriche correlate con smelliness (degree_centrality, pagerank, closeness_centrality)
- Rimozione di calcoli inutili (modularity, clustering, betweenness globali)
- Sistema di tracking hub più robusto
- Gestione errori migliorata
- Documentazione completa
"""

import gym
from gym import spaces
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from functools import lru_cache

# Sopprime warnings non critici per output più pulito
warnings.filterwarnings('ignore')

# Feature standard per ogni nodo (mantenute per compatibilità con discriminator)
HUB_FEATURES = [
    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
    'pagerank', 'betweenness_centrality', 'closeness_centrality'
]

class HubTracker:
    """
    Sistema robusto per il tracking dell'hub originale attraverso modifiche al grafo.

    NUOVO: Classe dedicata per gestire il tracking dell'hub in modo più pulito e robusto.
    """

    def __init__(self, initial_hub_idx: int):
        """
        Inizializza il tracker con l'hub originale.

        Args:
            initial_hub_idx: Indice iniziale del nodo hub
        """
        self.original_hub_idx = initial_hub_idx
        self.original_hub_id = f"hub_original_{initial_hub_idx}"
        self.current_hub_idx = initial_hub_idx
        self.hub_lost = False

        # Sistema di mapping per tutti i nodi
        self.node_id_mapping = {}  # stable_id -> current_index
        self.reverse_id_mapping = {}  # current_index -> stable_id
        self.next_node_id = 0

    def initialize_tracking(self, num_nodes: int):
        """
        Inizializza il sistema di tracking per tutti i nodi.

        Args:
            num_nodes: Numero totale di nodi nel grafo
        """
        self.node_id_mapping.clear()
        self.reverse_id_mapping.clear()
        self.next_node_id = 0

        # Assegna ID stabili a tutti i nodi esistenti
        for current_index in range(num_nodes):
            stable_id = f"node_{self.next_node_id}"
            self.node_id_mapping[stable_id] = current_index
            self.reverse_id_mapping[current_index] = stable_id
            self.next_node_id += 1

        # Memorizza l'ID stabile dell'hub originale
        self.original_hub_id = self.reverse_id_mapping[self.original_hub_idx]

    def get_current_hub_index(self, data: Data) -> int:
        """
        Trova l'indice corrente del nodo hub originale.

        Args:
            data: Oggetto Data PyG corrente

        Returns:
            Indice corrente dell'hub originale, o fallback se perso
        """
        if self.original_hub_id is None:
            return self._find_fallback_hub(data)

        current_index = self.node_id_mapping.get(self.original_hub_id, None)

        if current_index is None or current_index >= data.num_nodes:
            print(f"⚠️ Hub originale {self.original_hub_id} perso! Usando fallback.")
            self.hub_lost = True
            return self._find_fallback_hub(data)

        return current_index

    def _find_fallback_hub(self, data: Data) -> int:
        """
        Trova un hub di fallback quando quello originale è perso.

        Args:
            data: Oggetto Data PyG corrente

        Returns:
            Indice del nodo con grado massimo come hub di fallback
        """
        if data.edge_index.size(1) > 0:
            degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
            return degrees.argmax().item()
        else:
            return 0

    def update_after_node_addition(self, num_new_nodes: int, start_index: int):
        """
        Aggiorna mapping quando vengono aggiunti nuovi nodi.

        Args:
            num_new_nodes: Numero di nodi aggiunti
            start_index: Indice di partenza per i nuovi nodi
        """
        for i in range(num_new_nodes):
            new_index = start_index + i
            stable_id = f"node_{self.next_node_id}"
            self.node_id_mapping[stable_id] = new_index
            self.reverse_id_mapping[new_index] = stable_id
            self.next_node_id += 1

    def rebuild_mapping(self, old_num_nodes: int, new_num_nodes: int):
        """
        Ricostruisce il mapping dopo modifiche al grafo.

        Args:
            old_num_nodes: Numero precedente di nodi
            new_num_nodes: Numero attuale di nodi
        """
        new_mapping = {}
        new_reverse_mapping = {}

        # Mantieni mapping per nodi esistenti
        for old_index in range(min(old_num_nodes, new_num_nodes)):
            if old_index in self.reverse_id_mapping:
                stable_id = self.reverse_id_mapping[old_index]
                new_mapping[stable_id] = old_index
                new_reverse_mapping[old_index] = stable_id

        # Gestisci nuovi nodi se presenti
        if new_num_nodes > old_num_nodes:
            self.update_after_node_addition(
                new_num_nodes - old_num_nodes,
                old_num_nodes
            )
            # Aggiungi i nuovi mapping
            for new_index in range(old_num_nodes, new_num_nodes):
                if new_index in self.reverse_id_mapping:
                    stable_id = self.reverse_id_mapping[new_index]
                    new_mapping[stable_id] = new_index
                    new_reverse_mapping[new_index] = stable_id

        self.node_id_mapping = new_mapping
        self.reverse_id_mapping = new_reverse_mapping


class RefactorEnv(gym.Env):
    """
    Ambiente OpenAI Gym per la rifattorizzazione di sub-graph 1-hop.

    MIGLIORAMENTI:
    - Hub score ottimizzato basato su correlazioni con smelliness
    - Sistema di tracking hub più robusto
    - Rimozione di calcoli inutili
    - Gestione errori migliorata
    """

    def __init__(self,
                 data_path: str,
                 discriminator=None,
                 max_steps: int = 20,
                 reward_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Inizializza l'ambiente di refactoring.

        Args:
            data_path: Percorso ai dati di training
            discriminator: Modello discriminator opzionale per adversarial training
            max_steps: Numero massimo di step per episodio
            reward_weights: Pesi personalizzati per reward components
            device: Device PyTorch da utilizzare
        """
        super(RefactorEnv, self).__init__()

        self.device = device
        self.max_steps = max_steps
        self.discriminator = discriminator

        self.reward_weights = reward_weights or {
            'hub_weight': 10.0,  # Peso principale per hub improvement
            'step_valid': 0.05,  # Bonus per azioni valide
            'step_invalid': -0.1,  # Penalty per azioni invalide
            'time_penalty': -0.01,  # Penalty per ogni step
            'early_stop_penalty': -0.5,  # Penalty per STOP prematuro
            'cycle_penalty': -0.2,  # Penalty per cicli
            'duplicate_penalty': -0.1,  # Penalty per archi duplicati
            'adversarial_weight': 2.0,  # Peso per adversarial reward
            'patience': 15  # Steps senza miglioramento prima di early stop
        }

        # Tracking delle performance
        self.best_hub_score = 0.0
        self.no_improve_steps = 0
        self.disc_start = 0.5
        self.prev_disc_score = None

        # Carica e preprocessa i dati
        print("🔄 Caricamento e preprocessing dati...")
        self.original_data_list = self._load_and_preprocess_data(data_path)

        # Stato dell'ambiente
        self.current_data = None
        self.current_step = 0
        self.initial_metrics = {}
        self.prev_hub_score = 0.0

        # NUOVO: Hub tracker robusto
        self.hub_tracker = None

        # Action space: 7 azioni + STOP
        self.num_actions = 7
        self.action_space = spaces.Discrete(self.num_actions)

        # MIGLIORATO: Spazio di osservazione ottimizzato
        max_nodes = max([data.num_nodes for data in self.original_data_list])
        self.max_nodes = max_nodes

        # AGGIORNATO: Solo 4 global features invece di 10 (hub_score, num_nodes, num_edges, connected)
        obs_dim = max_nodes * 7 + max_nodes * max_nodes + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # Inizializza scaler per normalizzazione
        self.feature_scaler = None
        self._fit_feature_scaler()

        print(f"✅ Ambiente inizializzato: {len(self.original_data_list)} grafi, max_nodes={max_nodes}")

    def _get_discriminator_score(self) -> Optional[float]:
        """Ottieni score discriminator corrente"""
        if not hasattr(self, 'discriminator') or self.discriminator is None:
            return None

        try:
            with torch.no_grad():
                disc_output = self.discriminator(self.current_data)
                if isinstance(disc_output, dict):
                    p_smelly = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                else:
                    p_smelly = torch.softmax(disc_output, dim=1)[0, 1].item()
                return p_smelly
        except Exception:
            return None

    def _hub_potential_reward(self, prev_hub_score: float, current_hub_score: float) -> float:
        """Potential-based reward per hub score"""
        improvement = prev_hub_score - current_hub_score  # Positivo = miglioramento

        hub_weight = self.reward_weights.get('hub_weight', 10.0)
        hub_reward = hub_weight * improvement

        # Clipping SOFT per preservare gradiente
        return np.clip(hub_reward, -2.0, 2.0)

    def _adversarial_potential_reward(self, prev_disc_score: Optional[float],
                                      current_disc_score: Optional[float]) -> float:
        """Adversarial reward per step"""
        if prev_disc_score is None or current_disc_score is None:
            return 0.0

        # Riduzione in probabilità "smelly" = miglioramento
        disc_improvement = prev_disc_score - current_disc_score

        adv_weight = self.reward_weights.get('adversarial_weight', 2.0)
        adv_reward = adv_weight * disc_improvement

        return np.clip(adv_reward, -1.0, 1.0)

    def _action_reward(self, action: int, success: bool) -> float:
        """Reward per validità azione"""
        if success and action != 6:
            return self.reward_weights.get('step_valid', 0.05)
        elif action == 6:
            return 0.0  # STOP neutro
        else:
            return self.reward_weights.get('step_invalid', -0.1)

    def _anti_stop_penalty(self, action: int) -> float:
        """Penalità per STOP prematuro senza miglioramento"""
        if (action == 6 and
                self.current_step <= 2 and
                self.best_hub_score >= self.initial_metrics['hub_score'] - 0.001):
            return self.reward_weights.get('early_stop_penalty', -0.5)
        return 0.0

    def _fit_feature_scaler(self):
        """
        OTTIMIZZATO: Fit dello scaler sulle feature di un campione rappresentativo.
        """
        print("📊 Training feature scaler...")

        all_features = []
        sample_size = min(100, len(self.original_data_list))

        # Campiona in modo efficiente
        sampled_indices = np.random.choice(len(self.original_data_list), sample_size, replace=False)

        for idx in sampled_indices:
            data = self.original_data_list[idx]
            # Converti a NetworkX mantenendo orientamento
            G = to_networkx(data, to_undirected=False)
            node_features = self._compute_node_features(G)

            for node_feats in node_features.values():
                feature_vector = [node_feats[feat] for feat in HUB_FEATURES]
                all_features.append(feature_vector)

        if all_features:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(np.array(all_features))
            print(f"✅ Scaler trained su {len(all_features)} samples")
        else:
            print("⚠️ Nessuna feature trovata per training scaler")

    @staticmethod
    def _compute_centrality_metrics(G: nx.DiGraph) -> Tuple[Dict, Dict, Dict]:
        """
        OTTIMIZZATO: Calcola metriche di centralità in modo efficiente.

        Per grafi grandi (>100 nodi) usa approssimazioni veloci,
        per grafi piccoli calcola metriche esatte.

        Args:
            G: Grafo NetworkX diretto

        Returns:
            Tuple di (pagerank, betweenness, closeness) dictionaries
        """
        num_nodes = len(G)

        if num_nodes <= 100:
            # Calcolo esatto per grafi piccoli
            try:
                pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
                betweenness = nx.betweenness_centrality(G, normalized=True)
                closeness = nx.closeness_centrality(G)
            except:
                # Fallback in caso di errore
                pagerank = {n: 1.0 / num_nodes for n in G.nodes()}
                betweenness = {n: 0.0 for n in G.nodes()}
                closeness = {n: 1.0 / max(num_nodes - 1, 1) for n in G.nodes()}
        else:
            # Approssimazioni veloci per grafi grandi
            total_edges = G.number_of_edges()
            pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
            betweenness = {n: 0.0 for n in G.nodes()}  # Troppo costoso per grafi grandi
            closeness = {n: 1.0 / max(num_nodes - 1, 1) for n in G.nodes()}

        return pagerank, betweenness, closeness

    @staticmethod
    def _compute_node_features(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """
        MANTENUTO: Calcola feature per tutti i nodi (necessario per discriminator).

        Args:
            G: Grafo NetworkX diretto

        Returns:
            Dictionary con feature per ogni nodo
        """
        if not G.is_directed():
            raise ValueError("Grafo deve essere diretto")

        # Ottieni gradi in/out
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        # Calcola centralità
        pagerank, betweenness, closeness = RefactorEnv._compute_centrality_metrics(G)

        node_features = {}
        num_nodes = len(G)

        for node in G.nodes():
            fan_in = float(in_degrees.get(node, 0))
            fan_out = float(out_degrees.get(node, 0))
            total_degree = fan_in + fan_out

            node_features[str(node)] = {
                'fan_in': fan_in,
                'fan_out': fan_out,
                'degree_centrality': total_degree / max(num_nodes - 1, 1),
                'in_out_ratio': fan_in / (fan_out + 1e-8),
                'pagerank': float(pagerank.get(node, 0)),
                'betweenness_centrality': float(betweenness.get(node, 0)),
                'closeness_centrality': float(closeness.get(node, 0))
            }

        return node_features

    def compute_hub_score_from_tensor(self, data: Data, hub_index: int) -> float:
        """
        NUOVO: Hub score ottimizzato basato sulle correlazioni osservate con smelliness.

        Usa direttamente le feature già calcolate in data.x, zero overhead computazionale.
        Le metriche sono pesate secondo le correlazioni dalla matrice di correlazione:
        - degree_centrality: peso 0.35 (correlazione ~0.8-0.9 con smelly)
        - pagerank: peso 0.25 (correlazione ~0.6-0.7 con smelly)
        - closeness_centrality: peso 0.10 (correlazione ~0.6-0.7 con smelly)
        - total_degree: peso 0.30 (strategia Arcan per hub detection)

        Args:
            data: Oggetto Data PyG
            hub_index: Indice del nodo hub

        Returns:
            Hub score normalizzato in [0,1]
        """
        if hub_index >= data.num_nodes or hub_index < 0:
            return 0.0

        # Estrai feature del nodo hub da data.x (zero overhead!)
        hub_features = data.x[hub_index]  # Tensor [7] con le feature

        # Mapping delle feature secondo HUB_FEATURES:
        # [0]='fan_in', [1]='fan_out', [2]='degree_centrality', [3]='in_out_ratio',
        # [4]='pagerank', [5]='betweenness_centrality', [6]='closeness_centrality'

        fan_in = hub_features[0].item()
        fan_out = hub_features[1].item()
        degree_centrality = hub_features[2].item()      # Già calcolata!
        pagerank_hub = hub_features[4].item()           # Già calcolata!
        closeness_centrality = hub_features[6].item()   # Già calcolata!

        # 1. Total degree normalizzato (strategia Arcan)
        total_degree = fan_in + fan_out
        max_possible_degree = 2 * max(data.num_nodes - 1, 1)
        normalized_total_degree = total_degree / max_possible_degree

        # 2. Combinazione pesata basata sulle correlazioni empiriche
        hub_score = (
            0.30 * normalized_total_degree +    # Total degree (Arcan strategy)
            0.35 * degree_centrality +          # Alta correlazione con smelly (~0.8-0.9)
            0.25 * pagerank_hub +               # Correlazione moderata (~0.6-0.7)
            0.10 * closeness_centrality         # Correlazione moderata (~0.6-0.7)
        )

        return float(np.clip(hub_score, 0.0, 1.0))

    def _calculate_metrics(self, data: Data) -> Dict[str, float]:
        """
        SEMPLIFICATO: Calcola SOLO le metriche necessarie.

        Rimosse tutte le metriche inutili (density, modularity, clustering, etc.)
        Focus solo su hub_score + info di base per monitoring.

        Args:
            data: Oggetto Data PyG

        Returns:
            Dictionary con metriche essenziali
        """
        try:
            # Ottieni hub corrente
            current_hub = self.hub_tracker.get_current_hub_index(data)

            # 🎯 METRICA PRINCIPALE: Hub score dalle feature esistenti (zero overhead!)
            hub_score = self.compute_hub_score_from_tensor(data, current_hub)

            # 📊 Info di base del grafo (per monitoring/debug)
            num_nodes = int(data.num_nodes)
            num_edges = int(data.edge_index.shape[1])

            # 🔗 Verifica connettività (validazione grafo)
            try:
                G = to_networkx(data, to_undirected=True)
                connected = float(nx.is_connected(G))
            except:
                connected = 0.0

            return {
                'hub_score': float(hub_score),  # ← QUESTA È L'UNICA CHE CONTA PER L'OBIETTIVO
                'num_nodes': num_nodes,         # ← Info di base
                'num_edges': num_edges,         # ← Info di base
                'connected': connected          # ← Validazione grafo
            }

        except Exception as e:
            print(f"❌ Errore nel calcolo metriche: {e}")
            return {
                'hub_score': 0.0,
                'num_nodes': 0,
                'num_edges': 0,
                'connected': 0.0
            }

    def _create_fresh_data_object(self, x: torch.Tensor, edge_index: torch.Tensor) -> Data:
        """
        MIGLIORATO: Crea oggetto Data con feature fresche e normalizzate.

        Ricalcola tutte le feature nodali basandosi sulla nuova struttura del grafo,
        garantendo coerenza tra grafo e feature.

        Args:
            x: Tensor delle feature nodali
            edge_index: Tensor degli archi

        Returns:
            Nuovo oggetto Data con feature aggiornate
        """
        try:
            num_nodes = x.size(0)

            # Crea grafo NetworkX temporaneo per calcolo feature
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))

            # Aggiungi archi validi
            if edge_index.numel() > 0:
                valid_edges = []
                for i in range(edge_index.size(1)):
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                        valid_edges.append((src, dst))

                if valid_edges:
                    G.add_edges_from(valid_edges)

            # Calcola feature fresche per tutti i nodi
            node_features = self._compute_node_features(G)

            # Costruisci matrice feature
            feature_matrix = []
            for node_id in range(num_nodes):
                if str(node_id) in node_features:
                    feature_vector = [node_features[str(node_id)][feat] for feat in HUB_FEATURES]
                else:
                    # Feature di default per nodi isolati
                    feature_vector = [0.0] * len(HUB_FEATURES)
                feature_matrix.append(feature_vector)

            feature_matrix = np.array(feature_matrix)

            # Normalizza usando scaler pre-trained
            if self.feature_scaler is not None:
                try:
                    feature_matrix = self.feature_scaler.transform(feature_matrix)
                except Exception as e:
                    print(f"⚠️ Normalizzazione fallita: {e}")
                    # Fallback a normalizzazione standard
                    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (
                        feature_matrix.std(axis=0) + 1e-8)

            # Crea nuovo oggetto Data pulito
            new_data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float32, device=self.device),
                edge_index=edge_index.clone(),
                num_nodes=num_nodes
            )

            return new_data

        except Exception as e:
            print(f"❌ Errore creazione Data object: {e}")
            # Fallback con feature zero
            return Data(
                x=torch.zeros((x.size(0), len(HUB_FEATURES)), dtype=torch.float32, device=self.device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                num_nodes=x.size(0)
            )

    def _rebuild_graph_with_fresh_data(self, new_x: torch.Tensor, new_edge_index: torch.Tensor) -> None:
        """
        MIGLIORATO: Ricostruisce completamente current_data con tracking robusto.

        Args:
            new_x: Nuove feature nodali
            new_edge_index: Nuovi archi
        """
        old_num_nodes = self.current_data.num_nodes
        num_nodes = new_x.size(0)
        self.max_nodes = max(self.max_nodes, num_nodes)

        # Filtra archi per validità
        valid_edges = []
        for i in range(new_edge_index.size(1)):
            src, dst = new_edge_index[0, i].item(), new_edge_index[1, i].item()
            if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                valid_edges.append([src, dst])

        if valid_edges:
            filtered_edge_index = torch.tensor(valid_edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            filtered_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        # Ricostruisci Data con feature fresche
        self.current_data = self._create_fresh_data_object(new_x, filtered_edge_index)

        # Aggiorna tracking nodi
        self.hub_tracker.rebuild_mapping(old_num_nodes, self.current_data.num_nodes)

    def _load_and_preprocess_data(self, data_path: str) -> List[Data]:
        """
        MANTENUTO: Carica e preprocessa i dati (richiesto per compatibilità).

        Args:
            data_path: Percorso alla directory con file .pt

        Returns:
            Lista di oggetti Data preprocessati
        """
        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory non trovata: {data_path}")

        pt_files = list(data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"Nessun file .pt trovato in {data_path}")

        print(f"📂 Trovati {len(pt_files)} file .pt")

        # Carica tutti i file .pt
        data_list = []
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location=self.device)

                # Gestisci diversi formati di file
                if isinstance(data, dict):
                    if 'data' in data:
                        graph_data = data['data']
                    else:
                        graph_data = Data(x=data['x'], edge_index=data['edge_index'])
                elif isinstance(data, Data):
                    graph_data = data
                else:
                    print(f"⚠️ Formato non riconosciuto in {pt_file}")
                    continue

                # Valida che abbia attributi necessari
                if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                    if graph_data.x.size(1) == 7:  # Deve avere 7 feature per nodo
                        data_list.append(graph_data)
                    else:
                        print(f"⚠️ {pt_file}: {graph_data.x.size(1)} feature invece di 7")
                else:
                    print(f"⚠️ {pt_file}: mancano attributi x o edge_index")

            except Exception as e:
                print(f"❌ Errore caricando {pt_file}: {e}")
                continue

        if not data_list:
            raise ValueError("Nessun dato valido caricato")

        # Normalizzazione globale delle feature
        print("🔄 Normalizzazione feature...")
        scaler = StandardScaler()
        all_features = torch.cat([data.x for data in data_list], dim=0)
        scaler.fit(all_features.cpu().numpy())

        processed_data = []
        for data in data_list:
            normalized_features = scaler.transform(data.x.cpu().numpy())

            clean_data = Data(
                x=torch.tensor(normalized_features, dtype=torch.float32, device=self.device),
                edge_index=data.edge_index.to(self.device),
                num_nodes=data.x.size(0)
            )
            processed_data.append(clean_data)

        print(f"✅ Processati {len(processed_data)} sub-grafi")
        return processed_data

    def reset(self, graph_idx: Optional[int] = None) -> np.ndarray:
        """
        AGGIORNATO: Reset ambiente con inizializzazione discriminator tracking
        """
        if graph_idx is None:
            graph_idx = np.random.randint(0, len(self.original_data_list))

        original_data = self.original_data_list[graph_idx]

        # Clona data mantenendo device
        self.current_data = Data(
            x=original_data.x.clone(),
            edge_index=original_data.edge_index.clone(),
            num_nodes=original_data.x.size(0)
        )

        self.current_step = 0

        # Trova hub iniziale
        if self.current_data.edge_index.size(1) > 0:
            degrees = torch.bincount(self.current_data.edge_index[0], minlength=self.current_data.num_nodes)
            initial_hub = degrees.argmax().item()
        else:
            initial_hub = 0

        # Inizializza hub tracker
        self.hub_tracker = HubTracker(initial_hub)
        self.hub_tracker.initialize_tracking(self.current_data.num_nodes)

        # Calcola metriche iniziali
        self.initial_metrics = self._calculate_metrics(self.current_data)
        self.prev_hub_score = self.initial_metrics['hub_score']
        self.best_hub_score = self.prev_hub_score
        self.no_improve_steps = 0

        # 🔧 NUOVO: Inizializza discriminator baseline per per-step tracking
        self.prev_disc_score = self._get_discriminator_score()
        self.disc_start = self.prev_disc_score if self.prev_disc_score is not None else 0.5

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        RISTRUTTURATO: Esegue azione con reward shaping per-step
        """
        if self.current_data is None:
            raise RuntimeError("Ambiente non inizializzato. Chiama reset() prima.")

        # Salva stato precedente per potential-based shaping
        prev_hub_score = self._calculate_metrics(self.current_data)['hub_score']
        prev_disc_score = self._get_discriminator_score()

        self.current_step += 1

        # Esegui azione
        success = self._apply_action(action)

        # Calcola stato corrente
        current_metrics = self._calculate_metrics(self.current_data)
        current_hub_score = current_metrics['hub_score']
        current_disc_score = self._get_discriminator_score()

        # NUOVA COMPOSIZIONE REWARD PER-STEP
        # 1. Hub potential reward (componente principale)
        hub_reward = self._hub_potential_reward(prev_hub_score, current_hub_score)

        # 2. Adversarial potential reward
        adversarial_reward = self._adversarial_potential_reward(prev_disc_score, current_disc_score)

        # 3. Action validity reward
        action_reward = self._action_reward(action, success)

        # 4. Time penalty (costante) - 🔧 FIX con get()
        time_penalty = self.reward_weights.get('time_penalty', -0.01)

        # 5. Structural penalties
        structural_penalty = self._check_structural_penalties() if success else 0.0

        # 6. Anti-STOP penalty
        anti_stop_penalty = self._anti_stop_penalty(action)

        # Reward totale per questo step
        total_reward = (hub_reward + adversarial_reward + action_reward +
                        time_penalty + structural_penalty + anti_stop_penalty)

        # Update tracking
        if current_hub_score < self.best_hub_score:
            self.best_hub_score = current_hub_score
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        # Determina terminazione
        done = (action == 6) or (self.current_step >= self.max_steps)

        # Early stopping con patience - 🔧 FIX con get()
        patience = self.reward_weights.get('patience', 15)
        if self.no_improve_steps >= patience:
            done = True

        # Info dettagliate per debugging
        hub_improvement_step = prev_hub_score - current_hub_score
        hub_improvement_total = self.initial_metrics['hub_score'] - current_hub_score

        info = {
            'action_success': success,
            'metrics': current_metrics,
            'step': self.current_step,

            # BREAKDOWN REWARD DETTAGLIATO
            'hub_reward': hub_reward,
            'adversarial_reward': adversarial_reward,
            'action_reward': action_reward,
            'time_penalty': time_penalty,
            'structural_penalty': structural_penalty,
            'anti_stop_penalty': anti_stop_penalty,
            'total_reward': total_reward,

            # Metriche per monitoring
            'hub_improvement_step': hub_improvement_step,
            'hub_improvement_total': hub_improvement_total,
            'current_hub_score': current_hub_score,
            'best_hub_score': self.best_hub_score,
            'no_improve_steps': self.no_improve_steps,
            'is_early_stop': action == 6 and self.current_step <= 2,
            'hub_lost': self.hub_tracker.hub_lost if self.hub_tracker else False,

            # Discriminator info
            'prev_disc_score': prev_disc_score,
            'current_disc_score': current_disc_score,
            'disc_improvement': (
                        prev_disc_score - current_disc_score) if prev_disc_score and current_disc_score else 0.0
        }

        return self._get_state(), total_reward, done, info

    def _apply_action(self, action: int) -> bool:
        """
        MANTENUTO: Applica l'azione specificata al grafo.

        Args:
            action: Azione da applicare (0-6)
                   0: RemoveEdge, 1: AddEdge, 2: MoveEdge
                   3: ExtractMethod, 4: ExtractAbstractUnit
                   5: ExtractUnit, 6: STOP

        Returns:
            True se l'azione è stata applicata con successo
        """
        try:
            if action == 0:
                return self._remove_edge()
            elif action == 1:
                return self._add_edge()
            elif action == 2:
                return self._move_edge()
            elif action == 3:
                return self._extract_method()
            elif action == 4:
                return self._extract_abstract_unit()
            elif action == 5:
                return self._extract_unit()
            elif action == 6:
                return True  # STOP - sempre valida
            else:
                return False
        except Exception as e:
            print(f"❌ Errore applicando azione {action}: {e}")
            return False

    def _remove_edge(self) -> bool:
        """
        MIGLIORATO: Rimuove arco dal hub con hub tracking robusto.

        Returns:
            True se l'arco è stato rimosso con successo
        """
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
        edge_index = self.current_data.edge_index

        # Trova archi uscenti dall'hub (escludendo self-loops)
        hub_edges = []
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u == current_hub and u != v:
                hub_edges.append(i)

        if not hub_edges:
            return False

        # Rimuovi arco casuale
        edge_to_remove = np.random.choice(hub_edges)
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device)
        mask[edge_to_remove] = False
        new_edge_index = edge_index[:, mask]

        # Verifica connettività
        if self._is_connected(new_edge_index):
            self._rebuild_graph_with_fresh_data(self.current_data.x, new_edge_index)
            return True

        return False

    def _add_edge(self) -> bool:
        """
        MIGLIORATO: Aggiunge arco dall'hub con hub tracking robusto.

        Returns:
            True se l'arco è stato aggiunto con successo
        """
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
        edge_index = self.current_data.edge_index

        # Trova nodi non connessi all'hub
        connected_nodes = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u == current_hub:
                connected_nodes.add(v)

        possible_targets = []
        for node in range(self.current_data.num_nodes):
            if node not in connected_nodes and node != current_hub:
                possible_targets.append(node)

        if not possible_targets:
            return False

        # Aggiungi arco a target casuale
        target = np.random.choice(possible_targets)
        new_edge = torch.tensor([[current_hub], [target]], dtype=torch.long, device=self.device)
        new_edge_index = torch.cat([edge_index, new_edge], dim=1)

        self._rebuild_graph_with_fresh_data(self.current_data.x, new_edge_index)
        return True

    def _move_edge(self) -> bool:
        """
        MANTENUTO: Sposta arco dell'hub (rimuovi + aggiungi).

        Returns:
            True se l'operazione è riuscita
        """
        return self._remove_edge() and self._add_edge()

    def _extract_method(self) -> bool:
        """
        MANTENUTO: ExtractMethod refactoring - crea nodo intermediario.

        Returns:
            True se l'operazione è riuscita
        """
        edge_index = self.current_data.edge_index

        if edge_index.shape[1] == 0:
            return False

        # Seleziona arco casuale u→v
        edge_idx = np.random.randint(0, edge_index.shape[1])
        u, v = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()

        if u == v:  # Skip self-loops
            return False

        # Crea feature per nuovo nodo method (media di u e v)
        u_features = self.current_data.x[u]
        v_features = self.current_data.x[v]
        method_features = ((u_features + v_features) / 2).unsqueeze(0)

        # Ricostruisci edge list: u→method→v
        method_idx = self.current_data.x.size(0)
        new_edges = []

        # Mantieni tutti gli archi eccetto u→v
        for i in range(edge_index.shape[1]):
            if i != edge_idx:
                new_edges.append((edge_index[0, i].item(), edge_index[1, i].item()))

        # Aggiungi nuovi archi: u→method, method→v
        new_edges.append((u, method_idx))
        new_edges.append((method_idx, v))

        # Riassegna alcune dipendenze di v al method
        v_incoming = [(src, dst) for src, dst in new_edges if dst == v and src != method_idx]
        if len(v_incoming) > 1:
            num_reassign = min(2, len(v_incoming) // 2)
            to_reassign = np.random.choice(len(v_incoming), num_reassign, replace=False)
            for idx in to_reassign:
                src, _ = v_incoming[idx]
                new_edges.remove((src, v))
                new_edges.append((src, method_idx))

        # Ricostruisci grafo
        new_x = torch.cat([self.current_data.x, method_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _extract_abstract_unit(self) -> bool:
        """
        MANTENUTO: ExtractAbstractUnit - crea astrazione per dipendenze comuni.

        Returns:
            True se l'operazione è riuscita
        """
        edge_index = self.current_data.edge_index

        if edge_index.shape[1] < 3:
            return False

        # Trova nodi con target comuni
        targets = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if dst not in targets:
                targets[dst] = []
            targets[dst].append(src)

        # Trova target con almeno 2 source diversi
        common_targets = [(dst, srcs) for dst, srcs in targets.items()
                         if len(set(srcs)) >= 2]

        if not common_targets:
            return False

        # Seleziona target casuale con dipendenze comuni
        target_dst, source_nodes = common_targets[np.random.randint(len(common_targets))]
        unique_sources = list(set(source_nodes))

        if len(unique_sources) < 2:
            return False

        # Seleziona subset di source per astrazione
        num_to_abstract = min(3, len(unique_sources))
        selected_sources = np.random.choice(unique_sources, num_to_abstract, replace=False)

        # Crea nodo astratto con feature medie
        abstract_idx = self.current_data.x.size(0)
        selected_features = self.current_data.x[selected_sources]
        abstract_features = selected_features.mean(dim=0, keepdim=True)

        # Ricostruisci edge list
        new_edges = []
        removed_edges = set()

        # Marca archi da rimuovere (selected_sources → target_dst)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in selected_sources and dst == target_dst:
                removed_edges.add(i)

        # Mantieni archi non rimossi
        for i in range(edge_index.shape[1]):
            if i not in removed_edges:
                new_edges.append((edge_index[0, i].item(), edge_index[1, i].item()))

        # Aggiungi nuove connessioni: abstract → target, sources → abstract
        new_edges.append((abstract_idx, target_dst))
        for src in selected_sources:
            new_edges.append((src, abstract_idx))

        # Ricostruisci grafo
        new_x = torch.cat([self.current_data.x, abstract_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _extract_unit(self) -> bool:
        """
        MIGLIORATO: ExtractUnit - divide responsabilità dell'hub.

        Returns:
            True se l'operazione è riuscita
        """
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data)

        if current_hub >= self.current_data.num_nodes:
            return False

        edge_index = self.current_data.edge_index

        # Trova tutti i successori dell'hub
        hub_neighbors = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src == current_hub and dst != current_hub:
                hub_neighbors.append(dst)

        hub_neighbors = list(set(hub_neighbors))  # Rimuovi duplicati

        if len(hub_neighbors) < 2:
            return False

        # Dividi successori in due gruppi
        mid_point = len(hub_neighbors) // 2
        group1 = hub_neighbors[:mid_point]
        group2 = hub_neighbors[mid_point:]

        if not group1 or not group2:
            return False

        # Crea due nuovi nodi unit
        unit1_idx = self.current_data.x.size(0)
        unit2_idx = unit1_idx + 1

        # Feature dei nuovi unit (basate sui loro gruppi + hub)
        hub_features = self.current_data.x[current_hub]
        group1_features = self.current_data.x[group1].mean(dim=0) if group1 else hub_features
        group2_features = self.current_data.x[group2].mean(dim=0) if group2 else hub_features

        # Media pesata con feature dell'hub
        unit1_features = ((hub_features + group1_features) / 2).unsqueeze(0)
        unit2_features = ((hub_features + group2_features) / 2).unsqueeze(0)

        # Ricostruisci edge list
        new_edges = []

        # Mantieni archi che non coinvolgono l'hub
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src != current_hub and dst != current_hub:
                new_edges.append((src, dst))
            elif src == current_hub and dst in group1:
                new_edges.append((unit1_idx, dst))  # Unit1 → group1
            elif src == current_hub and dst in group2:
                new_edges.append((unit2_idx, dst))  # Unit2 → group2
            elif dst == current_hub:
                new_edges.append((src, dst))  # Mantieni incoming all'hub

        # Connetti hub ai nuovi unit
        new_edges.append((current_hub, unit1_idx))
        new_edges.append((current_hub, unit2_idx))

        # Ricostruisci grafo
        new_features = torch.cat([unit1_features, unit2_features], dim=0)
        new_x = torch.cat([self.current_data.x, new_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _check_structural_penalties(self) -> float:
        """
        AGGIORNATO: Verifica penalità per problemi strutturali con fallback
        """
        penalty = 0.0

        try:
            # Controlla cicli
            G = to_networkx(self.current_data, to_undirected=False)

            # Verifica cicli (early exit)
            try:
                next(nx.simple_cycles(G))  # Se trova un ciclo, alza StopIteration
                penalty += self.reward_weights.get('cycle_penalty', -0.2)  # 🔧 FIX
            except StopIteration:
                pass  # Nessun ciclo trovato

            # Controlla archi duplicati
            seen_edges = set()
            for u, v in self.current_data.edge_index.t().tolist():
                edge = (u, v)
                if edge in seen_edges:
                    penalty += self.reward_weights.get('duplicate_penalty', -0.1)
                    break
                seen_edges.add(edge)

        except Exception:
            # In caso di errore, applica penalty conservativa
            penalty += self.reward_weights.get('cycle_penalty', -0.2)

        return penalty

    def _is_connected(self, edge_index: torch.Tensor) -> bool:
        """
        OTTIMIZZATO: Verifica connettività del grafo.

        Args:
            edge_index: Tensor degli archi

        Returns:
            True se il grafo è debolmente connesso
        """
        try:
            if edge_index.size(1) == 0:
                return self.current_data.num_nodes <= 1

            temp_data = Data(edge_index=edge_index, num_nodes=self.current_data.num_nodes)
            G = to_networkx(temp_data, to_undirected=False)
            return nx.is_weakly_connected(G)
        except:
            return False

    def _calculate_comprehensive_final_reward(self) -> float:
        """
        MIGLIORATO: Calcola reward finale completo.

        COMPONENTI:
        1. Hub improvement reward (principale)
        2. Efficiency bonus (terminazione rapida)
        3. Adversarial reward (se discriminator presente)

        Returns:
            Reward finale totale
        """
        try:
            # ═══ 1. HUB IMPROVEMENT REWARD ═══
            initial_score = self.initial_metrics['hub_score']
            best_score = self.best_hub_score  # Usa il migliore dell'episodio
            improvement = initial_score - best_score  # Positivo = miglioramento

            improvement_reward = self._calculate_improvement_reward(improvement)

            # ═══ 2. EFFICIENCY BONUS ═══
            efficiency_bonus = 0.0
            if improvement > 0:  # Solo se c'è stato miglioramento
                steps_saved = self.max_steps - self.current_step
                efficiency_bonus = steps_saved * 0.1

            # ═══ 3. ADVERSARIAL REWARD ═══
            adversarial_reward = 0.0
            adversarial_info = {}

            if hasattr(self, 'discriminator') and self.discriminator is not None:
                adversarial_reward, adversarial_info = self._calculate_adversarial_reward()

            # ═══ 4. COMBINAZIONE FINALE ═══
            total_final_reward = improvement_reward + efficiency_bonus + adversarial_reward
            total_final_reward = np.clip(total_final_reward, -20.0, 50.0)  # Clamp per stabilità

            # Log dettagliato per debugging
            if abs(improvement) > 1e-6 or total_final_reward != 0:
                self._log_final_reward_breakdown({
                    'improvement': improvement,
                    'improvement_reward': improvement_reward,
                    'efficiency_bonus': efficiency_bonus,
                    'adversarial_reward': adversarial_reward,
                    'adversarial_info': adversarial_info,
                    'total': total_final_reward
                })

            return float(total_final_reward)

        except Exception as e:
            print(f"❌ Errore nel calcolo reward finale: {e}")
            return 0.0

    def _calculate_improvement_reward(self, improvement: float) -> float:
        """
        SEMPLIFICATO: Reward scalato per miglioramento hub score.

        Usa funzione tanh per smoothness invece di threshold rigidi.

        Args:
            improvement: Miglioramento hub score (positivo = miglioramento)

        Returns:
            Reward per il miglioramento
        """
        if improvement > 0:
            # Reward crescente con saturazione
            return 20.0 * np.tanh(improvement * 20.0)  # Max ~20 per improvement grandi
        else:
            # Penalty per peggioramento (più severa)
            return 30.0 * np.tanh(improvement * 30.0)  # Max penalty ~-30

    def _calculate_adversarial_reward(self) -> Tuple[float, Dict]:
        """
        MANTENUTO: Calcola reward adversariale dal discriminator.

        Returns:
            Tuple di (adversarial_reward, info_dict)
        """
        try:
            with torch.no_grad():
                # Forward pass discriminator
                disc_output = self.discriminator(self.current_data)
                logits = disc_output['logits'] if isinstance(disc_output, dict) else disc_output

                # Probabilità "smelly" finale
                p_smelly_final = torch.softmax(logits, dim=1)[0, 1].item()

                # Improvement: inizio → fine (positivo = meno smelly)
                disc_improvement = self.disc_start - p_smelly_final

                # Scala reward
                adv_weight = self.reward_weights.get('adversarial_weight', 0.15)
                raw_adversarial = adv_weight * disc_improvement * 10.0
                adversarial_reward = np.clip(raw_adversarial, -3.0, 3.0)

                # Info per debugging
                info = {
                    'p_smelly_start': self.disc_start,
                    'p_smelly_final': p_smelly_final,
                    'discriminator_improvement': disc_improvement,
                    'direction': 'positive' if disc_improvement > 0 else 'negative',
                    'magnitude': abs(disc_improvement),
                    'raw_reward': raw_adversarial,
                    'clipped_reward': adversarial_reward,
                    'weight_used': adv_weight
                }

                return adversarial_reward, info

        except Exception as e:
            error_info = {
                'error': str(e),
                'fallback_used': True,
                'p_smelly_start': self.disc_start,
                'p_smelly_final': 0.5
            }
            return 0.0, error_info

    def _log_final_reward_breakdown(self, breakdown: Dict):
        """
        MANTENUTO: Log dettagliato per debugging reward.

        Args:
            breakdown: Dictionary con componenti del reward
        """
        episode_num = getattr(self, 'current_episode', '?')

        print(f"\n🎯 FINAL REWARD BREAKDOWN (Episode {episode_num}):")
        print(f"   Hub Improvement: {breakdown['improvement']:.6f}")
        print(f"   → Improvement Reward: {breakdown['improvement_reward']:.3f}")
        print(f"   → Efficiency Bonus: {breakdown['efficiency_bonus']:.3f}")
        print(f"   → Adversarial Reward: {breakdown['adversarial_reward']:.3f}")

        if breakdown['adversarial_info']:
            adv_info = breakdown['adversarial_info']
            print(f"      • p_smelly: {adv_info.get('p_smelly_start', 0):.3f} → "
                  f"{adv_info.get('p_smelly_final', 0):.3f}")
            print(f"      • Direction: {adv_info.get('direction', 'unknown')} "
                  f"(Δ={adv_info.get('discriminator_improvement', 0):.3f})")

        print(f"   🎯 TOTAL FINAL: {breakdown['total']:.3f}")

    def _get_state(self) -> np.ndarray:
        """
        OTTIMIZZATO: Estrae stato con global features semplificate.

        Returns:
            Array numpy con stato dell'ambiente
        """
        try:
            # Numero reale di nodi
            real_num_nodes = self.current_data.x.size(0)

            # Node features con padding
            node_features = torch.zeros(self.max_nodes, 7, device=self.current_data.x.device)
            node_features[:real_num_nodes] = self.current_data.x

            # Adjacency matrix con padding
            adj_matrix = torch.zeros(self.max_nodes, self.max_nodes, device=self.current_data.edge_index.device)
            for u, v in self.current_data.edge_index.t().tolist():
                if u < self.max_nodes and v < self.max_nodes:
                    adj_matrix[u, v] = 1.0

            # NUOVO: Global features semplificate (solo 4 invece di 10)
            metrics = self._calculate_metrics(self.current_data)
            global_features = torch.tensor([
                metrics['hub_score'],     # Metrica principale
                real_num_nodes,          # Info strutturale
                metrics['num_edges'],    # Info strutturale
                metrics['connected']     # Validazione
            ], dtype=torch.float32, device=node_features.device)

            # Concatena tutte le componenti
            state = torch.cat([
                node_features.flatten(),
                adj_matrix.flatten(),
                global_features
            ])

            return state.cpu().numpy()

        except Exception as e:
            print(f"❌ Errore creazione stato: {e}")
            # Fallback con stato vuoto
            fallback_size = self.observation_space.shape[0]
            return np.zeros(fallback_size, dtype=np.float32)

    def render(self, mode: str = 'human'):
        """
        MIGLIORATO: Visualizzazione stato con info hub tracking.

        Args:
            mode: Modalità di rendering
        """
        if self.current_data is None:
            print("❌ Ambiente non inizializzato")
            return

        metrics = self._calculate_metrics(self.current_data)
        current_hub = self.hub_tracker.get_current_hub_index(self.current_data) if self.hub_tracker else 0

        print(f"\n{'='*50}")
        print(f"📊 STATO AMBIENTE - Step {self.current_step}")
        print(f"{'='*50}")
        print(f"🎯 Hub: ID {self.hub_tracker.original_hub_id if self.hub_tracker else 'N/A'} → indice {current_hub}")
        print(f"📈 Hub Score: {metrics['hub_score']:.4f}")
        print(f"📊 Grafo: {metrics['num_nodes']} nodi, {metrics['num_edges']} archi")
        print(f"🔗 Connesso: {'✅' if metrics['connected'] else '❌'}")

        if self.hub_tracker:
            print(f"🗺️  Nodi tracciati: {len(self.hub_tracker.node_id_mapping)}")
            print(f"⚠️  Hub perso: {'Sì' if self.hub_tracker.hub_lost else 'No'}")

        print(f"🏆 Best hub score: {self.best_hub_score:.4f}")
        print(f"⏱️  Step senza miglioramenti: {self.no_improve_steps}")
        print(f"{'='*50}")

    def get_hub_info(self) -> Dict:
        """
        NUOVO: Restituisce informazioni dettagliate sull'hub corrente.

        Returns:
            Dictionary con info hub per debugging/monitoring
        """
        if not self.hub_tracker or self.current_data is None:
            return {}

        current_hub_idx = self.hub_tracker.get_current_hub_index(self.current_data)

        info = {
            'original_hub_id': self.hub_tracker.original_hub_id,
            'current_hub_index': current_hub_idx,
            'hub_lost': self.hub_tracker.hub_lost,
            'hub_score': self.compute_hub_score_from_tensor(self.current_data, current_hub_idx),
            'total_nodes_tracked': len(self.hub_tracker.node_id_mapping)
        }

        # Aggiungi feature dell'hub se valido
        if current_hub_idx < self.current_data.num_nodes:
            hub_features = self.current_data.x[current_hub_idx]
            info.update({
                'hub_fan_in': hub_features[0].item(),
                'hub_fan_out': hub_features[1].item(),
                'hub_degree_centrality': hub_features[2].item(),
                'hub_pagerank': hub_features[4].item(),
                'hub_closeness_centrality': hub_features[6].item()
            })

        return info

    def validate_state_consistency(self) -> bool:
        """
        NUOVO: Valida la consistenza dello stato interno.

        Returns:
            True se lo stato è consistente
        """
        try:
            if self.current_data is None or self.hub_tracker is None:
                return False

            # Verifica dimensioni
            if self.current_data.x.size(0) != self.current_data.num_nodes:
                print("❌ Inconsistenza: x.size(0) != num_nodes")
                return False

            # Verifica hub tracking
            current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
            if current_hub >= self.current_data.num_nodes:
                print(f"❌ Inconsistenza: hub index {current_hub} >= num_nodes {self.current_data.num_nodes}")
                return False

            # Verifica edge_index validità
            if self.current_data.edge_index.size(1) > 0:
                max_node_in_edges = self.current_data.edge_index.max().item()
                if max_node_in_edges >= self.current_data.num_nodes:
                    print(f"❌ Inconsistenza: edge references node {max_node_in_edges} >= num_nodes {self.current_data.num_nodes}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Errore validazione consistenza: {e}")
            return False

    def get_action_mask(self) -> np.ndarray:
        """
        NUOVO: Restituisce maschera per azioni valide nello stato corrente.

        Utile per algoritmi che supportano action masking.

        Returns:
            Array boolean delle azioni disponibili
        """
        if self.current_data is None:
            return np.ones(self.num_actions, dtype=bool)

        mask = np.ones(self.num_actions, dtype=bool)

        try:
            current_hub = self.hub_tracker.get_current_hub_index(self.current_data)
            edge_index = self.current_data.edge_index

            # Action 0 (RemoveEdge): richiede archi uscenti dall'hub
            hub_outgoing = 0
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u == current_hub and u != v:
                    hub_outgoing += 1
                    break
            mask[0] = (hub_outgoing > 0)

            # Action 1 (AddEdge): richiede nodi non connessi all'hub
            connected_nodes = set()
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u == current_hub:
                    connected_nodes.add(v)

            available_targets = self.current_data.num_nodes - len(connected_nodes) - 1  # -1 per l'hub stesso
            mask[1] = (available_targets > 0)

            # Action 2 (MoveEdge): richiede sia remove che add possibili
            mask[2] = mask[0] and mask[1]

            # Action 3 (ExtractMethod): richiede almeno un arco
            mask[3] = (edge_index.shape[1] > 0)

            # Action 4 (ExtractAbstractUnit): richiede almeno 3 archi
            mask[4] = (edge_index.shape[1] >= 3)

            # Action 5 (ExtractUnit): richiede almeno 2 successori dell'hub
            hub_successors = set()
            for i in range(edge_index.shape[1]):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u == current_hub and u != v:
                    hub_successors.add(v)
            mask[5] = (len(hub_successors) >= 2)

            # Action 6 (STOP): sempre disponibile
            mask[6] = True

        except Exception as e:
            print(f"⚠️ Errore calcolo action mask: {e}")
            # In caso di errore, permetti tutte le azioni
            mask = np.ones(self.num_actions, dtype=bool)

        return mask

    def get_performance_stats(self) -> Dict:
        """
        NUOVO: Restituisce statistiche di performance per monitoring.

        Returns:
            Dictionary con statistiche utili per il training
        """
        if self.current_data is None:
            return {}

        current_metrics = self._calculate_metrics(self.current_data)

        stats = {
            # Metriche principali
            'current_hub_score': current_metrics['hub_score'],
            'initial_hub_score': self.initial_metrics.get('hub_score', 0.0),
            'best_hub_score': self.best_hub_score,
            'hub_improvement': self.initial_metrics.get('hub_score', 0.0) - self.best_hub_score,

            # Progresso episodio
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'progress': self.current_step / self.max_steps,
            'no_improve_steps': self.no_improve_steps,

            # Info strutturali
            'num_nodes': current_metrics['num_nodes'],
            'num_edges': current_metrics['num_edges'],
            'graph_connected': bool(current_metrics['connected']),

            # Hub tracking
            'hub_lost': self.hub_tracker.hub_lost if self.hub_tracker else False,
            'nodes_tracked': len(self.hub_tracker.node_id_mapping) if self.hub_tracker else 0,

            # Discriminator info (se presente)
            'discriminator_available': hasattr(self, 'discriminator') and self.discriminator is not None,
            'disc_start': self.disc_start
        }

        # Aggiungi info hub se tracking valido
        if self.hub_tracker and not self.hub_tracker.hub_lost:
            hub_info = self.get_hub_info()
            stats.update({
                'hub_fan_in': hub_info.get('hub_fan_in', 0),
                'hub_fan_out': hub_info.get('hub_fan_out', 0),
                'hub_total_degree': hub_info.get('hub_fan_in', 0) + hub_info.get('hub_fan_out', 0)
            })

        return stats

    def close(self):
        """
        NUOVO: Cleanup risorse dell'ambiente.
        """
        # Cleanup tracking
        if hasattr(self, 'hub_tracker') and self.hub_tracker:
            self.hub_tracker.node_id_mapping.clear()
            self.hub_tracker.reverse_id_mapping.clear()

        # Clear data references
        self.current_data = None
        self.original_data_list = None

        print("🧹 Ambiente chiuso e risorse liberate")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE HELPER PER TESTING E DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════════

class RefactorEnvTester:
    """
    NUOVO: Classe helper per testing e debugging dell'ambiente.

    Fornisce metodi per validare il comportamento dell'ambiente,
    testare azioni specifiche e raccogliere statistiche.
    """

    def __init__(self, env: RefactorEnv):
        """
        Inizializza il tester.

        Args:
            env: Istanza di RefactorEnv da testare
        """
        self.env = env
        self.test_results = []

    def test_basic_functionality(self) -> Dict:
        """
        Test di funzionalità base dell'ambiente.

        Returns:
            Dictionary con risultati dei test
        """
        results = {
            'reset_test': False,
            'step_test': False,
            'state_consistency': False,
            'hub_tracking': False,
            'action_validity': False
        }

        try:
            # Test reset
            initial_state = self.env.reset()
            results['reset_test'] = (initial_state is not None and
                                   initial_state.shape == self.env.observation_space.shape)

            # Test step con azione STOP
            next_state, reward, done, info = self.env.step(6)  # STOP action
            results['step_test'] = (next_state is not None and done)

            # Test consistenza stato
            results['state_consistency'] = self.env.validate_state_consistency()

            # Test hub tracking
            hub_info = self.env.get_hub_info()
            results['hub_tracking'] = (hub_info is not None and
                                     'current_hub_index' in hub_info)

            # Test action mask
            action_mask = self.env.get_action_mask()
            results['action_validity'] = (len(action_mask) == self.env.num_actions and
                                        action_mask[6])  # STOP sempre valida

        except Exception as e:
            print(f"❌ Errore durante test base: {e}")

        return results

    def test_action_sequence(self, actions: List[int]) -> Dict:
        """
        Testa una sequenza specifica di azioni.

        Args:
            actions: Lista di azioni da testare

        Returns:
            Dictionary con risultati del test
        """
        self.env.reset()

        results = {
            'actions_tested': len(actions),
            'successful_actions': 0,
            'failed_actions': 0,
            'hub_score_trajectory': [],
            'final_reward': 0.0,
            'episode_ended': False
        }

        initial_hub_score = self.env.get_performance_stats()['current_hub_score']
        results['hub_score_trajectory'].append(initial_hub_score)

        for i, action in enumerate(actions):
            try:
                _, reward, done, info = self.env.step(action)

                if info['action_success']:
                    results['successful_actions'] += 1
                else:
                    results['failed_actions'] += 1

                current_hub_score = self.env.get_performance_stats()['current_hub_score']
                results['hub_score_trajectory'].append(current_hub_score)

                if done:
                    results['final_reward'] = info.get('final_reward', 0.0)
                    results['episode_ended'] = True
                    break

            except Exception as e:
                print(f"❌ Errore durante azione {i} ({action}): {e}")
                results['failed_actions'] += 1

        return results

    def benchmark_hub_score_calculation(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark performance del calcolo hub score.

        Args:
            num_iterations: Numero di iterazioni per il benchmark

        Returns:
            Dictionary con statistiche di performance
        """
        import time

        self.env.reset()

        # Warm-up
        for _ in range(10):
            hub_idx = self.env.hub_tracker.get_current_hub_index(self.env.current_data)
            _ = self.env.compute_hub_score_from_tensor(self.env.current_data, hub_idx)

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            hub_idx = self.env.hub_tracker.get_current_hub_index(self.env.current_data)
            hub_score = self.env.compute_hub_score_from_tensor(self.env.current_data, hub_idx)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_iterations

        return {
            'total_time_seconds': total_time,
            'average_time_ms': avg_time * 1000,
            'iterations_per_second': num_iterations / total_time,
            'hub_score_sample': hub_score
        }

    def generate_test_report(self) -> str:
        """
        Genera un report completo dei test.

        Returns:
            String con report formattato
        """
        basic_results = self.test_basic_functionality()
        benchmark_results = self.benchmark_hub_score_calculation()

        report = []
        report.append("=" * 60)
        report.append("🧪 REFACTOR ENVIRONMENT TEST REPORT")
        report.append("=" * 60)

        # Test funzionalità base
        report.append("\n📋 BASIC FUNCTIONALITY TESTS:")
        for test_name, passed in basic_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"   {test_name}: {status}")

        # Performance benchmark
        report.append("\n⚡ PERFORMANCE BENCHMARK:")
        report.append(f"   Hub score calculation: {benchmark_results['average_time_ms']:.3f} ms avg")
        report.append(f"   Throughput: {benchmark_results['iterations_per_second']:.0f} ops/sec")

        # Environment info
        stats = self.env.get_performance_stats()
        report.append("\n📊 ENVIRONMENT STATUS:")
        report.append(f"   Dataset size: {len(self.env.original_data_list)} graphs")
        report.append(f"   Max nodes: {self.env.max_nodes}")
        report.append(f"   Action space: {self.env.num_actions} actions")
        report.append(f"   Observation space: {self.env.observation_space.shape}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE E TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def example_usage():
    """
    NUOVO: Esempio di utilizzo dell'ambiente refactored.
    """
    print("🚀 Esempio di utilizzo RefactorEnv refactored")

    # Inizializza ambiente (sostituisci con il tuo path)
    # env = RefactorEnv(
    #     data_path="/path/to/your/data",
    #     max_steps=15,
    #     reward_weights={
    #         'step_valid': 0.01,
    #         'step_invalid': -0.02,
    #         'adversarial_weight': 0.2
    #     }
    # )

    # # Test base
    # tester = RefactorEnvTester(env)
    # print(tester.generate_test_report())

    # # Episodio di esempio
    # state = env.reset()
    # print(f"📊 Stato iniziale: hub_score = {env.get_performance_stats()['current_hub_score']:.4f}")

    # for step in range(10):
    #     # Usa action mask per azioni valide
    #     valid_actions = env.get_action_mask()
    #     available_actions = np.where(valid_actions)[0]
    #     action = np.random.choice(available_actions)

    #     next_state, reward, done, info = env.step(action)

    #     print(f"Step {step}: Action {action}, Reward {reward:.3f}, Done {done}")

    #     if done:
    #         final_stats = env.get_performance_stats()
    #         print(f"🏁 Episodio terminato. Hub improvement: {final_stats['hub_improvement']:.4f}")
    #         break

    # env.close()

    pass

if __name__ == "__main__":
    example_usage()