"""
Ambiente di reinforcement learning per la rifattorizzazione automatica
di sub-graph 1-hop di dependency graph usando PyTorch Geometric.
"""

import gym
from gym import spaces
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, add_self_loops, remove_self_loops
import networkx as nx
import numpy as np
import copy
import warnings
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from pathlib import Path

warnings.filterwarnings('ignore')


class RefactorEnv(gym.Env):
    """
    Ambiente OpenAI Gym per la rifattorizzazione di sub-graph 1-hop
    attorno al nodo hub centrale.
    """

    def __init__(self,
                 data_path: str,
                 discriminator=None,
                 max_steps: int = 20,
                 reward_weights: Dict[str, float] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Inizializza l'ambiente di refactoring.

        Args:
            data_path: Percorso alla directory contenente i file .pt con i dati PyG
            discriminator: Modello discriminatore pre-addestrato (opzionale)
            max_steps: Numero massimo di step per episodio
            reward_weights: Pesi per le componenti del reward
            device: Dispositivo di calcolo
        """
        super(RefactorEnv, self).__init__()

        self.device = device
        self.max_steps = max_steps
        self.discriminator = discriminator

        # Pesi default per il calcolo del reward
        self.reward_weights = reward_weights or {
            'hub_score': 1.0,
            'modularity': 0.5,
            'density': -0.3,
            'avg_shortest_path': 0.2,
            'discriminator': 0.4
        }

        # Carica e preprocessa i dati
        self.original_data_list = self._load_and_preprocess_data(data_path)
        self.current_data = None
        self.current_step = 0
        self.center_node_idx = None
        self.initial_metrics = {}

        # Definisci spazi di azione e osservazione
        self.num_actions = 5  # 5 azioni atomiche
        self.action_space = spaces.Discrete(self.num_actions)

        # Lo spazio di osservazione sarà definito dinamicamente
        # basato sulle dimensioni dei grafi
        max_nodes = max([data.num_nodes for data in self.original_data_list])
        self.max_nodes = max_nodes

        # Feature space: node features + adjacency + global metrics
        obs_dim = max_nodes * 7 + max_nodes * max_nodes + 10  # 10 metriche globali
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    def _load_and_preprocess_data(self, data_path: str) -> List[Data]:
        """
        Carica e preprocessa i dati dalla directory contenente file .pt.

        Args:
            data_path: Percorso alla directory contenente i file .pt

        Returns:
            Lista di oggetti Data preprocessati
        """
        print(f"Caricamento dati da: {data_path}")

        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory non trovata: {data_path}")

        # Trova tutti i file .pt nella directory
        pt_files = list(data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"Nessun file .pt trovato in {data_path}")

        print(f"Trovati {len(pt_files)} file .pt")

        # Carica tutti i file .pt
        data_list = []
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location=self.device)

                # Gestisci diversi formati di dati
                if isinstance(data, dict):
                    if 'data' in data:
                        graph_data = data['data']
                    else:
                        # Assume che abbia 'x' e 'edge_index'
                        graph_data = Data(x=data['x'], edge_index=data['edge_index'])
                        if 'edge_attr' in data:
                            graph_data.edge_attr = data['edge_attr']
                elif isinstance(data, Data):
                    graph_data = data
                else:
                    print(f"Formato dati non riconosciuto in {pt_file}, saltato")
                    continue

                # Verifica che i dati siano validi
                if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                    if graph_data.x.size(1) == 7:  # Verifica che abbia 7 feature per nodo
                        data_list.append(graph_data)
                    else:
                        print(f"File {pt_file} ha {graph_data.x.size(1)} feature invece di 7, saltato")
                else:
                    print(f"File {pt_file} non ha attributi x o edge_index, saltato")

            except Exception as e:
                print(f"Errore caricando {pt_file}: {e}")
                continue

        if not data_list:
            raise ValueError("Nessun dato valido caricato")

        # Normalizza le feature nodali (7 dimensioni)
        scaler = StandardScaler()
        all_features = torch.cat([data.x for data in data_list], dim=0)
        all_features_np = all_features.cpu().numpy()
        scaler.fit(all_features_np)

        processed_data = []
        for data in data_list:
            data_copy = copy.deepcopy(data)

            # Normalizza feature
            normalized_features = scaler.transform(data_copy.x.cpu().numpy())
            data_copy.x = torch.tensor(normalized_features, dtype=torch.float32).to(self.device)

            # Aggiungi self-loops se necessario
            edge_index, _ = add_self_loops(data_copy.edge_index)
            data_copy.edge_index = edge_index

            processed_data.append(data_copy)

        print(f"Caricati e preprocessati {len(processed_data)} sub-graph")
        return processed_data

    def reset(self, graph_idx: Optional[int] = None) -> np.ndarray:
        """
        Resetta l'ambiente per un nuovo episodio.

        Args:
            graph_idx: Indice del grafo da usare (None per casuale)

        Returns:
            Stato iniziale
        """
        if graph_idx is None:
            graph_idx = np.random.randint(0, len(self.original_data_list))

        self.current_data = copy.deepcopy(self.original_data_list[graph_idx])
        self.current_step = 0

        # Identifica il nodo hub (quello con grado massimo)
        degrees = torch.bincount(self.current_data.edge_index[0])
        self.center_node_idx = degrees.argmax().item()

        # Calcola metriche iniziali
        self.initial_metrics = self._calculate_metrics(self.current_data)

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Esegue un'azione nell'ambiente.

        Args:
            action: Azione da eseguire (0-4)

        Returns:
            Tupla (stato, reward, done, info)
        """
        if self.current_data is None:
            raise RuntimeError("Ambiente non inizializzato. Chiama reset() prima.")

        self.current_step += 1

        # Salva stato precedente per calcolo reward
        prev_data = copy.deepcopy(self.current_data)
        prev_metrics = self._calculate_metrics(prev_data)

        # Applica l'azione
        success = self._apply_action(action)

        # Calcola nuovo stato e reward
        current_metrics = self._calculate_metrics(self.current_data)
        reward = self._calculate_reward(prev_metrics, current_metrics, success)

        # Controlla se l'episodio è terminato
        done = self.current_step >= self.max_steps

        info = {
            'action_success': success,
            'metrics': current_metrics,
            'step': self.current_step,
            'center_node': self.center_node_idx
        }

        return self._get_state(), reward, done, info

    def _apply_action(self, action: int) -> bool:
        """
        Applica l'azione specificata al grafo corrente.

        Args:
            action: Azione da applicare

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
                return self._create_helper_and_reassign()
            elif action == 4:
                return self._swap_edges_by_betweenness()
            else:
                return False
        except Exception as e:
            print(f"Errore nell'applicazione dell'azione {action}: {e}")
            return False

    def _remove_edge(self) -> bool:
        """
        Rimuove un arco casuale dal nodo hub.
        """
        # Trova archi del nodo hub
        hub_edges = []
        edge_index = self.current_data.edge_index

        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u == self.center_node_idx and u != v:  # No self-loops
                hub_edges.append(i)

        if not hub_edges:
            return False

        # Rimuovi arco casuale
        edge_to_remove = np.random.choice(hub_edges)
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        mask[edge_to_remove] = False

        new_edge_index = edge_index[:, mask]

        # Verifica che il grafo rimanga connesso
        if self._is_connected(new_edge_index):
            self.current_data.edge_index = new_edge_index
            return True

        return False

    def _add_edge(self) -> bool:
        """
        Aggiunge un arco dal nodo hub a un nodo casuale.
        """
        possible_targets = []
        edge_index = self.current_data.edge_index

        # Trova nodi non ancora connessi al hub
        connected_nodes = set()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u == self.center_node_idx:
                connected_nodes.add(v)

        for node in range(self.current_data.num_nodes):
            if node not in connected_nodes and node != self.center_node_idx:
                possible_targets.append(node)

        if not possible_targets:
            return False

        target = np.random.choice(possible_targets)
        new_edge = torch.tensor([[self.center_node_idx], [target]], device=self.device)

        self.current_data.edge_index = torch.cat([edge_index, new_edge], dim=1)
        return True

    def _move_edge(self) -> bool:
        """
        Sposta un arco dal nodo hub (rimuove e aggiunge in un step).
        """
        # Prima rimuovi un arco
        if not self._remove_edge():
            return False
        # Poi aggiungi un nuovo arco
        return self._add_edge()

    def _create_helper_and_reassign(self) -> bool:
        """
        Crea un nodo helper e riassegna alcuni figli del hub ad esso,
        mettendo sempre in coerenza `num_nodes` con la dimensione di x.
        """
        # Trova tutti i figli del nodo hub (escludendo self-loops)
        children = []
        ei = self.current_data.edge_index
        for u, v in ei.t().tolist():
            if u == self.center_node_idx and u != v:
                children.append(v)

        # Serve almeno 2 figli per split
        if len(children) < 2:
            return False

        # Seleziona un sottoinsieme (fino a 3) di figli da riassegnare
        num_to_reassign = min(len(children) // 2, 3)
        children_to_reassign = np.random.choice(children, num_to_reassign, replace=False)

        # Indice del nuovo helper = vecchio numero di righe di x
        helper_idx = self.current_data.x.size(0)

        # Costruisci le feature del helper come media delle feature dei figli
        children_feats = self.current_data.x[children_to_reassign]
        helper_feats = children_feats.mean(dim=0, keepdim=True)

        # Estendi x e riallinea num_nodes
        self.current_data.x = torch.cat([self.current_data.x, helper_feats], dim=0)
        # Assicuriamoci che num_nodes sia esattamente x.size(0)
        self.current_data.num_nodes = self.current_data.x.size(0)

        # Ricostruisci la lista di archi
        new_edges: List[Tuple[int,int]] = []
        # 1) arco hub -> helper
        new_edges.append((self.center_node_idx, helper_idx))
        # 2) per ogni arco originale, rimappalo o mantienilo
        for u, v in ei.t().tolist():
            if u == self.center_node_idx and v in children_to_reassign:
                # sposta il figlio dal hub all’helper
                new_edges.append((helper_idx, v))
            else:
                new_edges.append((u, v))

        # Crea il nuovo edge_index
        edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        self.current_data.edge_index = edge_index

        return True

    def _swap_edges_by_betweenness(self) -> bool:
        """
        Scambia due archi ad alta betweenness centrality, riassegnando
        i loro endpoint in modo incrociato, e mantiene invariati gli attributi x.
        """
        try:
            # Converte a NetworkX (grafo non orientato)
            G = to_networkx(self.current_data, to_undirected=True)

            # Calcola betweenness centrality per arco
            betweenness = nx.edge_betweenness_centrality(G)
            if len(betweenness) < 2:
                return False

            # Prendi i due archi con betweenness più alta
            sorted_edges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            (u1, v1), _ = sorted_edges[0][0], sorted_edges[0][1]
            (u2, v2), _ = sorted_edges[1][0], sorted_edges[1][1]

            # Rimuovi i due archi
            G.remove_edge(u1, v1)
            G.remove_edge(u2, v2)

            # Aggiungi le connessioni incrociate
            G.add_edge(u1, v2)
            G.add_edge(u2, v1)

            # Verifica che rimanga connesso
            if not nx.is_connected(G):
                return False

            # Ricostruisci solo edge_index, mantenendo data.x invariato
            new_edge_list = list(G.edges())
            edge_index = torch.tensor(new_edge_list, dtype=torch.long, device=self.device).t().contiguous()

            # Per grafi diretti, duplicare in entrambi i versi se serve:
            # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            self.current_data.edge_index = edge_index
            return True

        except Exception as e:
            print(f"Errore in swap_edges_by_betweenness: {e}")
            return False

    def _is_connected(self, edge_index: torch.Tensor) -> bool:
        """
        Verifica se il grafo è connesso.
        """
        try:
            temp_data = Data(edge_index=edge_index, num_nodes=self.current_data.num_nodes)
            G = to_networkx(temp_data, to_undirected=True)
            return nx.is_connected(G)
        except:
            return False

    def _calculate_metrics(self, data: Data) -> Dict[str, float]:
        """
        Calcola metriche globali del grafo.

        Args:
            data: Oggetto Data PyG

        Returns:
            Dizionario con le metriche
        """
        try:
            G = to_networkx(data, to_undirected=True)

            # Hub score del nodo centrale
            hub_score = G.degree(self.center_node_idx) if self.center_node_idx < data.num_nodes else 0

            # Densità
            density = nx.density(G)

            # Modularità (usa partizionamento greedy)
            try:
                communities = nx.community.greedy_modularity_communities(G)
                modularity = nx.community.modularity(G, communities)
            except:
                modularity = 0.0

            # Cammino minimo medio
            try:
                if nx.is_connected(G):
                    avg_shortest_path = nx.average_shortest_path_length(G)
                else:
                    avg_shortest_path = float('inf')
            except:
                avg_shortest_path = float('inf')

            # Clustering coefficient
            clustering = nx.average_clustering(G)

            # Betweenness centrality media
            betweenness = nx.betweenness_centrality(G)
            avg_betweenness = np.mean(list(betweenness.values()))

            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            avg_degree_centrality = np.mean(list(degree_centrality.values()))

            return {
                'hub_score': float(hub_score),
                'density': float(density),
                'modularity': float(modularity),
                'avg_shortest_path': float(avg_shortest_path),
                'clustering': float(clustering),
                'avg_betweenness': float(avg_betweenness),
                'avg_degree_centrality': float(avg_degree_centrality),
                'num_nodes': int(data.num_nodes),
                'num_edges': int(data.edge_index.shape[1]),
                'connected': float(nx.is_connected(G))
            }
        except Exception as e:
            print(f"Errore nel calcolo delle metriche: {e}")
            return {k: 0.0 for k in ['hub_score', 'density', 'modularity', 'avg_shortest_path',
                                     'clustering', 'avg_betweenness', 'avg_degree_centrality',
                                     'num_nodes', 'num_edges', 'connected']}

    def _extract_global_features(self, data: Data) -> torch.Tensor:
        """
        Estrae il vettore delle feature globali da un PyG Data,
        esattamente come usi in _get_state, ma operando su un Data arbitrario.
        """
        # Calcolo delle metriche
        metrics = self._calculate_metrics(data)

        # costruiamo il tensor in ordine coerente con _get_state()
        global_feats = torch.tensor([
            metrics['hub_score'],
            metrics['density'],
            metrics['modularity'],
            metrics['avg_shortest_path'] if metrics['avg_shortest_path'] != float('inf') else 10.0,
            metrics['clustering'],
            metrics['avg_betweenness'],
            metrics['avg_degree_centrality'],
            data.num_nodes,
            data.edge_index.shape[1],
            float(metrics['connected'])
        ], dtype=torch.float32, device=self.device)

        return global_feats

    def _calculate_reward(self, prev_metrics: Dict[str, float],
                          curr_metrics: Dict[str, float],
                          action_success: bool) -> float:
        """
        Calcola il reward basato sui cambiamenti delle metriche.

        Args:
            prev_metrics: Metriche precedenti
            curr_metrics: Metriche correnti
            action_success: Se l'azione è stata applicata con successo

        Returns:
            Valore del reward
        """
        if not action_success:
            return -0.1  # Penalità per azione fallita

        reward = 0.0

        # Componenti del reward basate sui delta delle metriche
        delta_hub_score = curr_metrics['hub_score'] - prev_metrics['hub_score']
        delta_modularity = curr_metrics['modularity'] - prev_metrics['modularity']
        delta_density = curr_metrics['density'] - prev_metrics['density']

        # Cammino minimo (più piccolo è meglio)
        if curr_metrics['avg_shortest_path'] != float('inf') and prev_metrics['avg_shortest_path'] != float('inf'):
            delta_avg_shortest_path = prev_metrics['avg_shortest_path'] - curr_metrics['avg_shortest_path']
        else:
            delta_avg_shortest_path = 0.0

        # Calcola reward pesato
        reward += self.reward_weights['hub_score'] * delta_hub_score
        reward += self.reward_weights['modularity'] * delta_modularity
        reward += self.reward_weights['density'] * delta_density
        reward += self.reward_weights['avg_shortest_path'] * delta_avg_shortest_path

        # Bonus per mantenere il grafo connesso
        if curr_metrics['connected'] == 0:
            reward -= 1.0

        # Segnale dal discriminatore (se disponibile)
        if self.discriminator is not None:
            try:
                with torch.no_grad():
                    discriminator_score = self.discriminator(self.current_data).sigmoid().item()
                    # Penalizza se il discriminatore classifica come "smelly"
                    reward += self.reward_weights['discriminator'] * (1 - discriminator_score)
            except:
                pass

        return reward

    def _get_state(self) -> np.ndarray:
        """
        Estrae lo stato corrente dell'ambiente.

        Returns:
            Rappresentazione numerica dello stato
        """
        # Numero reale di nodi (dalla dimensione di x)
        real_num_nodes = self.current_data.x.size(0)

        # Node features (padding a max_nodes)
        node_features = torch.zeros(self.max_nodes, 7, device=self.current_data.x.device)
        node_features[:real_num_nodes] = self.current_data.x

        # Adjacency matrix (padding a max_nodes x max_nodes)
        adj_matrix = torch.zeros(self.max_nodes, self.max_nodes, device=self.current_data.edge_index.device)
        for u, v in self.current_data.edge_index.t().tolist():
            if u < self.max_nodes and v < self.max_nodes:
                adj_matrix[u, v] = 1.0

        # Global metrics
        metrics = self._calculate_metrics(self.current_data)
        global_features = torch.tensor([
            metrics['hub_score'],
            metrics['density'],
            metrics['modularity'],
            metrics['avg_shortest_path'] if metrics['avg_shortest_path'] != float('inf') else 10.0,
            metrics['clustering'],
            metrics['avg_betweenness'],
            metrics['avg_degree_centrality'],
            real_num_nodes,
            metrics['num_edges'],
            metrics['connected']
        ], dtype=torch.float32, device=node_features.device)

        # Concatena tutte le features
        state = torch.cat([
            node_features.flatten(),
            adj_matrix.flatten(),
            global_features
        ])

        return state.cpu().numpy()

    def render(self, mode: str = 'human'):
        """
        Visualizza lo stato corrente dell'ambiente.
        """
        if self.current_data is None:
            print("Ambiente non inizializzato")
            return

        metrics = self._calculate_metrics(self.current_data)
        print(f"\n=== Step {self.current_step} ===")
        print(f"Centro hub: {self.center_node_idx}")
        print(f"Nodi: {metrics['num_nodes']}, Archi: {metrics['num_edges']}")
        print(f"Hub score: {metrics['hub_score']:.3f}")
        print(f"Density: {metrics['density']:.3f}")
        print(f"Modularity: {metrics['modularity']:.3f}")
        print(f"Avg shortest path: {metrics['avg_shortest_path']:.3f}")
        print(f"Connected: {bool(metrics['connected'])}")