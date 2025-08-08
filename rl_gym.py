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
HUB_FEATURES = [
    'fan_in', 'fan_out', 'degree_centrality', 'in_out_ratio',
    'pagerank', 'betweenness_centrality', 'closeness_centrality'
]

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
        """
        super(RefactorEnv, self).__init__()

        self.device = device
        self.max_steps = max_steps
        self.discriminator = discriminator

        # Pesi default per il calcolo del reward
        self.reward_weights = reward_weights or {
            'hub_score': 2.0,
            'step_valid': 0.1,
            'step_invalid': -0.1,
            'cycle_penalty': -0.5,
            'duplicate_penalty': -0.3,
            'adversarial_weight': 0.5
        }

        # Carica e preprocessa i dati
        self.original_data_list = self._load_and_preprocess_data(data_path)
        self.current_data = None
        self.current_step = 0
        self.center_node_idx = None
        self.initial_metrics = {}

        # 7 azioni: RemoveEdge, AddEdge, MoveEdge, ExtractMethod, ExtractAbstractUnit, ExtractUnit, STOP
        self.num_actions = 7
        self.action_space = spaces.Discrete(self.num_actions)

        # Spazio di osservazione
        max_nodes = max([data.num_nodes for data in self.original_data_list])
        self.max_nodes = max_nodes

        obs_dim = max_nodes * 7 + max_nodes * max_nodes + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

        # Inizializza scaler per normalizzazione
        self.feature_scaler = None
        self._fit_feature_scaler()

    @staticmethod
    def _compute_centrality_metrics(G: nx.Graph) -> Tuple[Dict, Dict, Dict]:
        """Compute centrality metrics efficiently based on graph size"""
        if len(G) <= 100:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
            betweenness = nx.betweenness_centrality(G, normalized=True)
            closeness = nx.closeness_centrality(G, distance=None, wf_improved=True)
        else:
            total_edges = G.number_of_edges()
            pagerank = {n: float(G.degree(n)) / (2 * total_edges + 1e-8) for n in G.nodes()}
            betweenness = {n: 0.0 for n in G.nodes()}
            closeness = {n: 1.0 / (len(G) - 1 + 1e-8) for n in G.nodes()}

        return pagerank, betweenness, closeness

    @staticmethod
    def _compute_node_features(G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Compute hub detection features for all nodes"""
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        pagerank, betweenness, closeness = RefactorEnv._compute_centrality_metrics(G)

        node_features = {}
        for node in G.nodes():
            fan_in = float(in_degrees.get(node, 0))
            fan_out = float(out_degrees.get(node, 0))
            total_degree = fan_in + fan_out

            node_features[str(node)] = {
                'fan_in': fan_in,
                'fan_out': fan_out,
                'degree_centrality': total_degree / (len(G) - 1 + 1e-8),
                'in_out_ratio': fan_in / (fan_out + 1e-8),
                'pagerank': float(pagerank.get(node, 0)),
                'betweenness_centrality': float(betweenness.get(node, 0)),
                'closeness_centrality': float(closeness.get(node, 0))
            }

        return node_features

    def _fit_feature_scaler(self):
        """Fit lo scaler sulle feature di tutti i grafi del dataset"""
        print("📊 Fitting feature scaler on dataset...")

        all_features = []
        sample_size = min(100, len(self.original_data_list))
        sampled_data = np.random.choice(self.original_data_list, sample_size, replace=False)

        for data in sampled_data:
            G = to_networkx(data, to_undirected=True)
            node_features = self._compute_node_features(G)

            for node_feats in node_features.values():
                feature_vector = [node_feats[feat] for feat in HUB_FEATURES]
                all_features.append(feature_vector)

        if all_features:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(np.array(all_features))
            print(f"✅ Feature scaler fitted on {len(all_features)} node samples")
        else:
            print("⚠️ No features found for scaler fitting")

    def _create_fresh_data_object(self, x: torch.Tensor, edge_index: torch.Tensor) -> Data:
        """Crea un nuovo oggetto Data fresco con metriche ricalcolate e normalizzate"""
        try:
            # Crea grafo NetworkX temporaneo
            G = nx.DiGraph()
            num_nodes = x.size(0)
            G.add_nodes_from(range(num_nodes))

            if edge_index.numel() > 0:
                edge_list = edge_index.t().cpu().numpy().tolist()
                G.add_edges_from(edge_list)

            # Calcola nuove features
            node_features = self._compute_node_features(G)

            # Crea matrice delle features
            feature_matrix = []
            for node_id in range(num_nodes):
                if str(node_id) in node_features:
                    feature_vector = [node_features[str(node_id)][feat] for feat in HUB_FEATURES]
                else:
                    feature_vector = [0.0] * len(HUB_FEATURES)
                feature_matrix.append(feature_vector)

            feature_matrix = np.array(feature_matrix)

            # Normalizza features
            if self.feature_scaler is not None:
                try:
                    feature_matrix = self.feature_scaler.transform(feature_matrix)
                except Exception as e:
                    print(f"Warning: Feature normalization failed: {e}")
                    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (
                                feature_matrix.std(axis=0) + 1e-8)

            # Crea nuovo oggetto Data
            new_data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float32, device=self.device),
                edge_index=edge_index.clone(),
                num_nodes=num_nodes
            )

            return new_data

        except Exception as e:
            print(f"Error creating fresh data object: {e}")
            return Data(
                x=x.clone(),
                edge_index=edge_index.clone(),
                num_nodes=x.size(0)
            )

    def _rebuild_graph_with_fresh_data(self, new_x: torch.Tensor, new_edge_index: torch.Tensor) -> None:
        """Ricostruisce completamente current_data con features fresche e normalizzate"""
        self.current_data = self._create_fresh_data_object(new_x, new_edge_index)

        # Aggiorna center_node_idx se necessario
        if self.center_node_idx >= self.current_data.num_nodes:
            degrees = torch.bincount(self.current_data.edge_index[0])
            if len(degrees) > 0:
                self.center_node_idx = degrees.argmax().item()
            else:
                self.center_node_idx = 0

    def _load_and_preprocess_data(self, data_path: str) -> List[Data]:
        """Carica e preprocessa i dati dalla directory contenente file .pt"""
        print(f"Caricamento dati da: {data_path}")

        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory non trovata: {data_path}")

        pt_files = list(data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"Nessun file .pt trovato in {data_path}")

        print(f"Trovati {len(pt_files)} file .pt")

        # Carica tutti i file .pt
        data_list = []
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location=self.device)

                if isinstance(data, dict):
                    if 'data' in data:
                        graph_data = data['data']
                    else:
                        graph_data = Data(x=data['x'], edge_index=data['edge_index'])
                        if 'edge_attr' in data:
                            graph_data.edge_attr = data['edge_attr']
                elif isinstance(data, Data):
                    graph_data = data
                else:
                    print(f"Formato dati non riconosciuto in {pt_file}, saltato")
                    continue

                if hasattr(graph_data, 'x') and hasattr(graph_data, 'edge_index'):
                    if graph_data.x.size(1) == 7:
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

        # Normalizza le feature nodali
        scaler = StandardScaler()
        all_features = torch.cat([data.x for data in data_list], dim=0)
        all_features_np = all_features.cpu().numpy()
        scaler.fit(all_features_np)

        processed_data = []
        for data in data_list:
            data_copy = copy.deepcopy(data)
            normalized_features = scaler.transform(data_copy.x.cpu().numpy())
            data_copy.x = torch.tensor(normalized_features, dtype=torch.float32).to(self.device)

            edge_index, _ = add_self_loops(data_copy.edge_index)
            data_copy.edge_index = edge_index

            # ✅ FIX: Assicura num_nodes
            data_copy.num_nodes = data_copy.x.size(0)

            processed_data.append(data_copy)

        print(f"Caricati e preprocessati {len(processed_data)} sub-graph")
        return processed_data

    def reset(self, graph_idx: Optional[int] = None) -> np.ndarray:
        """Resetta l'ambiente per un nuovo episodio"""
        if graph_idx is None:
            graph_idx = np.random.randint(0, len(self.original_data_list))

        self.current_data = copy.deepcopy(self.original_data_list[graph_idx])
        self.current_step = 0

        # ✅ FIX: Assicura num_nodes nel reset
        if not hasattr(self.current_data, 'num_nodes') or self.current_data.num_nodes is None:
            self.current_data.num_nodes = self.current_data.x.size(0)

        degrees = torch.bincount(self.current_data.edge_index[0])
        self.center_node_idx = degrees.argmax().item()

        self.initial_metrics = self._calculate_metrics(self.current_data)
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Esegue un'azione nell'ambiente.
        """
        if self.current_data is None:
            raise RuntimeError("Ambiente non inizializzato. Chiama reset() prima.")

        self.current_step += 1

        # Applica l'azione
        success = self._apply_action(action)

        # Step reward basato sulla validità dell'azione
        if success:
            step_reward = self.reward_weights['step_valid']
            # Controlli aggiuntivi per penalità
            step_reward += self._check_penalties()
        else:
            step_reward = self.reward_weights['step_invalid']

        # Controlla terminazione
        done = (action == 6) or (self.current_step >= self.max_steps)  # STOP action o max steps

        # Reward finale solo alla terminazione
        final_reward = 0.0
        if done:
            final_reward = self._calculate_final_reward()

        total_reward = step_reward + final_reward

        # Calcola metriche correnti per info
        current_metrics = self._calculate_metrics(self.current_data)

        info = {
            'action_success': success,
            'metrics': current_metrics,
            'step': self.current_step,
            'center_node': self.center_node_idx,
            'step_reward': step_reward,
            'final_reward': final_reward,
            'is_terminal': done
        }

        return self._get_state(), total_reward, done, info

    def _apply_action(self, action: int) -> bool:
        """
        Applica l'azione specificata al grafo corrente.

        Args:
            action: Azione da applicare (0-6)

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
                return True  # STOP action - sempre valida
            else:
                return False
        except Exception as e:
            print(f"Errore nell'applicazione dell'azione {action}: {e}")
            return False

    def _remove_edge(self) -> bool:
        """
        Rimuove un arco casuale dal nodo hub.
        """
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
            self._rebuild_graph_with_fresh_data(self.current_data.x, new_edge_index)
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
        new_edge_index = torch.cat([edge_index, new_edge], dim=1)

        self._rebuild_graph_with_fresh_data(self.current_data.x, new_edge_index)

        return True

    def _move_edge(self) -> bool:
        """
        Sposta un arco dal nodo hub (rimuove e aggiunge in un step).
        """
        if not self._remove_edge():
            return False
        return self._add_edge()

    def _extract_method(self) -> bool:
        """ExtractMethod: Crea un nuovo nodo method tra due nodi connessi."""
        edge_index = self.current_data.edge_index

        if edge_index.shape[1] == 0:
            return False

        # Seleziona un arco casuale u→v
        edge_idx = np.random.randint(0, edge_index.shape[1])
        u, v = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()

        # Non processare self-loops
        if u == v:
            return False

        # Crea feature per il nuovo nodo method (media delle features originali, non normalizzate)
        u_features_orig = self.current_data.x[u]
        v_features_orig = self.current_data.x[v]
        method_features = ((u_features_orig + v_features_orig) / 2).unsqueeze(0)

        # Ricostruisci edge_index: rimuovi u→v, aggiungi u→method→v
        method_idx = self.current_data.x.size(0)
        new_edges = []

        for i in range(edge_index.shape[1]):
            if i != edge_idx:  # Mantieni tutti gli archi eccetto u→v
                new_edges.append((edge_index[0, i].item(), edge_index[1, i].item()))

        # Aggiungi nuovi archi u→method→v
        new_edges.append((u, method_idx))
        new_edges.append((method_idx, v))

        # Opzionalmente riassegna alcune dipendenze
        v_incoming = [(src, dst) for src, dst in new_edges if dst == v and src != method_idx]
        if len(v_incoming) > 1:
            to_reassign = np.random.choice(len(v_incoming), min(2, len(v_incoming) // 2), replace=False)
            for idx in to_reassign:
                src, _ = v_incoming[idx]
                new_edges.remove((src, v))
                new_edges.append((src, method_idx))

        # Crea nuove tensori
        new_x = torch.cat([self.current_data.x, method_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()

        # --- RICOSTRUZIONE FRESCA CON METRICHE RICALCOLATE ---
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _extract_abstract_unit(self) -> bool:
        """ExtractAbstractUnit: Identifica nodi con dipendenze comuni e crea un'astrazione"""
        edge_index = self.current_data.edge_index

        if edge_index.shape[1] < 3:
            return False

        # Identifica nodi con target comuni
        targets = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if dst not in targets:
                targets[dst] = []
            targets[dst].append(src)

        common_targets = [(dst, srcs) for dst, srcs in targets.items() if len(set(srcs)) >= 2]

        if not common_targets:
            return False

        target_dst, source_nodes = common_targets[np.random.randint(len(common_targets))]
        unique_sources = list(set(source_nodes))

        if len(unique_sources) < 2:
            return False

        num_to_abstract = min(3, len(unique_sources))
        selected_sources = np.random.choice(unique_sources, num_to_abstract, replace=False)

        # Crea nodo astratto
        abstract_idx = self.current_data.x.size(0)
        selected_features = self.current_data.x[selected_sources]
        abstract_features = selected_features.mean(dim=0, keepdim=True)

        # Ricostruisci edge_index
        new_edges = []
        removed_edges = set()

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in selected_sources and dst == target_dst:
                removed_edges.add(i)

        for i in range(edge_index.shape[1]):
            if i not in removed_edges:
                new_edges.append((edge_index[0, i].item(), edge_index[1, i].item()))

        new_edges.append((abstract_idx, target_dst))
        for src in selected_sources:
            new_edges.append((src, abstract_idx))

        # ✅ FIX: Usa _rebuild_graph_with_fresh_data
        new_x = torch.cat([self.current_data.x, abstract_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()

        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _extract_unit(self) -> bool:
        """
        ExtractUnit: Divide le responsabilità del nodo hub in unità separate.
        """
        if self.center_node_idx is None or self.center_node_idx >= self.current_data.num_nodes:
            return False

        edge_index = self.current_data.edge_index

        # Trova tutti i vicini del hub
        hub_neighbors = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src == self.center_node_idx and dst != self.center_node_idx:
                hub_neighbors.append(dst)

        # Rimuovi duplicati
        hub_neighbors = list(set(hub_neighbors))

        if len(hub_neighbors) < 2:
            return False

        # Dividi i vicini in due gruppi (splitting delle responsabilità)
        mid_point = len(hub_neighbors) // 2
        group1 = hub_neighbors[:mid_point]
        group2 = hub_neighbors[mid_point:]

        if not group1 or not group2:
            return False

        # Crea due nuovi nodi unit
        unit1_idx = self.current_data.x.size(0)
        unit2_idx = unit1_idx + 1

        # Feature dei nuovi unit (basate sui loro gruppi di dipendenze)
        hub_features = self.current_data.x[self.center_node_idx]
        group1_features = self.current_data.x[group1].mean(dim=0) if group1 else hub_features
        group2_features = self.current_data.x[group2].mean(dim=0) if group2 else hub_features

        # Combina con feature del hub (weighted average)
        unit1_features = ((hub_features + group1_features) / 2).unsqueeze(0)
        unit2_features = ((hub_features + group2_features) / 2).unsqueeze(0)

        # Ricostruisci edge_index
        unit1_idx = self.current_data.x.size(0)
        unit2_idx = unit1_idx + 1
        new_edges = []

        # Mantieni tutti gli archi che non coinvolgono il hub
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src != self.center_node_idx and dst != self.center_node_idx:
                new_edges.append((src, dst))
            elif src == self.center_node_idx and dst in group1:
                new_edges.append((unit1_idx, dst))
            elif src == self.center_node_idx and dst in group2:
                new_edges.append((unit2_idx, dst))
            elif dst == self.center_node_idx:
                new_edges.append((src, dst))

        # Aggiungi connessioni hub → units
        new_edges.append((self.center_node_idx, unit1_idx))
        new_edges.append((self.center_node_idx, unit2_idx))

        # Crea nuovi tensori
        new_features = torch.cat([unit1_features, unit2_features], dim=0)
        new_x = torch.cat([self.current_data.x, new_features], dim=0)
        new_edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()

        # --- RICOSTRUZIONE FRESCA CON METRICHE RICALCOLATE ---
        self._rebuild_graph_with_fresh_data(new_x, new_edge_index)

        return True

    def _check_penalties(self) -> float:
        """
        Controlla penalità aggiuntive per azioni problematiche.
        """
        penalty = 0.0

        try:
            # Controllo cicli
            G = to_networkx(self.current_data, to_undirected=True)
            if not nx.is_forest(G):  # Se non è una foresta, ha cicli
                penalty += self.reward_weights['cycle_penalty']

            # Controllo archi duplicati
            edge_set = set()
            for i in range(self.current_data.edge_index.shape[1]):
                u, v = self.current_data.edge_index[0, i].item(), self.current_data.edge_index[1, i].item()
                edge = (min(u, v), max(u, v))
                if edge in edge_set:
                    penalty += self.reward_weights['duplicate_penalty']
                    break
                edge_set.add(edge)

        except Exception:
            # Se ci sono errori nel controllo, applica penalità generica
            penalty += self.reward_weights['cycle_penalty']

        return penalty

    def _calculate_final_reward(self) -> float:
        """
        Calcola il reward finale basato su miglioramento hub score e discriminatore.
        """
        try:
            current_metrics = self._calculate_metrics(self.current_data)

            # Delta hub score (diminuzione è positiva)
            delta_hub = self.initial_metrics['hub_score'] - current_metrics['hub_score']
            hub_reward = delta_hub * self.reward_weights['hub_score']

            # Reward adversarial dal discriminatore
            adversarial_reward = 0.0
            if self.discriminator is not None:
                try:
                    with torch.no_grad():
                        disc_output = self.discriminator(self.current_data)
                        if isinstance(disc_output, dict):
                            p_smelly = torch.softmax(disc_output['logits'], dim=1)[0, 1].item()
                        else:
                            p_smelly = torch.softmax(disc_output, dim=1)[0, 1].item()

                        # -log(1 - p_smelly) clampato in [-1, 1]
                        adversarial_term = -np.log(max(1 - p_smelly, 1e-8))
                        adversarial_reward = np.clip(
                            -self.reward_weights['adversarial_weight'] * adversarial_term,
                            -1.0, 1.0
                        )
                except Exception as e:
                    print(f"Errore nel calcolo adversarial reward: {e}")

            return hub_reward + adversarial_reward

        except Exception as e:
            print(f"Errore nel calcolo reward finale: {e}")
            return 0.0

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