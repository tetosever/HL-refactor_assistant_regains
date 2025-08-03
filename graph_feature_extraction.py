import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree


class EnhancedGraphFeatureExtractor:
    """Extract comprehensive structural features for hub detection and resolution"""

    @staticmethod
    def extract_node_features(data: Data) -> torch.Tensor:
        """Extract extended structural features from nodes"""
        edge_index = data.edge_index
        num_nodes = data.x.size(0) if hasattr(data, 'x') else data.num_nodes

        # Basic degree features
        in_degree = degree(edge_index[1], num_nodes, dtype=torch.float32)
        out_degree = degree(edge_index[0], num_nodes, dtype=torch.float32)
        total_degree = in_degree + out_degree

        # Degree ratios (important for hub detection)
        eps = 1e-8
        in_out_ratio = in_degree / (out_degree + eps)
        degree_centrality = total_degree / (num_nodes - 1 + eps)

        # Convert to NetworkX
        G = to_networkx(data, to_undirected=False)

        # Advanced centrality metrics
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            pagerank_tensor = torch.tensor([pagerank.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            pagerank_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        try:
            betweenness = nx.betweenness_centrality(G)
            betweenness_tensor = torch.tensor([betweenness.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            betweenness_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        try:
            closeness = nx.closeness_centrality(G)
            closeness_tensor = torch.tensor([closeness.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            closeness_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        # Hub-specific metrics
        try:
            hubs, authorities = nx.hits(G, max_iter=100)
            hub_tensor = torch.tensor([hubs.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
            auth_tensor = torch.tensor([authorities.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            hub_tensor = torch.zeros(num_nodes, dtype=torch.float32)
            auth_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        # Local clustering and structure
        try:
            clustering = nx.clustering(G)
            clustering_tensor = torch.tensor([clustering.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            clustering_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        # K-core decomposition (identifies densely connected components)
        try:
            core_number = nx.core_number(G)
            core_tensor = torch.tensor([core_number.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            core_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        # Eccentricity (max distance to any other node)
        try:
            if nx.is_strongly_connected(G):
                eccentricity = nx.eccentricity(G)
            else:
                # Use largest strongly connected component
                largest_scc = max(nx.strongly_connected_components(G), key=len)
                subG = G.subgraph(largest_scc)
                eccentricity = nx.eccentricity(subG)
            eccentricity_tensor = torch.tensor([eccentricity.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            eccentricity_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        # Load centrality (betweenness considering edges)
        try:
            load = nx.load_centrality(G)
            load_tensor = torch.tensor([load.get(i, 0) for i in range(num_nodes)], dtype=torch.float32)
        except:
            load_tensor = torch.zeros(num_nodes, dtype=torch.float32)

        # Stack all features
        features = torch.stack([
            in_degree,
            out_degree,
            total_degree,
            in_out_ratio,
            degree_centrality,
            pagerank_tensor,
            betweenness_tensor,
            closeness_tensor,
            hub_tensor,
            auth_tensor,
            clustering_tensor,
            core_tensor,
            eccentricity_tensor,
            load_tensor
        ], dim=1)

        return features

    @staticmethod
    def extract_hub_specific_features(data: Data) -> Dict[str, torch.Tensor]:
        """Extract features specifically relevant to hub detection"""
        G = to_networkx(data, to_undirected=False)
        num_nodes = data.num_nodes

        # Identify potential hubs (top 20% by degree)
        degrees = dict(G.degree())
        threshold = np.percentile(list(degrees.values()), 80)
        potential_hubs = [n for n, d in degrees.items() if d >= threshold]

        # Hub connectivity patterns
        hub_mask = torch.zeros(num_nodes, dtype=torch.bool)
        hub_mask[potential_hubs] = True

        # Inter-hub connectivity
        inter_hub_edges = sum(1 for u, v in G.edges() if u in potential_hubs and v in potential_hubs)

        # Hub dominance score
        total_edges = G.number_of_edges()
        hub_edge_ratio = sum(degrees[h] for h in potential_hubs) / (2 * total_edges + 1e-8)

        return {
            'hub_mask': hub_mask,
            'num_potential_hubs': len(potential_hubs),
            'inter_hub_edges': inter_hub_edges,
            'hub_dominance': hub_edge_ratio,
            'avg_hub_degree': np.mean([degrees[h] for h in potential_hubs]) if potential_hubs else 0
        }
