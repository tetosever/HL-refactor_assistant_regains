from typing import List

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from data_builder.graph_feature_extraction import EnhancedGraphFeatureExtractor


class EnhancedStructuralNormalizer(BaseTransform):
    """Normalize structural features with robust scaling and feature selection"""

    def __init__(self):
        self.node_scaler = RobustScaler()
        self.edge_scaler = RobustScaler()
        self.feature_extractor = EnhancedGraphFeatureExtractor()
        self.fitted = False

    def fit(self, data_list: List[Data]):
        """Fit normalizers on a list of graphs"""
        all_node_features = []
        all_edge_features = []

        for data in data_list:
            node_features = self.feature_extractor.extract_node_features(data)
            all_node_features.append(node_features.numpy())

            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_features.append(data.edge_attr.numpy())

        # Fit scalers
        all_node_features = np.vstack(all_node_features)
        self.node_scaler.fit(all_node_features)

        if all_edge_features:
            all_edge_features = np.vstack(all_edge_features)
            self.edge_scaler.fit(all_edge_features)

        self.fitted = True
        return self

    def __call__(self, data: Data) -> Data:
        """Transform a single graph"""
        data = data.clone()

        # Extract and normalize node features
        node_features = self.feature_extractor.extract_node_features(data)
        if self.fitted:
            node_features_norm = self.node_scaler.transform(node_features.numpy())
            data.x = torch.from_numpy(node_features_norm).float()
        else:
            data.x = node_features

        # Extract hub-specific features
        hub_features = self.feature_extractor.extract_hub_specific_features(data)
        for key, value in hub_features.items():
            setattr(data, key, value)

        # Handle edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and self.fitted:
            edge_features_norm = self.edge_scaler.transform(data.edge_attr.numpy())
            data.edge_attr = torch.from_numpy(edge_features_norm).float()
        elif not hasattr(data, 'edge_attr'):
            # Create basic edge features
            num_edges = data.edge_index.size(1)
            data.edge_attr = torch.ones(num_edges, 1, dtype=torch.float32)

        return data