from enum import Enum
from copy import deepcopy

import lightning as L
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.loader.dataloader import DataLoader

from datasets.dataset_info import print_node_feature_slice_dataset_info

class CustomDataset(L.LightningDataModule):
    def __init__(self, dataset: Data = None):
        self.dataset = dataset

    def print_info(self):
        print(f"Number of graphs: {len(self.dataset)}")
        print(f"Number of features: {self.dataset.num_features}")
        print(f"Number of classes: {self.dataset.num_classes}")

        data = self.dataset[0]
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(
            f"Training node "
            f"label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
        )
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")

    # def __len__(self):
    #     return ...

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset)


class PlanetoidDatasetType(Enum):
    CORA = "Cora"
    CITESEER = "CiteSeer"
    PUBMED = "PubMed"



class PlanetoidDataset():
    def __init__(self, name: PlanetoidDatasetType, num_clients: int):
        self.dataset = Planetoid(root="./datasets", name=name.value)
        self.num_clients = num_clients
        self.num_classes = self.dataset.num_classes
        (self.dataset_per_client, 
         self.num_features_per_client) = self._get_datasets()

    def _get_datasets(self) -> None:
        return [CustomDataset(dataset = deepcopy(self.dataset)) for _ in range(self.num_clients)], self.dataset.x.shape[0]


class NodeFeatureSliceDataset:
    def __init__(self, dataset: PlanetoidDataset, num_clients: int, overlap_percent: int = 0, verbose: bool = False) -> None:
        self.dataset = dataset
        self.num_clients = num_clients
        self.overlap_percent = overlap_percent
        self.verbose = verbose

        self.num_classes = self.dataset.dataset.num_classes

        (self.dataset_per_client, 
         self.num_features_per_client) = self._get_datasets()


    def _get_datasets(self) -> list[Data]:
        # defining overlap as %  of total dataset size
        data = self.dataset.dataset[0]
        features = data.x
        num_features = data.x.shape[1]

        shuffled_indices = torch.randperm(features.size(1))
        shuffled_features = features[:, shuffled_indices]

        num_overlap_features = int(num_features * (self.overlap_percent / 100))
        num_unique_features = num_features - num_overlap_features
        unique_features_per_partition = num_unique_features // self.num_clients
        overlap_features = shuffled_features[:, :num_overlap_features]

        dataset_per_client = []

        for i in range(self.num_clients):
            start_idx = num_overlap_features + i * unique_features_per_partition
            end_idx = (
                num_overlap_features + (i + 1) * unique_features_per_partition
                if i < self.num_clients - 1
                else num_features
            )
            unique_features = shuffled_features[:, start_idx:end_idx]
            partition_features = torch.cat(
                (unique_features, overlap_features), dim=1
            )

            #copying original Dataset object defined in constructor and changing the data.x 
            # part of it, keeping the rest the same
            new_dataset_object = deepcopy(self.dataset)
            new_dataset_object.dataset[0].x = partition_features
            dataset_per_client.append(new_dataset_object)

        if self.verbose:
            print_node_feature_slice_dataset_info()

        return [CustomDataset(dataset=data) for data in dataset_per_client], partition_features.shape[0]


class EdgeFeatureSliceDataset:
    def __init__(self, dataset: PlanetoidDataset) -> None:
        self.dataset = dataset

    def slice_dataset(num_partitions: int, overlap_amount: int):
        pass


class GraphPartitionSliceDataset:
    def __init__(self, dataset: PlanetoidDataset) -> None:
        self.dataset = dataset

    def slice_dataset(nums_partitions: int):
        pass
