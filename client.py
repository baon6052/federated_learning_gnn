from __future__ import annotations

import flwr as fl
import lightning as L

from client_utils import _set_parameters, get_model_parameters
from datasets.dataset import CustomDataset, PlanetoidDataset
from models.graph_attention_network import GAT
from models.graph_convolutional_neural_network import GCN


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, model: GCN | GAT, dataset: PlanetoidDataset, epochs: int
    ):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def set_parameters(self, parameters):
        _set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.global_model_parameters = parameters
        if "proximal_mu" in config:
            self.model.proximal_mu = config["proximal_mu"]

        trainer = L.Trainer(
            max_epochs=self.epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        trainer.fit(
            model=self.model,
            train_dataloaders=self.dataset.train_dataloader(),
            val_dataloaders=self.dataset.val_dataloader(),
        )

        return (
            self.get_parameters(config={}),
            int(self.dataset.dataset[0].train_mask.sum()),
            {},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = L.Trainer(accelerator="cpu")
        results = trainer.test(self.model, self.dataset.test_dataloader())

        accuracy = results[0]["test_accuracy"]

        return (
            float(accuracy),
            int(self.dataset.dataset[0].test_mask.sum()),
            {"loss": accuracy},
        )


def run_client(
    model_type: str,
    num_hidden_params: int,
    num_hidden_layers: int,
    client_datasets: list[CustomDataset],
    num_epochs: int,
    cid: int,
) -> None:
    custom_dataset = client_datasets[int(cid)]

    if model_type == "GAT":
        model = GAT(
            num_features=custom_dataset.dataset[0].num_features,
            num_hidden=num_hidden_params,
            num_hidden_layers=num_hidden_layers,
            num_classes=custom_dataset.dataset[0].num_classes,
        )
    elif model_type == "GCN":
        model = GCN(
            num_features=custom_dataset.dataset[0].num_features,
            num_classes=custom_dataset.dataset[0].num_classes,
            num_hidden_layers=num_hidden_layers,
        )

    return FlowerClient(model, custom_dataset, num_epochs)
