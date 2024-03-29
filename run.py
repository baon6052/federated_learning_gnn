import json
from functools import partial

import click
import flwr as fl

import wandb
from client import get_model_parameters, run_client
from datasets.dataset import (
    NodeFeatureSliceDataset,
    NodeFeatureSliceDataset2,
    PlanetoidDataset,
    PlanetoidDatasetType,
)
from models.graph_attention_network import GAT
from models.graph_convolutional_neural_network import GCN


def run_experiment(
    experiment_data: dict[str:any],
    experiment_name: str = "Bespoke Experiment",
    dry_run: bool = True,
    group: str = "unnamed-group",
):
    dry_run = experiment_data["dry_run"]
    print(f"Running Experiement: {experiment_name}")
    if dry_run:
        wandb.init(
            project="my-awesome-project-rev2",
            entity="ml-sys",
            name=experiment_name,
            group=group,
        )
    else:
        wandb.init(
            project="federated_learning_gnn-rev2",
            entity="ml-sys",
            name=experiment_name,
            group=group,
        )

    num_clients = experiment_data["num_clients"]
    dataset_name = experiment_data["dataset_name"]
    slice_method = experiment_data["slice_method"]
    percentage_overlap = experiment_data["percentage_overlap"]
    model_type = experiment_data["model_type"]
    num_hidden_params = experiment_data["num_hidden_params"]
    num_hidden_layers = experiment_data["num_hidden_layers"]
    learning_rate = experiment_data["learning_rate"]
    epochs_per_client = experiment_data["epochs_per_client"]
    num_rounds = experiment_data["num_rounds"]
    aggregation_strategy = experiment_data["aggregation_strategy"]

    custom_dataset = PlanetoidDataset(
        PlanetoidDatasetType(dataset_name), num_clients=num_clients
    )

    if slice_method == "node_feature":
        custom_dataset = NodeFeatureSliceDataset(
            PlanetoidDatasetType(dataset_name),
            num_clients=num_clients,
            overlap_percent=percentage_overlap,
        )
    elif slice_method == "node_feature2":
        custom_dataset = NodeFeatureSliceDataset2(
            PlanetoidDatasetType(dataset_name),
            num_clients=num_clients,
            overlap_percent=percentage_overlap,
        )

    # Initialise model for aggregation strategy
    model = GAT(
        num_features=custom_dataset.num_features_per_client,
        num_hidden=num_hidden_params,
        num_classes=custom_dataset.num_classes,
        num_hidden_layers=num_hidden_layers,
        learning_rate=learning_rate,
    )

    if model_type == "GCN":
        model = GCN(
            num_features=custom_dataset.num_features_per_client,
            num_classes=custom_dataset.num_classes,
            num_hidden=num_hidden_params,
            num_hidden_layers=num_hidden_layers,
            learning_rate=learning_rate,
        )

    client_fn_partial = partial(
        run_client,
        model_type,
        num_hidden_params,
        num_hidden_layers,
        custom_dataset.dataset_per_client,
        epochs_per_client,
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    if aggregation_strategy == "FedProx":
        strategy = fl.server.strategy.FedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            proximal_mu=0.1,  # TODO: speak about potential ablation study for this  # noqa:E501
        )
    elif aggregation_strategy == "FedYogi":
        strategy = fl.server.strategy.FedYogi(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_model_parameters(model)
            ),
        )
    elif aggregation_strategy == "FedAdam":
        strategy = fl.server.strategy.FedAdam(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_model_parameters(model)
            ),
        )
    elif aggregation_strategy == "FedOpt":
        strategy = fl.server.strategy.FedOpt(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_model_parameters(model)
            ),
        )
    elif aggregation_strategy == "FedAdagrad":
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_model_parameters(model)
            ),
        )

    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    metrics = fl.simulation.start_simulation(
        client_fn=client_fn_partial,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    for federated_round, accuracy in metrics.losses_distributed:
        wandb.log({"test_accuracy": accuracy})

    wandb.finish()


@click.command()
@click.option("--num_clients", default=10)
@click.option(
    "--dataset_name",
    default="Cora",
    type=click.Choice(["Cora", "CiteSeer", "PubMed"]),
)
@click.option(
    "--slice_method",
    default=None,
    type=click.Choice([None, "node_feature", "node_feature2"]),
)
@click.option("--percentage_overlap", default=0)
@click.option("--model_type", default="GAT", type=click.Choice(["GCN", "GAT"]))
@click.option("--num_hidden_params", default=16)
@click.option("--num_hidden_layers", default=1)
@click.option("--learning_rate", default=0.01)
@click.option("--epochs_per_client", default=10)
@click.option("--num_rounds", default=10)
@click.option("--aggregation_strategy", default="FedAvg")
@click.option("--experiment_config_filename", required=False, default=None)
@click.option("--experiment_name", required=False, default=None)
@click.option("--dry_run", default=True)
def run(
    num_clients: int,
    dataset_name: str,
    slice_method: str,
    percentage_overlap: int,
    model_type: str,
    num_hidden_params: int,
    num_hidden_layers: int,
    learning_rate: float,
    epochs_per_client: int,
    num_rounds: int,
    aggregation_strategy: str,
    experiment_config_filename: str,
    experiment_name: str,
    dry_run: bool,
):
    if experiment_config_filename is not None:
        with open(
            f"experiment_configs/{experiment_config_filename}.json"
        ) as json_file:
            experiments = json.load(json_file)

        if experiment_name:
            experiment_data = experiments[experiment_name]
            print(experiment_data)
            run_experiment(
                experiment_data,
                experiment_name=experiment_name,
                group=experiment_config_filename,
            )
        else:
            for experiment_name, experiment_data in experiments.items():
                run_experiment(
                    experiment_data,
                    experiment_name=experiment_name,
                    group=experiment_config_filename,
                )
    else:
        experiment_data = {
            "experiment_config_filename": experiment_config_filename,
            "num_clients": num_clients,
            "dataset_name": dataset_name,
            "slice_method": slice_method,
            "percentage_overlap": percentage_overlap,
            "model_type": model_type,
            "num_hidden_params": num_hidden_params,
            "num_hidden_layers": num_hidden_layers,
            "learning_rate": learning_rate,
            "epochs_per_client": epochs_per_client,
            "num_rounds": num_rounds,
            "aggregation_strategy": aggregation_strategy,
            "dry_run": dry_run,
        }
        run_experiment(experiment_data=experiment_data, group="unnamed-group")


if __name__ == "__main__":
    run()
