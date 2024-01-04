import json
from functools import partial

import click
import flwr as fl

import wandb
from client import run_client
from datasets.dataset import (
    EdgeFeatureSliceDataset,
    GraphPartitionSliceDataset,
    NodeFeatureSliceDataset,
    PlanetoidDataset,
    PlanetoidDatasetType,
)

wandb.init(project="federated_learning_gnn", entity="ml-sys")


def run_experiment(
    experiement_data: dict[str:any], experiment_name: str = "Bespoke Experiment"
):
    print(f"Running Experiement: {experiment_name}")

    num_clients = experiement_data["num_clients"]
    dataset_name = experiement_data["dataset_name"]
    slice_method = experiement_data["slice_method"]
    percentage_overlap = experiement_data["percentage_overlap"]
    model_type = experiement_data["model_type"]
    num_hidden_params = experiement_data["num_hidden_params"]
    epochs_per_client = experiement_data["epochs_per_client"]
    num_rounds = experiement_data["num_rounds"]

    custom_dataset = PlanetoidDataset(
        PlanetoidDatasetType(dataset_name), num_clients=num_clients
    )

    if slice_method == "node_feature":
        custom_dataset = NodeFeatureSliceDataset(
            PlanetoidDatasetType(dataset_name),
            num_clients=num_clients,
            overlap_percent=percentage_overlap,
        )
    elif slice_method == "edge_feature":
        custom_dataset = EdgeFeatureSliceDataset(custom_dataset)
    elif slice_method == "graph_partition":
        custom_dataset = GraphPartitionSliceDataset(custom_dataset)

    client_fn_partial = partial(
        run_client,
        model_type,
        num_hidden_params,
        custom_dataset.dataset_per_client,
        epochs_per_client,
    )

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
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
    type=click.Choice(
        [None, "node_feature", "edge_feature", "graph_partition"]
    ),
)
@click.option("--percentage_overlap", default=0)
@click.option("--model_type", default="GAT", type=click.Choice(["GCN", "GAT"]))
@click.option("--num_hidden_params", default=16)
@click.option("--epochs_per_client", default=10)
@click.option("--num_rounds", default=10)
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
    epochs_per_client: int,
    num_rounds: int,
    experiment_config_filename: str,
    experiment_name: str,
    dry_run: bool,
):
    if experiment_config_filename:
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
            )
        else:
            for experiment_name, experiment_data in experiments.items():
                run_experiment(
                    experiment_data,
                    experiment_name=experiment_name,
                )
    else:
        experiment_data = {
            "num_clients": num_clients,
            "dataset_name": dataset_name,
            "slice_method": slice_method,
            "percentage_overlap": percentage_overlap,
            "model_type": model_type,
            "num_hidden_params": num_hidden_params,
            "epochs_per_client": epochs_per_client,
            "num_rounds": num_rounds,
            "dry_run": dry_run,
        }
        run_experiment(experiement_data=experiment_data)


if __name__ == "__main__":
    run()
