import itertools
import json


fixed_params = {
    "num_clients": 10,
    "dataset_name": "Cora",
    "slice_method": "node_feature",
    "percentage_overlap": 0,
    "model_type": "GAT",
    "epochs_per_client": 10,
    "num_rounds": 100,
    "dry_run": True,
    "learning_rate": 0.01,
    "num_hidden_params": 16,
    "num_hidden_layers": 2,
}

client_poison_percs = [10, 20, 30]
aggregation_strategies = ["FedAvg", "FedMedian", "Krum"]

grid_search = list(itertools.product(client_poison_percs, aggregation_strategies))


all_experiments = {}
for client_poison_perc, aggregation_strategy in grid_search:
    new_params = fixed_params.copy()
    new_params["client_poison_perc"] = client_poison_perc
    new_params["aggregation_strategy"] = aggregation_strategy

    all_experiments[
        f"GAT-data_poisoning-client_poison_perc_{client_poison_perc}-agg_strategy-{aggregation_strategy}"
    ] = new_params

with open(
    f"experiment_configs/GAT_data_poisoning_experiments.json",
    "w",
) as outfile:
    json.dump(all_experiments, outfile, indent=4)
