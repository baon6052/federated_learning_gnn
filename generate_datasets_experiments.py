import itertools
import json


fixed_params = {
    "num_clients": 1,
    "slice_method": "node_feature",
    "percentage_overlap": 0,
    "model_type": "GAT",
    "epochs_per_client": 1,
    "num_rounds": 100,
    "aggregation_strategy": "FedAvg",
    "dry_run": True,
    "learning_rate": 0.01,
    "num_hidden_params": 16,
    "num_hidden_layers": 2,
}

datasets = ["Cora", "CiteSeer", "PubMed"]

all_experiments = {}
for dataset in datasets:
    new_params = fixed_params.copy()
    new_params["dataset_name"] = dataset

    all_experiments[f"GAT-dataset-{dataset}"] = new_params

with open(
    f"experiment_configs/GAT_dataset_experiments.json",
    "w",
) as outfile:
    json.dump(all_experiments, outfile, indent=4)
