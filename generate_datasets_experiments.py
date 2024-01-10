import itertools
import json
from constants import gat_cora_constants, gat_pubmed_constants, gat_citeseer_constants

fixed_params = {
    "num_clients": 1,
    "slice_method": "node_feature",
    "percentage_overlap": 0,
    "model_type": "GAT",
    "epochs_per_client": 1,
    "num_rounds": 100,
    "aggregation_strategy": "FedAvg",
    "dry_run": False,
}

datasets = ["Cora", "CiteSeer", "PubMed"]

all_experiments = {}
for dataset in datasets:
    new_params = fixed_params.copy()
    new_params["dataset_name"] = dataset
    if dataset == "Cora":
        new_params = new_params.update(gat_cora_constants)

    elif dataset == "CiteSeer":
        new_params = new_params.update(gat_citeseer_constants)

    elif dataset == "PubMed":
        new_params = new_params.update(gat_pubmed_constants)

    all_experiments[f"GAT-dataset-{dataset}"] = new_params

with open(
    f"experiment_configs/GAT_dataset_experiments.json",
    "w",
) as outfile:
    json.dump(all_experiments, outfile, indent=4)
