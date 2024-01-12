import itertools
import json

from constants import gat_cora_constants, gat_pubmed_constants, gat_citeseer_constants


fixed_params = {
    "num_clients": 20,
    "dataset_name": "Cora",
    "slice_method": "node_feature",
    "percentage_overlap": 0,
    "model_type": "GAT",
    "epochs_per_client": 10,
    "num_rounds": 100,
    "dry_run": False,
}

fixed_params.update(gat_cora_constants)

client_poison_percs = [10, 20, 5]
node_features_flip_fracs = [0.01, 0.02, 0.05]
aggregation_strategies = ["FedAvg", "FedMedian", "Krum", "FedTrimmedAvg"]

grid_search = list(
    itertools.product(
        client_poison_percs, aggregation_strategies, node_features_flip_fracs
    )
)
# breakpoint()

all_experiments = {}
for client_poison_perc, aggregation_strategy, node_features_flip_frac in grid_search:
    new_params = fixed_params.copy()
    new_params["client_poison_perc"] = client_poison_perc
    new_params["aggregation_strategy"] = aggregation_strategy
    new_params["node_features_flip_frac"] = node_features_flip_frac
    # breakpoint()
    all_experiments[
        f"GAT-data_poisoning-client_poison_perc_{client_poison_perc}-agg_strategy_{aggregation_strategy}-flip_frac_{node_features_flip_frac}"
    ] = new_params

with open(
    f"experiment_configs/GAT_data_poisoning_experiments_rev5.json",
    "w",
) as outfile:
    json.dump(all_experiments, outfile, indent=4)
