import itertools
import json


num_features = [4, 8, 16]

num_hidden_layers = [1, 2, 4]

grid_search = list(itertools.product(num_features, num_hidden_layers))


fixed_params = {
    "num_clients": 1,
    "dataset_name": "Cora",
    "slice_method": "node_feature",
    "percentage_overlap": 0,
    "model_type": "GAT",
    "epochs_per_client": 10,
    "num_rounds": 1,
    "aggregation_strategy": "FedAdagrad",
    "dry_run": True,
}

for model_type in ["GAT", "GCN"]:
    all_experiments = {}
    new_params = fixed_params.copy()
    new_params["model_type"] = model_type

    for features, hidden_layers in grid_search:
        new_params_w_l_f = new_params.copy()
        new_params_w_l_f["num_hidden_params"] = features
        new_params_w_l_f["num_hidden_layers"] = hidden_layers

        all_experiments[
            f"{model_type}_features-{features}_layers-{hidden_layers}"
        ] = new_params_w_l_f

    with open(
        f"experiment_configs/{model_type}-hyperparameter_tuning_experiments.json",
        "w",
    ) as outfile:
        json.dump(all_experiments, outfile)
