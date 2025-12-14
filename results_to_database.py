from qaoa_training_pipeline.pre_processing.feature_extraction import GraphFeatureExtractor
from qaoa_training_pipeline.utils.graph_utils import load_graph, graph_to_operator
from pathlib import Path
import json

best_results_path = Path("data/training/best_parameters_summary.json")
instances_path = Path("instances")
database_path = "./optimized_angles_database.json"


with open(best_results_path, "r") as f:
    best_results = json.load(f)

best_results_SV = best_results["SV"]
for input_file in best_results_SV.keys():
    instance_results = best_results_SV[input_file]
    for depth in instance_results.keys():
        if "regular" in input_file:
            instances_subdir = instances_path / "random_regular"
        elif "swap" in input_file:
            instances_subdir = instances_path / "line_to_full"
        elif "erdosrenyi" in input_file:
            instances_subdir = instances_path / "erdos_renyi"
        elif "heavyhex" in input_file:
            instances_subdir = instances_path / "heavy_hex"
            
        input_path = instances_subdir / input_file
        graph = load_graph(input_path)
    
        cost_op = graph_to_operator(graph, pre_factor=-0.5)
    
        feature_extractor = GraphFeatureExtractor(extract_num_nodes=False, extract_num_edges= False, extract_density=False)
    
        features = feature_extractor(cost_op, depth)
        features_str = ", ".join(map(str, features))
    
        with open(database_path, "r") as f:
            database_dict = json.load(f)
            
        opt_params = instance_results[depth]["qaoa_angles"]
        generating_method = instance_results[depth]["result_file_name"]
        
        new_entry = {
            "qaoa_angles": [opt_params],
            "metadata": [generating_method]
        }
        
        if features_str not in database_dict.keys():
            database_dict[features_str] = new_entry
        else:
            database_dict[features_str]["qaoa_angles"].append(opt_params)
            database_dict[features_str]["metadata"].append(generating_method)
        with open(database_path, "w") as f:
            json.dump(database_dict,f,indent=4, sort_keys=True)