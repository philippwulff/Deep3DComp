from train_deep_sdf import main_function
import json
import numpy as np
import random
import os
import deep_sdf
from math import log10
from collections.abc import MutableMapping
from itertools import product
import argparse
import copy
import re
import logging


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
    """Helper function"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Define default hparams.
SEARCH_DIR = "/home/shared/deepsdfcomp/searches/siren_500_gridsearch_quarter_params"
DEFAULT_SPECS_FILE = "/home/shared/deepsdfcomp/searches/siren_500_gridsearch_quarter_params/default_specs.json"

# Define hparams search options.
GRID_AXES = {
    "NetworkSpecs": {
        "norm_layers": [[],],
        "dropout": [[],],
        "dropout_prob" : [0,],
        "latent_in" : [[], [4]],
        # "xyz_in_all": [True, False],
        "latent_dropout" : [True, False],
    },
    # 1st lr schedule is for network and 2nd lr schedule is for latents
    "LearningRateSchedule": [       
        [{
            "Type": "Step",
            "Initial": 0.0005,
            "Interval": 500,
            "Factor": 0.5
        },
        {
            "Type": "Step",
            "Initial": 0.001,
            "Interval": 500,
            "Factor": 0.5
        }],
        [{
            "Type": "Step",
            "Initial": 0.005,
            "Interval": 500,
            "Factor": 0.5
        },
        {
            "Type": "Step",
            "Initial": 0.01,
            "Interval": 500,
            "Factor": 0.5
        }],
    ],
    "CodeLength" : [16, 200],
    "CodeRegularization" : [True, False],
}

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Hyperparameter Grid Search")
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    # Get all possible combinations of NetworkSpecs.
    network_specs_permutations = list(dict(zip(GRID_AXES["NetworkSpecs"].keys(), values)) 
                            for values in product(*GRID_AXES["NetworkSpecs"].values()))
    
    # Get all possible combinations of the other options.
    grid_axes_without_network_specs = {k: v for k, v in GRID_AXES.items() if k != "NetworkSpecs"}
    other_permutations = list(dict(zip(grid_axes_without_network_specs.keys(), values)) 
                            for values in product(*grid_axes_without_network_specs.values()))

    # Combine all combinations and generate experiment hparams list.
    all_permutations = list(product(network_specs_permutations, other_permutations))
    exps = []
    for perm in all_permutations:
        exps.append({
            "NetworkSpecs": perm[0], 
            **perm[1]
        })
        
    # Read default specifications in order to override them.
    with open(DEFAULT_SPECS_FILE) as f:
        default_specs = json.load(f)

    for exp in exps[16:]:
        # Produce specs for this experiment.
        specs = copy.deepcopy(default_specs)
        for k, v in exp.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    specs[k][kk] = vv
                continue
            specs[k] = v

        # Get exp name (counter + searched hparams).
        exp_number = 0
        if os.path.exists(SEARCH_DIR):
            dir_names = [name for name in os.listdir(SEARCH_DIR) if os.path.isdir(os.path.join(SEARCH_DIR, name))]
            exp_nos = [int(name.split("_")[1]) for name in dir_names]
            if len(exp_nos)>0:
                exp_number = max(exp_nos) + 1
        exp_name = f"exp_{exp_number:04}"
        
        searched_hparams = flatten_dict(exp)
        searched_hparams["LearningRateSchedule"] = searched_hparams["LearningRateSchedule"][0]['Initial']
        searched_hparams = {k.replace("NetworkSpecs.", ""):v for k, v in searched_hparams.items()}
        if len(searched_hparams) > 0:
            exp_name += "_" + "_".join(f"{k}={round(v, -int(log10(abs(v)))+3)}" if isinstance(v, float) else f"{k}={v}" for k, v in searched_hparams.items() if v)
        # Create exp directory and write specs.json
        exp_dir = os.path.join(SEARCH_DIR, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            continue
        with open(os.path.join(exp_dir, "specs.json"), "w+") as f:
            json.dump(specs, f, indent=4)
        
        # Start experiment
        logging.info(f"STARTING EXP: {exp_name}")
        main_function(exp_dir, continue_from=None, batch_split=3)     
