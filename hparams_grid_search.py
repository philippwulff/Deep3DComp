from train_deep_sdf import main_function
import json
import numpy as np
import random
import os
import deep_sdf
from math import log10
from collections.abc import MutableMapping
from itertools import product

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

SEARCH_DIR = "/home/shared/deepsdfcomp/searches/siren_500_gridsearch_quarter_params"
DEFAULT_SPECS_FILE = "/home/shared/deepsdfcomp/searches/siren_500_latentsize_quarter_params/exp_0000_noBN_noDO_adjLR_xyzIA_noWN_CodeLength=211/specs.json"

import argparse

arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
deep_sdf.add_common_args(arg_parser)
args = arg_parser.parse_args()
deep_sdf.configure_logging(args)

exps = []
grid_axes = {"NetworkSpecs": {"norm_layers": [[],],
                              "dropout": [[],],
                              "dropout_prob" : [0,],
                              "latent_in" : [[], [4]],
                              "xyz_in_all": [True, False], # maybe remove
                              "latent_dropout" : [True, False],},
             "LearningRateSchedule": {[
             ]},
             "CodeLength" : [10, 256],
             "CodeRegularization" : [True, False],
            }

network_specs_list = list(dict(zip(grid_axes["NetworkSpecs"].keys(), values)) 
                          for values in product(*grid_axes["NetworkSpecs"].values()))
grid_axes_copy = {**grid_axes}; 
if "NetworkSpecs" in grid_axes_copy.keys():
    del grid_axes_copy["NetworkSpecs"]
root_list = list(dict(zip(grid_axes_copy.keys(), values)) 
                          for values in product(*grid_axes_copy.values()))
exps_separate = list(product(network_specs_list, root_list))
for exp_separate in exps_separate:
    exps.append({"NetworkSpecs": exp_separate[0], ** exp_separate[1]})

    
# read data
with open(DEFAULT_SPECS_FILE) as f:
    default_specs = json.load(f)

codelengths = [4, 32, 128, 256]
for exp in exps:
    specs = {**default_specs}
    for k, v in exp.items():
        if type(v) == dict:
            for kk, vv in v.items():
                specs[k][kk] = vv
            continue
        specs[k] = v

    # find exp name (continuous counter and searched hparams)
    exp_number = 0
    if os.path.exists(SEARCH_DIR):
        dir_names = [name for name in os.listdir(SEARCH_DIR) if os.path.isdir(os.path.join(SEARCH_DIR, name))]
        exp_nos = [int(name.split("_")[1]) for name in dir_names]
        if len(exp_nos)>0:
            exp_number = max(exp_nos) + 1
    exp_name = f"exp_{exp_number:04}"
    
    # create exp directory and write specs.json
    searched_hparams = flatten_dict(exp)
    if len(searched_hparams) > 0:
        exp_name += "_" + "_".join(f"{k}={round(v, -int(log10(abs(v)))+3)}" if isinstance(v, float) else f"{k}={v}" for k, v in searched_hparams.items())
    exp_dir = os.path.join(SEARCH_DIR, exp_name)
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "specs.json"), "w+") as f:
        json.dump(specs, f, indent=4)
    
    # start experiment
    main_function(exp_dir, None, 3)
