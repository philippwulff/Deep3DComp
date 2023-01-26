from train_deep_sdf import main_function
import json
import numpy as np
import random
import os
import deep_sdf
from math import log10

def random_between(min, max):
    return min + random.random() * (max - min)


SEARCH_DIR = "/home/shared/deepsdf/searches/sirenVrelu"
HPARAM_RANGES_FILE = os.path.join(SEARCH_DIR, "hparam_ranges.json")
DEFAULT_SPECS_FILE = "/home/shared/deepsdf/searches/default_specs.json"
NUM_EXPERIMENTS = 100

import argparse

arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
deep_sdf.add_common_args(arg_parser)
args = arg_parser.parse_args()
deep_sdf.configure_logging(args)

# read data
with open(HPARAM_RANGES_FILE) as f:
    hparam_ranges = json.load(f)
with open(DEFAULT_SPECS_FILE) as f:
    default_specs = json.load(f)

for i in range(NUM_EXPERIMENTS):
    searched_hparams = {}
    specs = {**default_specs}
    
    # sample non-architecture hparams
    for k, v in hparam_ranges.items():
        if k in ("NetworkSpecs", "LearningRateSchedule"):
            continue
        if type(v[0]) == str:
            sampled_v = np.random.choice(v)
        elif type(v[0]) == int:
            sampled_v = int(random_between(*v))
        else:
            sampled_v = random_between(*v)
        specs[k] = sampled_v
        if len(v) > 1 and (v[0] != v[1]):
            searched_hparams[k] = sampled_v

    # sample architecture hparams
    for k, v in hparam_ranges["NetworkSpecs"].items():
        if type(v[0]) == str:
            sampled_v = np.random.choice(v)
        elif type(v[0]) == int:
            sampled_v = int(random_between(*v))
        else:
            sampled_v = random_between(*v)
        # handle build of dims array separateley
        if k in ("depth", "width"):
            if not "depth" in hparam_ranges["NetworkSpecs"] and "width" in hparam_ranges["NetworkSpecs"]:
                raise KeyError("Depth and width can only be searched when ranges for bot are given")
            depth = int(random_between(*hparam_ranges["NetworkSpecs"]["depth"]))
            width = int(random_between(*hparam_ranges["NetworkSpecs"]["width"]))
            sampled_dims = [width, ] * depth
            if len(v) > 1 and (v[0] != v[1]):
                searched_hparams["depth"] = depth
                searched_hparams["width"] = width
            specs["NetworkSpecs"]["dims"] = sampled_dims
            continue
        specs["NetworkSpecs"][k] = sampled_v
        if len(v) > 1 and (v[0] != v[1]):
            searched_hparams[k] = sampled_v

    # sample lr schedule hparams
    if len(hparam_ranges["LearningRateSchedule"]) != 2:
        raise ValueError("Please provide ranges for both LR shedulers or leave them empty")
    for i in range(2):
        for k, v in hparam_ranges["LearningRateSchedule"][i].items():
            if k == "InitialExp":
                sampled_v = 10**random_between(*v)
                specs["LearningRateSchedule"][i]["Initial"] = sampled_v
                if len(v) > 1 and (v[0] != v[1]):
                    searched_hparams["LrInitial"] = sampled_v
                continue
            elif type(v[0]) == str:
                sampled_v = np.random.choice(v)
            elif type(v[0]) == int:
                sampled_v = int(random_between(*v))
            else:
                sampled_v = random_between(*v)
            specs["LearningRateSchedule"][i][k] = sampled_v
            if len(v) > 1 and (v[0] != v[1]):
                searched_hparams[k] = sampled_v
    
    # find exp name (continuous counter and searched hparams)
    exp_number = 0
    if os.path.exists(SEARCH_DIR):
        dir_names = [name for name in os.listdir(SEARCH_DIR) if os.path.isdir(os.path.join(SEARCH_DIR, name))]
        exp_nos = [int(name.split("_")[1]) for name in dir_names]
        if len(exp_nos)>0:
            exp_number = max(exp_nos) + 1
    exp_name = f"exp_{exp_number:04}"
    
    # create exp directory and write specs.json
    if len(searched_hparams) > 0:
        exp_name += "_" + "_".join(f"{k}={round(v, -int(log10(abs(v)))+3)}" if isinstance(v, float) else f"{k}={v}" for k, v in searched_hparams.items())
    exp_dir = os.path.join(SEARCH_DIR, exp_name)
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "specs.json"), "w+") as f:
        json.dump(specs, f, indent=4)
    
    # start experimen
    main_function(exp_dir, None, 1)
