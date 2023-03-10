from train_deep_sdf import main_function
import json
import numpy as np
import random
import os
import deep_sdf
from math import log10

def random_between(min, max):
    return min + random.random() * (max - min)

def sample_value(values):
    if len(values) == 1:
        return values[0]
    if type(values[0]) in (str, bool):
        sampled_value = np.random.choice(values)
    elif type(values[0]) == int:
        sampled_value = int(random_between(*values))
    else:
        sampled_value = random_between(*values)
    return sampled_value

SEARCH_DIR = "/home/shared/deepsdfcomp/searches/siren_500_latentsize_quarter_params"
# HPARAM_RANGES_FILE = "/home/shared/deepsdfcomp/searches/ffe_500_shapes/hparam_ranges_ref.json"
HPARAM_RANGES_FILE = os.path.join(SEARCH_DIR, "hparam_ranges.json")
# DEFAULT_SPECS_FILE = "/home/freissmuth/deepsdf/examples/plane_dsdf/specs.json"
DEFAULT_SPECS_FILE = "/home/shared/deepsdfcomp/searches/siren_500_gridsearch_quarter_params/exp_0019_latent_in=[4]_latent_dropout=True_LearningRateSchedule=0.0005_CodeLength=200/specs.json"
NUM_EXPERIMENTS = 5

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

codelengths = [16, 32, 64, 127, 200]
for iExp in range(NUM_EXPERIMENTS):
    searched_hparams = {}
    specs = {**default_specs}
    
    # sample non-architecture hparams
    for k, v in hparam_ranges.items():
        if k in ("NetworkSpecs", "LearningRateSchedule"):
            continue
        sampled_v = sample_value(v)
        specs[k] = sampled_v
        if len(v) > 1 and (v[0] != v[1]):
            searched_hparams[k] = sampled_v

    # sample architecture hparams
    for k, v in hparam_ranges["NetworkSpecs"].items():
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
        if len(v) == 0:
            specs["NetworkSpecs"][k] = []
            continue
        sampled_v = sample_value(v)
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
            else:
                sampled_v = sample_value(v)
            specs["LearningRateSchedule"][i][k] = sampled_v
            if len(v) > 1 and (v[0] != v[1]):
                searched_hparams[k] = sampled_v
    
    # # TODO remove
    # if specs["NetworkSpecs"]["nonlinearity"] == "relu":
    #     continue
    specs["CodeLength"] = codelengths[iExp]
    searched_hparams["CodeLength"] = codelengths[iExp]

    

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
    
    # start experiment
    main_function(exp_dir, None, 3)
