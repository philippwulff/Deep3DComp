from train_deep_sdf import main_function
import json
import numpy as np
import random
import os

def random_between(min, max):
    return min + random.random() * (max - min)

SEARCH_DIR = "/home/shared/deepsdf/searches/test_search"
HPARAM_RANGES_FILE = "/home/shared/deepsdf/searches/test_search/hparam_ranges.json"
DEFAULT_SPECS_FILE = "/home/shared/deepsdf/searches/default_specs.json"
NUM_EXPERIMENTS = 10

# read data
with open(HPARAM_RANGES_FILE) as f:
    hparam_ranges = json.load(f)
with open(DEFAULT_SPECS_FILE) as f:
    default_specs = json.load(f)

for i in range(NUM_EXPERIMENTS):
    # sample non-architecture hparams
    searched_hparams = {}
    specs = {**default_specs}
    for k, v in hparam_ranges.items():
        if k == "NetworkSpecs":
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

    exp_number = 0
    if os.path.exists(SEARCH_DIR):
        dir_names = [name for name in os.listdir(SEARCH_DIR) if os.path.isdir(os.path.join(SEARCH_DIR, name))]
        exp_nos = [int(name.split("_")[1]) for name in dir_names]
        if len(exp_nos)>0:
            exp_number = max(exp_nos) + 1
    exp_name = f"exp_{exp_number:04}"
    if len(searched_hparams) > 0:
        exp_name += "_" + "_".join(f"{k}={v}" for k, v in searched_hparams.items())
    exp_dir = os.path.join(SEARCH_DIR, exp_name)
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "specs.json"), "w+") as f:
        json.dump(specs, f, indent=4)
    
    main_function(exp_dir, None, 1)
