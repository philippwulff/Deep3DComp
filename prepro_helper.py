# Runs preprocessing on multiple shape classes (Helper for preprocess_data.py)

import os
import json
import logging 
import argparse
import deep_sdf
import psutil
import subprocess
import time

# For headless rendering (turn this off, if you are not on a remote machine)
os.environ["PANGOLIN_WINDOW_URI"] = "headless://"


def main(data_dir, source_dir, splits_dir, debug=False):

    num_threads = psutil.cpu_count() - 4
    logging.info(f"Using {num_threads} cores.")

    all_splits_paths = []
    for fname in os.listdir(splits_dir):
        if fname.endswith(".json"):
            all_splits_paths.append(os.path.join(splits_dir, fname))
    all_splits_paths.sort()

    logging.info(f"Preprocessing data {source_dir} --> {data_dir}.")
    all_splits_paths_str = "\n\t" + "\n\t".join(all_splits_paths)
    logging.info(f"Found these splits-files to preprocess:{all_splits_paths_str}")

    for i, split_path in enumerate(all_splits_paths):
        start_time = time.time()
        with open(split_path, "r") as f:
            split = json.load(f)
            num_shapes = len(list(list(split.values())[0].values())[0])
        logging.info(f"[{i}/{len(all_splits_paths)}] Preprocessing split: {split_path} (containing {num_shapes} shapes).")

        cmd_train = f"python ./preprocess_data.py --data_dir {data_dir} --name ShapeNetV2 --source {source_dir} --split {split_path} --threads {num_threads} --skip"
        cmd_eval = f"{cmd_train} --surface"
        cmd_test = f"{cmd_train} --test"
        
        for cmd in [cmd_train, cmd_eval, cmd_test]:
            if debug: 
                logging.info(f"Running cmd: {cmd}")
            try:
                subprocess.run(cmd, shell=True, capture_output=not debug, check=True)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.STDOUT if debug else subprocess.DEVNULL)
                p.wait()
            except KeyboardInterrupt:
                p.terminate()


        duration_total = time.time() - start_time
        duration_min = int(duration_total % 60)
        duration_sec = duration_total - duration_min*60
        logging.info(f"Preprocessing {num_shapes} shapes took {duration_min}:{duration_sec} (min:sec).")


if __name__ == "__main__":

    data_dir = "../../shared/deepsdfcomp/data"
    source_dir = "/mnt/hdd/ShapeNetCore.v2"
    splits_dir = "./examples/splits"

    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)

    main(data_dir, source_dir, splits_dir, debug=True)
