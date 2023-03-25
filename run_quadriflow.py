# Runs ManifoldPlus on ShapeNet meshes.

import os
import json
import logging 
import argparse
import psutil
import subprocess
import concurrent
import time
from deep_sdf import metrics, utils
import trimesh
import pandas as pd


def run_manifold(quadriflow_exec_path: str, input_obj_path: str, output_obj_path: str, debug=False):

    start_time = time.time()
    if os.path.exists(output_obj_path):
        return

    cmd = f"{quadriflow_exec_path} -i {input_obj_path} -o {output_obj_path}"
            
    logging.info(f"Running cmd: {cmd}")
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)#stdout=subprocess.STDOUT if debug else subprocess.DEVNULL)
        p.wait()
    except KeyboardInterrupt:
        p.terminate()
    
    if not os.path.exists(output_obj_path):
        logging.debug(f"[run_quadriflow] Failure.")
    
    logging.debug(f"[run_quadriflow] Took {time.time() - start_time:.01f} seconds.")


if __name__ == "__main__":

    output_dir = "data/quadriflow_meshes"                             # This needs to be changed to where you want your data to be extracted to!
    output_dir = "../../shared/deepsdfcomp/data/quadriflow_meshes"        

    shapenet_dir = "/mnt/hdd/ShapeNetCore.v2"
    split_path = "examples/splits/sv2_planes_test.json"
    quadriflow_executable = "../QuadriFlow/build/quadriflow"

    os.makedirs(output_dir, exist_ok=True)

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "--n_jobs",
        dest="n_jobs",
        default=int(psutil.cpu_count() * 1/4),
        help="Number of threads to run quadriflow on.",
    )
    utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)
    logging.info(f"Using {args.n_jobs} cores.")

    # Prepare all input and output mesh files.
    with open(split_path, "r") as f:
        split = json.load(f)
        dataset_name = list(split.keys())[0]
        synset_id = list(split[dataset_name].keys())[0]
        shape_ids = split[dataset_name][synset_id]

    meshes_targets_and_specific_args = []
    file_not_found_cnt = 0
    for shape_id in shape_ids:
        # Path is hardcoded to work with ShapeNetCore.v2
        # input_obj_path = os.path.join(shapenet_dir, synset_id, shape_id, "models/model_normalized.obj")
        input_obj_path = os.path.join("../../shared/deepsdfcomp/data/manifold_meshes", synset_id, shape_id + ".obj")
        
        if not os.path.exists(input_obj_path):
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": input_obj_path,
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".obj"),
        })
        os.makedirs(os.path.join(output_dir, synset_id), exist_ok=True)

    # Logging to terminal.
    logging.info(f"Quadriflowing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")

    # Starting separate jobs.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.n_jobs)
    ) as executor:
        start_time = time.time()
        
        # Results logging list that is shared among all threads.
        shared_logs = []

        for i, mtsa in enumerate(meshes_targets_and_specific_args):
            executor.submit(
                run_manifold,
                quadriflow_executable,
                mtsa["input_obj_path"],
                mtsa["output_obj_path"],
                debug=args.debug,
            )

        executor.shutdown()

