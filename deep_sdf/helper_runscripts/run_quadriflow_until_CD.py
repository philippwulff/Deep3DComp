# Runs Quadriflow with decreasing vertex number on ShapeNet meshes until a certain chamfer distance between the GT mesh and the quadriflowed mesh is reached.

import os
import json
import logging 
import argparse
import deep_sdf
import psutil
import subprocess
import concurrent
import time


def run_quadriflow_until_cd(quadriflow_exec_path: str, input_obj_path: str, output_obj_path: str, target_chamfer_dist: float, debug=False):

    start_time = time.time()

    cmd = f"{quadriflow_exec_path} -i {input_obj_path} -o {output_obj_path}"
        
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

    output_dir = "../../../../shared/deepsdfcomp/data"        # This needs to be changed to where you want your data to be extracted to!
    shapenet_dir = "/mnt/hdd/ShapeNetCore.v2"
    split_path = "../../examples/splits"
    quadriflow_executable = "../../../Quadriflow/build/quadriflow"

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "--target_chamfer_dist", "-cd",
        dest="target_chamfer_dist",
        default=0.7,
        help="The mean reconstruction chamfer distance to compress to.",
    )
    arg_parser.add_argument(
        "--num_threads",
        dest="num_threads",
        default=int(psutil.cpu_count() * 3/4),
        help="Number of threads to run quadriflow on.",
    )
    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)
    logging.info(f"Using {args.num_threads} cores.")

    # Prepare all input and output mesh files.
    with open(split_path, "r") as f:
        split = json.load(f)
        dataset_name = list(split.keys())[0]
        synset_id = list(split[dataset_name].keys())[0]
        shape_ids = split[dataset_name][synset_id]

    meshes_targets_and_specific_args = []
    file_not_found_cnt = 0
    for shape_id in shape_ids:
        input_obj_path = os.path.join(shapenet_dir, synset_id, shape_id, "models/model_normalized.obj")
        if os.path.exists(input_obj_path):
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": input_obj_path,
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".obj"),
            "target_chamfer_dist": args.target_chamfer_dist
        })

    # Logging to terminal.
    logging.info(f"Quadriflowing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")

    # Starting separate jobs.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:

        for mtsa in meshes_targets_and_specific_args:
            executor.submit(
                run_quadriflow_until_cd,
                quadriflow_executable,
                mtsa["input_obj_path"],
                mtsa["output_obj_path"],
                mtsa["target_chamfer_dist"],
            )

        executor.shutdown()
