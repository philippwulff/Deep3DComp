# Runs Quadriflow with decreasing vertex number on ShapeNet meshes until a certain chamfer distance between the GT mesh and the quadriflowed mesh is reached.

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


def run_quadriflow_until_cd(quadriflow_exec_path: str, input_obj_path: str, output_obj_path: str, target_chamfer_dist: float, logs: list, debug=False):

    start_time = time.time()
    resolution = 1000      # the default in Quadriflow is around 100000
    # Target interval is 5% of the target chamfer distance.
    target_chamfer_dist_min = target_chamfer_dist * 0.95
    target_chamfer_dist_max = target_chamfer_dist * 1.05
    success = False

    # Do not try for more than 20 retries.
    for i in range(20):
        cmd = f"{quadriflow_exec_path} -i {input_obj_path} -o {output_obj_path} --resolution {int(resolution)}"
            
        if debug: 
            logging.info(f"Running cmd: {cmd}")
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL)#stdout=subprocess.STDOUT if debug else subprocess.DEVNULL)
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
        
        if not os.path.exists(output_obj_path):
            break
        cd, all_cd = metrics.compute_metric(input_obj_path, output_obj_path)

        if cd < target_chamfer_dist_min:
            pass
        if cd > target_chamfer_dist_max:
            pass
        elif target_chamfer_dist_min < cd < target_chamfer_dist_max:
            # Target achieved.
            num_faces = trimesh.load(output_obj_path).faces.shape[0]
            logs.append([
                input_obj_path, 
                output_obj_path, 
                resolution, 
                num_faces,
                cd
            ])
            success = True
            logging.debug(f"[run_quadriflow_until_cd] Converged after {i} iterations at resolution {resolution} and CD {cd}.")
            break
            
        # Decrease resolution is chamfer distance too low.
        # Increase resolution if chamfer distance too high.
        resolution += resolution * (cd - target_chamfer_dist) * 10      # the last factor is just a weight
        
    print("here")
    if not success:
        logging.debug(f"[run_quadriflow_until_cd] failed after {i} iterations.")
    logging.debug(f"[run_quadriflow_until_cd] Took {time.time() - start_time:.01f} seconds.")


if __name__ == "__main__":

    output_dir = "data/quadriflow_until_cd_meshes"      # This needs to be changed to where you want your data to be extracted to!
    #shapenet_dir = "/mnt/hdd/ShapeNetCore.v2"
    input_dir = "data/manifold_meshes"
    split_path = "examples/splits/sv2_planes_test_single.json"
    quadriflow_executable = "../QuadriFlow/build/quadriflow"

    os.makedirs(output_dir, exist_ok=True)

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "--target_chamfer_dist", "-cd",
        dest="target_chamfer_dist",
        default=0.02,
        help="The mean reconstruction chamfer distance to compress to.",
    )
    arg_parser.add_argument(
        "--num_threads",
        dest="num_threads",
        default=int(psutil.cpu_count() * 3/4),
        help="Number of threads to run quadriflow on.",
    )
    utils.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    utils.configure_logging(args)
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
        # Path is hardcoded to work with ShapeNetCore.v2
        # input_obj_path = os.path.join(input_dir, synset_id, shape_id, "models/model_normalized.obj")
        input_obj_path = os.path.join(input_dir, synset_id, shape_id + ".obj")
        if not os.path.exists(input_obj_path):
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": input_obj_path,
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".obj"),
            "target_chamfer_dist": args.target_chamfer_dist
        })
        os.makedirs(os.path.join(output_dir, synset_id), exist_ok=True)

    # Logging to terminal.
    logging.info(f"Quadriflowing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")

    # Starting separate jobs.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.num_threads)
    ) as executor:
        start_time = time.time()
        
        # Results logging list that is shared among all threads.
        shared_logs = []

        for i, mtsa in enumerate(meshes_targets_and_specific_args):
            executor.submit(
                run_quadriflow_until_cd,
                quadriflow_executable,
                mtsa["input_obj_path"],
                mtsa["output_obj_path"],
                mtsa["target_chamfer_dist"],
                shared_logs,
                debug=args.debug,
            )
            if i % 10 == 0:
                logging.info(f"Timing update: processed {i}/{len(shape_ids)} meshes in {time.time()-start_time:.0f} seconds. ETC: {(len(shape_ids)-i-1)*(time.time()-start_time)/(i+1):.0f}.")

        executor.shutdown()

    logs_df = pd.DataFrame(
        shared_logs, 
        columns=["input_obj_path", "output_obj_path", "resolution", "num_faces", "cd"],
    )
    logs_df.to_csv(os.path.join(output_dir, "run_quadriflow_until_CD_logs.csv"))
