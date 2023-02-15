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
import pyvista
from tqdm import tqdm
from collections import deque

def vtk_mesh_to_trimesh(vtk_mesh):
    # https://github.com/pyvista/pyvista/discussions/2268
    poly = vtk_mesh.extract_surface().triangulate()
    points = poly.points
    faces = poly.faces.reshape((poly.n_faces, 4))[:, 1:]
    return trimesh.Trimesh(points, faces) 

def run_remeshing_until_cd(input_obj_path: str, output_obj_path: str, target_chamfer_dist: float, logs: list):

    start_time = time.time()
    # Target interval is 10% of the target chamfer distance.
    target_chamfer_dist_min = target_chamfer_dist * 0.9
    target_chamfer_dist_max = target_chamfer_dist * 1.1
    target_reduction = 0.70

    gt_mesh = pyvista.read(input_obj_path)
    gt_trimesh = vtk_mesh_to_trimesh(gt_mesh)
    gt_faces = gt_trimesh.faces.shape[0]

    last_target_reduction = None
    last_num_faces = deque(maxlen=10)
    last_cd = None
    lr = 100
    success = False

    # Do not try for more than N retries.
    for i in range(100):

        if len(last_num_faces) == 10 and all([_ == last_num_faces[0] for _ in last_num_faces]):            
            logging.info(f"Returning early, since num_faces stays at {last_num_faces[0]}/{gt_faces}.")
            break

        if not target_reduction >= 1.0:
            decimated_mesh = gt_mesh.decimate(target_reduction)
            decimated_trimesh = vtk_mesh_to_trimesh(decimated_mesh)
            decimated_mesh.save(output_obj_path)
            last_num_faces.append(decimated_trimesh.faces.shape[0])
            cd, all_cd = metrics.compute_metric(gt_trimesh, decimated_trimesh, metric="chamfer")

            if target_chamfer_dist_min < cd < target_chamfer_dist_max:
                # Target achieved.
                logging.debug(f"[{i:2d}] LR: {lr:.0e}: {cd:6f} --> {target_chamfer_dist:.6f} (reduction: {target_reduction:.6f})")
                logs.append([
                    input_obj_path, 
                    output_obj_path,
                    target_reduction, 
                    gt_faces,
                    last_num_faces[-1],
                    cd,
                    i
                ])
                success = True
                logging.debug(f"Converged after {i} iterations at target reduction {target_reduction:.6f} and CD {cd:.6f}.")
                break
        
        logging.debug(f"[{i:2d}] LR: {lr:.0e}: {cd:6f} --> {target_chamfer_dist:.6f} (reduction: {target_reduction:.6f})")

        if target_reduction >= 1.0 or (cd > target_chamfer_dist and last_target_reduction and last_cd):
            # Only converge to the target from the left (from lower values).
            lr *= 0.1
            target_reduction = last_target_reduction
            cd = last_cd
        else:
            last_target_reduction = target_reduction
            last_cd = cd
        
        target_reduction += target_reduction * (target_chamfer_dist - cd) * lr
        
    if not success:
        logging.info(f"No convergence after {i} iterations.")
        os.remove(output_obj_path)
    logging.debug(f"Took {time.time() - start_time:.01f} seconds.")


if __name__ == "__main__":

    output_dir = "data/quadriflow_until_cd_meshes"      # This needs to be changed to where you want your data to be extracted to!
    input_dir = "/mnt/hdd/ShapeNetCore.v2"
    # input_dir = "data/manifold_meshes"
    split_path = "examples/splits/sv2_planes_test.json"
    quadriflow_executable = "../QuadriFlow/build/quadriflow"

    os.makedirs(output_dir, exist_ok=True)

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "--target_chamfer_dist", "-cd",
        dest="target_chamfer_dist",
        default=0.000146,
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
        input_obj_path = os.path.join(input_dir, synset_id, shape_id, "models/model_normalized.obj")
        # input_obj_path = os.path.join(input_dir, synset_id, shape_id + ".obj")
        if not os.path.exists(input_obj_path):
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": input_obj_path,
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".ply"),
            "target_chamfer_dist": args.target_chamfer_dist
        })
        os.makedirs(os.path.join(output_dir, synset_id), exist_ok=True)

    # Logging to terminal.
    logging.info(f"Quadriflowing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")

    start_time = time.time()

    shared_logs = []
    for mtsa in tqdm(meshes_targets_and_specific_args):
        run_remeshing_until_cd(
            mtsa["input_obj_path"],
            mtsa["output_obj_path"],
            mtsa["target_chamfer_dist"],
            shared_logs,
        )

    logs_df = pd.DataFrame(
        shared_logs, 
        columns=["input_obj_path", "output_obj_path", "resolution", "gt_faces", "decimated_faces", "cd", "iteration"],
    )
    logs_df.to_csv(os.path.join(output_dir, "run_quadriflow_until_CD_logs.csv"))
