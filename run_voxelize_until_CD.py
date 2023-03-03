# Runs voxelization with decreasing vertex number on ShapeNet meshes until a certain chamfer distance between the GT mesh and the quadriflowed mesh is reached.

import os
import json
import logging 
import argparse
import psutil
import time
from deep_sdf import metrics, utils
import trimesh
import pandas as pd
from tqdm import tqdm
import mesh_to_sdf
import skimage
import random


def run_voxelize_until_cd(input_obj_path: str, output_obj_path: str, target_chamfer_dist: float, logs: list):

    start_time = time.time()
    # Target interval is ±50% of the target chamfer distance.
    target_chamfer_dist_min = target_chamfer_dist * 0.5
    target_chamfer_dist_max = target_chamfer_dist * 1.5

    gt_mesh = utils.scale_to_unit_sphere(trimesh.load(input_obj_path))
    logging.debug(f"Voxelizing mesh: {input_obj_path}")

    # Initial bisection segment of voxel grid resolutions.
    bisection_segment = [32, 200]
    success = False
    bad_mesh_resolutions = []
    reconstruction = None
    cd = 1000

    for i in range(20):    # Do not try for more than N retries.
        if len(bad_mesh_resolutions) > 5 or any([int(bisection_segment[0]) == int(bisection_segment[1]) + _ for _ in [0, 1, -1]]):
            # If to many BadMeshExceptions or the bisection segment has become too small.
            break
        # The query point is in the middle of the bisection segment.
        voxel_resolution = int((bisection_segment[0] + bisection_segment[1]) / 2)
        if voxel_resolution in bad_mesh_resolutions:
            # Add some noise to voxel resolution is it caused a BadMeshException
            voxel_resolution = int(max(bisection_segment[0], voxel_resolution - random.randint(10, 20)))
            bisection_segment = [bisection_segment[0], 2*voxel_resolution - bisection_segment[0]]
        voxel_size = 2.0 / (voxel_resolution - 1)

        # Extract voxel grid.
        try:
            voxels = mesh_to_sdf.mesh_to_voxels(gt_mesh, voxel_resolution=voxel_resolution, check_result=True, pad=True, sign_method="depth")
        except mesh_to_sdf.BadMeshException:
            logging.debug(f"Caught BadMeshException at voxel-res {voxel_resolution} ({input_obj_path})")
            bad_mesh_resolutions.append(voxel_resolution)
            continue
        # Reconstruct mesh from voxel grid.
        verts, faces, normals, values = skimage.measure.marching_cubes(voxels, level=0.0, spacing=[voxel_size] * 3, method="lewiner")
        reconstruction = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        reconstruction = utils.scale_to_unit_sphere(reconstruction)
        # Compute reconstruction quality.
        cd, _ = metrics.compute_metric(gt_mesh, reconstruction, metric="chamfer")

        vert_cnt = len(reconstruction.vertices)
        logging.debug(f"CD: {cd:5f} (target={target_chamfer_dist:5f}) | Voxel-res={voxel_resolution} | Vertices={vert_cnt}")

        if target_chamfer_dist_min < cd < target_chamfer_dist_max:
            success = True
            logs.append([
                input_obj_path, 
                output_obj_path,
                voxel_resolution, 
                len(gt_mesh.vertices),
                vert_cnt,
                cd,
                i
            ])
            break
        # Perform bisection.
        elif cd < target_chamfer_dist:
            # If in the left half of the segment.
            bisection_segment = [bisection_segment[0], (bisection_segment[0] + bisection_segment[1]) / 2] 
        else:
            # If in the right half of the segment.
            bisection_segment = [(bisection_segment[0] + bisection_segment[1]) / 2, bisection_segment[1]] 
    
    if success:
        logging.debug(f"Convergence after {i} iterations.")
        logging.debug(f"Reduced with voxel-res {voxel_resolution} to chamfer distance of {cd:4f}. Reduced Mesh has {vert_cnt} Vertices.")
        with open(output_obj_path, "wb+") as f:
            f.write(trimesh.exchange.ply.export_ply(reconstruction))
    else:
        logging.debug(f"No convergence after {i} iterations.")
    logging.debug(f"Took {time.time() - start_time:.01f} seconds.")


if __name__ == "__main__":

    output_dir = "data/voxelize_until_cd_meshes"      # This needs to be changed to where you want your data to be extracted to!
    input_dir = "/mnt/hdd/ShapeNetCore.v2"
    input_dir = "../../shared/deepsdfcomp/data/manifold_meshes"
    split_path = "examples/splits/sv2_planes_test.json"

    # Setup args and logging.
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        "--target_chamfer_dist", "-c",
        dest="target_chamfer_dist",
        default=0.001,
        type=float,
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

    output_dir += f"_CD={args.target_chamfer_dist}"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare all input and output mesh files.
    with open(split_path, "r") as f:
        split = json.load(f)
        dataset_name = list(split.keys())[0]
        synset_id = list(split[dataset_name].keys())[0]
        shape_ids = split[dataset_name][synset_id]

    meshes_targets_and_specific_args = []
    file_not_found_cnt = 0
    for shape_id in shape_ids:
        input_obj_paths = [
            # Path that works with ShapeNetCore.v2
            os.path.join(input_dir, synset_id, shape_id, "models/model_normalized.obj"), 
            # Path that works with the DeepSDF dataset structure.
            os.path.join(input_dir, synset_id, shape_id + ".obj")
        ]
        existing_paths = [p for p in input_obj_paths if os.path.exists(p)]  # Should contain only one value.
        if not existing_paths:
            file_not_found_cnt += 1
            continue
        meshes_targets_and_specific_args.append({
            "input_obj_path": existing_paths[0],
            "output_obj_path": os.path.join(output_dir, synset_id, shape_id + ".ply"),
            "target_chamfer_dist": args.target_chamfer_dist
        })
        os.makedirs(os.path.join(output_dir, synset_id), exist_ok=True)

    # Logging to terminal.
    logging.info(f"Voxelizing a total of {len(shape_ids)-file_not_found_cnt} shapes.")
    if file_not_found_cnt:
        logging.info(f"Could not find {file_not_found_cnt} out of {len(shape_ids)} shapes.")

    start_time = time.time()

    shared_logs = []
    try:
        for mtsa in tqdm(meshes_targets_and_specific_args):
            if os.path.exists(mtsa["output_obj_path"]):
                continue
            run_voxelize_until_cd(
                mtsa["input_obj_path"],
                mtsa["output_obj_path"],
                mtsa["target_chamfer_dist"],
                shared_logs,
            )
    except KeyboardInterrupt:
        logging.info("Cleaning up and exiting.")
    finally:
        df_output_path = os.path.join(output_dir, "run_voxelize_until_CD_logs.csv")
        logs_df = pd.DataFrame(
            shared_logs, 
            columns=["input_obj_path", "output_obj_path", "voxel_resolution", "gt_vertices", "decimated_vertices", "cd", "iteration"],
        )
        if os.path.exists(df_output_path):
            logs_df_old = pd.read_csv(df_output_path)
            logs_df_all = pd.concat([logs_df_old, logs_df], ignore_index=True, axis=0)
            logs_df_all.to_csv(df_output_path)
        else:
            logs_df.to_csv(df_output_path, index=False)
