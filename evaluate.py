#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh

import deep_sdf
import deep_sdf.workspace as ws
if not os.name == "nt":
    # We do not import this on Windows.
    import pytorch3d

def evaluate(experiment_directory, checkpoint, data_dir, split_filename, curvature_sampling=0.):

    with open(split_filename, "r") as f:
        split = json.load(f)
        

    chamfer_results = []

    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                logging.debug(
                    "evaluating " + os.path.join(dataset, class_name, instance_name)
                )
                checkpoint_ = f"{checkpoint}_on_train_set" if "train" in split_filename else checkpoint
                reconstructed_mesh_filename = ws.get_reconstructed_mesh_filename(
                    experiment_directory, checkpoint_, dataset, class_name, instance_name
                )

                logging.debug(
                    'reconstructed mesh is "' + reconstructed_mesh_filename + '"'
                )

                ground_truth_samples_filename = os.path.join(
                    data_dir,
                    "SurfaceSamples",
                    dataset,
                    class_name,
                    instance_name + ".ply",
                )

                logging.debug(
                    "ground truth samples are " + ground_truth_samples_filename
                )

                normalization_params_filename = os.path.join(
                    data_dir,
                    "NormalizationParameters",
                    dataset,
                    class_name,
                    instance_name + ".npz",
                )

                logging.debug(
                    "normalization params are " + ground_truth_samples_filename
                )

                ground_truth_points = trimesh.load(ground_truth_samples_filename)
                reconstruction = trimesh.load(reconstructed_mesh_filename)

                normalization_params = np.load(normalization_params_filename)

                chamfer_dist, all_dists = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
                    ground_truth_points,
                    reconstruction,
                    normalization_params["offset"],
                    normalization_params["scale"],
                    curvature_sampling=curvature_sampling
                )
                percentiles = np.percentile(all_dists, [90, 95])
                normal_consistency = deep_sdf.metrics.compute_metric(gen_mesh=reconstruction, metric="normal_consistency")

                logging.debug("chamfer distance: " + str(chamfer_dist))

                chamfer_results.append(
                    (os.path.join(dataset, class_name, instance_name), (chamfer_dist, percentiles), normal_consistency)
                )

    output_filename = os.path.join(
            ws.get_evaluation_dir(experiment_directory, checkpoint, True),
            "chamfer"
        )
    output_filename += "_on_train_set" if "train" in split_filename else ""
    output_filename += f".csv" if curvature_sampling == 0. else f"_{curvature_sampling:.3f}_curvature.csv"
    logging.info(split_filename)
    logging.info(output_filename)
    with open(output_filename,"w",) as f:
        # semicolon-separated CSV file
        f.write("shape;chamfer_dist;90th_percentile;95th_percentile;normal_consistency\n")
        for result in chamfer_results:
            f.write("{};{};{};{}\n".format(result[0], result[1][0], result[1][1][0], result[1][1][1], result[2]))


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="2000",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        default="../../shared/deepsdfcomp/data/",
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        default="../../shared/deepsdfcomp/experiments/splits/sv2_planes_test.json",
        help="The split to evaluate.",
    )
    arg_parser.add_argument(
        "--curvature_sampling",
        "-cs",
        dest="curvature_sampling",
        default=0.0,
        required=False,
        help="Amount of sampling wrt mesh curvature. 0 means smapling wrt. face area, 1 wrt. face curvature.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    curvature_sampling = args.curvature_sampling
    try:
        curvature_sampling = float(curvature_sampling)
        evaluate(
            args.experiment_directory,
            args.checkpoint,
            args.data_source,
            args.split_filename,
            curvature_sampling
        )
    except ValueError as ve:
        logging.error(f"Could not cast {args.curvature_sampling} to float" + str(ve.args))
