#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import signal
import sys
import os
import logging
import math
import json
import time
import copy
import random

import deep_sdf
from deep_sdf import mesh, metrics, lr_scheduling
import deep_sdf.workspace as ws
import reconstruct

from torch.utils.tensorboard import SummaryWriter


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory: str, continue_from, batch_split: int):

    logging.debug("running experiment " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + str(specs["Description"]))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = lr_scheduling.get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))
    
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    # Get train evaluation settings.
    eval_grid_res = get_spec_with_default(specs, "EvalGridResolution", 256)
    eval_train_scene_num = get_spec_with_default(specs, "EvalTrainSceneNumber", 5)
    eval_train_frequency = get_spec_with_default(specs, "EvalTrainFrequency", 50)
    eval_train_scene_idxs = random.sample(range(len(sdf_dataset)), min(eval_train_scene_num, len(sdf_dataset)))
    logging.debug(f"Plotting {eval_train_scene_num} shapes with indices {eval_train_scene_idxs}")

    # Get test evaluation filenames.
    with open(test_split_file, "r") as f:
        test_split = json.load(f)
    eval_test_frequency = get_spec_with_default(specs, "EvalTestFrequency", 10)
    eval_test_scene_num = get_spec_with_default(specs, "EvalTestSceneNumber", 100)
    eval_test_filenames = deep_sdf.data.get_instance_filenames(data_source, test_split)
    eval_test_filenames = random.sample(eval_test_filenames, min(eval_test_scene_num, len(eval_test_filenames)))

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [{
            "params": decoder.parameters(),
            "lr": lr_schedules[0].get_learning_rate(0),
        },
        {
            "params": lat_vecs.parameters(),
            "lr": lr_schedules[1].get_learning_rate(0),
        }]
    )

    summary_writer = SummaryWriter(log_dir=os.path.join(experiment_directory, ws.tb_logs_dir))

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    
    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        # Required because evaluation puts the decoder into 'eval' mode.
        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)
        for sdf_data, indices in sdf_loader:
            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i])
                input = torch.cat([batch_vecs, xyz[i]], dim=1)
                
                # NN optimization
                pred_sdf = decoder(input)

                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)
                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()

                chunk_loss.backward()

                batch_loss += chunk_loss.item()
            
                
            logging.debug("loss = {}".format(batch_loss))
            summary_writer.add_scalar("Loss/train", batch_loss, global_step=epoch)
            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

        # LOG EPOCH
        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
        summary_writer.add_scalar("Learning Rate/Params", lr_schedules[0].get_learning_rate(epoch), global_step=epoch)
        summary_writer.add_scalar("Learning Rate/Code", lr_schedules[1].get_learning_rate(epoch), global_step=epoch)

        mlm = get_mean_latent_vector_magnitude(lat_vecs)
        lat_mag_log.append(mlm)
        summary_writer.add_scalar("Mean Latent Magnitude", mlm, global_step=epoch)

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )

        # EVALUATION 
        if epoch % eval_train_frequency == 0:
            # Training-set evaluation: Reconstruct mesh from learned latent and compute metrics.
            chamfer_dist_sum = 0
            for index in eval_train_scene_idxs:
                lat_vec = lat_vecs(torch.LongTensor([index])).cuda()
                mesh_class_id = sdf_dataset.npyfiles[index].split(".npz")[0].split(os.sep)[-2]
                mesh_shape_id = sdf_dataset.npyfiles[index].split(".npz")[0].split(os.sep)[-1]
                save_name = mesh_class_id + "_" + mesh_shape_id
                path = os.path.join(experiment_directory, ws.tb_logs_dir, ws.tb_logs_train_reconstructions, save_name)
                if not os.path.exists(path):
                    os.makedirs(path)

                start = time.time()
                with torch.no_grad():
                    train_mesh = mesh.create_mesh(
                        decoder, 
                        lat_vec, 
                        N=eval_grid_res, 
                        max_batch=int(2 ** 18), 
                        filename=os.path.join(path, f"epoch={epoch}"),
                        return_trimesh=True,
                    )
                logging.debug("[Train eval] Total time to create training mesh: {}".format(time.time() - start))

                if train_mesh is not None:
                    gt_mesh_path = f"/mnt/hdd/ShapeNetCore.v2/{mesh_class_id}/{mesh_shape_id}/models/model_normalized.obj"
                    chamfer_dist_sum += metrics.compute_metric(gt_mesh=gt_mesh_path, gen_mesh=train_mesh, metric="chamfer")
                
                del train_mesh, mesh_class_id, mesh_shape_id, save_name, gt_mesh_path

            logging.debug(f"Chamfer distance: {chamfer_dist_sum}.")            
            summary_writer.add_scalar("Mean Chamfer Dist/train", chamfer_dist_sum/len(eval_train_scene_idxs), epoch)
            # End of eval train.
        
        if epoch % eval_test_frequency == 0:
            # Test-set evaluation: Reconstruct latent and mesh from GT sdf values and compute metrics.
            eval_test_time_start = time.time()
            test_err_sum = 0
            chamfer_dist_sum = 0
            for test_fname in eval_test_filenames:
                mesh_class_id = test_fname.split(".npz")[0].split(os.sep)[-2]
                mesh_shape_id = test_fname.split(".npz")[0].split(os.sep)[-1]
                save_name = mesh_class_id + "_" + mesh_shape_id
                path = os.path.join(experiment_directory, ws.tb_logs_dir, ws.tb_logs_test_reconstructions, save_name)
                if not os.path.exists(path):
                    os.makedirs(path)
                test_fpath = os.path.join(data_source, ws.sdf_samples_subdir, test_fname)
                test_sdf_samples = deep_sdf.data.read_sdf_samples_into_ram(test_fpath)
                test_sdf_samples[0] = test_sdf_samples[0][torch.randperm(test_sdf_samples[0].shape[0])]
                test_sdf_samples[1] = test_sdf_samples[1][torch.randperm(test_sdf_samples[1].shape[0])]

                start = time.time()
                test_err, test_latent = reconstruct.reconstruct(
                    decoder,
                    int(2000),
                    latent_size,
                    test_sdf_samples,
                    0.01,  # [emp_mean,emp_var],
                    0.1,
                    num_samples=8000,
                    lr=5e-3,
                    l2reg=True,
                )
                logging.debug("[Test eval] Total reconstruction time: {}".format(time.time() - start))
                test_err_sum += test_err

                start = time.time()
                with torch.no_grad():
                    test_mesh = mesh.create_mesh(
                        decoder, 
                        test_latent, 
                        N=eval_grid_res, 
                        max_batch=int(2 ** 18), 
                        filename=os.path.join(path, f"epoch={epoch}"),
                        return_trimesh=True,
                    )
                logging.debug("[Test eval] Total time to create test mesh: {}".format(time.time() - start))

                if test_mesh is not None:
                    gt_mesh_path = f"/mnt/hdd/ShapeNetCore.v2/{mesh_class_id}/{mesh_shape_id}/models/model_normalized.obj"
                    chamfer_dist_sum += metrics.compute_metric(gt_mesh=gt_mesh_path, gen_mesh=test_mesh, metric="chamfer")
            
                del test_sdf_samples
                del test_latent
                del test_mesh

            logging.debug(f"Test Chamfer distance: {chamfer_dist_sum}.")
            summary_writer.add_scalar("Mean Chamfer Dist/test", chamfer_dist_sum/len(eval_test_filenames), epoch)
            summary_writer.add_scalar("Loss/test", test_err_sum/len(eval_test_filenames), epoch)
            summary_writer.add_scalar("Mean Reconstr Time (sec)", (time.time()-eval_test_time_start)/len(eval_test_filenames), epoch)
            # End of eval test.

        summary_writer.flush()    
        # End of epoch.

    # Log hparams to TensorBoard.
    writer_hparams = {k: (torch.tensor(v) if type(v) == list else v) for k, v in specs["NetworkSpecs"].items()}
    summary_writer.add_hparams(writer_hparams, {"BestTrainLoss":min(loss_log)}, run_name=str(specs["Description"]))
    summary_writer.flush()    
    summary_writer.close()
    # End of training.

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
