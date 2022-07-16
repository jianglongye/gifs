import os
import time
from glob import glob

import numpy as np
import torch
import torch.optim as optim
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    import wandb
except ImportError:
    pass

import configs.config_loader as cfg_loader
import models.data.voxelized_data_shapenet as voxelized_data
import models.local_model as model


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds


def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds


def setup(rank, world_size, port="10239"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def launch(main_fn, cfg, world_size):
    mp.spawn(main_fn, args=(world_size, cfg), nprocs=world_size, join=True)


def ddp_trainer(rank, world_size, cfg):
    setup(rank, world_size)

    net = model.GIFS()

    device = torch.device(rank)
    net = net.to(rank)
    net = DDP(net, device_ids=[rank])

    train_dataset = voxelized_data.VoxelizedDataset(
        "train",
        res=cfg.input_res,
        pointcloud_samples=cfg.num_points,
        data_path=cfg.data_dir,
        split_file=cfg.split_file,
        batch_size=cfg.batch_size,
        num_sample_points=cfg.num_sample_points_training,
        num_workers=30,
        sample_distribution=cfg.sample_ratio,
        sample_sigmas=cfg.sample_std_dev,
    )

    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler)

    val_dataset = voxelized_data.VoxelizedDataset(
        "val",
        res=cfg.input_res,
        pointcloud_samples=cfg.num_points,
        data_path=cfg.data_dir,
        split_file=cfg.split_file,
        batch_size=cfg.batch_size,
        num_sample_points=cfg.num_sample_points_training,
        num_workers=30,
        sample_distribution=cfg.sample_ratio,
        sample_sigmas=cfg.sample_std_dev,
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler)

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    if cfg.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters())
    if cfg.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), momentum=0.9)

    exp_path = os.path.dirname(__file__) + "/experiments/{}/".format(cfg.exp_name)
    checkpoint_path = exp_path + "checkpoints/".format(cfg.exp_name)
    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(checkpoint_path)
            os.makedirs(checkpoint_path)
    # if rank == 0:
    #     writer = SummaryWriter(exp_path + 'summary'.format(cfg.exp_name))
    if rank == 0 and cfg.use_wandb:
        wandb.init(project="general_shape", dir=exp_path, config=cfg, name=cfg.exp_name)

    val_min = 1e8
    max_dist = 0.1
    start, training_time = 0, 0

    # ===== load checkpoint =====
    checkpoints = glob(checkpoint_path + "/*")
    if len(checkpoints) == 0:
        if rank == 0:
            print("No checkpoints found at {}".format(checkpoint_path))
    else:
        checkpoints = [os.path.splitext(os.path.basename(path))[0].split("_")[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        path = checkpoint_path + "checkpoint_{}h:{}m:{}s_{}.tar".format(
            *[*convertSecs(checkpoints[-1]), checkpoints[-1]]
        )

        if rank == 0:
            print("Loaded checkpoint from: {}".format(path))
        checkpoint = torch.load(path, map_location=f"cuda:{rank}")
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start = checkpoint["epoch"]
        training_time = checkpoint["training_time"]
        val_min = checkpoint["val_min"]

    dist.barrier()

    # ===== train model =====
    loss = 0
    iteration_start_time = time.time()

    for epoch in range(start, cfg.num_epochs):
        sum_loss = 0
        if rank == 0:
            print("Start epoch {}".format(epoch))

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        for batch in train_data_loader:
            iteration_duration = time.time() - iteration_start_time

            # ==== save checkpoint and val ====
            save_ckpt_flag = iteration_duration > 60 * 60
            save_ckpt_flag_tensor = torch.tensor(save_ckpt_flag).int().to(device)
            dist.broadcast(save_ckpt_flag_tensor, src=0)
            if save_ckpt_flag_tensor.item() == 1:  # save model every X min and at start
                training_time += iteration_duration
                iteration_start_time = time.time()

                path = checkpoint_path + "checkpoint_{}h:{}m:{}s_{}.tar".format(
                    *[*convertSecs(training_time), training_time]
                )
                if rank == 0:
                    if not os.path.exists(path):
                        torch.save(
                            {  # 'state': torch.cuda.get_rng_state_all(),
                                "training_time": training_time,
                                "epoch": epoch,
                                "model_state_dict": net.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_min": val_min,
                            },
                            path,
                        )
                net.eval()

                sum_val_loss = 0
                num_batches = 125  # val_data_num = num_batches * batch_size
                val_data_iterator = val_data_loader.__iter__()
                for _ in range(num_batches):
                    try:
                        val_batch = val_data_iterator.next()
                    except:
                        val_data_iterator = val_data_loader.__iter__()
                        val_batch = val_data_iterator.next()

                    p = val_batch.get("grid_coords").to(device)
                    df_gt = val_batch.get("df").to(device)  # (Batch,num_points)
                    inputs = val_batch.get("inputs").to(device)

                    with torch.no_grad():
                        label_gt = val_batch.get("labels").squeeze(2).to(device)
                        with torch.cuda.amp.autocast():
                            pred_dict = net(p, inputs)

                        df_pred = pred_dict["udf"].float()
                        label_pred = pred_dict["pred"].float()

                        # out = (B,num_points) by componentwise comparing vecots of size num_samples:
                        loss_i = torch.nn.L1Loss()(torch.clamp(df_pred, max=max_dist), torch.clamp(df_gt, max=max_dist))
                        loss_gifs = torch.nn.L1Loss()(label_pred, label_gt)
                        loss = loss_i * 10 + loss_gifs

                    sum_val_loss += loss.item()

                val_loss = sum_val_loss / num_batches

                val_loss_tensor = torch.tensor(val_loss).to(device)
                dist.all_reduce(val_loss_tensor)
                val_loss = val_loss_tensor.item() / world_size

                if val_loss < val_min:
                    val_min = val_loss
                    if rank == 0:
                        for path in glob(exp_path + "val_min=*"):
                            os.remove(path)
                        np.save(
                            exp_path
                            + "val_min={}training_time={}h:{}m:{}s".format(*[epoch, *convertSecs(training_time)]),
                            [epoch, val_loss],
                        )

                if rank == 0 and cfg.use_wandb:
                    wandb.log({"val_loss": val_loss, "epoch": epoch})

                dist.barrier()

            # ==== optimize model ====
            net.train()
            optimizer.zero_grad()

            p = batch.get("grid_coords").to(device)
            df_gt = batch.get("df").to(device)  # (Batch,num_points)
            inputs = batch.get("inputs").to(device)

            label_gt = batch.get("labels").squeeze(2).to(device)
            pred_dict = net(p, inputs)

            df_pred = pred_dict["udf"]
            label_pred = pred_dict["pred"]

            loss_i = torch.nn.L1Loss()(torch.clamp(df_pred, max=max_dist), torch.clamp(df_gt, max=max_dist))
            loss_gifs = torch.nn.L1Loss()(label_pred, label_gt)

            loss = loss_i * 10 + loss_gifs

            loss.backward()
            optimizer.step()

            if rank == 0:
                print("Current loss: {}".format(loss / train_dataset.num_sample_points))
            sum_loss += loss

        if rank == 0 and cfg.use_wandb:
            wandb.log({"train_loss_last_batch": loss, "epoch": epoch})
            wandb.log({"train_loss_avg": sum_loss / len(train_data_loader), "epoch": epoch})

    cleanup()


if __name__ == "__main__":
    cfg = cfg_loader.get_config()

    n_gpus = torch.cuda.device_count()

    launch(ddp_trainer, cfg, n_gpus)
