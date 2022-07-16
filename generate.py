import os

import torch
import trimesh
from tqdm import tqdm

import configs.config_loader as cfg_loader
import models.data.voxelized_data_shapenet as voxelized_data
import models.local_model as model
from models.generation import GIFSGenerator

cfg = cfg_loader.get_config()

device = torch.device("cuda")

net = model.GIFS()

dataset = voxelized_data.VoxelizedDataset(
    "test",
    res=cfg.input_res,
    pointcloud_samples=cfg.num_points,
    data_path=cfg.data_dir,
    split_file=cfg.split_file,
    batch_size=1,
    num_sample_points=cfg.num_sample_points_generation,
    num_workers=30,
    sample_distribution=cfg.sample_ratio,
    sample_sigmas=cfg.sample_std_dev,
)

gen = GIFSGenerator(net, cfg.exp_name, device=device)

out_path = "experiments/{}/evaluation/".format(cfg.exp_name)


def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    for i, data in tqdm(enumerate(loader)):

        path = os.path.normpath(data["path"][0])
        export_path = out_path + "/generation/{}/{}/".format(path.split(os.sep)[-2], path.split(os.sep)[-1])

        if os.path.exists(export_path):
            if os.path.exists(export_path + "gifs_result.obj"):
                print("Path exists - skip! {}".format(export_path))
                continue
        else:
            os.makedirs(export_path)

        vs, fs, duration = gen.generate_mesh(data)
        mesh = trimesh.Trimesh(vs, fs)
        mesh.remove_degenerate_faces()
        mesh.export(export_path + "gifs_result.obj")

        print("duration", duration)


gen_iterator(out_path, dataset, gen)
