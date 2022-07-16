from __future__ import division

import os
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset


class VoxelizedDataset(Dataset):
    def __init__(
        self,
        mode,
        res,
        pointcloud_samples,
        data_path,
        split_file,
        batch_size,
        num_sample_points,
        num_workers,
        sample_distribution,
        sample_sigmas,
    ):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.mode = mode
        self.data = self.split[mode]
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples

        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:

            path = self.data[idx]
            input_path = path
            samples_path = path

            voxel_path = input_path + "/voxelized_point_cloud_{}res_{}points.npz".format(
                self.res, self.pointcloud_samples
            )
            occupancies = np.unpackbits(np.load(voxel_path)["compressed_occupancies"])
            input = np.reshape(occupancies, (self.res,) * 3)

            if self.mode == "test":
                return {"inputs": np.array(input, dtype=np.float32), "path": path}

            boundary_samples_path = samples_path + "/labels_gifs.npz"
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_pairs = boundary_samples_npz["pairs"]
            boundary_sample_coords = boundary_samples_npz["grid_coords"]
            boundary_sample_df = boundary_samples_npz["df"]
            boundary_sample_labels = boundary_samples_npz["labels"]

            subsample_indices = np.random.randint(0, len(boundary_sample_pairs), self.num_sample_points)
            pairs = boundary_sample_pairs[subsample_indices]
            coords = boundary_sample_coords[subsample_indices]
            df = boundary_sample_df[subsample_indices]
            labels = boundary_sample_labels[subsample_indices]
            assert len(pairs) == self.num_sample_points
            assert len(labels) == self.num_sample_points

            assert len(df) == self.num_sample_points
            assert len(coords) == self.num_sample_points
        except:
            print("Error with {}: {}".format(path, traceback.format_exc()))
            raise

        return {
            "grid_coords": np.array(coords, dtype=np.float32),
            "df": np.array(df, dtype=np.float32),
            "pairs": np.array(pairs, dtype=np.float32),
            "inputs": np.array(input, dtype=np.float32),
            "labels": np.array(labels, dtype=np.float32),
            "path": path,
        }

    def get_loader(self, shuffle=True):

        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
        )

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
