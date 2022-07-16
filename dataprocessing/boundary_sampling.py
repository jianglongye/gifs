import gc
import os

import igl
import numpy as np
import trimesh

# number of distance field samples generated per object
sample_num = 100000


def generate_labels(path):
    def points2grid_coords(points):
        grid_coords = points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = points[:, 2], points[:, 0]
        grid_coords = 2 * grid_coords
        return grid_coords

    def downsample_data(data, sample_num):
        if data.shape[0] >= sample_num:
            indices = np.random.choice(data.shape[0], sample_num, replace=False)
        else:
            print(f"WARNING: data.shape[0] ({data.shape[0]}) < sample_num ({sample_num})")
            indices = np.random.choice(data.shape[0], sample_num, replace=True)
        return data[indices]

    def balance_data(random_pairs, labels, sample_points_num):
        assert random_pairs.shape[0] == labels.shape[0]
        pos_random_pairs = random_pairs[labels > 0]
        neg_random_pairs = random_pairs[labels <= 0]

        pos_random_pairs = downsample_data(pos_random_pairs, int(sample_points_num / 2))
        neg_random_pairs = downsample_data(neg_random_pairs, int(sample_points_num / 2))
        downsample_pairs = np.concatenate([pos_random_pairs, neg_random_pairs], axis=0)
        downsample_labels = np.concatenate(
            [np.ones([pos_random_pairs.shape[0], 1]), np.zeros([neg_random_pairs.shape[0], 1])], axis=0
        )

        return downsample_pairs, downsample_labels

    surface_sample_scales = [0.005, 0.01, 0.03]
    surface_sample_ratios = [0.5, 0.3, 0.1]  # sum: 0.9

    bbox_sample_scale, bbox_sample_ratio, bbox_padding = 0.07, 0.08, 0.15
    space_sample_scale, space_sample_ratio, space_size = 0.09, 0.02, 0.65

    cmd_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "intersection_detection", "build", "intersection"
    )
    cmd_tmpl = cmd_path + " {} {} {}"

    out_path = os.path.dirname(path)
    file_name = os.path.splitext(os.path.basename(path))[0]
    input_file = os.path.join(out_path, file_name + "_scaled.off")
    temp_sample_file = os.path.join(out_path, "temp_sample.bin")
    temp_label_file = os.path.join(out_path, "temp_label.bin")
    label_file = out_path + "/labels_gifs.npz"

    if os.path.exists(label_file):
        print("Exists: {}".format(label_file))
        # return

    mesh = trimesh.load(input_file)

    surface_pairs = []
    # sample near surface
    for sample_ratio, sample_scale in zip(surface_sample_ratios, surface_sample_scales):
        sample_points_num = int(sample_num * sample_ratio)
        surface_points = mesh.sample(sample_points_num * 4)
        random_pairs = np.tile(surface_points, (1, 2))
        assert random_pairs.shape[1] == 6  # shape: N x 6
        random_pairs = random_pairs + np.random.randn(*random_pairs.shape) * sample_scale
        surface_pairs.append(random_pairs)
    surface_pairs = np.concatenate(surface_pairs, axis=0)

    # sample in bbox
    bbox_points_num = int(sample_num * bbox_sample_ratio)
    extents, transform = trimesh.bounds.to_extents(mesh.bounds)
    padding_extents = extents + bbox_padding
    bbox_points = trimesh.sample.volume_rectangular(padding_extents, bbox_points_num * 6, transform=transform)
    bbox_pairs = np.tile(bbox_points, (1, 2))
    bbox_pairs = bbox_pairs + np.random.randn(*bbox_pairs.shape) * bbox_sample_scale

    # sample in space
    space_points_num = int(sample_num * space_sample_ratio)
    space_points = (np.random.rand(int(space_points_num * 2), 3) * 2 - 1) * space_size
    space_pairs0 = np.tile(space_points, (1, 2))
    space_pairs0 = space_pairs0 + np.random.randn(*space_pairs0.shape) * space_sample_scale

    # sample points in bbox and space
    extents, transform = trimesh.bounds.to_extents(mesh.bounds)
    bbox_points = trimesh.sample.volume_rectangular(extents, space_points_num * 2, transform=transform)
    space_points = (np.random.rand(int(space_points_num * 2), 3) * 2 - 1) * space_size
    space_pairs1 = np.concatenate([bbox_points, space_points], axis=1)
    space_pairs = np.concatenate([space_pairs0, space_pairs1], axis=0)

    sample_pairs = np.concatenate([surface_pairs, bbox_pairs, space_pairs], axis=0)
    sample_pairs.astype(np.float32).tofile(temp_sample_file)

    # get labels
    command = cmd_tmpl.format(input_file, temp_sample_file, temp_label_file)
    return_code = os.system(command)
    if return_code != 0:
        print("Error: {}".format(command))
        os.remove(temp_sample_file)
        os.remove(temp_label_file)
        return
    labels = np.fromfile(temp_label_file, np.bool)
    os.remove(temp_sample_file)
    os.remove(temp_label_file)

    # balance data
    ds_surface_pairs, ds_surface_labels = [], []
    last_sample_points_num = 0
    for sample_ratio, sample_scale in zip(surface_sample_ratios, surface_sample_scales):
        sample_points_num = int(sample_num * sample_ratio)

        ds_pairs, ds_labels = balance_data(
            surface_pairs[last_sample_points_num : last_sample_points_num + sample_points_num * 4],
            labels[last_sample_points_num : last_sample_points_num + sample_points_num * 4],
            sample_points_num,
        )

        last_sample_points_num += sample_points_num * 4
        ds_surface_pairs.append(ds_pairs)
        ds_surface_labels.append(ds_labels)
    assert last_sample_points_num == surface_pairs.shape[0]
    ds_surface_pairs = np.concatenate(ds_surface_pairs, axis=0)
    ds_surface_labels = np.concatenate(ds_surface_labels, axis=0)

    ds_bbox_pairs, ds_bbox_labels = balance_data(
        bbox_pairs, labels[surface_pairs.shape[0] : bbox_pairs.shape[0] + surface_pairs.shape[0]], bbox_points_num
    )
    ds_space_pairs, ds_space_labels = balance_data(space_pairs, labels[-space_pairs.shape[0] :], space_points_num)

    ds_sample_pairs = np.concatenate([ds_surface_pairs, ds_bbox_pairs, ds_space_pairs], axis=0)
    ds_labels = np.concatenate([ds_surface_labels, ds_bbox_labels, ds_space_labels], axis=0)

    # get udf
    df0 = np.abs(igl.signed_distance(ds_sample_pairs[:, :3], mesh.vertices, mesh.faces)[0])
    df1 = np.abs(igl.signed_distance(ds_sample_pairs[:, 3:], mesh.vertices, mesh.faces)[0])
    df = np.concatenate([df0[:, None], df1[:, None]], axis=1)

    # get grid_coords
    grid_coords0 = points2grid_coords(ds_sample_pairs[:, :3])
    grid_coords1 = points2grid_coords(ds_sample_pairs[:, 3:])
    grid_coords = np.concatenate([grid_coords0, grid_coords1], axis=1)

    np.savez(label_file, pairs=ds_sample_pairs, df=df, grid_coords=grid_coords, labels=ds_labels)
    assert ds_sample_pairs.shape[0] == sample_num
    assert ds_labels.shape[0] == sample_num
    assert df.shape[0] == sample_num
    assert grid_coords.shape[0] == sample_num

    del mesh, df, ds_sample_pairs, grid_coords, ds_labels
    gc.collect()
