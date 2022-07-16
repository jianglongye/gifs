import math
import os
import time
from glob import glob
from itertools import permutations

import numba
import numpy as np
import torch
import tqdm

import models.lookup as lookup


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


def extract_udfs_in_udf_combs(udf_combs):
    udfs = []
    for idx in idx_in_combs:
        udfs.append(udf_combs[idx[0], idx[1]])
    return udfs


inc = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])

combs = []
comb_to_idx = [0] * 64
dist = [0] * 64
for i in range(7):
    for j in range(i + 1, 8):
        comb_to_idx[i * 8 + j] = len(combs)
        dist[i * 8 + j] = np.linalg.norm(inc[i] - inc[j])
        combs.append([i, j])

possible_assignments = []
for pos_num in range(0, 9):
    assignments = set(permutations([0] * (8 - pos_num) + [1] * pos_num))
    possible_assignments.extend([list(x) for x in assignments])

new_possible_assignments = []
for assignment in possible_assignments:
    if [1 - x for x in assignment] not in new_possible_assignments:
        new_possible_assignments.append(assignment)

possible_assignments = np.ascontiguousarray(np.array(new_possible_assignments))

idx_in_combs = [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]


def vertex_interpolate(p1, p2, v1, v2, isovalue):
    if np.any(p1 > p2):
        p1, p2, v1, v2 = p2, p1, v2, v1
    p = p1
    if np.abs(v1 - v2) > 1e-5:
        p = p1 + (p2 - p1) * (isovalue - v1) / (v2 - v1)
    return p


def get_grid_centers(res=20, size=1.1, selected_indices=None):
    # centers range from [-size/2 + step_size/2, -size/2 + step_size/2, -size/2 + step_size/2]
    # to [size/2 - step_size/2, size/2 - step_size/2, size/2 - step_size/2]

    step_size = size / res

    mgrid = np.mgrid[:res, :res, :res]
    mgrid = np.moveaxis(mgrid, 0, -1)
    if selected_indices is None:
        mgrid.reshape(-1, 3)
    else:
        mgrid = mgrid[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]

    mgrid = mgrid / res * size - (size / 2 - step_size / 2)

    return mgrid


@numba.jit(nopython=True, fastmath=True)
def cal_loss(assignment, comb_values, udf, step_size):
    # inc = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    comb_to_idx = [0] * 64
    combs = []
    # dist = [0] * 64
    for i in range(7):
        for j in range(i + 1, 8):
            comb_to_idx[i * 8 + j] = len(combs)
            # dist[i * 8 + j] = np.sqrt(np.sum((inc[i] - inc[j]) ** 2))
            combs.append([i, j])

    neg_idxes = np.where(assignment < 0.5)[0]
    pos_idxes = np.where(assignment > 0.5)[0]

    gifs_loss = 0.0
    # udf_loss = 0.0
    for i in range(len(neg_idxes)):
        neg_idx = neg_idxes[i]
        for j in range(i + 1, len(neg_idxes)):
            gifs_loss = (
                gifs_loss + comb_values[comb_to_idx[min(neg_idx, neg_idxes[j]) * 8 + max(neg_idx, neg_idxes[j])], 0]
            )
        for pos_idx in pos_idxes:
            gifs_loss = gifs_loss + 1 - comb_values[comb_to_idx[min(neg_idx, pos_idx) * 8 + max(neg_idx, pos_idx)], 0]
            # udf_item = udf[comb_to_idx[min(neg_idx, pos_idx) * 8 + max(neg_idx, pos_idx)]]
            # max_udf_diff = step_size / 2 * dist[comb_to_idx[min(neg_idx, pos_idx) * 8 + max(neg_idx, pos_idx)]]
            # udf_loss = udf_loss + max([np.abs(udf_item[0]) + np.abs(udf_item[1]), max_udf_diff]) - max_udf_diff

    for i in range(len(pos_idxes)):
        pos_idx = pos_idxes[i]
        for j in range(i + 1, len(pos_idxes)):
            gifs_loss += comb_values[comb_to_idx[min(pos_idx, pos_idxes[j]) * 8 + max(pos_idx, pos_idxes[j])], 0]

    return gifs_loss


@numba.jit(nopython=True, fastmath=True)
def combs_to_verts_glb_opt(comb_values, udf, step_size, possible_assignments):
    idx_in_combs = [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]

    min_loss = np.inf
    min_idx = 0

    if np.max(comb_values) > 0.5:
        for assign_idx in range(128):
            assignment = possible_assignments[assign_idx]
            loss = cal_loss(assignment, comb_values, udf, step_size)
            if loss < min_loss:
                min_idx = assign_idx
                min_loss = loss

    result = np.ones(8) * -1
    result[possible_assignments[min_idx] > 0] = 1

    vert_udf = []
    for idx in idx_in_combs:
        vert_udf.append(udf[idx[0], idx[1]])
    vert_udf = np.array(vert_udf)

    result = result * vert_udf
    return result


# TODO: pre-cache a combs_to_verts lookup table
def combs_to_verts(comb_values, udf=None):
    # comb_values.shape: 24
    max_comb_values = comb_values.max()
    if max_comb_values > 0.5:
        anchor_vert0, anchor_vert1 = combs[np.argmax(comb_values)]

        verts_class0 = [anchor_vert0]
        verts_class1 = [anchor_vert1]

        for temp_vert in range(8):
            if temp_vert == anchor_vert0 or temp_vert == anchor_vert1:
                continue
            temp_comb_value0 = comb_values[comb_to_idx[min(temp_vert, anchor_vert0) * 8 + max(temp_vert, anchor_vert0)]]
            temp_comb_value1 = comb_values[comb_to_idx[min(temp_vert, anchor_vert1) * 8 + max(temp_vert, anchor_vert1)]]
            if temp_comb_value0 > temp_comb_value1:
                verts_class1.append(temp_vert)
            else:
                verts_class0.append(temp_vert)

        if udf is None:
            result = np.zeros(8)
            for temp_vert in verts_class1:
                result[temp_vert] = 1
            return result
        else:
            result = np.ones(8) * -1
            for temp_vert in verts_class1:
                result[temp_vert] = 1
            vert_udf = np.array(extract_udfs_in_udf_combs(udf))
            result = result * vert_udf
            return result
    else:
        return np.zeros(8)


def contrastive_marching_cubes(comb_values, isovalue=0.5, res=100, size=2.4, selected_indices=None, udf=None):
    vs = {}
    fs = []
    mgrid = np.mgrid[: (res + 1), : (res + 1), : (res + 1)]
    mgrid = mgrid / res * size - size / 2
    mgrid = np.moveaxis(mgrid, 0, -1)

    if selected_indices is None:
        if udf is not None:
            udf = udf.reshape(res, res, res, len(combs), 2)

        for step_x in range(res):
            for step_y in range(res):
                for step_z in range(res):
                    grid_inc = np.array([step_x, step_y, step_z]) + inc
                    grid_verts = mgrid[grid_inc[:, 0], grid_inc[:, 1], grid_inc[:, 2]]

                    temp_comb_values = comb_values[step_x, step_y, step_z]
                    if udf is not None:
                        temp_udf = udf[step_x, step_y, step_z]

                    if udf is None:
                        vert_values = combs_to_verts(temp_comb_values)
                    else:
                        vert_values = combs_to_verts(temp_comb_values, udf=temp_udf)

                    pow2 = 2 ** np.arange(8)
                    inside = (vert_values < isovalue).astype(np.int)
                    top_id = np.sum(inside * pow2)

                    edges = lookup.EDGE_TABLE[top_id]
                    if edges == 0:
                        continue

                    quick_lookup_key = np.packbits((temp_comb_values < isovalue)).tostring()

                    edge_cut = np.zeros((12, 3))
                    for i in range(12):
                        if edges & (1 << i):
                            p1, p2 = lookup.EDGE_VERTEX[i]
                            edge_cut[i] = vertex_interpolate(
                                grid_verts[p1], grid_verts[p2], vert_values[p1], vert_values[p2], isovalue
                            )

                    tri_edges = lookup.TRI_TABLE[top_id] + [-1, -1]
                    tri_edges = [tri_edges[3 * i : 3 * i + 3] for i in range(len(tri_edges) // 3)]
                    triangles = [edge_cut[e] for e in tri_edges if e[0] >= 0]
                    triangles = np.stack(triangles)

                    for t in triangles:
                        vid_list = []
                        for v in t:
                            v = tuple(v)
                            if v not in vs:
                                vs[v] = len(vs) + 1
                            vid_list.append(vs[v])
                        fs.append(vid_list)
    else:
        comb_values = comb_values.reshape(selected_indices.shape[0], len(combs), -1)
        udf = udf.reshape(selected_indices.shape[0], len(combs), -1)

        step_size = size / res
        mask = udf.min(axis=1).min(axis=1) < step_size
        selected_indices = selected_indices[mask]
        udf = udf[mask]
        comb_values = comb_values[mask]

        if udf is None:
            zip_datas = zip(comb_values, selected_indices)
        else:
            zip_datas = zip(comb_values, selected_indices, udf)
        for zip_data in tqdm.tqdm(zip_datas, total=len(comb_values)):
            if udf is None:
                temp_comb_values, selected_index = zip_data
            else:
                temp_comb_values, selected_index, temp_udf = zip_data
            grid_inc = selected_index + inc
            grid_verts = mgrid[grid_inc[:, 0], grid_inc[:, 1], grid_inc[:, 2]]

            if udf is None:
                vert_values = combs_to_verts(temp_comb_values)
            else:
                # vert_values = combs_to_verts(temp_comb_values, temp_udf)
                vert_values = combs_to_verts_glb_opt(temp_comb_values, temp_udf, size / res, possible_assignments)

            pow2 = 2 ** np.arange(8)
            inside = (vert_values < isovalue).astype(np.int)
            top_id = np.sum(inside * pow2)

            edges = lookup.EDGE_TABLE[top_id]
            if edges == 0:
                continue

            edge_cut = np.zeros((12, 3))
            for i in range(12):
                if edges & (1 << i):
                    p1, p2 = lookup.EDGE_VERTEX[i]
                    edge_cut[i] = vertex_interpolate(
                        grid_verts[p1], grid_verts[p2], vert_values[p1], vert_values[p2], isovalue
                    )

            tri_edges = lookup.TRI_TABLE[top_id] + [-1, -1]
            tri_edges = [tri_edges[3 * i : 3 * i + 3] for i in range(len(tri_edges) // 3)]
            triangles = [edge_cut[e] for e in tri_edges if e[0] >= 0]
            triangles = np.stack(triangles)

            for t in triangles:
                vid_list = []
                for v in t:
                    v = tuple(v)
                    if v not in vs:
                        vs[v] = len(vs) + 1
                    vid_list.append(vs[v])
                fs.append(vid_list)
    return vs, fs


def get_grid_comb_verts(res=100, size=2.4, selected_indices=None):
    # vertices range from [-size/2, -size/2, -size/2] to [size/2, size/2, size/2]
    mgrid = np.mgrid[: (res + 1), : (res + 1), : (res + 1)]
    mgrid = mgrid / res * size - size / 2
    mgrid = np.moveaxis(mgrid, 0, -1)

    grid_verts_inc = np.mgrid[:res, :res, :res]
    grid_verts_inc = np.moveaxis(grid_verts_inc, 0, -1)
    if selected_indices is None:
        grid_verts_inc = grid_verts_inc.reshape(-1, 3)
    else:
        grid_verts_inc = grid_verts_inc[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]

    comb_inc0 = []
    comb_inc1 = []
    for comb in combs:
        comb_inc0.append(inc[comb[0]])
        comb_inc1.append(inc[comb[1]])
    comb_inc0 = np.array(comb_inc0)
    comb_inc1 = np.array(comb_inc1)

    comb_verts_inc0 = np.repeat(grid_verts_inc[:, None, :], len(combs), axis=1) + comb_inc0[None]
    comb_verts_inc1 = np.repeat(grid_verts_inc[:, None, :], len(combs), axis=1) + comb_inc1[None]
    comb_verts0 = mgrid[comb_verts_inc0[:, :, 0], comb_verts_inc0[:, :, 1], comb_verts_inc0[:, :, 2]]
    comb_verts1 = mgrid[comb_verts_inc1[:, :, 0], comb_verts_inc1[:, :, 1], comb_verts_inc1[:, :, 2]]

    if selected_indices is None:
        comb_grid_verts = (
            np.concatenate([comb_verts0, comb_verts1], axis=-1).reshape(res, res, res, len(combs), 6).copy()
        )
    else:
        comb_grid_verts = np.concatenate([comb_verts0, comb_verts1], axis=-1).reshape(-1, len(combs), 6).copy()
    return comb_grid_verts


class GIFSGenerator(object):
    def __init__(self, model, exp_name, checkpoint=None, device=torch.device("cuda")):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.checkpoint_path = os.path.dirname(__file__) + "/../experiments/{}/checkpoints/".format(exp_name)
        self.load_checkpoint(checkpoint)

    def mesh_refine(self, vs, fs, inputs, refinement_step=30):
        v = torch.FloatTensor(vs).clone().cuda()
        v.requires_grad = True
        f = torch.LongTensor(fs).cuda()

        optimizer = torch.optim.RMSprop([v], lr=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        for it_r in range(refinement_step):
            optimizer.zero_grad()

            face_vertex = v[f]
            # Loss
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=fs.shape[0])
            eps = torch.FloatTensor(eps).cuda()
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            result_dict = self.model.get_udf(face_point.unsqueeze(0), inputs)
            face_value = result_dict["udf"].squeeze(0)

            loss_target = (face_value).pow(2).mean()

            loss = loss_target
            # print(100 * float(loss_target))

            # Update
            loss.backward()
            optimizer.step()
            scheduler.step()

        return v.detach().cpu().numpy(), fs

    # if use udf, remember to set isovalue to 0! Yes, remember it.
    def generate_mesh(self, data, init_res=15, size=2.4, refine_levels=3, thres=3.0, isovalue=0):

        start = time.time()
        inputs = data["inputs"].to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        with torch.cuda.amp.autocast():
            encoding = self.model.encoder(inputs)

        selected_indices = np.mgrid[: int(init_res / 2), : int(init_res / 2), : int(init_res / 2)]
        selected_indices = np.moveaxis(selected_indices, 0, -1).reshape(-1, 3)

        for level in range(refine_levels):
            print(f"level: {level}")
            res = init_res * (2 ** level)
            step_size = size / res
            selected_indices = (selected_indices[:, None] * 2 + inc[None]).reshape(-1, 3)

            grid_centers = get_grid_centers(res, size, selected_indices)
            grid_centers = torch.from_numpy(grid_centers).float().reshape(-1, 6)

            num_chunks = math.ceil(grid_centers.shape[0] / 10000)
            chunked_points = torch.chunk(grid_centers, chunks=num_chunks, dim=0)
            udfs = []
            with torch.no_grad():
                for p in chunked_points:
                    with torch.cuda.amp.autocast():
                        pred_dict = self.model.query(p.unsqueeze(0).cuda(), encoding)
                    udfs.append(pred_dict["udf"].squeeze(0))
            udf = torch.cat(udfs, dim=0).float()

            udf = udf.detach().cpu().numpy().reshape(-1)

            selected_indices = selected_indices[udf < step_size * thres]

        res = init_res * (2 ** (refine_levels))
        selected_indices = (selected_indices[:, None] * 2 + inc[None]).reshape(-1, 3)

        grid_comb_verts = get_grid_comb_verts(res, size, selected_indices).reshape(-1, 6)
        grid_comb_verts = torch.from_numpy(grid_comb_verts).float()

        num_chunks = math.ceil(grid_comb_verts.shape[0] / 10000)
        chunked_points = torch.chunk(grid_comb_verts, chunks=num_chunks, dim=0)
        preds = []
        udfs = []
        print("final level")
        with torch.no_grad():
            for p in chunked_points:
                with torch.cuda.amp.autocast():
                    pred_dict = self.model.query(p.unsqueeze(0).cuda(), encoding)
                preds.append(pred_dict["pred"].squeeze(0))
                udfs.append(pred_dict["udf"].squeeze(0))
        preds = torch.cat(preds, dim=0).float()
        udfs = torch.cat(udfs, dim=0).float()
        preds = preds.cpu().numpy()
        udfs = udfs.cpu().numpy()

        vs, fs = contrastive_marching_cubes(
            preds, res=res, size=size, selected_indices=selected_indices, udf=udfs, isovalue=isovalue
        )
        vs, fs = np.array(list(vs.keys())), np.array(fs) - 1

        duration = time.time() - start

        return vs, fs, duration

    def load_checkpoint(self, checkpoint):
        checkpoints = glob(self.checkpoint_path + "/*")
        if checkpoint is None:
            if len(checkpoints) == 0:
                print("No checkpoints found at {}".format(self.checkpoint_path))
                return 0, 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0].split("_")[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=float)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + "checkpoint_{}h:{}m:{}s_{}.tar".format(
                *[*convertSecs(checkpoints[-1]), checkpoints[-1]]
            )
        else:
            path = self.checkpoint_path + "{}.tar".format(checkpoint)
        print("Loaded checkpoint from: {}".format(path))
        checkpoint = torch.load(path)
        state_dict = {
            k.replace("module.", ""): checkpoint["model_state_dict"][k] for k in checkpoint["model_state_dict"]
        }
        self.model.load_state_dict(state_dict)
        epoch = checkpoint["epoch"]
        training_time = checkpoint["training_time"]
        return epoch, training_time
