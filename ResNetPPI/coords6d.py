# Copyright 2021 Zefeng Zhu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @Created Date: 2021-12-14 06:53:34 pm
# @Filename: coords6d.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-26 05:39:11 pm
import numpy as np
import scipy
import scipy.spatial


def get_dist6d(Cb: np.ndarray, dmax: float = 20.0, dist_fill_value: float = 0.0, dtype=np.float32):
    nres = Cb.shape[0]

    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    mask_for_sym = idx0 > idx1
    idx0_sym = idx0[mask_for_sym]
    idx1_sym = idx1[mask_for_sym]

    dist6d = np.full((nres, nres), dist_fill_value, dtype=dtype)
    dist6d[idx1_sym, idx0_sym] = dist6d[idx0_sym, idx1_sym] = np.linalg.norm(Cb[idx1_sym]-Cb[idx0_sym], axis=-1)
    return idx, dist6d


def get_contact6d(dist6d: np.ndarray, idx: np.ndarray, dmax: float = 20.0, fill_diag: bool = False, boolean: bool = False):
    ret = np.zeros_like(dist6d)
    idx0 = idx[0]
    idx1 = idx[1]
    ret[idx0, idx1] = (dmax - dist6d[idx0, idx1])/dmax if (not boolean) else 1
    if fill_diag and dist6d.shape[0] == dist6d.shape[1]:
        np.fill_diagonal(ret, 1)
    return ret


def get_dist6d_2(Cb_1: np.ndarray, Cb_2: np.ndarray, dmax: float = 20.0, dist_fill_value: float = 0.0):
    dist6d_2 = scipy.spatial.distance.cdist(Cb_1, Cb_2, 'euclidean')
    mask = dist6d_2 > dmax
    dist6d_2[mask] = dist_fill_value
    return np.where(~mask), dist6d_2

