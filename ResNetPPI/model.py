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

# @Created Date: 2021-12-16 12:19:15 am
# @Filename: model.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-16 11:41:08 am
import numpy as np
import torch
from ResNetPPI.net import ResNet1D, ResNet2D

ONEHOT_DIM = 22
ENCODE_DIM = 44 # 46 if add hydrophobic features
ONEHOT = np.eye(ONEHOT_DIM, dtype=np.float32)
# ONEHOT = torch.eye(ONEHOT_DIM, dtype=torch.float32)
resnet1d = ResNet1D(44, [8])

def onehot_encoding(aln: np.ndarray):
    encoding = ONEHOT[aln].transpose((0, 2, 1))
    encoding = encoding.reshape(-1, encoding.shape[-1])
    return torch.from_numpy(encoding)


# def onehot_encoding(aln):
#     aln = torch.from_numpy(aln)
#     encoding = ONEHOT[aln.to(torch.int64)].transpose(-1, -2)
#     encoding = encoding.reshape(-1, encoding.shape[-1])
#     return encoding


def pw_encoding(aln: np.ndarray):
    return onehot_encoding(aln).reshape(1, ENCODE_DIM, -1)


def gen_pw_embedding(pw_msa):
    length = (pw_msa[0][0] != 20).sum() # ref_msa.shape[1]
    for pw_aln in pw_msa:
        msa_embedding = resnet1d(pw_encoding(pw_aln))
        if pw_aln.shape[1] != length:
            yield msa_embedding[:, :, pw_aln[0] != 20]
        else:
            yield msa_embedding

