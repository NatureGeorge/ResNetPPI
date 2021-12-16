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
# @Last Modified: 2021-12-16 03:15:25 pm
import numpy as np
import scipy.spatial
import torch
from torch import nn
from ResNetPPI.net import ResNet1D, ResNet2D
from ResNetPPI.utils import identity_score, gen_ref_msa_from_pairwise_aln, load_pairwise_aln_from_a3m


# SETTINGS
ONEHOT_DIM = 22
ENCODE_DIM = 44 # 46 if add hydrophobic features
ONEHOT = np.eye(ONEHOT_DIM, dtype=np.float32)


# NETWORKS
resnet1d = ResNet1D(ENCODE_DIM, [8])


# FUNCTIONS
def onehot_encoding(aln: np.ndarray):
    encoding = ONEHOT[aln].transpose((0, 2, 1))
    encoding = encoding.reshape(-1, encoding.shape[-1])
    return torch.from_numpy(encoding)


# def onehot_encoding(aln):
#     ONEHOT = torch.eye(ONEHOT_DIM, dtype=torch.float32)
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


def msa_embedding(pw_msa):
    return torch.cat(tuple(gen_pw_embedding(pw_msa)))


def get_eff_weights(pw_msa):
    '''
    NOTE:
    following identity calculation include the reference sequence 
    and ignore those insertion regions of the homologous sequences
    '''
    ref_msa = gen_ref_msa_from_pairwise_aln(pw_msa)
    iden_score_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ref_msa, metric=identity_score))
    np.fill_diagonal(iden_score_mat, 1)
    iden_eff_weights = 1.0/(iden_score_mat >= 0.8).sum(axis=0)
    # M_eff = iden_eff_weights.sum()
    return iden_eff_weights.astype(np.float32)


class ResNetPPI: # (nn.Module)
    def forward(self, msa_file):
        pw_msa = tuple(load_pairwise_aln_from_a3m(msa_file))
        iden_eff_weights = torch.from_numpy(get_eff_weights(pw_msa)[1:])
        m_eff = iden_eff_weights.sum()
        iden_eff_weights = iden_eff_weights.reshape(1, iden_eff_weights.shape[0])
        msa_embeddings = msa_embedding(pw_msa)
        one_body_term = torch.matmul(iden_eff_weights, msa_embeddings.transpose(0,1))
        # $C \times L$
        one_body_term = one_body_term.reshape(one_body_term.shape[0], one_body_term.shape[2])/m_eff
        # $C \times C$
        # two_body_term
        # 
