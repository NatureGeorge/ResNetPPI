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
# @Last Modified: 2021-12-24 03:39:35 pm
import numpy as np
import scipy.spatial
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from ResNetPPI.net import ResNet1D, ResNet2D
from ResNetPPI.utils import (identity_score,
                             load_pairwise_aln_from_a3m,
                             gen_ref_msa_from_pairwise_aln)


# SETTINGS
ONEHOT_DIM = 22
ENCODE_DIM = 44 # 46 if add hydrophobic features
ONEHOT = np.eye(ONEHOT_DIM, dtype=np.float32)


# FUNCTIONS
# def onehot_encoding(aln):
#     ONEHOT = torch.eye(ONEHOT_DIM, dtype=torch.float32)
#     aln = torch.from_numpy(aln)
#     encoding = ONEHOT[aln.to(torch.int64)].transpose(-1, -2)
#     encoding = encoding.reshape(-1, encoding.shape[-1])
#     return encoding


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
    # m_eff = iden_eff_weights.sum()
    return iden_eff_weights.astype(np.float32)


class ResNetPPI(pl.LightningModule):
    def __init__(self, encode_dim: int = ENCODE_DIM):
        super().__init__()
        self.resnet1d = ResNet1D(encode_dim, [8])
        self.resnet2d = ResNet2D(4224, [(1,2,4,8)]*18)
        self.conv2d_37 = nn.Conv2d(96, 37, kernel_size=3, padding=1, bias=False)
        # self.conv2d_41 = nn.Conv2d(96, 41, kernel_size=3, padding=1, bias=False)
        self.softmax_func = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def onehot_encoding(self, aln: np.ndarray):
        encoding = ONEHOT[aln].transpose((0, 2, 1))
        encoding = encoding.reshape(-1, encoding.shape[-1])
        return torch.from_numpy(encoding)

    def pw_encoding(self, aln: np.ndarray):
        return self.onehot_encoding(aln).reshape(1, ENCODE_DIM, -1)

    def gen_pw_embedding(self, pw_msa):
        ref_length = (pw_msa[0][0] != 20).sum()
        for pw_aln in pw_msa:
            # $1 \times C \times L_k$
            msa_embedding = self.resnet1d(self.pw_encoding(pw_aln))  # TODO: optimize with torch.utils.data.DataLoader
            if pw_aln.shape[1] != ref_length:
                yield msa_embedding[:, :, pw_aln[0] != 20]  # NOTE: maybe a point to optimize (CPU <-> GPU)
            else:
                yield msa_embedding
    
    def msa_embedding(self, pw_msa):
        return torch.cat(tuple(self.gen_pw_embedding(pw_msa)))

    def gen_coevolution_aggregator(self, ref_length, iden_eff_weights, msa_embeddings):
        msa_embeddings = msa_embeddings.transpose(0, 1)
        # Weights: $1 \times K$
        m_eff = iden_eff_weights.sum()
        # One-Body Term: $C \times L$ -> $C \times 1$
        ## $(1 \times K) \times (C \times K \times L)$ -> $C \times L$
        ### one_body_term = torch.matmul(iden_eff_weights, msa_embeddings).squeeze(1).transpose(0, 1)/m_eff
        one_body_term = torch.einsum('ckl,k->lc', msa_embeddings, iden_eff_weights)/m_eff
        msa_embeddings = msa_embeddings.transpose(0, 2)  # $L \times K \times C$
        for idx_i in range(ref_length-1):
            f_i = one_body_term[idx_i]
            x_k_i = msa_embeddings[idx_i]
            for idx_j in range(idx_i+1, ref_length):
                f_j = one_body_term[idx_j]
                x_k_j = msa_embeddings[idx_j]
                # Two-Body Term: $C \times C$
                ## $(K \times C) \otimes (K \times C)$ -> $K \times C \times C$
                x_k_ij = torch.einsum('ki,kj->ikj', x_k_i, x_k_j)
                ## $(1 \times K) \times (C \times K \times C)$ -> $C \times C$
                ### two_body_term_ij = torch.einsum('ikj,k->ji', x_k_ij, iden_eff_weights).transpose(0, 1)/m_eff
                ### two_body_term_ij = torch.matmul(iden_eff_weights, x_k_ij)/m_eff
                two_body_term_ij = (iden_eff_weights @ x_k_ij) / m_eff
                del x_k_ij
                ## $C + C + C^2$
                yield (idx_i, idx_j), torch.cat((f_i, f_j, two_body_term_ij.flatten()))

    def forward_single_protein(self, pw_msa):
        iden_eff_weights = torch.from_numpy(get_eff_weights(pw_msa)[1:])
        # MSA Embeddings: $K \times C \times L$
        msa_embeddings = self.msa_embedding(pw_msa)
        ref_length = msa_embeddings.shape[2]
        coevo_agg = self.gen_coevolution_aggregator(ref_length, iden_eff_weights, msa_embeddings)
        coevo_couplings = torch.zeros((ref_length, ref_length, 4224), dtype=torch.float32)
        # TODO: optimization for symmetric tensors
        # coevo_couplings = torch.stack(tuple(coevo_agg), dim=0).transpose(-1, -2)
        for (idx_i, idx_j), coevo_cp in coevo_agg:  # tqdm(coevo_agg, total=ref_length*(ref_length-1)//2):
            coevo_couplings[idx_i, idx_j, :] = coevo_couplings[idx_j, idx_i, :] = coevo_cp
        r2s = self.resnet2d(coevo_couplings.transpose(-1, -3).unsqueeze(0))
        print(r2s.shape)
        return self.softmax_func(
            self.conv2d_37(
                0.5*(r2s + r2s.transpose(-1, -2))
            ))

    def loss_single_protein(self, l_idx, pred, target):
        if l_idx.shape[0] == pred.shape[2]:
            pass
        else:
            # TODO: optimization?
            pred = pred[:, :, l_idx, :][:, :, :, l_idx]
        return self.loss_func(pred, target)

    def training_step(self, train_batch, batch_idx):
        msa_file, label_dist6d_1 = train_batch
        loading_a3m = load_pairwise_aln_from_a3m(msa_file)
        ref_seq_info = next(loading_a3m)
        pw_msa = tuple(loading_a3m)
        pred_dist6d_1 = self.forward_single_protein(pw_msa)
        l_idx = torch.from_numpy(ref_seq_info['obs_mask'])
        loss = self.loss_single_protein(l_idx, pred_dist6d_1, label_dist6d_1)
        self.log('train_loss', loss)
        return loss
        

    def validation_step(self, val_batch, batch_idx):
        msa_file, label_dist6d_1 = val_batch
        loading_a3m = load_pairwise_aln_from_a3m(msa_file)
        ref_seq_info = next(loading_a3m)
        pw_msa = tuple(loading_a3m)
        pred_dist6d_1 = self.forward_single_protein(pw_msa)
        l_idx = torch.from_numpy(ref_seq_info['obs_mask'])
        loss = self.loss_single_protein(l_idx, pred_dist6d_1, label_dist6d_1)
        self.log('val_loss', loss)

    def forward(self, msa_file):
        # NOTE: for prediction/inference actions
        loading_a3m = load_pairwise_aln_from_a3m(msa_file)
        next(loading_a3m)
        pw_msa = tuple(loading_a3m)
        return self.forward_single_protein(pw_msa)

