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
# @Last Modified: 2021-12-27 04:15:41 pm
from collections import defaultdict
import numpy as np
import scipy.spatial
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from ResNetPPI.net import ResNet1D, ResNet2D
from ResNetPPI.utils import (identity_score,
                             load_pairwise_aln_from_a3m,
                             sample_pairwise_aln,
                             gen_ref_msa_from_pairwise_aln,
                             get_random_crop_idx)


# SETTINGS
ONEHOT_DIM = 22
ENCODE_DIM = 44 # 46 if add hydrophobic features
CROP_SIZE = 128 # 64 if CUDA run out of memory
ONEHOT = np.eye(ONEHOT_DIM, dtype=np.float32)


# FUNCTIONS
# def onehot_encoding(aln):
#     ONEHOT = torch.eye(ONEHOT_DIM, dtype=torch.float)
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
    return iden_eff_weights


class ResNetPPI(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet1d = ResNet1D(ENCODE_DIM, [8])
        self.resnet2d = ResNet2D(4224, [(1,2,4,8)]*18)
        self.conv2d_37 = nn.Conv2d(96, 37, kernel_size=3, padding=1)
        # self.conv2d_41 = nn.Conv2d(96, 41, kernel_size=3, padding=1)
        self.softmax_func = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def onehot_encoding(self, aln: np.ndarray):
        encoding = ONEHOT[aln].transpose((0, 2, 1))
        encoding = encoding.reshape(-1, encoding.shape[-1])
        return torch.from_numpy(encoding).float()

    def pw_encoding(self, aln: np.ndarray):
        return self.onehot_encoding(aln).reshape(ENCODE_DIM, -1)

    '''
    def gen_pw_embedding_1by1(self, pw_msa):
        ref_length = (pw_msa[0][0] != 20).sum()
        for pw_aln in pw_msa:
            # $1 \times C \times L_k$
            msa_embedding = self.resnet1d(self.pw_encoding(pw_aln).unsqueeze(0))
            if pw_aln.shape[1] != ref_length:
                yield msa_embedding[:, :, pw_aln[0] != 20]  # NOTE: maybe a point to optimize (CPU <-> GPU)
            else:
                yield msa_embedding
    '''

    def gen_pw_embedding_group(self, pw_msa, iden_eff_weights_idx):
        # NOTE: the order of homologous sequences would change
        ref_length = (pw_msa[0][0] != 20).sum()
        group_pw_msa = defaultdict(list)
        for pw_idx, pw_aln in enumerate(pw_msa):
            group_pw_msa[pw_aln.shape[1]].append(pw_idx)
        for l_k, group in group_pw_msa.items():
            # $g \times C \times L_k$
            pw_encodings = torch.zeros((len(group), ENCODE_DIM, l_k), dtype=torch.float)
            for pw_idx_idx, pw_idx in enumerate(group):
                pw_encodings[pw_idx_idx] = self.pw_encoding(pw_msa[pw_idx])
                iden_eff_weights_idx.append(pw_idx)
            msa_embedding = self.resnet1d(pw_encodings)
            if msa_embedding.shape[2] == ref_length:
                yield msa_embedding
            else:
                for pw_idx_idx, pw_idx in enumerate(group):
                    yield msa_embedding[[pw_idx_idx]][:, :, pw_msa[pw_idx][0] != 20]  # TODO: optimize
    
    def msa_embedding(self, pw_msa, iden_eff_weights_idx):
        return torch.cat(tuple(self.gen_pw_embedding_group(pw_msa, iden_eff_weights_idx)))

    def gen_coevolution_aggregator(self, iden_eff_weights, msa_embeddings, crop_d: bool):
        cur_k, _, ref_length = msa_embeddings.shape
        """
        if cur_k > max_k:
            use_indices = torch.randperm(cur_k)[:max_k]
            iden_eff_weights = iden_eff_weights[use_indices]
            msa_embeddings = msa_embeddings[use_indices]
            cur_k = max_k
        """
        msa_embeddings = msa_embeddings.transpose(0, 1)
        # Weights: $1 \times K$
        m_eff = iden_eff_weights.sum()
        # One-Body Term: $C \times L$ -> $C \times 1$
        ## $(1 \times K) \times (C \times K \times L)$ -> $C \times L$
        ### one_body_term = torch.matmul(iden_eff_weights, msa_embeddings).squeeze(1).transpose(0, 1)/m_eff
        one_body_term = torch.einsum('ckl,k->lc', msa_embeddings, iden_eff_weights)/m_eff
        msa_embeddings = msa_embeddings.transpose(0, 2)  # $L \times K \times C$
        # Handle Cropping
        cropping_info = {}
        if crop_d and (ref_length > CROP_SIZE):
            crop_idx_x, crop_idx_y = get_random_crop_idx(ref_length, CROP_SIZE)
            if crop_idx_x <= (ref_length - CROP_SIZE - 1):
                idx_i_range = range(crop_idx_x, crop_idx_x+CROP_SIZE)
                cropping_info['idx_i_range'] = (crop_idx_x, crop_idx_x+CROP_SIZE)
            else:
                idx_i_range = range(crop_idx_x-CROP_SIZE, crop_idx_x)
                cropping_info['idx_i_range'] = (crop_idx_x-CROP_SIZE, crop_idx_x)
            if crop_idx_y <= (ref_length - CROP_SIZE - 1):
                idx_j_range = range(crop_idx_y, crop_idx_y+CROP_SIZE)
                cropping_info['idx_j_range'] = (crop_idx_y, crop_idx_y+CROP_SIZE)
            else:
                idx_j_range = range(crop_idx_y-CROP_SIZE, crop_idx_y)
                cropping_info['idx_j_range'] = (crop_idx_y-CROP_SIZE, crop_idx_y)
            ref_length = CROP_SIZE
        else:
            idx_i_range = range(ref_length)
            idx_j_range = range(ref_length)
        # Aggregating
        coevo_couplings = torch.zeros((1, ref_length, ref_length, 4224), dtype=torch.float)
        ij_record = {}
        for use_idx_i, idx_i in enumerate(idx_i_range):
            f_i = one_body_term[idx_i]
            x_k_i = msa_embeddings[idx_i]
            for use_idx_j, idx_j in enumerate(idx_j_range):
                if idx_i == idx_j:
                    continue
                if (idx_j, idx_i) in ij_record:
                    coevo_couplings[0, use_idx_i, use_idx_j] = coevo_couplings[ij_record[idx_j, idx_i]]
                    continue
                f_j = one_body_term[idx_j]
                x_k_j = msa_embeddings[idx_j]
                # Two-Body Term: $C \times C$
                ## $(K \times C) \otimes (K \times C)$ -> $K \times C \times C$
                x_k_ij = torch.einsum('ki,kj->ikj', x_k_i, x_k_j)
                ## $(1 \times K) \times (C \times K \times C)$ -> $C \times C$
                ### two_body_term_ij = torch.einsum('ikj,k->ji', x_k_ij, iden_eff_weights).transpose(0, 1)/m_eff
                ### two_body_term_ij = torch.matmul(iden_eff_weights, x_k_ij)/m_eff
                two_body_term_ij = (iden_eff_weights @ x_k_ij) / m_eff
                """
                two_body_term_ij = torch.zeros((64, 64), dtype=torch.float)
                for k_idx in range(cur_k):
                    two_body_term_ij += (iden_eff_weights[k_idx] * torch.outer(x_k_i[k_idx], x_k_j[k_idx]))
                two_body_term_ij /= m_eff
                """
                ## $C + C + C^2$
                coevo_couplings[0, use_idx_i, use_idx_j, :64] = f_i
                coevo_couplings[0, use_idx_i, use_idx_j, 64:128] = f_j
                coevo_couplings[0, use_idx_i, use_idx_j, 128:] = two_body_term_ij.flatten()
                ij_record[idx_i, idx_j] = (0, use_idx_i, use_idx_j)
        return coevo_couplings, cropping_info

    def forward_single_protein(self, pw_msa, crop_d: bool):
        iden_eff_weights = torch.from_numpy(get_eff_weights(pw_msa)[1:]).float()
        # MSA Embeddings: $K \times C \times L$
        iden_eff_weights_idx = []
        msa_embeddings = self.msa_embedding(pw_msa, iden_eff_weights_idx)
        iden_eff_weights = iden_eff_weights[iden_eff_weights_idx]
        # TODO: optimization for symmetric tensors
        coevo_couplings, cropping_info = self.gen_coevolution_aggregator(iden_eff_weights, msa_embeddings, crop_d)
        r2s = self.resnet2d(coevo_couplings.transpose(-1, -3))
        return self.softmax_func(self.conv2d_37(r2s)), cropping_info

    def loss_single_protein(self, cropping_info, l_idx, pred, target):
        # TODO: optimize double index
        if len(cropping_info) == 0:
            if l_idx.shape[0] != pred.shape[2]:
                pred = pred[:, :, l_idx, :][:, :, :, l_idx]
        else:
            idx_i_beg, idx_i_end = cropping_info['idx_i_range']
            idx_j_beg, idx_j_end = cropping_info['idx_j_range']
            mask_i = (l_idx >= idx_i_beg) & (l_idx < idx_i_end)
            mask_j = (l_idx >= idx_j_beg) & (l_idx < idx_j_end)
            l_idx_i = l_idx[mask_i] - idx_i_beg
            l_idx_j = l_idx[mask_j] - idx_j_beg
            pred = pred[:, :, l_idx_i, :][:, :, :, l_idx_j]
            t_l_idx = torch.arange(l_idx.shape[0], dtype=torch.int64)
            t_l_idx_i = t_l_idx[mask_i]
            t_l_idx_j = t_l_idx[mask_j]
            target = target[:, t_l_idx_i, :][:, :, t_l_idx_j]
        assert pred.shape[-1] > 0 and pred.shape[-2] > 0
        return self.loss_func(pred, target)

    def training_step(self, train_batch, batch_idx):
        msa_file, label_dist6d_1 = train_batch
        loading_a3m = load_pairwise_aln_from_a3m(msa_file)
        ref_seq_info = next(loading_a3m)
        pw_msa = sample_pairwise_aln(tuple(loading_a3m))
        pred_dist6d_1, cropping_info = self.forward_single_protein(pw_msa, True)
        l_idx = torch.from_numpy(ref_seq_info['obs_mask'])
        loss = self.loss_single_protein(cropping_info, l_idx, pred_dist6d_1, label_dist6d_1)
        #self.log('train_loss', loss)
        return loss
        

    def validation_step(self, val_batch, batch_idx):
        msa_file, label_dist6d_1 = val_batch
        loading_a3m = load_pairwise_aln_from_a3m(msa_file)
        ref_seq_info = next(loading_a3m)
        pw_msa = sample_pairwise_aln(tuple(loading_a3m))
        pred_dist6d_1, cropping_info = self.forward_single_protein(pw_msa, True)
        l_idx = torch.from_numpy(ref_seq_info['obs_mask'])
        loss = self.loss_single_protein(cropping_info, l_idx, pred_dist6d_1, label_dist6d_1)
        self.log('val_loss', loss)

    def forward(self, msa_file):
        # NOTE: for prediction/inference actions
        loading_a3m = load_pairwise_aln_from_a3m(msa_file)
        next(loading_a3m)
        pw_msa = tuple(loading_a3m)
        return self.forward_single_protein(pw_msa, False)[0]

