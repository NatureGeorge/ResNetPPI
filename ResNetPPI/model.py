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
# @Last Modified: 2021-12-29 03:02:15 pm
import torch
from torch import nn
import pytorch_lightning as pl
from ResNetPPI import ENCODE_DIM, CROP_SIZE
from ResNetPPI.net import ResNet1D, ResNet2D
from ResNetPPI.utils import get_random_crop_idx


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

    '''
    def gen_pw_embedding_1by1(self, pw_encodings):
        ref_length = (pw_encodings[0][20] != 1).sum()
        for pw_aln in pw_encodings:
            # $1 \times C \times L_k$
            msa_embedding = self.resnet1d(pw_aln.unsqueeze(0))
            if pw_aln.shape[1] != ref_length:
                yield msa_embedding[:, :, pw_aln[20] != 1]
            else:
                yield msa_embedding
    '''

    def gen_pw_embedding_group(self, pw_encodings_group):
        ref_length = (pw_encodings_group[0][0][20] != 1).sum()
        for group in pw_encodings_group:
            msa_embedding = self.resnet1d(group)
            cur_k, channel_size, l_k = msa_embedding.shape
            if l_k == ref_length:
                yield msa_embedding
            else:
                #for idx_k in range(msa_embedding.shape[0]):
                #    yield msa_embedding[[idx_k]][:, :, group[idx_k][20] != 1]
                mask = torch.zeros_like(msa_embedding, dtype=torch.bool)
                for idx in range(cur_k):
                    mask[idx] = (group[idx][20] != 1).unsqueeze(0).repeat(channel_size, 1)
                yield msa_embedding[mask].reshape(cur_k, channel_size, ref_length)
    
    def get_msa_embeddings(self, pw_encodings_group):
        return torch.cat(tuple(self.gen_pw_embedding_group(pw_encodings_group)))

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
        coevo_couplings = torch.zeros((1, ref_length, ref_length, 4224), dtype=torch.float, device=msa_embeddings.device)
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

    def forward_single_protein(self, pw_encodings_group, iden_eff_weights, crop_d: bool):
        # MSA Embeddings: $K \times C \times L$
        msa_embeddings = self.get_msa_embeddings(pw_encodings_group)
        # TODO: optimization for symmetric tensors
        coevo_couplings, cropping_info = self.gen_coevolution_aggregator(iden_eff_weights, msa_embeddings, crop_d)
        r2s = self.resnet2d(coevo_couplings.movedim(3, 1))
        return self.conv2d_37(r2s), cropping_info

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
        ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1, label_dist6d_1 = train_batch
        pred_dist6d_1, cropping_info_1 = self.forward_single_protein(pw_encodings_group_1, iden_eff_weights_1, True)
        l_idx_1 = ref_seq_info_1['obs_mask']
        loss_1 = self.loss_single_protein(cropping_info_1, l_idx_1, pred_dist6d_1, label_dist6d_1.unsqueeze(0))
        self.log('train_loss', loss_1, batch_size=1)
        return loss_1

    def validation_step(self, val_batch, batch_idx):
        ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1, label_dist6d_1 = val_batch
        pred_dist6d_1, cropping_info_1 = self.forward_single_protein(pw_encodings_group_1, iden_eff_weights_1, True)
        l_idx_1 = ref_seq_info_1['obs_mask']
        loss_1 = self.loss_single_protein(cropping_info_1, l_idx_1, pred_dist6d_1, label_dist6d_1.unsqueeze(0))
        self.log('val_loss', loss_1, batch_size=1)
        return loss_1

    def forward(self, inputs):
        # NOTE: for prediction/inference actions
        pw_encodings_group, iden_eff_weights = inputs
        pred = self.forward_single_protein(pw_encodings_group, iden_eff_weights, False)[0]
        return self.softmax_func(0.5*(pred + pred.transpose(-1, -2)))

