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
# @Last Modified: 2022-01-09 10:47:30 pm
import torch
from torch import nn
import pytorch_lightning as pl
from ResNetPPI import ENCODE_DIM, HYDRO_DIM, CROP_SIZE
from ResNetPPI.net import ResNet1D, ResNet2D
from ResNetPPI.utils import get_random_crop_idx


def handle_cropping(ref_length: int, crop_d: bool, obs_idx: list):
    '''
    Handle cropping for single protein
    '''
    cropping_info = {}
    if crop_d and (ref_length > CROP_SIZE):
        crop_idx_x, crop_idx_y = get_random_crop_idx(ref_length, CROP_SIZE, obs_idx)
        if crop_idx_x <= (ref_length - CROP_SIZE):
            idx_i_range = torch.arange(crop_idx_x, crop_idx_x+CROP_SIZE)
            cropping_info['idx_i_range'] = (crop_idx_x, crop_idx_x+CROP_SIZE)
        else:
            idx_i_range = torch.arange(crop_idx_x-CROP_SIZE+1, crop_idx_x+1)
            cropping_info['idx_i_range'] = (crop_idx_x-CROP_SIZE+1, crop_idx_x+1)
        if crop_idx_y <= (ref_length - CROP_SIZE):
            idx_j_range = torch.arange(crop_idx_y, crop_idx_y+CROP_SIZE)
            cropping_info['idx_j_range'] = (crop_idx_y, crop_idx_y+CROP_SIZE)
        else:
            idx_j_range = torch.arange(crop_idx_y-CROP_SIZE+1, crop_idx_y+1)
            cropping_info['idx_j_range'] = (crop_idx_y-CROP_SIZE+1, crop_idx_y+1)
    else:
        idx_i_range = idx_j_range = torch.arange(ref_length)
    meshgrid = torch.cat((
        torch.cartesian_prod(idx_i_range, idx_j_range),
        torch.cartesian_prod(torch.arange(idx_i_range.shape[0]),
                             torch.arange(idx_j_range.shape[0]))),
        dim=1)
    meshgrid = meshgrid[meshgrid[:, 0]!=meshgrid[:, 1]]
    ij_record = {}
    ij_record_idx = []
    ij_record_mask = []
    for idx_idx, (idx_i, idx_j) in enumerate(meshgrid[:,:2]):
        idx_i = idx_i.item()
        idx_j = idx_j.item()
        if (idx_j, idx_i) in ij_record:
            ij_record_idx.append(ij_record[idx_j, idx_i])
            ij_record_mask.append(False)
        else:
            # ij_record_idx.append(idx_idx)
            ij_record_mask.append(True)
            ij_record[idx_i, idx_j] = idx_idx
    return cropping_info, idx_i_range.shape[0], meshgrid, torch.as_tensor(ij_record_idx), torch.as_tensor(ij_record_mask)


def gen_coevolution_aggregator(iden_eff_weights, msa_embeddings, cur_length: int, meshgrid, record_idx, record_mask):
    msa_embeddings = msa_embeddings.transpose(0, 1)
    # Weights: $1 \times K$
    m_eff = iden_eff_weights.sum()
    # One-Body Term: $C \times L$ -> $C \times 1$
    ## $(1 \times K) \times (C \times K \times L)$ -> $C \times L$
    ### one_body_term = torch.matmul(iden_eff_weights, msa_embeddings).squeeze(1).transpose(0, 1)/m_eff
    one_body_term = torch.einsum('ckl,k->lc', msa_embeddings, iden_eff_weights)/m_eff
    msa_embeddings = msa_embeddings.transpose(0, 2)  # $L \times K \times C$
    # Aggregating
    coevo_couplings = torch.zeros((1, cur_length, cur_length, 4224), dtype=msa_embeddings.dtype, device=msa_embeddings.device)
    cur_meshgrid = meshgrid[record_mask]
    for idx_idx in range(cur_meshgrid.shape[0]):
        cur_meshgrid_idx = cur_meshgrid[idx_idx]
        idx_i = cur_meshgrid_idx[0]
        idx_j = cur_meshgrid_idx[1]
        use_idx_i = cur_meshgrid_idx[2]
        use_idx_j = cur_meshgrid_idx[3]
        x_k_i = msa_embeddings[idx_i]
        x_k_j = msa_embeddings[idx_j]
        # Two-Body Term: $C \times C$
        ## $(K \times C) \otimes (K \times C)$ -> $K \times C \times C$
        x_k_ij = torch.einsum('ki,kj->ikj', x_k_i, x_k_j/m_eff)
        ## $(1 \times K) \times (C \times K \times C)$ -> $C \times C$
        ### two_body_term_ij = torch.matmul(iden_eff_weights, x_k_ij)/m_eff
        two_body_term_ij = iden_eff_weights @ x_k_ij
        ## $C + C + C^2$
        coevo_couplings[0, use_idx_i, use_idx_j, :64] = one_body_term[idx_i]
        coevo_couplings[0, use_idx_i, use_idx_j, 64:128] = one_body_term[idx_j]
        coevo_couplings[0, use_idx_i, use_idx_j, 128:] = two_body_term_ij.flatten()
    del x_k_ij, one_body_term, two_body_term_ij
    torch.cuda.empty_cache()
    cur_meshgrid = meshgrid[~record_mask]
    for idx_idx in range(cur_meshgrid.shape[0]):
        cur_meshgrid_idx = cur_meshgrid[idx_idx]
        use_idx_i = cur_meshgrid_idx[2]
        use_idx_j = cur_meshgrid_idx[3]
        meshgrid_idx = meshgrid[record_idx[idx_idx]]
        loc_idx_i = meshgrid_idx[2]
        loc_idx_j = meshgrid_idx[3]
        coevo_couplings[0, use_idx_i, use_idx_j] = coevo_couplings[0, loc_idx_i, loc_idx_j]
    return coevo_couplings


def handle_cropping_partner(ref_length: int, crop_d: bool, obs_idx: list):
    '''
    Handle cropping for the partner in a ppi
    '''
    if crop_d and (ref_length > CROP_SIZE):
        crop_idx = get_random_crop_idx(ref_length, CROP_SIZE, obs_idx, 1)[0]
        if crop_idx <= (ref_length - CROP_SIZE):
            return range(crop_idx, crop_idx+CROP_SIZE)
        else:
            return range(crop_idx-CROP_SIZE+1, crop_idx+1)
    else:
        return range(ref_length)


def gen_paired_evolution_aggregator(iden_eff_weights_1, iden_eff_weights_2,
                                    msa_embeddings_1, msa_embeddings_2,
                                    idx_range_1, idx_range_2):
    cur_length_1, cur_length_2 = idx_range_1.stop-idx_range_1.start, idx_range_2.stop-idx_range_2.start
    msa_embeddings_1 = msa_embeddings_1.transpose(0, 1)
    msa_embeddings_2 = msa_embeddings_2.transpose(0, 1)
    m_eff_1 = iden_eff_weights_1.sum()
    m_eff_2 = iden_eff_weights_2.sum()
    one_body_term_1 = torch.einsum('ckl,k->lc', msa_embeddings_1, iden_eff_weights_1)/m_eff_1
    one_body_term_2 = torch.einsum('ckl,k->lc', msa_embeddings_2, iden_eff_weights_2)/m_eff_2
    max_msa_embeddings_1 = torch.max(msa_embeddings_1.transpose(0, 2), dim=1)[0]  # $L_1 \times C$
    max_msa_embeddings_2 = torch.max(msa_embeddings_2.transpose(0, 2), dim=1)[0]  # $L_2 \times C$
    paired_evo_couplings = torch.zeros((1, cur_length_1, cur_length_2, 4224), dtype=msa_embeddings_1.dtype, device=msa_embeddings_1.device)
    m_eff_12 = torch.sqrt(m_eff_1*m_eff_2)
    for use_idx_1, idx_1 in enumerate(idx_range_1):
        x1 = one_body_term_1[idx_1]
        x1_max = max_msa_embeddings_1[idx_1]/m_eff_12  ## avoid overflow
        for use_idx_2, idx_2 in enumerate(idx_range_2):
            x2 = one_body_term_2[idx_2]
            x2_max = max_msa_embeddings_2[idx_2]
            x12_max = torch.einsum('i,j->ij', x1_max, x2_max)  # $C \times C$
            paired_evo_couplings[0, use_idx_1, use_idx_2, :64] = x1
            paired_evo_couplings[0, use_idx_1, use_idx_2, 64:128] = x2
            paired_evo_couplings[0, use_idx_1, use_idx_2, 128:] = x12_max.flatten()
    return paired_evo_couplings


class ResNetPPI(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.resnet1d = ResNet1D(ENCODE_DIM+HYDRO_DIM, [8])
        self.resnet2d = ResNet2D(4224, [(1,2,4,8)]*4) # 18
        self.conv2d_37 = nn.Sequential(
            nn.Conv2d(96, 37, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(37),
            nn.ELU(inplace=True),
        )
        self.softmax_func = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
        """
        cuda: bool = False, half: bool = False, cache: bool = True
        if cache and (loaded_gen_coevolution_aggregator[1] is not None):
            self.gen_coevolution_aggregator = loaded_gen_coevolution_aggregator[1]
        else:
            self.gen_coevolution_aggregator = torch.jit.script(
                gen_coevolution_aggregator,
                example_inputs=demo_input_for_gen_coevolution_aggregator(cuda, half))
            self.gen_coevolution_aggregator.save(loaded_gen_coevolution_aggregator[0])
        """
    
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

    def forward_single_protein(self, pw_encodings_group, iden_eff_weights, obs_idx, crop_d: bool):
        # MSA Embeddings: $K \times C \times L$
        msa_embeddings = self.get_msa_embeddings(pw_encodings_group)
        cropping_info, cur_length, meshgrid, record_idx, record_mask = handle_cropping(msa_embeddings.shape[2], crop_d, obs_idx)
        coevo_couplings = gen_coevolution_aggregator(iden_eff_weights.to(msa_embeddings.dtype), msa_embeddings, cur_length, meshgrid, record_idx, record_mask)
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

    def training_step_sp(self, train_batch, batch_idx):
        '''single protein'''
        ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1, label_dist6d_1 = train_batch
        l_idx_1 = ref_seq_info_1['obs_mask']
        pred_dist6d_1, cropping_info_1 = self.forward_single_protein(pw_encodings_group_1, iden_eff_weights_1, l_idx_1.tolist(), True)
        #assert not torch.isinf(pred_dist6d_1).any()
        #assert not torch.isnan(pred_dist6d_1).any()
        loss_1 = self.loss_single_protein(cropping_info_1, l_idx_1, pred_dist6d_1, label_dist6d_1.unsqueeze(0))
        #assert not torch.isinf(loss_1).any()
        #assert not torch.isnan(loss_1).any()
        self.log('train_loss', loss_1, batch_size=1, on_step=True, on_epoch=True)
        return loss_1

    def validation_step_sp(self, val_batch, batch_idx):
        '''single protein'''
        ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1, label_dist6d_1 = val_batch
        l_idx_1 = ref_seq_info_1['obs_mask']
        pred_dist6d_1, cropping_info_1 = self.forward_single_protein(pw_encodings_group_1, iden_eff_weights_1, l_idx_1.tolist(), True)
        loss_1 = self.loss_single_protein(cropping_info_1, l_idx_1, pred_dist6d_1, label_dist6d_1.unsqueeze(0))
        self.log('val_loss', loss_1, batch_size=1, on_step=True, on_epoch=True)
        return loss_1

    def forward_sp(self, inputs):
        '''single protein'''
        # NOTE: for prediction/inference actions
        # pw_encodings_group, iden_eff_weights = inputs
        pred = self.forward_single_protein(*inputs, [], False)[0]
        return self.softmax_func(0.5*(pred + pred.transpose(-1, -2)))

    def forward_ppi(self, pw_encodings_group_1, pw_encodings_group_2, iden_eff_weights_1, iden_eff_weights_2, obs_idx_1, obs_idx_2, crop_d: bool):
        msa_embeddings_1 = self.get_msa_embeddings(pw_encodings_group_1)
        msa_embeddings_2 = self.get_msa_embeddings(pw_encodings_group_2)
        idx_range_1 = handle_cropping_partner(msa_embeddings_1.shape[2], crop_d, obs_idx_1)
        idx_range_2 = handle_cropping_partner(msa_embeddings_2.shape[2], crop_d, obs_idx_2)
        paired_evo_couplings = gen_paired_evolution_aggregator(
            iden_eff_weights_1, iden_eff_weights_2,
            msa_embeddings_1, msa_embeddings_2,
            idx_range_1, idx_range_2)
        #assert not torch.isinf(paired_evo_couplings).any()
        #assert not torch.isnan(paired_evo_couplings).any()
        torch.cuda.empty_cache()
        ret = self.conv2d_37(self.resnet2d(paired_evo_couplings.movedim(3, 1))), idx_range_1, idx_range_2
        #assert not torch.isinf(ret[0]).any()
        if torch.isnan(ret[0]).any():
            torch.save(paired_evo_couplings.detach(), 'paired_evo_couplings.pt')
            raise AssertionError((paired_evo_couplings.max(), paired_evo_couplings.min()))
        return ret

    def loss_ppi(self, obs_idx_1, obs_idx_2, idx_range_1, idx_range_2, pred, target):
        mask_1 = (obs_idx_1 >= idx_range_1.start) & (obs_idx_1 < idx_range_1.stop)
        mask_2 = (obs_idx_2 >= idx_range_2.start) & (obs_idx_2 < idx_range_2.stop)
        l_idx_1 = obs_idx_1[mask_1] - idx_range_1.start
        l_idx_2 = obs_idx_2[mask_2] - idx_range_2.start
        pred = pred[:, :, l_idx_1, :][:, :, :, l_idx_2]
        t_l_idx_1 = torch.arange(obs_idx_1.shape[0], dtype=torch.int64)[mask_1]
        t_l_idx_2 = torch.arange(obs_idx_2.shape[0], dtype=torch.int64)[mask_2]
        target = target[:, t_l_idx_1, :][:, :, t_l_idx_2]
        assert pred.shape[-1] > 0 and pred.shape[-2] > 0
        assert pred.shape[-2:] == target.shape[-2:]
        return self.loss_func(pred, target)

    def training_step(self, train_batch, batch_idx):
        (ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1,
         ref_seq_info_2, pw_encodings_group_2, iden_eff_weights_2,
         label_dist6d_12) = train_batch
        obs_idx_1, obs_idx_2 = ref_seq_info_1['obs_mask'], ref_seq_info_2['obs_mask']
        pred_dist6d_12, idx_range_1, idx_range_2 = self.forward_ppi(pw_encodings_group_1, pw_encodings_group_2, iden_eff_weights_1, iden_eff_weights_2, obs_idx_1.tolist(), obs_idx_2.tolist(), True)
        loss = self.loss_ppi(obs_idx_1, obs_idx_2, idx_range_1, idx_range_2, pred_dist6d_12, label_dist6d_12.unsqueeze(0))
        assert not torch.isinf(loss).any()
        assert not torch.isnan(loss).any()
        self.log('train_loss', loss, batch_size=1, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        (ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1,
         ref_seq_info_2, pw_encodings_group_2, iden_eff_weights_2,
         label_dist6d_12) = val_batch
        obs_idx_1, obs_idx_2 = ref_seq_info_1['obs_mask'], ref_seq_info_2['obs_mask']
        pred_dist6d_12, idx_range_1, idx_range_2 = self.forward_ppi(pw_encodings_group_1, pw_encodings_group_2, iden_eff_weights_1, iden_eff_weights_2, obs_idx_1.tolist(), obs_idx_2.tolist(), True)
        loss = self.loss_ppi(obs_idx_1, obs_idx_2, idx_range_1, idx_range_2, pred_dist6d_12, label_dist6d_12.unsqueeze(0))
        self.log('val_loss', loss, batch_size=1, on_step=True, on_epoch=True)
        return loss

    def forward(self, inputs):
        # NOTE: for prediction/inference actions
        # pw_encodings_group_1, pw_encodings_group_2, iden_eff_weights_1, iden_eff_weights_2 = inputs
        return self.softmax_func(self.forward_ppi(*inputs, [], [], False)[0])

