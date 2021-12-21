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
# @Last Modified: 2021-12-21 01:16:34 pm
import re
import json
import zlib
from pathlib import Path
import numpy as np
import scipy.spatial
import torch
from torch import nn
import pytorch_lightning as pl
from ResNetPPI.net import ResNet1D, ResNet2D
from ResNetPPI.utils import identity_score, gen_ref_msa_from_pairwise_aln


# SETTINGS
ONEHOT_DIM = 22
ENCODE_DIM = 44 # 46 if add hydrophobic features
ONEHOT = np.eye(ONEHOT_DIM, dtype=np.float32)
REF_SEQ_PAT = re.compile(r"[ARNDCQEGHILKMFPSTWYVX]+")
AA_ALPHABET = np.array(list("ARNDCQEGHILKMFPSTWYV-X"), dtype='|S1').view(np.uint8)


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


def aa2index(seq):
    for i in range(AA_ALPHABET.shape[0]):
        seq[seq == AA_ALPHABET[i]] = i
    seq[seq > 21] = 21


class ResNetPPI: # (pl.LightningModule)
    def __init__(self, device_id: int = -1):
        self.device = torch.device(f'cuda:{device_id}') if (
            device_id >= 0 and 
            torch.cuda.is_available() and 
            torch.cuda.device_count() > 0) else torch.device('cpu')
        self.resnet1d = ResNet1D(ENCODE_DIM, [8])#.to(self.device)
        self.resnet2d = ResNet2D(4224, [4]*18)#.to(self.device)
        self.conv2d_37 = nn.Conv2d(96, 37, kernel_size=3, padding=1, bias=False)
        self.conv2d_41 = nn.Conv2d(96, 41, kernel_size=3, padding=1, bias=False)
        self.softmax_func = nn.Softmax(dim=1)
        self.loss_func = nn.CrossEntropyLoss()
    
    def training_step(self, train_batch, batch_idx):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def load_pairwise_aln_from_a3m(self, path):
        with Path(path).open('rt') as handle:
            self.ref_seq_info = json.loads(next(handle)[1:])
            self.ref_seq_info['obs_mask'] = torch.from_numpy(
                np.where(np.array(list(zlib.decompress(eval(
                    self.ref_seq_info['obs_mask'])).decode('utf-8')), dtype=np.uint8))[0])
            ref_seq = next(handle).rstrip()
            assert bool(REF_SEQ_PAT.fullmatch(ref_seq)), 'Unexpected seq!'
            ref_seq_vec = np.array(list(ref_seq), dtype='|S1').view(np.uint8)
            aa2index(ref_seq_vec)
            for line in handle:
                if line.startswith('>'):
                    continue
                oth_seq = line.rstrip()
                mask_insertion = np.array([False if aa.isupper() or aa == '-' else True for aa in oth_seq])
                if mask_insertion.any():
                    ret_ref_seq_vec = np.full(mask_insertion.shape, 20, dtype=np.uint8)
                    ret_ref_seq_vec[np.where(~mask_insertion)] = ref_seq_vec
                    oth_seq_vec = np.array([(aa.upper() if ins else aa) for aa, ins in zip(oth_seq, mask_insertion)], dtype='|S1').view(np.uint8)
                    aa2index(oth_seq_vec)
                    yield np.asarray([ret_ref_seq_vec, oth_seq_vec])
                else:
                    oth_seq_vec = np.array(list(oth_seq), dtype='|S1').view(np.uint8)
                    aa2index(oth_seq_vec)
                    # assert ref_seq_vec.shape == oth_seq_vec.shape, 'Unexpected situation!'
                    yield np.asarray([ref_seq_vec, oth_seq_vec])

    def onehot_encoding(self, aln: np.ndarray):
        encoding = ONEHOT[aln].transpose((0, 2, 1))
        encoding = encoding.reshape(-1, encoding.shape[-1])
        return torch.from_numpy(encoding)#.to(self.device)

    def pw_encoding(self, aln: np.ndarray):
        return self.onehot_encoding(aln).reshape(1, ENCODE_DIM, -1)

    def gen_pw_embedding(self, pw_msa):
        self.ref_length = (pw_msa[0][0] != 20).sum() # ref_msa.shape[1]
        for pw_aln in pw_msa:
            # $1 \times C \times L_k$
            msa_embedding = self.resnet1d(self.pw_encoding(pw_aln))  # TODO: optimize with torch.utils.data.DataLoader
            if pw_aln.shape[1] != self.ref_length:
                yield msa_embedding[:, :, pw_aln[0] != 20]  # NOTE: maybe a point to optimize (CPU <-> GPU)
            else:
                yield msa_embedding
    
    def msa_embedding(self, pw_msa):
        return torch.cat(tuple(self.gen_pw_embedding(pw_msa)))

    def gen_coevolution_aggregator(self, iden_eff_weights, msa_embeddings):
        msa_embeddings = msa_embeddings.transpose(0, 1)
        # Weights: $1 \times K$
        m_eff = iden_eff_weights.sum()
        # ? with torch.no_grad():
        # One-Body Term: $C \times L$ -> $C \times 1$
        ## $(1 \times K) \times (C \times K \times L)$ -> $C \times L$
        ### one_body_term = torch.matmul(iden_eff_weights, msa_embeddings).squeeze(1).transpose(0, 1)/m_eff
        one_body_term = torch.einsum('ckl,k->lc', msa_embeddings, iden_eff_weights)/m_eff
        msa_embeddings = msa_embeddings.transpose(0, 2)  # $L \times K \times C$
        for idx_i in range(self.ref_length-1):
            f_i = one_body_term[idx_i]
            x_k_i = msa_embeddings[idx_i]
            for idx_j in range(idx_i+1, self.ref_length):
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

    def forward_single_protein(self, msa_file):
        pw_msa = tuple(self.load_pairwise_aln_from_a3m(msa_file))
        iden_eff_weights = torch.from_numpy(get_eff_weights(pw_msa)[1:])#.to(self.device)
        # MSA Embeddings: $K \times C \times L$
        msa_embeddings = self.msa_embedding(pw_msa)  # set self.ref_length
        coevo_agg = self.gen_coevolution_aggregator(iden_eff_weights, msa_embeddings)
        #coevo_couplings = torch.stack(tuple(coevo_agg), dim=0)
        #return coevo_couplings.transpose(-1, -2)
        coevo_couplings = torch.zeros((self.ref_length, self.ref_length, 4224), dtype=torch.float32)
        # TODO: optimization for symmetric tensors
        for (idx_i, idx_j), coevo_cp in coevo_agg:  # tqdm(coevo_agg, total=self.ref_length*(self.ref_length-1)//2):
            coevo_couplings[idx_i, idx_j, :] = coevo_couplings[idx_j, idx_i, :] = coevo_cp
        r2s = self.resnet2d(coevo_couplings.transpose(-1, -3).unsqueeze(0))
        mid = self.conv2d_37(r2s)
        return r2s, self.softmax_func(0.5*(mid + mid.transpose(-1, -2)))

    def loss_single_protein(self, pred, target):
        l_idx = self.ref_seq_info['obs_mask']
        if l_idx.shape[0] == pred.shape[2]:
            pass
        else:
            # TODO: optimization?
            pred = pred[:, :, l_idx, :][:, :, :, l_idx]
            target = target[:, :, l_idx, :][:, :, :, l_idx]
        return self.loss_func(pred, target)

    def forward(self, msa_file):
        pass

