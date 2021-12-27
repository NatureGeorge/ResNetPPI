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

# @Created Date: 2021-12-27 05:02:29 pm
# @Filename: dataset.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-27 07:07:06 pm
import torch
from torch.utils.data import Dataset
from ResNetPPI.msa import *
from ResNetPPI.utils import (get_representative_xyz,
                             get_dist6d,
                             get_dist6d_2,
                             get_label_bin_map,
                             load_pairwise_aln_from_a3m,
                             sample_pairwise_aln)

DIST_CUTOFF = 20.0


class SeqStructDataset(Dataset):
    def __init__(self, pdb_list, msa_dir, pdb_dir):
        self.dataframe = read_csv(pdb_list, dtype=str, keep_default_na=False)
        self.msa_dir = Path(msa_dir)
        self.pdb_dir = Path(pdb_dir)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        record = self.dataframe.loc[index]
        pdb_chain_1 = PDB_CHAIN(record.pdb_id, record.entity_id_1, record.struct_asym_id_1, record.chain_id_1)
        pdb_chain_2 = PDB_CHAIN(record.pdb_id, record.entity_id_2, record.struct_asym_id_2, record.chain_id_2)
        pdb_binary_int = PDB_BINARY_CHAIN(pdb_chain_1, pdb_chain_2)
        gemmi_obj = gemmi.read_structure(str(self.pdb_dir/f"{record.pdb_id}.cif.gz"))
        msa_file_1 = self.msa_dir/record.pdb_id/pdb_binary_int.chain_1.struct_asym_id/'t000_.msa0.a3m'
        xyz_1 = get_representative_xyz(gemmi_obj[MODEL_ID].get_subchain(pdb_binary_int.chain_1.struct_asym_id))
        #xyz_2 = get_representative_xyz(gemmi_obj[MODEL_ID].get_subchain(pdb_binary_int.chain_2.struct_asym_id))
        idx_1, dist6d_1 = get_dist6d(xyz_1, DIST_CUTOFF)
        #idx_2, dist6d_2 = get_dist6d(xyz_2, DIST_CUTOFF)
        #idx_12, dist6d_12 = get_dist6d_2(xyz_2, xyz_1, DIST_CUTOFF)
        binned_dist6d_1 = torch.from_numpy(get_label_bin_map(idx_1, dist6d_1, 0.5, 2, 20, non_contact_at_first=False))
        #binned_dist6d_2 = torch.from_numpy(get_label_bin_map(idx_2, dist6d_2, 0.5, 2, 20, non_contact_at_first=False))
        #binned_dist6d_12 = torch.from_numpy(get_label_bin_map(idx_12, dist6d_12, 0.5, 0, 20, non_contact_at_first=False))
        loading_a3m_1 = load_pairwise_aln_from_a3m(msa_file_1)
        ref_seq_info_1 = next(loading_a3m_1)
        pw_msa_1 = sample_pairwise_aln(tuple(loading_a3m_1))
        return ref_seq_info_1, pw_msa_1, binned_dist6d_1
