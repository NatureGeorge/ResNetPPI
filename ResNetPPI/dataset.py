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
# @Last Modified: 2022-01-08 02:04:52 pm
from torch.utils.data import Dataset
from ResNetPPI import DIST_CUTOFF, MAX_K
from ResNetPPI.msa import *
from ResNetPPI.utils import (get_representative_xyz,
                             get_dist6d,
                             get_dist6d_2,
                             get_label_bin_map,
                             load_pairwise_aln_from_a3m,
                             sample_pairwise_aln,
                             get_eff_weights,
                             onehot_encoding,
                             gen_pw_encodings_group,
                             add_hydro_encoding)


class SeqStructDataset(Dataset):
    def __init__(self, pdb_list, msa_dir, pdb_dir, max_k: int = MAX_K, **kwargs):
        self.dataframe = read_csv(pdb_list, dtype=str, keep_default_na=False, **kwargs)
        self.msa_dir = Path(msa_dir)
        self.pdb_dir = Path(pdb_dir)
        self.max_k = max_k

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
        idx_1, dist6d_1 = get_dist6d(xyz_1, DIST_CUTOFF)
        label_dist6d_1 = get_label_bin_map(idx_1, dist6d_1, 0.5, 2, 20, non_contact_at_first=False)
        # binned_dist6d_1 = get_bin_map(idx_1, dist6d_1, 0.5, 2, 20, non_contact_at_first=False)
        loading_a3m_1 = load_pairwise_aln_from_a3m(msa_file_1)
        ref_seq_info_1 = next(loading_a3m_1)
        pw_msa_1 = sample_pairwise_aln(tuple(loading_a3m_1), self.max_k)
        iden_eff_weights_1 = get_eff_weights(pw_msa_1)[1:]
        pw_encodings_1 = tuple(add_hydro_encoding(onehot_encoding(pw_aln)) for pw_aln in pw_msa_1)
        iden_eff_weights_idx_1 = []
        pw_encodings_group_1 = tuple(gen_pw_encodings_group(pw_encodings_1, iden_eff_weights_idx_1))
        # assert len(iden_eff_weights_idx_1) == len(iden_eff_weights_1)
        iden_eff_weights_1 = iden_eff_weights_1[iden_eff_weights_idx_1]
        # assert sum(i.shape[0] for i in pw_encodings_group_1) == iden_eff_weights_1.shape[0]
        return ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1, label_dist6d_1#, binned_dist6d_1
        #xyz_2 = get_representative_xyz(gemmi_obj[MODEL_ID].get_subchain(pdb_binary_int.chain_2.struct_asym_id))
        #idx_2, dist6d_2 = get_dist6d(xyz_2, DIST_CUTOFF)
        #idx_12, dist6d_12 = get_dist6d_2(xyz_2, xyz_1, DIST_CUTOFF)
        #label_dist6d_2 = get_label_bin_map(idx_2, dist6d_2, 0.5, 2, 20, non_contact_at_first=False)
        #label_dist6d_12 = get_label_bin_map(idx_12, dist6d_12, 0.5, 0, 20, non_contact_at_first=False)
