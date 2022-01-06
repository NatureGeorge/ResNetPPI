# Copyright 2022 Zefeng Zhu
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

# @Created Date: 2022-01-06 04:25:05 pm
# @Filename: inference_pipeline.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2022-01-06 04:34:49 pm
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb_list_test", type=str, help="testing dataset.")
    parser.add_argument('weights_dir', type=str, help='location of trained weights.')
    parser.add_argument('output', type=str, help='output file.')
    parser.add_argument("-msa_dir", type=str, required=False, dest='msa_dir', default="./workdir/msa_dataset/", help="MSA file directory.")
    parser.add_argument("-pdb_dir", type=str, required=False, dest='pdb_dir', default="./workdir/pdb_dataset/", help="PDB file directory.")
    parser.add_argument("-src_dir", type=str, required=False, dest='sc_dir', default="./workdir/ResNetPPI-main", help="`ResNetPPI` source code directory.")
    args = parser.parse_args()
    sys.path.append(str(Path(args.src_dir).absolute()))
    from ResNetPPI.model import ResNetPPI
    from ResNetPPI.dataset import SeqStructDataset
    test_data = SeqStructDataset(args.pdb_list_test, args.msa_dir, args.pdb_dir, 10000)
    test_loader = iter(DataLoader(dataset=test_data, batch_size=None))
    ref_seq_info_1, pw_encodings_group_1, iden_eff_weights_1, label_dist6d_1 = next(test_loader)
    model = ResNetPPI.load_from_checkpoint(args.weights_dir)
    model.eval()
    with torch.no_grad():
        reconstruction = model((pw_encodings_group_1, iden_eff_weights_1))
    del ref_seq_info_1['obs_mask']
    print("cross-entropy loss of {}: {}".format(ref_seq_info_1, torch.nn.NLLLoss()(torch.log(reconstruction), label_dist6d_1.unsqueeze(0)).item()))
    torch.save(reconstruction, args.output)
    
