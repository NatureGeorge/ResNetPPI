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

# @Created Date: 2022-01-05 08:44:10 pm
# @Filename: training_pipeline.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2022-01-13 04:14:23 pm
import argparse
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb_list_train", type=str, help="training dataset.")
    parser.add_argument("pdb_list_val", type=str, help="validation dataset.")
    parser.add_argument("-msa_dir", type=str, required=False, dest='msa_dir', default="./workdir/msa_dataset/", help="MSA file directory.")
    parser.add_argument("-pdb_dir", type=str, required=False, dest='pdb_dir', default="./workdir/pdb_dataset/", help="PDB file directory.")
    parser.add_argument("-src_dir", type=str, required=False, dest='sc_dir', default="./workdir/ResNetPPI-main", help="`ResNetPPI` source code directory.")
    parser.add_argument('-num_workers', type=int, required=False, dest='num_workers', default=2, help='number of workers to use for DataLoader.')
    parser.add_argument('-gpus', type=str, required=False, dest='gpus', default="1", help='list of GPU (comma-seperate) to use during training.')
    parser.add_argument('-max_epochs', type=int, required=False, dest='max_epochs', default=20, help='number of epochs.')
    parser.add_argument('-precision', type=int, required=False, dest='precision', default=16, help='precision')
    parser.add_argument('-checkpoint_dir', type=str, required=False, dest='checkpoint_dir', default='', help='location of trained weights')
    args = parser.parse_args()
    sys.path.append(str(Path(args.src_dir).absolute()))
    from ResNetPPI.model import ResNetPPI
    from ResNetPPI.dataset import SeqStructDataset
    train_data = SeqStructDataset(args.pdb_list_train, args.msa_dir, args.pdb_dir)
    train_loader = DataLoader(dataset=train_data, batch_size=None, num_workers=args.num_workers)
    val_data = SeqStructDataset(args.pdb_list_val, args.msa_dir, args.pdb_dir)
    val_loader = DataLoader(dataset=train_data, batch_size=None, num_workers=args.num_workers)
    model = ResNetPPI() if args.checkpoint_dir == '' else ResNetPPI.load_from_checkpoint(args.checkpoint_dir)
    trainer = Trainer(gpus=[int(i) for i in args.gpus.split(',')], max_epochs=args.max_epochs, precision=args.precision)
    trainer.fit(model, train_loader, val_loader)
