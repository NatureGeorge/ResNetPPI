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

# @Created Date: 2021-12-23 03:02:22 pm
# @Filename: msa.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-25 03:57:14 pm
import os
import datetime
import argparse
import zlib
from pathlib import Path
from collections import namedtuple
import joblib
import numpy as np
import gemmi
from pandas import read_csv
import orjson as json


MODEL_ID = 0
PDB_CHAIN = namedtuple('PDB_CHAIN', 'pdb_id entity_id struct_asym_id chain_id')
PDB_BINARY_CHAIN = namedtuple('PDB_BINARY_CHAIN', 'chain_1 chain_2')


def prepare_input_seq_and_folder(folder, seq_header, seq):
    fasta_file = Path(folder)/'seq.fasta'
    fasta_file.parent.mkdir(parents=True, exist_ok=True)
    (fasta_file.parent/'log').mkdir(parents=True, exist_ok=True)
    with fasta_file.open('wt') as handle:
        handle.write(f'>{seq_header}\n')
        handle.write(seq)
    return fasta_file


def running_msa_pipeline(args):
    dataframe = read_csv(args.pdb_list, dtype=str, keep_default_na=False, chunksize=300)
    for dataset in dataframe:
        msa_cmds = set()
        for record in dataset.itertuples(index=False):
            pdb_chain_1 = PDB_CHAIN(record.pdb_id, record.entity_id_1, record.struct_asym_id_1, record.chain_id_1)
            pdb_chain_2 = PDB_CHAIN(record.pdb_id, record.entity_id_2, record.struct_asym_id_2, record.chain_id_2)
            pdb_binary_int = PDB_BINARY_CHAIN(pdb_chain_1, pdb_chain_2)
            gemmi_obj = None
            for chain in pdb_binary_int:
                chain_wdir = Path(args.out_dir)/f'{record.pdb_id}/{chain.struct_asym_id}'
                if not chain_wdir.exists():
                    if gemmi_obj is None:
                        gemmi_obj = gemmi.read_structure(str(Path(args.pdb_dir)/f"{record.pdb_id}.cif.gz"))
                    chain_obj = gemmi_obj[MODEL_ID].get_subchain(chain.struct_asym_id) # chain_obj.make_one_letter_sequence()
                    entity_seq = gemmi.one_letter_code(gemmi_obj.get_entity(chain.entity_id).full_sequence)
                    res_i_beg = chain_obj[0].label_seq
                    res_i_end = chain_obj[-1].label_seq
                    obs_mask = np.zeros(res_i_end-res_i_beg+1, dtype=np.uint8)
                    obs_mask[[(res_i.label_seq-res_i_beg) for res_i in chain_obj]] = 1
                    prepare_input_seq_and_folder(chain_wdir,
                        json.dumps(dict(
                            pdb_id=record.pdb_id,
                            struct_asym_id=chain.struct_asym_id,
                            obs_mask=str(zlib.compress(bytes(''.join(map(str, obs_mask)), encoding='utf-8'))))
                        ).decode('utf-8'),
                        entity_seq[res_i_beg-1: res_i_end])
                if not (chain_wdir/'t000_.msa0.a3m').exists():
                    msa_cmds.add(f"{args.sc_dir}/make_msa.sh {chain_wdir}/seq.fasta {chain_wdir} {args.n_cpu} {args.n_mem} {args.seqdb_dir} > {chain_wdir}/log/make_msa.stdout 2> {chain_wdir}/log/make_msa.stderr")
        if msa_cmds:
            print(f"[{datetime.datetime.now()}] would run {len(msa_cmds)} cmds")
            joblib.Parallel(n_jobs=min(args.n_job, len(msa_cmds)))(joblib.delayed(os.system)(msa_cmd_i) for msa_cmd_i in msa_cmds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("pdb_list", type=str, help="PDBlist file.")
    parser.add_argument("out_dir", type=str, help="output directory.")
    parser.add_argument("-pdb_dir", type=str, required=False, dest='pdb_dir', default="./pdb", help="PDB file directory.")
    parser.add_argument("-sc_dir", type=str, required=False, dest='sc_dir', default="./scripts", help="make_msa.sh directory.")
    parser.add_argument("-seqdb_dir", type=str, required=False, dest='seqdb_dir', default="./seqs_database/UniRef30_2020_06/UniRef30_2020_06", help="sequence database directory.")
    parser.add_argument('-n_cpu', type=int, required=False, dest='n_cpu', default=2, help='number of CPU to use when running `make_msa.sh`.')
    parser.add_argument('-n_mem', type=int, required=False, dest='n_mem', default=8, help='size of memory to use when running `make_msa.sh`.')
    parser.add_argument('-n_job', type=int, required=False, dest='n_job', default=2, help='the maximum number of concurrently running jobs (joblib).')
    running_msa_pipeline(parser.parse_args())

