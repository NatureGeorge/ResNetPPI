# @Created Date: 2021-10-15 06:38:47 pm
# @Filename: pipeline_configs.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: ZeFeng Zhu
# @Last Modified: 2021-10-15 06:38:56 pm
from coords6d import *
import gemmi
import prody
import re
import string
from pathlib import Path
import numpy as np
from functools import partial
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numba
import scipy.special
from collections import namedtuple

sns.set_style('whitegrid')
sns.set_style({'font.family': 'serif', 'font.serif': ['Times New Roman']})
CONSOLE = logging.StreamHandler()
CONSOLE.setLevel(logging.WARNING)
LOGGER = logging.getLogger('ZZFLog')
LOGGER.addHandler(CONSOLE)

PDB_CHAIN = namedtuple('PDB_CHAIN', 'pdb_id entity_id struct_asym_id chain_id')
PDB_BINARY_CHAIN = namedtuple('PDB_BINARY_CHAIN', 'chain_1 chain_2')


def gen_res_idx_range(obs_index):
    last_end = 0
    last_index = 0
    for beg_i, end_i in obs_index:
        yield (last_end, last_end + end_i - beg_i + 1), (beg_i-last_index-1)
        last_end = last_end + end_i - beg_i + 1
        last_index = end_i


def get_res_idx_range(obs_index):
    res_idxes, missing_segs = zip(*gen_res_idx_range(obs_index))
    missing_segs = missing_segs[1:]
    return res_idxes, missing_segs


def prepare_input_seq_and_folder(folder, seq_header, seq):
    fasta_file = Path(folder)/'seq.fasta'
    fasta_file.parent.mkdir(parents=True, exist_ok=True)
    (fasta_file.parent/'log').mkdir(parents=True, exist_ok=True)
    with fasta_file.open('wt') as handle:
        handle.write(f'>{seq_header}\n')
        handle.write(seq)
    return fasta_file


def get_real_or_virtual_CB(res):
    if res.name != 'GLY':
        try:
            return res['CB'][0].pos.tolist()
        except Exception:
            LOGGER.warning(f"no CB for {res}")
    try:
        Ca = res['CA'][0].pos
        b = Ca - res['N'][0].pos
        c = res['C'][0].pos - Ca
    except Exception as e:
        LOGGER.error(f"no enough anchor atoms for {res}")
        raise e
    a = b.cross(c)
    return (-0.58273431*a + 0.56802827*b - 0.54067466*c + Ca).tolist()


def get_representative_xyz(chain_obj, representative_atom='CB', dtype=np.float32):
    if representative_atom == 'CB':
        xyz = np.array([get_real_or_virtual_CB(res) for res in chain_obj], dtype=dtype)
    elif representative_atom == 'CA':
        xyz = np.array([res['CA'][0].pos.tolist() for res in chain_obj], dtype=dtype)
    else:
        return
    assert xyz.shape[0] == chain_obj.length(), "Missing anchor atoms!"
    return xyz


def parse_a3m(filename):
    '''
    read A3M and convert letters into integers in the 0..20 range, also keep track of 
    * source code from <https://github.com/RosettaCommons/trRosetta2/blob/main/trRosetta/parsers.py>
    '''

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return {'msa':msa, 'ins':ins}


def get_bin_map(idx: np.ndarray, mat: np.ndarray, size_bins: float, v_min: float, v_max: float, non_contact_at_first : bool = True) -> np.ndarray:
    idx0 = idx[0]
    idx1 = idx[1]
    assert v_max > v_min and size_bins > 0
    num_bins = round((v_max - v_min) / size_bins) + 1
    bin_mat = np.zeros(mat.shape+(num_bins,), dtype=np.bool_)
    use_mat = mat[idx0, idx1]
    if non_contact_at_first:
        bin_mat[idx0, idx1, 0] = 1
        bin_mat[:, :, 0] = ~bin_mat[:, :, 0]
        for bin_i in range(1, num_bins):
            bin_mat[idx0, idx1, bin_i] = np.where(
                (use_mat >= v_min + size_bins * (bin_i - 1)) & (use_mat < v_min + size_bins * bin_i), True, False)
    else:
        bin_mat[idx0, idx1, num_bins-1] = 1
        bin_mat[:, :, num_bins-1] = ~bin_mat[:, :, num_bins-1]
        for bin_i in range(1, num_bins):
            bin_mat[idx0, idx1, bin_i-1] = np.where(
                (use_mat >= v_min + size_bins * (bin_i - 1)) & (use_mat < v_min + size_bins * bin_i), True, False)
    return bin_mat.astype(np.float32)


def get_bins_tex(size_bins: float, v_min: float, v_max: float, init='$[0,2) \cup [20,+\infty)$', non_contact_at_first:bool=True):
    assert v_max > v_min and size_bins > 0
    num_bins = round((v_max - v_min) / size_bins) + 1
    if non_contact_at_first:
        bins = [init]
        for bin_i in range(1, num_bins):
            cur = v_min + size_bins * (bin_i-1)
            cur_next = v_min + size_bins * bin_i
            bins.append(f'$[{cur},{cur_next})$')
    else:
        bins = []
        for bin_i in range(1, num_bins):
            cur = v_min + size_bins * (bin_i-1)
            cur_next = v_min + size_bins * bin_i
            bins.append(f'$[{cur},{cur_next})$')
        bins.append(init)
    return bins


def loc_ij(n, cn2, i, j):
    if j > i:
        return cn2 - scipy.special.comb(n-i, 2, exact=True) + j - i - 1
    elif i > j:
        return cn2 - scipy.special.comb(n-j, 2, exact=True) + i - j - 1
    else:
        return


@numba.njit(cache=True)
def identity_score(a, b):
    '''
    usage:
    >>> ar = scipy.spatial.distance.pdist(msa, metric=identity_score)
    >>> sar = scipy.spatial.distance.squareform(ar)
    >>> np.fill_diagonal(sar, 1)
    >>> # OR
    >>> n = msa.shape[0]
    >>> cn2 = scipy.special.comb(n, 2, exact=True)
    >>> loc_ij_func = partial(loc_ij, n, cn2)
    >>> sar = np.ones((n, n), dtype=np.float32)
    >>> for i in range(n-1):
            for j in range(i+1, n):
                sar[i,j] = sar[j,i] = ar[loc_ij_func(i,j)]
    '''
    mask_ab_sum = ((a == 20) & (b == 20)).sum()
    return ((a == b).sum() - mask_ab_sum)/(a.shape[0] - mask_ab_sum)
