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

# @Created Date: 2021-10-15 06:38:47 pm
# @Filename: utils.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: ZeFeng Zhu
# @Last Modified: 2021-10-15 06:38:56 pm
from ResNetPPI.coords6d import *
import re
from pathlib import Path
import numpy as np
import logging
import numba
import scipy.special
from collections import namedtuple

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


def get_real_or_virtual_CB(res, gly_ca: bool = False):
    if res.name != 'GLY':
        try:
            return res['CB'][0].pos.tolist()
        except Exception:
            LOGGER.warning(f"no CB for {res}")
    else:
        if gly_ca:
            try:
                return res['CA'][0].pos.tolist()
            except Exception as e:
                LOGGER.error(f"no CA for {res}")
                raise e
    try:
        Ca = res['CA'][0].pos
        b = Ca - res['N'][0].pos
        c = res['C'][0].pos - Ca
    except Exception as e:
        LOGGER.error(f"no enough anchor atoms for {res}")
        raise e
    a = b.cross(c)
    return (-0.58273431*a + 0.56802827*b - 0.54067466*c + Ca).tolist()


def get_representative_xyz(chain_obj, representative_atom='CB', gly_ca: bool = False, dtype=np.float32):
    if representative_atom == 'CB':
        xyz = np.array([get_real_or_virtual_CB(res, gly_ca) for res in chain_obj], dtype=dtype)
    elif representative_atom == 'CA':
        xyz = np.array([res['CA'][0].pos.tolist() for res in chain_obj], dtype=dtype)
    else:
        return
    assert xyz.shape[0] == chain_obj.length(), "Missing anchor atoms!"
    return xyz


ref_seq_pat = re.compile(r"[ARNDCQEGHILKMFPSTWYVX]+")
aa_alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-X"), dtype='|S1').view(np.uint8)


def aa2index(seq):
    for i in range(aa_alphabet.shape[0]):
        seq[seq == aa_alphabet[i]] = i
    seq[seq > 21] = 21


def load_pairwise_aln_from_a3m(path):
    with Path(path).open('rt') as handle:
        next(handle)
        ref_seq = next(handle).rstrip()
        assert bool(ref_seq_pat.fullmatch(ref_seq)), 'Unexpected seq!'
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


def gen_ref_msa_from_pairwise_aln(pw_aln):
    use_idx = np.where(pw_aln[0][0] != 20)[0]
    ref_msa = np.ones((len(pw_aln)+1, use_idx.shape[0]), dtype=np.uint8)
    ref_msa[0] = pw_aln[0][0][use_idx]
    for idx, (pw_ref, pw_hmo) in enumerate(pw_aln):
        use_idx = np.where(pw_ref!=20)[0]
        ref_msa[idx+1] = pw_hmo[use_idx]
    return ref_msa


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


def to_interval(lyst):
    if not isinstance(lyst, (set, frozenset)):
        lyst = frozenset(int(i) for i in lyst if i is not None)
    if len(lyst) == 0:
        return tuple()

    start = []
    interval_lyst = []
    max_edge = max(lyst)
    min_edge = min(lyst)

    if len(lyst) == (max_edge + 1 - min_edge):
        return ((min_edge, max_edge),)

    lyst_list = sorted(lyst)

    for j in lyst_list:
        if len(start) == 0:
            i = j
            start.append(j)
            i += 1
        else:
            if (i != j) or (j == max(lyst_list)):
                if j == max(lyst_list):
                    if (i != j):
                        interval_lyst.append(start)
                        interval_lyst.append([j])
                        break
                    else:
                        start.append(j)
                interval_lyst.append(start)
                start = [j]
                i = j + 1
            else:
                start.append(j)
                i += 1

    return tuple((min(li), max(li)) for li in interval_lyst)
