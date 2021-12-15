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

# @Created Date: 2021-12-14 09:46:48 pm
# @Filename: plot.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-15 11:01:00 am
from ResNetPPI.demo_configs import *
from ResNetPPI.utils import get_bins_tex, binned_dist6d_1, binned_dist6d_12
from matplotlib.animation import FuncAnimation


def plot2dist6d(pdb_id, pdb_binary_int, outdir='./figs'):
    titles = get_bins_tex(0.5, 0, 20, '$[20,+\infty)$', non_contact_at_first=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal')
    ax.set_title(titles[0])
    cax = ax.pcolor(binned_dist6d_12[:, :, 0], vmin=0, vmax=1, cmap='Blues')
    ax.invert_yaxis()
    #fig.colorbar(cax)
    def animate(i):
        ax.set_title(titles[i])
        cax.set_array(binned_dist6d_12[:, :, i].flatten())
    anim = FuncAnimation(fig, animate, interval=150, frames=binned_dist6d_12.shape[2], repeat=True)
    fig.show()
    anim.save(f'{outdir}/{pdb_id}.{pdb_binary_int.chain_1.struct_asym_id}.{pdb_binary_int.chain_2.struct_asym_id}_dist6d_maps.gif', writer='pillow')
    # ---
    titles = get_bins_tex(0.5, 2, 20, non_contact_at_first=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal')
    ax.set_title(titles[0])
    cax = ax.pcolor(binned_dist6d_1[:, :, 0], vmin=0, vmax=1, cmap='Blues')
    ax.invert_yaxis()
    #fig.colorbar(cax)
    def animate(i):
        ax.set_title(titles[i])
        cax.set_array(binned_dist6d_1[:, :, i].flatten())
    anim = FuncAnimation(fig, animate, interval=150, frames=binned_dist6d_1.shape[2], repeat=True)
    fig.show()
    anim.save(f'{outdir}/{pdb_id}.{pdb_binary_int.chain_1.struct_asym_id}_dist6d_maps.gif', writer='pillow')
