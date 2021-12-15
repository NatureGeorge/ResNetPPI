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

# @Created Date: 2021-12-14 06:58:33 pm
# @Filename: featuredata.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2021-12-14 06:58:55 pm

# Hydrophobic residues and hydrophilic residues were classified according to the ordering of hydrophobicity in the following paper.:
# Miyazawa, S., & Jernigan, R. L. (1996). 
# Residue-residue potentials with a favorable contact pair term 
# and an unfavorable high packing density term, for simulation and threading. 
# Journal of molecular biology, 256(3), 623–644. 
# https://doi.org/10.1006/jmbi.1996.0114
hydrophobic_group = (0, 4, 9, 10, 12, 13, 17, 18, 19) # seq order: ARNDCQEGHILKMFPSTWYV
hydrophilic_group = tuple(i for i in range(20) if i not in hydrophobic_group)
hydrophobic_bool_tab = dict(tuple(zip(hydrophobic_group, [1]*len(hydrophobic_group)))+tuple(zip(hydrophilic_group, [0]*len(hydrophilic_group))))
