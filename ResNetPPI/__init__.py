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

# @Created Date: 2021-12-14 07:03:49 pm
# @Filename: __init__.py
# @Email:  zhuzefeng@stu.pku.edu.cn
# @Author: Zefeng Zhu
# @Last Modified: 2022-01-03 02:47:18 pm

# SETTINGS
ONEHOT_DIM = 22
ENCODE_DIM = 44 # 46 if add hydrophobic features
CROP_SIZE = 128 # 64 if CUDA run out of memory
DIST_CUTOFF = 20.0
MAX_K = 1000
