#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# env var ROCM_VERSION - The ROCm version to consider.

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return
fi

set -xe

# preinstall tzdata without install recommendations
if [ ! -z ${SET_TIMEZONE+x} ]; then
  TZ=Etc/UTC
  sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
fi
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y --no-install-recommends tzdata

# install latest rocm installation tool
sudo apt update

script_dir=$(dirname "$0")
source ${script_dir}/librocm.sh
install_rocm_ubuntu

# patch ROCm C API compilation issues for legacy ROCm versions
sed -s -i "s,typedef rocrand_generator_base_type hiprandGenerator_st;,typedef struct rocrand_generator_base_type hiprandGenerator_st;," /opt/rocm/include/hiprand/hiprand_hcc.h || true
sed -s -i 's,HIPSPARSE_ORDER_COLUMN\s*\[\[deprecated("Please use HIPSPARSE_ORDER_COL instead")\]\],HIPSPARSE_ORDER_COLUMN,g' /opt/rocm/include/hipsparse/hipsparse.h || true

