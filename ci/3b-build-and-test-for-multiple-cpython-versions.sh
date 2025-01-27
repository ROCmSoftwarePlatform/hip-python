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

# env var NUM_JOBS

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return
fi

set -xe

CONDA_DIR=${CONDA_DIR:-~/miniconda3}
source ${CONDA_DIR}/etc/profile.d/conda.sh

MIN_PYVER=${MIN_PYVER:-8}
MAX_PYVER=${MAX_PYVER:-13}

for v in $(seq ${MIN_PYVER} ${MAX_PYVER});
do
  conda activate py3${v}
  if [[ "${v}" == "8" ]]; then
    ./build_hip_python_pkgs.sh --pre-clean --post-clean --hip --cuda --run-tests -j ${NUM_JOBS:-16}
  else
    ./build_hip_python_pkgs.sh --pre-clean --post-clean --hip --cuda --run-tests -j ${NUM_JOBS:-16} --no-archive
  fi
  conda deactivate
done
