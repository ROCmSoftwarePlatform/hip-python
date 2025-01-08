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
# env var AMDGPU_INSTALL_USE_INTERNAL, ("true","false"), optional - If the internal amdgpu install tools should be used.
# env var AMDGPU_INSTALL_INTERNAL_SPECIFY_BUILD, ("true","false"), optional - If a particular build should be selected.
# env var AMDGPU_INSTALL_INTERNAL_REPO_ARGS, required if AMDGPU_INSTALL_INTERNAL_SPECIFY_BUILD == true - The arguments to pass to amdgpu-repo, e.g.: '--amdgpu-build=1708475 --rocm-build=compute-rocm-rel-6.0/98'
# env var ROCM_PKGS - A space-separated list of APT packages to install instead of the full ROCm installation.
#                     A very minimal installation is "rocm-hip-runtime-dev". For many C++ apps you may want to add "hipcub-dev"
#                     and "rocthrust-dev". If this env var is undefined or "", a full ROCm installation will be performed.

##### Helpers (copied from 'libutil.sh')

# Exits with status 1 if a bash library is executed.
function err_if_lib_is_executed() {
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf "ERROR: this bash library must be sourced!";
    exit 1
  fi
}

# Matches undefined vars, empty vars, and vars that contain spaces.
# Usage: `[ is_undefined_or_blank <VARNAME> ]`
# NOTE: A variable name must be passed.
function is_undefined_or_blank() {
  [ -z ${!1} ] && printf "1"
}

# Checks if a value equals 'true', '1', 'y', 'yes', or 'on' (case-insensitive).
# Usage: `[ $(is_true $<VALUE>) ]`
# NOTE: A value must be passed.
function is_true() {
  local lowered=$(printf "$1" | tr '[:upper:]' '[:lower:]')
  [[ "${lowered}" == "true" || "${lowered}" == "1" || "${lowered}" == "y"  || "${lowered}" == "yes" || "${lowered}" == "on"  ]] && printf "1"
}

# Matches undefined vars, empty vars, and vars that contain spaces.
# Usage: `err_if_undefined_or_blank <VARNAME>`
# NOTE: A variable name must be passed.
function err_if_undefined_or_blank() {
  if [ -z ${!1} ]; then
    printf "${libutil_error_prefix:-"ERROR"}: Variable '$1' is undefined or empty or contains only spaces."
    if [ -z ${WARN_ON_ERROR+x} ]; then
      exit 1
    fi
  fi
}

# Checks if a variable is undefined, if its value is empty or consists only of spaces, or if its value equals 'true', '1', 'y', 'yes', or 'on' (case-insensitive), or
# 'false', '0', 'n', 'no', or 'off' (case-insensitive).
#
# This can be used for scenarios where an undefined or empty variable or one that contains only spaces also implies a 'false' (or 'true') value.
#
# Usage: `err_if_neither_undefined_nor_blank_nor_boolean <VARNAME>`
# NOTE: A variable name must be passed.
function err_if_neither_undefined_nor_blank_nor_boolean() {
  if [ -z ${!1} ]; then
    return
  fi
  local lowered=$(printf "${!1}" | tr '[:upper:]' '[:lower:]')
  if [[ "${lowered}" == "true" || "${lowered}" == "1" || "${lowered}" == "y"  || "${lowered}" == "yes" || "${lowered}" == "on"  ]]; then
    : # nop
  elif [[ "${lowered}" == "false" || "${lowered}" == "0" || "${lowered}" == "n"  || "${lowered}" == "no" || "${lowered}" == "off"  ]]; then
    : # nop
  else
    printf "${libutil_error_prefix:-"ERROR"}: Argument '$1' is defined, not empty, does not solely consist of spaces, and is no boolean value; allowed boolean expressions: 'true' / 'false', '1' / '0', 'y' / 'n', 'yes' / 'no', 'on' / 'off' (case-insensitive)."
    if [ -z ${WARN_ON_ERROR+x} ]; then
      exit 1
    fi
  fi
}

##### Main

err_if_lib_is_executed

# Prints location of the 'rocm_version.h' file.
function __get_rocm_version_header() {
  local rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
  err_if_undefined_or_blank rocm_version_h
  printf ${rocm_version_h}
}

# Prints the HIP version.
# Example: 6.2.41133-dd7f95766
function get_hip_version() {
  hipconfig --version
}

# Prints '1' if the hipconfig tool was found.
# Prints nothing otherwise.
# NOTE: Expected to be used as follows: ``[ $(is_hip_installed) ]``.
function is_hip_installed() {
  local hip_version=$(get_hip_version)
  [ ! $(is_undefined_or_blank hip_version) ] && printf "1"
}

# Prints '1' if the 'rocm_version.h' header file was found.
# Prints nothing otherwise.
# NOTE: Expected to be used as follows: ``[ $(is_rocm_installed) ]``.
function is_rocm_installed() {
  if [ $(is_hip_installed) ]; then
    local rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
    [ ! $(is_undefined_or_blank rocm_version_h) ] && printf "1"
  fi
}

# Prints the >>X<<.Y.Z of the ROCm version X.Y.Z.
function get_rocm_version_major() {
  local rocm_version_h=$(__get_rocm_version_header)
  grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+"
}

# Prints the X.>>Y<<.Z of the ROCm version X.Y.Z.
function get_rocm_version_minor() {
  local rocm_version_h=$(__get_rocm_version_header)
  grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+"
}

# Prints the X.Y.>>Z<< of the ROCm version X.Y.Z.
function get_rocm_version_patch() {
  local rocm_version_h=$(__get_rocm_version_header)
  grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+"
}

function __make_short_version() {
  local major=${1}
  local minor=${2}
  local patch=${3}
  if [[ "${patch}" == "0" ]]; then
    printf "${major}.${minor}"
  else
    printf "${major}.${minor}.${patch}"
  fi
}

# Prints X.Y if the last number Z in the ROCm version X.Y.Z is 0,
# prints X.Y.Z otherwise.
function get_rocm_version_short() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  __make_short_version ${major} ${minor} ${patch}
}

# Prints the full ROCm version X.Y.Z.
function get_rocm_version() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  printf "${major}.${minor}.${patch}"
}

# Prints a key 'rocm-X-Y-Z'
function get_rocm_pip_key() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  printf "rocm-${major}-${minor}-${patch}"
}

function __get_rocm_version_linearized() {
  local major=$1
  local minor=$2
  local patch=$3
  let result=(major * 10000 + minor * 100 + patch)
  echo "${result}"
}

function get_rocm_version_linearized() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  __get_rocm_version_linearized ${major} ${minor} ${patch}
}

# Replaces the placeholders [MAJOR], [MINOR], [PATCH] by their corresponding
# value from the rocm version.
# USAGE: `var=$(insert_rocm_version "<text>")`
function insert_rocm_version() {
    local rocm_version_h=$(__get_rocm_version_header)
    local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
    local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
    local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")

    local result=${1/\[MAJOR\]/"${major}"}
    result=${result/\[MINOR\]/"${minor}"}
    result=${result/\[PATCH\]/"${patch}"}
    echo ${result}
}

function install_rocm_ubuntu() {
  err_if_undefined_or_blank ROCM_VERSION
  err_if_neither_undefined_nor_blank_nor_boolean AMDGPU_INSTALL_USE_INTERNAL
  err_if_neither_undefined_nor_blank_nor_boolean AMDGPU_INSTALL_INTERNAL_SPECIFY_BUILD

  if [[ "${ROCM_VERSION}" == "preinstalled" ]]; then
    printf "INFO: Use pre-installed ROCm."
    return
  fi

  export DEBIAN_FRONTEND=noninteractive

  # install OS packages
  err_if_neither_undefined_nor_blank_nor_boolean SKIP_UPDATE_PACKAGING_TOOLS

  if [ $(is_true ${SKIP_UPDATE_PACKAGING_TOOLS}) ]; then
    apt update -y
    apt dist-upgrade -y
  fi
  apt install -y --no-install-recommends wget grep sed

  # install latest rocm installation tool
  os_version_codename=$(grep VERSION_CODENAME /etc/os-release | sed "s,VERSION_CODENAME=,,")
  os_version_id=$(grep VERSION_ID /etc/os-release | sed 's,VERSION_ID="\([0-9\.]\+\)",\1,')
  rocm_ver_short=$(printf ${ROCM_VERSION} | sed "s,\([0-9]\+\.[0-9]\+\)\.0,\1,g")

  if [ $(is_true ${AMDGPU_INSTALL_USE_INTERNAL}) ]; then
    err_if_undefined_or_blank AMDGPU_INSTALL_INTERNAL_URL
    AMDGPU_INSTALL_INTERNAL_PREFIX=${AMDGPU_INSTALL_INTERNAL_PREFIX:-amd-nonfree-radeon_}
    # other option: 'amdgpu-install-internal_6.1-''
    wget -N -P /tmp/ ${AMDGPU_INSTALL_INTERNAL_URL}/${AMDGPU_INSTALL_INTERNAL_PREFIX}${os_version_id}-1_all.deb
    apt install -y --no-install-recommends /tmp/${AMDGPU_INSTALL_INTERNAL_PREFIX}${os_version_id}-1_all.deb
    if [ $(is_true ${AMDGPU_INSTALL_INTERNAL_SPECIFY_BUILD}) ]; then
      err_if_undefined_or_blank AMDGPU_INSTALL_INTERNAL_REPO_ARGS
      amdgpu-repo ${AMDGPU_INSTALL_INTERNAL_REPO_ARGS}
    fi
    # install ROCm (full vs individual packages, no driver)
    if [ $(is_undefined_or_blank ROCM_PKGS) ]; then
      amdgpu-install -y --usecase=rocm --no-dkms
    else
      apt install -y ${ROCM_PKGS}
    fi
  else # use public tool
    # wget -np -r -nH --cut-dirs=4 -A "amdgpu-install*deb" https://repo.radeon.com/amdgpu-install/latest/ubuntu/${os_version_codename}/
    # NOTE (Nov 2024): below seems to be more robust than above:
    wget -np -r -nH --cut-dirs=4 -A "amdgpu-install*deb" https://repo.radeon.com/amdgpu-install/${rocm_ver_short}/ubuntu/${os_version_codename}/
    apt install -y --no-install-recommends ./amdgpu-install_*.deb
    rm ./amdgpu-install_*.deb
    if (( $(__get_rocm_version_linearized ${ROCM_VERSION//./ }) < 60300 )); then
      printf "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/${rocm_ver_short} ${os_version_codename} main" | tee /etc/apt/sources.list.d/rocm.list
      printf -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600
    fi
    # install ROCm (full vs individual packages, no driver)
    apt update
    if [ $(is_undefined_or_blank ROCM_PKGS) ]; then
      amdgpu-install -y --usecase=rocm --rocmrelease=${ROCM_VERSION} --no-dkms
    else
      apt install -y ${ROCM_PKGS}
    fi
  fi
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib
}

