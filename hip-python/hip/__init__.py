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

# This file has been autogenerated, do not modify.

__author__ = "Advanced Micro Devices, Inc. <hip-python.maintainer@amd.com>"

from ._version import *
ROCM_VERSION = 50700000
ROCM_VERSION_NAME = rocm_version_name = "5.7.0"
ROCM_VERSION_TUPLE = rocm_version_tuple = (5,7,0)
HIP_VERSION = 50731921
HIP_VERSION_NAME = hip_version_name = "5.7.31921-d1770ee1b"
HIP_VERSION_TUPLE = hip_version_tuple = (5,7,31921,"d1770ee1b")


from . import _util
try:
    from . import hip
except ImportError:
    pass # may have been excluded from build
try:
    from . import hiprtc
except ImportError:
    pass # may have been excluded from build
else: # no import error
    from . import hiprtc_pyext
    setattr(hiprtc,"ext",hiprtc_pyext)
try:
    from . import hipblas
except ImportError:
    pass # may have been excluded from build
try:
    from . import rccl
except ImportError:
    pass # may have been excluded from build
try:
    from . import hiprand
except ImportError:
    pass # may have been excluded from build
try:
    from . import hipfft
except ImportError:
    pass # may have been excluded from build
try:
    from . import hipsparse
except ImportError:
    pass # may have been excluded from build
try:
    from . import roctx
except ImportError:
    pass # may have been excluded from build
try:
    from . import hipsolver
except ImportError:
    pass # may have been excluded from build