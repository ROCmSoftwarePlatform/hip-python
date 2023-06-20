# AMD_COPYRIGHT

__author__ = "AMD_AUTHOR"

import os, sys

from setuptools import Extension, setup
from Cython.Build import cythonize

ROCM_PATH=os.environ.get("ROCM_PATH", "/opt/rocm")
HIP_PLATFORM = os.environ.get("HIP_PLATFORM", "amd")

if HIP_PLATFORM not in ("amd", "hcc"):
    raise RuntimeError("Currently only HIP_PLATFORM=amd is supported")

def create_extension(name, sources):
    global ROCM_PATH
    global HIP_PLATFORM
    rocm_inc = os.path.join(ROCM_PATH,"include")
    rocm_lib_dir = os.path.join(ROCM_PATH,"lib")
    rocm_libs = ["amdhip64"]
    platform = HIP_PLATFORM.upper()
    cflags = ["-D", f"__HIP_PLATFORM_{platform}__"]
 
    return Extension(
        name,
        sources=sources,
        include_dirs=[rocm_inc],
        library_dirs=[rocm_lib_dir],
        libraries=rocm_libs,
        language="c",
        extra_compile_args=cflags,
    )

setup(
  ext_modules = cythonize(
    [create_extension("ccuda_stream", ["ccuda_stream.pyx"]),],
    compiler_directives=dict(language_level=3),
    compile_time_env=dict(HIP_PYTHON=True),
  )
)