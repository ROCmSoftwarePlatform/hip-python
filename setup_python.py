# AMD_COPYRIGHT

"""This is the project's setup script.

After pointing this setup script to an HIP installation,
it generates Cython files and compiles them to Python
modules. The generated Cython declaration files can be used
by Cython users of this project.
"""

__author__ = "AMD_AUTHOR"

import os
import warnings
import enum
import textwrap
import argparse

import _controls
import _cuda_interop_layer_gen

from _codegen.cython import (
    CythonPackageGenerator,
    DEFAULT_PTR_COMPLICATED_TYPE_HANDLER,
)

from _codegen.cparser import TypeHandler

TypeCategory = TypeHandler.TypeCategory

from _codegen.tree import (
    Node,
    MacroDefinition,
    Parm,
)

from _parse_hipify_perl import parse_hipify_perl

def parse_options():
    global ROCM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS
    global LIBS
    global HIP_2_CUDA

    def get_bool_environ_var(env_var, default):
        yes_vals = ("true", "1", "t", "y", "yes")
        no_vals = ("false", "0", "f", "n", "no")
        value = os.environ.get(env_var, default).lower()
        if value in yes_vals:
            return True
        elif value in no_vals:
            return False
        else:
            allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals) + list(no_vals))])
            raise RuntimeError(
                f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}"
            )

    parser = argparse.ArgumentParser(description="Generator for HIP Python packages")
    parser.add_argument("--rocm-path",type=str,required=False,dest="rocm_path",
                        help="The ROCm installation directory. Can be set via environment variables 'ROCM_PATH', 'ROCM_HOME' too.")
    parser.add_argument("--platform",type=str,required=False,dest="platform",
                        help="The HIP platform, 'amd' or 'nvidia'. Can be set via environment variable HIP_PLATFORM too.")
    parser.add_argument("--clang-resource-dir",required=False,dest="clang_resource_dir",
                        help="The clang resource directory. Can also be set via environment variable 'HIP_PYTHON_CLANG_RES_DIR'.")
    parser.add_argument("--libs",type=str,required=False,dest="libs",
                        help="The ROCm libaries to generate interfaces for, as comma-separated list, e.g. 'hip,hiprtc'. Pass '*' to generate all, pass '' to generate none.")
    parser.add_argument("--no-rt-linking",required=False,action="store_false",dest="runtime_linking",
                        help="If HIP libraries should not be linked at runtime by the HIP Python modules.")
    parser.add_argument("-v","--verbose",required=False,action="store_true",dest="verbose",
                        default=False,
                        help="Verbose output.")
    parser.set_defaults(
        rocm_path=os.environ.get("ROCM_PATH", os.environ.get("ROCM_HOME",None)),
        platform=os.environ.get("HIP_PLATFORM","amd"),
        clang_resource_dir=os.environ.get("HIP_PYTHON_CLANG_RES_DIR", None),
        libs="*",
        runtime_linking=get_bool_environ_var("HIP_PYTHON_RUNTIME_LINKING","true"),
        verbose=False,
    )
    args = parser.parse_args()

    RUNTIME_LINKING = args.runtime_linking
    LIBS = args.libs

    if not args.rocm_path:
        raise RuntimeError("ROCm path is not set")
    ROCM_INC = os.path.join(args.rocm_path, "include")

    hipify_perl_path = os.path.join(args.rocm_path, "bin", "hipify-perl")
    (_, HIP_2_CUDA) = parse_hipify_perl(hipify_perl_path)

    if args.platform not in ("amd", "hcc"):
        raise RuntimeError("Currently only platform 'amd' is supported")

    class HipPlatform(enum.IntEnum):
        AMD = 0
        NVIDIA = 1

        @staticmethod
        def from_string(key: str):
            valid_inputs = ("amd", "hcc", "nvidia", "nvcc")
            key = key.lower()
            if key in valid_inputs[0:2]:
                return HipPlatform.AMD
            elif key in valid_inputs[2:4]:
                return HipPlatform.NVIDIA
            else:
                raise ValueError(
                    f"Input must be one of: {','.join(valid_inputs)} (any case)"
                )

        @property
        def cflags(self):
            return ["-D", f"__HIP_PLATFORM_{self.name}__"]

    hip_platform = HipPlatform.from_string(args.platform)

    GENERATOR_ARGS = hip_platform.cflags + [f"-I{ROCM_INC}"]
    if not args.clang_resource_dir:
        raise RuntimeError(
            textwrap.dedent(
                """\
            Clang resource directory is not set.
            
            Hint: If `clang` is in the PATH, you can 
            run `clang -print-resource-dir` to obtain the path to
            the resource directory.

            Hint: If you have the HIP SDK installed, you have `amdclang` installed in
            `ROCM_PATH/bin/`. You can use it to run the above command too.
            
            Hint: If you have the HIP SDK installed, the last include folder listed in ``hipconfig --cpp_config``
            points to the `amdclang` compiler's resource dir too.
            """
            )
        )
    GENERATOR_ARGS += ["-resource-dir", args.clang_resource_dir]

# hip
def generate_hip_package_files():
    global ROCM_INC
    global RUNTIME_LINKING
    global GENERATOR_ARGS

    global HIP_VERSION_MAJOR
    global HIP_VERSION_MINOR
    global HIP_VERSION_PATCH
    global HIP_VERSION_GITHASH

    def toclassname(name: str):
        return name[0].upper() + name[1:]

    def hip_ptr_complicated_type_handler(parm: Node):
        if (parm.parent.name, parm.name) == ("hipModuleLaunchKernel", "extra"):
            return f"hip._hip_helpers.{toclassname(parm.parent.name)}_{parm.name}"
        if (parm.parent.name, parm.name) in (
            ("hipMalloc", "ptr"),
            ("hipExtMallocWithFlags", "ptr"),
            ("hipMallocManaged", "dev_ptr"),
            ("hipMallocAsync", "dev_ptr"),
            ("hipMallocFromPoolAsync", "dev_ptr"),
        ):
            if parm.parent.name == "hipExtMallocWithFlags":
                size = "sizeBytes"
            else:
                size = "size"
            parm.parent.python_body_prepend_before_return(
                f"{parm.name}.configure(_force=True,shape=(cpython.long.PyLong_FromUnsignedLong({size}),))"
            )
            return "hip._util.types.DeviceArray"
        return DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    generator = CythonPackageGenerator(
        "hip",
        ROCM_INC,
        "hip/hip_runtime.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libamdhip64.so",
        node_filter=_controls.hip.node_filter,
        ptr_parm_intent=_controls.hip.ptr_parm_intent,
        ptr_rank=_controls.hip.ptr_rank,
        ptr_complicated_type_handler=hip_ptr_complicated_type_handler,
        macro_type=_controls.hip.macro_type,
        cflags=GENERATOR_ARGS,
    )
    generator.python_interface_impl_preamble += textwrap.dedent(
        """\
    cimport hip._hip_helpers
    """
    )
    HIP_VERSION_MAJOR = 0
    HIP_VERSION_MINOR = 0
    HIP_VERSION_PATCH = 0
    HIP_VERSION_GITHASH = ""
    for node in generator.backend.root.walk():
        if isinstance(node, MacroDefinition):
            last_token = list(node.cursor.get_tokens())[-1].spelling
            if node.name == "HIP_VERSION_MAJOR":
                HIP_VERSION_MAJOR = int(last_token)
            elif node.name == "HIP_VERSION_MINOR":
                HIP_VERSION_MINOR = int(last_token)
            elif node.name == "HIP_VERSION_PATCH":
                HIP_VERSION_PATCH = int(last_token)
            elif node.name == "HIP_VERSION_GITHASH":
                HIP_VERSION_GITHASH = last_token.strip('"')
    _cuda_interop_layer_gen.generate_cuda_interop_package_files("cuda", generator, HIP_2_CUDA)
    _cuda_interop_layer_gen.generate_cuda_interop_package_files(
        "cudart", generator, HIP_2_CUDA, warn=False
    )  # already warned before, regenerate to have correctly named pxd/pyx files too. Could be done via symlinks & __init__.py mod too.
    return generator

# hiprtc
def generate_hiprtc_package_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    def hiprtc_ptr_complicated_type_handler(parm: Parm):
        list_of_str_parms = (
            ("hiprtcCompileProgram", "options"),
            ("hiprtcCreateProgram", "headers"),
            ("hiprtcCreateProgram", "includeNames"),
        )
        if (parm.parent.name, parm.name) in list_of_str_parms:
            return "hip._util.types.ListOfBytes"
        return DEFAULT_PTR_COMPLICATED_TYPE_HANDLER(parm)

    generator = CythonPackageGenerator(
        "hiprtc",
        ROCM_INC,
        "hip/hiprtc.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libhiprtc.so",
        node_filter=_controls.hiprtc.node_filter,
        ptr_parm_intent=_controls.hiprtc.ptr_parm_intent,
        ptr_rank=_controls.hiprtc.ptr_rank,
        ptr_complicated_type_handler=hiprtc_ptr_complicated_type_handler,
        cflags=GENERATOR_ARGS,
    )
    _cuda_interop_layer_gen.generate_cuda_interop_package_files("nvrtc", generator, HIP_2_CUDA)
    return generator

# hipblas
def generate_hipblas_package_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonPackageGenerator(
        "hipblas",
        ROCM_INC,
        "hipblas/hipblas.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libhipblas.so",
        node_filter=_controls.hipblas.node_filter,
        ptr_parm_intent=_controls.hipblas.ptr_parm_intent,
        ptr_rank=_controls.hipblas.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t
    """
    )
    return generator


# rccl
def generate_rccl_package_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonPackageGenerator(
        "rccl",
        ROCM_INC,
        "rccl/rccl.h",
        runtime_linking=RUNTIME_LINKING,
        dll="librccl.so",
        node_filter=_controls.rccl.node_filter,
        macro_type=_controls.rccl.macro_type,
        ptr_parm_intent=_controls.rccl.ptr_parm_intent,
        ptr_rank=_controls.rccl.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t
    """
    )
    return generator


# hiprand
def generate_hiprand_package_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonPackageGenerator(
        "hiprand",
        ROCM_INC,
        "hiprand/hiprand.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libhiprand.so",
        node_filter=_controls.hiprand.node_filter,
        macro_type=_controls.hiprand.macro_type,
        ptr_parm_intent=_controls.hiprand.ptr_parm_intent,
        ptr_rank=_controls.hiprand.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t
    """
    )
    return generator


# hipfft
def generate_hipfft_package_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonPackageGenerator(
        "hipfft",
        ROCM_INC,
        "hipfft/hipfft.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libhipfft.so",
        node_filter=_controls.hipfft.node_filter,
        macro_type=_controls.hipfft.macro_type,
        ptr_parm_intent=_controls.hipfft.ptr_parm_intent,
        ptr_rank=_controls.hipfft.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip cimport ihipStream_t, float2, double2
    """
    )
    return generator


# hipsparse
def generate_hipsparse_package_files():
    global ROCM_INC
    global GENERATOR_ARGS
    global RUNTIME_LINKING

    generator = CythonPackageGenerator(
        "hipsparse",
        ROCM_INC,
        "hipsparse/hipsparse.h",
        runtime_linking=RUNTIME_LINKING,
        dll="libhipsparse.so",
        node_filter=_controls.hipsparse.node_filter,
        macro_type=_controls.hipsparse.macro_type,
        ptr_parm_intent=_controls.hipsparse.ptr_parm_intent,
        ptr_rank=_controls.hipsparse.ptr_rank,
        cflags=GENERATOR_ARGS,
    )
    generator.c_interface_decl_preamble += textwrap.dedent(
        """\
    from .chip cimport *
    """
    )
    generator.python_interface_decl_preamble += textwrap.dedent(
        """\
    from .hip import hipError_t, _hipDataType__Base # PY import enums
    from .hip cimport ihipStream_t, float2, double2 # C import structs/union types
    """
    )
    return generator

if __name__ == "__main__":
    ROCM_INC = None
    RUNTIME_LINKING = None
    GENERATOR_ARGS = None
    LIBS = None
    HIP_2_CUDA = None

    HIP_VERSION_MAJOR = 0
    HIP_VERSION_MINOR = 0
    HIP_VERSION_PATCH = 0
    HIP_VERSION_GITHASH = ""

    parse_options()

    AVAILABLE_GENERATORS = dict(
        hip=generate_hip_package_files,
        hiprtc=generate_hiprtc_package_files,
        hipblas=generate_hipblas_package_files,
        rccl=generate_rccl_package_files,
        hiprand=generate_hiprand_package_files,
        hipfft=generate_hipfft_package_files,
        hipsparse=generate_hipsparse_package_files,
    )

    if len(LIBS.strip()):
        lib_names = (
            AVAILABLE_GENERATORS.keys()
            if LIBS == "*"
            else LIBS.split(",")
        )
    else:
        lib_names = []

    output_dir = os.path.join("packages","hip-python","hip")
    for entry in lib_names:
        libname = entry.strip()
        if libname not in AVAILABLE_GENERATORS:
            available_libs = ", ".join([f"'{a}'" for a in AVAILABLE_GENERATORS.keys()])
            msg = f"no codegenerator found for library '{libname}'; please choose from: {available_libs}, or '*', which implies that all code generators will be used."
            raise KeyError(msg)
        generator = AVAILABLE_GENERATORS[libname]()
        generator.write_package_files(output_dir=output_dir)

    HIP_VERSION_NAME = (
        f"{HIP_VERSION_MAJOR}.{HIP_VERSION_MINOR}.{HIP_VERSION_PATCH}-{HIP_VERSION_GITHASH}"
    )
    HIP_VERSION = (
        HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH
    )

    with open(os.path.join(output_dir,"__init__.py"), "w") as f:
        init_content = textwrap.dedent(
            f"""\
            from ._version import *
            HIP_VERSION = {HIP_VERSION}
            HIP_VERSION_NAME = hip_version_name = "{HIP_VERSION_NAME}"
            HIP_VERSION_TUPLE = hip_version_tuple = ({HIP_VERSION_MAJOR},{HIP_VERSION_MINOR},{HIP_VERSION_PATCH},"{HIP_VERSION_GITHASH}")

            from . import _util
            """
        )

        for pkg_name in AVAILABLE_GENERATORS.keys():
            init_content += f"from . import {pkg_name}\n"

        f.write(init_content)