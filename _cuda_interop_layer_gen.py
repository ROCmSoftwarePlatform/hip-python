import os
import textwrap

from _codegen.cython import (
    CythonPackageGenerator,
)

from _codegen.tree import (
    MacroDefinition,
    Function,
    Typedef,
    Enum,
    AnonymousEnum,
    Record,
    FunctionPointer,
)

try:
    # package to calculate word distances
    import Levenshtein

    HAVE_LEVENSHTEIN = True
except ImportError:
    HAVE_LEVENSHTEIN = False

def generate_cuda_interop_package_files(
    cuda_pkg_name: str, generator: CythonPackageGenerator, 
    hip2cuda: dict,
    warn: bool = True
):
    global HAVE_LEVENSHTEIN
    pkg_dir = "cuda"
    output_dir = os.path.join("packages","hip-python-as-nv",pkg_dir)
    indent = " " * 4
    pkg_name = generator.pkg_name
    cpkg_name = f"hip.c{pkg_name}"
    pkg_cimport_name = f"hip.{pkg_name}" 
    backend = generator.backend

    c_interface_decl_part = [
        textwrap.dedent(
            f"""\
            # AMD_COPYRIGHT
            
            cimport {cpkg_name}
            """
        ),
    ]
    python_interface_decl_part = [
        textwrap.dedent(
            f"""\
            # AMD_COPYRIGHT

            __author__ = "AMD_AUTHOR"

            cimport {cpkg_name}
            cimport {pkg_cimport_name}
            """
        ),
        f"cimport {pkg_dir}.c{cuda_pkg_name}",  # for checking compiler errors
    ]
    python_interface_impl_part = [
        textwrap.dedent(
            f"""\
            # AMD_COPYRIGHT

            __author__ = "AMD_AUTHOR"

            import os
            import enum

            import hip.{pkg_name}
            {pkg_name} = hip.{pkg_name} # makes {pkg_name} types and routines accessible without import
                                        # allows checks such as `hasattr(cuda.{cuda_pkg_name},"{pkg_name}")`

            HIP_PYTHON_MOD = {pkg_name}
            globals()["HIP_PYTHON"] = True
            """
        ),
        textwrap.dedent(
            """\
            def _hip_python_get_bool_environ_var(env_var, default):
                yes_vals = ("true", "1", "t", "y", "yes")
                no_vals = ("false", "0", "f", "n", "no")
                value = os.environ.get(env_var, default).lower()
                if value in yes_vals:
                    return True
                elif value in no_vals:
                    return False
                else:
                    allowed_vals = ", ".join([f"'{a}'" for a in (list(yes_vals)+list(no_vals))])
                    raise RuntimeError(f"value of '{env_var}' must be one of (case-insensitive): {allowed_vals}")
            """
        ),
    ]

    # impl part is always empty
    def warn_(hip_name):
        global HAVE_LEVENSHTEIN
        global HIP_2_CUDA
        msg = f"hipify-perl: no CUDA symbol found for HIP symbol {hip_name}"
        if HAVE_LEVENSHTEIN:
            cutoff = 0.9
            candidates = []
            for other_hip_name in HIP_2_CUDA:
                if (
                    Levenshtein.ratio(
                        hip_name,
                        other_hip_name,
                        processor=lambda tk: tk.lower(),  # do everything in lowercase
                        score_cutoff=cutoff,  # everything below cutoff is set to 0
                    )
                    > 0
                ):
                    candidates.append(other_hip_name)
            candidates_formatted = ", ".join(["'" + c + "'" for c in candidates])
            msg += f"; most similar hipify-perl HIP symbols (Levenshtein ratio > {cutoff}): [{candidates_formatted}]"
        warnings.warn(msg)

    def handle_enum_(node, hip_name, cuda_name):
        global HIP_2_CUDA
        nonlocal indent
        nonlocal c_interface_decl_part
        nonlocal python_interface_impl_part
        enum = node if isinstance(node, Enum) else node.lookup_innermost_type()
        c_constants = []
        python_constants = []
        for child_cursor in enum.cursor.get_children():
            hip_constant_name = child_cursor.spelling
            # append hip constant too, to help workarounds
            c_constants.append(
                f"from {cpkg_name} cimport {hip_constant_name}"
            )
            python_constants.append(
                f"{hip_constant_name} = {cpkg_name}.{hip_constant_name}"
            )
            if hip_constant_name in HIP_2_CUDA:
                for cuda_constant_name in HIP_2_CUDA[hip_constant_name]:
                    c_constants.append(
                        f"from {cpkg_name} cimport {hip_constant_name} as {cuda_constant_name}"
                    )
                    python_constants.append(
                        f"{cuda_constant_name} = {cpkg_name}.{hip_constant_name}"
                    )
            else:
                warn_(hip_constant_name)
        if isinstance(node, AnonymousEnum):  # cannot be typedefed
            python_interface_impl_part += python_constants
        else:
            python_enum_metaclass_name = f"_{cuda_name}_EnumMeta"
            python_enum_hallucinate_var_name = (
                f"HIP_PYTHON_{cuda_name}_HALLUCINATE"
            )
            python_enum_metaclass = textwrap.dedent(
                f"""\
                
                {python_enum_hallucinate_var_name} = _hip_python_get_bool_environ_var("{python_enum_hallucinate_var_name}","false")

                class {python_enum_metaclass_name}(enum.EnumMeta):
                
                    def __getattribute__(cls,name):
                        global _get_hip_name
                        global {python_enum_hallucinate_var_name}
                        try:
                            result = super().__getattribute__(name)
                            return result
                        except AttributeError as ae:
                            if not {python_enum_hallucinate_var_name}:
                                raise ae
                            else:
                                used_vals = list(cls._value2member_map_.keys())
                                if not len(used_vals):
                                    raise ae
                                new_val = min(used_vals)
                                while new_val in used_vals: # find a free enum value
                                    new_val += 1
                                
                                class HallucinatedEnumConstant():
                                    \"""Mimicks the orginal enum type this is derived from.
                                    \"""
                                    def __init__(self):
                                        pass
                                    
                                    @property
                                    def name(self):
                                        return self._name_
                                    
                                    @property
                                    def value(self):
                                        return self._value_

                                    def __eq__(self,other):
                                        if isinstance(other,{pkg_name}.{hip_name}):
                                            return self.value == other.value
                                        return False

                                    def __repr__(self):
                                        \"""Mimicks enum.Enum.__repr__\"""
                                        return "<%s.%s: %r>" % (
                                                self.__class__._name_, self._name_, self._value_)
                                                
                                    def __str__(self):
                                        \"""Mimicks enum.Enum.__str__\"""
                                        return "%s.%s" % (self.__class__._name_, self._name_)

                                    def __hash__(self):
                                        return hash(str(self))

                                    @property
                                    def __class__(self):
                                        \"""Make this type appear as a constant of the actual 
                                        CUDA enum type in isinstance checks.
                                        \"""
                                        return {cuda_name}
                                setattr(HallucinatedEnumConstant,"_name_",name)
                                setattr(HallucinatedEnumConstant,"_value_",new_val)
                                return HallucinatedEnumConstant()
                """
            )
            python_enum_class = textwrap.dedent(
                f"""
                class {cuda_name}({pkg_name}.{enum.python_base_class_name},metaclass={python_enum_metaclass_name}):
                """
            )
            python_enum_class += textwrap.indent("\n".join(python_constants), indent)

            python_interface_impl_part.append(python_enum_metaclass)
            python_interface_impl_part.append(python_enum_class)

        if isinstance(node, Enum) and i == 0:
            if not isinstance(node,AnonymousEnum):
                cython_enum = f"from {cpkg_name} cimport {hip_name} as {cuda_name}"
                c_interface_decl_part.append(cython_enum)
            c_interface_decl_part += c_constants
        else:  # if it is a typedef or there are multiple CUDA names
            hip_underlying_type_name = enum.name
            if hip_underlying_type_name in HIP_2_CUDA:
                cuda_underlying_type_name = HIP_2_CUDA[hip_underlying_type_name][
                    0
                ]  # take first
                cython_enum = f"ctypedef {cuda_underlying_type_name} {cuda_name}"
                c_interface_decl_part.append(cython_enum)
            else:
                warn_(hip_underlying_type_name)

    # main loop over nodes
    for node in backend.walk_filtered_nodes():
        hip_name = node.name
        if isinstance(node, AnonymousEnum):
            # Anonymous enums won't have a different CUDA name but their constants might
            handle_enum_(node, hip_name, hip_name)  # hip_name is auto_generated in this case
        if hip_name in hip2cuda:
            cuda_names = hip2cuda[hip_name]
            for i, cuda_name in enumerate(cuda_names):
                if isinstance(node, Enum) or (
                    isinstance(node, Typedef)
                    and node.is_pointer_to_enum(degree=(0, -1))
                ):
                    # enums require special care as they are modelled as "class <type>"
                    # and not as "cdef class" in the Python interface, just like in CUDA Python.
                    handle_enum_(node, hip_name, cuda_name)
                elif (
                    isinstance(
                        node,
                        (
                            MacroDefinition,
                            Function,
                        ),
                    )
                    or isinstance(node, Typedef)
                    and node.is_pointer_to_record(degree=(0, -1))
                ):
                    # These are Python objects/functions in the Python interface
                    if i == 0:
                        c_interface_decl_part.append(
                            f"from {cpkg_name} cimport {hip_name}"
                        )
                    c_interface_decl_part.append(
                        f"from {cpkg_name} cimport {hip_name} as {cuda_name}"
                    )
                    python_interface_impl_part.append(
                        f"{cuda_name} = {pkg_name}.{hip_name}"
                    )
                elif isinstance(node, Typedef) and (
                    node.is_pointer_to_basic_type(degree=(0, -1))
                    or node.is_pointer_to_void(degree=(0, -1))
                ):
                    canonical_type = node.cursor.type.get_canonical().spelling
                    c_interface_decl_part.append(
                        f"ctypedef {canonical_type} {cuda_name}"
                    )
                elif isinstance(node, (FunctionPointer, Record)):
                    # These are cdef classes ("extension types").
                    # So Python interface declaration must be cimported.
                    # and a subclass needs to be created to define a Python object. (TODO other options?)
                    if i == 0:
                        c_interface_decl_part.append(
                            f"from {cpkg_name} cimport {hip_name}"
                        )
                    c_interface_decl_part.append(
                        f"from {cpkg_name} cimport {hip_name} as {cuda_name}"
                    )
                    #
                    if i == 0 and hip_name not in cuda_names:
                        python_interface_decl_part.append(
                            f"from {pkg_cimport_name} cimport {hip_name} # here"
                        ) 
                    cdef_subclass = f"cdef class {cuda_name}({pkg_cimport_name}.{hip_name}):\n{indent}pass"
                    python_interface_decl_part.append(cdef_subclass)
                    python_interface_impl_part.append(cdef_subclass)
        elif warn:
            warn_(hip_name)

    python_interface_decl_path = os.path.join(output_dir, f"{cuda_pkg_name}.pxd")
    python_interface_impl_path = os.path.join(output_dir, f"{cuda_pkg_name}.pyx")
    c_interface_decl_path = os.path.join(output_dir, f"c{cuda_pkg_name}.pxd")
    with open(c_interface_decl_path, "w") as outfile:
        outfile.write("\n".join(c_interface_decl_part))
    with open(python_interface_decl_path, "w") as outfile:
        outfile.write("\n".join(python_interface_decl_part))
    with open(python_interface_impl_path, "w") as outfile:
        outfile.write("\n".join(python_interface_impl_part))