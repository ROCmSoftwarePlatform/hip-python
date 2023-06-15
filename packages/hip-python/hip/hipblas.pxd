# AMD_COPYRIGHT
from libc cimport stdlib
from libc cimport string
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip cimport ihipStream_t

from . cimport chipblas
cdef class hipblasBfloat16:
    cdef chipblas.hipblasBfloat16* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasBfloat16 from_ptr(chipblas.hipblasBfloat16* ptr, bint owner=*)
    @staticmethod
    cdef hipblasBfloat16 from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasBfloat16** ptr)
    @staticmethod
    cdef hipblasBfloat16 new()
    @staticmethod
    cdef hipblasBfloat16 from_value(chipblas.hipblasBfloat16 other)


cdef class hipblasComplex:
    cdef chipblas.hipblasComplex* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasComplex from_ptr(chipblas.hipblasComplex* ptr, bint owner=*)
    @staticmethod
    cdef hipblasComplex from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasComplex** ptr)
    @staticmethod
    cdef hipblasComplex new()
    @staticmethod
    cdef hipblasComplex from_value(chipblas.hipblasComplex other)


cdef class hipblasDoubleComplex:
    cdef chipblas.hipblasDoubleComplex* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipblasDoubleComplex from_ptr(chipblas.hipblasDoubleComplex* ptr, bint owner=*)
    @staticmethod
    cdef hipblasDoubleComplex from_pyobj(object pyobj)
    @staticmethod
    cdef __allocate(chipblas.hipblasDoubleComplex** ptr)
    @staticmethod
    cdef hipblasDoubleComplex new()
    @staticmethod
    cdef hipblasDoubleComplex from_value(chipblas.hipblasDoubleComplex other)
