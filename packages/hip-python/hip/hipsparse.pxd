# AMD_COPYRIGHT
from libc cimport stdlib
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
cimport hip._util.types
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip import hipError_t, _hipDataType__Base # PY import enums
from .hip cimport ihipStream_t, float2, double2 # C import structs/union types

from . cimport chipsparse
cdef class bsrsv2Info:
    cdef chipsparse.bsrsv2Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef bsrsv2Info from_ptr(chipsparse.bsrsv2Info* ptr, bint owner=*)
    @staticmethod
    cdef bsrsv2Info from_pyobj(object pyobj)


cdef class bsrsm2Info:
    cdef chipsparse.bsrsm2Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef bsrsm2Info from_ptr(chipsparse.bsrsm2Info* ptr, bint owner=*)
    @staticmethod
    cdef bsrsm2Info from_pyobj(object pyobj)


cdef class bsrilu02Info:
    cdef chipsparse.bsrilu02Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef bsrilu02Info from_ptr(chipsparse.bsrilu02Info* ptr, bint owner=*)
    @staticmethod
    cdef bsrilu02Info from_pyobj(object pyobj)


cdef class bsric02Info:
    cdef chipsparse.bsric02Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef bsric02Info from_ptr(chipsparse.bsric02Info* ptr, bint owner=*)
    @staticmethod
    cdef bsric02Info from_pyobj(object pyobj)


cdef class csrsv2Info:
    cdef chipsparse.csrsv2Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef csrsv2Info from_ptr(chipsparse.csrsv2Info* ptr, bint owner=*)
    @staticmethod
    cdef csrsv2Info from_pyobj(object pyobj)


cdef class csrsm2Info:
    cdef chipsparse.csrsm2Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef csrsm2Info from_ptr(chipsparse.csrsm2Info* ptr, bint owner=*)
    @staticmethod
    cdef csrsm2Info from_pyobj(object pyobj)


cdef class csrilu02Info:
    cdef chipsparse.csrilu02Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef csrilu02Info from_ptr(chipsparse.csrilu02Info* ptr, bint owner=*)
    @staticmethod
    cdef csrilu02Info from_pyobj(object pyobj)


cdef class csric02Info:
    cdef chipsparse.csric02Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef csric02Info from_ptr(chipsparse.csric02Info* ptr, bint owner=*)
    @staticmethod
    cdef csric02Info from_pyobj(object pyobj)


cdef class csrgemm2Info:
    cdef chipsparse.csrgemm2Info* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef csrgemm2Info from_ptr(chipsparse.csrgemm2Info* ptr, bint owner=*)
    @staticmethod
    cdef csrgemm2Info from_pyobj(object pyobj)


cdef class pruneInfo:
    cdef chipsparse.pruneInfo* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef pruneInfo from_ptr(chipsparse.pruneInfo* ptr, bint owner=*)
    @staticmethod
    cdef pruneInfo from_pyobj(object pyobj)


cdef class csru2csrInfo:
    cdef chipsparse.csru2csrInfo* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef csru2csrInfo from_ptr(chipsparse.csru2csrInfo* ptr, bint owner=*)
    @staticmethod
    cdef csru2csrInfo from_pyobj(object pyobj)


cdef class hipsparseSpGEMMDescr:
    cdef chipsparse.hipsparseSpGEMMDescr* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipsparseSpGEMMDescr from_ptr(chipsparse.hipsparseSpGEMMDescr* ptr, bint owner=*)
    @staticmethod
    cdef hipsparseSpGEMMDescr from_pyobj(object pyobj)


cdef class hipsparseSpSVDescr:
    cdef chipsparse.hipsparseSpSVDescr* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipsparseSpSVDescr from_ptr(chipsparse.hipsparseSpSVDescr* ptr, bint owner=*)
    @staticmethod
    cdef hipsparseSpSVDescr from_pyobj(object pyobj)


cdef class hipsparseSpSMDescr:
    cdef chipsparse.hipsparseSpSMDescr* _ptr
    cdef bint ptr_owner
    cdef Py_buffer _py_buffer
    cdef bint _py_buffer_acquired

    @staticmethod
    cdef hipsparseSpSMDescr from_ptr(chipsparse.hipsparseSpSMDescr* ptr, bint owner=*)
    @staticmethod
    cdef hipsparseSpSMDescr from_pyobj(object pyobj)