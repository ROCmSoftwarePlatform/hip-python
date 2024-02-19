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

from libc cimport stdlib
from libc cimport string
from libc.stdint cimport *
cimport cpython.long
cimport cpython.buffer
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .hip cimport ihipStream_t

cimport hip._util.types
from hip cimport crccl

cdef class ncclComm(hip._util.types.Pointer):
    cdef bint _is_ptr_owner

    cdef crccl.ncclComm* getElementPtr(self)

    @staticmethod
    cdef ncclComm fromPtr(void* ptr, bint owner=*)
    @staticmethod
    cdef ncclComm fromPyobj(object pyobj)


cdef class ncclUniqueId(hip._util.types.Pointer):
    cdef bint _is_ptr_owner

    cdef crccl.ncclUniqueId* getElementPtr(self)

    @staticmethod
    cdef ncclUniqueId fromPtr(void* ptr, bint owner=*)
    @staticmethod
    cdef ncclUniqueId fromPyobj(object pyobj)
    @staticmethod
    cdef __allocate(void* ptr)
    @staticmethod
    cdef ncclUniqueId new()
    @staticmethod
    cdef ncclUniqueId fromValue(crccl.ncclUniqueId other)
