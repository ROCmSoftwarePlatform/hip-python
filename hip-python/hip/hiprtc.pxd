# MIT License
#
# Copyright (c) 2021-2024 Advanced Micro Devices, Inc.
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

cimport hip._util.types
from hip cimport chiprtc

cdef class ihiprtcLinkState(hip._util.types.Pointer):
    cdef bint _is_ptr_owner

    cdef chiprtc.ihiprtcLinkState* getElementPtr(self)

    @staticmethod
    cdef ihiprtcLinkState fromPtr(void* ptr, bint owner=*)
    @staticmethod
    cdef ihiprtcLinkState fromPyobj(object pyobj)


cdef class _hiprtcProgram(hip._util.types.Pointer):
    cdef bint _is_ptr_owner

    cdef chiprtc._hiprtcProgram* getElementPtr(self)

    @staticmethod
    cdef _hiprtcProgram fromPtr(void* ptr, bint owner=*)
    @staticmethod
    cdef _hiprtcProgram fromPyobj(object pyobj)
