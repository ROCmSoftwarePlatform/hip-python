# AMD_COPYRIGHT
import cython
import ctypes
import enum
NCCL_MAJOR = crccl.NCCL_MAJOR

NCCL_MINOR = crccl.NCCL_MINOR

NCCL_PATCH = crccl.NCCL_PATCH

NCCL_SUFFIX = crccl.NCCL_SUFFIX

NCCL_VERSION_CODE = crccl.NCCL_VERSION_CODE

RCCL_BFLOAT16 = crccl.RCCL_BFLOAT16

RCCL_GATHER_SCATTER = crccl.RCCL_GATHER_SCATTER

RCCL_ALLTOALLV = crccl.RCCL_ALLTOALLV

RCCL_MULTIRANKPERGPU = crccl.RCCL_MULTIRANKPERGPU

NCCL_UNIQUE_ID_BYTES = crccl.NCCL_UNIQUE_ID_BYTES

cdef class ncclComm:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef ncclComm from_ptr(crccl.ncclComm* ptr, bint owner=False):
        """Factory function to create ``ncclComm`` objects from
        given ``crccl.ncclComm`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ncclComm wrapper = ncclComm.__new__(ncclComm)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef ncclComm from_pyobj(object pyobj):
        """Derives a ncclComm from a Python object.

        Derives a ncclComm from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``ncclComm`` reference, this method
        returns it directly. No new ``ncclComm`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ncclComm``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of ncclComm!
        """
        cdef ncclComm wrapper = ncclComm.__new__(ncclComm)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,ncclComm):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <crccl.ncclComm*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <crccl.ncclComm*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <crccl.ncclComm*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <crccl.ncclComm*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<ncclComm object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    @staticmethod
    def PROPERTIES():
        return []

    def __contains__(self,item):
        properties = self.PROPERTIES()
        return item in properties

    def __getitem__(self,item):
        properties = self.PROPERTIES()
        if isinstance(item,int):
            if item < 0 or item >= len(properties):
                raise IndexError()
            return getattr(self,properties[item])
        raise ValueError("'item' type must be 'int'")


ncclComm_t = ncclComm

cdef class ncclUniqueId:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef ncclUniqueId from_ptr(crccl.ncclUniqueId* ptr, bint owner=False):
        """Factory function to create ``ncclUniqueId`` objects from
        given ``crccl.ncclUniqueId`` pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``ptr``
        when the wrapper object is deallocated.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef ncclUniqueId wrapper = ncclUniqueId.__new__(ncclUniqueId)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef ncclUniqueId from_pyobj(object pyobj):
        """Derives a ncclUniqueId from a Python object.

        Derives a ncclUniqueId from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``ncclUniqueId`` reference, this method
        returns it directly. No new ``ncclUniqueId`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``ncclUniqueId``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of ncclUniqueId!
        """
        cdef ncclUniqueId wrapper = ncclUniqueId.__new__(ncclUniqueId)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,ncclUniqueId):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <crccl.ncclUniqueId*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <crccl.ncclUniqueId*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <crccl.ncclUniqueId*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                wrapper.ptr,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <crccl.ncclUniqueId*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
        # De-allocate if not null and flag is set
        if self._ptr is not NULL and self.ptr_owner is True:
            stdlib.free(self._ptr)
            self._ptr = NULL
    @staticmethod
    cdef __allocate(crccl.ncclUniqueId** ptr):
        ptr[0] = <crccl.ncclUniqueId*>stdlib.malloc(sizeof(crccl.ncclUniqueId))

        if ptr[0] is NULL:
            raise MemoryError
        # TODO init values, if present

    @staticmethod
    cdef ncclUniqueId new():
        """Factory function to create ncclUniqueId objects with
        newly allocated crccl.ncclUniqueId"""
        cdef crccl.ncclUniqueId* ptr
        ncclUniqueId.__allocate(&ptr)
        return ncclUniqueId.from_ptr(ptr, owner=True)
   
    def __init__(self,*args,**kwargs):
        ncclUniqueId.__allocate(&self._ptr)
        self.ptr_owner = True
        attribs = self.PROPERTIES()
        used_attribs = set()
        if len(args) > len(attribs):
            raise ValueError("More positional arguments specified than this type has properties.")
        for i,v in enumerate(args):
            setattr(self,attribs[i],v)
            used_attribs.add(attribs[i])
        valid_names = ", ".join(["'"+p+"'" for p in attribs])
        for k,v in kwargs.items():
            if k in used_attribs:
                raise KeyError(f"argument '{k}' has already been specified as positional argument.")
            elif k not in attribs:
                raise KeyError(f"'{k}' is no valid property name. Valid names: {valid_names}")
            setattr(self,k,v)
    
    @property
    def ptr(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __int__(self):
        return self.ptr
    def __repr__(self):
        return f"<ncclUniqueId object, self.ptr={self.ptr()}>"
    @property
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(self.ptr)
    def get_internal(self, i):
        """Get value of ``internal`` of ``self._ptr[i]``.
        """
        return self._ptr[i].internal
    # TODO add setters
    #def set_internal(self, i, char[128] value):
    #    """Set value ``internal`` of ``self._ptr[i]``.
    #    """
    #    self._ptr[i].internal = value
    @property
    def internal(self):
        return self.get_internal(0)
    # TODO add setters
    #@internal.setter
    #def internal(self, char[128] value):
    #    self.set_internal(0,value)

    @staticmethod
    def PROPERTIES():
        return ["internal"]

    def __contains__(self,item):
        properties = self.PROPERTIES()
        return item in properties

    def __getitem__(self,item):
        properties = self.PROPERTIES()
        if isinstance(item,int):
            if item < 0 or item >= len(properties):
                raise IndexError()
            return getattr(self,properties[item])
        raise ValueError("'item' type must be 'int'")


class ncclResult_t(enum.IntEnum):
    ncclSuccess = crccl.ncclSuccess
    ncclUnhandledCudaError = crccl.ncclUnhandledCudaError
    ncclSystemError = crccl.ncclSystemError
    ncclInternalError = crccl.ncclInternalError
    ncclInvalidArgument = crccl.ncclInvalidArgument
    ncclInvalidUsage = crccl.ncclInvalidUsage
    ncclRemoteError = crccl.ncclRemoteError
    ncclNumResults = crccl.ncclNumResults
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def ncclGetVersion():
    """! @brief Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
    @details This integer is coded with the MAJOR, MINOR and PATCH level of the
    NCCL library
    """
    cdef int version
    _ncclGetVersion__retval = ncclResult_t(crccl.ncclGetVersion(&version))    # fully specified
    return (_ncclGetVersion__retval,version)


@cython.embedsignature(True)
def pncclGetVersion():
    """@cond include_hidden
    """
    cdef int version
    _pncclGetVersion__retval = ncclResult_t(crccl.pncclGetVersion(&version))    # fully specified
    return (_pncclGetVersion__retval,version)


@cython.embedsignature(True)
def ncclGetUniqueId(object uniqueId):
    """! @brief Generates an ID for ncclCommInitRank

        @details
        Generates an ID to be used in ncclCommInitRank. ncclGetUniqueId should be
        called once and the Id should be distributed to all ranks in the
        communicator before calling ncclCommInitRank.

        @param[in]
        uniqueId     ncclUniqueId*
                     pointer to uniqueId
    """
    _ncclGetUniqueId__retval = ncclResult_t(crccl.ncclGetUniqueId(
        ncclUniqueId.from_pyobj(uniqueId)._ptr))    # fully specified
    return (_ncclGetUniqueId__retval,)


@cython.embedsignature(True)
def pncclGetUniqueId(object uniqueId):
    """@cond include_hidden
    """
    _pncclGetUniqueId__retval = ncclResult_t(crccl.pncclGetUniqueId(
        ncclUniqueId.from_pyobj(uniqueId)._ptr))    # fully specified
    return (_pncclGetUniqueId__retval,)


@cython.embedsignature(True)
def ncclCommInitRank(int nranks, object commId, int rank):
    """! @brief Creates a new communicator (multi thread/process version).

        @details
        rank must be between 0 and nranks-1 and unique within a communicator clique.
        Each rank is associated to a CUDA device, which has to be set before calling
        ncclCommInitRank.
        ncclCommInitRank implicitly syncronizes with other ranks, so it must be
        called by different threads/processes or use ncclGroupStart/ncclGroupEnd.

        @param[in]
        comm        ncclComm_t*
                    communicator struct pointer
    """
    comm = ncclComm.from_ptr(NULL)
    _ncclCommInitRank__retval = ncclResult_t(crccl.ncclCommInitRank(&comm._ptr,nranks,
        ncclUniqueId.from_pyobj(commId)._ptr[0],rank))    # fully specified
    return (_ncclCommInitRank__retval,comm)


@cython.embedsignature(True)
def pncclCommInitRank(int nranks, object commId, int rank):
    """@cond include_hidden
    """
    comm = ncclComm.from_ptr(NULL)
    _pncclCommInitRank__retval = ncclResult_t(crccl.pncclCommInitRank(&comm._ptr,nranks,
        ncclUniqueId.from_pyobj(commId)._ptr[0],rank))    # fully specified
    return (_pncclCommInitRank__retval,comm)


@cython.embedsignature(True)
def ncclCommInitRankMulti(int nranks, object commId, int rank, int virtualId):
    """! @brief Creates a new communicator (multi thread/process version) allowing multiple ranks per device.

        @details
        rank must be between 0 and nranks-1 and unique within a communicator clique.
        Each rank is associated to a HIP device, which has to be set before calling
        ncclCommInitRankMulti.
        Since this version of the function allows multiple ranks to utilize the same
        HIP device, a unique virtualId per device has to be provided by each calling
        rank.
        ncclCommInitRankMulti implicitly syncronizes with other ranks, so it must be
        called by different threads/processes or use ncclGroupStart/ncclGroupEnd.

        @param[in]
        comm        ncclComm_t*
                    communicator struct pointer
    """
    comm = ncclComm.from_ptr(NULL)
    _ncclCommInitRankMulti__retval = ncclResult_t(crccl.ncclCommInitRankMulti(&comm._ptr,nranks,
        ncclUniqueId.from_pyobj(commId)._ptr[0],rank,virtualId))    # fully specified
    return (_ncclCommInitRankMulti__retval,comm)


@cython.embedsignature(True)
def pncclCommInitRankMulti(int nranks, object commId, int rank, int virtualId):
    """@cond include_hidden
    """
    comm = ncclComm.from_ptr(NULL)
    _pncclCommInitRankMulti__retval = ncclResult_t(crccl.pncclCommInitRankMulti(&comm._ptr,nranks,
        ncclUniqueId.from_pyobj(commId)._ptr[0],rank,virtualId))    # fully specified
    return (_pncclCommInitRankMulti__retval,comm)


@cython.embedsignature(True)
def ncclCommInitAll(int ndev, object devlist):
    """! @brief Creates a clique of communicators (single process version).
    @details This is a convenience function to create a single-process communicator clique.
    Returns an array of ndev newly initialized communicators in comm.
    comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
    If devlist is NULL, the first ndev HIP devices are used.
    Order of devlist defines user-order of processors within the communicator.
    """
    comm = ncclComm.from_ptr(NULL)
    _ncclCommInitAll__retval = ncclResult_t(crccl.ncclCommInitAll(&comm._ptr,ndev,
        <const int *>hip._util.types.ListOfInt.from_pyobj(devlist)._ptr))    # fully specified
    return (_ncclCommInitAll__retval,comm)


@cython.embedsignature(True)
def pncclCommInitAll(int ndev, object devlist):
    """@cond include_hidden
    """
    comm = ncclComm.from_ptr(NULL)
    _pncclCommInitAll__retval = ncclResult_t(crccl.pncclCommInitAll(&comm._ptr,ndev,
        <const int *>hip._util.types.ListOfInt.from_pyobj(devlist)._ptr))    # fully specified
    return (_pncclCommInitAll__retval,comm)


@cython.embedsignature(True)
def ncclCommDestroy(object comm):
    """! @brief Frees resources associated with communicator object, but waits for any operations that might still be running on the device */
    """
    _ncclCommDestroy__retval = ncclResult_t(crccl.ncclCommDestroy(
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_ncclCommDestroy__retval,)


@cython.embedsignature(True)
def pncclCommDestroy(object comm):
    """@cond include_hidden
    """
    _pncclCommDestroy__retval = ncclResult_t(crccl.pncclCommDestroy(
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_pncclCommDestroy__retval,)


@cython.embedsignature(True)
def ncclCommAbort(object comm):
    """! @brief Frees resources associated with communicator object and aborts any operations that might still be running on the device. */
    """
    _ncclCommAbort__retval = ncclResult_t(crccl.ncclCommAbort(
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_ncclCommAbort__retval,)


@cython.embedsignature(True)
def pncclCommAbort(object comm):
    """@cond include_hidden
    """
    _pncclCommAbort__retval = ncclResult_t(crccl.pncclCommAbort(
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_pncclCommAbort__retval,)


@cython.embedsignature(True)
def ncclGetErrorString(object result):
    """! @brief Returns a string for each error code. */
    """
    if not isinstance(result,ncclResult_t):
        raise TypeError("argument 'result' must be of type 'ncclResult_t'")
    cdef const char * _ncclGetErrorString__retval = crccl.ncclGetErrorString(result.value)    # fully specified
    return (_ncclGetErrorString__retval,)


@cython.embedsignature(True)
def pncclGetErrorString(object result):
    """@cond include_hidden
    """
    if not isinstance(result,ncclResult_t):
        raise TypeError("argument 'result' must be of type 'ncclResult_t'")
    cdef const char * _pncclGetErrorString__retval = crccl.pncclGetErrorString(result.value)    # fully specified
    return (_pncclGetErrorString__retval,)


@cython.embedsignature(True)
def ncclGetLastError(object comm):
    """! @brief Returns a human-readable message of the last error that occurred.
    comm is currently unused and can be set to NULL
    """
    cdef const char * _ncclGetLastError__retval = crccl.ncclGetLastError(
        ncclComm.from_pyobj(comm)._ptr)    # fully specified
    return (_ncclGetLastError__retval,)


@cython.embedsignature(True)
def pncclGetError(object comm):
    """@cond include_hidden
    """
    cdef const char * _pncclGetError__retval = crccl.pncclGetError(
        ncclComm.from_pyobj(comm)._ptr)    # fully specified
    return (_pncclGetError__retval,)


@cython.embedsignature(True)
def ncclCommGetAsyncError(object comm, object asyncError):
    """@endcond
    """
    _ncclCommGetAsyncError__retval = ncclResult_t(crccl.ncclCommGetAsyncError(
        ncclComm.from_pyobj(comm)._ptr,
        <crccl.ncclResult_t *>hip._util.types.DataHandle.from_pyobj(asyncError)._ptr))    # fully specified
    return (_ncclCommGetAsyncError__retval,)


@cython.embedsignature(True)
def pncclCommGetAsyncError(object comm, object asyncError):
    """@cond include_hidden
    """
    _pncclCommGetAsyncError__retval = ncclResult_t(crccl.pncclCommGetAsyncError(
        ncclComm.from_pyobj(comm)._ptr,
        <crccl.ncclResult_t *>hip._util.types.DataHandle.from_pyobj(asyncError)._ptr))    # fully specified
    return (_pncclCommGetAsyncError__retval,)


@cython.embedsignature(True)
def ncclCommCount(object comm):
    """! @brief Gets the number of ranks in the communicator clique. */
    """
    cdef int count
    _ncclCommCount__retval = ncclResult_t(crccl.ncclCommCount(
        ncclComm.from_pyobj(comm)._ptr,&count))    # fully specified
    return (_ncclCommCount__retval,count)


@cython.embedsignature(True)
def pncclCommCount(object comm):
    """@cond include_hidden
    """
    cdef int count
    _pncclCommCount__retval = ncclResult_t(crccl.pncclCommCount(
        ncclComm.from_pyobj(comm)._ptr,&count))    # fully specified
    return (_pncclCommCount__retval,count)


@cython.embedsignature(True)
def ncclCommCuDevice(object comm):
    """! @brief Returns the rocm device number associated with the communicator. */
    """
    cdef int device
    _ncclCommCuDevice__retval = ncclResult_t(crccl.ncclCommCuDevice(
        ncclComm.from_pyobj(comm)._ptr,&device))    # fully specified
    return (_ncclCommCuDevice__retval,device)


@cython.embedsignature(True)
def pncclCommCuDevice(object comm):
    """@cond include_hidden
    """
    cdef int device
    _pncclCommCuDevice__retval = ncclResult_t(crccl.pncclCommCuDevice(
        ncclComm.from_pyobj(comm)._ptr,&device))    # fully specified
    return (_pncclCommCuDevice__retval,device)


@cython.embedsignature(True)
def ncclCommUserRank(object comm):
    """! @brief Returns the user-ordered "rank" associated with the communicator. */
    """
    cdef int rank
    _ncclCommUserRank__retval = ncclResult_t(crccl.ncclCommUserRank(
        ncclComm.from_pyobj(comm)._ptr,&rank))    # fully specified
    return (_ncclCommUserRank__retval,rank)


@cython.embedsignature(True)
def pncclCommUserRank(object comm):
    """@cond include_hidden
    """
    cdef int rank
    _pncclCommUserRank__retval = ncclResult_t(crccl.pncclCommUserRank(
        ncclComm.from_pyobj(comm)._ptr,&rank))    # fully specified
    return (_pncclCommUserRank__retval,rank)


class ncclRedOp_dummy_t(enum.IntEnum):
    ncclNumOps_dummy = crccl.ncclNumOps_dummy
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class ncclRedOp_t(enum.IntEnum):
    ncclSum = crccl.ncclSum
    ncclProd = crccl.ncclProd
    ncclMax = crccl.ncclMax
    ncclMin = crccl.ncclMin
    ncclAvg = crccl.ncclAvg
    ncclNumOps = crccl.ncclNumOps
    ncclMaxRedOp = crccl.ncclMaxRedOp
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class ncclDataType_t(enum.IntEnum):
    ncclInt8 = crccl.ncclInt8
    ncclChar = crccl.ncclChar
    ncclUint8 = crccl.ncclUint8
    ncclInt32 = crccl.ncclInt32
    ncclInt = crccl.ncclInt
    ncclUint32 = crccl.ncclUint32
    ncclInt64 = crccl.ncclInt64
    ncclUint64 = crccl.ncclUint64
    ncclFloat16 = crccl.ncclFloat16
    ncclHalf = crccl.ncclHalf
    ncclFloat32 = crccl.ncclFloat32
    ncclFloat = crccl.ncclFloat
    ncclFloat64 = crccl.ncclFloat64
    ncclDouble = crccl.ncclDouble
    ncclBfloat16 = crccl.ncclBfloat16
    ncclNumTypes = crccl.ncclNumTypes
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


class ncclScalarResidence_t(enum.IntEnum):
    ncclScalarDevice = crccl.ncclScalarDevice
    ncclScalarHostImmediate = crccl.ncclScalarHostImmediate
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


@cython.embedsignature(True)
def ncclRedOpCreatePreMulSum(object op, object scalar, object datatype, object residence, object comm):
    """
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(residence,ncclScalarResidence_t):
        raise TypeError("argument 'residence' must be of type 'ncclScalarResidence_t'")
    _ncclRedOpCreatePreMulSum__retval = ncclResult_t(crccl.ncclRedOpCreatePreMulSum(
        <crccl.ncclRedOp_t *>hip._util.types.DataHandle.from_pyobj(op)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(scalar)._ptr,datatype.value,residence.value,
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_ncclRedOpCreatePreMulSum__retval,)


@cython.embedsignature(True)
def pncclRedOpCreatePreMulSum(object op, object scalar, object datatype, object residence, object comm):
    """
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(residence,ncclScalarResidence_t):
        raise TypeError("argument 'residence' must be of type 'ncclScalarResidence_t'")
    _pncclRedOpCreatePreMulSum__retval = ncclResult_t(crccl.pncclRedOpCreatePreMulSum(
        <crccl.ncclRedOp_t *>hip._util.types.DataHandle.from_pyobj(op)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(scalar)._ptr,datatype.value,residence.value,
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_pncclRedOpCreatePreMulSum__retval,)


@cython.embedsignature(True)
def ncclRedOpDestroy(object op, object comm):
    """
    """
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _ncclRedOpDestroy__retval = ncclResult_t(crccl.ncclRedOpDestroy(op.value,
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_ncclRedOpDestroy__retval,)


@cython.embedsignature(True)
def pncclRedOpDestroy(object op, object comm):
    """
    """
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _pncclRedOpDestroy__retval = ncclResult_t(crccl.pncclRedOpDestroy(op.value,
        ncclComm.from_pyobj(comm)._ptr))    # fully specified
    return (_pncclRedOpDestroy__retval,)


@cython.embedsignature(True)
def ncclReduce(object sendbuff, object recvbuff, unsigned long count, object datatype, object op, int root, object comm, object stream):
    """!
    @brief Reduce
    @details Reduces data arrays of length count in sendbuff into recvbuff using op
    operation.
    recvbuff may be NULL on all calls except for root device.
    root is the rank (not the CUDA device) where data will reside after the
    operation is complete.
    In-place operation will happen if sendbuff == recvbuff.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _ncclReduce__retval = ncclResult_t(crccl.ncclReduce(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,op.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclReduce__retval,)


@cython.embedsignature(True)
def pncclReduce(object sendbuff, object recvbuff, unsigned long count, object datatype, object op, int root, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _pncclReduce__retval = ncclResult_t(crccl.pncclReduce(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,op.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclReduce__retval,)


@cython.embedsignature(True)
def ncclBcast(object buff, unsigned long count, object datatype, int root, object comm, object stream):
    """! @brief (deprecated) Broadcast (in-place)
    @details Copies count values from root to all other devices.
    root is the rank (not the CUDA device) where data resides before the
    operation is started.
    This operation is implicitely in place.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclBcast__retval = ncclResult_t(crccl.ncclBcast(
        <void *>hip._util.types.DataHandle.from_pyobj(buff)._ptr,count,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclBcast__retval,)


@cython.embedsignature(True)
def pncclBcast(object buff, unsigned long count, object datatype, int root, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclBcast__retval = ncclResult_t(crccl.pncclBcast(
        <void *>hip._util.types.DataHandle.from_pyobj(buff)._ptr,count,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclBcast__retval,)


@cython.embedsignature(True)
def ncclBroadcast(object sendbuff, object recvbuff, unsigned long count, object datatype, int root, object comm, object stream):
    """! @brief Broadcast
    @details Copies count values from root to all other devices.
    root is the rank (not the HIP device) where data resides before the
    operation is started.
    In-place operation will happen if sendbuff == recvbuff.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclBroadcast__retval = ncclResult_t(crccl.ncclBroadcast(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclBroadcast__retval,)


@cython.embedsignature(True)
def pncclBroadcast(object sendbuff, object recvbuff, unsigned long count, object datatype, int root, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclBroadcast__retval = ncclResult_t(crccl.pncclBroadcast(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclBroadcast__retval,)


@cython.embedsignature(True)
def ncclAllReduce(object sendbuff, object recvbuff, unsigned long count, object datatype, object op, object comm, object stream):
    """! @brief All-Reduce
    @details Reduces data arrays of length count in sendbuff using op operation, and
    leaves identical copies of result on each recvbuff.
    In-place operation will happen if sendbuff == recvbuff.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _ncclAllReduce__retval = ncclResult_t(crccl.ncclAllReduce(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,op.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclAllReduce__retval,)


@cython.embedsignature(True)
def pncclAllReduce(object sendbuff, object recvbuff, unsigned long count, object datatype, object op, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _pncclAllReduce__retval = ncclResult_t(crccl.pncclAllReduce(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,op.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclAllReduce__retval,)


@cython.embedsignature(True)
def ncclReduceScatter(object sendbuff, object recvbuff, unsigned long recvcount, object datatype, object op, object comm, object stream):
    """!
    @brief Reduce-Scatter
    @details Reduces data in sendbuff using op operation and leaves reduced result
    scattered over the devices so that recvbuff on rank i will contain the i-th
    block of the result.
    Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
    should have a size of at least nranks*recvcount elements.
    In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _ncclReduceScatter__retval = ncclResult_t(crccl.ncclReduceScatter(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,recvcount,datatype.value,op.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclReduceScatter__retval,)


@cython.embedsignature(True)
def pncclReduceScatter(object sendbuff, object recvbuff, unsigned long recvcount, object datatype, object op, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")                    
    if not isinstance(op,ncclRedOp_t):
        raise TypeError("argument 'op' must be of type 'ncclRedOp_t'")
    _pncclReduceScatter__retval = ncclResult_t(crccl.pncclReduceScatter(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,recvcount,datatype.value,op.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclReduceScatter__retval,)


@cython.embedsignature(True)
def ncclAllGather(object sendbuff, object recvbuff, unsigned long sendcount, object datatype, object comm, object stream):
    """! @brief All-Gather
    @details Each device gathers sendcount values from other GPUs into recvbuff,
    receiving data from rank i at offset i*sendcount.
    Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
    should have a size of at least nranks*sendcount elements.
    In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclAllGather__retval = ncclResult_t(crccl.ncclAllGather(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,sendcount,datatype.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclAllGather__retval,)


@cython.embedsignature(True)
def pncclAllGather(object sendbuff, object recvbuff, unsigned long sendcount, object datatype, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclAllGather__retval = ncclResult_t(crccl.pncclAllGather(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,sendcount,datatype.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclAllGather__retval,)


@cython.embedsignature(True)
def ncclSend(object sendbuff, unsigned long count, object datatype, int peer, object comm, object stream):
    """! @brief Send
    @details Send data from sendbuff to rank peer.
    Rank peer needs to call ncclRecv with the same datatype and the same count from this
    rank.
    This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
    need to progress concurrently to complete, they must be fused within a ncclGroupStart/
    ncclGroupEnd section.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclSend__retval = ncclResult_t(crccl.ncclSend(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,count,datatype.value,peer,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclSend__retval,)


@cython.embedsignature(True)
def pncclSend(object sendbuff, unsigned long count, object datatype, int peer, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclSend__retval = ncclResult_t(crccl.pncclSend(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,count,datatype.value,peer,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclSend__retval,)


@cython.embedsignature(True)
def ncclRecv(object recvbuff, unsigned long count, object datatype, int peer, object comm, object stream):
    """! @brief Receive
    @details Receive data from rank peer into recvbuff.
    Rank peer needs to call ncclSend with the same datatype and the same count to this
    rank.
    This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
    need to progress concurrently to complete, they must be fused within a ncclGroupStart/
    ncclGroupEnd section.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclRecv__retval = ncclResult_t(crccl.ncclRecv(
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,peer,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclRecv__retval,)


@cython.embedsignature(True)
def pncclRecv(object recvbuff, unsigned long count, object datatype, int peer, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclRecv__retval = ncclResult_t(crccl.pncclRecv(
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,peer,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclRecv__retval,)


@cython.embedsignature(True)
def ncclGather(object sendbuff, object recvbuff, unsigned long sendcount, object datatype, int root, object comm, object stream):
    """! @brief Gather
    @details Root device gathers sendcount values from other GPUs into recvbuff,
    receiving data from rank i at offset i*sendcount.
    Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
    should have a size of at least nranks*sendcount elements.
    In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclGather__retval = ncclResult_t(crccl.ncclGather(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,sendcount,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclGather__retval,)


@cython.embedsignature(True)
def pncclGather(object sendbuff, object recvbuff, unsigned long sendcount, object datatype, int root, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclGather__retval = ncclResult_t(crccl.pncclGather(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,sendcount,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclGather__retval,)


@cython.embedsignature(True)
def ncclScatter(object sendbuff, object recvbuff, unsigned long recvcount, object datatype, int root, object comm, object stream):
    """! @brief Scatter
    @details Scattered over the devices so that recvbuff on rank i will contain the i-th
    block of the data on root.
    Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
    should have a size of at least nranks*recvcount elements.
    In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclScatter__retval = ncclResult_t(crccl.ncclScatter(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,recvcount,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclScatter__retval,)


@cython.embedsignature(True)
def pncclScatter(object sendbuff, object recvbuff, unsigned long recvcount, object datatype, int root, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclScatter__retval = ncclResult_t(crccl.pncclScatter(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,recvcount,datatype.value,root,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclScatter__retval,)


@cython.embedsignature(True)
def ncclAllToAll(object sendbuff, object recvbuff, unsigned long count, object datatype, object comm, object stream):
    """! @brief All-To-All
    @details Device (i) send (j)th block of data to device (j) and be placed as (i)th
    block. Each block for sending/receiving has count elements, which means
    that recvbuff and sendbuff should have a size of nranks*count elements.
    In-place operation will happen if sendbuff == recvbuff.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclAllToAll__retval = ncclResult_t(crccl.ncclAllToAll(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclAllToAll__retval,)


@cython.embedsignature(True)
def pncclAllToAll(object sendbuff, object recvbuff, unsigned long count, object datatype, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclAllToAll__retval = ncclResult_t(crccl.pncclAllToAll(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,count,datatype.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclAllToAll__retval,)


@cython.embedsignature(True)
def ncclAllToAllv(object sendbuff, object sendcounts, object sdispls, object recvbuff, object recvcounts, object rdispls, object datatype, object comm, object stream):
    """! @brief All-To-Allv
    @details Device (i) sends sendcounts[j] of data from offset sdispls[j]
    to device (j). In the same time, device (i) receives recvcounts[j] of data
    from device (j) to be placed at rdispls[j].

    sendcounts, sdispls, recvcounts and rdispls are all measured in the units
    of datatype, not bytes.
    In-place operation will happen if sendbuff == recvbuff.
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _ncclAllToAllv__retval = ncclResult_t(crccl.ncclAllToAllv(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(sendcounts)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(sdispls)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(recvcounts)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(rdispls)._ptr,datatype.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_ncclAllToAllv__retval,)


@cython.embedsignature(True)
def pncclAllToAllv(object sendbuff, object sendcounts, object sdispls, object recvbuff, object recvcounts, object rdispls, object datatype, object comm, object stream):
    """@cond include_hidden
    """
    if not isinstance(datatype,ncclDataType_t):
        raise TypeError("argument 'datatype' must be of type 'ncclDataType_t'")
    _pncclAllToAllv__retval = ncclResult_t(crccl.pncclAllToAllv(
        <const void *>hip._util.types.DataHandle.from_pyobj(sendbuff)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(sendcounts)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(sdispls)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(recvbuff)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(recvcounts)._ptr,
        <const unsigned long*>hip._util.types.ListOfUnsignedLong.from_pyobj(rdispls)._ptr,datatype.value,
        ncclComm.from_pyobj(comm)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_pncclAllToAllv__retval,)


@cython.embedsignature(True)
def ncclGroupStart():
    """! @brief Group Start
    Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
    a single NCCL operation. Nothing will be started on the CUDA stream until
    ncclGroupEnd.
    """
    _ncclGroupStart__retval = ncclResult_t(crccl.ncclGroupStart())    # fully specified
    return (_ncclGroupStart__retval,)


@cython.embedsignature(True)
def pncclGroupStart():
    """@cond include_hidden
    """
    _pncclGroupStart__retval = ncclResult_t(crccl.pncclGroupStart())    # fully specified
    return (_pncclGroupStart__retval,)


@cython.embedsignature(True)
def ncclGroupEnd():
    """! @brief Group End
    End a group call. Start a fused NCCL operation consisting of all calls since
    ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
    need to be called after ncclGroupEnd.
    """
    _ncclGroupEnd__retval = ncclResult_t(crccl.ncclGroupEnd())    # fully specified
    return (_ncclGroupEnd__retval,)


@cython.embedsignature(True)
def pncclGroupEnd():
    """@cond include_hidden
    """
    _pncclGroupEnd__retval = ncclResult_t(crccl.pncclGroupEnd())    # fully specified
    return (_pncclGroupEnd__retval,)
