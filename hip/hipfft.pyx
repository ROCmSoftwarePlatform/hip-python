# AMD_COPYRIGHT
import cython
import ctypes
import enum
import hip.hipify
HIPFFT_FORWARD = chipfft.HIPFFT_FORWARD

HIPFFT_BACKWARD = chipfft.HIPFFT_BACKWARD

class hipfftResult_t(hip.hipify.IntEnum):
    HIPFFT_SUCCESS = chipfft.HIPFFT_SUCCESS
    HIPFFT_INVALID_PLAN = chipfft.HIPFFT_INVALID_PLAN
    HIPFFT_ALLOC_FAILED = chipfft.HIPFFT_ALLOC_FAILED
    HIPFFT_INVALID_TYPE = chipfft.HIPFFT_INVALID_TYPE
    HIPFFT_INVALID_VALUE = chipfft.HIPFFT_INVALID_VALUE
    HIPFFT_INTERNAL_ERROR = chipfft.HIPFFT_INTERNAL_ERROR
    HIPFFT_EXEC_FAILED = chipfft.HIPFFT_EXEC_FAILED
    HIPFFT_SETUP_FAILED = chipfft.HIPFFT_SETUP_FAILED
    HIPFFT_INVALID_SIZE = chipfft.HIPFFT_INVALID_SIZE
    HIPFFT_UNALIGNED_DATA = chipfft.HIPFFT_UNALIGNED_DATA
    HIPFFT_INCOMPLETE_PARAMETER_LIST = chipfft.HIPFFT_INCOMPLETE_PARAMETER_LIST
    HIPFFT_INVALID_DEVICE = chipfft.HIPFFT_INVALID_DEVICE
    HIPFFT_PARSE_ERROR = chipfft.HIPFFT_PARSE_ERROR
    HIPFFT_NO_WORKSPACE = chipfft.HIPFFT_NO_WORKSPACE
    HIPFFT_NOT_IMPLEMENTED = chipfft.HIPFFT_NOT_IMPLEMENTED
    HIPFFT_NOT_SUPPORTED = chipfft.HIPFFT_NOT_SUPPORTED
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hipfftResult = hipfftResult_t

class hipfftType_t(hip.hipify.IntEnum):
    HIPFFT_R2C = chipfft.HIPFFT_R2C
    HIPFFT_C2R = chipfft.HIPFFT_C2R
    HIPFFT_C2C = chipfft.HIPFFT_C2C
    HIPFFT_D2Z = chipfft.HIPFFT_D2Z
    HIPFFT_Z2D = chipfft.HIPFFT_Z2D
    HIPFFT_Z2Z = chipfft.HIPFFT_Z2Z
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hipfftType = hipfftType_t

class hipfftLibraryPropertyType_t(hip.hipify.IntEnum):
    HIPFFT_MAJOR_VERSION = chipfft.HIPFFT_MAJOR_VERSION
    HIPFFT_MINOR_VERSION = chipfft.HIPFFT_MINOR_VERSION
    HIPFFT_PATCH_LEVEL = chipfft.HIPFFT_PATCH_LEVEL
    @staticmethod
    def ctypes_type():
        """The type of the enum constants as ctypes type."""
        return ctypes.c_uint 


hipfftLibraryPropertyType = hipfftLibraryPropertyType_t

cdef class hipfftHandle_t:
    # members declared in pxd file

    def __cinit__(self):
        self._ptr = NULL
        self.ptr_owner = False
        self._py_buffer_acquired = False

    @staticmethod
    cdef hipfftHandle_t from_ptr(chipfft.hipfftHandle_t* ptr, bint owner=False):
        """Factory function to create ``hipfftHandle_t`` objects from
        given ``chipfft.hipfftHandle_t`` pointer.
        """
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef hipfftHandle_t wrapper = hipfftHandle_t.__new__(hipfftHandle_t)
        wrapper._ptr = ptr
        wrapper.ptr_owner = owner
        return wrapper

    @staticmethod
    cdef hipfftHandle_t from_pyobj(object pyobj):
        """Derives a hipfftHandle_t from a Python object.

        Derives a hipfftHandle_t from the given Python object ``pyobj``.
        In case ``pyobj`` is itself an ``hipfftHandle_t`` reference, this method
        returns it directly. No new ``hipfftHandle_t`` is created in this case.

        Args:
            pyobj (object): Must be either ``None``, a simple, contiguous buffer according to the buffer protocol,
                            or of type ``hipfftHandle_t``, ``int``, or ``ctypes.c_void_p``

        Note:
            This routine does not perform a copy but returns the original ``pyobj``
            if ``pyobj`` is an instance of hipfftHandle_t!
        """
        cdef hipfftHandle_t wrapper = hipfftHandle_t.__new__(hipfftHandle_t)
        cdef dict cuda_array_interface = getattr(pyobj, "__cuda_array_interface__", None)

        if pyobj is None:
            wrapper._ptr = NULL
        elif isinstance(pyobj,hipfftHandle_t):
            return pyobj
        elif isinstance(pyobj,int):
            wrapper._ptr = <chipfft.hipfftHandle_t*>cpython.long.PyLong_AsVoidPtr(pyobj)
        elif isinstance(pyobj,ctypes.c_void_p):
            wrapper._ptr = <chipfft.hipfftHandle_t*>cpython.long.PyLong_AsVoidPtr(pyobj.value) if pyobj.value != None else NULL
        elif cuda_array_interface != None:
            if not "data" in cuda_array_interface:
                raise ValueError("input object has '__cuda_array_interface__' attribute but the dict has no 'data' key")
            ptr_as_int = cuda_array_interface["data"][0]
            wrapper._ptr = <chipfft.hipfftHandle_t*>cpython.long.PyLong_AsVoidPtr(ptr_as_int)
        elif cpython.buffer.PyObject_CheckBuffer(pyobj):
            err = cpython.buffer.PyObject_GetBuffer( 
                pyobj,
                &wrapper._py_buffer, 
                cpython.buffer.PyBUF_SIMPLE | cpython.buffer.PyBUF_ANY_CONTIGUOUS
            )
            if err == -1:
                raise RuntimeError("failed to create simple, contiguous Py_buffer from Python object")
            wrapper._py_buffer_acquired = True
            wrapper._ptr = <chipfft.hipfftHandle_t*>wrapper._py_buffer.buf
        else:
            raise TypeError(f"unsupported input type: '{str(type(pyobj))}'")
        return wrapper
    def __dealloc__(self):
        # Release the buffer handle
        if self._py_buffer_acquired is True:
            cpython.buffer.PyBuffer_Release(&self._py_buffer)
    
    def __int__(self):
        """Returns the data's address as long integer."""
        return cpython.long.PyLong_FromVoidPtr(self._ptr)
    def __repr__(self):
        return f"<hipfftHandle_t object, self.ptr={int(self)}>"
    def as_c_void_p(self):
        """Returns the data's address as `ctypes.c_void_p`"""
        return ctypes.c_void_p(int(self))
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


hipfftHandle = hipfftHandle_t

hipfftComplex = float2

hipfftDoubleComplex = double2

@cython.embedsignature(True)
def hipfftPlan1d(int nx, object type, int batch):
    """! @brief Create a new one-dimensional FFT plan.
    @details Allocate and initialize a new one-dimensional FFT plan.
    @param[out] plan Pointer to the FFT plan handle.
    @param[in] nx FFT length.
    @param[in] type FFT type.
    @param[in] batch Number of batched transforms to compute.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")
    _hipfftPlan1d__retval = hipfftResult_t(chipfft.hipfftPlan1d(&plan._ptr,nx,type.value,batch))    # fully specified
    return (_hipfftPlan1d__retval,plan)


@cython.embedsignature(True)
def hipfftPlan2d(int nx, int ny, object type):
    """! @brief Create a new two-dimensional FFT plan.
    @details Allocate and initialize a new two-dimensional FFT plan.
    Two-dimensional data should be stored in C ordering (row-major
    format), so that indexes in y-direction (j index) vary the
    fastest.
    @param[out] plan Pointer to the FFT plan handle.
    @param[in] nx Number of elements in the x-direction (slow index).
    @param[in] ny Number of elements in the y-direction (fast index).
    @param[in] type FFT type.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")
    _hipfftPlan2d__retval = hipfftResult_t(chipfft.hipfftPlan2d(&plan._ptr,nx,ny,type.value))    # fully specified
    return (_hipfftPlan2d__retval,plan)


@cython.embedsignature(True)
def hipfftPlan3d(int nx, int ny, int nz, object type):
    """! @brief Create a new three-dimensional FFT plan.
    @details Allocate and initialize a new three-dimensional FFT plan.
    Three-dimensional data should be stored in C ordering (row-major
    format), so that indexes in z-direction (k index) vary the
    fastest.
    @param[out] plan Pointer to the FFT plan handle.
    @param[in] nx Number of elements in the x-direction (slowest index).
    @param[in] ny Number of elements in the y-direction.
    @param[in] nz Number of elements in the z-direction (fastest index).
    @param[in] type FFT type.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")
    _hipfftPlan3d__retval = hipfftResult_t(chipfft.hipfftPlan3d(&plan._ptr,nx,ny,nz,type.value))    # fully specified
    return (_hipfftPlan3d__retval,plan)


@cython.embedsignature(True)
def hipfftPlanMany(int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    """! @brief Create a new batched rank-dimensional FFT plan with advanced data layout.
    @details Allocate and initialize a new batched rank-dimensional
    FFT plan. The number of elements to transform in each direction of
    the input data is specified in n.
    The batch parameter tells hipFFT how many transforms to perform. 
    The distance between the first elements of two consecutive batches 
    of the input and output data are specified with the idist and odist 
    parameters.
    The inembed and onembed parameters define the input and output data
    layouts. The number of elements in the data is assumed to be larger 
    than the number of elements in the transform. Strided data layouts 
    are also supported. Strides along the fastest direction in the input
    and output data are specified via the istride and ostride parameters.  
    If both inembed and onembed parameters are set to NULL, all the 
    advanced data layout parameters are ignored and reverted to default 
    values, i.e., the batched transform is performed with non-strided data
    access and the number of data/transform elements are assumed to be  
    equivalent.
    @param[out] plan Pointer to the FFT plan handle.
    @param[in] rank Dimension of transform (1, 2, or 3).
    @param[in] n Number of elements to transform in the x/y/z directions.
    @param[in] inembed Number of elements in the input data in the x/y/z directions.
    @param[in] istride Distance between two successive elements in the input data.
    @param[in] idist Distance between input batches.
    @param[in] onembed Number of elements in the output data in the x/y/z directions.
    @param[in] ostride Distance between two successive elements in the output data.
    @param[in] odist Distance between output batches.
    @param[in] type FFT type.
    @param[in] batch Number of batched transforms to perform.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")
    _hipfftPlanMany__retval = hipfftResult_t(chipfft.hipfftPlanMany(&plan._ptr,rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch))    # fully specified
    return (_hipfftPlanMany__retval,plan)


@cython.embedsignature(True)
def hipfftCreate():
    """! @brief Allocate a new plan.
    """
    plan = hipfftHandle_t.from_ptr(NULL)
    _hipfftCreate__retval = hipfftResult_t(chipfft.hipfftCreate(&plan._ptr))    # fully specified
    return (_hipfftCreate__retval,plan)


@cython.embedsignature(True)
def hipfftExtPlanScaleFactor(object plan, double scalefactor):
    """! @brief Set scaling factor.
    @details hipFFT multiplies each element of the result by the given factor at the end of the transform.
    The supplied factor must be a finite number.  That is, it must neither be infinity nor NaN.
    This function must be called after the plan is allocated using
    ::hipfftCreate, but before the plan is initialized by any of the
    "MakePlan" functions.
    """
    _hipfftExtPlanScaleFactor__retval = hipfftResult_t(chipfft.hipfftExtPlanScaleFactor(
        hipfftHandle_t.from_pyobj(plan)._ptr,scalefactor))    # fully specified
    return (_hipfftExtPlanScaleFactor__retval,)


@cython.embedsignature(True)
def hipfftMakePlan1d(object plan, int nx, object type, int batch):
    """! @brief Initialize a new one-dimensional FFT plan.
    @details Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle.
    @param[in] plan Handle of the FFT plan.
    @param[in] nx FFT length.
    @param[in] type FFT type.
    @param[in] batch Number of batched transforms to compute.
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftMakePlan1d__retval = hipfftResult_t(chipfft.hipfftMakePlan1d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,type.value,batch,&workSize))    # fully specified
    return (_hipfftMakePlan1d__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlan2d(object plan, int nx, int ny, object type):
    """! @brief Initialize a new two-dimensional FFT plan.
    @details Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle.
    Two-dimensional data should be stored in C ordering (row-major
    format), so that indexes in y-direction (j index) vary the
    fastest.
    @param[in] plan Handle of the FFT plan.
    @param[in] nx Number of elements in the x-direction (slow index).
    @param[in] ny Number of elements in the y-direction (fast index).
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftMakePlan2d__retval = hipfftResult_t(chipfft.hipfftMakePlan2d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,type.value,&workSize))    # fully specified
    return (_hipfftMakePlan2d__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlan3d(object plan, int nx, int ny, int nz, object type):
    """! @brief Initialize a new two-dimensional FFT plan.
    @details Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle.
    Three-dimensional data should be stored in C ordering (row-major
    format), so that indexes in z-direction (k index) vary the
    fastest.
    @param[in] plan Handle of the FFT plan.
    @param[in] nx Number of elements in the x-direction (slowest index).
    @param[in] ny Number of elements in the y-direction.
    @param[in] nz Number of elements in the z-direction (fastest index).
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftMakePlan3d__retval = hipfftResult_t(chipfft.hipfftMakePlan3d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,nz,type.value,&workSize))    # fully specified
    return (_hipfftMakePlan3d__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlanMany(object plan, int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    """! @brief Initialize a new batched rank-dimensional FFT plan with advanced data layout.
    @details Assumes that the plan has been created already, and
    modifies the plan associated with the plan handle. The number 
    of elements to transform in each direction of the input data 
    in the FFT plan is specified in n.
    The batch parameter tells hipFFT how many transforms to perform. 
    The distance between the first elements of two consecutive batches 
    of the input and output data are specified with the idist and odist 
    parameters.
    The inembed and onembed parameters define the input and output data
    layouts. The number of elements in the data is assumed to be larger 
    than the number of elements in the transform. Strided data layouts 
    are also supported. Strides along the fastest direction in the input
    and output data are specified via the istride and ostride parameters.  
    If both inembed and onembed parameters are set to NULL, all the 
    advanced data layout parameters are ignored and reverted to default 
    values, i.e., the batched transform is performed with non-strided data
    access and the number of data/transform elements are assumed to be  
    equivalent.
    @param[out] plan Pointer to the FFT plan handle.
    @param[in] rank Dimension of transform (1, 2, or 3).
    @param[in] n Number of elements to transform in the x/y/z directions.
    @param[in] inembed Number of elements in the input data in the x/y/z directions.
    @param[in] istride Distance between two successive elements in the input data.
    @param[in] idist Distance between input batches.
    @param[in] onembed Number of elements in the output data in the x/y/z directions.
    @param[in] ostride Distance between two successive elements in the output data.
    @param[in] odist Distance between output batches.
    @param[in] type FFT type.
    @param[in] batch Number of batched transforms to perform.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftMakePlanMany__retval = hipfftResult_t(chipfft.hipfftMakePlanMany(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftMakePlanMany__retval,workSize)


@cython.embedsignature(True)
def hipfftMakePlanMany64(object plan, int rank, object n, object inembed, long long istride, long long idist, object onembed, long long ostride, long long odist, object type, long long batch):
    """
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftMakePlanMany64__retval = hipfftResult_t(chipfft.hipfftMakePlanMany64(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <long long *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <long long *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <long long *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftMakePlanMany64__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimate1d(int nx, object type, int batch):
    """! @brief Return an estimate of the work area size required for a 1D plan.
    @param[in] nx Number of elements in the x-direction.
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftEstimate1d__retval = hipfftResult_t(chipfft.hipfftEstimate1d(nx,type.value,batch,&workSize))    # fully specified
    return (_hipfftEstimate1d__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimate2d(int nx, int ny, object type):
    """! @brief Return an estimate of the work area size required for a 2D plan.
    @param[in] nx Number of elements in the x-direction.
    @param[in] ny Number of elements in the y-direction.
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftEstimate2d__retval = hipfftResult_t(chipfft.hipfftEstimate2d(nx,ny,type.value,&workSize))    # fully specified
    return (_hipfftEstimate2d__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimate3d(int nx, int ny, int nz, object type):
    """! @brief Return an estimate of the work area size required for a 3D plan.
    @param[in] nx Number of elements in the x-direction.
    @param[in] ny Number of elements in the y-direction.
    @param[in] nz Number of elements in the z-direction.
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftEstimate3d__retval = hipfftResult_t(chipfft.hipfftEstimate3d(nx,ny,nz,type.value,&workSize))    # fully specified
    return (_hipfftEstimate3d__retval,workSize)


@cython.embedsignature(True)
def hipfftEstimateMany(int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    """! @brief Return an estimate of the work area size required for a rank-dimensional plan.
    @param[in] rank Dimension of FFT transform (1, 2, or 3).
    @param[in] n Number of elements in the x/y/z directions.
    @param[in] inembed
    @param[in] istride
    @param[in] idist Distance between input batches.
    @param[in] onembed
    @param[in] ostride
    @param[in] odist Distance between output batches.
    @param[in] type FFT type.
    @param[in] batch Number of batched transforms to perform.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftEstimateMany__retval = hipfftResult_t(chipfft.hipfftEstimateMany(rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftEstimateMany__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize1d(object plan, int nx, object type, int batch):
    """! @brief Return size of the work area size required for a 1D plan.
    @param[in] plan Pointer to the FFT plan.
    @param[in] nx Number of elements in the x-direction.
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftGetSize1d__retval = hipfftResult_t(chipfft.hipfftGetSize1d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,type.value,batch,&workSize))    # fully specified
    return (_hipfftGetSize1d__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize2d(object plan, int nx, int ny, object type):
    """! @brief Return size of the work area size required for a 2D plan.
    @param[in] plan Pointer to the FFT plan.
    @param[in] nx Number of elements in the x-direction.
    @param[in] ny Number of elements in the y-direction.
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftGetSize2d__retval = hipfftResult_t(chipfft.hipfftGetSize2d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,type.value,&workSize))    # fully specified
    return (_hipfftGetSize2d__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize3d(object plan, int nx, int ny, int nz, object type):
    """! @brief Return size of the work area size required for a 3D plan.
    @param[in] plan Pointer to the FFT plan.
    @param[in] nx Number of elements in the x-direction.
    @param[in] ny Number of elements in the y-direction.
    @param[in] nz Number of elements in the z-direction.
    @param[in] type FFT type.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftGetSize3d__retval = hipfftResult_t(chipfft.hipfftGetSize3d(
        hipfftHandle_t.from_pyobj(plan)._ptr,nx,ny,nz,type.value,&workSize))    # fully specified
    return (_hipfftGetSize3d__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSizeMany(object plan, int rank, object n, object inembed, int istride, int idist, object onembed, int ostride, int odist, object type, int batch):
    """! @brief Return size of the work area size required for a rank-dimensional plan.
    @param[in] plan Pointer to the FFT plan.
    @param[in] rank Dimension of FFT transform (1, 2, or 3).
    @param[in] n Number of elements in the x/y/z directions.
    @param[in] inembed
    @param[in] istride
    @param[in] idist Distance between input batches.
    @param[in] onembed
    @param[in] ostride
    @param[in] odist Distance between output batches.
    @param[in] type FFT type.
    @param[in] batch Number of batched transforms to perform.
    @param[out] workSize Pointer to work area size (returned value).
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftGetSizeMany__retval = hipfftResult_t(chipfft.hipfftGetSizeMany(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <int *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <int *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <int *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftGetSizeMany__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSizeMany64(object plan, int rank, object n, object inembed, long long istride, long long idist, object onembed, long long ostride, long long odist, object type, long long batch):
    """
    """
    if not isinstance(type,hipfftType_t):
        raise TypeError("argument 'type' must be of type 'hipfftType_t'")                    
    cdef unsigned long workSize
    _hipfftGetSizeMany64__retval = hipfftResult_t(chipfft.hipfftGetSizeMany64(
        hipfftHandle_t.from_pyobj(plan)._ptr,rank,
        <long long *>hip._util.types.DataHandle.from_pyobj(n)._ptr,
        <long long *>hip._util.types.DataHandle.from_pyobj(inembed)._ptr,istride,idist,
        <long long *>hip._util.types.DataHandle.from_pyobj(onembed)._ptr,ostride,odist,type.value,batch,&workSize))    # fully specified
    return (_hipfftGetSizeMany64__retval,workSize)


@cython.embedsignature(True)
def hipfftGetSize(object plan):
    """! @brief Return size of the work area size required for a rank-dimensional plan.
    @param[in] plan Pointer to the FFT plan.
    """
    cdef unsigned long workSize
    _hipfftGetSize__retval = hipfftResult_t(chipfft.hipfftGetSize(
        hipfftHandle_t.from_pyobj(plan)._ptr,&workSize))    # fully specified
    return (_hipfftGetSize__retval,workSize)


@cython.embedsignature(True)
def hipfftSetAutoAllocation(object plan, int autoAllocate):
    """! @brief Set the plan's auto-allocation flag.  The plan will allocate its own workarea.
    @param[in] plan Pointer to the FFT plan.
    @param[in] autoAllocate 0 to disable auto-allocation, non-zero to enable.
    """
    _hipfftSetAutoAllocation__retval = hipfftResult_t(chipfft.hipfftSetAutoAllocation(
        hipfftHandle_t.from_pyobj(plan)._ptr,autoAllocate))    # fully specified
    return (_hipfftSetAutoAllocation__retval,)


@cython.embedsignature(True)
def hipfftSetWorkArea(object plan, object workArea):
    """! @brief Set the plan's work area.
    @param[in] plan Pointer to the FFT plan.
    @param[in] workArea Pointer to the work area (on device).
    """
    _hipfftSetWorkArea__retval = hipfftResult_t(chipfft.hipfftSetWorkArea(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        <void *>hip._util.types.DataHandle.from_pyobj(workArea)._ptr))    # fully specified
    return (_hipfftSetWorkArea__retval,)


@cython.embedsignature(True)
def hipfftExecC2C(object plan, object idata, object odata, int direction):
    """! @brief Execute a (float) complex-to-complex FFT.
    @details If the input and output buffers are equal, an in-place
    transform is performed.
    @param plan The FFT plan.
    @param idata Input data (on device).
    @param odata Output data (on device).
    @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
    """
    _hipfftExecC2C__retval = hipfftResult_t(chipfft.hipfftExecC2C(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        float2.from_pyobj(idata)._ptr,
        float2.from_pyobj(odata)._ptr,direction))    # fully specified
    return (_hipfftExecC2C__retval,)


@cython.embedsignature(True)
def hipfftExecR2C(object plan, object idata, object odata):
    """! @brief Execute a (float) real-to-complex FFT.
    @details If the input and output buffers are equal, an in-place
    transform is performed.
    @param plan The FFT plan.
    @param idata Input data (on device).
    @param odata Output data (on device).
    """
    _hipfftExecR2C__retval = hipfftResult_t(chipfft.hipfftExecR2C(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(idata)._ptr,
        float2.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecR2C__retval,)


@cython.embedsignature(True)
def hipfftExecC2R(object plan, object idata, object odata):
    """! @brief Execute a (float) complex-to-real FFT.
    @details If the input and output buffers are equal, an in-place
    transform is performed.
    @param plan The FFT plan.
    @param idata Input data (on device).
    @param odata Output data (on device).
    """
    _hipfftExecC2R__retval = hipfftResult_t(chipfft.hipfftExecC2R(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        float2.from_pyobj(idata)._ptr,
        <float *>hip._util.types.DataHandle.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecC2R__retval,)


@cython.embedsignature(True)
def hipfftExecZ2Z(object plan, object idata, object odata, int direction):
    """! @brief Execute a (double) complex-to-complex FFT.
    @details If the input and output buffers are equal, an in-place
    transform is performed.
    @param plan The FFT plan.
    @param idata Input data (on device).
    @param odata Output data (on device).
    @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
    """
    _hipfftExecZ2Z__retval = hipfftResult_t(chipfft.hipfftExecZ2Z(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        double2.from_pyobj(idata)._ptr,
        double2.from_pyobj(odata)._ptr,direction))    # fully specified
    return (_hipfftExecZ2Z__retval,)


@cython.embedsignature(True)
def hipfftExecD2Z(object plan, object idata, object odata):
    """! @brief Execute a (double) real-to-complex FFT.
    @details If the input and output buffers are equal, an in-place
    transform is performed.
    @param plan The FFT plan.
    @param idata Input data (on device).
    @param odata Output data (on device).
    """
    _hipfftExecD2Z__retval = hipfftResult_t(chipfft.hipfftExecD2Z(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(idata)._ptr,
        double2.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecD2Z__retval,)


@cython.embedsignature(True)
def hipfftExecZ2D(object plan, object idata, object odata):
    """! @brief Execute a (double) complex-to-real FFT.
    @details If the input and output buffers are equal, an in-place
    transform is performed.
    @param plan The FFT plan.
    @param idata Input data (on device).
    @param odata Output data (on device).
    """
    _hipfftExecZ2D__retval = hipfftResult_t(chipfft.hipfftExecZ2D(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        double2.from_pyobj(idata)._ptr,
        <double *>hip._util.types.DataHandle.from_pyobj(odata)._ptr))    # fully specified
    return (_hipfftExecZ2D__retval,)


@cython.embedsignature(True)
def hipfftSetStream(object plan, object stream):
    """! @brief Set HIP stream to execute plan on.
    @details Associates a HIP stream with a hipFFT plan.  All kernels
    launched by this plan are associated with the provided stream.
    @param plan The FFT plan.
    @param stream The HIP stream.
    """
    _hipfftSetStream__retval = hipfftResult_t(chipfft.hipfftSetStream(
        hipfftHandle_t.from_pyobj(plan)._ptr,
        ihipStream_t.from_pyobj(stream)._ptr))    # fully specified
    return (_hipfftSetStream__retval,)


@cython.embedsignature(True)
def hipfftDestroy(object plan):
    """! @brief Destroy and deallocate an existing plan.
    """
    _hipfftDestroy__retval = hipfftResult_t(chipfft.hipfftDestroy(
        hipfftHandle_t.from_pyobj(plan)._ptr))    # fully specified
    return (_hipfftDestroy__retval,)


@cython.embedsignature(True)
def hipfftGetVersion(object version):
    """! @brief Get rocFFT/cuFFT version.
    @param[out] version cuFFT/rocFFT version (returned value).
    """
    _hipfftGetVersion__retval = hipfftResult_t(chipfft.hipfftGetVersion(
        <int *>hip._util.types.DataHandle.from_pyobj(version)._ptr))    # fully specified
    return (_hipfftGetVersion__retval,)


@cython.embedsignature(True)
def hipfftGetProperty(object type, object value):
    """! @brief Get library property.
    @param[in] type Property type.
    @param[out] value Returned value.
    """
    if not isinstance(type,hipfftLibraryPropertyType_t):
        raise TypeError("argument 'type' must be of type 'hipfftLibraryPropertyType_t'")
    _hipfftGetProperty__retval = hipfftResult_t(chipfft.hipfftGetProperty(type.value,
        <int *>hip._util.types.DataHandle.from_pyobj(value)._ptr))    # fully specified
    return (_hipfftGetProperty__retval,)