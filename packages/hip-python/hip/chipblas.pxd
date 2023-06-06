# AMD_COPYRIGHT
from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .chip cimport hipStream_t
cdef extern from "hipblas/hipblas.h":

    cdef int hipblasVersionMajor

    cdef int hipblaseVersionMinor

    cdef int hipblasVersionMinor

    cdef int hipblasVersionPatch

    ctypedef short __int16_t

    ctypedef unsigned short __uint16_t

    ctypedef void * hipblasHandle_t

    ctypedef uint16_t hipblasHalf

    ctypedef int8_t hipblasInt8

    ctypedef int64_t hipblasStride

    cdef struct hipblasBfloat16:
        unsigned short data

    cdef struct hipblasComplex:
        float x
        float y

    cdef struct hipblasDoubleComplex:
        double x
        double y

    ctypedef enum hipblasStatus_t:
        HIPBLAS_STATUS_SUCCESS
        HIPBLAS_STATUS_NOT_INITIALIZED
        HIPBLAS_STATUS_ALLOC_FAILED
        HIPBLAS_STATUS_INVALID_VALUE
        HIPBLAS_STATUS_MAPPING_ERROR
        HIPBLAS_STATUS_EXECUTION_FAILED
        HIPBLAS_STATUS_INTERNAL_ERROR
        HIPBLAS_STATUS_NOT_SUPPORTED
        HIPBLAS_STATUS_ARCH_MISMATCH
        HIPBLAS_STATUS_HANDLE_IS_NULLPTR
        HIPBLAS_STATUS_INVALID_ENUM
        HIPBLAS_STATUS_UNKNOWN

    ctypedef enum hipblasOperation_t:
        HIPBLAS_OP_N
        HIPBLAS_OP_T
        HIPBLAS_OP_C

    ctypedef enum hipblasPointerMode_t:
        HIPBLAS_POINTER_MODE_HOST
        HIPBLAS_POINTER_MODE_DEVICE

    ctypedef enum hipblasFillMode_t:
        HIPBLAS_FILL_MODE_UPPER
        HIPBLAS_FILL_MODE_LOWER
        HIPBLAS_FILL_MODE_FULL

    ctypedef enum hipblasDiagType_t:
        HIPBLAS_DIAG_NON_UNIT
        HIPBLAS_DIAG_UNIT

    ctypedef enum hipblasSideMode_t:
        HIPBLAS_SIDE_LEFT
        HIPBLAS_SIDE_RIGHT
        HIPBLAS_SIDE_BOTH

    ctypedef enum hipblasDatatype_t:
        HIPBLAS_R_16F
        HIPBLAS_R_32F
        HIPBLAS_R_64F
        HIPBLAS_C_16F
        HIPBLAS_C_32F
        HIPBLAS_C_64F
        HIPBLAS_R_8I
        HIPBLAS_R_8U
        HIPBLAS_R_32I
        HIPBLAS_R_32U
        HIPBLAS_C_8I
        HIPBLAS_C_8U
        HIPBLAS_C_32I
        HIPBLAS_C_32U
        HIPBLAS_R_16B
        HIPBLAS_C_16B

    ctypedef enum hipblasGemmAlgo_t:
        HIPBLAS_GEMM_DEFAULT

    ctypedef enum hipblasAtomicsMode_t:
        HIPBLAS_ATOMICS_NOT_ALLOWED
        HIPBLAS_ATOMICS_ALLOWED

    ctypedef enum hipblasInt8Datatype_t:
        HIPBLAS_INT8_DATATYPE_DEFAULT
        HIPBLAS_INT8_DATATYPE_INT8
        HIPBLAS_INT8_DATATYPE_PACK_INT8x4

# \brief Create hipblas handle. */
cdef hipblasStatus_t hipblasCreate(void ** handle) nogil


# \brief Destroys the library context created using hipblasCreate() */
cdef hipblasStatus_t hipblasDestroy(void * handle) nogil


# \brief Set stream for handle */
cdef hipblasStatus_t hipblasSetStream(void * handle,hipStream_t streamId) nogil


# \brief Get stream[0] for handle */
cdef hipblasStatus_t hipblasGetStream(void * handle,hipStream_t* streamId) nogil


# \brief Set hipblas pointer mode */
cdef hipblasStatus_t hipblasSetPointerMode(void * handle,hipblasPointerMode_t mode) nogil


# \brief Get hipblas pointer mode */
cdef hipblasStatus_t hipblasGetPointerMode(void * handle,hipblasPointerMode_t * mode) nogil


# \brief Set hipblas int8 Datatype */
cdef hipblasStatus_t hipblasSetInt8Datatype(void * handle,hipblasInt8Datatype_t int8Type) nogil


# \brief Get hipblas int8 Datatype*/
cdef hipblasStatus_t hipblasGetInt8Datatype(void * handle,hipblasInt8Datatype_t * int8Type) nogil


# \brief copy vector from host to device
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize    [int]
#             Size of both vectors in bytes
# @param[in]
# x           pointer to vector on the host
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the device
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
#   ******************************************************************
cdef hipblasStatus_t hipblasSetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil


# \brief copy vector from device to host
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize    [int]
#             Size of both vectors in bytes
# @param[in]
# x           pointer to vector on the device
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the host
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
#   ******************************************************************
cdef hipblasStatus_t hipblasGetVector(int n,int elemSize,const void * x,int incx,void * y,int incy) nogil


# \brief copy matrix from host to device
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP          pointer to matrix on the host
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP           pointer to matrix on the GPU
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
#   ******************************************************************
cdef hipblasStatus_t hipblasSetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil


# \brief copy matrix from device to host
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP          pointer to matrix on the GPU
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP          pointer to matrix on the host
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
#   ******************************************************************
cdef hipblasStatus_t hipblasGetMatrix(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb) nogil


# \brief asynchronously copy vector from host to device
# \details
# hipblasSetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# x           pointer to vector on the host
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the device
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
# @param[in]
# stream      specifies the stream into which this transfer request is queued
#   ******************************************************************
cdef hipblasStatus_t hipblasSetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil


# \brief asynchronously copy vector from device to host
# \details
# hipblasGetVectorAsync copies a vector from pinned host memory to device memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# n           [int]
#             number of elements in the vector
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# x           pointer to vector on the device
# @param[in]
# incx        [int]
#             specifies the increment for the elements of the vector
# @param[out]
# y           pointer to vector on the host
# @param[in]
# incy        [int]
#             specifies the increment for the elements of the vector
# @param[in]
# stream      specifies the stream into which this transfer request is queued
#   ******************************************************************
cdef hipblasStatus_t hipblasGetVectorAsync(int n,int elemSize,const void * x,int incx,void * y,int incy,hipStream_t stream) nogil


# \brief asynchronously copy matrix from host to device
# \details
# hipblasSetMatrixAsync copies a matrix from pinned host memory to device memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP           pointer to matrix on the host
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP           pointer to matrix on the GPU
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
# @param[in]
# stream      specifies the stream into which this transfer request is queued
#   ******************************************************************
cdef hipblasStatus_t hipblasSetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil


# \brief asynchronously copy matrix from device to host
# \details
# hipblasGetMatrixAsync copies a matrix from device memory to pinned host memory asynchronously.
# Memory on the host must be allocated with hipHostMalloc or the transfer will be synchronous.
# @param[in]
# rows        [int]
#             number of rows in matrices
# @param[in]
# cols        [int]
#             number of columns in matrices
# @param[in]
# elemSize   [int]
#             number of bytes per element in the matrix
# @param[in]
# AP          pointer to matrix on the GPU
# @param[in]
# lda         [int]
#             specifies the leading dimension of A, lda >= rows
# @param[out]
# BP           pointer to matrix on the host
# @param[in]
# ldb         [int]
#             specifies the leading dimension of B, ldb >= rows
# @param[in]
# stream      specifies the stream into which this transfer request is queued
#   ******************************************************************
cdef hipblasStatus_t hipblasGetMatrixAsync(int rows,int cols,int elemSize,const void * AP,int lda,void * BP,int ldb,hipStream_t stream) nogil


# \brief Set hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasSetAtomicsMode(void * handle,hipblasAtomicsMode_t atomics_mode) nogil


# \brief Get hipblasSetAtomicsMode*/
cdef hipblasStatus_t hipblasGetAtomicsMode(void * handle,hipblasAtomicsMode_t * atomics_mode) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# amax finds the first index of the element of maximum magnitude of a vector x.
# 
# - Supported precisions in rocBLAS : s,d,c,z.
# - Supported precisions in cuBLAS  : s,d,c,z.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the amax index.
#           return is 0.0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasIsamax(void * handle,int n,const float * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIdamax(void * handle,int n,const double * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIcamax(void * handle,int n,hipblasComplex * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIzamax(void * handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
#  amaxBatched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z.
# - Supported precisions in cuBLAS  : No support.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch, must be > 0.
# @param[out]
# result
#           device or host array of pointers of batchCount size for results.
#           return is 0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasIsamaxBatched(void * handle,int n,const float *const* x,int incx,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIdamaxBatched(void * handle,int n,const double *const* x,int incx,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIcamaxBatched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIzamaxBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,int * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
#  amaxStridedBatched finds the first index of the element of maximum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between one x_i and the next x_(i + 1).
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device or host pointer for storing contiguous batchCount results.
#           return is 0 if n <= 0, incx<=0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasIsamaxStridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIdamaxStridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIcamaxStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIzamaxStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,int * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# amin finds the first index of the element of minimum magnitude of a vector x.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the amin index.
#           return is 0.0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasIsamin(void * handle,int n,const float * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIdamin(void * handle,int n,const double * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIcamin(void * handle,int n,hipblasComplex * x,int incx,int * result) nogil



cdef hipblasStatus_t hipblasIzamin(void * handle,int n,hipblasDoubleComplex * x,int incx,int * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# aminBatched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch, must be > 0.
# @param[out]
# result
#           device or host pointers to array of batchCount size for results.
#           return is 0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasIsaminBatched(void * handle,int n,const float *const* x,int incx,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIdaminBatched(void * handle,int n,const double *const* x,int incx,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIcaminBatched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIzaminBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,int * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
#  aminStridedBatched finds the first index of the element of minimum magnitude of each vector x_i in a batch, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between one x_i and the next x_(i + 1)
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device or host pointer to array for storing contiguous batchCount results.
#           return is 0 if n <= 0, incx<=0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasIsaminStridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIdaminStridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIcaminStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,int * result) nogil



cdef hipblasStatus_t hipblasIzaminStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,int * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# asum computes the sum of the magnitudes of elements of a real vector x,
#      or the sum of magnitudes of the real and imaginary parts of elements if x is a complex vector.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x. incx must be > 0.
# @param[inout]
# result
#           device pointer or host pointer to store the asum product.
#           return is 0.0 if n <= 0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSasum(void * handle,int n,const float * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDasum(void * handle,int n,const double * x,int incx,double * result) nogil



cdef hipblasStatus_t hipblasScasum(void * handle,int n,hipblasComplex * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDzasum(void * handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# asumBatched computes the sum of the magnitudes of the elements in a batch of real vectors x_i,
#     or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
#     vector, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch.
# @param[out]
# result
#           device array or host array of batchCount size for results.
#           return is 0.0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasSasumBatched(void * handle,int n,const float *const* x,int incx,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDasumBatched(void * handle,int n,const double *const* x,int incx,int batchCount,double * result) nogil



cdef hipblasStatus_t hipblasScasumBatched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDzasumBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,double * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# asumStridedBatched computes the sum of the magnitudes of elements of a real vectors x_i,
#     or the sum of magnitudes of the real and imaginary parts of elements if x_i is a complex
#     vector, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each vector x_i
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array for storing contiguous batchCount results.
#           return is 0.0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasSasumStridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDasumStridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,double * result) nogil



cdef hipblasStatus_t hipblasScasumStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDzasumStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,double * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# axpy   computes constant alpha multiplied by vector x, plus vector y
# 
#     y := alpha * x + y
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[out]
# y         device pointer storing vector y.
# @param[inout]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHaxpy(void * handle,int n,const unsigned short * alpha,const unsigned short * x,int incx,unsigned short * y,int incy) nogil



cdef hipblasStatus_t hipblasSaxpy(void * handle,int n,const float * alpha,const float * x,int incx,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDaxpy(void * handle,int n,const double * alpha,const double * x,int incx,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCaxpy(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZaxpy(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# axpyBatched   compute y := alpha * x + y over a set of batched vectors.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# alpha     specifies the scalar alpha.
# @param[in]
# x         pointer storing vector x on the GPU.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[out]
# y         pointer storing vector y on the GPU.
# @param[inout]
# incy      [int]
#           specifies the increment for the elements of y.
# 
# @param[in]
# batchCount [int]
#           number of instances in the batch
#  ******************************************************************
cdef hipblasStatus_t hipblasHaxpyBatched(void * handle,int n,const unsigned short * alpha,const unsigned short *const* x,int incx,unsigned short *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasSaxpyBatched(void * handle,int n,const float * alpha,const float *const* x,int incx,float *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDaxpyBatched(void * handle,int n,const double * alpha,const double *const* x,int incx,double *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasCaxpyBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZaxpyBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# axpyStridedBatched   compute y := alpha * x + y over a set of strided batched vectors.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
# @param[in]
# alpha     specifies the scalar alpha.
# @param[in]
# x         pointer storing vector x on the GPU.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# stridex   [hipblasStride]
#           specifies the increment between vectors of x.
# @param[out]
# y         pointer storing vector y on the GPU.
# @param[inout]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey   [hipblasStride]
#           specifies the increment between vectors of y.
# 
# @param[in]
# batchCount [int]
#           number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHaxpyStridedBatched(void * handle,int n,const unsigned short * alpha,const unsigned short * x,int incx,long stridex,unsigned short * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasSaxpyStridedBatched(void * handle,int n,const float * alpha,const float * x,int incx,long stridex,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDaxpyStridedBatched(void * handle,int n,const double * alpha,const double * x,int incx,long stridex,double * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasCaxpyStridedBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZaxpyStridedBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# copy  copies each element x[i] into y[i], for  i = 1 , ... , n
# 
#     y := x,
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x to be copied to y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[out]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasScopy(void * handle,int n,const float * x,int incx,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDcopy(void * handle,int n,const double * x,int incx,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCcopy(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZcopy(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# copyBatched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batchCount
# 
#     y_i := x_i,
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i to be copied to y_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i.
# @param[out]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasScopyBatched(void * handle,int n,const float *const* x,int incx,float *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDcopyBatched(void * handle,int n,const double *const* x,int incx,double *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasCcopyBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZcopyBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# copyStridedBatched copies each element x_i[j] into y_i[j], for  j = 1 , ... , n; i = 1 , ... , batchCount
# 
#     y_i := x_i,
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i to be copied to y_i.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx      [int]
#           specifies the increments for the elements of vectors x_i.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stride_x, however the user should
#             take care to ensure that stride_x is of appropriate size, for a typical
#             case this means stride_x >= n * incx.
# @param[out]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of vectors y_i.
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stride_y, however the user should
#             take care to ensure that stride_y is of appropriate size, for a typical
#             case this means stride_y >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasScopyStridedBatched(void * handle,int n,const float * x,int incx,long stridex,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDcopyStridedBatched(void * handle,int n,const double * x,int incx,long stridex,double * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasCcopyStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZcopyStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# dot(u)  performs the dot product of vectors x and y
# 
#     result = x * y;
# 
# dotc  performs the dot product of the conjugate of complex vector x and complex vector y
# 
#     result = conjugate (x) * y;
# 
# - Supported precisions in rocBLAS : h,bf,s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the dot product.
#           return is 0.0 if n <= 0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHdot(void * handle,int n,const unsigned short * x,int incx,const unsigned short * y,int incy,unsigned short * result) nogil



cdef hipblasStatus_t hipblasBfdot(void * handle,int n,hipblasBfloat16 * x,int incx,hipblasBfloat16 * y,int incy,hipblasBfloat16 * result) nogil



cdef hipblasStatus_t hipblasSdot(void * handle,int n,const float * x,int incx,const float * y,int incy,float * result) nogil



cdef hipblasStatus_t hipblasDdot(void * handle,int n,const double * x,int incx,const double * y,int incy,double * result) nogil



cdef hipblasStatus_t hipblasCdotc(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasCdotu(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasZdotc(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil



cdef hipblasStatus_t hipblasZdotu(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# dotBatched(u) performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotcBatched  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : h,bf,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHdotBatched(void * handle,int n,const unsigned short *const* x,int incx,const unsigned short *const* y,int incy,int batchCount,unsigned short * result) nogil



cdef hipblasStatus_t hipblasBfdotBatched(void * handle,int n,hipblasBfloat16 *const* x,int incx,hipblasBfloat16 *const* y,int incy,int batchCount,hipblasBfloat16 * result) nogil



cdef hipblasStatus_t hipblasSdotBatched(void * handle,int n,const float *const* x,int incx,const float *const* y,int incy,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDdotBatched(void * handle,int n,const double *const* x,int incx,const double *const* y,int incy,int batchCount,double * result) nogil



cdef hipblasStatus_t hipblasCdotcBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasCdotuBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,int batchCount,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasZdotcBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount,hipblasDoubleComplex * result) nogil



cdef hipblasStatus_t hipblasZdotuBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,int batchCount,hipblasDoubleComplex * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# dotStridedBatched(u)  performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotcStridedBatched  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : h,bf,s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHdotStridedBatched(void * handle,int n,const unsigned short * x,int incx,long stridex,const unsigned short * y,int incy,long stridey,int batchCount,unsigned short * result) nogil



cdef hipblasStatus_t hipblasBfdotStridedBatched(void * handle,int n,hipblasBfloat16 * x,int incx,long stridex,hipblasBfloat16 * y,int incy,long stridey,int batchCount,hipblasBfloat16 * result) nogil



cdef hipblasStatus_t hipblasSdotStridedBatched(void * handle,int n,const float * x,int incx,long stridex,const float * y,int incy,long stridey,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDdotStridedBatched(void * handle,int n,const double * x,int incx,long stridex,const double * y,int incy,long stridey,int batchCount,double * result) nogil



cdef hipblasStatus_t hipblasCdotcStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasCdotuStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount,hipblasComplex * result) nogil



cdef hipblasStatus_t hipblasZdotcStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount,hipblasDoubleComplex * result) nogil



cdef hipblasStatus_t hipblasZdotuStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount,hipblasDoubleComplex * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# nrm2 computes the euclidean norm of a real or complex vector
# 
#           result := sqrt( x'*x ) for real vectors
#           result := sqrt( x**H*x ) for complex vectors
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : s,d,sc,dz
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the nrm2 product.
#           return is 0.0 if n, incx<=0.
#  ******************************************************************
cdef hipblasStatus_t hipblasSnrm2(void * handle,int n,const float * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDnrm2(void * handle,int n,const double * x,int incx,double * result) nogil



cdef hipblasStatus_t hipblasScnrm2(void * handle,int n,hipblasComplex * x,int incx,float * result) nogil



cdef hipblasStatus_t hipblasDznrm2(void * handle,int n,hipblasDoubleComplex * x,int incx,double * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# nrm2Batched computes the euclidean norm over a batch of real or complex vectors
# 
#           result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array of batchCount size for nrm2 results.
#           return is 0.0 for each element if n <= 0, incx<=0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSnrm2Batched(void * handle,int n,const float *const* x,int incx,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDnrm2Batched(void * handle,int n,const double *const* x,int incx,int batchCount,double * result) nogil



cdef hipblasStatus_t hipblasScnrm2Batched(void * handle,int n,hipblasComplex *const* x,int incx,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDznrm2Batched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,int batchCount,double * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# nrm2StridedBatched computes the euclidean norm over a batch of real or complex vectors
# 
#           := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array for storing contiguous batchCount results.
#           return is 0.0 for each element if n <= 0, incx<=0.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSnrm2StridedBatched(void * handle,int n,const float * x,int incx,long stridex,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDnrm2StridedBatched(void * handle,int n,const double * x,int incx,long stridex,int batchCount,double * result) nogil



cdef hipblasStatus_t hipblasScnrm2StridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,int batchCount,float * result) nogil



cdef hipblasStatus_t hipblasDznrm2StridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,int batchCount,double * result) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rot applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# - Supported precisions in rocBLAS : s,d,c,z,sc,dz
# - Supported precisions in cuBLAS  : s,d,c,z,cs,zd
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer storing vector x.
# @param[in]
# incx    [int]
#         specifies the increment between elements of x.
# @param[inout]
# y       device pointer storing vector y.
# @param[in]
# incy    [int]
#         specifies the increment between elements of y.
# @param[in]
# c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer storing scalar sine component of the rotation matrix.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrot(void * handle,int n,float * x,int incx,float * y,int incy,const float * c,const float * s) nogil



cdef hipblasStatus_t hipblasDrot(void * handle,int n,double * x,int incx,double * y,int incy,const double * c,const double * s) nogil



cdef hipblasStatus_t hipblasCrot(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,hipblasComplex * s) nogil



cdef hipblasStatus_t hipblasCsrot(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy,const float * c,const float * s) nogil



cdef hipblasStatus_t hipblasZrot(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,hipblasDoubleComplex * s) nogil



cdef hipblasStatus_t hipblasZdrot(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,const double * c,const double * s) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotBatched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# - Supported precisions in rocBLAS : s,d,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device array of deivce pointers storing each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[inout]
# y       device array of device pointers storing each vector y_i.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotBatched(void * handle,int n,float *const* x,int incx,float *const* y,int incy,const float * c,const float * s,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotBatched(void * handle,int n,double *const* x,int incx,double *const* y,int incy,const double * c,const double * s,int batchCount) nogil



cdef hipblasStatus_t hipblasCrotBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,const float * c,hipblasComplex * s,int batchCount) nogil



cdef hipblasStatus_t hipblasCsrotBatched(void * handle,int n,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,const float * c,const float * s,int batchCount) nogil



cdef hipblasStatus_t hipblasZrotBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,const double * c,hipblasDoubleComplex * s,int batchCount) nogil



cdef hipblasStatus_t hipblasZdrotBatched(void * handle,int n,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,const double * c,const double * s,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotStridedBatched applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# - Supported precisions in rocBLAS : s,d,sc,dz
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device pointer to the first vector x_1.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[in]
# stridex [hipblasStride]
#          specifies the increment from the beginning of x_i to the beginning of x_(i+1)
# @param[inout]
# y       device pointer to the first vector y_1.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#          specifies the increment from the beginning of y_i to the beginning of y_(i+1)
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# batchCount [int]
#         the number of x and y arrays, i.e. the number of batches.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotStridedBatched(void * handle,int n,float * x,int incx,long stridex,float * y,int incy,long stridey,const float * c,const float * s,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotStridedBatched(void * handle,int n,double * x,int incx,long stridex,double * y,int incy,long stridey,const double * c,const double * s,int batchCount) nogil



cdef hipblasStatus_t hipblasCrotStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,const float * c,hipblasComplex * s,int batchCount) nogil



cdef hipblasStatus_t hipblasCsrotStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,const float * c,const float * s,int batchCount) nogil



cdef hipblasStatus_t hipblasZrotStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,const double * c,hipblasDoubleComplex * s,int batchCount) nogil



cdef hipblasStatus_t hipblasZdrotStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,const double * c,const double * s,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotg creates the Givens rotation matrix for the vector (a b).
#      Scalars c and s and arrays a and b may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# a       device pointer or host pointer to input vector element, overwritten with r.
# @param[inout]
# b       device pointer or host pointer to input vector element, overwritten with z.
# @param[inout]
# c       device pointer or host pointer to cosine element of Givens rotation.
# @param[inout]
# s       device pointer or host pointer sine element of Givens rotation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotg(void * handle,float * a,float * b,float * c,float * s) nogil



cdef hipblasStatus_t hipblasDrotg(void * handle,double * a,double * b,double * c,double * s) nogil



cdef hipblasStatus_t hipblasCrotg(void * handle,hipblasComplex * a,hipblasComplex * b,float * c,hipblasComplex * s) nogil



cdef hipblasStatus_t hipblasZrotg(void * handle,hipblasDoubleComplex * a,hipblasDoubleComplex * b,double * c,hipblasDoubleComplex * s) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotgBatched creates the Givens rotation matrix for the batched vectors (a_i b_i), for i = 1, ..., batchCount.
#      a, b, c, and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# a       device array of device pointers storing each single input vector element a_i, overwritten with r_i.
# @param[inout]
# b       device array of device pointers storing each single input vector element b_i, overwritten with z_i.
# @param[inout]
# c       device array of device pointers storing each cosine element of Givens rotation for the batch.
# @param[inout]
# s       device array of device pointers storing each sine element of Givens rotation for the batch.
# @param[in]
# batchCount [int]
#             number of batches (length of arrays a, b, c, and s).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotgBatched(void * handle,float *const* a,float *const* b,float *const* c,float *const* s,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotgBatched(void * handle,double *const* a,double *const* b,double *const* c,double *const* s,int batchCount) nogil



cdef hipblasStatus_t hipblasCrotgBatched(void * handle,hipblasComplex *const* a,hipblasComplex *const* b,float *const* c,hipblasComplex *const* s,int batchCount) nogil



cdef hipblasStatus_t hipblasZrotgBatched(void * handle,hipblasDoubleComplex *const* a,hipblasDoubleComplex *const* b,double *const* c,hipblasDoubleComplex *const* s,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotgStridedBatched creates the Givens rotation matrix for the strided batched vectors (a_i b_i), for i = 1, ..., batchCount.
#      a, b, c, and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#      If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# a       device strided_batched pointer or host strided_batched pointer to first single input vector element a_1, overwritten with r.
# @param[in]
# stridea [hipblasStride]
#          distance between elements of a in batch (distance between a_i and a_(i + 1))
# @param[inout]
# b       device strided_batched pointer or host strided_batched pointer to first single input vector element b_1, overwritten with z.
# @param[in]
# strideb [hipblasStride]
#          distance between elements of b in batch (distance between b_i and b_(i + 1))
# @param[inout]
# c       device strided_batched pointer or host strided_batched pointer to first cosine element of Givens rotations c_1.
# @param[in]
# stridec [hipblasStride]
#          distance between elements of c in batch (distance between c_i and c_(i + 1))
# @param[inout]
# s       device strided_batched pointer or host strided_batched pointer to sine element of Givens rotations s_1.
# @param[in]
# strides [hipblasStride]
#          distance between elements of s in batch (distance between s_i and s_(i + 1))
# @param[in]
# batchCount [int]
#             number of batches (length of arrays a, b, c, and s).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotgStridedBatched(void * handle,float * a,long stridea,float * b,long strideb,float * c,long stridec,float * s,long strides,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotgStridedBatched(void * handle,double * a,long stridea,double * b,long strideb,double * c,long stridec,double * s,long strides,int batchCount) nogil



cdef hipblasStatus_t hipblasCrotgStridedBatched(void * handle,hipblasComplex * a,long stridea,hipblasComplex * b,long strideb,float * c,long stridec,hipblasComplex * s,long strides,int batchCount) nogil



cdef hipblasStatus_t hipblasZrotgStridedBatched(void * handle,hipblasDoubleComplex * a,long stridea,hipblasDoubleComplex * b,long strideb,double * c,long stridec,hipblasDoubleComplex * s,long strides,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotm applies the modified Givens rotation matrix defined by param to vectors x and y.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer storing vector x.
# @param[in]
# incx    [int]
#         specifies the increment between elements of x.
# @param[inout]
# y       device pointer storing vector y.
# @param[in]
# incy    [int]
#         specifies the increment between elements of y.
# @param[in]
# param   device vector or host vector of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotm(void * handle,int n,float * x,int incx,float * y,int incy,const float * param) nogil



cdef hipblasStatus_t hipblasDrotm(void * handle,int n,double * x,int incx,double * y,int incy,const double * param) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmBatched applies the modified Givens rotation matrix defined by param_i to batched vectors x_i and y_i, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device array of device pointers storing each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[inout]
# y       device array of device pointers storing each vector y_1.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# param   device array of device vectors of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may ONLY be stored on the device for the batched version of this function.
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotmBatched(void * handle,int n,float *const* x,int incx,float *const* y,int incy,const float *const* param,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotmBatched(void * handle,int n,double *const* x,int incx,double *const* y,int incy,const double *const* param,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmStridedBatched applies the modified Givens rotation matrix defined by param_i to strided batched vectors x_i and y_i, for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer pointing to first strided batched vector x_1.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[in]
# stridex [hipblasStride]
#          specifies the increment between the beginning of x_i and x_(i + 1)
# @param[inout]
# y       device pointer pointing to first strided batched vector y_1.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#          specifies the increment between the beginning of y_i and y_(i + 1)
# @param[in]
# param   device pointer pointing to first array of 5 elements defining the rotation (param_1).
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may ONLY be stored on the device for the strided_batched version of this function.
# @param[in]
# strideParam [hipblasStride]
#              specifies the increment between the beginning of param_i and param_(i + 1)
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotmStridedBatched(void * handle,int n,float * x,int incx,long stridex,float * y,int incy,long stridey,const float * param,long strideParam,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotmStridedBatched(void * handle,int n,double * x,int incx,long stridex,double * y,int incy,long stridey,const double * param,long strideParam,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmg creates the modified Givens rotation matrix for the vector (d1 * x1, d2 * y1).
#       Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# d1      device pointer or host pointer to input scalar that is overwritten.
# @param[inout]
# d2      device pointer or host pointer to input scalar that is overwritten.
# @param[inout]
# x1      device pointer or host pointer to input scalar that is overwritten.
# @param[in]
# y1      device pointer or host pointer to input scalar.
# @param[out]
# param   device vector or host vector of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotmg(void * handle,float * d1,float * d2,float * x1,const float * y1,float * param) nogil



cdef hipblasStatus_t hipblasDrotmg(void * handle,double * d1,double * d2,double * x1,const double * y1,double * param) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmgBatched creates the modified Givens rotation matrix for the batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batchCount.
#       Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# d1      device batched array or host batched array of input scalars that is overwritten.
# @param[inout]
# d2      device batched array or host batched array of input scalars that is overwritten.
# @param[inout]
# x1      device batched array or host batched array of input scalars that is overwritten.
# @param[in]
# y1      device batched array or host batched array of input scalars.
# @param[out]
# param   device batched array or host batched array of vectors of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# @param[in]
# batchCount [int]
#             the number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotmgBatched(void * handle,float *const* d1,float *const* d2,float *const* x1,const float *const* y1,float *const* param,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotmgBatched(void * handle,double *const* d1,double *const* d2,double *const* x1,const double *const* y1,double *const* param,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# rotmgStridedBatched creates the modified Givens rotation matrix for the strided batched vectors (d1_i * x1_i, d2_i * y1_i), for i = 1, ..., batchCount.
#       Parameters may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_HOST, this function blocks the CPU until the GPU has finished and the results are available in host memory.
#       If the pointer mode is set to HIPBLAS_POINTER_MODE_DEVICE, this function returns immediately and synchronization is required to read the results.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[inout]
# d1      device strided_batched array or host strided_batched array of input scalars that is overwritten.
# @param[in]
# strided1 [hipblasStride]
#           specifies the increment between the beginning of d1_i and d1_(i+1)
# @param[inout]
# d2      device strided_batched array or host strided_batched array of input scalars that is overwritten.
# @param[in]
# strided2 [hipblasStride]
#           specifies the increment between the beginning of d2_i and d2_(i+1)
# @param[inout]
# x1      device strided_batched array or host strided_batched array of input scalars that is overwritten.
# @param[in]
# stridex1 [hipblasStride]
#           specifies the increment between the beginning of x1_i and x1_(i+1)
# @param[in]
# y1      device strided_batched array or host strided_batched array of input scalars.
# @param[in]
# stridey1 [hipblasStride]
#           specifies the increment between the beginning of y1_i and y1_(i+1)
# @param[out]
# param   device stridedBatched array or host stridedBatched array of vectors of 5 elements defining the rotation.
#         param[0] = flag
#         param[1] = H11
#         param[2] = H21
#         param[3] = H12
#         param[4] = H22
#         The flag parameter defines the form of H:
#         flag = -1 => H = ( H11 H12 H21 H22 )
#         flag =  0 => H = ( 1.0 H12 H21 1.0 )
#         flag =  1 => H = ( H11 1.0 -1.0 H22 )
#         flag = -2 => H = ( 1.0 0.0 0.0 1.0 )
#         param may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# @param[in]
# strideParam [hipblasStride]
#              specifies the increment between the beginning of param_i and param_(i + 1)
# @param[in]
# batchCount [int]
#             the number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSrotmgStridedBatched(void * handle,float * d1,long strided1,float * d2,long strided2,float * x1,long stridex1,const float * y1,long stridey1,float * param,long strideParam,int batchCount) nogil



cdef hipblasStatus_t hipblasDrotmgStridedBatched(void * handle,double * d1,long strided1,double * d2,long strided2,double * x1,long stridex1,const double * y1,long stridey1,double * param,long strideParam,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# scal  scales each element of vector x with scalar alpha.
# 
#     x := alpha * x
# 
# - Supported precisions in rocBLAS : s,d,c,z,cs,zd
# - Supported precisions in cuBLAS  : s,d,c,z,cs,zd
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSscal(void * handle,int n,const float * alpha,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDscal(void * handle,int n,const double * alpha,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCscal(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasCsscal(void * handle,int n,const float * alpha,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZscal(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZdscal(void * handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 1 API
# \details
# scalBatched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batchCount.
# 
#      x_i := alpha * x_i
# 
#  where (x_i) is the i-th instance of the batch.
# 
# - Supported precisions in rocBLAS : s,d,c,z,cs,zd
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# n           [int]
#             the number of elements in each x_i.
# @param[in]
# alpha       host pointer or device pointer for the scalar alpha.
# @param[inout]
# x           device array of device pointers storing each vector x_i.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             specifies the number of batches in x.
#   ******************************************************************
cdef hipblasStatus_t hipblasSscalBatched(void * handle,int n,const float * alpha,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDscalBatched(void * handle,int n,const double * alpha,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCscalBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZscalBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCsscalBatched(void * handle,int n,const float * alpha,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZdscalBatched(void * handle,int n,const double * alpha,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# \details
# scalStridedBatched  scales each element of vector x_i with scalar alpha, for i = 1, ... , batchCount.
# 
#      x_i := alpha * x_i ,
# 
#  where (x_i) is the i-th instance of the batch.
# 
# - Supported precisions in rocBLAS : s,d,c,z,cs,zd
# - Supported precisions in cuBLAS  : No support
# 
#  @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# n           [int]
#             the number of elements in each x_i.
# @param[in]
# alpha       host pointer or device pointer for the scalar alpha.
# @param[inout]
# x           device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of x.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stride_x, however the user should
#             take care to ensure that stride_x is of appropriate size, for a typical
#             case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#             specifies the number of batches in x.
#   ******************************************************************
cdef hipblasStatus_t hipblasSscalStridedBatched(void * handle,int n,const float * alpha,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDscalStridedBatched(void * handle,int n,const double * alpha,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCscalStridedBatched(void * handle,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZscalStridedBatched(void * handle,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCsscalStridedBatched(void * handle,int n,const float * alpha,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZdscalStridedBatched(void * handle,int n,const double * alpha,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# swap  interchanges vectors x and y.
# 
#     y := x; x := y
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSswap(void * handle,int n,float * x,int incx,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDswap(void * handle,int n,double * x,int incx,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCswap(void * handle,int n,hipblasComplex * x,int incx,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZswap(void * handle,int n,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# swapBatched interchanges vectors x_i and y_i, for i = 1 , ... , batchCount
# 
#     y_i := x_i; x_i := y_i
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[inout]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSswapBatched(void * handle,int n,float ** x,int incx,float ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDswapBatched(void * handle,int n,double ** x,int incx,double ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasCswapBatched(void * handle,int n,hipblasComplex ** x,int incx,hipblasComplex ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZswapBatched(void * handle,int n,hipblasDoubleComplex ** x,int incx,hipblasDoubleComplex ** y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 1 API
# 
# \details
# swapStridedBatched interchanges vectors x_i and y_i, for i = 1 , ... , batchCount
# 
#     y_i := x_i; x_i := y_i
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[inout]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[inout]
# y         device pointer to the first vector y_1.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey   [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_y is of appropriate size, for a typical
#           case this means stride_y >= n * incy. stridey should be non zero.
#  @param[in]
#  batchCount [int]
#              number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSswapStridedBatched(void * handle,int n,float * x,int incx,long stridex,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDswapStridedBatched(void * handle,int n,double * x,int incx,long stridex,double * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasCswapStridedBatched(void * handle,int n,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZswapStridedBatched(void * handle,int n,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gbmv performs one of the matrix-vector operations
# 
#     y := alpha*A*x    + beta*y,   or
#     y := alpha*A**T*x + beta*y,   or
#     y := alpha*A**H*x + beta*y,
# 
# where alpha and beta are scalars, x and y are vectors and A is an
# m by n banded matrix with kl sub-diagonals and ku super-diagonals.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of matrix A
# @param[in]
# n         [int]
#           number of columns of matrix A
# @param[in]
# kl        [int]
#           number of sub-diagonals of A
# @param[in]
# ku        [int]
#           number of super-diagonals of A
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
#     AP    device pointer storing banded matrix A.
#           Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#           of the banded matrix. The leading diagonal resides in row (ku + 1) with
#           the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#           resides below on the LHS of row ku + 2. This propogates up and down across
#           sub/super-diagonals.
#             Ex: (m = n = 7; ku = 2, kl = 2)
#             1 2 3 0 0 0 0             0 0 3 3 3 3 3
#             4 1 2 3 0 0 0             0 2 2 2 2 2 2
#             5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#             0 5 4 1 2 3 0             4 4 4 4 4 4 0
#             0 0 5 4 1 2 0             5 5 5 5 5 0 0
#             0 0 0 5 4 1 2             0 0 0 0 0 0 0
#             0 0 0 0 5 4 1             0 0 0 0 0 0 0
#           Note that the empty elements which don't correspond to data will not
#           be referenced.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be >= (kl + ku + 1)
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZgbmv(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gbmvBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n banded matrix with kl sub-diagonals and ku super-diagonals,
# for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of each matrix A_i
# @param[in]
# n         [int]
#           number of columns of each matrix A_i
# @param[in]
# kl        [int]
#           number of sub-diagonals of each A_i
# @param[in]
# ku        [int]
#           number of super-diagonals of each A_i
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
#     AP    device array of device pointers storing each banded matrix A_i.
#           Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#           of the banded matrix. The leading diagonal resides in row (ku + 1) with
#           the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#           resides below on the LHS of row ku + 2. This propogates up and down across
#           sub/super-diagonals.
#             Ex: (m = n = 7; ku = 2, kl = 2)
#             1 2 3 0 0 0 0             0 0 3 3 3 3 3
#             4 1 2 3 0 0 0             0 2 2 2 2 2 2
#             5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#             0 5 4 1 2 3 0             4 4 4 4 4 4 0
#             0 0 5 4 1 2 0             5 5 5 5 5 0 0
#             0 0 0 5 4 1 2             0 0 0 0 0 0 0
#             0 0 0 0 5 4 1             0 0 0 0 0 0 0
#           Note that the empty elements which don't correspond to data will not
#           be referenced.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be >= (kl + ku + 1)
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasCgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZgbmvBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gbmvStridedBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n banded matrix with kl sub-diagonals and ku super-diagonals,
# for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of matrix A
# @param[in]
# n         [int]
#           number of columns of matrix A
# @param[in]
# kl        [int]
#           number of sub-diagonals of A
# @param[in]
# ku        [int]
#           number of super-diagonals of A
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
#     AP    device pointer to first banded matrix (A_1).
#           Leading (kl + ku + 1) by n part of the matrix contains the coefficients
#           of the banded matrix. The leading diagonal resides in row (ku + 1) with
#           the first super-diagonal above on the RHS of row ku. The first sub-diagonal
#           resides below on the LHS of row ku + 2. This propogates up and down across
#           sub/super-diagonals.
#             Ex: (m = n = 7; ku = 2, kl = 2)
#             1 2 3 0 0 0 0             0 0 3 3 3 3 3
#             4 1 2 3 0 0 0             0 2 2 2 2 2 2
#             5 4 1 2 3 0 0    ---->    1 1 1 1 1 1 1
#             0 5 4 1 2 3 0             4 4 4 4 4 4 0
#             0 0 5 4 1 2 0             5 5 5 5 5 0 0
#             0 0 0 5 4 1 2             0 0 0 0 0 0 0
#             0 0 0 0 5 4 1             0 0 0 0 0 0 0
#           Note that the empty elements which don't correspond to data will not
#           be referenced.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be >= (kl + ku + 1)
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         device pointer to first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer to first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (x_i+1)
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasCgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZgbmvStridedBatched(void * handle,hipblasOperation_t trans,int m,int n,int kl,int ku,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gemv performs one of the matrix-vector operations
# 
#     y := alpha*A*x    + beta*y,   or
#     y := alpha*A**T*x + beta*y,   or
#     y := alpha*A**H*x + beta*y,
# 
# where alpha and beta are scalars, x and y are vectors and A is an
# m by n matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# trans     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not
# @param[in]
# m         [int]
#           number of rows of matrix A
# @param[in]
# n         [int]
#           number of columns of matrix A
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgemv(void * handle,hipblasOperation_t trans,int m,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDgemv(void * handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCgemv(void * handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZgemv(void * handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gemvBatched performs a batch of matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# trans       [hipblasOperation_t]
#             indicates whether matrices A_i are tranposed (conjugated) or not
# @param[in]
# m           [int]
#             number of rows of each matrix A_i
# @param[in]
# n           [int]
#             number of columns of each matrix A_i
# @param[in]
# alpha       device pointer or host pointer to scalar alpha.
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda         [int]
#             specifies the leading dimension of each matrix A_i.
# @param[in]
# x           device array of device pointers storing each vector x_i.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of each vector x_i.
# @param[in]
# beta        device pointer or host pointer to scalar beta.
# @param[inout]
# y           device array of device pointers storing each vector y_i.
# @param[in]
# incy        [int]
#             specifies the increment for the elements of each vector y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasCgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZgemvBatched(void * handle,hipblasOperation_t trans,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gemvStridedBatched performs a batch of matrix-vector operations
# 
#     y_i := alpha*A_i*x_i    + beta*y_i,   or
#     y_i := alpha*A_i**T*x_i + beta*y_i,   or
#     y_i := alpha*A_i**H*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle      [hipblasHandle_t]
#             handle to the hipblas library context queue.
# @param[in]
# transA      [hipblasOperation_t]
#             indicates whether matrices A_i are tranposed (conjugated) or not
# @param[in]
# m           [int]
#             number of rows of matrices A_i
# @param[in]
# n           [int]
#             number of columns of matrices A_i
# @param[in]
# alpha       device pointer or host pointer to scalar alpha.
# @param[in]
# AP          device pointer to the first matrix (A_1) in the batch.
# @param[in]
# lda         [int]
#             specifies the leading dimension of matrices A_i.
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x           device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx        [int]
#             specifies the increment for the elements of vectors x_i.
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size. When trans equals HIPBLAS_OP_N
#             this typically means stridex >= n * incx, otherwise stridex >= m * incx.
# @param[in]
# beta        device pointer or host pointer to scalar beta.
# @param[inout]
# y           device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy        [int]
#             specifies the increment for the elements of vectors y_i.
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size. When trans equals HIPBLAS_OP_N
#             this typically means stridey >= m * incy, otherwise stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasCgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZgemvStridedBatched(void * handle,hipblasOperation_t transA,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# ger,geru,gerc performs the matrix-vector operations
# 
#     A := A + alpha*x*y**T , OR
#     A := A + alpha*x*y**H for gerc
# 
# where alpha is a scalar, x and y are vectors, and A is an
# m by n matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# m         [int]
#           the number of rows of the matrix A.
# @param[in]
# n         [int]
#           the number of columns of the matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSger(void * handle,int m,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil



cdef hipblasStatus_t hipblasDger(void * handle,int m,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil



cdef hipblasStatus_t hipblasCgeru(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasCgerc(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZgeru(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZgerc(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gerBatched,geruBatched,gercBatched performs a batch of the matrix-vector operations
# 
#     A := A + alpha*x*y**T , OR
#     A := A + alpha*x*y**H for gerc
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha is a scalar, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# m         [int]
#           the number of rows of each matrix A_i.
# @param[in]
# n         [int]
#           the number of columns of eaceh matrix A_i.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i.
# @param[inout]
# AP        device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgerBatched(void * handle,int m,int n,const float * alpha,const float *const* x,int incx,const float *const* y,int incy,float *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasDgerBatched(void * handle,int m,int n,const double * alpha,const double *const* x,int incx,const double *const* y,int incy,double *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasCgeruBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasCgercBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasZgeruBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasZgercBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# gerStridedBatched,geruStridedBatched,gercStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**T, OR
#     A_i := A_i + alpha*x_i*y_i**H  for gerc
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha is a scalar, x_i and y_i are vectors and A_i is an
# m by n matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# m         [int]
#           the number of rows of each matrix A_i.
# @param[in]
# n         [int]
#           the number of columns of each matrix A_i.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# incx      [int]
#           specifies the increments for the elements of each vector x_i.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stridex, however the user should
#           take care to ensure that stridex is of appropriate size, for a typical
#           case this means stridex >= m * incx.
# @param[inout]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i.
# @param[in]
# stridey   [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
#           There are no restrictions placed on stridey, however the user should
#           take care to ensure that stridey is of appropriate size, for a typical
#           case this means stridey >= n * incy.
# @param[inout]
# AP        device pointer to the first matrix (A_1) in the batch.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgerStridedBatched(void * handle,int m,int n,const float * alpha,const float * x,int incx,long stridex,const float * y,int incy,long stridey,float * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasDgerStridedBatched(void * handle,int m,int n,const double * alpha,const double * x,int incx,long stridex,const double * y,int incy,long stridey,double * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasCgeruStridedBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasCgercStridedBatched(void * handle,int m,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZgeruStridedBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZgercStridedBatched(void * handle,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hbmv performs the matrix-vector operations
# 
#     y := alpha*A*x + beta*y
# 
# where alpha and beta are scalars, x and y are n element vectors and A is an
# n by n Hermitian band matrix, with k super-diagonals.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is being supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is being supplied.
# @param[in]
# n         [int]
#           the order of the matrix A.
# @param[in]
# k         [int]
#           the number of super-diagonals of the matrix A. Must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing matrix A. Of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The leading (k + 1) by n part of A must contain the upper
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (k + 1), the first super-diagonal on the RHS
#             of row k, etc.
#             The top left k by x triangle of A will not be referenced.
#                 Ex (upper, lda = n = 4, k = 1):
#                 A                             Represented matrix
#                 (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                 (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The leading (k + 1) by n part of A must contain the lower
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (1), the first sub-diagonal on the LHS of
#             row 2, etc.
#             The bottom right k by k triangle of A will not be referenced.
#                 Ex (lower, lda = 2, n = 4, k = 1):
#                 A                               Represented matrix
#                 (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                 (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                 (0, 0) (6, 8) (3, 0) (7,-7)
#                                                 (0, 0) (0, 0) (7, 7) (4, 0)
# 
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of A will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. must be >= k + 1
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChbmv(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZhbmv(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hbmvBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is being supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is being supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# k         [int]
#           the number of super-diagonals of each matrix A_i. Must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array of device pointers storing each matrix_i A of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The leading (k + 1) by n part of each A_i must contain the upper
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (k + 1), the first super-diagonal on the RHS
#             of row k, etc.
#             The top left k by x triangle of each A_i will not be referenced.
#                 Ex (upper, lda = n = 4, k = 1):
#                 A                             Represented matrix
#                 (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                 (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The leading (k + 1) by n part of each A_i must contain the lower
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (1), the first sub-diagonal on the LHS of
#             row 2, etc.
#             The bottom right k by k triangle of each A_i will not be referenced.
#                 Ex (lower, lda = 2, n = 4, k = 1):
#                 A                               Represented matrix
#                 (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                 (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                 (0, 0) (6, 8) (3, 0) (7,-7)
#                                                 (0, 0) (0, 0) (7, 7) (4, 0)
# 
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZhbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hbmvStridedBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian band matrix with k super-diagonals, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is being supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is being supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# k         [int]
#           the number of super-diagonals of each matrix A_i. Must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array pointing to the first matrix A_1. Each A_i is of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The leading (k + 1) by n part of each A_i must contain the upper
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (k + 1), the first super-diagonal on the RHS
#             of row k, etc.
#             The top left k by x triangle of each A_i will not be referenced.
#                 Ex (upper, lda = n = 4, k = 1):
#                 A                             Represented matrix
#                 (0,0) (5,9) (6,8) (7,7)       (1, 0) (5, 9) (0, 0) (0, 0)
#                 (1,0) (2,0) (3,0) (4,0)       (5,-9) (2, 0) (6, 8) (0, 0)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (6,-8) (3, 0) (7, 7)
#                 (0,0) (0,0) (0,0) (0,0)       (0, 0) (0, 0) (7,-7) (4, 0)
# 
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The leading (k + 1) by n part of each A_i must contain the lower
#             triangular band part of the Hermitian matrix, with the leading
#             diagonal in row (1), the first sub-diagonal on the LHS of
#             row 2, etc.
#             The bottom right k by k triangle of each A_i will not be referenced.
#                 Ex (lower, lda = 2, n = 4, k = 1):
#                 A                               Represented matrix
#                 (1,0) (2,0) (3,0) (4,0)         (1, 0) (5,-9) (0, 0) (0, 0)
#                 (5,9) (6,8) (7,7) (0,0)         (5, 9) (2, 0) (6,-8) (0, 0)
#                                                 (0, 0) (6, 8) (3, 0) (7,-7)
#                                                 (0, 0) (0, 0) (7, 7) (4, 0)
# 
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         device array pointing to the first vector y_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array pointing to the first vector y_1.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZhbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hemv performs one of the matrix-vector operations
# 
#     y := alpha*A*x + beta*y
# 
# where alpha and beta are scalars, x and y are n element vectors and A is an
# n by n Hermitian matrix.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
# @param[in]
# n         [int]
#           the order of the matrix A.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing matrix A. Of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular part of A must contain
#             the upper triangular part of a Hermitian matrix. The lower
#             triangular part of A will not be referenced.
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular part of A must contain
#             the lower triangular part of a Hermitian matrix. The upper
#             triangular part of A will not be referenced.
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of A will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. must be >= max(1, n)
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChemv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZhemv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hemvBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array of device pointers storing each matrix A_i of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular part of each A_i must contain
#             the upper triangular part of a Hermitian matrix. The lower
#             triangular part of each A_i will not be referenced.
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular part of each A_i must contain
#             the lower triangular part of a Hermitian matrix. The upper
#             triangular part of each A_i will not be referenced.
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChemvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZhemvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hemvStridedBatched performs one of the matrix-vector operations
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device array of device pointers storing each matrix A_i of dimension (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular part of each A_i must contain
#             the upper triangular part of a Hermitian matrix. The lower
#             triangular part of each A_i will not be referenced.
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular part of each A_i must contain
#             the lower triangular part of a Hermitian matrix. The upper
#             triangular part of each A_i will not be referenced.
#           As a Hermitian matrix, the imaginary part of the main diagonal
#           of each A_i will not be referenced and is assumed to be == 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. must be >= max(1, n)
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) to the next (A_i+1)
# 
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChemvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZhemvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# her performs the matrix-vector operations
# 
#     A := A + alpha*x*x**H
# 
# where alpha is a real scalar, x is a vector, and A is an
# n by n Hermitian matrix.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in A.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in A.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP        device pointer storing the specified triangular portion of
#           the Hermitian matrix A. Of size (lda * n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied. The lower
#             triangluar portion will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied. The upper
#             triangular portion will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be at least max(1, n).
#  ******************************************************************
cdef hipblasStatus_t hipblasCher(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZher(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# herBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in A.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in A.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP       device array of device pointers storing the specified triangular portion of
#           each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular portion
#             of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular portion
#             of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be at least max(1, n).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasCherBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasZherBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# herStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n Hermitian matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in A.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in A.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[inout]
# AP        device array of device pointers storing the specified triangular portion of
#           each Hermitian matrix A_i. Points to the first matrix (A_1).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
#             portion of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
#             portion of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasCherStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZherStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# her2 performs the matrix-vector operations
# 
#     A := A + alpha*x*y**H + conj(alpha)*y*x**H
# 
# where alpha is a complex scalar, x and y are vectors, and A is an
# n by n Hermitian matrix.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP         device pointer storing the specified triangular portion of
#           the Hermitian matrix A. Of size (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied. The lower triangular
#             portion of A will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied. The upper triangular
#             portion of A will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. Must be at least max(lda, 1).
#  ******************************************************************
cdef hipblasStatus_t hipblasCher2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZher2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# her2Batched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n Hermitian matrix for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP         device array of device pointers storing the specified triangular portion of
#           each Hermitian matrix A_i of size (lda, n).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
#             portion of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
#             portion of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be at least max(lda, 1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasCher2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasZher2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# her2StridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n Hermitian matrix for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           specifies the stride between the beginning of one vector (x_i) and the next (x_i+1).
# @param[in]
# y         device pointer pointing to the first vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#           specifies the stride between the beginning of one vector (y_i) and the next (y_i+1).
# @param[inout]
# AP        device pointer pointing to the first matrix (A_1). Stores the specified triangular portion of
#           each Hermitian matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied. The lower triangular
#             portion of each A_i will not be touched.
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied. The upper triangular
#             portion of each A_i will not be touched.
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. Must be at least max(lda, 1).
# @param[in]
# strideA  [hipblasStride]
#           specifies the stride between the beginning of one matrix (A_i) and the next (A_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasCher2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZher2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpmv performs the matrix-vector operation
# 
#     y := alpha*A*x + beta*y
# 
# where alpha and beta are scalars, x and y are n element vectors and A is an
# n by n Hermitian matrix, supplied in packed form (see description below).
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of the Hermitian matrix A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of the Hermitian matrix A is supplied in AP.
# @param[in]
# n         [int]
#           the order of the matrix A, must be >= 0.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChpmv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZhpmv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpmvBatched performs the matrix-vector operation
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, supplied in packed form (see description below),
# for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP      device pointer of device pointers storing the packed version of the specified triangular
#         portion of each Hermitian matrix A_i. Each A_i is of at least size ((n * (n + 1)) / 2).
#         if uplo == HIPBLAS_FILL_MODE_UPPER:
#         The upper triangular portion of each Hermitian matrix A_i is supplied.
#         The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#         so that:
#         AP(0) = A(0,0)
#         AP(1) = A(0,1)
#         AP(2) = A(1,1), etc.
#             Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                 (1, 0) (2, 1) (3, 2)
#                 (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                 (3,-2) (5, 1) (6, 0)
#     if uplo == HIPBLAS_FILL_MODE_LOWER:
#         The lower triangular portion of each Hermitian matrix A_i is supplied.
#         The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#         so that:
#         AP(0) = A(0,0)
#         AP(1) = A(1,0)
#         AP(2) = A(2,1), etc.
#             Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                 (1, 0) (2, 1) (3, 2)
#                 (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                 (3,-2) (5, 1) (6, 0)
#     Note that the imaginary part of the diagonal elements are not accessed and are assumed
#     to be 0.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChpmvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* AP,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex *const* y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZhpmvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpmvStridedBatched performs the matrix-vector operation
# 
#     y_i := alpha*A_i*x_i + beta*y_i
# 
# where alpha and beta are scalars, x_i and y_i are n element vectors and A_i is an
# n by n Hermitian matrix, supplied in packed form (see description below),
# for each batch in i = [1, batchCount].
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: the upper triangular part of each Hermitian matrix A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: the lower triangular part of each Hermitian matrix A_i is supplied in AP.
# @param[in]
# n         [int]
#           the order of each matrix A_i.
# @param[in]
# alpha     device pointer or host pointer to scalar alpha.
# @param[in]
# AP        device pointer pointing to the beginning of the first matrix (AP_1). Stores the packed
#           version of the specified triangular portion of each Hermitian matrix AP_i of size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,1), (4,0), (3,2), (5,-1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that each AP_i contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (3, 2)
#                     (2,-1) (4, 0) (5,-1)    -----> [(1,0), (2,-1), (3,-2), (4,0), (5,1), (6,0)]
#                     (3,-2) (5, 1) (6, 0)
#     Note that the imaginary part of the diagonal elements are not accessed and are assumed
#     to be 0.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (AP_i) and the next one (AP_i+1).
# @param[in]
# x         device array pointing to the beginning of the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# beta      device pointer or host pointer to scalar beta.
# @param[inout]
# y         device array pointing to the beginning of the first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChpmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZhpmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr performs the matrix-vector operations
# 
#     A := A + alpha*x*x**H
# 
# where alpha is a real scalar, x is a vector, and A is an
# n by n Hermitian matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
#  ******************************************************************
cdef hipblasStatus_t hipblasChpr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil



cdef hipblasStatus_t hipblasZhpr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hprBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasChprBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int batchCount) nogil



cdef hipblasStatus_t hipblasZhprBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hprStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**H
# 
# where alpha is a real scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i. Points to the first matrix (A_1).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# strideA   [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasChprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZhprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr2 performs the matrix-vector operations
# 
#     A := A + alpha*x*y**H + conj(alpha)*y*x**H
# 
# where alpha is a complex scalar, x and y are vectors, and A is an
# n by n Hermitian matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the Hermitian matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the Hermitian matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
#  ******************************************************************
cdef hipblasStatus_t hipblasChpr2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP) nogil



cdef hipblasStatus_t hipblasZhpr2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr2Batched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasChpr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int batchCount) nogil



cdef hipblasStatus_t hipblasZhpr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# hpr2StridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*y_i**H + conj(alpha)*y_i*x_i**H
# 
# where alpha is a complex scalar, x_i and y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# y         device pointer pointing to the first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each Hermitian matrix A_i. Points to the first matrix (A_1).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,1), (3,0), (4,9), (5,3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each Hermitian matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 3)
#                     (1, 0) (2, 1) (4,9)
#                     (2,-1) (3, 0) (5,3)  -----> [(1,0), (2,-1), (4,-9), (3,0), (5,-3), (6,0)]
#                     (4,-9) (5,-3) (6,0)
#         Note that the imaginary part of the diagonal elements are not accessed and are assumed
#         to be 0.
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasChpr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZhpr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# sbmv performs the matrix-vector operation:
# 
#     y := alpha*A*x + beta*y,
# 
# where alpha and beta are scalars, x and y are n element vectors and
# A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
# @param[in]
# k         [int]
#           specifies the number of sub- and super-diagonals
# @param[in]
# alpha
#           specifies the scalar alpha
# @param[in]
# AP         pointer storing matrix A on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of matrix A
# @param[in]
# x         pointer storing vector x on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x
# @param[in]
# beta      specifies the scalar beta
# @param[out]
# y         pointer storing vector y on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsbmv(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDsbmv(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# sbmvBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric banded matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# k         [int]
#           specifies the number of sub- and super-diagonals
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP         device array of device pointers storing each matrix A_i
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# x         device array of device pointers storing each vector x_i
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         device array of device pointers storing each vector y_i
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDsbmvBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double ** y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# sbmvStridedBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric banded matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n symmetric banded matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# k         [int]
#           specifies the number of sub- and super-diagonals
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP        Device pointer to the first matrix A_1 on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         Device pointer to the first vector x_1 on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size.
#             This typically means stridex >= n * incx. stridex should be non zero.
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         Device pointer to the first vector y_1 on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size.
#             This typically means stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDsbmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spmv performs the matrix-vector operation:
# 
#     y := alpha*A*x + beta*y,
# 
# where alpha and beta are scalars, x and y are n element vectors and
# A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
# @param[in]
# alpha
#           specifies the scalar alpha
# @param[in]
# AP         pointer storing matrix A on the GPU
# @param[in]
# x         pointer storing vector x on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x
# @param[in]
# beta      specifies the scalar beta
# @param[out]
# y         pointer storing vector y on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSspmv(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDspmv(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,const double * x,int incx,const double * beta,double * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spmvBatched performs the matrix-vector operation:
# 
#     y_i := alpha*AP_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP         device array of device pointers storing each matrix A_i
# @param[in]
# x         device array of device pointers storing each vector x_i
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         device array of device pointers storing each vector y_i
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSspmvBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* AP,const float *const* x,int incx,const float * beta,float ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDspmvBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* AP,const double *const* x,int incx,const double * beta,double ** y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spmvStridedBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A should contain an upper or lower triangular n by n packed symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP        Device pointer to the first matrix A_1 on the GPU
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         Device pointer to the first vector x_1 on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size.
#             This typically means stridex >= n * incx. stridex should be non zero.
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         Device pointer to the first vector y_1 on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size.
#             This typically means stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSspmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDspmvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spr performs the matrix-vector operations
# 
#     A := A + alpha*x*x**T
# 
# where alpha is a scalar, x is a vector, and A is an
# n by n symmetric matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
#  ******************************************************************
cdef hipblasStatus_t hipblasSspr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP) nogil



cdef hipblasStatus_t hipblasDspr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP) nogil



cdef hipblasStatus_t hipblasCspr(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP) nogil



cdef hipblasStatus_t hipblasZspr(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# sprBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**T
# 
# where alpha is a scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSsprBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,float *const* AP,int batchCount) nogil



cdef hipblasStatus_t hipblasDsprBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,double *const* AP,int batchCount) nogil



cdef hipblasStatus_t hipblasCsprBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int batchCount) nogil



cdef hipblasStatus_t hipblasZsprBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# sprStridedBatched performs the matrix-vector operations
# 
#     A_i := A_i + alpha*x_i*x_i**T
# 
# where alpha is a scalar, x_i is a vector, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i. Points to the first A_1.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(2) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# strideA    [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,float * AP,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasDsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,double * AP,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasCsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZsprStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spr2 performs the matrix-vector operation
# 
#     A := A + alpha*x*y**T + alpha*y*x**T
# 
# where alpha is a scalar, x and y are vectors, and A is an
# n by n symmetric matrix, supplied in packed form.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : s,d
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of A is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of A is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           the symmetric matrix A. Of at least size ((n * (n + 1)) / 2).
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of the symmetric matrix A is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(n) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
#  ******************************************************************
cdef hipblasStatus_t hipblasSspr2(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP) nogil



cdef hipblasStatus_t hipblasDspr2(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spr2Batched performs the matrix-vector operation
# 
#     A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T
# 
# where alpha is a scalar, x_i and y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP        device array of device pointers storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i of at least size ((n * (n + 1)) / 2). Array is of at least size batchCount.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(n) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSspr2Batched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,const float *const* y,int incy,float *const* AP,int batchCount) nogil



cdef hipblasStatus_t hipblasDspr2Batched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,const double *const* y,int incy,double *const* AP,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# spr2StridedBatched performs the matrix-vector operation
# 
#     A_i := A_i + alpha*x_i*y_i**T + alpha*y_i*x_i**T
# 
# where alpha is a scalar, x_i amd y_i are vectors, and A_i is an
# n by n symmetric matrix, supplied in packed form, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           HIPBLAS_FILL_MODE_UPPER: The upper triangular part of each A_i is supplied in AP.
#           HIPBLAS_FILL_MODE_LOWER: The lower triangular part of each A_i is supplied in AP.
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A_i, must be at least 0.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer pointing to the first vector (x_1).
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
# @param[in]
# y         device pointer pointing to the first vector (y_1).
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey  [hipblasStride]
#           stride from the start of one vector (y_i) and the next one (y_i+1).
# @param[inout]
# AP        device pointer storing the packed version of the specified triangular portion of
#           each symmetric matrix A_i. Points to the first A_1.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The upper triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(0,1)
#             AP(2) = A(1,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_UPPER; n = 4)
#                     1 2 4 7
#                     2 3 5 8   -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     4 5 6 9
#                     7 8 9 0
#         if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The lower triangular portion of each symmetric matrix A_i is supplied.
#             The matrix is compacted so that AP contains the triangular portion column-by-column
#             so that:
#             AP(0) = A(0,0)
#             AP(1) = A(1,0)
#             AP(n) = A(2,1), etc.
#                 Ex: (HIPBLAS_FILL_MODE_LOWER; n = 4)
#                     1 2 3 4
#                     2 5 6 7    -----> [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
#                     3 6 8 9
#                     4 7 9 0
# @param[in]
# strideA   [hipblasStride]
#             stride from the start of one (A_i) and the next (A_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSspr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,const float * y,int incy,long stridey,float * AP,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasDspr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,const double * y,int incy,long stridey,double * AP,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# symv performs the matrix-vector operation:
# 
#     y := alpha*A*x + beta*y,
# 
# where alpha and beta are scalars, x and y are n element vectors and
# A should contain an upper or lower triangular n by n symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
# @param[in]
# alpha
#           specifies the scalar alpha
# @param[in]
# AP         pointer storing matrix A on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of A
# @param[in]
# x         pointer storing vector x on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x
# @param[in]
# beta      specifies the scalar beta
# @param[out]
# y         pointer storing vector y on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsymv(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,int lda,const float * x,int incx,const float * beta,float * y,int incy) nogil



cdef hipblasStatus_t hipblasDsymv(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,const double * x,int incx,const double * beta,double * y,int incy) nogil



cdef hipblasStatus_t hipblasCsymv(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * beta,hipblasComplex * y,int incy) nogil



cdef hipblasStatus_t hipblasZsymv(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# symvBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A a should contain an upper or lower triangular symmetric matrix
# and the opposing triangular part of A is not referenced
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP        device array of device pointers storing each matrix A_i
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# x         device array of device pointers storing each vector x_i
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         device array of device pointers storing each vector y_i
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsymvBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* AP,int lda,const float *const* x,int incx,const float * beta,float ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasDsymvBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* AP,int lda,const double *const* x,int incx,const double * beta,double ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasCsymvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex * beta,hipblasComplex ** y,int incy,int batchCount) nogil



cdef hipblasStatus_t hipblasZsymvBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex * beta,hipblasDoubleComplex ** y,int incy,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# symvStridedBatched performs the matrix-vector operation:
# 
#     y_i := alpha*A_i*x_i + beta*y_i,
# 
# where (A_i, x_i, y_i) is the i-th instance of the batch.
# alpha and beta are scalars, x_i and y_i are vectors and A_i is an
# n by n symmetric matrix, for i = 1, ..., batchCount.
# A a should contain an upper or lower triangular symmetric matrix
# and the opposing triangular part of A is not referenced
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           number of rows and columns of each matrix A_i
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha
# @param[in]
# AP         Device pointer to the first matrix A_1 on the GPU
# @param[in]
# lda       [int]
#           specifies the leading dimension of each matrix A_i
# @param[in]
# strideA     [hipblasStride]
#             stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         Device pointer to the first vector x_1 on the GPU
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each vector x_i
# @param[in]
# stridex     [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1).
#             There are no restrictions placed on stridex, however the user should
#             take care to ensure that stridex is of appropriate size.
#             This typically means stridex >= n * incx. stridex should be non zero.
# @param[in]
# beta      device pointer or host pointer to scalar beta
# @param[out]
# y         Device pointer to the first vector y_1 on the GPU
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each vector y_i
# @param[in]
# stridey     [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1).
#             There are no restrictions placed on stridey, however the user should
#             take care to ensure that stridey is of appropriate size.
#             This typically means stridey >= n * incy. stridey should be non zero.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,const float * beta,float * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasDsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,const double * beta,double * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasCsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * beta,hipblasComplex * y,int incy,long stridey,int batchCount) nogil



cdef hipblasStatus_t hipblasZsymvStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * beta,hipblasDoubleComplex * y,int incy,long stridey,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# syr performs the matrix-vector operations
# 
#     A := A + alpha*x*x**T
# 
# where alpha is a scalar, x is a vector, and A is an
# n by n symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# 
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,float * AP,int lda) nogil



cdef hipblasStatus_t hipblasDsyr(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,double * AP,int lda) nogil



cdef hipblasStatus_t hipblasCsyr(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZsyr(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * AP,int lda) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# syrBatched performs a batch of matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*x[i]**T
# 
# where alpha is a scalar, x is an array of vectors, and A is an array of
# n by n symmetric matrices, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,float *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyrBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,double *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyrBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyrBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# syrStridedBatched performs the matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*x[i]**T
# 
# where alpha is a scalar, vectors, and A is an array of
# n by n symmetric matrices, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between vectors (x_i) and (x_i+1).
# @param[inout]
# AP         device pointer to the first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA   [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# batchCount [int]
#           number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,float * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,double * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyrStridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# syr2 performs the matrix-vector operations
# 
#     A := A + alpha*x*y**T + alpha*y*x**T
# 
# where alpha is a scalar, x and y are vectors, and A is an
# n by n symmetric matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# 
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr2(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,const float * y,int incy,float * AP,int lda) nogil



cdef hipblasStatus_t hipblasDsyr2(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,const double * y,int incy,double * AP,int lda) nogil



cdef hipblasStatus_t hipblasCsyr2(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,hipblasComplex * y,int incy,hipblasComplex * AP,int lda) nogil



cdef hipblasStatus_t hipblasZsyr2(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * y,int incy,hipblasDoubleComplex * AP,int lda) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# syr2Batched performs a batch of matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T
# 
# where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
# n by n symmetric matrix, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[inout]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float *const* x,int incx,const float *const* y,int incy,float *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double *const* x,int incx,const double *const* y,int incy,double *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex *const* x,int incx,hipblasComplex *const* y,int incy,hipblasComplex *const* AP,int lda,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyr2Batched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* y,int incy,hipblasDoubleComplex *const* AP,int lda,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# syr2StridedBatched the matrix-vector operations
# 
#     A[i] := A[i] + alpha*x[i]*y[i]**T + alpha*y[i]*x[i]**T
# 
# where alpha is a scalar, x[i] and y[i] are vectors, and A[i] is a
# n by n symmetric matrices, for i = 1 , ... , batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#           if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#           if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
# @param[in]
# n         [int]
#           the number of rows and columns of each matrix A.
# @param[in]
# alpha
#           device pointer or host pointer to scalar alpha.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           specifies the pointer increment between vectors (x_i) and (x_i+1).
# @param[in]
# y         device pointer to the first vector y_1.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey   [hipblasStride]
#           specifies the pointer increment between vectors (y_i) and (y_i+1).
# @param[inout]
# AP         device pointer to the first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA   [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# batchCount [int]
#           number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const float * alpha,const float * x,int incx,long stridex,const float * y,int incy,long stridey,float * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,const double * alpha,const double * x,int incx,long stridex,const double * y,int incy,long stridey,double * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasComplex * alpha,hipblasComplex * x,int incx,long stridex,hipblasComplex * y,int incy,long stridey,hipblasComplex * AP,int lda,long strideA,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyr2StridedBatched(void * handle,hipblasFillMode_t uplo,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * y,int incy,long stridey,hipblasDoubleComplex * AP,int lda,long strideA,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tbmv performs one of the matrix-vector operations
# 
#     x := A*x      or
#     x := A**T*x   or
#     x := A**H*x,
# 
# x is a vectors and A is a banded m by m matrix (see description below).
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: A is an upper banded triangular matrix.
#           HIPBLAS_FILL_MODE_LOWER: A is a  lower banded triangular matrix.
# @param[in]
# transA     [hipblasOperation_t]
#           indicates whether matrix A is tranposed (conjugated) or not.
# @param[in]
# diag      [hipblasDiagType_t]
#           HIPBLAS_DIAG_UNIT: The main diagonal of A is assumed to consist of only
#                                  1's and is not referenced.
#           HIPBLAS_DIAG_NON_UNIT: No assumptions are made of A's main diagonal.
# @param[in]
# m         [int]
#           the number of rows and columns of the matrix represented by A.
# @param[in]
# k         [int]
#           if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#           of the matrix A.
#           if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#           of the matrix A.
#           k must satisfy k > 0 && k < lda.
# @param[in]
# AP         device pointer storing banded triangular matrix A.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The matrix represented is an upper banded triangular matrix
#             with the main diagonal and k super-diagonals, everything
#             else can be assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the k'th
#             row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#             with the k'th diagonal on the RHS of the 0'th row.
#                Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                   1 6 9 0 0              0 0 9 8 7
#                   0 2 7 8 0              0 6 7 8 9
#                   0 0 3 8 7     ---->    1 2 3 4 5
#                   0 0 0 4 9              0 0 0 0 0
#                   0 0 0 0 5              0 0 0 0 0
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The matrix represnted is a lower banded triangular matrix
#             with the main diagonal and k sub-diagonals, everything else can be
#             assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the 0'th row,
#             working up to the k'th diagonal residing on the LHS of the k'th row.
#                Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                   1 0 0 0 0              1 2 3 4 5
#                   6 2 0 0 0              6 7 8 9 0
#                   9 7 3 0 0     ---->    9 8 7 0 0
#                   0 8 8 4 0              0 0 0 0 0
#                   0 0 7 9 5              0 0 0 0 0
# @param[in]
# lda       [int]
#           specifies the leading dimension of A. lda must satisfy lda > k.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtbmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tbmvBatched performs one of the matrix-vector operations
# 
#     x_i := A_i*x_i      or
#     x_i := A_i**T*x_i   or
#     x_i := A_i**H*x_i,
# 
# where (A_i, x_i) is the i-th instance of the batch.
# x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: each A_i is an upper banded triangular matrix.
#           HIPBLAS_FILL_MODE_LOWER: each A_i is a  lower banded triangular matrix.
# @param[in]
# transA     [hipblasOperation_t]
#           indicates whether each matrix A_i is tranposed (conjugated) or not.
# @param[in]
# diag      [hipblasDiagType_t]
#           HIPBLAS_DIAG_UNIT: The main diagonal of each A_i is assumed to consist of only
#                                  1's and is not referenced.
#           HIPBLAS_DIAG_NON_UNIT: No assumptions are made of each A_i's main diagonal.
# @param[in]
# m         [int]
#           the number of rows and columns of the matrix represented by each A_i.
# @param[in]
# k         [int]
#           if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#           of each matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#           of each matrix A_i.
#           k must satisfy k > 0 && k < lda.
# @param[in]
# AP         device array of device pointers storing each banded triangular matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The matrix represented is an upper banded triangular matrix
#             with the main diagonal and k super-diagonals, everything
#             else can be assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the k'th
#             row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#             with the k'th diagonal on the RHS of the 0'th row.
#                Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                   1 6 9 0 0              0 0 9 8 7
#                   0 2 7 8 0              0 6 7 8 9
#                   0 0 3 8 7     ---->    1 2 3 4 5
#                   0 0 0 4 9              0 0 0 0 0
#                   0 0 0 0 5              0 0 0 0 0
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The matrix represnted is a lower banded triangular matrix
#             with the main diagonal and k sub-diagonals, everything else can be
#             assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the 0'th row,
#             working up to the k'th diagonal residing on the LHS of the k'th row.
#                Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                   1 0 0 0 0              1 2 3 4 5
#                   6 2 0 0 0              6 7 8 9 0
#                   9 7 3 0 0     ---->    9 8 7 0 0
#                   0 8 8 4 0              0 0 0 0 0
#                   0 0 7 9 5              0 0 0 0 0
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. lda must satisfy lda > k.
# @param[inout]
# x         device array of device pointer storing each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDtbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCtbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZtbmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tbmvStridedBatched performs one of the matrix-vector operations
# 
#     x_i := A_i*x_i      or
#     x_i := A_i**T*x_i   or
#     x_i := A_i**H*x_i,
# 
# where (A_i, x_i) is the i-th instance of the batch.
# x_i is a vector and A_i is an m by m matrix, for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           HIPBLAS_FILL_MODE_UPPER: each A_i is an upper banded triangular matrix.
#           HIPBLAS_FILL_MODE_LOWER: each A_i is a  lower banded triangular matrix.
# @param[in]
# transA     [hipblasOperation_t]
#           indicates whether each matrix A_i is tranposed (conjugated) or not.
# @param[in]
# diag      [hipblasDiagType_t]
#           HIPBLAS_DIAG_UNIT: The main diagonal of each A_i is assumed to consist of only
#                                  1's and is not referenced.
#           HIPBLAS_DIAG_NON_UNIT: No assumptions are made of each A_i's main diagonal.
# @param[in]
# m         [int]
#           the number of rows and columns of the matrix represented by each A_i.
# @param[in]
# k         [int]
#           if uplo == HIPBLAS_FILL_MODE_UPPER, k specifies the number of super-diagonals
#           of each matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_LOWER, k specifies the number of sub-diagonals
#           of each matrix A_i.
#           k must satisfy k > 0 && k < lda.
# @param[in]
# AP         device array to the first matrix A_i of the batch. Stores each banded triangular matrix A_i.
#           if uplo == HIPBLAS_FILL_MODE_UPPER:
#             The matrix represented is an upper banded triangular matrix
#             with the main diagonal and k super-diagonals, everything
#             else can be assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the k'th
#             row, the first super diagonal resides on the RHS of the k-1'th row, etc,
#             with the k'th diagonal on the RHS of the 0'th row.
#                Ex: (HIPBLAS_FILL_MODE_UPPER; m = 5; k = 2)
#                   1 6 9 0 0              0 0 9 8 7
#                   0 2 7 8 0              0 6 7 8 9
#                   0 0 3 8 7     ---->    1 2 3 4 5
#                   0 0 0 4 9              0 0 0 0 0
#                   0 0 0 0 5              0 0 0 0 0
#           if uplo == HIPBLAS_FILL_MODE_LOWER:
#             The matrix represnted is a lower banded triangular matrix
#             with the main diagonal and k sub-diagonals, everything else can be
#             assumed to be 0.
#             The matrix is compacted so that the main diagonal resides on the 0'th row,
#             working up to the k'th diagonal residing on the LHS of the k'th row.
#                Ex: (HIPBLAS_FILL_MODE_LOWER; m = 5; k = 2)
#                   1 0 0 0 0              1 2 3 4 5
#                   6 2 0 0 0              6 7 8 9 0
#                   9 7 3 0 0     ---->    9 8 7 0 0
#                   0 8 8 4 0              0 0 0 0 0
#                   0 0 7 9 5              0 0 0 0 0
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i. lda must satisfy lda > k.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_(i + 1).
# @param[inout]
# x         device array to the first vector x_i of the batch.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one x_i matrix to the next x_(i + 1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDtbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCtbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZtbmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int k,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tbsv solves
# 
#      A*x = b or A**T*x = b or A**H*x = b,
# 
# where x and b are vectors and A is a banded triangular matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
#            HIPBLAS_OP_N: Solves A*x = b
#            HIPBLAS_OP_T: Solves A**T*x = b
#            HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
#                                    of A are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
# 
# @param[in]
# n         [int]
#           n specifies the number of rows of b. n >= 0.
# @param[in]
# k         [int]
#           if(uplo == HIPBLAS_FILL_MODE_UPPER)
#             k specifies the number of super-diagonals of A.
#           if(uplo == HIPBLAS_FILL_MODE_LOWER)
#             k specifies the number of sub-diagonals of A.
#           k >= 0.
# 
# @param[in]
# AP         device pointer storing the matrix A in banded format.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#           lda >= (k + 1).
# 
# @param[inout]
# x         device pointer storing input vector b. Overwritten by the output vector x.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtbsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tbsvBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a banded triangular matrix,
# for i = [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
#            HIPBLAS_OP_N: Solves A_i*x_i = b_i
#            HIPBLAS_OP_T: Solves A_i**T*x_i = b_i
#            HIPBLAS_OP_C: Solves A_i**H*x_i = b_i
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# n         [int]
#           n specifies the number of rows of each b_i. n >= 0.
# @param[in]
# k         [int]
#           if(uplo == HIPBLAS_FILL_MODE_UPPER)
#             k specifies the number of super-diagonals of each A_i.
#           if(uplo == HIPBLAS_FILL_MODE_LOWER)
#             k specifies the number of sub-diagonals of each A_i.
#           k >= 0.
# 
# @param[in]
# AP         device vector of device pointers storing each matrix A_i in banded format.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda >= (k + 1).
# 
# @param[inout]
# x         device vector of device pointers storing each input vector b_i. Overwritten by each output
#           vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDtbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCtbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZtbsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tbsvStridedBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a banded triangular matrix,
# for i = [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
#            HIPBLAS_OP_N: Solves A_i*x_i = b_i
#            HIPBLAS_OP_T: Solves A_i**T*x_i = b_i
#            HIPBLAS_OP_C: Solves A_i**H*x_i = b_i
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# n         [int]
#           n specifies the number of rows of each b_i. n >= 0.
# @param[in]
# k         [int]
#           if(uplo == HIPBLAS_FILL_MODE_UPPER)
#             k specifies the number of super-diagonals of each A_i.
#           if(uplo == HIPBLAS_FILL_MODE_LOWER)
#             k specifies the number of sub-diagonals of each A_i.
#           k >= 0.
# 
# @param[in]
# AP         device pointer pointing to the first banded matrix A_1.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda >= (k + 1).
# @param[in]
# strideA  [hipblasStride]
#           specifies the distance between the start of one matrix (A_i) and the next (A_i+1).
# 
# @param[inout]
# x         device pointer pointing to the first input vector b_1. Overwritten by output vectors x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           specifies the distance between the start of one vector (x_i) and the next (x_i+1).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDtbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCtbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZtbsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int n,int k,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tpmv performs one of the matrix-vector operations
# 
#      x = A*x or x = A**T*x,
# 
# where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix, supplied in the pack form.
# 
# The vector x is overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of A. m >= 0.
# 
# @param[in]
# AP       device pointer storing matrix A,
#         of dimension at leat ( m * ( m + 1 ) / 2 ).
#       Before entry with uplo = HIPBLAS_FILL_MODE_UPPER, the array A
#       must contain the upper triangular matrix packed sequentially,
#       column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
#       a_{0,1} and a_{1, 1} respectively, and so on.
#       Before entry with uplo = HIPBLAS_FILL_MODE_LOWER, the array A
#       must contain the lower triangular matrix packed sequentially,
#       column by column, so that A[0] contains a_{0,0}, A[1] and A[2] contain
#       a_{1,0} and a_{2,0} respectively, and so on.
#       Note that when DIAG = HIPBLAS_DIAG_UNIT, the diagonal elements of A are
#       not referenced, but are assumed to be unity.
# 
# @param[in]
# x       device pointer storing vector x.
# 
# @param[in]
# incx    [int]
#         specifies the increment for the elements of x. incx must not be zero.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtpmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tpmvBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP         device pointer storing pointer of matrices A_i,
#           of dimension ( lda, m )
# 
# @param[in]
# x         device pointer storing vectors x_i.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of vectors x_i.
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDtpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCtpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZtpmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tpmvStridedBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP         device pointer of the matrix A_0,
#           of dimension ( lda, m )
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_{i + 1}
# 
# @param[in]
# x         device pointer storing the vector x_0.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of one vector x.
# 
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one x_i vector to the next x_{i + 1}
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,long strideA,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDtpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,long strideA,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCtpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZtpmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tpsv solves
# 
#      A*x = b or A**T*x = b, or A**H*x = b,
# 
# where x and b are vectors and A is a triangular matrix stored in the packed format.
# 
# The input vector b is overwritten by the output vector x.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: Solves A*x = b
#         HIPBLAS_OP_T: Solves A**T*x = b
#         HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular (i.e. the diagonal elements
#                                    of A are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of b. m >= 0.
# 
# @param[in]
# AP        device pointer storing the packed version of matrix A,
#           of dimension >= (n * (n + 1) / 2)
# 
# @param[inout]
# x         device pointer storing vector b on input, overwritten by x on output.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtpsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tpsvBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
# for i in [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: Solves A*x = b
#         HIPBLAS_OP_T: Solves A**T*x = b
#         HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of each b_i. m >= 0.
# 
# @param[in]
# AP        device array of device pointers storing the packed versions of each matrix A_i,
#           of dimension >= (n * (n + 1) / 2)
# 
# @param[inout]
# x         device array of device pointers storing each input vector b_i, overwritten by x_i on output.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDtpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCtpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZtpsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# tpsvStridedBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i, or A_i**H*x_i = b_i,
# 
# where x_i and b_i are vectors and A_i is a triangular matrix stored in the packed format,
# for i in [1, batchCount].
# 
# The input vectors b_i are overwritten by the output vectors x_i.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: Solves A*x = b
#         HIPBLAS_OP_T: Solves A**T*x = b
#         HIPBLAS_OP_C: Solves A**H*x = b
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular (i.e. the diagonal elements
#                                    of each A_i are not used in computations).
#         HIPBLAS_DIAG_NON_UNIT: each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of each b_i. m >= 0.
# 
# @param[in]
# AP        device pointer pointing to the first packed matrix A_1,
#           of dimension >= (n * (n + 1) / 2)
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the beginning of one packed matrix (AP_i) and the next (AP_i+1).
# 
# @param[inout]
# x         device pointer pointing to the first input vector b_1. Overwritten by each x_i on output.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex  [hipblasStride]
#           stride from the beginning of one vector (x_i) and the next (x_i+1).
# @param[in]
# batchCount [int]
#             specifies the number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,long strideA,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDtpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,long strideA,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCtpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZtpsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# trmv performs one of the matrix-vector operations
# 
#      x = A*x or x = A**T*x,
# 
# where x is an n element vector and A is an n by n unit, or non-unit, upper or lower triangular matrix.
# 
# The vector x is overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of A. m >= 0.
# 
# @param[in]
# AP        device pointer storing matrix A,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#           lda = max( 1, m ).
# 
# @param[in]
# x         device pointer storing vector x.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtrmv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# trmvBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP        device pointer storing pointer of matrices A_i,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A_i.
#           lda >= max( 1, m ).
# 
# @param[in]
# x         device pointer storing vectors x_i.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of vectors x_i.
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrmvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# trmvStridedBatched performs one of the matrix-vector operations
# 
#      x_i = A_i*x_i or x_i = A**T*x_i, 0 \le i < batchCount
# 
# where x_i is an n element vector and A_i is an n by n (unit, or non-unit, upper or lower triangular matrix)
# with strides specifying how to retrieve $x_i$ (resp. $A_i$) from $x_{i-1}$ (resp. $A_i$).
# 
# The vectors x_i are overwritten.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of matrices A_i. m >= 0.
# 
# @param[in]
# AP        device pointer of the matrix A_0,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A_i.
#           lda >= max( 1, m ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_{i + 1}
# 
# @param[in]
# x         device pointer storing the vector x_0.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of one vector x.
# 
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one x_i vector to the next x_{i + 1}
# 
# @param[in]
# batchCount [int]
#           The number of batched matrices/vectors.
# 
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrmvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# trsv solves
# 
#      A*x = b or A**T*x = b,
# 
# where x and b are vectors and A is a triangular matrix.
# 
# The vector x is overwritten on b.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of b. m >= 0.
# 
# @param[in]
# AP        device pointer storing matrix A,
#           of dimension ( lda, m )
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
#           lda = max( 1, m ).
# 
# @param[in]
# x         device pointer storing vector x.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,float * x,int incx) nogil



cdef hipblasStatus_t hipblasDtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,double * x,int incx) nogil



cdef hipblasStatus_t hipblasCtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,hipblasComplex * x,int incx) nogil



cdef hipblasStatus_t hipblasZtrsv(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# trsvBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i,
# 
# where (A_i, x_i, b_i) is the i-th instance of the batch.
# x_i and b_i are vectors and A_i is an
# m by m triangular matrix.
# 
# The vector x is overwritten on b.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of b. m >= 0.
# 
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda = max(1, m)
# 
# @param[in]
# x         device array of device pointers storing each vector x_i.
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float *const* AP,int lda,float *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double *const* AP,int lda,double *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrsvBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,int batchCount) nogil


# @{
# \brief BLAS Level 2 API
# 
# \details
# trsvStridedBatched solves
# 
#      A_i*x_i = b_i or A_i**T*x_i = b_i,
# 
# where (A_i, x_i, b_i) is the i-th instance of the batch.
# x_i and b_i are vectors and A_i is an m by m triangular matrix, for i = 1, ..., batchCount.
# 
# The vector x is overwritten on b.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA     [hipblasOperation_t]
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m         [int]
#           m specifies the number of rows of each b_i. m >= 0.
# 
# @param[in]
# AP         device pointer to the first matrix (A_1) in the batch, of dimension ( lda, m )
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_(i + 1)
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
#           lda = max( 1, m ).
# 
# @param[in, out]
# x         device pointer to the first vector (x_1) in the batch.
# 
# @param[in]
# stridex [hipblasStride]
#          stride from the start of one x_i vector to the next x_(i + 1)
# 
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const float * AP,int lda,long strideA,float * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,const double * AP,int lda,long strideA,double * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrsvStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# gemm performs one of the matrix-matrix operations
# 
#     C = alpha*op( A )*op( B ) + beta*C,
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B and C are matrices, with
# op( A ) an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : h,s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
# 
#           .
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           number or rows of matrices op( A ) and C
# @param[in]
# n         [int]
#           number of columns of matrices op( B ) and C
# @param[in]
# k         [int]
#           number of columns of matrix op( A ) and number of rows of matrix op( B )
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# BP         device pointer storing matrix B.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in, out]
# CP         device pointer storing matrix C on the GPU.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short * AP,int lda,const unsigned short * BP,int ldb,const unsigned short * beta,unsigned short * CP,int ldc) nogil



cdef hipblasStatus_t hipblasSgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZgemm(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
#  \details
# gemmBatched performs one of the batched matrix-matrix operations
#      C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount.
#  where op( X ) is one of
#      op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
#  alpha and beta are scalars, and A, B and C are strided batched matrices, with
# op( A ) an m by k by batchCount strided_batched matrix,
# op( B ) an k by n by batchCount strided_batched matrix and
# C an m by n by batchCount strided_batched matrix.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : h,s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimention m.
# @param[in]
# n         [int]
#           matrix dimention n.
# @param[in]
# k         [int]
#           matrix dimention k.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# BP         device array of device pointers storing each matrix B_i.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in, out]
# CP         device array of device pointers storing each matrix C_i.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# batchCount
#           [int]
#           number of gemm operations in the batch
#   ******************************************************************
cdef hipblasStatus_t hipblasHgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short *const* AP,int lda,const unsigned short *const* BP,int ldb,const unsigned short * beta,unsigned short *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasSgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZgemmBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# gemmStridedBatched performs one of the strided batched matrix-matrix operations
# 
#     C_i = alpha*op( A_i )*op( B_i ) + beta*C_i, for i = 1, ..., batchCount.
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B and C are strided batched matrices, with
# op( A ) an m by k by batchCount strided_batched matrix,
# op( B ) an k by n by batchCount strided_batched matrix and
# C an m by n by batchCount strided_batched matrix.
# 
# - Supported precisions in rocBLAS : h,s,d,c,z
# - Supported precisions in cuBLAS  : h,s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimention m.
# @param[in]
# n         [int]
#           matrix dimention n.
# @param[in]
# k         [int]
#           matrix dimention k.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device pointer pointing to the first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one A_i matrix to the next A_(i + 1).
# @param[in]
# BP         device pointer pointing to the first matrix B_1.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one B_i matrix to the next B_(i + 1).
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in, out]
# CP         device pointer pointing to the first matrix C_1.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# strideC  [hipblasStride]
#           stride from the start of one C_i matrix to the next C_(i + 1).
# @param[in]
# batchCount
#           [int]
#           number of gemm operatons in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasHgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const unsigned short * alpha,const unsigned short * AP,int lda,long long strideA,const unsigned short * BP,int ldb,long long strideB,const unsigned short * beta,unsigned short * CP,int ldc,long long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasSgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const float * alpha,const float * AP,int lda,long long strideA,const float * BP,int ldb,long long strideB,const float * beta,float * CP,int ldc,long long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const double * alpha,const double * AP,int lda,long long strideA,const double * BP,int ldb,long long strideB,const double * beta,double * CP,int ldc,long long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long long strideA,hipblasComplex * BP,int ldb,long long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZgemmStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long long strideA,hipblasDoubleComplex * BP,int ldb,long long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herk performs one of the matrix-matrix operations for a Hermitian rank-k update
# 
# C := alpha*op( A )*op( A )^H + beta*C
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A ) = A,  and A is n by k if transA == HIPBLAS_OP_N
#     op( A ) = A^H and A is k by n if transA == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C:  op(A) = A^H
#         HIPBLAS_ON_N:  op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCherk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex * AP,int lda,const float * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZherk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^H and A_i is k by n if transA == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCherkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex *const* AP,int lda,const float * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZherkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex *const* AP,int lda,const double * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^H and A_i is k by n if transA == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCherkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,hipblasComplex * AP,int lda,long strideA,const float * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZherkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,hipblasDoubleComplex * AP,int lda,long strideA,const double * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkx performs one of the matrix-matrix operations for a Hermitian rank-k update
# 
# C := alpha*op( A )*op( B )^H + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
# C is a n x n Hermitian matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
#         HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP      pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCherkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZherkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkxBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCherkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,const float * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZherkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,const double * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# herkxStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be Hermitian.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op( A_i ) = A_i^H, op( B_i ) = B_i^H
#         HIPBLAS_OP_N: op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCherkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,const float * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZherkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,const double * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# her2k performs one of the matrix-matrix operations for a Hermitian rank-2k update
# 
# C := alpha*op( A )*op( B )^H + conj(alpha)*op( B )*op( A )^H + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrices, and
# C is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^H, op( B ) = B^H,  and A and B are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C:  op( A ) = A^H, op( B ) = B^H
#         HIPBLAS_OP_N:  op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCher2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,const float * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZher2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,const double * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# her2kBatched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op(A) = A^H
#         HIPBLAS_OP_N: op(A) = A
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCher2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,const float * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZher2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,const double * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# her2kStridedBatched performs a batch of the matrix-matrix operations for a Hermitian rank-2k update
# 
# C_i := alpha*op( A_i )*op( B_i )^H + conj(alpha)*op( B_i )*op( A_i )^H + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrices, and
# C_i is a n x n Hermitian matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^H, op( B_i ) = B_i^H,  and A_i and B_i are k by n if trans == HIPBLAS_OP_C
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_C: op( A_i ) = A_i^H, op( B_i ) = B_i^H
#         HIPBLAS_OP_N: op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
#         The imaginary component of the diagonal elements are not used but are set to zero unless quick return.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasCher2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,const float * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZher2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,const double * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# symm performs one of the matrix-matrix operations:
# 
# C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
# C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B and C are m by n matrices, and
# A is a symmetric matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
#         HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B and C. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B and C. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A and B are not referenced.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         A is m by m if side == HIPBLAS_SIDE_LEFT
#         A is n by n if side == HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m )
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsymm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# symmBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a symmetric matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i and C_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i and C_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP      device array of device pointers storing each matrix A_i on the GPU.
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix B_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C_i need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C_i. ldc >= max( 1, m )
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZsymmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# symmStridedBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a symmetric matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i and C_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i and C_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP       device pointer to first matrix A_1
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       device pointer to first matrix B_1 of dimension (ldb, n) on the GPU.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i. ldb >= max( 1, m )
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP        device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const float * alpha,const float * AP,int lda,long strideA,const float * BP,int ldb,long strideB,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,const double * alpha,const double * AP,int lda,long strideA,const double * BP,int ldb,long strideB,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZsymmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrk performs one of the matrix-matrix operations for a symmetric rank-k update
# 
# C := alpha*op( A )*op( A )^T + beta*C
# 
# where  alpha and beta are scalars, op(A) is an n by k matrix, and
# C is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A ) = A, and A is n by k if transA == HIPBLAS_OP_N
#     op( A ) = A^T and A is k by n if transA == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T: op(A) = A^T
#         HIPBLAS_OP_N: op(A) = A
#         HIPBLAS_OP_C: op(A) = A^T
# 
#         HIPBLAS_OP_C is not supported for complex types, see cherk
#         and zherk.
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if transA = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsyrk(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^T and A_i is k by n if transA == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T: op(A) = A^T
#         HIPBLAS_OP_N: op(A) = A
#         HIPBLAS_OP_C: op(A) = A^T
# 
#         HIPBLAS_OP_C is not supported for complex types, see cherk
#         and zherk.
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float *const* AP,int lda,const float * beta,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double *const* AP,int lda,const double * beta,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyrkBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( A_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) is an n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, and A_i is n by k if transA == HIPBLAS_OP_N
#     op( A_i ) = A_i^T and A_i is k by n if transA == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T: op(A) = A^T
#         HIPBLAS_OP_N: op(A) = A
#         HIPBLAS_OP_C: op(A) = A^T
# 
#         HIPBLAS_OP_C is not supported for complex types, see cherk
#         and zherk.
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when transA is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if transA = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU. on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyrkStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syr2k performs one of the matrix-matrix operations for a symmetric rank-2k update
# 
# C := alpha*(op( A )*op( B )^T + op( B )*op( A )^T) + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
# C is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
#         HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A) and op(B). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsyr2k(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syr2kBatched performs a batch of the matrix-matrix operations for a symmetric rank-2k update
# 
# C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# @param[in]
# BP      device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP      device array of device pointers storing each matrix C_i on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyr2kBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syr2kStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-2k update
# 
# C_i := alpha*(op( A_i )*op( B_i )^T + op( B_i )*op( A_i )^T) + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * BP,int ldb,long strideB,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * BP,int ldb,long strideB,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyr2kStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkx performs one of the matrix-matrix operations for a symmetric rank-k update
# 
# C := alpha*op( A )*op( B )^T + beta*C
# 
# where  alpha and beta are scalars, op(A) and op(B) are n by k matrix, and
# C is a symmetric n x n matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A )*op( B )^T will be symmetric.
# 
#     op( A ) = A, op( B ) = B, and A and B are n by k if trans == HIPBLAS_OP_N
#     op( A ) = A^T, op( B ) = B^T,  and A and B are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A ) = A^T, op( B ) = B^T
#         HIPBLAS_OP_N:           op( A ) = A, op( B ) = B
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A) and op(B). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         Martrix dimension is ( lda, k ) when if trans = HIPBLAS_OP_N, otherwise (lda, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Martrix dimension is ( ldb, k ) when if trans = HIPBLAS_OP_N, otherwise (ldb, n)
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,const float * BP,int ldb,const float * beta,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,const double * BP,int ldb,const double * beta,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZsyrkx(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkxBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       device array of device pointers storing each matrix_i A of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix_i B of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[in]
# batchCount [int]
#         number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float *const* AP,int lda,const float *const* BP,int ldb,const float * beta,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double *const* AP,int lda,const double *const* BP,int ldb,const double * beta,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyrkxBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# syrkxStridedBatched performs a batch of the matrix-matrix operations for a symmetric rank-k update
# 
# C_i := alpha*op( A_i )*op( B_i )^T + beta*C_i
# 
# where  alpha and beta are scalars, op(A_i) and op(B_i) are n by k matrix, and
# C_i is a symmetric n x n matrix stored as either upper or lower.
# This routine should only be used when the caller can guarantee that the result of op( A_i )*op( B_i )^T will be symmetric.
# 
#     op( A_i ) = A_i, op( B_i ) = B_i, and A_i and B_i are n by k if trans == HIPBLAS_OP_N
#     op( A_i ) = A_i^T, op( B_i ) = B_i^T,  and A_i and B_i are k by n if trans == HIPBLAS_OP_T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  C_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  C_i is a  lower triangular matrix
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_T:      op( A_i ) = A_i^T, op( B_i ) = B_i^T
#         HIPBLAS_OP_N:           op( A_i ) = A_i, op( B_i ) = B_i
# 
# @param[in]
# n       [int]
#         n specifies the number of rows and columns of C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of op(A). k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and A need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_1 on the GPU of dimension (lda, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (lda, n)
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if trans = HIPBLAS_OP_N,  lda >= max( 1, n ),
#         otherwise lda >= max( 1, k ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       Device pointer to the first matrix B_1 on the GPU of dimension (ldb, k)
#         when trans is HIPBLAS_OP_N, otherwise of dimension (ldb, n)
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if trans = HIPBLAS_OP_N,  ldb >= max( 1, n ),
#         otherwise ldb >= max( 1, k ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       Device pointer to the first matrix C_1 on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, n ).
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const float * alpha,const float * AP,int lda,long strideA,const float * BP,int ldb,long strideB,const float * beta,float * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,const double * alpha,const double * AP,int lda,long strideA,const double * BP,int ldb,long strideB,const double * beta,double * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZsyrkxStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasOperation_t transA,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# geam performs one of the matrix-matrix operations
# 
#     C = alpha*op( A ) + beta*op( B ),
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B and C are matrices, with
# op( A ) an m by n matrix, op( B ) an m by n matrix, and C an m by n matrix.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in]
# BP         device pointer storing matrix B.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in, out]
# CP         device pointer storing matrix C.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float * AP,int lda,const float * beta,const float * BP,int ldb,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,const double * beta,const double * BP,int ldb,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * beta,hipblasComplex * BP,int ldb,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZgeam(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# geamBatched performs one of the batched matrix-matrix operations
# 
#     C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batchCount - 1
# 
# where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
# and op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# @param[in]
# AP         device array of device pointers storing each matrix A_i on the GPU.
#           Each A_i is of dimension ( lda, k ), where k is m
#           when  transA == HIPBLAS_OP_N and
#           is  n  when  transA == HIPBLAS_OP_T.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# @param[in]
# BP         device array of device pointers storing each matrix B_i on the GPU.
#           Each B_i is of dimension ( ldb, k ), where k is m
#           when  transB == HIPBLAS_OP_N and
#           is  n  when  transB == HIPBLAS_OP_T.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in, out]
# CP         device array of device pointers storing each matrix C_i on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float *const* AP,int lda,const float * beta,const float *const* BP,int ldb,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double *const* AP,int lda,const double * beta,const double *const* BP,int ldb,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex * beta,hipblasComplex *const* BP,int ldb,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZgeamBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# geamStridedBatched performs one of the batched matrix-matrix operations
# 
#     C_i = alpha*op( A_i ) + beta*op( B_i )  for i = 0, 1, ... batchCount - 1
# 
# where alpha and beta are scalars, and op(A_i), op(B_i) and C_i are m by n matrices
# and op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A )
# 
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B )
# 
# @param[in]
# m         [int]
#           matrix dimension m.
# 
# @param[in]
# n         [int]
#           matrix dimension n.
# 
# @param[in]
# alpha     device pointer or host pointer specifying the scalar alpha.
# 
# @param[in]
# AP         device pointer to the first matrix A_0 on the GPU.
#           Each A_i is of dimension ( lda, k ), where k is m
#           when  transA == HIPBLAS_OP_N and
#           is  n  when  transA == HIPBLAS_OP_T.
# 
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# beta      device pointer or host pointer specifying the scalar beta.
# 
# @param[in]
# BP         pointer to the first matrix B_0 on the GPU.
#           Each B_i is of dimension ( ldb, k ), where k is m
#           when  transB == HIPBLAS_OP_N and
#           is  n  when  transB == HIPBLAS_OP_T.
# 
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in, out]
# CP         pointer to the first matrix C_0 on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# 
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
# @param[in]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const float * alpha,const float * AP,int lda,long strideA,const float * beta,const float * BP,int ldb,long strideB,float * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,const double * alpha,const double * AP,int lda,long strideA,const double * beta,const double * BP,int ldb,long strideB,double * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * beta,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZgeamStridedBatched(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * beta,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# hemm performs one of the matrix-matrix operations:
# 
# C := alpha*A*B + beta*C if side == HIPBLAS_SIDE_LEFT,
# C := alpha*B*A + beta*C if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B and C are m by n matrices, and
# A is a Hermitian matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C := alpha*A*B + beta*C
#         HIPBLAS_SIDE_RIGHT:     C := alpha*B*A + beta*C
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix
# 
# @param[in]
# n       [int]
#         n specifies the number of rows of B and C. n >= 0.
# 
# @param[in]
# k       [int]
#         n specifies the number of columns of B and C. k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A and B are not referenced.
# 
# @param[in]
# AP       pointer storing matrix A on the GPU.
#         A is m by m if side == HIPBLAS_SIDE_LEFT
#         A is n by n if side == HIPBLAS_SIDE_RIGHT
#         Only the upper/lower triangular part is accessed.
#         The imaginary component of the diagonal elements is not used.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       pointer storing matrix B on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP       pointer storing matrix C on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m )
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChemm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb,hipblasComplex * beta,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZhemm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# hemmBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a Hermitian matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# n       [int]
#         n specifies the number of rows of B_i and C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of B_i and C_i. k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP       device array of device pointers storing each matrix A_i on the GPU.
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         Only the upper/lower triangular part is accessed.
#         The imaginary component of the diagonal elements is not used.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# BP       device array of device pointers storing each matrix B_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i. ldb >= max( 1, m )
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C_i need not be set before entry.
# 
# @param[in]
# CP       device array of device pointers storing each matrix C_i on the GPU.
#         Matrix dimension is m by n
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C_i. ldc >= max( 1, m )
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChemmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,hipblasComplex * beta,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZhemmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,hipblasDoubleComplex * beta,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# hemmStridedBatched performs a batch of the matrix-matrix operations:
# 
# C_i := alpha*A_i*B_i + beta*C_i if side == HIPBLAS_SIDE_LEFT,
# C_i := alpha*B_i*A_i + beta*C_i if side == HIPBLAS_SIDE_RIGHT,
# 
# where alpha and beta are scalars, B_i and C_i are m by n matrices, and
# A_i is a Hermitian matrix stored as either upper or lower.
# 
# - Supported precisions in rocBLAS : c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side  [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:      C_i := alpha*A_i*B_i + beta*C_i
#         HIPBLAS_SIDE_RIGHT:     C_i := alpha*B_i*A_i + beta*C_i
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A_i is an upper triangular matrix
#         HIPBLAS_FILL_MODE_LOWER:  A_i is a  lower triangular matrix
# 
# @param[in]
# n       [int]
#         n specifies the number of rows of B_i and C_i. n >= 0.
# 
# @param[in]
# k       [int]
#         k specifies the number of columns of B_i and C_i. k >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i and B_i are not referenced.
# 
# @param[in]
# AP       device pointer to first matrix A_1
#         A_i is m by m if side == HIPBLAS_SIDE_LEFT
#         A_i is n by n if side == HIPBLAS_SIDE_RIGHT
#         Only the upper/lower triangular part is accessed.
#         The imaginary component of the diagonal elements is not used.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         otherwise lda >= max( 1, n ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[in]
# BP       device pointer to first matrix B_1 of dimension (ldb, n) on the GPU
# 
# @param[in]
# ldb     [int]
#         ldb specifies the first dimension of B_i.
#         if side = HIPBLAS_OP_N,  ldb >= max( 1, m ),
#         otherwise ldb >= max( 1, n ).
# 
# @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# 
# @param[in]
# beta
#         beta specifies the scalar beta. When beta is
#         zero then C need not be set before entry.
# 
# @param[in]
# CP        device pointer to first matrix C_1 of dimension (ldc, n) on the GPU.
# 
# @param[in]
# ldc    [int]
#        ldc specifies the first dimension of C. ldc >= max( 1, m )
# 
# @param[inout]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# 
# @param[in]
# batchCount [int]
#             number of instances in the batch
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasChemmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,hipblasComplex * beta,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZhemmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,int n,int k,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,hipblasDoubleComplex * beta,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trmm performs one of the matrix-matrix operations
# 
# B := alpha*op( A )*B,   or   B := alpha*B*op( A )
# 
# where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
# non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         Specifies whether op(A) multiplies B from the left or right as follows:
#         HIPBLAS_SIDE_LEFT:       B := alpha*op( A )*B.
#         HIPBLAS_SIDE_RIGHT:      B := alpha*B*op( A ).
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         Specifies the form of op(A) to be used in the matrix multiplication as follows:
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C:  op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         Specifies whether or not A is unit triangular as follows:
#         HIPBLAS_DIAG_UNIT:      A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A is not referenced and B need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to matrix A on the GPU.
#         A has dimension ( lda, k ), where k is m
#         when  side == HIPBLAS_SIDE_LEFT  and
#         is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#     When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#     upper triangular part of the array  A must contain the upper
#     triangular matrix  and the strictly lower triangular part of
#     A is not referenced.
# 
#     When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#     lower triangular part of the array  A must contain the lower
#     triangular matrix  and the strictly upper triangular part of
#     A is not referenced.
# 
#     Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#     A  are not referenced either,  but are assumed to be  unity.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[inout]
# BP       Device pointer to the first matrix B_0 on the GPU.
#         On entry,  the leading  m by n part of the array  B must
#        contain the matrix  B,  and  on exit  is overwritten  by the
#        transformed matrix.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B. ldb >= max( 1, m ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float * AP,int lda,float * BP,int ldb) nogil



cdef hipblasStatus_t hipblasDtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,double * BP,int ldb) nogil



cdef hipblasStatus_t hipblasCtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil



cdef hipblasStatus_t hipblasZtrmm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trmmBatched performs one of the batched matrix-matrix operations
# 
# B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batchCount -1
# 
# where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
# non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         Specifies whether op(A_i) multiplies B_i from the left or right as follows:
#         HIPBLAS_SIDE_LEFT:       B_i := alpha*op( A_i )*B_i.
#         HIPBLAS_SIDE_RIGHT:      B_i := alpha*B_i*op( A_i ).
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
#         HIPBLAS_OP_N:    op(A_i) = A_i.
#         HIPBLAS_OP_T:      op(A_i) = A_i^T.
#         HIPBLAS_OP_C:  op(A_i) = A_i^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         Specifies whether or not A_i is unit triangular as follows:
#         HIPBLAS_DIAG_UNIT:      A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i is not referenced and B_i need not be set before
#         entry.
# 
# @param[in]
# AP       Device array of device pointers storing each matrix A_i on the GPU.
#         Each A_i is of dimension ( lda, k ), where k is m
#         when  side == HIPBLAS_SIDE_LEFT  and
#         is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#     When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#     upper triangular part of the array  A must contain the upper
#     triangular matrix  and the strictly lower triangular part of
#     A is not referenced.
# 
#     When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#     lower triangular part of the array  A must contain the lower
#     triangular matrix  and the strictly upper triangular part of
#     A is not referenced.
# 
#     Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#     A_i  are not referenced either,  but are assumed to be  unity.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[inout]
# BP       device array of device pointers storing each matrix B_i on the GPU.
#         On entry,  the leading  m by n part of the array  B_i must
#        contain the matrix  B_i,  and  on exit  is overwritten  by the
#        transformed matrix.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B_i. ldb >= max( 1, m ).
# 
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasStrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float *const* AP,int lda,float *const* BP,int ldb,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double *const* AP,int lda,double *const* BP,int ldb,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex *const* BP,int ldb,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrmmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* BP,int ldb,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trmmStridedBatched performs one of the strided_batched matrix-matrix operations
# 
# B_i := alpha*op( A_i )*B_i,   or   B_i := alpha*B_i*op( A_i )  for i = 0, 1, ... batchCount -1
# 
# where  alpha  is a scalar,  B_i  is an m by n matrix,  A_i  is a unit, or
# non-unit,  upper or lower triangular matrix  and  op( A_i )  is one  of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         Specifies whether op(A_i) multiplies B_i from the left or right as follows:
#         HIPBLAS_SIDE_LEFT:       B_i := alpha*op( A_i )*B_i.
#         HIPBLAS_SIDE_RIGHT:      B_i := alpha*B_i*op( A_i ).
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         Specifies whether the matrix A is an upper or lower triangular matrix as follows:
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         Specifies the form of op(A_i) to be used in the matrix multiplication as follows:
#         HIPBLAS_OP_N:    op(A_i) = A_i.
#         HIPBLAS_OP_T:      op(A_i) = A_i^T.
#         HIPBLAS_OP_C:  op(A_i) = A_i^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         Specifies whether or not A_i is unit triangular as follows:
#         HIPBLAS_DIAG_UNIT:      A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B_i. n >= 0.
# 
# @param[in]
# alpha
#         alpha specifies the scalar alpha. When alpha is
#         zero then A_i is not referenced and B_i need not be set before
#         entry.
# 
# @param[in]
# AP       Device pointer to the first matrix A_0 on the GPU.
#         Each A_i is of dimension ( lda, k ), where k is m
#         when  side == HIPBLAS_SIDE_LEFT  and
#         is  n  when  side == HIPBLAS_SIDE_RIGHT.
# 
#     When uplo == HIPBLAS_FILL_MODE_UPPER the  leading  k by k
#     upper triangular part of the array  A must contain the upper
#     triangular matrix  and the strictly lower triangular part of
#     A is not referenced.
# 
#     When uplo == HIPBLAS_FILL_MODE_LOWER the  leading  k by k
#     lower triangular part of the array  A must contain the lower
#     triangular matrix  and the strictly upper triangular part of
#     A is not referenced.
# 
#     Note that when  diag == HIPBLAS_DIAG_UNIT  the diagonal elements of
#     A_i  are not referenced either,  but are assumed to be  unity.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side == HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side == HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# 
# @param[inout]
# BP       Device pointer to the first matrix B_0 on the GPU.
#         On entry,  the leading  m by n part of the array  B_i must
#        contain the matrix  B_i,  and  on exit  is overwritten  by the
#        transformed matrix.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B_i. ldb >= max( 1, m ).
# 
#        @param[in]
# strideB  [hipblasStride]
#           stride from the start of one matrix (B_i) and the next one (B_i+1)
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasStrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,const float * AP,int lda,long strideA,float * BP,int ldb,long strideB,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,const double * AP,int lda,long strideA,double * BP,int ldb,long strideB,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrmmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# 
# trsm solves
# 
#     op(A)*X = alpha*B or  X*op(A) = alpha*B,
# 
# where alpha is a scalar, X and B are m by n matrices,
# A is triangular matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# The matrix X is overwritten on B.
# 
# Note about memory allocation:
# When trsm is launched with a k evenly divisible by the internal block size of 128,
# and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
# memory found in the handle to increase overall performance. This memory can be managed by using
# the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
# used for temporary storage will default to 1 MB and may result in chunking, which in turn may
# reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
# to the desired chunk of right hand sides to be used at a time.
# 
# (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a  lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B. n >= 0.
# 
# @param[in]
# alpha
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced and B need not be set before
#         entry.
# 
# @param[in]
# AP       device pointer storing matrix A.
#         of dimension ( lda, k ), where k is m
#         when  HIPBLAS_SIDE_LEFT  and
#         is  n  when  HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in,out]
# BP       device pointer storing matrix B.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B. ldb >= max( 1, m ).
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasStrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float * AP,int lda,float * BP,int ldb) nogil



cdef hipblasStatus_t hipblasDtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,double * BP,int ldb) nogil



cdef hipblasStatus_t hipblasCtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,hipblasComplex * BP,int ldb) nogil



cdef hipblasStatus_t hipblasZtrsm(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * BP,int ldb) nogil


# @{
# \brief BLAS Level 3 API
# \details
# trsmBatched performs the following batched operation:
# 
#     op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batchCount.
# 
# where alpha is a scalar, X and B are batched m by n matrices,
# A is triangular batched matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# Each matrix X_i is overwritten on B_i for i = 1, ..., batchCount.
# 
# Note about memory allocation:
# When trsm is launched with a k evenly divisible by the internal block size of 128,
# and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
# memory found in the handle to increase overall performance. This memory can be managed by using
# the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
# used for temporary storage will default to 1 MB and may result in chunking, which in turn may
# reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
# to the desired chunk of right hand sides to be used at a time.
# (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# @param[in]
# alpha
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced and B need not be set before
#         entry.
# @param[in]
# AP       device array of device pointers storing each matrix A_i on the GPU.
#         Matricies are of dimension ( lda, k ), where k is m
#         when  HIPBLAS_SIDE_LEFT  and is  n  when  HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# @param[in]
# lda     [int]
#         lda specifies the first dimension of each A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# @param[in,out]
# BP       device array of device pointers storing each matrix B_i on the GPU.
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# @param[in]
# batchCount [int]
#             number of trsm operatons in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasStrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float *const* AP,int lda,float ** BP,int ldb,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double *const* AP,int lda,double ** BP,int ldb,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex *const* AP,int lda,hipblasComplex ** BP,int ldb,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrsmBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex ** BP,int ldb,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# \details
# trsmSridedBatched performs the following strided batched operation:
# 
#     op(A_i)*X_i = alpha*B_i or  X_i*op(A_i) = alpha*B_i, for i = 1, ..., batchCount.
# 
# where alpha is a scalar, X and B are strided batched m by n matrices,
# A is triangular strided batched matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# Each matrix X_i is overwritten on B_i for i = 1, ..., batchCount.
# 
# Note about memory allocation:
# When trsm is launched with a k evenly divisible by the internal block size of 128,
# and is no larger than 10 of these blocks, the API takes advantage of utilizing pre-allocated
# memory found in the handle to increase overall performance. This memory can be managed by using
# the environment variable WORKBUF_TRSM_B_CHNK. When this variable is not set the device memory
# used for temporary storage will default to 1 MB and may result in chunking, which in turn may
# reduce performance. Under these circumstances it is recommended that WORKBUF_TRSM_B_CHNK be set
# to the desired chunk of right hand sides to be used at a time.
# (where k is m when HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT)
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a  lower triangular matrix.
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# @param[in]
# alpha
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced and B need not be set before
#         entry.
# @param[in]
# AP       device pointer pointing to the first matrix A_1.
#         of dimension ( lda, k ), where k is m
#         when  HIPBLAS_SIDE_LEFT  and
#         is  n  when  HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# @param[in]
# lda     [int]
#         lda specifies the first dimension of each A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# @param[in]
# strideA [hipblasStride]
#          stride from the start of one A_i matrix to the next A_(i + 1).
# @param[in,out]
# BP       device pointer pointing to the first matrix B_1.
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# @param[in]
# strideB [hipblasStride]
#          stride from the start of one B_i matrix to the next B_(i + 1).
# @param[in]
# batchCount [int]
#             number of trsm operatons in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasStrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const float * alpha,float * AP,int lda,long strideA,float * BP,int ldb,long strideB,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const double * alpha,double * AP,int lda,long strideA,double * BP,int ldb,long strideB,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasComplex * alpha,hipblasComplex * AP,int lda,long strideA,hipblasComplex * BP,int ldb,long strideB,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrsmStridedBatched(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,hipblasDoubleComplex * alpha,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * BP,int ldb,long strideB,int batchCount) nogil


#    @{
#    \brief BLAS Level 3 API
# 
#    \details
#    trtri  compute the inverse of a matrix A, namely, invA
# 
#        and write the result into invA;
# 
#    - Supported precisions in rocBLAS : s,d,c,z
#    - Supported precisions in cuBLAS  : No support
# 
#    @param[in]
#    handle    [hipblasHandle_t]
#              handle to the hipblas library context queue.
#    @param[in]
#    uplo      [hipblasFillMode_t]
#              specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
#              if HIPBLAS_FILL_MODE_UPPER, the lower part of A is not referenced
#              if HIPBLAS_FILL_MODE_LOWER, the upper part of A is not referenced
#    @param[in]
#    diag      [hipblasDiagType_t]
#              = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#              = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
#    @param[in]
#    n         [int]
#              size of matrix A and invA
#    @param[in]
#    AP         device pointer storing matrix A.
#    @param[in]
#    lda       [int]
#              specifies the leading dimension of A.
#    @param[out]
#    invA      device pointer storing matrix invA.
#    @param[in]
#    ldinvA    [int]
#              specifies the leading dimension of invA.
# 
# ******************************************************************
cdef hipblasStatus_t hipblasStrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float * AP,int lda,float * invA,int ldinvA) nogil



cdef hipblasStatus_t hipblasDtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,double * invA,int ldinvA) nogil



cdef hipblasStatus_t hipblasCtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,hipblasComplex * invA,int ldinvA) nogil



cdef hipblasStatus_t hipblasZtrtri(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * invA,int ldinvA) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# trtriBatched  compute the inverse of A_i and write into invA_i where
#                A_i and invA_i are the i-th matrices in the batch,
#                for i = 1, ..., batchCount.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
# @param[in]
# diag      [hipblasDiagType_t]
#           = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#           = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
# @param[in]
# n         [int]
# @param[in]
# AP         device array of device pointers storing each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[out]
# invA      device array of device pointers storing the inverse of each matrix A_i.
#           Partial inplace operation is supported, see below.
#           If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
#           the inverse of the upper triangular matrix, and the strictly lower
#           triangular part of invA is cleared.
#           If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
#           the inverse of the lower triangular matrix, and the strictly upper
#           triangular part of invA is cleared.
# @param[in]
# ldinvA    [int]
#           specifies the leading dimension of each invA_i.
# @param[in]
# batchCount [int]
#           numbers of matrices in the batch
#  ******************************************************************
cdef hipblasStatus_t hipblasStrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float *const* AP,int lda,float ** invA,int ldinvA,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double *const* AP,int lda,double ** invA,int ldinvA,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex *const* AP,int lda,hipblasComplex ** invA,int ldinvA,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrtriBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex ** invA,int ldinvA,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# trtriStridedBatched compute the inverse of A_i and write into invA_i where
#                A_i and invA_i are the i-th matrices in the batch,
#                for i = 1, ..., batchCount
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# uplo      [hipblasFillMode_t]
#           specifies whether the upper 'HIPBLAS_FILL_MODE_UPPER' or lower 'HIPBLAS_FILL_MODE_LOWER'
# @param[in]
# diag      [hipblasDiagType_t]
#           = 'HIPBLAS_DIAG_NON_UNIT', A is non-unit triangular;
#           = 'HIPBLAS_DIAG_UNIT', A is unit triangular;
# @param[in]
# n         [int]
# @param[in]
# AP         device pointer pointing to address of first matrix A_1.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A.
# @param[in]
# strideA  [hipblasStride]
#          "batch stride a": stride from the start of one A_i matrix to the next A_(i + 1).
# @param[out]
# invA      device pointer storing the inverses of each matrix A_i.
#           Partial inplace operation is supported, see below.
#           If UPLO = 'U', the leading N-by-N upper triangular part of the invA will store
#           the inverse of the upper triangular matrix, and the strictly lower
#           triangular part of invA is cleared.
#           If UPLO = 'L', the leading N-by-N lower triangular part of the invA will store
#           the inverse of the lower triangular matrix, and the strictly upper
#           triangular part of invA is cleared.
# @param[in]
# ldinvA    [int]
#           specifies the leading dimension of each invA_i.
# @param[in]
# stride_invA  [hipblasStride]
#              "batch stride invA": stride from the start of one invA_i matrix to the next invA_(i + 1).
# @param[in]
# batchCount  [int]
#              numbers of matrices in the batch
#  ******************************************************************
cdef hipblasStatus_t hipblasStrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const float * AP,int lda,long strideA,float * invA,int ldinvA,long stride_invA,int batchCount) nogil



cdef hipblasStatus_t hipblasDtrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,const double * AP,int lda,long strideA,double * invA,int ldinvA,long stride_invA,int batchCount) nogil



cdef hipblasStatus_t hipblasCtrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasComplex * AP,int lda,long strideA,hipblasComplex * invA,int ldinvA,long stride_invA,int batchCount) nogil



cdef hipblasStatus_t hipblasZtrtriStridedBatched(void * handle,hipblasFillMode_t uplo,hipblasDiagType_t diag,int n,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * invA,int ldinvA,long stride_invA,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# dgmm performs one of the matrix-matrix operations
# 
#     C = A * diag(x) if side == HIPBLAS_SIDE_RIGHT
#     C = diag(x) * A if side == HIPBLAS_SIDE_LEFT
# 
# where C and A are m by n dimensional matrices. diag( x ) is a diagonal matrix
# and x is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
# if side == HIPBLAS_SIDE_LEFT.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : s,d,c,z
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side      [hipblasSideMode_t]
#           specifies the side of diag(x)
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# AP         device pointer storing matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# incx      [int]
#           specifies the increment between values of x
# @param[in, out]
# CP         device pointer storing matrix C.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSdgmm(void * handle,hipblasSideMode_t side,int m,int n,const float * AP,int lda,const float * x,int incx,float * CP,int ldc) nogil



cdef hipblasStatus_t hipblasDdgmm(void * handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,const double * x,int incx,double * CP,int ldc) nogil



cdef hipblasStatus_t hipblasCdgmm(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,hipblasComplex * x,int incx,hipblasComplex * CP,int ldc) nogil



cdef hipblasStatus_t hipblasZdgmm(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,hipblasDoubleComplex * x,int incx,hipblasDoubleComplex * CP,int ldc) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# dgmmBatched performs one of the batched matrix-matrix operations
# 
#     C_i = A_i * diag(x_i) for i = 0, 1, ... batchCount-1 if side == HIPBLAS_SIDE_RIGHT
#     C_i = diag(x_i) * A_i for i = 0, 1, ... batchCount-1 if side == HIPBLAS_SIDE_LEFT
# 
# where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
# and x_i is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
# if side == HIPBLAS_SIDE_LEFT.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side      [hipblasSideMode_t]
#           specifies the side of diag(x)
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# AP         device array of device pointers storing each matrix A_i on the GPU.
#           Each A_i is of dimension ( lda, n )
# @param[in]
# lda       [int]
#           specifies the leading dimension of A_i.
# @param[in]
# x         device array of device pointers storing each vector x_i on the GPU.
#           Each x_i is of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension
#           m if side == HIPBLAS_SIDE_LEFT
# @param[in]
# incx      [int]
#           specifies the increment between values of x_i
# @param[in, out]
# CP         device array of device pointers storing each matrix C_i on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,const float *const* AP,int lda,const float *const* x,int incx,float *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasDdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,const double *const* AP,int lda,const double *const* x,int incx,double *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasCdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex *const* AP,int lda,hipblasComplex *const* x,int incx,hipblasComplex *const* CP,int ldc,int batchCount) nogil



cdef hipblasStatus_t hipblasZdgmmBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex *const* AP,int lda,hipblasDoubleComplex *const* x,int incx,hipblasDoubleComplex *const* CP,int ldc,int batchCount) nogil


# @{
# \brief BLAS Level 3 API
# 
# \details
# dgmmStridedBatched performs one of the batched matrix-matrix operations
# 
#     C_i = A_i * diag(x_i)   if side == HIPBLAS_SIDE_RIGHT   for i = 0, 1, ... batchCount-1
#     C_i = diag(x_i) * A_i   if side == HIPBLAS_SIDE_LEFT    for i = 0, 1, ... batchCount-1
# 
# where C_i and A_i are m by n dimensional matrices. diag(x_i) is a diagonal matrix
# and x_i is vector of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension m
# if side == HIPBLAS_SIDE_LEFT.
# 
# - Supported precisions in rocBLAS : s,d,c,z
# - Supported precisions in cuBLAS  : No support
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# side      [hipblasSideMode_t]
#           specifies the side of diag(x)
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# AP         device pointer to the first matrix A_0 on the GPU.
#           Each A_i is of dimension ( lda, n )
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# strideA  [hipblasStride]
#           stride from the start of one matrix (A_i) and the next one (A_i+1)
# @param[in]
# x         pointer to the first vector x_0 on the GPU.
#           Each x_i is of dimension n if side == HIPBLAS_SIDE_RIGHT and dimension
#           m if side == HIPBLAS_SIDE_LEFT
# @param[in]
# incx      [int]
#           specifies the increment between values of x
# @param[in]
# stridex  [hipblasStride]
#           stride from the start of one vector(x_i) and the next one (x_i+1)
# @param[in, out]
# CP         device pointer to the first matrix C_0 on the GPU.
#           Each C_i is of dimension ( ldc, n ).
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# @param[in]
# strideC  [hipblasStride]
#           stride from the start of one matrix (C_i) and the next one (C_i+1)
# @param[in]
# batchCount [int]
#             number of instances i in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,const float * AP,int lda,long strideA,const float * x,int incx,long stridex,float * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasDdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,const double * AP,int lda,long strideA,const double * x,int incx,long stridex,double * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasCdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasComplex * AP,int lda,long strideA,hipblasComplex * x,int incx,long stridex,hipblasComplex * CP,int ldc,long strideC,int batchCount) nogil



cdef hipblasStatus_t hipblasZdgmmStridedBatched(void * handle,hipblasSideMode_t side,int m,int n,hipblasDoubleComplex * AP,int lda,long strideA,hipblasDoubleComplex * x,int incx,long stridex,hipblasDoubleComplex * CP,int ldc,long strideC,int batchCount) nogil


# @{
# \brief SOLVER API
# 
# \details
# getrf computes the LU factorization of a general n-by-n matrix A
# using partial pivoting with row interchanges. The LU factorization can
# be done without pivoting if ipiv is passed as a nullptr.
# 
# In the case that ipiv is not null, the factorization has the form:
# 
# \f[
#     A = PLU
# \f]
# 
# where P is a permutation matrix, L is lower triangular with unit
# diagonal elements, and U is upper triangular.
# 
# In the case that ipiv is null, the factorization is done without pivoting:
# 
# \f[
#     A = LU
# \f]
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns and rows of the matrix A.
# @param[inout]
# A         pointer to type. Array on the GPU of dimension lda*n.\n
#           On entry, the n-by-n matrix A to be factored.
#           On exit, the factors L and U from the factorization.
#           The unit diagonal elements of L are not stored.
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of A.
# @param[out]
# ipiv      pointer to int. Array on the GPU of dimension n.\n
#           The vector of pivot indices. Elements of ipiv are 1-based indices.
#           For 1 <= i <= n, the row i of the
#           matrix was interchanged with row ipiv[i].
#           Matrix P of the factorization can be derived from ipiv.
#           The factorization here can be done without pivoting if ipiv is passed
#           in as a nullptr.
# @param[out]
# info      pointer to a int on the GPU.\n
#           If info = 0, successful exit.
#           If info = j > 0, U is singular. U[j,j] is the first zero pivot.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgetrf(void * handle,const int n,float * A,const int lda,int * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasDgetrf(void * handle,const int n,double * A,const int lda,int * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasCgetrf(void * handle,const int n,hipblasComplex * A,const int lda,int * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasZgetrf(void * handle,const int n,hipblasDoubleComplex * A,const int lda,int * ipiv,int * info) nogil


# @{
# \brief SOLVER API
# 
# \details
# getrfBatched computes the LU factorization of a batch of general
# n-by-n matrices using partial pivoting with row interchanges. The LU factorization can
# be done without pivoting if ipiv is passed as a nullptr.
# 
# In the case that ipiv is not null, the factorization of matrix \f$A_i\f$ in the batch has the form:
# 
# \f[
#     A_i = P_iL_iU_i
# \f]
# 
# where \f$P_i\f$ is a permutation matrix, \f$L_i\f$ is lower triangular with unit
# diagonal elements, and \f$U_i\f$ is upper triangular.
# 
# In the case that ipiv is null, the factorization is done without pivoting:
# 
# \f[
#     A_i = L_iU_i
# \f]
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns and rows of all matrices A_i in the batch.
# @param[inout]
# A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#           On entry, the n-by-n matrices A_i to be factored.
#           On exit, the factors L_i and U_i from the factorizations.
#           The unit diagonal elements of L_i are not stored.
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of matrices A_i.
# @param[out]
# ipiv      pointer to int. Array on the GPU.\n
#           Contains the vectors of pivot indices ipiv_i (corresponding to A_i).
#           Dimension of ipiv_i is n.
#           Elements of ipiv_i are 1-based indices.
#           For each instance A_i in the batch and for 1 <= j <= n, the row j of the
#           matrix A_i was interchanged with row ipiv_i[j].
#           Matrix P_i of the factorization can be derived from ipiv_i.
#           The factorization here can be done without pivoting if ipiv is passed
#           in as a nullptr.
# @param[out]
# info      pointer to int. Array of batchCount integers on the GPU.\n
#           If info[i] = 0, successful exit for factorization of A_i.
#           If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of matrices in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgetrfBatched(void * handle,const int n,float *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgetrfBatched(void * handle,const int n,double *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgetrfBatched(void * handle,const int n,hipblasComplex *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgetrfBatched(void * handle,const int n,hipblasDoubleComplex *const* A,const int lda,int * ipiv,int * info,const int batchCount) nogil


# @{
# \brief SOLVER API
# 
# \details
# getrfStridedBatched computes the LU factorization of a batch of
# general n-by-n matrices using partial pivoting with row interchanges. The LU factorization can
# be done without pivoting if ipiv is passed as a nullptr.
# 
# In the case that ipiv is not null, the factorization of matrix \f$A_i\f$ in the batch has the form:
# 
# \f[
#     A_i = P_iL_iU_i
# \f]
# 
# where \f$P_i\f$ is a permutation matrix, \f$L_i\f$ is lower triangular with unit
# diagonal elements, and \f$U_i\f$ is upper triangular.
# 
# In the case that ipiv is null, the factorization is done without pivoting:
# 
# \f[
#     A_i = L_iU_i
# \f]
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns and rows of all matrices A_i in the batch.
# @param[inout]
# A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#           On entry, the n-by-n matrices A_i to be factored.
#           On exit, the factors L_i and U_i from the factorization.
#           The unit diagonal elements of L_i are not stored.
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of matrices A_i.
# @param[in]
# strideA   hipblasStride.\n
#           Stride from the start of one matrix A_i to the next one A_(i+1).
#           There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
# @param[out]
# ipiv      pointer to int. Array on the GPU (the size depends on the value of strideP).\n
#           Contains the vectors of pivots indices ipiv_i (corresponding to A_i).
#           Dimension of ipiv_i is n.
#           Elements of ipiv_i are 1-based indices.
#           For each instance A_i in the batch and for 1 <= j <= n, the row j of the
#           matrix A_i was interchanged with row ipiv_i[j].
#           Matrix P_i of the factorization can be derived from ipiv_i.
#           The factorization here can be done without pivoting if ipiv is passed
#           in as a nullptr.
# @param[in]
# strideP   hipblasStride.\n
#           Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
#           There is no restriction for the value of strideP. Normal use case is strideP >= n.
# @param[out]
# info      pointer to int. Array of batchCount integers on the GPU.\n
#           If info[i] = 0, successful exit for factorization of A_i.
#           If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of matrices in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgetrfStridedBatched(void * handle,const int n,float * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgetrfStridedBatched(void * handle,const int n,double * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgetrfStridedBatched(void * handle,const int n,hipblasComplex * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgetrfStridedBatched(void * handle,const int n,hipblasDoubleComplex * A,const int lda,const long strideA,int * ipiv,const long strideP,int * info,const int batchCount) nogil


# @{
# \brief SOLVER API
# 
# \details
# getrs solves a system of n linear equations on n variables in its factorized form.
# 
# It solves one of the following systems, depending on the value of trans:
# 
# \f[
#     \begin{array}{cl}
#     A X = B & \: \text{not transposed,}\\
#     A^T X = B & \: \text{transposed, or}\\
#     A^H X = B & \: \text{conjugate transposed.}
#     \end{array}
# \f]
# 
# Matrix A is defined by its triangular factors as returned by \ref hipblasSgetrf "getrf".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# n           int. n >= 0.\n
#             The order of the system, i.e. the number of columns and rows of A.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of right hand sides, i.e., the number of columns
#             of the matrix B.
# @param[in]
# A           pointer to type. Array on the GPU of dimension lda*n.\n
#             The factors L and U of the factorization A = P*L*U returned by \ref hipblasSgetrf "getrf".
# @param[in]
# lda         int. lda >= n.\n
#             The leading dimension of A.
# @param[in]
# ipiv        pointer to int. Array on the GPU of dimension n.\n
#             The pivot indices returned by \ref hipblasSgetrf "getrf".
# @param[in,out]
# B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
#             On entry, the right hand side matrix B.
#             On exit, the solution matrix X.
# @param[in]
# ldb         int. ldb >= n.\n
#             The leading dimension of B.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
# ******************************************************************
cdef hipblasStatus_t hipblasSgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float * A,const int lda,const int * ipiv,float * B,const int ldb,int * info) nogil



cdef hipblasStatus_t hipblasDgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const int * ipiv,double * B,const int ldb,int * info) nogil



cdef hipblasStatus_t hipblasCgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const int * ipiv,hipblasComplex * B,const int ldb,int * info) nogil



cdef hipblasStatus_t hipblasZgetrs(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const int * ipiv,hipblasDoubleComplex * B,const int ldb,int * info) nogil


# @{
# \brief SOLVER API
# 
# \details getrsBatched solves a batch of systems of n linear equations on n
# variables in its factorized forms.
# 
# For each instance i in the batch, it solves one of the following systems, depending on the value of trans:
# 
# \f[
#     \begin{array}{cl}
#     A_i X_i = B_i & \: \text{not transposed,}\\
#     A_i^T X_i = B_i & \: \text{transposed, or}\\
#     A_i^H X_i = B_i & \: \text{conjugate transposed.}
#     \end{array}
# \f]
# 
# Matrix \f$A_i\f$ is defined by its triangular factors as returned by \ref hipblasSgetrfBatched "getrfBatched".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations of each instance in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The order of the system, i.e. the number of columns and rows of all A_i matrices.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of right hand sides, i.e., the number of columns
#             of all the matrices B_i.
# @param[in]
# A           Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#             The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by \ref hipblasSgetrfBatched "getrfBatched".
# @param[in]
# lda         int. lda >= n.\n
#             The leading dimension of matrices A_i.
# @param[in]
# ipiv        pointer to int. Array on the GPU.\n
#             Contains the vectors ipiv_i of pivot indices returned by \ref hipblasSgetrfBatched "getrfBatched".
# @param[in,out]
# B           Array of pointers to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
#             On entry, the right hand side matrices B_i.
#             On exit, the solution matrix X_i of each system in the batch.
# @param[in]
# ldb         int. ldb >= n.\n
#             The leading dimension of matrices B_i.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of instances (systems) in the batch.
# 
# ******************************************************************
cdef hipblasStatus_t hipblasSgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float *const* A,const int lda,const int * ipiv,float *const* B,const int ldb,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double *const* A,const int lda,const int * ipiv,double *const* B,const int ldb,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex *const* A,const int lda,const int * ipiv,hipblasComplex *const* B,const int ldb,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgetrsBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex *const* A,const int lda,const int * ipiv,hipblasDoubleComplex *const* B,const int ldb,int * info,const int batchCount) nogil


# @{
# \brief SOLVER API
# 
# \details
# getrsStridedBatched solves a batch of systems of n linear equations
# on n variables in its factorized forms.
# 
# For each instance i in the batch, it solves one of the following systems, depending on the value of trans:
# 
# \f[
#     \begin{array}{cl}
#     A_i X_i = B_i & \: \text{not transposed,}\\
#     A_i^T X_i = B_i & \: \text{transposed, or}\\
#     A_i^H X_i = B_i & \: \text{conjugate transposed.}
#     \end{array}
# \f]
# 
# Matrix \f$A_i\f$ is defined by its triangular factors as returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : No support
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations of each instance in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The order of the system, i.e. the number of columns and rows of all A_i matrices.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of right hand sides, i.e., the number of columns
#             of all the matrices B_i.
# @param[in]
# A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#             The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".
# @param[in]
# lda         int. lda >= n.\n
#             The leading dimension of matrices A_i.
# @param[in]
# strideA     hipblasStride.\n
#             Stride from the start of one matrix A_i to the next one A_(i+1).
#             There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
# @param[in]
# ipiv        pointer to int. Array on the GPU (the size depends on the value of strideP).\n
#             Contains the vectors ipiv_i of pivot indices returned by \ref hipblasSgetrfStridedBatched "getrfStridedBatched".
# @param[in]
# strideP     hipblasStride.\n
#             Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
#             There is no restriction for the value of strideP. Normal use case is strideP >= n.
# @param[in,out]
# B           pointer to type. Array on the GPU (size depends on the value of strideB).\n
#             On entry, the right hand side matrices B_i.
#             On exit, the solution matrix X_i of each system in the batch.
# @param[in]
# ldb         int. ldb >= n.\n
#             The leading dimension of matrices B_i.
# @param[in]
# strideB     hipblasStride.\n
#             Stride from the start of one matrix B_i to the next one B_(i+1).
#             There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of instances (systems) in the batch.
# 
# ******************************************************************
cdef hipblasStatus_t hipblasSgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,float * A,const int lda,const long strideA,const int * ipiv,const long strideP,float * B,const int ldb,const long strideB,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,double * A,const int lda,const long strideA,const int * ipiv,const long strideP,double * B,const int ldb,const long strideB,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasComplex * A,const int lda,const long strideA,const int * ipiv,const long strideP,hipblasComplex * B,const int ldb,const long strideB,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgetrsStridedBatched(void * handle,hipblasOperation_t trans,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const long strideA,const int * ipiv,const long strideP,hipblasDoubleComplex * B,const int ldb,const long strideB,int * info,const int batchCount) nogil


# @{
# \brief SOLVER API
# 
# \details
# getriBatched computes the inverse \f$C_i = A_i^{-1}\f$ of a batch of general n-by-n matrices \f$A_i\f$.
# 
# The inverse is computed by solving the linear system
# 
# \f[
#     A_i C_i = I
# \f]
# 
# where I is the identity matrix, and \f$A_i\f$ is factorized as \f$A_i = P_i  L_i  U_i\f$ as given by \ref hipblasSgetrfBatched "getrfBatched".
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# n         int. n >= 0.\n
#           The number of rows and columns of all matrices A_i in the batch.
# @param[in]
# A         array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#           The factors L_i and U_i of the factorization A_i = P_i*L_i*U_i returned by \ref hipblasSgetrfBatched "getrfBatched".
# @param[in]
# lda       int. lda >= n.\n
#           Specifies the leading dimension of matrices A_i.
# @param[in]
# ipiv      pointer to int. Array on the GPU (the size depends on the value of strideP).\n
#           The pivot indices returned by \ref hipblasSgetrfBatched "getrfBatched".
#           ipiv can be passed in as a nullptr, this will assume that getrfBatched was called without partial pivoting.
# @param[out]
# C         array of pointers to type. Each pointer points to an array on the GPU of dimension ldc*n.\n
#           If info[i] = 0, the inverse of matrices A_i. Otherwise, undefined.
# @param[in]
# ldc       int. ldc >= n.\n
#           Specifies the leading dimension of C_i.
# @param[out]
# info      pointer to int. Array of batchCount integers on the GPU.\n
#           If info[i] = 0, successful exit for inversion of A_i.
#           If info[i] = j > 0, U_i is singular. U_i[j,j] is the first zero pivot.
# @param[in]
# batchCount int. batchCount >= 0.\n
#             Number of matrices in the batch.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgetriBatched(void * handle,const int n,float *const* A,const int lda,int * ipiv,float *const* C,const int ldc,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgetriBatched(void * handle,const int n,double *const* A,const int lda,int * ipiv,double *const* C,const int ldc,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgetriBatched(void * handle,const int n,hipblasComplex *const* A,const int lda,int * ipiv,hipblasComplex *const* C,const int ldc,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgetriBatched(void * handle,const int n,hipblasDoubleComplex *const* A,const int lda,int * ipiv,hipblasDoubleComplex *const* C,const int ldc,int * info,const int batchCount) nogil


# @{
# \brief GELS solves an overdetermined (or underdetermined) linear system defined by an m-by-n
# matrix A, and a corresponding matrix B, using the QR factorization computed by \ref hipblasSgeqrf "GEQRF" (or the LQ
# factorization computed by "GELQF").
# 
# \details
# Depending on the value of trans, the problem solved by this function is either of the form
# 
# \f[
#     \begin{array}{cl}
#     A X = B & \: \text{not transposed, or}\\
#     A' X = B & \: \text{transposed if real, or conjugate transposed if complex}
#     \end{array}
# \f]
# 
# If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
# and a least-squares solution approximating X is found by minimizing
# 
# \f[
#     || B - A  X || \quad \text{(or} \: || B - A' X ||\text{)}
# \f]
# 
# If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
# and a unique solution for X is chosen such that \f$|| X ||\f$ is minimal.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : currently unsupported
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# m           int. m >= 0.\n
#             The number of rows of matrix A.
# @param[in]
# n           int. n >= 0.\n
#             The number of columns of matrix A.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of columns of matrices B and X;
#             i.e., the columns on the right hand side.
# @param[inout]
# A           pointer to type. Array on the GPU of dimension lda*n.\n
#             On entry, the matrix A.
#             On exit, the QR (or LQ) factorization of A as returned by "GEQRF" (or "GELQF").
# @param[in]
# lda         int. lda >= m.\n
#             Specifies the leading dimension of matrix A.
# @param[inout]
# B           pointer to type. Array on the GPU of dimension ldb*nrhs.\n
#             On entry, the matrix B.
#             On exit, when info = 0, B is overwritten by the solution vectors (and the residuals in
#             the overdetermined cases) stored as columns.
# @param[in]
# ldb         int. ldb >= max(m,n).\n
#             Specifies the leading dimension of matrix B.
# @param[out]
# info        pointer to an int on the host.\n
#             If info = 0, successful exit.
#             If info = j < 0, the j-th argument is invalid.
# @param[out]
# deviceInfo  pointer to int on the GPU.\n
#             If info = 0, successful exit.
#             If info = i > 0, the solution could not be computed because input matrix A is
#             rank deficient; the i-th diagonal element of its triangular factor is zero.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float * A,const int lda,float * B,const int ldb,int * info,int * deviceInfo) nogil



cdef hipblasStatus_t hipblasDgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,double * B,const int ldb,int * info,int * deviceInfo) nogil



cdef hipblasStatus_t hipblasCgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,hipblasComplex * B,const int ldb,int * info,int * deviceInfo) nogil



cdef hipblasStatus_t hipblasZgels(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * B,const int ldb,int * info,int * deviceInfo) nogil


# @{
# \brief gelsBatched solves a batch of overdetermined (or underdetermined) linear systems
# defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$, using the
# QR factorizations computed by "GEQRF_BATCHED" (or the LQ factorizations computed by "GELQF_BATCHED").
# 
# \details
# For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form
# 
# \f[
#     \begin{array}{cl}
#     A_j X_j = B_j & \: \text{not transposed, or}\\
#     A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
#     \end{array}
# \f]
# 
# If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
# and a least-squares solution approximating X_j is found by minimizing
# 
# \f[
#     || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
# \f]
# 
# If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
# and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# Note that cuBLAS backend supports only the non-transpose operation and only solves over-determined systems (m >= n).
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# m           int. m >= 0.\n
#             The number of rows of all matrices A_j in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The number of columns of all matrices A_j in the batch.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of columns of all matrices B_j and X_j in the batch;
#             i.e., the columns on the right hand side.
# @param[inout]
# A           array of pointer to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#             On entry, the matrices A_j.
#             On exit, the QR (or LQ) factorizations of A_j as returned by "GEQRF_BATCHED"
#             (or "GELQF_BATCHED").
# @param[in]
# lda         int. lda >= m.\n
#             Specifies the leading dimension of matrices A_j.
# @param[inout]
# B           array of pointer to type. Each pointer points to an array on the GPU of dimension ldb*nrhs.\n
#             On entry, the matrices B_j.
#             On exit, when info[j] = 0, B_j is overwritten by the solution vectors (and the residuals in
#             the overdetermined cases) stored as columns.
# @param[in]
# ldb         int. ldb >= max(m,n).\n
#             Specifies the leading dimension of matrices B_j.
# @param[out]
# info        pointer to an int on the host.\n
#             If info = 0, successful exit.
#             If info = j < 0, the j-th argument is invalid.
# @param[out]
# deviceInfo  pointer to int. Array of batchCount integers on the GPU.\n
#             If deviceInfo[j] = 0, successful exit for solution of A_j.
#             If deviceInfo[j] = i > 0, the solution of A_j could not be computed because input
#             matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#             Number of matrices in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float *const* A,const int lda,float *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double *const* A,const int lda,double *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex *const* A,const int lda,hipblasComplex *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgelsBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex *const* A,const int lda,hipblasDoubleComplex *const* B,const int ldb,int * info,int * deviceInfo,const int batchCount) nogil


# @{
# \brief gelsStridedBatched solves a batch of overdetermined (or underdetermined) linear
# systems defined by a set of m-by-n matrices \f$A_j\f$, and corresponding matrices \f$B_j\f$,
# using the QR factorizations computed by "GEQRF_STRIDED_BATCHED"
# (or the LQ factorizations computed by "GELQF_STRIDED_BATCHED").
# 
# \details
# For each instance in the batch, depending on the value of trans, the problem solved by this function is either of the form
# 
# \f[
#     \begin{array}{cl}
#     A_j X_j = B_j & \: \text{not transposed, or}\\
#     A_j' X_j = B_j & \: \text{transposed if real, or conjugate transposed if complex}
#     \end{array}
# \f]
# 
# If m >= n (or m < n in the case of transpose/conjugate transpose), the system is overdetermined
# and a least-squares solution approximating X_j is found by minimizing
# 
# \f[
#     || B_j - A_j  X_j || \quad \text{(or} \: || B_j - A_j' X_j ||\text{)}
# \f]
# 
# If m < n (or m >= n in the case of transpose/conjugate transpose), the system is underdetermined
# and a unique solution for X_j is chosen such that \f$|| X_j ||\f$ is minimal.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : currently unsupported
# 
# @param[in]
# handle      hipblasHandle_t.
# @param[in]
# trans       hipblasOperation_t.\n
#             Specifies the form of the system of equations.
# @param[in]
# m           int. m >= 0.\n
#             The number of rows of all matrices A_j in the batch.
# @param[in]
# n           int. n >= 0.\n
#             The number of columns of all matrices A_j in the batch.
# @param[in]
# nrhs        int. nrhs >= 0.\n
#             The number of columns of all matrices B_j and X_j in the batch;
#             i.e., the columns on the right hand side.
# @param[inout]
# A           pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#             On entry, the matrices A_j.
#             On exit, the QR (or LQ) factorizations of A_j as returned by "GEQRF_STRIDED_BATCHED"
#             (or "GELQF_STRIDED_BATCHED").
# @param[in]
# lda         int. lda >= m.\n
#             Specifies the leading dimension of matrices A_j.
# @param[in]
# strideA     hipblasStride.\n
#             Stride from the start of one matrix A_j to the next one A_(j+1).
#             There is no restriction for the value of strideA. Normal use case is strideA >= lda*n
# @param[inout]
# B           pointer to type. Array on the GPU (the size depends on the value of strideB).\n
#             On entry, the matrices B_j.
#             On exit, when info[j] = 0, each B_j is overwritten by the solution vectors (and the residuals in
#             the overdetermined cases) stored as columns.
# @param[in]
# ldb         int. ldb >= max(m,n).\n
#             Specifies the leading dimension of matrices B_j.
# @param[in]
# strideB     hipblasStride.\n
#             Stride from the start of one matrix B_j to the next one B_(j+1).
#             There is no restriction for the value of strideB. Normal use case is strideB >= ldb*nrhs
# @param[out]
# info        pointer to an int on the host.\n
#             If info = 0, successful exit.
#             If info = j < 0, the j-th argument is invalid.
# @param[out]
# deviceInfo  pointer to int. Array of batchCount integers on the GPU.\n
#             If deviceInfo[j] = 0, successful exit for solution of A_j.
#             If deviceInfo[j] = i > 0, the solution of A_j could not be computed because input
#             matrix A_j is rank deficient; the i-th diagonal element of its triangular factor is zero.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#             Number of matrices in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,float * A,const int lda,const long strideA,float * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil



cdef hipblasStatus_t hipblasDgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,double * A,const int lda,const long strideA,double * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil



cdef hipblasStatus_t hipblasCgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasComplex * A,const int lda,const long strideA,hipblasComplex * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil



cdef hipblasStatus_t hipblasZgelsStridedBatched(void * handle,hipblasOperation_t trans,const int m,const int n,const int nrhs,hipblasDoubleComplex * A,const int lda,const long strideA,hipblasDoubleComplex * B,const int ldb,const long strideB,int * info,int * deviceInfo,const int batch_count) nogil


# @{
# \brief SOLVER API
# 
# \details
# geqrf computes a QR factorization of a general m-by-n matrix A.
# 
# The factorization has the form
# 
# \f[
#     A = Q\left[\begin{array}{c}
#     R\\
#     0
#     \end{array}\right]
# \f]
# 
# where R is upper triangular (upper trapezoidal if m < n), and Q is
# a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
# \f[
#     Q = H_1H_2\cdots H_k, \quad \text{with} \: k = \text{min}(m,n)
# \f]
# 
# Each Householder matrix \f$H_i\f$ is given by
# 
# \f[
#     H_i = I - \text{ipiv}[i] \cdot v_i v_i'
# \f]
# 
# where the first i-1 elements of the Householder vector \f$v_i\f$ are zero, and \f$v_i[i] = 1\f$.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# m         int. m >= 0.\n
#           The number of rows of the matrix A.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns of the matrix A.
# @param[inout]
# A         pointer to type. Array on the GPU of dimension lda*n.\n
#           On entry, the m-by-n matrix to be factored.
#           On exit, the elements on and above the diagonal contain the
#           factor R; the elements below the diagonal are the last m - i elements
#           of Householder vector v_i.
# @param[in]
# lda       int. lda >= m.\n
#           Specifies the leading dimension of A.
# @param[out]
# ipiv      pointer to type. Array on the GPU of dimension min(m,n).\n
#           The Householder scalars.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = j < 0, the j-th argument is invalid.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasSgeqrf(void * handle,const int m,const int n,float * A,const int lda,float * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasDgeqrf(void * handle,const int m,const int n,double * A,const int lda,double * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasCgeqrf(void * handle,const int m,const int n,hipblasComplex * A,const int lda,hipblasComplex * ipiv,int * info) nogil



cdef hipblasStatus_t hipblasZgeqrf(void * handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,hipblasDoubleComplex * ipiv,int * info) nogil


# @{
# \brief SOLVER API
# 
# \details
# geqrfBatched computes the QR factorization of a batch of general
# m-by-n matrices.
# 
# The factorization of matrix \f$A_i\f$ in the batch has the form
# 
# \f[
#     A_i = Q_i\left[\begin{array}{c}
#     R_i\\
#     0
#     \end{array}\right]
# \f]
# 
# where \f$R_i\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_i\f$ is
# a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
# \f[
#     Q_i = H_{i_1}H_{i_2}\cdots H_{i_k}, \quad \text{with} \: k = \text{min}(m,n)
# \f]
# 
# Each Householder matrix \f$H_{i_j}\f$ is given by
# 
# \f[
#     H_{i_j} = I - \text{ipiv}_i[j] \cdot v_{i_j} v_{i_j}'
# \f]
# 
# where the first j-1 elements of Householder vector \f$v_{i_j}\f$ are zero, and \f$v_{i_j}[j] = 1\f$.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : s,d,c,z
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# m         int. m >= 0.\n
#           The number of rows of all the matrices A_i in the batch.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns of all the matrices A_i in the batch.
# @param[inout]
# A         Array of pointers to type. Each pointer points to an array on the GPU of dimension lda*n.\n
#           On entry, the m-by-n matrices A_i to be factored.
#           On exit, the elements on and above the diagonal contain the
#           factor R_i. The elements below the diagonal are the last m - j elements
#           of Householder vector v_(i_j).
# @param[in]
# lda       int. lda >= m.\n
#           Specifies the leading dimension of matrices A_i.
# @param[out]
# ipiv      array of pointers to type. Each pointer points to an array on the GPU
#           of dimension min(m, n).\n
#           Contains the vectors ipiv_i of corresponding Householder scalars.
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = k < 0, the k-th argument is invalid.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#              Number of matrices in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgeqrfBatched(void * handle,const int m,const int n,float *const* A,const int lda,float *const* ipiv,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgeqrfBatched(void * handle,const int m,const int n,double *const* A,const int lda,double *const* ipiv,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgeqrfBatched(void * handle,const int m,const int n,hipblasComplex *const* A,const int lda,hipblasComplex *const* ipiv,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgeqrfBatched(void * handle,const int m,const int n,hipblasDoubleComplex *const* A,const int lda,hipblasDoubleComplex *const* ipiv,int * info,const int batchCount) nogil


# @{
# \brief SOLVER API
# 
# \details
# geqrfStridedBatched computes the QR factorization of a batch of
# general m-by-n matrices.
# 
# The factorization of matrix \f$A_i\f$ in the batch has the form
# 
# \f[
#     A_i = Q_i\left[\begin{array}{c}
#     R_i\\
#     0
#     \end{array}\right]
# \f]
# 
# where \f$R_i\f$ is upper triangular (upper trapezoidal if m < n), and \f$Q_i\f$ is
# a m-by-m orthogonal/unitary matrix represented as the product of Householder matrices
# 
# \f[
#     Q_i = H_{i_1}H_{i_2}\cdots H_{i_k}, \quad \text{with} \: k = \text{min}(m,n)
# \f]
# 
# Each Householder matrix \f$H_{i_j}\f$ is given by
# 
# \f[
#     H_{i_j} = I - \text{ipiv}_j[j] \cdot v_{i_j} v_{i_j}'
# \f]
# 
# where the first j-1 elements of Householder vector \f$v_{i_j}\f$ are zero, and \f$v_{i_j}[j] = 1\f$.
# 
# - Supported precisions in rocSOLVER : s,d,c,z
# - Supported precisions in cuBLAS    : No support
# 
# @param[in]
# handle    hipblasHandle_t.
# @param[in]
# m         int. m >= 0.\n
#           The number of rows of all the matrices A_i in the batch.
# @param[in]
# n         int. n >= 0.\n
#           The number of columns of all the matrices A_i in the batch.
# @param[inout]
# A         pointer to type. Array on the GPU (the size depends on the value of strideA).\n
#           On entry, the m-by-n matrices A_i to be factored.
#           On exit, the elements on and above the diagonal contain the
#           factor R_i. The elements below the diagonal are the last m - j elements
#           of Householder vector v_(i_j).
# @param[in]
# lda       int. lda >= m.\n
#           Specifies the leading dimension of matrices A_i.
# @param[in]
# strideA   hipblasStride.\n
#           Stride from the start of one matrix A_i to the next one A_(i+1).
#           There is no restriction for the value of strideA. Normal use case is strideA >= lda*n.
# @param[out]
# ipiv      pointer to type. Array on the GPU (the size depends on the value of strideP).\n
#           Contains the vectors ipiv_i of corresponding Householder scalars.
# @param[in]
# strideP   hipblasStride.\n
#           Stride from the start of one vector ipiv_i to the next one ipiv_(i+1).
#           There is no restriction for the value
#           of strideP. Normal use is strideP >= min(m,n).
# @param[out]
# info      pointer to a int on the host.\n
#           If info = 0, successful exit.
#           If info = k < 0, the k-th argument is invalid.
# @param[in]
# batchCount  int. batchCount >= 0.\n
#              Number of matrices in the batch.
#  ******************************************************************
cdef hipblasStatus_t hipblasSgeqrfStridedBatched(void * handle,const int m,const int n,float * A,const int lda,const long strideA,float * ipiv,const long strideP,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasDgeqrfStridedBatched(void * handle,const int m,const int n,double * A,const int lda,const long strideA,double * ipiv,const long strideP,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasCgeqrfStridedBatched(void * handle,const int m,const int n,hipblasComplex * A,const int lda,const long strideA,hipblasComplex * ipiv,const long strideP,int * info,const int batchCount) nogil



cdef hipblasStatus_t hipblasZgeqrfStridedBatched(void * handle,const int m,const int n,hipblasDoubleComplex * A,const int lda,const long strideA,hipblasDoubleComplex * ipiv,const long strideP,int * info,const int batchCount) nogil


# \brief BLAS EX API
# 
# \details
# gemmEx performs one of the matrix-matrix operations
# 
#     C = alpha*op( A )*op( B ) + beta*C,
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B, and C are matrices, with
# op( A ) an m by k matrix, op( B ) a k by n matrix and C is a m by n matrix.
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# Note for int8 users - For rocBLAS backend, please read rocblas_gemm_ex documentation on int8
# data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
# format for a given device as documented in rocBLAS.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A ).
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B ).
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# k         [int]
#           matrix dimension k.
# @param[in]
# alpha     [const void *]
#           device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
# @param[in]
# A         [void *]
#           device pointer storing matrix A.
# @param[in]
# aType    [hipblasDatatype_t]
#           specifies the datatype of matrix A.
# @param[in]
# lda       [int]
#           specifies the leading dimension of A.
# @param[in]
# B         [void *]
#           device pointer storing matrix B.
# @param[in]
# bType    [hipblasDatatype_t]
#           specifies the datatype of matrix B.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of B.
# @param[in]
# beta      [const void *]
#           device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
# @param[in]
# C         [void *]
#           device pointer storing matrix C.
# @param[in]
# cType    [hipblasDatatype_t]
#           specifies the datatype of matrix C.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of C.
# @param[in]
# computeType
#           [hipblasDatatype_t]
#           specifies the datatype of computation.
# @param[in]
# algo      [hipblasGemmAlgo_t]
#           enumerant specifying the algorithm type.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasGemmEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void * A,hipblasDatatype_t aType,int lda,const void * B,hipblasDatatype_t bType,int ldb,const void * beta,void * C,hipblasDatatype_t cType,int ldc,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil


# \brief BLAS EX API
# \details
# gemmBatchedEx performs one of the batched matrix-matrix operations
#     C_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batchCount.
# where op( X ) is one of
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# alpha and beta are scalars, and A, B, and C are batched pointers to matrices, with
# op( A ) an m by k by batchCount batched matrix,
# op( B ) a k by n by batchCount batched matrix and
# C a m by n by batchCount batched matrix.
# The batched matrices are an array of pointers to matrices.
# The number of pointers to matrices is batchCount.
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# Note for int8 users - For rocBLAS backend, please read rocblas_gemm_batched_ex documentation on int8
# data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
# format for a given device as documented in rocBLAS.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A ).
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B ).
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# k         [int]
#           matrix dimension k.
# @param[in]
# alpha     [const void *]
#           device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
# @param[in]
# A         [void *]
#           device pointer storing array of pointers to each matrix A_i.
# @param[in]
# aType    [hipblasDatatype_t]
#           specifies the datatype of each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# B         [void *]
#           device pointer storing array of pointers to each matrix B_i.
# @param[in]
# bType    [hipblasDatatype_t]
#           specifies the datatype of each matrix B_i.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# beta      [const void *]
#           device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
# @param[in]
# C         [void *]
#           device array of device pointers to each matrix C_i.
# @param[in]
# cType    [hipblasDatatype_t]
#           specifies the datatype of each matrix C_i.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# batchCount
#           [int]
#           number of gemm operations in the batch.
# @param[in]
# computeType
#           [hipblasDatatype_t]
#           specifies the datatype of computation.
# @param[in]
# algo      [hipblasGemmAlgo_t]
#           enumerant specifying the algorithm type.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasGemmBatchedEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void ** A,hipblasDatatype_t aType,int lda,const void ** B,hipblasDatatype_t bType,int ldb,const void * beta,void ** C,hipblasDatatype_t cType,int ldc,int batchCount,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil


# \brief BLAS EX API
# 
# \details
# gemmStridedBatchedEx performs one of the strided_batched matrix-matrix operations
# 
#     C_i = alpha*op(A_i)*op(B_i) + beta*C_i, for i = 1, ..., batchCount
# 
# where op( X ) is one of
# 
#     op( X ) = X      or
#     op( X ) = X**T   or
#     op( X ) = X**H,
# 
# alpha and beta are scalars, and A, B, and C are strided_batched matrices, with
# op( A ) an m by k by batchCount strided_batched matrix,
# op( B ) a k by n by batchCount strided_batched matrix and
# C a m by n by batchCount strided_batched matrix.
# 
# The strided_batched matrices are multiple matrices separated by a constant stride.
# The number of matrices is batchCount.
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# Note for int8 users - For rocBLAS backend, please read rocblas_gemm_strided_batched_ex documentation on int8
# data layout requirements. hipBLAS makes the assumption that the data layout is in the preferred
# format for a given device as documented in rocBLAS.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# transA    [hipblasOperation_t]
#           specifies the form of op( A ).
# @param[in]
# transB    [hipblasOperation_t]
#           specifies the form of op( B ).
# @param[in]
# m         [int]
#           matrix dimension m.
# @param[in]
# n         [int]
#           matrix dimension n.
# @param[in]
# k         [int]
#           matrix dimension k.
# @param[in]
# alpha     [const void *]
#           device pointer or host pointer specifying the scalar alpha. Same datatype as computeType.
# @param[in]
# A         [void *]
#           device pointer pointing to first matrix A_1.
# @param[in]
# aType    [hipblasDatatype_t]
#           specifies the datatype of each matrix A_i.
# @param[in]
# lda       [int]
#           specifies the leading dimension of each A_i.
# @param[in]
# strideA  [hipblasStride]
#           specifies stride from start of one A_i matrix to the next A_(i + 1).
# @param[in]
# B         [void *]
#           device pointer pointing to first matrix B_1.
# @param[in]
# bType    [hipblasDatatype_t]
#           specifies the datatype of each matrix B_i.
# @param[in]
# ldb       [int]
#           specifies the leading dimension of each B_i.
# @param[in]
# strideB  [hipblasStride]
#           specifies stride from start of one B_i matrix to the next B_(i + 1).
# @param[in]
# beta      [const void *]
#           device pointer or host pointer specifying the scalar beta. Same datatype as computeType.
# @param[in]
# C         [void *]
#           device pointer pointing to first matrix C_1.
# @param[in]
# cType    [hipblasDatatype_t]
#           specifies the datatype of each matrix C_i.
# @param[in]
# ldc       [int]
#           specifies the leading dimension of each C_i.
# @param[in]
# strideC  [hipblasStride]
#           specifies stride from start of one C_i matrix to the next C_(i + 1).
# @param[in]
# batchCount
#           [int]
#           number of gemm operations in the batch.
# @param[in]
# computeType
#           [hipblasDatatype_t]
#           specifies the datatype of computation.
# @param[in]
# algo      [hipblasGemmAlgo_t]
#           enumerant specifying the algorithm type.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasGemmStridedBatchedEx(void * handle,hipblasOperation_t transA,hipblasOperation_t transB,int m,int n,int k,const void * alpha,const void * A,hipblasDatatype_t aType,int lda,long strideA,const void * B,hipblasDatatype_t bType,int ldb,long strideB,const void * beta,void * C,hipblasDatatype_t cType,int ldc,long strideC,int batchCount,hipblasDatatype_t computeType,hipblasGemmAlgo_t algo) nogil


# BLAS EX API
# 
# \details
# trsmEx solves
# 
#     op(A)*X = alpha*B or X*op(A) = alpha*B,
# 
# where alpha is a scalar, X and B are m by n matrices,
# A is triangular matrix and op(A) is one of
# 
#     op( A ) = A   or   op( A ) = A^T   or   op( A ) = A^H.
# 
# The matrix X is overwritten on B.
# 
# This function gives the user the ability to reuse the invA matrix between runs.
# If invA == NULL, hipblasTrsmEx will automatically calculate invA on every run.
# 
# Setting up invA:
# The accepted invA matrix consists of the packed 128x128 inverses of the diagonal blocks of
# matrix A, followed by any smaller diagonal block that remains.
# To set up invA it is recommended that hipblasTrtriBatched be used with matrix A as the input.
# 
# Device memory of size 128 x k should be allocated for invA ahead of time, where k is m when
# HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in invA
# should be passed as invAsize.
# 
# To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of
# matrix A. Below are the restricted parameters:
#   - n = 128
#   - ldinvA = 128
#   - stride_invA = 128x128
#   - batchCount = k / 128,
# 
# Then any remaining block may be added:
#   - n = k % 128
#   - invA = invA + stride_invA * previousBatchCount
#   - ldinvA = 128
#   - batchCount = 1
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  A is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  A is a lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_ON_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     A is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  A is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of B. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of B. n >= 0.
# 
# @param[in]
# alpha   [void *]
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced, and B need not be set before
#         entry.
# 
# @param[in]
# A       [void *]
#         device pointer storing matrix A.
#         of dimension ( lda, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in, out]
# B       [void *]
#         device pointer storing matrix B.
#         B is of dimension ( ldb, n ).
#         Before entry, the leading m by n part of the array B must
#         contain the right-hand side matrix B, and on exit is
#         overwritten by the solution matrix X.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of B. ldb >= max( 1, m ).
# 
# @param[in]
# invA    [void *]
#         device pointer storing the inverse diagonal blocks of A.
#         invA is of dimension ( ld_invA, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT.
#         ld_invA must be equal to 128.
# 
# @param[in]
# invAsize [int]
#         invAsize specifies the number of elements of device memory in invA.
# 
# @param[in]
# computeType [hipblasDatatype_t]
#         specifies the datatype of computation
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasTrsmEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,void * B,int ldb,const void * invA,int invAsize,hipblasDatatype_t computeType) nogil


# BLAS EX API
# 
# \details
# trsmBatchedEx solves
# 
#     op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,
# 
# for i = 1, ..., batchCount; and where alpha is a scalar, X and B are arrays of m by n matrices,
# A is an array of triangular matrix and each op(A_i) is one of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# Each matrix X_i is overwritten on B_i.
# 
# This function gives the user the ability to reuse the invA matrix between runs.
# If invA == NULL, hipblasTrsmBatchedEx will automatically calculate each invA_i on every run.
# 
# Setting up invA:
# Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
# matrix A_i, followed by any smaller diagonal block that remains.
# To set up each invA_i it is recommended that hipblasTrtriBatched be used with matrix A_i as the input.
# invA is an array of pointers of batchCount length holding each invA_i.
# 
# Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
# HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in each invA_i
# should be passed as invAsize.
# 
# To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of each
# matrix A_i. Below are the restricted parameters:
#   - n = 128
#   - ldinvA = 128
#   - stride_invA = 128x128
#   - batchCount = k / 128,
# 
# Then any remaining block may be added:
#   - n = k % 128
#   - invA = invA + stride_invA * previousBatchCount
#   - ldinvA = 128
#   - batchCount = 1
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# 
# @param[in]
# alpha   [void *]
#         device pointer or host pointer alpha specifying the scalar alpha. When alpha is
#         &zero then A is not referenced, and B need not be set before
#         entry.
# 
# @param[in]
# A       [void *]
#         device array of device pointers storing each matrix A_i.
#         each A_i is of dimension ( lda, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of each A_i.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in, out]
# B       [void *]
#         device array of device pointers storing each matrix B_i.
#         each B_i is of dimension ( ldb, n ).
#         Before entry, the leading m by n part of the array B_i must
#         contain the right-hand side matrix B_i, and on exit is
#         overwritten by the solution matrix X_i
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# 
# @param[in]
# batchCount [int]
#         specifies how many batches.
# 
# @param[in]
# invA    [void *]
#         device array of device pointers storing the inverse diagonal blocks of each A_i.
#         each invA_i is of dimension ( ld_invA, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT.
#         ld_invA must be equal to 128.
# 
# @param[in]
# invAsize [int]
#         invAsize specifies the number of elements of device memory in each invA_i.
# 
# @param[in]
# computeType [hipblasDatatype_t]
#         specifies the datatype of computation
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasTrsmBatchedEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,void * B,int ldb,int batchCount,const void * invA,int invAsize,hipblasDatatype_t computeType) nogil


# BLAS EX API
# 
# \details
# trsmStridedBatchedEx solves
# 
#     op(A_i)*X_i = alpha*B_i or X_i*op(A_i) = alpha*B_i,
# 
# for i = 1, ..., batchCount; and where alpha is a scalar, X and B are strided batched m by n matrices,
# A is a strided batched triangular matrix and op(A_i) is one of
# 
#     op( A_i ) = A_i   or   op( A_i ) = A_i^T   or   op( A_i ) = A_i^H.
# 
# Each matrix X_i is overwritten on B_i.
# 
# This function gives the user the ability to reuse each invA_i matrix between runs.
# If invA == NULL, hipblasTrsmStridedBatchedEx will automatically calculate each invA_i on every run.
# 
# Setting up invA:
# Each accepted invA_i matrix consists of the packed 128x128 inverses of the diagonal blocks of
# matrix A_i, followed by any smaller diagonal block that remains.
# To set up invA_i it is recommended that hipblasTrtriBatched be used with matrix A_i as the input.
# invA is a contiguous piece of memory holding each invA_i.
# 
# Device memory of size 128 x k should be allocated for each invA_i ahead of time, where k is m when
# HIPBLAS_SIDE_LEFT and is n when HIPBLAS_SIDE_RIGHT. The actual number of elements in each invA_i
# should be passed as invAsize.
# 
# To begin, hipblasTrtriBatched must be called on the full 128x128 sized diagonal blocks of each
# matrix A_i. Below are the restricted parameters:
#   - n = 128
#   - ldinvA = 128
#   - stride_invA = 128x128
#   - batchCount = k / 128,
# 
# Then any remaining block may be added:
#   - n = k % 128
#   - invA = invA + stride_invA * previousBatchCount
#   - ldinvA = 128
#   - batchCount = 1
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# 
# @param[in]
# side    [hipblasSideMode_t]
#         HIPBLAS_SIDE_LEFT:       op(A)*X = alpha*B.
#         HIPBLAS_SIDE_RIGHT:      X*op(A) = alpha*B.
# 
# @param[in]
# uplo    [hipblasFillMode_t]
#         HIPBLAS_FILL_MODE_UPPER:  each A_i is an upper triangular matrix.
#         HIPBLAS_FILL_MODE_LOWER:  each A_i is a lower triangular matrix.
# 
# @param[in]
# transA  [hipblasOperation_t]
#         HIPBLAS_OP_N: op(A) = A.
#         HIPBLAS_OP_T: op(A) = A^T.
#         HIPBLAS_OP_C: op(A) = A^H.
# 
# @param[in]
# diag    [hipblasDiagType_t]
#         HIPBLAS_DIAG_UNIT:     each A_i is assumed to be unit triangular.
#         HIPBLAS_DIAG_NON_UNIT:  each A_i is not assumed to be unit triangular.
# 
# @param[in]
# m       [int]
#         m specifies the number of rows of each B_i. m >= 0.
# 
# @param[in]
# n       [int]
#         n specifies the number of columns of each B_i. n >= 0.
# 
# @param[in]
# alpha   [void *]
#         device pointer or host pointer specifying the scalar alpha. When alpha is
#         &zero then A is not referenced, and B need not be set before
#         entry.
# 
# @param[in]
# A       [void *]
#         device pointer storing matrix A.
#         of dimension ( lda, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT
#         only the upper/lower triangular part is accessed.
# 
# @param[in]
# lda     [int]
#         lda specifies the first dimension of A.
#         if side = HIPBLAS_SIDE_LEFT,  lda >= max( 1, m ),
#         if side = HIPBLAS_SIDE_RIGHT, lda >= max( 1, n ).
# 
# @param[in]
# strideA [hipblasStride]
#         The stride between each A matrix.
# 
# @param[in, out]
# B       [void *]
#         device pointer pointing to first matrix B_i.
#         each B_i is of dimension ( ldb, n ).
#         Before entry, the leading m by n part of each array B_i must
#         contain the right-hand side of matrix B_i, and on exit is
#         overwritten by the solution matrix X_i.
# 
# @param[in]
# ldb    [int]
#        ldb specifies the first dimension of each B_i. ldb >= max( 1, m ).
# 
# @param[in]
# strideB [hipblasStride]
#         The stride between each B_i matrix.
# 
# @param[in]
# batchCount [int]
#         specifies how many batches.
# 
# @param[in]
# invA    [void *]
#         device pointer storing the inverse diagonal blocks of each A_i.
#         invA points to the first invA_1.
#         each invA_i is of dimension ( ld_invA, k ), where k is m
#         when HIPBLAS_SIDE_LEFT and
#         is n when HIPBLAS_SIDE_RIGHT.
#         ld_invA must be equal to 128.
# 
# @param[in]
# invAsize [int]
#         invAsize specifies the number of elements of device memory in each invA_i.
# 
# @param[in]
# strideInvA [hipblasStride]
#         The stride between each invA matrix.
# 
# @param[in]
# computeType [hipblasDatatype_t]
#         specifies the datatype of computation
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasTrsmStridedBatchedEx(void * handle,hipblasSideMode_t side,hipblasFillMode_t uplo,hipblasOperation_t transA,hipblasDiagType_t diag,int m,int n,const void * alpha,void * A,int lda,long strideA,void * B,int ldb,long strideB,int batchCount,const void * invA,int invAsize,long strideInvA,hipblasDatatype_t computeType) nogil


# \brief BLAS EX API
# 
# \details
# axpyEx computes constant alpha multiplied by vector x, plus vector y
# 
#     y := alpha * x + y
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#           specifies the datatype of alpha.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[inout]
# y         device pointer storing vector y.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasAxpyEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# axpyBatchedEx computes constant alpha multiplied by vector x, plus vector y over
#                   a set of batched vectors.
# 
#     y := alpha * x + y
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#           specifies the datatype of alpha.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[inout]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasAxpyBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,int batchCount,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# axpyStridedBatchedEx computes constant alpha multiplied by vector x, plus vector y over
#                   a set of strided batched vectors.
# 
#     y := alpha * x + y
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# alpha     device pointer or host pointer to specify the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#           specifies the datatype of alpha.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) to the next one (x_i+1).
#           There are no restrictions placed on stridex, however the user should
#           take care to ensure that stridex is of appropriate size, for a typical
#           case this means stridex >= n * incx.
# @param[inout]
# y         device pointer to the first vector y_1.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey   [hipblasStride]
#           stride from the start of one vector (y_i) to the next one (y_i+1).
#           There are no restrictions placed on stridey, however the user should
#           take care to ensure that stridey is of appropriate size, for a typical
#           case this means stridey >= n * incy.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasAxpyStridedBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,const void * x,hipblasDatatype_t xType,int incx,long stridex,void * y,hipblasDatatype_t yType,int incy,long stridey,int batchCount,hipblasDatatype_t executionType) nogil


# @{
# \brief BLAS EX API
# 
# \details
# dotEx  performs the dot product of vectors x and y
# 
#     result = x * y;
# 
# dotcEx  performs the dot product of the conjugate of complex vector x and complex vector y
# 
#     result = conjugate (x) * y;
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x and y.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[in]
# y         device pointer storing vector y.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of vector y.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the dot product.
#           return is 0.0 if n <= 0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasDotEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil



cdef hipblasStatus_t hipblasDotcEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# @{
# \brief BLAS EX API
# 
# \details
# dotBatchedEx performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotcBatchedEx  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# y         device array of device pointers storing each vector y_i.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasDotBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil



cdef hipblasStatus_t hipblasDotcBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,const void * y,hipblasDatatype_t yType,int incy,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# @{
# \brief BLAS EX API
# 
# \details
# dotStridedBatchedEx  performs a batch of dot products of vectors x and y
# 
#     result_i = x_i * y_i;
# 
# dotc_strided_batched_ex  performs a batch of dot products of the conjugate of complex vector x and complex vector y
# 
#     result_i = conjugate (x_i) * y_i;
# 
# where (x_i, y_i) is the i-th instance of the batch.
# x_i and y_i are vectors, for i = 1, ..., batchCount
# 
#     - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in each x_i and y_i.
# @param[in]
# x         device pointer to the first vector (x_1) in the batch.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex    [hipblasStride]
#             stride from the start of one vector (x_i) and the next one (x_i+1)
# @param[in]
# y         device pointer to the first vector (y_1) in the batch.
# @param[in]
# yType [hipblasDatatype_t]
#       specifies the datatype of each vector y_i.
# @param[in]
# incy      [int]
#           specifies the increment for the elements of each y_i.
# @param[in]
# stridey    [hipblasStride]
#             stride from the start of one vector (y_i) and the next one (y_i+1)
# @param[in]
# batchCount [int]
#             number of instances in the batch
# @param[inout]
# result
#           device array or host array of batchCount size to store the dot products of each batch.
#           return 0.0 for each element if n <= 0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasDotStridedBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,long stridex,const void * y,hipblasDatatype_t yType,int incy,long stridey,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil



cdef hipblasStatus_t hipblasDotcStridedBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,long stridex,const void * y,hipblasDatatype_t yType,int incy,long stridey,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# \brief BLAS_EX API
# 
# \details
# nrm2Ex computes the euclidean norm of a real or complex vector
# 
#           result := sqrt( x'*x ) for real vectors
#           result := sqrt( x**H*x ) for complex vectors
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of the vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of y.
# @param[inout]
# result
#           device pointer or host pointer to store the nrm2 product.
#           return is 0.0 if n, incx<=0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
#  ******************************************************************
cdef hipblasStatus_t hipblasNrm2Ex(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# \brief BLAS_EX API
# 
# \details
# nrm2BatchedEx computes the euclidean norm over a batch of real or complex vectors
# 
#           result := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           result := sqrt( x_i**H*x_i ) for complex vectors x, for i = 1, ..., batchCount
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array of batchCount size for nrm2 results.
#           return is 0.0 for each element if n <= 0, incx<=0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasNrm2BatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# \brief BLAS_EX API
# 
# \details
# nrm2StridedBatchedEx computes the euclidean norm over a batch of real or complex vectors
# 
#           := sqrt( x_i'*x_i ) for real vectors x, for i = 1, ..., batchCount
#           := sqrt( x_i**H*x_i ) for complex vectors, for i = 1, ..., batchCount
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           number of elements in each x_i.
# @param[in]
# x         device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i. incx must be > 0.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) and the next one (x_i+1).
#           There are no restrictions placed on stride_x, however the user should
#           take care to ensure that stride_x is of appropriate size, for a typical
#           case this means stride_x >= n * incx.
# @param[in]
# batchCount [int]
#           number of instances in the batch
# @param[out]
# result
#           device pointer or host pointer to array for storing contiguous batchCount results.
#           return is 0.0 for each element if n <= 0, incx<=0.
# @param[in]
# resultType [hipblasDatatype_t]
#             specifies the datatype of the result.
# @param[in]
# executionType [hipblasDatatype_t]
#               specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasNrm2StridedBatchedEx(void * handle,int n,const void * x,hipblasDatatype_t xType,int incx,long stridex,int batchCount,void * result,hipblasDatatype_t resultType,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# rotEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to vectors x and y.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# In the case where cs_type is real:
#     x := c * x + s * y
#         y := c * y - s * x
# 
# In the case where cs_type is complex, the imaginary part of c is ignored:
#     x := real(c) * x + s * y
#         y := real(c) * y - conj(s) * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in the x and y vectors.
# @param[inout]
# x       device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx    [int]
#         specifies the increment between elements of x.
# @param[inout]
# y       device pointer storing vector y.
# @param[in]
# yType [hipblasDatatype_t]
#        specifies the datatype of vector y.
# @param[in]
# incy    [int]
#         specifies the increment between elements of y.
# @param[in]
# c       device pointer or host pointer storing scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer storing scalar sine component of the rotation matrix.
# @param[in]
# csType [hipblasDatatype_t]
#         specifies the datatype of c and s.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasRotEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,const void * c,const void * s,hipblasDatatype_t csType,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# rotBatchedEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# In the case where cs_type is real:
#         x := c * x + s * y
#         y := c * y - s * x
# 
#     In the case where cs_type is complex, the imaginary part of c is ignored:
#         x := real(c) * x + s * y
#         y := real(c) * y - conj(s) * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device array of deivce pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[inout]
# y       device array of device pointers storing each vector y_i.
# @param[in]
# yType [hipblasDatatype_t]
#        specifies the datatype of each vector y_i.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# csType [hipblasDatatype_t]
#         specifies the datatype of c and s.
# @param[in]
# batchCount [int]
#             the number of x and y arrays, i.e. the number of batches.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasRotBatchedEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,void * y,hipblasDatatype_t yType,int incy,const void * c,const void * s,hipblasDatatype_t csType,int batchCount,hipblasDatatype_t executionType) nogil


# \brief BLAS Level 1 API
# 
# \details
# rotStridedBatchedEx applies the Givens rotation matrix defined by c=cos(alpha) and s=sin(alpha) to strided batched vectors x_i and y_i, for i = 1, ..., batchCount.
#     Scalars c and s may be stored in either host or device memory, location is specified by calling hipblasSetPointerMode.
# 
# In the case where cs_type is real:
#         x := c * x + s * y
#         y := c * y - s * x
# 
#     In the case where cs_type is complex, the imaginary part of c is ignored:
#         x := real(c) * x + s * y
#         y := real(c) * y - conj(s) * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle  [hipblasHandle_t]
#         handle to the hipblas library context queue.
# @param[in]
# n       [int]
#         number of elements in each x_i and y_i vectors.
# @param[inout]
# x       device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx    [int]
#         specifies the increment between elements of each x_i.
# @param[in]
# stridex [hipblasStride]
#          specifies the increment from the beginning of x_i to the beginning of x_(i+1)
# @param[inout]
# y       device pointer to the first vector y_1.
# @param[in]
# yType [hipblasDatatype_t]
#        specifies the datatype of each vector y_i.
# @param[in]
# incy    [int]
#         specifies the increment between elements of each y_i.
# @param[in]
# stridey [hipblasStride]
#          specifies the increment from the beginning of y_i to the beginning of y_(i+1)
# @param[in]
# c       device pointer or host pointer to scalar cosine component of the rotation matrix.
# @param[in]
# s       device pointer or host pointer to scalar sine component of the rotation matrix.
# @param[in]
# csType [hipblasDatatype_t]
#         specifies the datatype of c and s.
# @param[in]
# batchCount [int]
#         the number of x and y arrays, i.e. the number of batches.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasRotStridedBatchedEx(void * handle,int n,void * x,hipblasDatatype_t xType,int incx,long stridex,void * y,hipblasDatatype_t yType,int incy,long stridey,const void * c,const void * s,hipblasDatatype_t csType,int batchCount,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# scalEx  scales each element of vector x with scalar alpha.
# 
#     x := alpha * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#            specifies the datatype of alpha.
# @param[inout]
# x         device pointer storing vector x.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of vector x.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of x.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasScalEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# scalBatchedEx  scales each element of each vector x_i with scalar alpha.
# 
#     x_i := alpha * x_i
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#            specifies the datatype of alpha.
# @param[inout]
# x         device array of device pointers storing each vector x_i.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasScalBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,int batchCount,hipblasDatatype_t executionType) nogil


# \brief BLAS EX API
# 
# \details
# scalStridedBatchedEx  scales each element of vector x with scalar alpha over a set
#                          of strided batched vectors.
# 
#     x := alpha * x
# 
# - Supported types are determined by the backend. See rocBLAS/cuBLAS documentation.
# 
# @param[in]
# handle    [hipblasHandle_t]
#           handle to the hipblas library context queue.
# @param[in]
# n         [int]
#           the number of elements in x.
# @param[in]
# alpha     device pointer or host pointer for the scalar alpha.
# @param[in]
# alphaType [hipblasDatatype_t]
#            specifies the datatype of alpha.
# @param[inout]
# x         device pointer to the first vector x_1.
# @param[in]
# xType [hipblasDatatype_t]
#        specifies the datatype of each vector x_i.
# @param[in]
# incx      [int]
#           specifies the increment for the elements of each x_i.
# @param[in]
# stridex   [hipblasStride]
#           stride from the start of one vector (x_i) to the next one (x_i+1).
#           There are no restrictions placed on stridex, however the user should
#           take care to ensure that stridex is of appropriate size, for a typical
#           case this means stridex >= n * incx.
# @param[in]
# batchCount [int]
#             number of instances in the batch.
# @param[in]
# executionType [hipblasDatatype_t]
#                specifies the datatype of computation.
# 
#  ******************************************************************
cdef hipblasStatus_t hipblasScalStridedBatchedEx(void * handle,int n,const void * alpha,hipblasDatatype_t alphaType,void * x,hipblasDatatype_t xType,int incx,long stridex,int batchCount,hipblasDatatype_t executionType) nogil


#    HIPBLAS Auxiliary API
# 
#    \details
#    hipblasStatusToString
# 
#    Returns string representing hipblasStatus_t value
# 
#    @param[in]
#    status  [hipblasStatus_t]
#            hipBLAS status to convert to string
# /
cdef const char * hipblasStatusToString(hipblasStatus_t status) nogil
