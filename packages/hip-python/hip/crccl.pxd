# AMD_COPYRIGHT
from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .chip cimport hipStream_t
cdef extern from "rccl/rccl.h":

    cdef int NCCL_MAJOR

    cdef int NCCL_MINOR

    cdef int NCCL_PATCH

    cdef char* NCCL_SUFFIX

    cdef int NCCL_VERSION_CODE

    cdef int RCCL_BFLOAT16

    cdef int RCCL_GATHER_SCATTER

    cdef int RCCL_ALLTOALLV

    cdef int RCCL_MULTIRANKPERGPU

    cdef int NCCL_UNIQUE_ID_BYTES

    cdef struct ncclComm:
        pass

    ctypedef ncclComm * ncclComm_t

    ctypedef struct ncclUniqueId:
        char[128] internal

    ctypedef enum ncclResult_t:
        ncclSuccess
        ncclUnhandledCudaError
        ncclSystemError
        ncclInternalError
        ncclInvalidArgument
        ncclInvalidUsage
        ncclRemoteError
        ncclNumResults

#  @brief Return the NCCL_VERSION_CODE of the NCCL library in the supplied integer.
# 
# @details This integer is coded with the MAJOR, MINOR and PATCH level of the
# NCCL library
cdef ncclResult_t ncclGetVersion(int * version) nogil


# @cond include_hidden
cdef ncclResult_t pncclGetVersion(int * version) nogil


#    @brief Generates an ID for ncclCommInitRank
# 
#    @details
#    Generates an ID to be used in ncclCommInitRank. ncclGetUniqueId should be
#    called once and the Id should be distributed to all ranks in the
#    communicator before calling ncclCommInitRank.
# 
#    @param[in]
#    uniqueId     ncclUniqueId*
#                 pointer to uniqueId
# 
# /
cdef ncclResult_t ncclGetUniqueId(ncclUniqueId * uniqueId) nogil


# @cond include_hidden
cdef ncclResult_t pncclGetUniqueId(ncclUniqueId * uniqueId) nogil


# @brief Creates a new communicator (multi thread/process version).
# 
# @details
# rank must be between 0 and nranks-1 and unique within a communicator clique.
# Each rank is associated to a CUDA device, which has to be set before calling
# ncclCommInitRank.
# ncclCommInitRank implicitly syncronizes with other ranks, so it must be
# called by different threads/processes or use ncclGroupStart/ncclGroupEnd.
# 
# @param[in]
# comm        ncclComm_t*
#             communicator struct pointer
cdef ncclResult_t ncclCommInitRank(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommInitRank(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank) nogil


# @brief Creates a new communicator (multi thread/process version) allowing multiple ranks per device.
# 
# @details
# rank must be between 0 and nranks-1 and unique within a communicator clique.
# Each rank is associated to a HIP device, which has to be set before calling
# ncclCommInitRankMulti.
# Since this version of the function allows multiple ranks to utilize the same
# HIP device, a unique virtualId per device has to be provided by each calling
# rank.
# ncclCommInitRankMulti implicitly syncronizes with other ranks, so it must be
# called by different threads/processes or use ncclGroupStart/ncclGroupEnd.
# 
# @param[in]
# comm        ncclComm_t*
#             communicator struct pointer
cdef ncclResult_t ncclCommInitRankMulti(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank,int virtualId) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommInitRankMulti(ncclComm_t* comm,int nranks,ncclUniqueId commId,int rank,int virtualId) nogil


#  @brief Creates a clique of communicators (single process version).
# 
# @details This is a convenience function to create a single-process communicator clique.
# Returns an array of ndev newly initialized communicators in comm.
# comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
# If devlist is NULL, the first ndev HIP devices are used.
# Order of devlist defines user-order of processors within the communicator.
cdef ncclResult_t ncclCommInitAll(ncclComm_t* comm,int ndev,const int * devlist) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommInitAll(ncclComm_t* comm,int ndev,const int * devlist) nogil


# @brief Frees resources associated with communicator object, but waits for any operations that might still be running on the device */
cdef ncclResult_t ncclCommDestroy(ncclComm_t comm) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommDestroy(ncclComm_t comm) nogil


# @brief Frees resources associated with communicator object and aborts any operations that might still be running on the device. */
cdef ncclResult_t ncclCommAbort(ncclComm_t comm) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommAbort(ncclComm_t comm) nogil


# @brief Returns a string for each error code. */
cdef const char * ncclGetErrorString(ncclResult_t result) nogil


# @cond include_hidden
cdef const char * pncclGetErrorString(ncclResult_t result) nogil


#  @brief Returns a human-readable message of the last error that occurred.
# comm is currently unused and can be set to NULL
cdef const char * ncclGetLastError(ncclComm_t comm) nogil


# @cond include_hidden
cdef const char * pncclGetError(ncclComm_t comm) nogil


# @endcond
cdef ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,ncclResult_t * asyncError) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommGetAsyncError(ncclComm_t comm,ncclResult_t * asyncError) nogil


# @brief Gets the number of ranks in the communicator clique. */
cdef ncclResult_t ncclCommCount(ncclComm_t comm,int * count) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommCount(ncclComm_t comm,int * count) nogil


# @brief Returns the rocm device number associated with the communicator. */
cdef ncclResult_t ncclCommCuDevice(ncclComm_t comm,int * device) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommCuDevice(ncclComm_t comm,int * device) nogil


# @brief Returns the user-ordered "rank" associated with the communicator. */
cdef ncclResult_t ncclCommUserRank(ncclComm_t comm,int * rank) nogil


# @cond include_hidden
cdef ncclResult_t pncclCommUserRank(ncclComm_t comm,int * rank) nogil


cdef extern from "rccl/rccl.h":

    ctypedef enum ncclRedOp_dummy_t:
        ncclNumOps_dummy

    ctypedef enum ncclRedOp_t:
        ncclSum
        ncclProd
        ncclMax
        ncclMin
        ncclAvg
        ncclNumOps
        ncclMaxRedOp

    ctypedef enum ncclDataType_t:
        ncclInt8
        ncclChar
        ncclUint8
        ncclInt32
        ncclInt
        ncclUint32
        ncclInt64
        ncclUint64
        ncclFloat16
        ncclHalf
        ncclFloat32
        ncclFloat
        ncclFloat64
        ncclDouble
        ncclBfloat16
        ncclNumTypes

    ctypedef enum ncclScalarResidence_t:
        ncclScalarDevice
        ncclScalarHostImmediate


cdef ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t * op,void * scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm) nogil



cdef ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t * op,void * scalar,ncclDataType_t datatype,ncclScalarResidence_t residence,ncclComm_t comm) nogil



cdef ncclResult_t ncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm) nogil



cdef ncclResult_t pncclRedOpDestroy(ncclRedOp_t op,ncclComm_t comm) nogil


# 
# @brief Reduce
# 
# @details Reduces data arrays of length count in sendbuff into recvbuff using op
# operation.
# recvbuff may be NULL on all calls except for root device.
# root is the rank (not the CUDA device) where data will reside after the
# operation is complete.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,int root,ncclComm_t comm,hipStream_t stream) nogil


#  @brief (deprecated) Broadcast (in-place)
# 
# @details Copies count values from root to all other devices.
# root is the rank (not the CUDA device) where data resides before the
# operation is started.
# 
# This operation is implicitely in place.
cdef ncclResult_t ncclBcast(void * buff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclBcast(void * buff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


#  @brief Broadcast
# 
# @details Copies count values from root to all other devices.
# root is the rank (not the HIP device) where data resides before the
# operation is started.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclBroadcast(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclBroadcast(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


#  @brief All-Reduce
# 
# @details Reduces data arrays of length count in sendbuff using op operation, and
# leaves identical copies of result on each recvbuff.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclAllReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclAllReduce(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil


# 
# @brief Reduce-Scatter
# 
# @details Reduces data in sendbuff using op operation and leaves reduced result
# scattered over the devices so that recvbuff on rank i will contain the i-th
# block of the result.
# Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
# should have a size of at least nranks*recvcount elements.
# 
# In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
cdef ncclResult_t ncclReduceScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclReduceScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,ncclRedOp_t op,ncclComm_t comm,hipStream_t stream) nogil


#  @brief All-Gather
# 
# @details Each device gathers sendcount values from other GPUs into recvbuff,
# receiving data from rank i at offset i*sendcount.
# Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
# should have a size of at least nranks*sendcount elements.
# 
# In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
cdef ncclResult_t ncclAllGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclAllGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil


#  @brief Send
# 
# @details Send data from sendbuff to rank peer.
# Rank peer needs to call ncclRecv with the same datatype and the same count from this
# rank.
# 
# This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
# need to progress concurrently to complete, they must be fused within a ncclGroupStart/
# ncclGroupEnd section.
cdef ncclResult_t ncclSend(const void * sendbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclSend(const void * sendbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil


#  @brief Receive
# 
# @details Receive data from rank peer into recvbuff.
# Rank peer needs to call ncclSend with the same datatype and the same count to this
# rank.
# 
# This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
# need to progress concurrently to complete, they must be fused within a ncclGroupStart/
# ncclGroupEnd section.
cdef ncclResult_t ncclRecv(void * recvbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclRecv(void * recvbuff,unsigned long count,ncclDataType_t datatype,int peer,ncclComm_t comm,hipStream_t stream) nogil


#  @brief Gather
# 
# @details Root device gathers sendcount values from other GPUs into recvbuff,
# receiving data from rank i at offset i*sendcount.
# 
# Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
# should have a size of at least nranks*sendcount elements.
# 
# In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
cdef ncclResult_t ncclGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclGather(const void * sendbuff,void * recvbuff,unsigned long sendcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


#  @brief Scatter
# 
# @details Scattered over the devices so that recvbuff on rank i will contain the i-th
# block of the data on root.
# 
# Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
# should have a size of at least nranks*recvcount elements.
# 
# In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
cdef ncclResult_t ncclScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclScatter(const void * sendbuff,void * recvbuff,unsigned long recvcount,ncclDataType_t datatype,int root,ncclComm_t comm,hipStream_t stream) nogil


#  @brief All-To-All
# 
# @details Device (i) send (j)th block of data to device (j) and be placed as (i)th
# block. Each block for sending/receiving has count elements, which means
# that recvbuff and sendbuff should have a size of nranks*count elements.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclAllToAll(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclAllToAll(const void * sendbuff,void * recvbuff,unsigned long count,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil


#  @brief All-To-Allv
# 
# @details Device (i) sends sendcounts[j] of data from offset sdispls[j]
# to device (j). In the same time, device (i) receives recvcounts[j] of data
# from device (j) to be placed at rdispls[j].
# 
# sendcounts, sdispls, recvcounts and rdispls are all measured in the units
# of datatype, not bytes.
# 
# In-place operation will happen if sendbuff == recvbuff.
cdef ncclResult_t ncclAllToAllv(const void * sendbuff,const unsigned long* sendcounts,const unsigned long* sdispls,void * recvbuff,const unsigned long* recvcounts,const unsigned long* rdispls,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil


# @cond include_hidden
cdef ncclResult_t pncclAllToAllv(const void * sendbuff,const unsigned long* sendcounts,const unsigned long* sdispls,void * recvbuff,const unsigned long* recvcounts,const unsigned long* rdispls,ncclDataType_t datatype,ncclComm_t comm,hipStream_t stream) nogil


#  @brief Group Start
# 
# Start a group call. All calls to NCCL until ncclGroupEnd will be fused into
# a single NCCL operation. Nothing will be started on the CUDA stream until
# ncclGroupEnd.
cdef ncclResult_t ncclGroupStart() nogil


# @cond include_hidden
cdef ncclResult_t pncclGroupStart() nogil


#  @brief Group End
# 
# End a group call. Start a fused NCCL operation consisting of all calls since
# ncclGroupStart. Operations on the CUDA stream depending on the NCCL operations
# need to be called after ncclGroupEnd.
cdef ncclResult_t ncclGroupEnd() nogil


# @cond include_hidden
cdef ncclResult_t pncclGroupEnd() nogil
