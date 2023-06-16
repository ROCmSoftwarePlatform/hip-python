# AMD_COPYRIGHT
cimport hip._util.posixloader as loader
cdef void* _lib_handle = NULL

cdef void __init() nogil:
    global _lib_handle
    if _lib_handle == NULL:
        with gil:
            _lib_handle = loader.open_library("libhipsparse.so")

cdef void __init_symbol(void** result, const char* name) nogil:
    global _lib_handle
    if _lib_handle == NULL:
        __init()
    if result[0] == NULL:
        with gil:
            result[0] = loader.load_symbol(_lib_handle, name) 


cdef void* _hipsparseCreate__funptr = NULL
# \ingroup aux_module
# \brief Create a hipsparse handle
# 
# \details
# \p hipsparseCreate creates the hipSPARSE library context. It must be
# initialized before any other hipSPARSE API function is invoked and must be passed to
# all subsequent library function calls. The handle should be destroyed at the end
# using hipsparseDestroy().
cdef hipsparseStatus_t hipsparseCreate(void ** handle) nogil:
    global _hipsparseCreate__funptr
    __init_symbol(&_hipsparseCreate__funptr,"hipsparseCreate")
    return (<hipsparseStatus_t (*)(void **) nogil> _hipsparseCreate__funptr)(handle)


cdef void* _hipsparseDestroy__funptr = NULL
# \ingroup aux_module
# \brief Destroy a hipsparse handle
# 
# \details
# \p hipsparseDestroy destroys the hipSPARSE library context and releases all
# resources used by the hipSPARSE library.
cdef hipsparseStatus_t hipsparseDestroy(void * handle) nogil:
    global _hipsparseDestroy__funptr
    __init_symbol(&_hipsparseDestroy__funptr,"hipsparseDestroy")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroy__funptr)(handle)


cdef void* _hipsparseGetVersion__funptr = NULL
# \ingroup aux_module
# \brief Get hipSPARSE version
# 
# \details
# \p hipsparseGetVersion gets the hipSPARSE library version number.
# - patch = version % 100
# - minor = version / 100 % 1000
# - major = version / 100000
cdef hipsparseStatus_t hipsparseGetVersion(void * handle,int * version) nogil:
    global _hipsparseGetVersion__funptr
    __init_symbol(&_hipsparseGetVersion__funptr,"hipsparseGetVersion")
    return (<hipsparseStatus_t (*)(void *,int *) nogil> _hipsparseGetVersion__funptr)(handle,version)


cdef void* _hipsparseGetGitRevision__funptr = NULL
# \ingroup aux_module
# \brief Get hipSPARSE git revision
# 
# \details
# \p hipsparseGetGitRevision gets the hipSPARSE library git commit revision (SHA-1).
cdef hipsparseStatus_t hipsparseGetGitRevision(void * handle,char * rev) nogil:
    global _hipsparseGetGitRevision__funptr
    __init_symbol(&_hipsparseGetGitRevision__funptr,"hipsparseGetGitRevision")
    return (<hipsparseStatus_t (*)(void *,char *) nogil> _hipsparseGetGitRevision__funptr)(handle,rev)


cdef void* _hipsparseSetStream__funptr = NULL
# \ingroup aux_module
# \brief Specify user defined HIP stream
# 
# \details
# \p hipsparseSetStream specifies the stream to be used by the hipSPARSE library
# context and all subsequent function calls.
cdef hipsparseStatus_t hipsparseSetStream(void * handle,hipStream_t streamId) nogil:
    global _hipsparseSetStream__funptr
    __init_symbol(&_hipsparseSetStream__funptr,"hipsparseSetStream")
    return (<hipsparseStatus_t (*)(void *,hipStream_t) nogil> _hipsparseSetStream__funptr)(handle,streamId)


cdef void* _hipsparseGetStream__funptr = NULL
# \ingroup aux_module
# \brief Get current stream from library context
# 
# \details
# \p hipsparseGetStream gets the hipSPARSE library context stream which is currently
# used for all subsequent function calls.
cdef hipsparseStatus_t hipsparseGetStream(void * handle,hipStream_t* streamId) nogil:
    global _hipsparseGetStream__funptr
    __init_symbol(&_hipsparseGetStream__funptr,"hipsparseGetStream")
    return (<hipsparseStatus_t (*)(void *,hipStream_t*) nogil> _hipsparseGetStream__funptr)(handle,streamId)


cdef void* _hipsparseSetPointerMode__funptr = NULL
# \ingroup aux_module
# \brief Specify pointer mode
# 
# \details
# \p hipsparseSetPointerMode specifies the pointer mode to be used by the hipSPARSE
# library context and all subsequent function calls. By default, all values are passed
# by reference on the host. Valid pointer modes are \ref HIPSPARSE_POINTER_MODE_HOST
# or \p HIPSPARSE_POINTER_MODE_DEVICE.
cdef hipsparseStatus_t hipsparseSetPointerMode(void * handle,hipsparsePointerMode_t mode) nogil:
    global _hipsparseSetPointerMode__funptr
    __init_symbol(&_hipsparseSetPointerMode__funptr,"hipsparseSetPointerMode")
    return (<hipsparseStatus_t (*)(void *,hipsparsePointerMode_t) nogil> _hipsparseSetPointerMode__funptr)(handle,mode)


cdef void* _hipsparseGetPointerMode__funptr = NULL
# \ingroup aux_module
# \brief Get current pointer mode from library context
# 
# \details
# \p hipsparseGetPointerMode gets the hipSPARSE library context pointer mode which
# is currently used for all subsequent function calls.
cdef hipsparseStatus_t hipsparseGetPointerMode(void * handle,hipsparsePointerMode_t * mode) nogil:
    global _hipsparseGetPointerMode__funptr
    __init_symbol(&_hipsparseGetPointerMode__funptr,"hipsparseGetPointerMode")
    return (<hipsparseStatus_t (*)(void *,hipsparsePointerMode_t *) nogil> _hipsparseGetPointerMode__funptr)(handle,mode)


cdef void* _hipsparseCreateMatDescr__funptr = NULL
# \ingroup aux_module
# \brief Create a matrix descriptor
# \details
# \p hipsparseCreateMatDescr creates a matrix descriptor. It initializes
# \ref hipsparseMatrixType_t to \ref HIPSPARSE_MATRIX_TYPE_GENERAL and
# \ref hipsparseIndexBase_t to \ref HIPSPARSE_INDEX_BASE_ZERO. It should be destroyed
# at the end using hipsparseDestroyMatDescr().
cdef hipsparseStatus_t hipsparseCreateMatDescr(void ** descrA) nogil:
    global _hipsparseCreateMatDescr__funptr
    __init_symbol(&_hipsparseCreateMatDescr__funptr,"hipsparseCreateMatDescr")
    return (<hipsparseStatus_t (*)(void **) nogil> _hipsparseCreateMatDescr__funptr)(descrA)


cdef void* _hipsparseDestroyMatDescr__funptr = NULL
# \ingroup aux_module
# \brief Destroy a matrix descriptor
# 
# \details
# \p hipsparseDestroyMatDescr destroys a matrix descriptor and releases all
# resources used by the descriptor.
cdef hipsparseStatus_t hipsparseDestroyMatDescr(void * descrA) nogil:
    global _hipsparseDestroyMatDescr__funptr
    __init_symbol(&_hipsparseDestroyMatDescr__funptr,"hipsparseDestroyMatDescr")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroyMatDescr__funptr)(descrA)


cdef void* _hipsparseCopyMatDescr__funptr = NULL
# \ingroup aux_module
# \brief Copy a matrix descriptor
# \details
# \p hipsparseCopyMatDescr copies a matrix descriptor. Both, source and destination
# matrix descriptors must be initialized prior to calling \p hipsparseCopyMatDescr.
cdef hipsparseStatus_t hipsparseCopyMatDescr(void * dest,void *const src) nogil:
    global _hipsparseCopyMatDescr__funptr
    __init_symbol(&_hipsparseCopyMatDescr__funptr,"hipsparseCopyMatDescr")
    return (<hipsparseStatus_t (*)(void *,void *const) nogil> _hipsparseCopyMatDescr__funptr)(dest,src)


cdef void* _hipsparseSetMatType__funptr = NULL
# \ingroup aux_module
# \brief Specify the matrix type of a matrix descriptor
# 
# \details
# \p hipsparseSetMatType sets the matrix type of a matrix descriptor. Valid
# matrix types are \ref HIPSPARSE_MATRIX_TYPE_GENERAL,
# \ref HIPSPARSE_MATRIX_TYPE_SYMMETRIC, \ref HIPSPARSE_MATRIX_TYPE_HERMITIAN or
# \ref HIPSPARSE_MATRIX_TYPE_TRIANGULAR.
cdef hipsparseStatus_t hipsparseSetMatType(void * descrA,hipsparseMatrixType_t type) nogil:
    global _hipsparseSetMatType__funptr
    __init_symbol(&_hipsparseSetMatType__funptr,"hipsparseSetMatType")
    return (<hipsparseStatus_t (*)(void *,hipsparseMatrixType_t) nogil> _hipsparseSetMatType__funptr)(descrA,type)


cdef void* _hipsparseGetMatType__funptr = NULL
# \ingroup aux_module
# \brief Get the matrix type of a matrix descriptor
# 
# \details
# \p hipsparseGetMatType returns the matrix type of a matrix descriptor.
cdef hipsparseMatrixType_t hipsparseGetMatType(void *const descrA) nogil:
    global _hipsparseGetMatType__funptr
    __init_symbol(&_hipsparseGetMatType__funptr,"hipsparseGetMatType")
    return (<hipsparseMatrixType_t (*)(void *const) nogil> _hipsparseGetMatType__funptr)(descrA)


cdef void* _hipsparseSetMatFillMode__funptr = NULL
# \ingroup aux_module
# \brief Specify the matrix fill mode of a matrix descriptor
# 
# \details
# \p hipsparseSetMatFillMode sets the matrix fill mode of a matrix descriptor.
# Valid fill modes are \ref HIPSPARSE_FILL_MODE_LOWER or
# \ref HIPSPARSE_FILL_MODE_UPPER.
cdef hipsparseStatus_t hipsparseSetMatFillMode(void * descrA,hipsparseFillMode_t fillMode) nogil:
    global _hipsparseSetMatFillMode__funptr
    __init_symbol(&_hipsparseSetMatFillMode__funptr,"hipsparseSetMatFillMode")
    return (<hipsparseStatus_t (*)(void *,hipsparseFillMode_t) nogil> _hipsparseSetMatFillMode__funptr)(descrA,fillMode)


cdef void* _hipsparseGetMatFillMode__funptr = NULL
# \ingroup aux_module
# \brief Get the matrix fill mode of a matrix descriptor
# 
# \details
# \p hipsparseGetMatFillMode returns the matrix fill mode of a matrix descriptor.
cdef hipsparseFillMode_t hipsparseGetMatFillMode(void *const descrA) nogil:
    global _hipsparseGetMatFillMode__funptr
    __init_symbol(&_hipsparseGetMatFillMode__funptr,"hipsparseGetMatFillMode")
    return (<hipsparseFillMode_t (*)(void *const) nogil> _hipsparseGetMatFillMode__funptr)(descrA)


cdef void* _hipsparseSetMatDiagType__funptr = NULL
# \ingroup aux_module
# \brief Specify the matrix diagonal type of a matrix descriptor
# 
# \details
# \p hipsparseSetMatDiagType sets the matrix diagonal type of a matrix
# descriptor. Valid diagonal types are \ref HIPSPARSE_DIAG_TYPE_UNIT or
# \ref HIPSPARSE_DIAG_TYPE_NON_UNIT.
cdef hipsparseStatus_t hipsparseSetMatDiagType(void * descrA,hipsparseDiagType_t diagType) nogil:
    global _hipsparseSetMatDiagType__funptr
    __init_symbol(&_hipsparseSetMatDiagType__funptr,"hipsparseSetMatDiagType")
    return (<hipsparseStatus_t (*)(void *,hipsparseDiagType_t) nogil> _hipsparseSetMatDiagType__funptr)(descrA,diagType)


cdef void* _hipsparseGetMatDiagType__funptr = NULL
# \ingroup aux_module
# \brief Get the matrix diagonal type of a matrix descriptor
# 
# \details
# \p hipsparseGetMatDiagType returns the matrix diagonal type of a matrix
# descriptor.
cdef hipsparseDiagType_t hipsparseGetMatDiagType(void *const descrA) nogil:
    global _hipsparseGetMatDiagType__funptr
    __init_symbol(&_hipsparseGetMatDiagType__funptr,"hipsparseGetMatDiagType")
    return (<hipsparseDiagType_t (*)(void *const) nogil> _hipsparseGetMatDiagType__funptr)(descrA)


cdef void* _hipsparseSetMatIndexBase__funptr = NULL
# \ingroup aux_module
# \brief Specify the index base of a matrix descriptor
# 
# \details
# \p hipsparseSetMatIndexBase sets the index base of a matrix descriptor. Valid
# options are \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
cdef hipsparseStatus_t hipsparseSetMatIndexBase(void * descrA,hipsparseIndexBase_t base) nogil:
    global _hipsparseSetMatIndexBase__funptr
    __init_symbol(&_hipsparseSetMatIndexBase__funptr,"hipsparseSetMatIndexBase")
    return (<hipsparseStatus_t (*)(void *,hipsparseIndexBase_t) nogil> _hipsparseSetMatIndexBase__funptr)(descrA,base)


cdef void* _hipsparseGetMatIndexBase__funptr = NULL
# \ingroup aux_module
# \brief Get the index base of a matrix descriptor
# 
# \details
# \p hipsparseGetMatIndexBase returns the index base of a matrix descriptor.
cdef hipsparseIndexBase_t hipsparseGetMatIndexBase(void *const descrA) nogil:
    global _hipsparseGetMatIndexBase__funptr
    __init_symbol(&_hipsparseGetMatIndexBase__funptr,"hipsparseGetMatIndexBase")
    return (<hipsparseIndexBase_t (*)(void *const) nogil> _hipsparseGetMatIndexBase__funptr)(descrA)


cdef void* _hipsparseCreateHybMat__funptr = NULL
# \ingroup aux_module
# \brief Create a \p HYB matrix structure
# 
# \details
# \p hipsparseCreateHybMat creates a structure that holds the matrix in \p HYB
# storage format. It should be destroyed at the end using hipsparseDestroyHybMat().
cdef hipsparseStatus_t hipsparseCreateHybMat(void ** hybA) nogil:
    global _hipsparseCreateHybMat__funptr
    __init_symbol(&_hipsparseCreateHybMat__funptr,"hipsparseCreateHybMat")
    return (<hipsparseStatus_t (*)(void **) nogil> _hipsparseCreateHybMat__funptr)(hybA)


cdef void* _hipsparseDestroyHybMat__funptr = NULL
# \ingroup aux_module
# \brief Destroy a \p HYB matrix structure
# 
# \details
# \p hipsparseDestroyHybMat destroys a \p HYB structure.
cdef hipsparseStatus_t hipsparseDestroyHybMat(void * hybA) nogil:
    global _hipsparseDestroyHybMat__funptr
    __init_symbol(&_hipsparseDestroyHybMat__funptr,"hipsparseDestroyHybMat")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroyHybMat__funptr)(hybA)


cdef void* _hipsparseCreateBsrsv2Info__funptr = NULL
# \ingroup aux_module
# \brief Create a bsrsv2 info structure
# 
# \details
# \p hipsparseCreateBsrsv2Info creates a structure that holds the bsrsv2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsrsv2Info().
cdef hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info) nogil:
    global _hipsparseCreateBsrsv2Info__funptr
    __init_symbol(&_hipsparseCreateBsrsv2Info__funptr,"hipsparseCreateBsrsv2Info")
    return (<hipsparseStatus_t (*)(bsrsv2Info_t*) nogil> _hipsparseCreateBsrsv2Info__funptr)(info)


cdef void* _hipsparseDestroyBsrsv2Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a bsrsv2 info structure
# 
# \details
# \p hipsparseDestroyBsrsv2Info destroys a bsrsv2 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info) nogil:
    global _hipsparseDestroyBsrsv2Info__funptr
    __init_symbol(&_hipsparseDestroyBsrsv2Info__funptr,"hipsparseDestroyBsrsv2Info")
    return (<hipsparseStatus_t (*)(bsrsv2Info_t) nogil> _hipsparseDestroyBsrsv2Info__funptr)(info)


cdef void* _hipsparseCreateBsrsm2Info__funptr = NULL
# \ingroup aux_module
# \brief Create a bsrsm2 info structure
# 
# \details
# \p hipsparseCreateBsrsm2Info creates a structure that holds the bsrsm2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsrsm2Info().
cdef hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info) nogil:
    global _hipsparseCreateBsrsm2Info__funptr
    __init_symbol(&_hipsparseCreateBsrsm2Info__funptr,"hipsparseCreateBsrsm2Info")
    return (<hipsparseStatus_t (*)(bsrsm2Info_t*) nogil> _hipsparseCreateBsrsm2Info__funptr)(info)


cdef void* _hipsparseDestroyBsrsm2Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a bsrsm2 info structure
# 
# \details
# \p hipsparseDestroyBsrsm2Info destroys a bsrsm2 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info) nogil:
    global _hipsparseDestroyBsrsm2Info__funptr
    __init_symbol(&_hipsparseDestroyBsrsm2Info__funptr,"hipsparseDestroyBsrsm2Info")
    return (<hipsparseStatus_t (*)(bsrsm2Info_t) nogil> _hipsparseDestroyBsrsm2Info__funptr)(info)


cdef void* _hipsparseCreateBsrilu02Info__funptr = NULL
# \ingroup aux_module
# \brief Create a bsrilu02 info structure
# 
# \details
# \p hipsparseCreateBsrilu02Info creates a structure that holds the bsrilu02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsrilu02Info().
cdef hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info) nogil:
    global _hipsparseCreateBsrilu02Info__funptr
    __init_symbol(&_hipsparseCreateBsrilu02Info__funptr,"hipsparseCreateBsrilu02Info")
    return (<hipsparseStatus_t (*)(bsrilu02Info_t*) nogil> _hipsparseCreateBsrilu02Info__funptr)(info)


cdef void* _hipsparseDestroyBsrilu02Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a bsrilu02 info structure
# 
# \details
# \p hipsparseDestroyBsrilu02Info destroys a bsrilu02 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info) nogil:
    global _hipsparseDestroyBsrilu02Info__funptr
    __init_symbol(&_hipsparseDestroyBsrilu02Info__funptr,"hipsparseDestroyBsrilu02Info")
    return (<hipsparseStatus_t (*)(bsrilu02Info_t) nogil> _hipsparseDestroyBsrilu02Info__funptr)(info)


cdef void* _hipsparseCreateBsric02Info__funptr = NULL
# \ingroup aux_module
# \brief Create a bsric02 info structure
# 
# \details
# \p hipsparseCreateBsric02Info creates a structure that holds the bsric02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyBsric02Info().
cdef hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info) nogil:
    global _hipsparseCreateBsric02Info__funptr
    __init_symbol(&_hipsparseCreateBsric02Info__funptr,"hipsparseCreateBsric02Info")
    return (<hipsparseStatus_t (*)(bsric02Info_t*) nogil> _hipsparseCreateBsric02Info__funptr)(info)


cdef void* _hipsparseDestroyBsric02Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a bsric02 info structure
# 
# \details
# \p hipsparseDestroyBsric02Info destroys a bsric02 info structure.
cdef hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info) nogil:
    global _hipsparseDestroyBsric02Info__funptr
    __init_symbol(&_hipsparseDestroyBsric02Info__funptr,"hipsparseDestroyBsric02Info")
    return (<hipsparseStatus_t (*)(bsric02Info_t) nogil> _hipsparseDestroyBsric02Info__funptr)(info)


cdef void* _hipsparseCreateCsrsv2Info__funptr = NULL
# \ingroup aux_module
# \brief Create a csrsv2 info structure
# 
# \details
# \p hipsparseCreateCsrsv2Info creates a structure that holds the csrsv2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrsv2Info().
cdef hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info) nogil:
    global _hipsparseCreateCsrsv2Info__funptr
    __init_symbol(&_hipsparseCreateCsrsv2Info__funptr,"hipsparseCreateCsrsv2Info")
    return (<hipsparseStatus_t (*)(csrsv2Info_t*) nogil> _hipsparseCreateCsrsv2Info__funptr)(info)


cdef void* _hipsparseDestroyCsrsv2Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a csrsv2 info structure
# 
# \details
# \p hipsparseDestroyCsrsv2Info destroys a csrsv2 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info) nogil:
    global _hipsparseDestroyCsrsv2Info__funptr
    __init_symbol(&_hipsparseDestroyCsrsv2Info__funptr,"hipsparseDestroyCsrsv2Info")
    return (<hipsparseStatus_t (*)(csrsv2Info_t) nogil> _hipsparseDestroyCsrsv2Info__funptr)(info)


cdef void* _hipsparseCreateCsrsm2Info__funptr = NULL
# \ingroup aux_module
# \brief Create a csrsm2 info structure
# 
# \details
# \p hipsparseCreateCsrsm2Info creates a structure that holds the csrsm2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrsm2Info().
cdef hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info) nogil:
    global _hipsparseCreateCsrsm2Info__funptr
    __init_symbol(&_hipsparseCreateCsrsm2Info__funptr,"hipsparseCreateCsrsm2Info")
    return (<hipsparseStatus_t (*)(csrsm2Info_t*) nogil> _hipsparseCreateCsrsm2Info__funptr)(info)


cdef void* _hipsparseDestroyCsrsm2Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a csrsm2 info structure
# 
# \details
# \p hipsparseDestroyCsrsm2Info destroys a csrsm2 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info) nogil:
    global _hipsparseDestroyCsrsm2Info__funptr
    __init_symbol(&_hipsparseDestroyCsrsm2Info__funptr,"hipsparseDestroyCsrsm2Info")
    return (<hipsparseStatus_t (*)(csrsm2Info_t) nogil> _hipsparseDestroyCsrsm2Info__funptr)(info)


cdef void* _hipsparseCreateCsrilu02Info__funptr = NULL
# \ingroup aux_module
# \brief Create a csrilu02 info structure
# 
# \details
# \p hipsparseCreateCsrilu02Info creates a structure that holds the csrilu02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrilu02Info().
cdef hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info) nogil:
    global _hipsparseCreateCsrilu02Info__funptr
    __init_symbol(&_hipsparseCreateCsrilu02Info__funptr,"hipsparseCreateCsrilu02Info")
    return (<hipsparseStatus_t (*)(csrilu02Info_t*) nogil> _hipsparseCreateCsrilu02Info__funptr)(info)


cdef void* _hipsparseDestroyCsrilu02Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a csrilu02 info structure
# 
# \details
# \p hipsparseDestroyCsrilu02Info destroys a csrilu02 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info) nogil:
    global _hipsparseDestroyCsrilu02Info__funptr
    __init_symbol(&_hipsparseDestroyCsrilu02Info__funptr,"hipsparseDestroyCsrilu02Info")
    return (<hipsparseStatus_t (*)(csrilu02Info_t) nogil> _hipsparseDestroyCsrilu02Info__funptr)(info)


cdef void* _hipsparseCreateCsric02Info__funptr = NULL
# \ingroup aux_module
# \brief Create a csric02 info structure
# 
# \details
# \p hipsparseCreateCsric02Info creates a structure that holds the csric02 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsric02Info().
cdef hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info) nogil:
    global _hipsparseCreateCsric02Info__funptr
    __init_symbol(&_hipsparseCreateCsric02Info__funptr,"hipsparseCreateCsric02Info")
    return (<hipsparseStatus_t (*)(csric02Info_t*) nogil> _hipsparseCreateCsric02Info__funptr)(info)


cdef void* _hipsparseDestroyCsric02Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a csric02 info structure
# 
# \details
# \p hipsparseDestroyCsric02Info destroys a csric02 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info) nogil:
    global _hipsparseDestroyCsric02Info__funptr
    __init_symbol(&_hipsparseDestroyCsric02Info__funptr,"hipsparseDestroyCsric02Info")
    return (<hipsparseStatus_t (*)(csric02Info_t) nogil> _hipsparseDestroyCsric02Info__funptr)(info)


cdef void* _hipsparseCreateCsru2csrInfo__funptr = NULL
# \ingroup aux_module
# \brief Create a csru2csr info structure
# 
# \details
# \p hipsparseCreateCsru2csrInfo creates a structure that holds the csru2csr info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsru2csrInfo().
cdef hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info) nogil:
    global _hipsparseCreateCsru2csrInfo__funptr
    __init_symbol(&_hipsparseCreateCsru2csrInfo__funptr,"hipsparseCreateCsru2csrInfo")
    return (<hipsparseStatus_t (*)(csru2csrInfo_t*) nogil> _hipsparseCreateCsru2csrInfo__funptr)(info)


cdef void* _hipsparseDestroyCsru2csrInfo__funptr = NULL
# \ingroup aux_module
# \brief Destroy a csru2csr info structure
# 
# \details
# \p hipsparseDestroyCsru2csrInfo destroys a csru2csr info structure.
cdef hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info) nogil:
    global _hipsparseDestroyCsru2csrInfo__funptr
    __init_symbol(&_hipsparseDestroyCsru2csrInfo__funptr,"hipsparseDestroyCsru2csrInfo")
    return (<hipsparseStatus_t (*)(csru2csrInfo_t) nogil> _hipsparseDestroyCsru2csrInfo__funptr)(info)


cdef void* _hipsparseCreateColorInfo__funptr = NULL
# \ingroup aux_module
# \brief Create a color info structure
# 
# \details
# \p hipsparseCreateColorInfo creates a structure that holds the color info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyColorInfo().
cdef hipsparseStatus_t hipsparseCreateColorInfo(void ** info) nogil:
    global _hipsparseCreateColorInfo__funptr
    __init_symbol(&_hipsparseCreateColorInfo__funptr,"hipsparseCreateColorInfo")
    return (<hipsparseStatus_t (*)(void **) nogil> _hipsparseCreateColorInfo__funptr)(info)


cdef void* _hipsparseDestroyColorInfo__funptr = NULL
# \ingroup aux_module
# \brief Destroy a color info structure
# 
# \details
# \p hipsparseDestroyColorInfo destroys a color info structure.
cdef hipsparseStatus_t hipsparseDestroyColorInfo(void * info) nogil:
    global _hipsparseDestroyColorInfo__funptr
    __init_symbol(&_hipsparseDestroyColorInfo__funptr,"hipsparseDestroyColorInfo")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroyColorInfo__funptr)(info)


cdef void* _hipsparseCreateCsrgemm2Info__funptr = NULL
# \ingroup aux_module
# \brief Create a csrgemm2 info structure
# 
# \details
# \p hipsparseCreateCsrgemm2Info creates a structure that holds the csrgemm2 info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyCsrgemm2Info().
cdef hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info) nogil:
    global _hipsparseCreateCsrgemm2Info__funptr
    __init_symbol(&_hipsparseCreateCsrgemm2Info__funptr,"hipsparseCreateCsrgemm2Info")
    return (<hipsparseStatus_t (*)(csrgemm2Info_t*) nogil> _hipsparseCreateCsrgemm2Info__funptr)(info)


cdef void* _hipsparseDestroyCsrgemm2Info__funptr = NULL
# \ingroup aux_module
# \brief Destroy a csrgemm2 info structure
# 
# \details
# \p hipsparseDestroyCsrgemm2Info destroys a csrgemm2 info structure.
cdef hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info) nogil:
    global _hipsparseDestroyCsrgemm2Info__funptr
    __init_symbol(&_hipsparseDestroyCsrgemm2Info__funptr,"hipsparseDestroyCsrgemm2Info")
    return (<hipsparseStatus_t (*)(csrgemm2Info_t) nogil> _hipsparseDestroyCsrgemm2Info__funptr)(info)


cdef void* _hipsparseCreatePruneInfo__funptr = NULL
# \ingroup aux_module
# \brief Create a prune info structure
# 
# \details
# \p hipsparseCreatePruneInfo creates a structure that holds the prune info data
# that is gathered during the analysis routines available. It should be destroyed
# at the end using hipsparseDestroyPruneInfo().
cdef hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info) nogil:
    global _hipsparseCreatePruneInfo__funptr
    __init_symbol(&_hipsparseCreatePruneInfo__funptr,"hipsparseCreatePruneInfo")
    return (<hipsparseStatus_t (*)(pruneInfo_t*) nogil> _hipsparseCreatePruneInfo__funptr)(info)


cdef void* _hipsparseDestroyPruneInfo__funptr = NULL
# \ingroup aux_module
# \brief Destroy a prune info structure
# 
# \details
# \p hipsparseDestroyPruneInfo destroys a prune info structure.
cdef hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info) nogil:
    global _hipsparseDestroyPruneInfo__funptr
    __init_symbol(&_hipsparseDestroyPruneInfo__funptr,"hipsparseDestroyPruneInfo")
    return (<hipsparseStatus_t (*)(pruneInfo_t) nogil> _hipsparseDestroyPruneInfo__funptr)(info)


cdef void* _hipsparseSaxpyi__funptr = NULL
#    \ingroup level1_module
#   \brief Scale a sparse vector and add it to a dense vector.
# 
#   \details
#   \p hipsparseXaxpyi multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
#   adds the result to the dense vector \f$y\f$, such that
# 
#   \f[
#       y := y + \alpha \cdot x
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSaxpyi(void * handle,int nnz,const float * alpha,const float * xVal,const int * xInd,float * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseSaxpyi__funptr
    __init_symbol(&_hipsparseSaxpyi__funptr,"hipsparseSaxpyi")
    return (<hipsparseStatus_t (*)(void *,int,const float *,const float *,const int *,float *,hipsparseIndexBase_t) nogil> _hipsparseSaxpyi__funptr)(handle,nnz,alpha,xVal,xInd,y,idxBase)


cdef void* _hipsparseDaxpyi__funptr = NULL
cdef hipsparseStatus_t hipsparseDaxpyi(void * handle,int nnz,const double * alpha,const double * xVal,const int * xInd,double * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDaxpyi__funptr
    __init_symbol(&_hipsparseDaxpyi__funptr,"hipsparseDaxpyi")
    return (<hipsparseStatus_t (*)(void *,int,const double *,const double *,const int *,double *,hipsparseIndexBase_t) nogil> _hipsparseDaxpyi__funptr)(handle,nnz,alpha,xVal,xInd,y,idxBase)


cdef void* _hipsparseCaxpyi__funptr = NULL
cdef hipsparseStatus_t hipsparseCaxpyi(void * handle,int nnz,float2 * alpha,float2 * xVal,const int * xInd,float2 * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCaxpyi__funptr
    __init_symbol(&_hipsparseCaxpyi__funptr,"hipsparseCaxpyi")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,float2 *,const int *,float2 *,hipsparseIndexBase_t) nogil> _hipsparseCaxpyi__funptr)(handle,nnz,alpha,xVal,xInd,y,idxBase)


cdef void* _hipsparseZaxpyi__funptr = NULL
cdef hipsparseStatus_t hipsparseZaxpyi(void * handle,int nnz,double2 * alpha,double2 * xVal,const int * xInd,double2 * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZaxpyi__funptr
    __init_symbol(&_hipsparseZaxpyi__funptr,"hipsparseZaxpyi")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,double2 *,const int *,double2 *,hipsparseIndexBase_t) nogil> _hipsparseZaxpyi__funptr)(handle,nnz,alpha,xVal,xInd,y,idxBase)


cdef void* _hipsparseSdoti__funptr = NULL
#    \ingroup level1_module
#   \brief Compute the dot product of a sparse vector with a dense vector.
# 
#   \details
#   \p hipsparseXdoti computes the dot product of the sparse vector \f$x\f$ with the
#   dense vector \f$y\f$, such that
#   \f[
#     \text{result} := y^T x
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           result += x_val[i] * y[x_ind[i]];
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSdoti(void * handle,int nnz,const float * xVal,const int * xInd,const float * y,float * result,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseSdoti__funptr
    __init_symbol(&_hipsparseSdoti__funptr,"hipsparseSdoti")
    return (<hipsparseStatus_t (*)(void *,int,const float *,const int *,const float *,float *,hipsparseIndexBase_t) nogil> _hipsparseSdoti__funptr)(handle,nnz,xVal,xInd,y,result,idxBase)


cdef void* _hipsparseDdoti__funptr = NULL
cdef hipsparseStatus_t hipsparseDdoti(void * handle,int nnz,const double * xVal,const int * xInd,const double * y,double * result,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDdoti__funptr
    __init_symbol(&_hipsparseDdoti__funptr,"hipsparseDdoti")
    return (<hipsparseStatus_t (*)(void *,int,const double *,const int *,const double *,double *,hipsparseIndexBase_t) nogil> _hipsparseDdoti__funptr)(handle,nnz,xVal,xInd,y,result,idxBase)


cdef void* _hipsparseCdoti__funptr = NULL
cdef hipsparseStatus_t hipsparseCdoti(void * handle,int nnz,float2 * xVal,const int * xInd,float2 * y,float2 * result,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCdoti__funptr
    __init_symbol(&_hipsparseCdoti__funptr,"hipsparseCdoti")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,const int *,float2 *,float2 *,hipsparseIndexBase_t) nogil> _hipsparseCdoti__funptr)(handle,nnz,xVal,xInd,y,result,idxBase)


cdef void* _hipsparseZdoti__funptr = NULL
cdef hipsparseStatus_t hipsparseZdoti(void * handle,int nnz,double2 * xVal,const int * xInd,double2 * y,double2 * result,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZdoti__funptr
    __init_symbol(&_hipsparseZdoti__funptr,"hipsparseZdoti")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,const int *,double2 *,double2 *,hipsparseIndexBase_t) nogil> _hipsparseZdoti__funptr)(handle,nnz,xVal,xInd,y,result,idxBase)


cdef void* _hipsparseCdotci__funptr = NULL
#    \ingroup level1_module
#   \brief Compute the dot product of a complex conjugate sparse vector with a dense
#   vector.
# 
#   \details
#   \p hipsparseXdotci computes the dot product of the complex conjugate sparse vector
#   \f$x\f$ with the dense vector \f$y\f$, such that
#   \f[
#     \text{result} := \bar{x}^H y
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           result += conj(x_val[i]) * y[x_ind[i]];
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseCdotci(void * handle,int nnz,float2 * xVal,const int * xInd,float2 * y,float2 * result,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCdotci__funptr
    __init_symbol(&_hipsparseCdotci__funptr,"hipsparseCdotci")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,const int *,float2 *,float2 *,hipsparseIndexBase_t) nogil> _hipsparseCdotci__funptr)(handle,nnz,xVal,xInd,y,result,idxBase)


cdef void* _hipsparseZdotci__funptr = NULL
cdef hipsparseStatus_t hipsparseZdotci(void * handle,int nnz,double2 * xVal,const int * xInd,double2 * y,double2 * result,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZdotci__funptr
    __init_symbol(&_hipsparseZdotci__funptr,"hipsparseZdotci")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,const int *,double2 *,double2 *,hipsparseIndexBase_t) nogil> _hipsparseZdotci__funptr)(handle,nnz,xVal,xInd,y,result,idxBase)


cdef void* _hipsparseSgthr__funptr = NULL
#    \ingroup level1_module
#   \brief Gather elements from a dense vector and store them into a sparse vector.
# 
#   \details
#   \p hipsparseXgthr gathers the elements that are listed in \p x_ind from the dense
#   vector \f$y\f$ and stores them in the sparse vector \f$x\f$.
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           x_val[i] = y[x_ind[i]];
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgthr(void * handle,int nnz,const float * y,float * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseSgthr__funptr
    __init_symbol(&_hipsparseSgthr__funptr,"hipsparseSgthr")
    return (<hipsparseStatus_t (*)(void *,int,const float *,float *,const int *,hipsparseIndexBase_t) nogil> _hipsparseSgthr__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseDgthr__funptr = NULL
cdef hipsparseStatus_t hipsparseDgthr(void * handle,int nnz,const double * y,double * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDgthr__funptr
    __init_symbol(&_hipsparseDgthr__funptr,"hipsparseDgthr")
    return (<hipsparseStatus_t (*)(void *,int,const double *,double *,const int *,hipsparseIndexBase_t) nogil> _hipsparseDgthr__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseCgthr__funptr = NULL
cdef hipsparseStatus_t hipsparseCgthr(void * handle,int nnz,float2 * y,float2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCgthr__funptr
    __init_symbol(&_hipsparseCgthr__funptr,"hipsparseCgthr")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,float2 *,const int *,hipsparseIndexBase_t) nogil> _hipsparseCgthr__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseZgthr__funptr = NULL
cdef hipsparseStatus_t hipsparseZgthr(void * handle,int nnz,double2 * y,double2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZgthr__funptr
    __init_symbol(&_hipsparseZgthr__funptr,"hipsparseZgthr")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,double2 *,const int *,hipsparseIndexBase_t) nogil> _hipsparseZgthr__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseSgthrz__funptr = NULL
#    \ingroup level1_module
#   \brief Gather and zero out elements from a dense vector and store them into a sparse
#   vector.
# 
#   \details
#   \p hipsparseXgthrz gathers the elements that are listed in \p x_ind from the dense
#   vector \f$y\f$ and stores them in the sparse vector \f$x\f$. The gathered elements
#   in \f$y\f$ are replaced by zero.
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           x_val[i]    = y[x_ind[i]];
#           y[x_ind[i]] = 0;
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgthrz(void * handle,int nnz,float * y,float * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseSgthrz__funptr
    __init_symbol(&_hipsparseSgthrz__funptr,"hipsparseSgthrz")
    return (<hipsparseStatus_t (*)(void *,int,float *,float *,const int *,hipsparseIndexBase_t) nogil> _hipsparseSgthrz__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseDgthrz__funptr = NULL
cdef hipsparseStatus_t hipsparseDgthrz(void * handle,int nnz,double * y,double * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDgthrz__funptr
    __init_symbol(&_hipsparseDgthrz__funptr,"hipsparseDgthrz")
    return (<hipsparseStatus_t (*)(void *,int,double *,double *,const int *,hipsparseIndexBase_t) nogil> _hipsparseDgthrz__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseCgthrz__funptr = NULL
cdef hipsparseStatus_t hipsparseCgthrz(void * handle,int nnz,float2 * y,float2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCgthrz__funptr
    __init_symbol(&_hipsparseCgthrz__funptr,"hipsparseCgthrz")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,float2 *,const int *,hipsparseIndexBase_t) nogil> _hipsparseCgthrz__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseZgthrz__funptr = NULL
cdef hipsparseStatus_t hipsparseZgthrz(void * handle,int nnz,double2 * y,double2 * xVal,const int * xInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZgthrz__funptr
    __init_symbol(&_hipsparseZgthrz__funptr,"hipsparseZgthrz")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,double2 *,const int *,hipsparseIndexBase_t) nogil> _hipsparseZgthrz__funptr)(handle,nnz,y,xVal,xInd,idxBase)


cdef void* _hipsparseSroti__funptr = NULL
#    \ingroup level1_module
#   \brief Apply Givens rotation to a dense and a sparse vector.
# 
#   \details
#   \p hipsparseXroti applies the Givens rotation matrix \f$G\f$ to the sparse vector
#   \f$x\f$ and the dense vector \f$y\f$, where
#   \f[
#     G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           x_tmp = x_val[i];
#           y_tmp = y[x_ind[i]];
# 
#           x_val[i]    = c * x_tmp + s * y_tmp;
#           y[x_ind[i]] = c * y_tmp - s * x_tmp;
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSroti(void * handle,int nnz,float * xVal,const int * xInd,float * y,const float * c,const float * s,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseSroti__funptr
    __init_symbol(&_hipsparseSroti__funptr,"hipsparseSroti")
    return (<hipsparseStatus_t (*)(void *,int,float *,const int *,float *,const float *,const float *,hipsparseIndexBase_t) nogil> _hipsparseSroti__funptr)(handle,nnz,xVal,xInd,y,c,s,idxBase)


cdef void* _hipsparseDroti__funptr = NULL
cdef hipsparseStatus_t hipsparseDroti(void * handle,int nnz,double * xVal,const int * xInd,double * y,const double * c,const double * s,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDroti__funptr
    __init_symbol(&_hipsparseDroti__funptr,"hipsparseDroti")
    return (<hipsparseStatus_t (*)(void *,int,double *,const int *,double *,const double *,const double *,hipsparseIndexBase_t) nogil> _hipsparseDroti__funptr)(handle,nnz,xVal,xInd,y,c,s,idxBase)


cdef void* _hipsparseSsctr__funptr = NULL
#    \ingroup level1_module
#   \brief Scatter elements from a dense vector across a sparse vector.
# 
#   \details
#   \p hipsparseXsctr scatters the elements that are listed in \p x_ind from the sparse
#   vector \f$x\f$ into the dense vector \f$y\f$. Indices of \f$y\f$ that are not listed
#   in \p x_ind remain unchanged.
# 
#   \code{.c}
#       for(i = 0; i < nnz; ++i)
#       {
#           y[x_ind[i]] = x_val[i];
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSsctr(void * handle,int nnz,const float * xVal,const int * xInd,float * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseSsctr__funptr
    __init_symbol(&_hipsparseSsctr__funptr,"hipsparseSsctr")
    return (<hipsparseStatus_t (*)(void *,int,const float *,const int *,float *,hipsparseIndexBase_t) nogil> _hipsparseSsctr__funptr)(handle,nnz,xVal,xInd,y,idxBase)


cdef void* _hipsparseDsctr__funptr = NULL
cdef hipsparseStatus_t hipsparseDsctr(void * handle,int nnz,const double * xVal,const int * xInd,double * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDsctr__funptr
    __init_symbol(&_hipsparseDsctr__funptr,"hipsparseDsctr")
    return (<hipsparseStatus_t (*)(void *,int,const double *,const int *,double *,hipsparseIndexBase_t) nogil> _hipsparseDsctr__funptr)(handle,nnz,xVal,xInd,y,idxBase)


cdef void* _hipsparseCsctr__funptr = NULL
cdef hipsparseStatus_t hipsparseCsctr(void * handle,int nnz,float2 * xVal,const int * xInd,float2 * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCsctr__funptr
    __init_symbol(&_hipsparseCsctr__funptr,"hipsparseCsctr")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,const int *,float2 *,hipsparseIndexBase_t) nogil> _hipsparseCsctr__funptr)(handle,nnz,xVal,xInd,y,idxBase)


cdef void* _hipsparseZsctr__funptr = NULL
cdef hipsparseStatus_t hipsparseZsctr(void * handle,int nnz,double2 * xVal,const int * xInd,double2 * y,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZsctr__funptr
    __init_symbol(&_hipsparseZsctr__funptr,"hipsparseZsctr")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,const int *,double2 *,hipsparseIndexBase_t) nogil> _hipsparseZsctr__funptr)(handle,nnz,xVal,xInd,y,idxBase)


cdef void* _hipsparseScsrmv__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse matrix vector multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
#   matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
#   result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
#   such that
#   \f[
#     y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < m; ++i)
#       {
#           y[i] = beta * y[i];
# 
#           for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
#           {
#               y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
#           }
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * x,const float * beta,float * y) nogil:
    global _hipsparseScsrmv__funptr
    __init_symbol(&_hipsparseScsrmv__funptr,"hipsparseScsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,const float *,void *const,const float *,const int *,const int *,const float *,const float *,float *) nogil> _hipsparseScsrmv__funptr)(handle,transA,m,n,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,x,beta,y)


cdef void* _hipsparseDcsrmv__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * x,const double * beta,double * y) nogil:
    global _hipsparseDcsrmv__funptr
    __init_symbol(&_hipsparseDcsrmv__funptr,"hipsparseDcsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,const double *,void *const,const double *,const int *,const int *,const double *,const double *,double *) nogil> _hipsparseDcsrmv__funptr)(handle,transA,m,n,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,x,beta,y)


cdef void* _hipsparseCcsrmv__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * x,float2 * beta,float2 * y) nogil:
    global _hipsparseCcsrmv__funptr
    __init_symbol(&_hipsparseCcsrmv__funptr,"hipsparseCcsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,float2 *,void *const,float2 *,const int *,const int *,float2 *,float2 *,float2 *) nogil> _hipsparseCcsrmv__funptr)(handle,transA,m,n,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,x,beta,y)


cdef void* _hipsparseZcsrmv__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrmv(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * x,double2 * beta,double2 * y) nogil:
    global _hipsparseZcsrmv__funptr
    __init_symbol(&_hipsparseZcsrmv__funptr,"hipsparseZcsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,double2 *,void *const,double2 *,const int *,const int *,double2 *,double2 *,double2 *) nogil> _hipsparseZcsrmv__funptr)(handle,transA,m,n,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,x,beta,y)


cdef void* _hipsparseXcsrsv2_zeroPivot__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
#   structural or numerical zero has been found during hipsparseScsrsv2_solve(),
#   hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() or hipsparseZcsrsv2_solve()
#   computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
#   using same index base as the CSR matrix.
# 
#   \p position can be in host or device memory. If no zero pivot has been found,
#   \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
#   \note \p hipsparseXcsrsv2_zeroPivot is a blocking function. It might influence
#   performance negatively.
# /
cdef hipsparseStatus_t hipsparseXcsrsv2_zeroPivot(void * handle,csrsv2Info_t info,int * position) nogil:
    global _hipsparseXcsrsv2_zeroPivot__funptr
    __init_symbol(&_hipsparseXcsrsv2_zeroPivot__funptr,"hipsparseXcsrsv2_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,csrsv2Info_t,int *) nogil> _hipsparseXcsrsv2_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseScsrsv2_bufferSize__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsv2_bufferSize returns the size of the temporary storage buffer that
#   is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
#   hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
#   hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseScsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseScsrsv2_bufferSize__funptr,"hipsparseScsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,float *,const int *,const int *,csrsv2Info_t,int *) nogil> _hipsparseScsrsv2_bufferSize__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseDcsrsv2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDcsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseDcsrsv2_bufferSize__funptr,"hipsparseDcsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,double *,const int *,const int *,csrsv2Info_t,int *) nogil> _hipsparseDcsrsv2_bufferSize__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseCcsrsv2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCcsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseCcsrsv2_bufferSize__funptr,"hipsparseCcsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,float2 *,const int *,const int *,csrsv2Info_t,int *) nogil> _hipsparseCcsrsv2_bufferSize__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseZcsrsv2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsv2_bufferSize(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZcsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseZcsrsv2_bufferSize__funptr,"hipsparseZcsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,double2 *,const int *,const int *,csrsv2Info_t,int *) nogil> _hipsparseZcsrsv2_bufferSize__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseScsrsv2_bufferSizeExt__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsv2_bufferSizeExt returns the size of the temporary storage buffer that
#   is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
#   hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
#   hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseScsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsrsv2_bufferSizeExt__funptr,"hipsparseScsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,float *,const int *,const int *,csrsv2Info_t,unsigned long *) nogil> _hipsparseScsrsv2_bufferSizeExt__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseDcsrsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseDcsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsrsv2_bufferSizeExt__funptr,"hipsparseDcsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,double *,const int *,const int *,csrsv2Info_t,unsigned long *) nogil> _hipsparseDcsrsv2_bufferSizeExt__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseCcsrsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseCcsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsrsv2_bufferSizeExt__funptr,"hipsparseCcsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,float2 *,const int *,const int *,csrsv2Info_t,unsigned long *) nogil> _hipsparseCcsrsv2_bufferSizeExt__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseZcsrsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseZcsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsrsv2_bufferSizeExt__funptr,"hipsparseZcsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,double2 *,const int *,const int *,csrsv2Info_t,unsigned long *) nogil> _hipsparseZcsrsv2_bufferSizeExt__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseScsrsv2_analysis__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsv2_analysis performs the analysis step for hipsparseScsrsv2_solve(),
#   hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve().
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsrsv2_analysis__funptr
    __init_symbol(&_hipsparseScsrsv2_analysis__funptr,"hipsparseScsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,const float *,const int *,const int *,csrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsrsv2_analysis__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseDcsrsv2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsrsv2_analysis__funptr
    __init_symbol(&_hipsparseDcsrsv2_analysis__funptr,"hipsparseDcsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,const double *,const int *,const int *,csrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsrsv2_analysis__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseCcsrsv2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsrsv2_analysis__funptr
    __init_symbol(&_hipsparseCcsrsv2_analysis__funptr,"hipsparseCcsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,float2 *,const int *,const int *,csrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsrsv2_analysis__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseZcsrsv2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsv2_analysis(void * handle,hipsparseOperation_t transA,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsrsv2_analysis__funptr
    __init_symbol(&_hipsparseZcsrsv2_analysis__funptr,"hipsparseZcsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,void *const,double2 *,const int *,const int *,csrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsrsv2_analysis__funptr)(handle,transA,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseScsrsv2_solve__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsv2_solve solves a sparse triangular linear system of a sparse
#   \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
#   \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
#   \f[
#     op(A) \cdot y = \alpha \cdot x,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \p hipsparseXcsrsv2_solve requires a user allocated temporary buffer. Its size is
#   returned by hipsparseXcsrsv2_bufferSize() or hipsparseXcsrsv2_bufferSizeExt().
#   Furthermore, analysis meta data is required. It can be obtained by
#   hipsparseXcsrsv2_analysis(). \p hipsparseXcsrsv2_solve reports the first zero pivot
#   (either numerical or structural zero). The zero pivot status can be checked calling
#   hipsparseXcsrsv2_zeroPivot(). If
#   \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
#   reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
#   \note
#   The sparse CSR matrix has to be sorted. This can be achieved by calling
#   hipsparseXcsrsort().
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
#   \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,const float * f,float * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsrsv2_solve__funptr
    __init_symbol(&_hipsparseScsrsv2_solve__funptr,"hipsparseScsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,const float *,void *const,const float *,const int *,const int *,csrsv2Info_t,const float *,float *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsrsv2_solve__funptr)(handle,transA,m,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,f,x,policy,pBuffer)


cdef void* _hipsparseDcsrsv2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,const double * f,double * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsrsv2_solve__funptr
    __init_symbol(&_hipsparseDcsrsv2_solve__funptr,"hipsparseDcsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,const double *,void *const,const double *,const int *,const int *,csrsv2Info_t,const double *,double *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsrsv2_solve__funptr)(handle,transA,m,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,f,x,policy,pBuffer)


cdef void* _hipsparseCcsrsv2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,float2 * f,float2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsrsv2_solve__funptr
    __init_symbol(&_hipsparseCcsrsv2_solve__funptr,"hipsparseCcsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,float2 *,void *const,float2 *,const int *,const int *,csrsv2Info_t,float2 *,float2 *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsrsv2_solve__funptr)(handle,transA,m,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,f,x,policy,pBuffer)


cdef void* _hipsparseZcsrsv2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsv2_solve(void * handle,hipsparseOperation_t transA,int m,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrsv2Info_t info,double2 * f,double2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsrsv2_solve__funptr
    __init_symbol(&_hipsparseZcsrsv2_solve__funptr,"hipsparseZcsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,double2 *,void *const,double2 *,const int *,const int *,csrsv2Info_t,double2 *,double2 *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsrsv2_solve__funptr)(handle,transA,m,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,f,x,policy,pBuffer)


cdef void* _hipsparseShybmv__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse matrix vector multiplication using HYB storage format
# 
#   \details
#   \p hipsparseXhybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
#   matrix, defined in HYB storage format, and the dense vector \f$x\f$ and adds the
#   result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
#   such that
#   \f[
#     y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseShybmv(void * handle,hipsparseOperation_t transA,const float * alpha,void *const descrA,void *const hybA,const float * x,const float * beta,float * y) nogil:
    global _hipsparseShybmv__funptr
    __init_symbol(&_hipsparseShybmv__funptr,"hipsparseShybmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const float *,void *const,void *const,const float *,const float *,float *) nogil> _hipsparseShybmv__funptr)(handle,transA,alpha,descrA,hybA,x,beta,y)


cdef void* _hipsparseDhybmv__funptr = NULL
cdef hipsparseStatus_t hipsparseDhybmv(void * handle,hipsparseOperation_t transA,const double * alpha,void *const descrA,void *const hybA,const double * x,const double * beta,double * y) nogil:
    global _hipsparseDhybmv__funptr
    __init_symbol(&_hipsparseDhybmv__funptr,"hipsparseDhybmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const double *,void *const,void *const,const double *,const double *,double *) nogil> _hipsparseDhybmv__funptr)(handle,transA,alpha,descrA,hybA,x,beta,y)


cdef void* _hipsparseChybmv__funptr = NULL
cdef hipsparseStatus_t hipsparseChybmv(void * handle,hipsparseOperation_t transA,float2 * alpha,void *const descrA,void *const hybA,float2 * x,float2 * beta,float2 * y) nogil:
    global _hipsparseChybmv__funptr
    __init_symbol(&_hipsparseChybmv__funptr,"hipsparseChybmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,float2 *,void *const,void *const,float2 *,float2 *,float2 *) nogil> _hipsparseChybmv__funptr)(handle,transA,alpha,descrA,hybA,x,beta,y)


cdef void* _hipsparseZhybmv__funptr = NULL
cdef hipsparseStatus_t hipsparseZhybmv(void * handle,hipsparseOperation_t transA,double2 * alpha,void *const descrA,void *const hybA,double2 * x,double2 * beta,double2 * y) nogil:
    global _hipsparseZhybmv__funptr
    __init_symbol(&_hipsparseZhybmv__funptr,"hipsparseZhybmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,double2 *,void *const,void *const,double2 *,double2 *,double2 *) nogil> _hipsparseZhybmv__funptr)(handle,transA,alpha,descrA,hybA,x,beta,y)


cdef void* _hipsparseSbsrmv__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse matrix vector multiplication using BSR storage format
# 
#   \details
#   \p hipsparseXbsrmv multiplies the scalar \f$\alpha\f$ with a sparse
#   \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
#   matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
#   result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
#   such that
#   \f[
#     y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,const float * alpha,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,const float * x,const float * beta,float * y) nogil:
    global _hipsparseSbsrmv__funptr
    __init_symbol(&_hipsparseSbsrmv__funptr,"hipsparseSbsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,const float *,void *const,const float *,const int *,const int *,int,const float *,const float *,float *) nogil> _hipsparseSbsrmv__funptr)(handle,dirA,transA,mb,nb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,x,beta,y)


cdef void* _hipsparseDbsrmv__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,const double * alpha,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,const double * x,const double * beta,double * y) nogil:
    global _hipsparseDbsrmv__funptr
    __init_symbol(&_hipsparseDbsrmv__funptr,"hipsparseDbsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,const double *,void *const,const double *,const int *,const int *,int,const double *,const double *,double *) nogil> _hipsparseDbsrmv__funptr)(handle,dirA,transA,mb,nb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,x,beta,y)


cdef void* _hipsparseCbsrmv__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,float2 * alpha,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,float2 * x,float2 * beta,float2 * y) nogil:
    global _hipsparseCbsrmv__funptr
    __init_symbol(&_hipsparseCbsrmv__funptr,"hipsparseCbsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,float2 *,void *const,float2 *,const int *,const int *,int,float2 *,float2 *,float2 *) nogil> _hipsparseCbsrmv__funptr)(handle,dirA,transA,mb,nb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,x,beta,y)


cdef void* _hipsparseZbsrmv__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrmv(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nb,int nnzb,double2 * alpha,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,double2 * x,double2 * beta,double2 * y) nogil:
    global _hipsparseZbsrmv__funptr
    __init_symbol(&_hipsparseZbsrmv__funptr,"hipsparseZbsrmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,double2 *,void *const,double2 *,const int *,const int *,int,double2 *,double2 *,double2 *) nogil> _hipsparseZbsrmv__funptr)(handle,dirA,transA,mb,nb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,x,beta,y)


cdef void* _hipsparseSbsrxmv__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse matrix vector multiplication with mask operation using BSR storage format
# 
#   \details
#   \p hipsparseXbsrxmv multiplies the scalar \f$\alpha\f$ with a sparse
#   \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
#   modified matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
#   result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
#   such that
#   \f[
#     y := \left( \alpha \cdot op(A) \cdot x + \beta \cdot y \right)\left( \text{mask} \right),
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   The \f$\text{mask}\f$ is defined as an array of block row indices.
#   The input sparse matrix is defined with a modified BSR storage format where the beginning and the end of each row
#   is defined with two arrays, \p bsr_row_ptr and \p bsr_end_ptr (both of size \p mb), rather the usual \p bsr_row_ptr of size \p mb + 1.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
#   Currently, \p block_dim == 1 is not supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,const float * alpha,void *const descr,const float * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,const float * x,const float * beta,float * y) nogil:
    global _hipsparseSbsrxmv__funptr
    __init_symbol(&_hipsparseSbsrxmv__funptr,"hipsparseSbsrxmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,int,const float *,void *const,const float *,const int *,const int *,const int *,const int *,int,const float *,const float *,float *) nogil> _hipsparseSbsrxmv__funptr)(handle,dir,trans,sizeOfMask,mb,nb,nnzb,alpha,descr,bsrVal,bsrMaskPtr,bsrRowPtr,bsrEndPtr,bsrColInd,blockDim,x,beta,y)


cdef void* _hipsparseDbsrxmv__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,const double * alpha,void *const descr,const double * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,const double * x,const double * beta,double * y) nogil:
    global _hipsparseDbsrxmv__funptr
    __init_symbol(&_hipsparseDbsrxmv__funptr,"hipsparseDbsrxmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,int,const double *,void *const,const double *,const int *,const int *,const int *,const int *,int,const double *,const double *,double *) nogil> _hipsparseDbsrxmv__funptr)(handle,dir,trans,sizeOfMask,mb,nb,nnzb,alpha,descr,bsrVal,bsrMaskPtr,bsrRowPtr,bsrEndPtr,bsrColInd,blockDim,x,beta,y)


cdef void* _hipsparseCbsrxmv__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,float2 * alpha,void *const descr,float2 * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,float2 * x,float2 * beta,float2 * y) nogil:
    global _hipsparseCbsrxmv__funptr
    __init_symbol(&_hipsparseCbsrxmv__funptr,"hipsparseCbsrxmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,int,float2 *,void *const,float2 *,const int *,const int *,const int *,const int *,int,float2 *,float2 *,float2 *) nogil> _hipsparseCbsrxmv__funptr)(handle,dir,trans,sizeOfMask,mb,nb,nnzb,alpha,descr,bsrVal,bsrMaskPtr,bsrRowPtr,bsrEndPtr,bsrColInd,blockDim,x,beta,y)


cdef void* _hipsparseZbsrxmv__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrxmv(void * handle,hipsparseDirection_t dir,hipsparseOperation_t trans,int sizeOfMask,int mb,int nb,int nnzb,double2 * alpha,void *const descr,double2 * bsrVal,const int * bsrMaskPtr,const int * bsrRowPtr,const int * bsrEndPtr,const int * bsrColInd,int blockDim,double2 * x,double2 * beta,double2 * y) nogil:
    global _hipsparseZbsrxmv__funptr
    __init_symbol(&_hipsparseZbsrxmv__funptr,"hipsparseZbsrxmv")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,int,int,double2 *,void *const,double2 *,const int *,const int *,const int *,const int *,int,double2 *,double2 *,double2 *) nogil> _hipsparseZbsrxmv__funptr)(handle,dir,trans,sizeOfMask,mb,nb,nnzb,alpha,descr,bsrVal,bsrMaskPtr,bsrRowPtr,bsrEndPtr,bsrColInd,blockDim,x,beta,y)


cdef void* _hipsparseXbsrsv2_zeroPivot__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
#   structural or numerical zero has been found during hipsparseXbsrsv2_analysis() or
#   hipsparseXbsrsv2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
#   is stored in \p position, using same index base as the BSR matrix.
# 
#   \p position can be in host or device memory. If no zero pivot has been found,
#   \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
#   \note \p hipsparseXbsrsv2_zeroPivot is a blocking function. It might influence
#   performance negatively.
# /
cdef hipsparseStatus_t hipsparseXbsrsv2_zeroPivot(void * handle,bsrsv2Info_t info,int * position) nogil:
    global _hipsparseXbsrsv2_zeroPivot__funptr
    __init_symbol(&_hipsparseXbsrsv2_zeroPivot__funptr,"hipsparseXbsrsv2_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,bsrsv2Info_t,int *) nogil> _hipsparseXbsrsv2_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseSbsrsv2_bufferSize__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsv2_bufferSize returns the size of the temporary storage buffer that
#   is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseSbsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseSbsrsv2_bufferSize__funptr,"hipsparseSbsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,float *,const int *,const int *,int,bsrsv2Info_t,int *) nogil> _hipsparseSbsrsv2_bufferSize__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseDbsrsv2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDbsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseDbsrsv2_bufferSize__funptr,"hipsparseDbsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,double *,const int *,const int *,int,bsrsv2Info_t,int *) nogil> _hipsparseDbsrsv2_bufferSize__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseCbsrsv2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCbsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseCbsrsv2_bufferSize__funptr,"hipsparseCbsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,float2 *,const int *,const int *,int,bsrsv2Info_t,int *) nogil> _hipsparseCbsrsv2_bufferSize__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseZbsrsv2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsv2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZbsrsv2_bufferSize__funptr
    __init_symbol(&_hipsparseZbsrsv2_bufferSize__funptr,"hipsparseZbsrsv2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,double2 *,const int *,const int *,int,bsrsv2Info_t,int *) nogil> _hipsparseZbsrsv2_bufferSize__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseSbsrsv2_bufferSizeExt__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsv2_bufferSizeExt returns the size of the temporary storage buffer that
#   is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseSbsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSbsrsv2_bufferSizeExt__funptr,"hipsparseSbsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,float *,const int *,const int *,int,bsrsv2Info_t,unsigned long *) nogil> _hipsparseSbsrsv2_bufferSizeExt__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSize)


cdef void* _hipsparseDbsrsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseDbsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDbsrsv2_bufferSizeExt__funptr,"hipsparseDbsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,double *,const int *,const int *,int,bsrsv2Info_t,unsigned long *) nogil> _hipsparseDbsrsv2_bufferSizeExt__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSize)


cdef void* _hipsparseCbsrsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseCbsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCbsrsv2_bufferSizeExt__funptr,"hipsparseCbsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,float2 *,const int *,const int *,int,bsrsv2Info_t,unsigned long *) nogil> _hipsparseCbsrsv2_bufferSizeExt__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSize)


cdef void* _hipsparseZbsrsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseZbsrsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZbsrsv2_bufferSizeExt__funptr,"hipsparseZbsrsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,double2 *,const int *,const int *,int,bsrsv2Info_t,unsigned long *) nogil> _hipsparseZbsrsv2_bufferSizeExt__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSize)


cdef void* _hipsparseSbsrsv2_analysis__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsv2_analysis performs the analysis step for hipsparseXbsrsv2_solve().
# 
#   \note
#   If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsrsv2_analysis__funptr
    __init_symbol(&_hipsparseSbsrsv2_analysis__funptr,"hipsparseSbsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,const float *,const int *,const int *,int,bsrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsrsv2_analysis__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseDbsrsv2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsrsv2_analysis__funptr
    __init_symbol(&_hipsparseDbsrsv2_analysis__funptr,"hipsparseDbsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,const double *,const int *,const int *,int,bsrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsrsv2_analysis__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseCbsrsv2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsrsv2_analysis__funptr
    __init_symbol(&_hipsparseCbsrsv2_analysis__funptr,"hipsparseCbsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,float2 *,const int *,const int *,int,bsrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsrsv2_analysis__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseZbsrsv2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsv2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsrsv2_analysis__funptr
    __init_symbol(&_hipsparseZbsrsv2_analysis__funptr,"hipsparseZbsrsv2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,void *const,double2 *,const int *,const int *,int,bsrsv2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsrsv2_analysis__funptr)(handle,dirA,transA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseSbsrsv2_solve__funptr = NULL
#    \ingroup level2_module
#   \brief Sparse triangular solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsv2_solve solves a sparse triangular linear system of a sparse
#   \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution vector
#   \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
#   \f[
#     op(A) \cdot y = \alpha \cdot x,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \p hipsparseXbsrsv2_solve requires a user allocated temporary buffer. Its size is
#   returned by hipsparseXbsrsv2_bufferSize() or hipsparseXbsrsv2_bufferSizeExt().
#   Furthermore, analysis meta data is required. It can be obtained by
#   hipsparseXbsrsv2_analysis(). \p hipsparseXbsrsv2_solve reports the first zero pivot
#   (either numerical or structural zero). The zero pivot status can be checked calling
#   hipsparseXbsrsv2_zeroPivot(). If
#   \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
#   reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
#   \note
#   The sparse BSR matrix has to be sorted.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
#   \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,const float * alpha,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,const float * f,float * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsrsv2_solve__funptr
    __init_symbol(&_hipsparseSbsrsv2_solve__funptr,"hipsparseSbsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,const float *,void *const,const float *,const int *,const int *,int,bsrsv2Info_t,const float *,float *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsrsv2_solve__funptr)(handle,dirA,transA,mb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,f,x,policy,pBuffer)


cdef void* _hipsparseDbsrsv2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,const double * alpha,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,const double * f,double * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsrsv2_solve__funptr
    __init_symbol(&_hipsparseDbsrsv2_solve__funptr,"hipsparseDbsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,const double *,void *const,const double *,const int *,const int *,int,bsrsv2Info_t,const double *,double *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsrsv2_solve__funptr)(handle,dirA,transA,mb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,f,x,policy,pBuffer)


cdef void* _hipsparseCbsrsv2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,float2 * alpha,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,float2 * f,float2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsrsv2_solve__funptr
    __init_symbol(&_hipsparseCbsrsv2_solve__funptr,"hipsparseCbsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,float2 *,void *const,float2 *,const int *,const int *,int,bsrsv2Info_t,float2 *,float2 *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsrsv2_solve__funptr)(handle,dirA,transA,mb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,f,x,policy,pBuffer)


cdef void* _hipsparseZbsrsv2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsv2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,int mb,int nnzb,double2 * alpha,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsv2Info_t info,double2 * f,double2 * x,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsrsv2_solve__funptr
    __init_symbol(&_hipsparseZbsrsv2_solve__funptr,"hipsparseZbsrsv2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,int,int,double2 *,void *const,double2 *,const int *,const int *,int,bsrsv2Info_t,double2 *,double2 *,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsrsv2_solve__funptr)(handle,dirA,transA,mb,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,f,x,policy,pBuffer)


cdef void* _hipsparseSgemvi_bufferSize__funptr = NULL
#  \ingroup level2_module
#  \brief Dense matrix sparse vector multiplication
# 
#  \details
#  \p hipsparseXgemvi_bufferSize returns the size of the temporary storage buffer
#  required by hipsparseXgemvi(). The temporary storage buffer must be allocated by the
#  user.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil:
    global _hipsparseSgemvi_bufferSize__funptr
    __init_symbol(&_hipsparseSgemvi_bufferSize__funptr,"hipsparseSgemvi_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int *) nogil> _hipsparseSgemvi_bufferSize__funptr)(handle,transA,m,n,nnz,pBufferSize)


cdef void* _hipsparseDgemvi_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil:
    global _hipsparseDgemvi_bufferSize__funptr
    __init_symbol(&_hipsparseDgemvi_bufferSize__funptr,"hipsparseDgemvi_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int *) nogil> _hipsparseDgemvi_bufferSize__funptr)(handle,transA,m,n,nnz,pBufferSize)


cdef void* _hipsparseCgemvi_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil:
    global _hipsparseCgemvi_bufferSize__funptr
    __init_symbol(&_hipsparseCgemvi_bufferSize__funptr,"hipsparseCgemvi_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int *) nogil> _hipsparseCgemvi_bufferSize__funptr)(handle,transA,m,n,nnz,pBufferSize)


cdef void* _hipsparseZgemvi_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZgemvi_bufferSize(void * handle,hipsparseOperation_t transA,int m,int n,int nnz,int * pBufferSize) nogil:
    global _hipsparseZgemvi_bufferSize__funptr
    __init_symbol(&_hipsparseZgemvi_bufferSize__funptr,"hipsparseZgemvi_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int *) nogil> _hipsparseZgemvi_bufferSize__funptr)(handle,transA,m,n,nnz,pBufferSize)


cdef void* _hipsparseSgemvi__funptr = NULL
#  \ingroup level2_module
#  \brief Dense matrix sparse vector multiplication
# 
#  \details
#  \p hipsparseXgemvi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times n\f$
#  matrix \f$A\f$ and the sparse vector \f$x\f$ and adds the result to the dense vector
#  \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
#  \f[
#    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
#  \f]
#  with
#  \f[
#    op(A) = \left\{
#    \begin{array}{ll}
#        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
#        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#    \end{array}
#    \right.
#  \f]
# 
#  \p hipsparseXgemvi requires a user allocated temporary buffer. Its size is returned
#  by hipsparseXgemvi_bufferSize().
# 
#  \note
#  This function is non blocking and executed asynchronously with respect to the host.
#  It may return before the actual computation has finished.
# 
#  \note
#  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSgemvi(void * handle,hipsparseOperation_t transA,int m,int n,const float * alpha,const float * A,int lda,int nnz,const float * x,const int * xInd,const float * beta,float * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil:
    global _hipsparseSgemvi__funptr
    __init_symbol(&_hipsparseSgemvi__funptr,"hipsparseSgemvi")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,const float *,const float *,int,int,const float *,const int *,const float *,float *,hipsparseIndexBase_t,void *) nogil> _hipsparseSgemvi__funptr)(handle,transA,m,n,alpha,A,lda,nnz,x,xInd,beta,y,idxBase,pBuffer)


cdef void* _hipsparseDgemvi__funptr = NULL
cdef hipsparseStatus_t hipsparseDgemvi(void * handle,hipsparseOperation_t transA,int m,int n,const double * alpha,const double * A,int lda,int nnz,const double * x,const int * xInd,const double * beta,double * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil:
    global _hipsparseDgemvi__funptr
    __init_symbol(&_hipsparseDgemvi__funptr,"hipsparseDgemvi")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,const double *,const double *,int,int,const double *,const int *,const double *,double *,hipsparseIndexBase_t,void *) nogil> _hipsparseDgemvi__funptr)(handle,transA,m,n,alpha,A,lda,nnz,x,xInd,beta,y,idxBase,pBuffer)


cdef void* _hipsparseCgemvi__funptr = NULL
cdef hipsparseStatus_t hipsparseCgemvi(void * handle,hipsparseOperation_t transA,int m,int n,float2 * alpha,float2 * A,int lda,int nnz,float2 * x,const int * xInd,float2 * beta,float2 * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil:
    global _hipsparseCgemvi__funptr
    __init_symbol(&_hipsparseCgemvi__funptr,"hipsparseCgemvi")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,float2 *,float2 *,int,int,float2 *,const int *,float2 *,float2 *,hipsparseIndexBase_t,void *) nogil> _hipsparseCgemvi__funptr)(handle,transA,m,n,alpha,A,lda,nnz,x,xInd,beta,y,idxBase,pBuffer)


cdef void* _hipsparseZgemvi__funptr = NULL
cdef hipsparseStatus_t hipsparseZgemvi(void * handle,hipsparseOperation_t transA,int m,int n,double2 * alpha,double2 * A,int lda,int nnz,double2 * x,const int * xInd,double2 * beta,double2 * y,hipsparseIndexBase_t idxBase,void * pBuffer) nogil:
    global _hipsparseZgemvi__funptr
    __init_symbol(&_hipsparseZgemvi__funptr,"hipsparseZgemvi")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,double2 *,double2 *,int,int,double2 *,const int *,double2 *,double2 *,hipsparseIndexBase_t,void *) nogil> _hipsparseZgemvi__funptr)(handle,transA,m,n,alpha,A,lda,nnz,x,xInd,beta,y,idxBase,pBuffer)


cdef void* _hipsparseSbsrmm__funptr = NULL
#  \ingroup level3_module
#  \brief Sparse matrix dense matrix multiplication using BSR storage format
# 
#  \details
#  \p hipsparseXbsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
#  matrix \f$A\f$, defined in BSR storage format, and the dense \f$k \times n\f$
#  matrix \f$B\f$ (where \f$k = block\_dim \times kb\f$) and adds the result to the dense
#  \f$m \times n\f$ matrix \f$C\f$ (where \f$m = block\_dim \times mb\f$) that
#  is multiplied by the scalar \f$\beta\f$, such that
#  \f[
#    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
#  \f]
#  with
#  \f[
#    op(A) = \left\{
#    \begin{array}{ll}
#        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#    \end{array}
#    \right.
#  \f]
#  and
#  \f[
#    op(B) = \left\{
#    \begin{array}{ll}
#        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#    \end{array}
#    \right.
#  \f]
# 
#  \note
#  This function is non blocking and executed asynchronously with respect to the host.
#  It may return before the actual computation has finished.
# 
#  \note
#  Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,const float * alpha,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,const float * B,int ldb,const float * beta,float * C,int ldc) nogil:
    global _hipsparseSbsrmm__funptr
    __init_symbol(&_hipsparseSbsrmm__funptr,"hipsparseSbsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,const float *,void *const,const float *,const int *,const int *,int,const float *,int,const float *,float *,int) nogil> _hipsparseSbsrmm__funptr)(handle,dirA,transA,transB,mb,n,kb,nnzb,alpha,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,B,ldb,beta,C,ldc)


cdef void* _hipsparseDbsrmm__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,const double * alpha,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,const double * B,int ldb,const double * beta,double * C,int ldc) nogil:
    global _hipsparseDbsrmm__funptr
    __init_symbol(&_hipsparseDbsrmm__funptr,"hipsparseDbsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,const double *,void *const,const double *,const int *,const int *,int,const double *,int,const double *,double *,int) nogil> _hipsparseDbsrmm__funptr)(handle,dirA,transA,transB,mb,n,kb,nnzb,alpha,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,B,ldb,beta,C,ldc)


cdef void* _hipsparseCbsrmm__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,float2 * alpha,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,float2 * B,int ldb,float2 * beta,float2 * C,int ldc) nogil:
    global _hipsparseCbsrmm__funptr
    __init_symbol(&_hipsparseCbsrmm__funptr,"hipsparseCbsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,float2 *,void *const,float2 *,const int *,const int *,int,float2 *,int,float2 *,float2 *,int) nogil> _hipsparseCbsrmm__funptr)(handle,dirA,transA,transB,mb,n,kb,nnzb,alpha,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,B,ldb,beta,C,ldc)


cdef void* _hipsparseZbsrmm__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrmm(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transB,int mb,int n,int kb,int nnzb,double2 * alpha,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,double2 * B,int ldb,double2 * beta,double2 * C,int ldc) nogil:
    global _hipsparseZbsrmm__funptr
    __init_symbol(&_hipsparseZbsrmm__funptr,"hipsparseZbsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,double2 *,void *const,double2 *,const int *,const int *,int,double2 *,int,double2 *,double2 *,int) nogil> _hipsparseZbsrmm__funptr)(handle,dirA,transA,transB,mb,n,kb,nnzb,alpha,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,B,ldb,beta,C,ldc)


cdef void* _hipsparseScsrmm__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse matrix dense matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
#   matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
#   matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
#   is multiplied by the scalar \f$\beta\f$, such that
#   \f[
#     C := \alpha \cdot op(A) \cdot B + \beta \cdot C,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < ldc; ++i)
#       {
#           for(j = 0; j < n; ++j)
#           {
#               C[i][j] = beta * C[i][j];
# 
#               for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
#               {
#                   C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
#               }
#           }
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,const float * beta,float * C,int ldc) nogil:
    global _hipsparseScsrmm__funptr
    __init_symbol(&_hipsparseScsrmm__funptr,"hipsparseScsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int,const float *,void *const,const float *,const int *,const int *,const float *,int,const float *,float *,int) nogil> _hipsparseScsrmm__funptr)(handle,transA,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseDcsrmm__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,const double * beta,double * C,int ldc) nogil:
    global _hipsparseDcsrmm__funptr
    __init_symbol(&_hipsparseDcsrmm__funptr,"hipsparseDcsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int,const double *,void *const,const double *,const int *,const int *,const double *,int,const double *,double *,int) nogil> _hipsparseDcsrmm__funptr)(handle,transA,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseCcsrmm__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,float2 * beta,float2 * C,int ldc) nogil:
    global _hipsparseCcsrmm__funptr
    __init_symbol(&_hipsparseCcsrmm__funptr,"hipsparseCcsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int,float2 *,void *const,float2 *,const int *,const int *,float2 *,int,float2 *,float2 *,int) nogil> _hipsparseCcsrmm__funptr)(handle,transA,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseZcsrmm__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrmm(void * handle,hipsparseOperation_t transA,int m,int n,int k,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,double2 * beta,double2 * C,int ldc) nogil:
    global _hipsparseZcsrmm__funptr
    __init_symbol(&_hipsparseZcsrmm__funptr,"hipsparseZcsrmm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,int,int,int,int,double2 *,void *const,double2 *,const int *,const int *,double2 *,int,double2 *,double2 *,int) nogil> _hipsparseZcsrmm__funptr)(handle,transA,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseScsrmm2__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse matrix dense matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
#   matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
#   matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
#   is multiplied by the scalar \f$\beta\f$, such that
#   \f[
#     C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
#   and
#   \f[
#     op(B) = \left\{
#     \begin{array}{ll}
#         B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \code{.c}
#       for(i = 0; i < ldc; ++i)
#       {
#           for(j = 0; j < n; ++j)
#           {
#               C[i][j] = beta * C[i][j];
# 
#               for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
#               {
#                   C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
#               }
#           }
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,const float * beta,float * C,int ldc) nogil:
    global _hipsparseScsrmm2__funptr
    __init_symbol(&_hipsparseScsrmm2__funptr,"hipsparseScsrmm2")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,const float *,void *const,const float *,const int *,const int *,const float *,int,const float *,float *,int) nogil> _hipsparseScsrmm2__funptr)(handle,transA,transB,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseDcsrmm2__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,const double * beta,double * C,int ldc) nogil:
    global _hipsparseDcsrmm2__funptr
    __init_symbol(&_hipsparseDcsrmm2__funptr,"hipsparseDcsrmm2")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,const double *,void *const,const double *,const int *,const int *,const double *,int,const double *,double *,int) nogil> _hipsparseDcsrmm2__funptr)(handle,transA,transB,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseCcsrmm2__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,float2 * beta,float2 * C,int ldc) nogil:
    global _hipsparseCcsrmm2__funptr
    __init_symbol(&_hipsparseCcsrmm2__funptr,"hipsparseCcsrmm2")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,float2 *,void *const,float2 *,const int *,const int *,float2 *,int,float2 *,float2 *,int) nogil> _hipsparseCcsrmm2__funptr)(handle,transA,transB,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseZcsrmm2__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrmm2(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,double2 * beta,double2 * C,int ldc) nogil:
    global _hipsparseZcsrmm2__funptr
    __init_symbol(&_hipsparseZcsrmm2__funptr,"hipsparseZcsrmm2")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,int,double2 *,void *const,double2 *,const int *,const int *,double2 *,int,double2 *,double2 *,int) nogil> _hipsparseZcsrmm2__funptr)(handle,transA,transB,m,n,k,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,beta,C,ldc)


cdef void* _hipsparseXbsrsm2_zeroPivot__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
#   structural or numerical zero has been found during hipsparseXbsrsm2_analysis() or
#   hipsparseXbsrsm2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
#   is stored in \p position, using same index base as the BSR matrix.
# 
#   \p position can be in host or device memory. If no zero pivot has been found,
#   \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
#   \note \p hipsparseXbsrsm2_zeroPivot is a blocking function. It might influence
#   performance negatively.
# /
cdef hipsparseStatus_t hipsparseXbsrsm2_zeroPivot(void * handle,bsrsm2Info_t info,int * position) nogil:
    global _hipsparseXbsrsm2_zeroPivot__funptr
    __init_symbol(&_hipsparseXbsrsm2_zeroPivot__funptr,"hipsparseXbsrsm2_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,bsrsm2Info_t,int *) nogil> _hipsparseXbsrsm2_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseSbsrsm2_bufferSize__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsm2_buffer_size returns the size of the temporary storage buffer that
#   is required by hipsparseXbsrsm2_analysis() and hipsparseXbsrsm2_solve(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseSbsrsm2_bufferSize__funptr
    __init_symbol(&_hipsparseSbsrsm2_bufferSize__funptr,"hipsparseSbsrsm2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,float *,const int *,const int *,int,bsrsm2Info_t,int *) nogil> _hipsparseSbsrsm2_bufferSize__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseDbsrsm2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDbsrsm2_bufferSize__funptr
    __init_symbol(&_hipsparseDbsrsm2_bufferSize__funptr,"hipsparseDbsrsm2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,double *,const int *,const int *,int,bsrsm2Info_t,int *) nogil> _hipsparseDbsrsm2_bufferSize__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseCbsrsm2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCbsrsm2_bufferSize__funptr
    __init_symbol(&_hipsparseCbsrsm2_bufferSize__funptr,"hipsparseCbsrsm2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,float2 *,const int *,const int *,int,bsrsm2Info_t,int *) nogil> _hipsparseCbsrsm2_bufferSize__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseZbsrsm2_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsm2_bufferSize(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZbsrsm2_bufferSize__funptr
    __init_symbol(&_hipsparseZbsrsm2_bufferSize__funptr,"hipsparseZbsrsm2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,double2 *,const int *,const int *,int,bsrsm2Info_t,int *) nogil> _hipsparseZbsrsm2_bufferSize__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseSbsrsm2_analysis__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsm2_analysis performs the analysis step for hipsparseXbsrsm2_solve().
# 
#   \note
#   If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsrsm2_analysis__funptr
    __init_symbol(&_hipsparseSbsrsm2_analysis__funptr,"hipsparseSbsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,const float *,const int *,const int *,int,bsrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsrsm2_analysis__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseDbsrsm2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsrsm2_analysis__funptr
    __init_symbol(&_hipsparseDbsrsm2_analysis__funptr,"hipsparseDbsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,const double *,const int *,const int *,int,bsrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsrsm2_analysis__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseCbsrsm2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsrsm2_analysis__funptr
    __init_symbol(&_hipsparseCbsrsm2_analysis__funptr,"hipsparseCbsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,float2 *,const int *,const int *,int,bsrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsrsm2_analysis__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseZbsrsm2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsm2_analysis(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsrsm2_analysis__funptr
    __init_symbol(&_hipsparseZbsrsm2_analysis__funptr,"hipsparseZbsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,double2 *,const int *,const int *,int,bsrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsrsm2_analysis__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseSbsrsm2_solve__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using BSR storage format
# 
#   \details
#   \p hipsparseXbsrsm2_solve solves a sparse triangular linear system of a sparse
#   \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution matrix
#   \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
#   \f[
#     op(A) \cdot op(X) = \alpha \cdot op(B),
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
#   ,
#   \f[
#     op(X) = \left\{
#     \begin{array}{ll}
#         X,   & \text{if trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         X^T, & \text{if trans_X == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         X^H, & \text{if trans_X == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \p hipsparseXbsrsm2_solve requires a user allocated temporary buffer. Its size is
#   returned by hipsparseXbsrsm2_bufferSize(). Furthermore, analysis meta data is
#   required. It can be obtained by hipsparseXbsrsm2_analysis(). \p hipsparseXbsrsm2_solve
#   reports the first zero pivot (either numerical or structural zero). The zero pivot
#   status can be checked calling hipsparseXbsrsm2_zeroPivot(). If
#   \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
#   reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
#   \note
#   The sparse BSR matrix has to be sorted.
# 
#   \note
#   Operation type of B and X must match, if \f$op(B)=B, op(X)=X\f$.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans_A != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
#   \p trans_X != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,const float * alpha,void *const descrA,const float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,const float * B,int ldb,float * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsrsm2_solve__funptr
    __init_symbol(&_hipsparseSbsrsm2_solve__funptr,"hipsparseSbsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const float *,void *const,const float *,const int *,const int *,int,bsrsm2Info_t,const float *,int,float *,int,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsrsm2_solve__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,B,ldb,X,ldx,policy,pBuffer)


cdef void* _hipsparseDbsrsm2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,const double * alpha,void *const descrA,const double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,const double * B,int ldb,double * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsrsm2_solve__funptr
    __init_symbol(&_hipsparseDbsrsm2_solve__funptr,"hipsparseDbsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const double *,void *const,const double *,const int *,const int *,int,bsrsm2Info_t,const double *,int,double *,int,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsrsm2_solve__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,B,ldb,X,ldx,policy,pBuffer)


cdef void* _hipsparseCbsrsm2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,float2 * alpha,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,float2 * B,int ldb,float2 * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsrsm2_solve__funptr
    __init_symbol(&_hipsparseCbsrsm2_solve__funptr,"hipsparseCbsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,float2 *,void *const,float2 *,const int *,const int *,int,bsrsm2Info_t,float2 *,int,float2 *,int,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsrsm2_solve__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,B,ldb,X,ldx,policy,pBuffer)


cdef void* _hipsparseZbsrsm2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrsm2_solve(void * handle,hipsparseDirection_t dirA,hipsparseOperation_t transA,hipsparseOperation_t transX,int mb,int nrhs,int nnzb,double2 * alpha,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrsm2Info_t info,double2 * B,int ldb,double2 * X,int ldx,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsrsm2_solve__funptr
    __init_symbol(&_hipsparseZbsrsm2_solve__funptr,"hipsparseZbsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,hipsparseOperation_t,hipsparseOperation_t,int,int,int,double2 *,void *const,double2 *,const int *,const int *,int,bsrsm2Info_t,double2 *,int,double2 *,int,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsrsm2_solve__funptr)(handle,dirA,transA,transX,mb,nrhs,nnzb,alpha,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,B,ldb,X,ldx,policy,pBuffer)


cdef void* _hipsparseXcsrsm2_zeroPivot__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
#   structural or numerical zero has been found during hipsparseXcsrsm2_analysis() or
#   hipsparseXcsrsm2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
#   is stored in \p position, using same index base as the CSR matrix.
# 
#   \p position can be in host or device memory. If no zero pivot has been found,
#   \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
#   \note \p hipsparseXcsrsm2_zeroPivot is a blocking function. It might influence
#   performance negatively.
# /
cdef hipsparseStatus_t hipsparseXcsrsm2_zeroPivot(void * handle,csrsm2Info_t info,int * position) nogil:
    global _hipsparseXcsrsm2_zeroPivot__funptr
    __init_symbol(&_hipsparseXcsrsm2_zeroPivot__funptr,"hipsparseXcsrsm2_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,csrsm2Info_t,int *) nogil> _hipsparseXcsrsm2_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseScsrsm2_bufferSizeExt__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsm2_bufferSizeExt returns the size of the temporary storage buffer
#   that is required by hipsparseXcsrsm2_analysis() and hipsparseXcsrsm2_solve(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil:
    global _hipsparseScsrsm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsrsm2_bufferSizeExt__funptr,"hipsparseScsrsm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const float *,void *const,const float *,const int *,const int *,const float *,int,csrsm2Info_t,hipsparseSolvePolicy_t,unsigned long *) nogil> _hipsparseScsrsm2_bufferSizeExt__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBufferSize)


cdef void* _hipsparseDcsrsm2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil:
    global _hipsparseDcsrsm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsrsm2_bufferSizeExt__funptr,"hipsparseDcsrsm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const double *,void *const,const double *,const int *,const int *,const double *,int,csrsm2Info_t,hipsparseSolvePolicy_t,unsigned long *) nogil> _hipsparseDcsrsm2_bufferSizeExt__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBufferSize)


cdef void* _hipsparseCcsrsm2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil:
    global _hipsparseCcsrsm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsrsm2_bufferSizeExt__funptr,"hipsparseCcsrsm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,float2 *,void *const,float2 *,const int *,const int *,float2 *,int,csrsm2Info_t,hipsparseSolvePolicy_t,unsigned long *) nogil> _hipsparseCcsrsm2_bufferSizeExt__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBufferSize)


cdef void* _hipsparseZcsrsm2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,unsigned long * pBufferSize) nogil:
    global _hipsparseZcsrsm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsrsm2_bufferSizeExt__funptr,"hipsparseZcsrsm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,double2 *,void *const,double2 *,const int *,const int *,double2 *,int,csrsm2Info_t,hipsparseSolvePolicy_t,unsigned long *) nogil> _hipsparseZcsrsm2_bufferSizeExt__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBufferSize)


cdef void* _hipsparseScsrsm2_analysis__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsm2_analysis performs the analysis step for hipsparseXcsrsm2_solve().
# 
#   \note
#   If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsrsm2_analysis__funptr
    __init_symbol(&_hipsparseScsrsm2_analysis__funptr,"hipsparseScsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const float *,void *const,const float *,const int *,const int *,const float *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsrsm2_analysis__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseDcsrsm2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsrsm2_analysis__funptr
    __init_symbol(&_hipsparseDcsrsm2_analysis__funptr,"hipsparseDcsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const double *,void *const,const double *,const int *,const int *,const double *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsrsm2_analysis__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseCcsrsm2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsrsm2_analysis__funptr
    __init_symbol(&_hipsparseCcsrsm2_analysis__funptr,"hipsparseCcsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,float2 *,void *const,float2 *,const int *,const int *,float2 *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsrsm2_analysis__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseZcsrsm2_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsm2_analysis(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsrsm2_analysis__funptr
    __init_symbol(&_hipsparseZcsrsm2_analysis__funptr,"hipsparseZcsrsm2_analysis")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,double2 *,void *const,double2 *,const int *,const int *,double2 *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsrsm2_analysis__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseScsrsm2_solve__funptr = NULL
#    \ingroup level3_module
#   \brief Sparse triangular system solve using CSR storage format
# 
#   \details
#   \p hipsparseXcsrsm2_solve solves a sparse triangular linear system of a sparse
#   \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution matrix
#   \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
#   \f[
#     op(A) \cdot op(X) = \alpha \cdot op(B),
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
#   ,
#   \f[
#     op(B) = \left\{
#     \begin{array}{ll}
#         B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
#   and
#   \f[
#     op(X) = \left\{
#     \begin{array}{ll}
#         X,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         X^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         X^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \p hipsparseXcsrsm2_solve requires a user allocated temporary buffer. Its size is
#   returned by hipsparseXcsrsm2_bufferSizeExt(). Furthermore, analysis meta data is
#   required. It can be obtained by hipsparseXcsrsm2_analysis().
#   \p hipsparseXcsrsm2_solve reports the first zero pivot (either numerical or structural
#   zero). The zero pivot status can be checked calling hipsparseXcsrsm2_zeroPivot(). If
#   \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
#   reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
# 
#   \note
#   The sparse CSR matrix has to be sorted. This can be achieved by calling
#   hipsparseXcsrsort().
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Currently, only \p trans_A != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
#   \p trans_B != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const float * alpha,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsrsm2_solve__funptr
    __init_symbol(&_hipsparseScsrsm2_solve__funptr,"hipsparseScsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const float *,void *const,const float *,const int *,const int *,float *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsrsm2_solve__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseDcsrsm2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,const double * alpha,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsrsm2_solve__funptr
    __init_symbol(&_hipsparseDcsrsm2_solve__funptr,"hipsparseDcsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,const double *,void *const,const double *,const int *,const int *,double *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsrsm2_solve__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseCcsrsm2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,float2 * alpha,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsrsm2_solve__funptr
    __init_symbol(&_hipsparseCcsrsm2_solve__funptr,"hipsparseCcsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,float2 *,void *const,float2 *,const int *,const int *,float2 *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsrsm2_solve__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseZcsrsm2_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrsm2_solve(void * handle,int algo,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int nrhs,int nnz,double2 * alpha,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * B,int ldb,csrsm2Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsrsm2_solve__funptr
    __init_symbol(&_hipsparseZcsrsm2_solve__funptr,"hipsparseZcsrsm2_solve")
    return (<hipsparseStatus_t (*)(void *,int,hipsparseOperation_t,hipsparseOperation_t,int,int,int,double2 *,void *const,double2 *,const int *,const int *,double2 *,int,csrsm2Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsrsm2_solve__funptr)(handle,algo,transA,transB,m,nrhs,nnz,alpha,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,B,ldb,info,policy,pBuffer)


cdef void* _hipsparseSgemmi__funptr = NULL
#    \ingroup level3_module
#   \brief Dense matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXgemmi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times k\f$
#   matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
#   storage format and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
#   is multiplied by the scalar \f$\beta\f$, such that
#   \f[
#     C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
#   and
#   \f[
#     op(B) = \left\{
#     \begin{array}{ll}
#         B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgemmi(void * handle,int m,int n,int k,int nnz,const float * alpha,const float * A,int lda,const float * cscValB,const int * cscColPtrB,const int * cscRowIndB,const float * beta,float * C,int ldc) nogil:
    global _hipsparseSgemmi__funptr
    __init_symbol(&_hipsparseSgemmi__funptr,"hipsparseSgemmi")
    return (<hipsparseStatus_t (*)(void *,int,int,int,int,const float *,const float *,int,const float *,const int *,const int *,const float *,float *,int) nogil> _hipsparseSgemmi__funptr)(handle,m,n,k,nnz,alpha,A,lda,cscValB,cscColPtrB,cscRowIndB,beta,C,ldc)


cdef void* _hipsparseDgemmi__funptr = NULL
cdef hipsparseStatus_t hipsparseDgemmi(void * handle,int m,int n,int k,int nnz,const double * alpha,const double * A,int lda,const double * cscValB,const int * cscColPtrB,const int * cscRowIndB,const double * beta,double * C,int ldc) nogil:
    global _hipsparseDgemmi__funptr
    __init_symbol(&_hipsparseDgemmi__funptr,"hipsparseDgemmi")
    return (<hipsparseStatus_t (*)(void *,int,int,int,int,const double *,const double *,int,const double *,const int *,const int *,const double *,double *,int) nogil> _hipsparseDgemmi__funptr)(handle,m,n,k,nnz,alpha,A,lda,cscValB,cscColPtrB,cscRowIndB,beta,C,ldc)


cdef void* _hipsparseCgemmi__funptr = NULL
cdef hipsparseStatus_t hipsparseCgemmi(void * handle,int m,int n,int k,int nnz,float2 * alpha,float2 * A,int lda,float2 * cscValB,const int * cscColPtrB,const int * cscRowIndB,float2 * beta,float2 * C,int ldc) nogil:
    global _hipsparseCgemmi__funptr
    __init_symbol(&_hipsparseCgemmi__funptr,"hipsparseCgemmi")
    return (<hipsparseStatus_t (*)(void *,int,int,int,int,float2 *,float2 *,int,float2 *,const int *,const int *,float2 *,float2 *,int) nogil> _hipsparseCgemmi__funptr)(handle,m,n,k,nnz,alpha,A,lda,cscValB,cscColPtrB,cscRowIndB,beta,C,ldc)


cdef void* _hipsparseZgemmi__funptr = NULL
cdef hipsparseStatus_t hipsparseZgemmi(void * handle,int m,int n,int k,int nnz,double2 * alpha,double2 * A,int lda,double2 * cscValB,const int * cscColPtrB,const int * cscRowIndB,double2 * beta,double2 * C,int ldc) nogil:
    global _hipsparseZgemmi__funptr
    __init_symbol(&_hipsparseZgemmi__funptr,"hipsparseZgemmi")
    return (<hipsparseStatus_t (*)(void *,int,int,int,int,double2 *,double2 *,int,double2 *,const int *,const int *,double2 *,double2 *,int) nogil> _hipsparseZgemmi__funptr)(handle,m,n,k,nnz,alpha,A,lda,cscValB,cscColPtrB,cscRowIndB,beta,C,ldc)


cdef void* _hipsparseXcsrgeamNnz__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix addition using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgeamNnz computes the total CSR non-zero elements and the CSR row
#   offsets, that point to the start of every row of the sparse CSR matrix, of the
#   resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
#   size \p m + 1.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
#   \note
#   Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# /
cdef hipsparseStatus_t hipsparseXcsrgeamNnz(void * handle,int m,int n,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr) nogil:
    global _hipsparseXcsrgeamNnz__funptr
    __init_symbol(&_hipsparseXcsrgeamNnz__funptr,"hipsparseXcsrgeamNnz")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,int,const int *,const int *,void *const,int,const int *,const int *,void *const,int *,int *) nogil> _hipsparseXcsrgeamNnz__funptr)(handle,m,n,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,descrC,csrRowPtrC,nnzTotalDevHostPtr)


cdef void* _hipsparseScsrgeam__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix addition using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
#   \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
#   scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
#   storage format, and adds both resulting matrices to obtain the sparse
#   \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
#   \f[
#     C := \alpha \cdot A + \beta \cdot B.
#   \f]
# 
#   It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
#   \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
#   \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
#   the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeamNnz().
# 
#   \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
#   \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
#   \note This function is non blocking and executed asynchronously with respect to the
#         host. It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrgeam(void * handle,int m,int n,const float * alpha,void *const descrA,int nnzA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * beta,void *const descrB,int nnzB,const float * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseScsrgeam__funptr
    __init_symbol(&_hipsparseScsrgeam__funptr,"hipsparseScsrgeam")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,void *const,int,const float *,const int *,const int *,const float *,void *const,int,const float *,const int *,const int *,void *const,float *,int *,int *) nogil> _hipsparseScsrgeam__funptr)(handle,m,n,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,beta,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseDcsrgeam__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrgeam(void * handle,int m,int n,const double * alpha,void *const descrA,int nnzA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * beta,void *const descrB,int nnzB,const double * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseDcsrgeam__funptr
    __init_symbol(&_hipsparseDcsrgeam__funptr,"hipsparseDcsrgeam")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,void *const,int,const double *,const int *,const int *,const double *,void *const,int,const double *,const int *,const int *,void *const,double *,int *,int *) nogil> _hipsparseDcsrgeam__funptr)(handle,m,n,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,beta,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseCcsrgeam__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrgeam(void * handle,int m,int n,float2 * alpha,void *const descrA,int nnzA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,float2 * beta,void *const descrB,int nnzB,float2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseCcsrgeam__funptr
    __init_symbol(&_hipsparseCcsrgeam__funptr,"hipsparseCcsrgeam")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,void *const,int,float2 *,const int *,const int *,float2 *,void *const,int,float2 *,const int *,const int *,void *const,float2 *,int *,int *) nogil> _hipsparseCcsrgeam__funptr)(handle,m,n,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,beta,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseZcsrgeam__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrgeam(void * handle,int m,int n,double2 * alpha,void *const descrA,int nnzA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,double2 * beta,void *const descrB,int nnzB,double2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseZcsrgeam__funptr
    __init_symbol(&_hipsparseZcsrgeam__funptr,"hipsparseZcsrgeam")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,void *const,int,double2 *,const int *,const int *,double2 *,void *const,int,double2 *,const int *,const int *,void *const,double2 *,int *,int *) nogil> _hipsparseZcsrgeam__funptr)(handle,m,n,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,beta,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseScsrgeam2_bufferSizeExt__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgeam2_bufferSizeExt returns the size of the temporary storage buffer
#   that is required by hipsparseXcsrgeam2Nnz() and hipsparseXcsrgeam2(). The temporary
#   storage buffer must be allocated by the user.
# 
#   \note
#   Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(void * handle,int m,int n,const float * alpha,void *const descrA,int nnzA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * beta,void *const descrB,int nnzB,const float * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,const float * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseScsrgeam2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsrgeam2_bufferSizeExt__funptr,"hipsparseScsrgeam2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,void *const,int,const float *,const int *,const int *,const float *,void *const,int,const float *,const int *,const int *,void *const,const float *,const int *,const int *,unsigned long *) nogil> _hipsparseScsrgeam2_bufferSizeExt__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBufferSizeInBytes)


cdef void* _hipsparseDcsrgeam2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(void * handle,int m,int n,const double * alpha,void *const descrA,int nnzA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * beta,void *const descrB,int nnzB,const double * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,const double * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDcsrgeam2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsrgeam2_bufferSizeExt__funptr,"hipsparseDcsrgeam2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,void *const,int,const double *,const int *,const int *,const double *,void *const,int,const double *,const int *,const int *,void *const,const double *,const int *,const int *,unsigned long *) nogil> _hipsparseDcsrgeam2_bufferSizeExt__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBufferSizeInBytes)


cdef void* _hipsparseCcsrgeam2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(void * handle,int m,int n,float2 * alpha,void *const descrA,int nnzA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * beta,void *const descrB,int nnzB,float2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,float2 * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCcsrgeam2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsrgeam2_bufferSizeExt__funptr,"hipsparseCcsrgeam2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,void *const,int,float2 *,const int *,const int *,float2 *,void *const,int,float2 *,const int *,const int *,void *const,float2 *,const int *,const int *,unsigned long *) nogil> _hipsparseCcsrgeam2_bufferSizeExt__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBufferSizeInBytes)


cdef void* _hipsparseZcsrgeam2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(void * handle,int m,int n,double2 * alpha,void *const descrA,int nnzA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * beta,void *const descrB,int nnzB,double2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,double2 * csrSortedValC,const int * csrSortedRowPtrC,const int * csrSortedColIndC,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZcsrgeam2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsrgeam2_bufferSizeExt__funptr,"hipsparseZcsrgeam2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,void *const,int,double2 *,const int *,const int *,double2 *,void *const,int,double2 *,const int *,const int *,void *const,double2 *,const int *,const int *,unsigned long *) nogil> _hipsparseZcsrgeam2_bufferSizeExt__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBufferSizeInBytes)


cdef void* _hipsparseXcsrgeam2Nnz__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix addition using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgeam2Nnz computes the total CSR non-zero elements and the CSR row
#   offsets, that point to the start of every row of the sparse CSR matrix, of the
#   resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
#   size \p m + 1.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
#   \note
#   Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# /
cdef hipsparseStatus_t hipsparseXcsrgeam2Nnz(void * handle,int m,int n,void *const descrA,int nnzA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void *const descrB,int nnzB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,int * csrSortedRowPtrC,int * nnzTotalDevHostPtr,void * workspace) nogil:
    global _hipsparseXcsrgeam2Nnz__funptr
    __init_symbol(&_hipsparseXcsrgeam2Nnz__funptr,"hipsparseXcsrgeam2Nnz")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,int,const int *,const int *,void *const,int,const int *,const int *,void *const,int *,int *,void *) nogil> _hipsparseXcsrgeam2Nnz__funptr)(handle,m,n,descrA,nnzA,csrSortedRowPtrA,csrSortedColIndA,descrB,nnzB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedRowPtrC,nnzTotalDevHostPtr,workspace)


cdef void* _hipsparseScsrgeam2__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix addition using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgeam2 multiplies the scalar \f$\alpha\f$ with the sparse
#   \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
#   scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
#   storage format, and adds both resulting matrices to obtain the sparse
#   \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
#   \f[
#     C := \alpha \cdot A + \beta \cdot B.
#   \f]
# 
#   It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
#   \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
#   \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
#   the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeam2Nnz().
# 
#   \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
#   \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
#   \note This function is non blocking and executed asynchronously with respect to the
#         host. It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrgeam2(void * handle,int m,int n,const float * alpha,void *const descrA,int nnzA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const float * beta,void *const descrB,int nnzB,const float * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,float * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil:
    global _hipsparseScsrgeam2__funptr
    __init_symbol(&_hipsparseScsrgeam2__funptr,"hipsparseScsrgeam2")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,void *const,int,const float *,const int *,const int *,const float *,void *const,int,const float *,const int *,const int *,void *const,float *,int *,int *,void *) nogil> _hipsparseScsrgeam2__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBuffer)


cdef void* _hipsparseDcsrgeam2__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrgeam2(void * handle,int m,int n,const double * alpha,void *const descrA,int nnzA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,const double * beta,void *const descrB,int nnzB,const double * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,double * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil:
    global _hipsparseDcsrgeam2__funptr
    __init_symbol(&_hipsparseDcsrgeam2__funptr,"hipsparseDcsrgeam2")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,void *const,int,const double *,const int *,const int *,const double *,void *const,int,const double *,const int *,const int *,void *const,double *,int *,int *,void *) nogil> _hipsparseDcsrgeam2__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBuffer)


cdef void* _hipsparseCcsrgeam2__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrgeam2(void * handle,int m,int n,float2 * alpha,void *const descrA,int nnzA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,float2 * beta,void *const descrB,int nnzB,float2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,float2 * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil:
    global _hipsparseCcsrgeam2__funptr
    __init_symbol(&_hipsparseCcsrgeam2__funptr,"hipsparseCcsrgeam2")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,void *const,int,float2 *,const int *,const int *,float2 *,void *const,int,float2 *,const int *,const int *,void *const,float2 *,int *,int *,void *) nogil> _hipsparseCcsrgeam2__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBuffer)


cdef void* _hipsparseZcsrgeam2__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrgeam2(void * handle,int m,int n,double2 * alpha,void *const descrA,int nnzA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,double2 * beta,void *const descrB,int nnzB,double2 * csrSortedValB,const int * csrSortedRowPtrB,const int * csrSortedColIndB,void *const descrC,double2 * csrSortedValC,int * csrSortedRowPtrC,int * csrSortedColIndC,void * pBuffer) nogil:
    global _hipsparseZcsrgeam2__funptr
    __init_symbol(&_hipsparseZcsrgeam2__funptr,"hipsparseZcsrgeam2")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,void *const,int,double2 *,const int *,const int *,double2 *,void *const,int,double2 *,const int *,const int *,void *const,double2 *,int *,int *,void *) nogil> _hipsparseZcsrgeam2__funptr)(handle,m,n,alpha,descrA,nnzA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,beta,descrB,nnzB,csrSortedValB,csrSortedRowPtrB,csrSortedColIndB,descrC,csrSortedValC,csrSortedRowPtrC,csrSortedColIndC,pBuffer)


cdef void* _hipsparseXcsrgemmNnz__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgemmNnz computes the total CSR non-zero elements and the CSR row
#   offsets, that point to the start of every row of the sparse CSR matrix, of the
#   resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
#   with size \p m + 1.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Please note, that for matrix products with more than 8192 intermediate products per
#   row, additional temporary storage buffer is allocated by the algorithm.
# 
#   \note
#   Currently, only \p trans_A == \p trans_B == \ref HIPSPARSE_OPERATION_NONE is
#   supported.
# 
#   \note
#   Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# /
cdef hipsparseStatus_t hipsparseXcsrgemmNnz(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr) nogil:
    global _hipsparseXcsrgemmNnz__funptr
    __init_symbol(&_hipsparseXcsrgemmNnz__funptr,"hipsparseXcsrgemmNnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,int,const int *,const int *,void *const,int,const int *,const int *,void *const,int *,int *) nogil> _hipsparseXcsrgemmNnz__funptr)(handle,transA,transB,m,n,k,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,descrC,csrRowPtrC,nnzTotalDevHostPtr)


cdef void* _hipsparseScsrgemm__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgemm multiplies the sparse \f$m \times k\f$ matrix \f$A\f$, defined in
#   CSR storage format with the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
#   storage format, and stores the result in the sparse \f$m \times n\f$ matrix \f$C\f$,
#   defined in CSR storage format, such that
#   \f[
#     C := op(A) \cdot op(B),
#   \f]
#   with
#   \f[
#     op(A) = \left\{
#     \begin{array}{ll}
#         A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
#   and
#   \f[
#     op(B) = \left\{
#     \begin{array}{ll}
#         B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
#         B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
#         B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
#     \end{array}
#     \right.
#   \f]
# 
#   It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
#   \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
#   \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
#   the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemmNnz().
# 
#   \note Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
#   \note Currently, only \p trans_B == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
#   \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
#   \note This function is non blocking and executed asynchronously with respect to the
#         host. It may return before the actual computation has finished.
#   \note Please note, that for matrix products with more than 4096 non-zero entries per
#   row, additional temporary storage buffer is allocated by the algorithm.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const float * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseScsrgemm__funptr
    __init_symbol(&_hipsparseScsrgemm__funptr,"hipsparseScsrgemm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,int,const float *,const int *,const int *,void *const,int,const float *,const int *,const int *,void *const,float *,const int *,int *) nogil> _hipsparseScsrgemm__funptr)(handle,transA,transB,m,n,k,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseDcsrgemm__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const double * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseDcsrgemm__funptr
    __init_symbol(&_hipsparseDcsrgemm__funptr,"hipsparseDcsrgemm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,int,const double *,const int *,const int *,void *const,int,const double *,const int *,const int *,void *const,double *,const int *,int *) nogil> _hipsparseDcsrgemm__funptr)(handle,transA,transB,m,n,k,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseCcsrgemm__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,float2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,float2 * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseCcsrgemm__funptr
    __init_symbol(&_hipsparseCcsrgemm__funptr,"hipsparseCcsrgemm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,int,float2 *,const int *,const int *,void *const,int,float2 *,const int *,const int *,void *const,float2 *,const int *,int *) nogil> _hipsparseCcsrgemm__funptr)(handle,transA,transB,m,n,k,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseZcsrgemm__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrgemm(void * handle,hipsparseOperation_t transA,hipsparseOperation_t transB,int m,int n,int k,void *const descrA,int nnzA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,double2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,void *const descrC,double2 * csrValC,const int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseZcsrgemm__funptr
    __init_symbol(&_hipsparseZcsrgemm__funptr,"hipsparseZcsrgemm")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,int,int,int,void *const,int,double2 *,const int *,const int *,void *const,int,double2 *,const int *,const int *,void *const,double2 *,const int *,int *) nogil> _hipsparseZcsrgemm__funptr)(handle,transA,transB,m,n,k,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseScsrgemm2_bufferSizeExt__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgemm2_bufferSizeExt returns the size of the temporary storage buffer
#   that is required by hipsparseXcsrgemm2Nnz() and hipsparseXcsrgemm2(). The temporary
#   storage buffer must be allocated by the user.
# 
#   \note
#   Please note, that for matrix products with more than 4096 non-zero entries per row,
#   additional temporary storage buffer is allocated by the algorithm.
# 
#   \note
#   Please note, that for matrix products with more than 8192 intermediate products per
#   row, additional temporary storage buffer is allocated by the algorithm.
# 
#   \note
#   Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,const float * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,const float * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseScsrgemm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsrgemm2_bufferSizeExt__funptr,"hipsparseScsrgemm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const float *,void *const,int,const int *,const int *,void *const,int,const int *,const int *,const float *,void *const,int,const int *,const int *,csrgemm2Info_t,unsigned long *) nogil> _hipsparseScsrgemm2_bufferSizeExt__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrRowPtrD,csrColIndD,info,pBufferSizeInBytes)


cdef void* _hipsparseDcsrgemm2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,const double * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,const double * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDcsrgemm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsrgemm2_bufferSizeExt__funptr,"hipsparseDcsrgemm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const double *,void *const,int,const int *,const int *,void *const,int,const int *,const int *,const double *,void *const,int,const int *,const int *,csrgemm2Info_t,unsigned long *) nogil> _hipsparseDcsrgemm2_bufferSizeExt__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrRowPtrD,csrColIndD,info,pBufferSizeInBytes)


cdef void* _hipsparseCcsrgemm2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,float2 * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,float2 * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCcsrgemm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsrgemm2_bufferSizeExt__funptr,"hipsparseCcsrgemm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float2 *,void *const,int,const int *,const int *,void *const,int,const int *,const int *,float2 *,void *const,int,const int *,const int *,csrgemm2Info_t,unsigned long *) nogil> _hipsparseCcsrgemm2_bufferSizeExt__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrRowPtrD,csrColIndD,info,pBufferSizeInBytes)


cdef void* _hipsparseZcsrgemm2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(void * handle,int m,int n,int k,double2 * alpha,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,double2 * beta,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,csrgemm2Info_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZcsrgemm2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsrgemm2_bufferSizeExt__funptr,"hipsparseZcsrgemm2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double2 *,void *const,int,const int *,const int *,void *const,int,const int *,const int *,double2 *,void *const,int,const int *,const int *,csrgemm2Info_t,unsigned long *) nogil> _hipsparseZcsrgemm2_bufferSizeExt__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrRowPtrD,csrColIndD,info,pBufferSizeInBytes)


cdef void* _hipsparseXcsrgemm2Nnz__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgemm2Nnz computes the total CSR non-zero elements and the CSR row
#   offsets, that point to the start of every row of the sparse CSR matrix, of the
#   resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
#   with size \p m + 1.
#   The required buffer size can be obtained by hipsparseXcsrgemm2_bufferSizeExt().
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# 
#   \note
#   Please note, that for matrix products with more than 8192 intermediate products per
#   row, additional temporary storage buffer is allocated by the algorithm.
# 
#   \note
#   Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
# /
cdef hipsparseStatus_t hipsparseXcsrgemm2Nnz(void * handle,int m,int n,int k,void *const descrA,int nnzA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const int * csrRowPtrB,const int * csrColIndB,void *const descrD,int nnzD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,csrgemm2Info_t info,void * pBuffer) nogil:
    global _hipsparseXcsrgemm2Nnz__funptr
    __init_symbol(&_hipsparseXcsrgemm2Nnz__funptr,"hipsparseXcsrgemm2Nnz")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,int,const int *,const int *,void *const,int,const int *,const int *,void *const,int,const int *,const int *,void *const,int *,int *,csrgemm2Info_t,void *) nogil> _hipsparseXcsrgemm2Nnz__funptr)(handle,m,n,k,descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,descrD,nnzD,csrRowPtrD,csrColIndD,descrC,csrRowPtrC,nnzTotalDevHostPtr,info,pBuffer)


cdef void* _hipsparseScsrgemm2__funptr = NULL
#    \ingroup extra_module
#   \brief Sparse matrix sparse matrix multiplication using CSR storage format
# 
#   \details
#   \p hipsparseXcsrgemm2 multiplies the scalar \f$\alpha\f$ with the sparse
#   \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
#   \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
#   to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
#   final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
#   storage format, such
#   that
#   \f[
#     C := \alpha \cdot A \cdot B + \beta \cdot D
#   \f]
# 
#   It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
#   \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
#   \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
#   the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemm2Nnz(). The
#   required buffer size for the computation can be obtained by
#   hipsparseXcsrgemm2_bufferSizeExt().
# 
#   \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
#   \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot A \cdot B\f$ will be computed.
#   \note \f$\alpha == beta == 0\f$ is invalid.
#   \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
#   \note This function is non blocking and executed asynchronously with respect to the
#         host. It may return before the actual computation has finished.
#   \note Please note, that for matrix products with more than 4096 non-zero entries per
#   row, additional temporary storage buffer is allocated by the algorithm.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrgemm2(void * handle,int m,int n,int k,const float * alpha,void *const descrA,int nnzA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const float * csrValB,const int * csrRowPtrB,const int * csrColIndB,const float * beta,void *const descrD,int nnzD,const float * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil:
    global _hipsparseScsrgemm2__funptr
    __init_symbol(&_hipsparseScsrgemm2__funptr,"hipsparseScsrgemm2")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const float *,void *const,int,const float *,const int *,const int *,void *const,int,const float *,const int *,const int *,const float *,void *const,int,const float *,const int *,const int *,void *const,float *,const int *,int *,csrgemm2Info_t,void *) nogil> _hipsparseScsrgemm2__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrValD,csrRowPtrD,csrColIndD,descrC,csrValC,csrRowPtrC,csrColIndC,info,pBuffer)


cdef void* _hipsparseDcsrgemm2__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrgemm2(void * handle,int m,int n,int k,const double * alpha,void *const descrA,int nnzA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,const double * csrValB,const int * csrRowPtrB,const int * csrColIndB,const double * beta,void *const descrD,int nnzD,const double * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil:
    global _hipsparseDcsrgemm2__funptr
    __init_symbol(&_hipsparseDcsrgemm2__funptr,"hipsparseDcsrgemm2")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const double *,void *const,int,const double *,const int *,const int *,void *const,int,const double *,const int *,const int *,const double *,void *const,int,const double *,const int *,const int *,void *const,double *,const int *,int *,csrgemm2Info_t,void *) nogil> _hipsparseDcsrgemm2__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrValD,csrRowPtrD,csrColIndD,descrC,csrValC,csrRowPtrC,csrColIndC,info,pBuffer)


cdef void* _hipsparseCcsrgemm2__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrgemm2(void * handle,int m,int n,int k,float2 * alpha,void *const descrA,int nnzA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,float2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,float2 * beta,void *const descrD,int nnzD,float2 * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,float2 * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil:
    global _hipsparseCcsrgemm2__funptr
    __init_symbol(&_hipsparseCcsrgemm2__funptr,"hipsparseCcsrgemm2")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float2 *,void *const,int,float2 *,const int *,const int *,void *const,int,float2 *,const int *,const int *,float2 *,void *const,int,float2 *,const int *,const int *,void *const,float2 *,const int *,int *,csrgemm2Info_t,void *) nogil> _hipsparseCcsrgemm2__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrValD,csrRowPtrD,csrColIndD,descrC,csrValC,csrRowPtrC,csrColIndC,info,pBuffer)


cdef void* _hipsparseZcsrgemm2__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrgemm2(void * handle,int m,int n,int k,double2 * alpha,void *const descrA,int nnzA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,void *const descrB,int nnzB,double2 * csrValB,const int * csrRowPtrB,const int * csrColIndB,double2 * beta,void *const descrD,int nnzD,double2 * csrValD,const int * csrRowPtrD,const int * csrColIndD,void *const descrC,double2 * csrValC,const int * csrRowPtrC,int * csrColIndC,csrgemm2Info_t info,void * pBuffer) nogil:
    global _hipsparseZcsrgemm2__funptr
    __init_symbol(&_hipsparseZcsrgemm2__funptr,"hipsparseZcsrgemm2")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double2 *,void *const,int,double2 *,const int *,const int *,void *const,int,double2 *,const int *,const int *,double2 *,void *const,int,double2 *,const int *,const int *,void *const,double2 *,const int *,int *,csrgemm2Info_t,void *) nogil> _hipsparseZcsrgemm2__funptr)(handle,m,n,k,alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,beta,descrD,nnzD,csrValD,csrRowPtrD,csrColIndD,descrC,csrValC,csrRowPtrC,csrColIndC,info,pBuffer)


cdef void* _hipsparseXbsrilu02_zeroPivot__funptr = NULL
# \ingroup precond_module
# \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
# format
# 
# \details
# \p hipsparseXbsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXbsrilu02_analysis() or
# hipsparseXbsrilu02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
# stored in \p position, using same index base as the BSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note
# If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
# \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
# invertible (numerical zero).
# 
# \note \p hipsparseXbsrilu02_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXbsrilu02_zeroPivot(void * handle,bsrilu02Info_t info,int * position) nogil:
    global _hipsparseXbsrilu02_zeroPivot__funptr
    __init_symbol(&_hipsparseXbsrilu02_zeroPivot__funptr,"hipsparseXbsrilu02_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,bsrilu02Info_t,int *) nogil> _hipsparseXbsrilu02_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseSbsrilu02_numericBoost__funptr = NULL
#    \ingroup precond_module
#    \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
#    format
# 
#    \details
#    \p hipsparseXbsrilu02_numericBoost enables the user to replace a numerical value in
#    an incomplete LU factorization. \p tol is used to determine whether a numerical value
#    is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
#    \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
# 
#    \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
#    setting \p enable_boost to 0.
# 
#    \note \p tol and \p boost_val can be in host or device memory.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,float * boost_val) nogil:
    global _hipsparseSbsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseSbsrilu02_numericBoost__funptr,"hipsparseSbsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,bsrilu02Info_t,int,double *,float *) nogil> _hipsparseSbsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseDbsrilu02_numericBoost__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,double * boost_val) nogil:
    global _hipsparseDbsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseDbsrilu02_numericBoost__funptr,"hipsparseDbsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,bsrilu02Info_t,int,double *,double *) nogil> _hipsparseDbsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseCbsrilu02_numericBoost__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,float2 * boost_val) nogil:
    global _hipsparseCbsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseCbsrilu02_numericBoost__funptr,"hipsparseCbsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,bsrilu02Info_t,int,double *,float2 *) nogil> _hipsparseCbsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseZbsrilu02_numericBoost__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrilu02_numericBoost(void * handle,bsrilu02Info_t info,int enable_boost,double * tol,double2 * boost_val) nogil:
    global _hipsparseZbsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseZbsrilu02_numericBoost__funptr,"hipsparseZbsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,bsrilu02Info_t,int,double *,double2 *) nogil> _hipsparseZbsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseSbsrilu02_bufferSize__funptr = NULL
#    \ingroup precond_module
#    \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
#    format
# 
#    \details
#    \p hipsparseXbsrilu02_bufferSize returns the size of the temporary storage buffer
#    that is required by hipsparseXbsrilu02_analysis() and hipsparseXbsrilu02_solve().
#    The temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseSbsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseSbsrilu02_bufferSize__funptr,"hipsparseSbsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float *,const int *,const int *,int,bsrilu02Info_t,int *) nogil> _hipsparseSbsrilu02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseDbsrilu02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDbsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseDbsrilu02_bufferSize__funptr,"hipsparseDbsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double *,const int *,const int *,int,bsrilu02Info_t,int *) nogil> _hipsparseDbsrilu02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseCbsrilu02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCbsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseCbsrilu02_bufferSize__funptr,"hipsparseCbsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,bsrilu02Info_t,int *) nogil> _hipsparseCbsrilu02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseZbsrilu02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrilu02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZbsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseZbsrilu02_bufferSize__funptr,"hipsparseZbsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,bsrilu02Info_t,int *) nogil> _hipsparseZbsrilu02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseSbsrilu02_analysis__funptr = NULL
#  \ingroup precond_module
#  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
#  format
# 
#  \details
#  \p hipsparseXbsrilu02_analysis performs the analysis step for hipsparseXbsrilu02().
# 
#  \note
#  If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#  \note
#  This function is non blocking and executed asynchronously with respect to the host.
#  It may return before the actual computation has finished.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsrilu02_analysis__funptr
    __init_symbol(&_hipsparseSbsrilu02_analysis__funptr,"hipsparseSbsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsrilu02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseDbsrilu02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsrilu02_analysis__funptr
    __init_symbol(&_hipsparseDbsrilu02_analysis__funptr,"hipsparseDbsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsrilu02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseCbsrilu02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsrilu02_analysis__funptr
    __init_symbol(&_hipsparseCbsrilu02_analysis__funptr,"hipsparseCbsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsrilu02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseZbsrilu02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrilu02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsrilu02_analysis__funptr
    __init_symbol(&_hipsparseZbsrilu02_analysis__funptr,"hipsparseZbsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsrilu02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseSbsrilu02__funptr = NULL
#  \ingroup precond_module
#  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
#  format
# 
#  \details
#  \p hipsparseXbsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
#  pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
#  \f[
#    A \approx LU
#  \f]
# 
#  \p hipsparseXbsrilu02 requires a user allocated temporary buffer. Its size is
#  returned by hipsparseXbsrilu02_bufferSize(). Furthermore, analysis meta data is
#  required. It can be obtained by hipsparseXbsrilu02_analysis(). \p hipsparseXbsrilu02
#  reports the first zero pivot (either numerical or structural zero). The zero pivot
#  status can be obtained by calling hipsparseXbsrilu02_zeroPivot().
# 
#  \note
#  This function is non blocking and executed asynchronously with respect to the host.
#  It may return before the actual computation has finished.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsrilu02__funptr
    __init_symbol(&_hipsparseSbsrilu02__funptr,"hipsparseSbsrilu02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsrilu02__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA_valM,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseDbsrilu02__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsrilu02__funptr
    __init_symbol(&_hipsparseDbsrilu02__funptr,"hipsparseDbsrilu02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsrilu02__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA_valM,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseCbsrilu02__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsrilu02__funptr
    __init_symbol(&_hipsparseCbsrilu02__funptr,"hipsparseCbsrilu02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsrilu02__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA_valM,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseZbsrilu02__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsrilu02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrSortedValA_valM,const int * bsrSortedRowPtrA,const int * bsrSortedColIndA,int blockDim,bsrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsrilu02__funptr
    __init_symbol(&_hipsparseZbsrilu02__funptr,"hipsparseZbsrilu02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,bsrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsrilu02__funptr)(handle,dirA,mb,nnzb,descrA,bsrSortedValA_valM,bsrSortedRowPtrA,bsrSortedColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseXcsrilu02_zeroPivot__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
#   structural or numerical zero has been found during hipsparseXcsrilu02() computation.
#   The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
#   index base as the CSR matrix.
# 
#   \p position can be in host or device memory. If no zero pivot has been found,
#   \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
#   \note \p hipsparseXcsrilu02_zeroPivot is a blocking function. It might influence
#   performance negatively.
# /
cdef hipsparseStatus_t hipsparseXcsrilu02_zeroPivot(void * handle,csrilu02Info_t info,int * position) nogil:
    global _hipsparseXcsrilu02_zeroPivot__funptr
    __init_symbol(&_hipsparseXcsrilu02_zeroPivot__funptr,"hipsparseXcsrilu02_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,csrilu02Info_t,int *) nogil> _hipsparseXcsrilu02_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseScsrilu02_numericBoost__funptr = NULL
#  \ingroup precond_module
#  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
#  format
# 
#  \details
#  \p hipsparseXcsrilu02_numericBoost enables the user to replace a numerical value in
#  an incomplete LU factorization. \p tol is used to determine whether a numerical value
#  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
#  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
# 
#  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
#  setting \p enable_boost to 0.
# 
#  \note \p tol and \p boost_val can be in host or device memory.
# 
# @{*/
cdef hipsparseStatus_t hipsparseScsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,float * boost_val) nogil:
    global _hipsparseScsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseScsrilu02_numericBoost__funptr,"hipsparseScsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,csrilu02Info_t,int,double *,float *) nogil> _hipsparseScsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseDcsrilu02_numericBoost__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,double * boost_val) nogil:
    global _hipsparseDcsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseDcsrilu02_numericBoost__funptr,"hipsparseDcsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,csrilu02Info_t,int,double *,double *) nogil> _hipsparseDcsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseCcsrilu02_numericBoost__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,float2 * boost_val) nogil:
    global _hipsparseCcsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseCcsrilu02_numericBoost__funptr,"hipsparseCcsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,csrilu02Info_t,int,double *,float2 *) nogil> _hipsparseCcsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseZcsrilu02_numericBoost__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrilu02_numericBoost(void * handle,csrilu02Info_t info,int enable_boost,double * tol,double2 * boost_val) nogil:
    global _hipsparseZcsrilu02_numericBoost__funptr
    __init_symbol(&_hipsparseZcsrilu02_numericBoost__funptr,"hipsparseZcsrilu02_numericBoost")
    return (<hipsparseStatus_t (*)(void *,csrilu02Info_t,int,double *,double2 *) nogil> _hipsparseZcsrilu02_numericBoost__funptr)(handle,info,enable_boost,tol,boost_val)


cdef void* _hipsparseScsrilu02_bufferSize__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsrilu02_bufferSize returns the size of the temporary storage buffer
#   that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseScsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseScsrilu02_bufferSize__funptr,"hipsparseScsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float *,const int *,const int *,csrilu02Info_t,int *) nogil> _hipsparseScsrilu02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseDcsrilu02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDcsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseDcsrilu02_bufferSize__funptr,"hipsparseDcsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double *,const int *,const int *,csrilu02Info_t,int *) nogil> _hipsparseDcsrilu02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseCcsrilu02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCcsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseCcsrilu02_bufferSize__funptr,"hipsparseCcsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csrilu02Info_t,int *) nogil> _hipsparseCcsrilu02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseZcsrilu02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrilu02_bufferSize(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZcsrilu02_bufferSize__funptr
    __init_symbol(&_hipsparseZcsrilu02_bufferSize__funptr,"hipsparseZcsrilu02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csrilu02Info_t,int *) nogil> _hipsparseZcsrilu02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseScsrilu02_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsrilu02_bufferSizeExt returns the size of the temporary storage buffer
#   that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseScsrilu02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsrilu02_bufferSizeExt__funptr,"hipsparseScsrilu02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float *,const int *,const int *,csrilu02Info_t,unsigned long *) nogil> _hipsparseScsrilu02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseDcsrilu02_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseDcsrilu02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsrilu02_bufferSizeExt__funptr,"hipsparseDcsrilu02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double *,const int *,const int *,csrilu02Info_t,unsigned long *) nogil> _hipsparseDcsrilu02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseCcsrilu02_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseCcsrilu02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsrilu02_bufferSizeExt__funptr,"hipsparseCcsrilu02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csrilu02Info_t,unsigned long *) nogil> _hipsparseCcsrilu02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseZcsrilu02_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseZcsrilu02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsrilu02_bufferSizeExt__funptr,"hipsparseZcsrilu02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csrilu02Info_t,unsigned long *) nogil> _hipsparseZcsrilu02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseScsrilu02_analysis__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsrilu02_analysis performs the analysis step for hipsparseXcsrilu02().
# 
#   \note
#   If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsrilu02_analysis__funptr
    __init_symbol(&_hipsparseScsrilu02_analysis__funptr,"hipsparseScsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsrilu02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseDcsrilu02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsrilu02_analysis__funptr
    __init_symbol(&_hipsparseDcsrilu02_analysis__funptr,"hipsparseDcsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsrilu02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseCcsrilu02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsrilu02_analysis__funptr
    __init_symbol(&_hipsparseCcsrilu02_analysis__funptr,"hipsparseCcsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsrilu02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseZcsrilu02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrilu02_analysis(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsrilu02_analysis__funptr
    __init_symbol(&_hipsparseZcsrilu02_analysis__funptr,"hipsparseZcsrilu02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsrilu02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseScsrilu02__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
#   pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
#   \f[
#     A \approx LU
#   \f]
# 
#   \p hipsparseXcsrilu02 requires a user allocated temporary buffer. Its size is returned
#   by hipsparseXcsrilu02_bufferSize() or hipsparseXcsrilu02_bufferSizeExt(). Furthermore,
#   analysis meta data is required. It can be obtained by hipsparseXcsrilu02_analysis().
#   \p hipsparseXcsrilu02 reports the first zero pivot (either numerical or structural
#   zero). The zero pivot status can be obtained by calling hipsparseXcsrilu02_zeroPivot().
# 
#   \note
#   The sparse CSR matrix has to be sorted. This can be achieved by calling
#   hipsparseXcsrsort().
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrilu02(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsrilu02__funptr
    __init_symbol(&_hipsparseScsrilu02__funptr,"hipsparseScsrilu02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsrilu02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseDcsrilu02__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrilu02(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsrilu02__funptr
    __init_symbol(&_hipsparseDcsrilu02__funptr,"hipsparseDcsrilu02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsrilu02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseCcsrilu02__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrilu02(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsrilu02__funptr
    __init_symbol(&_hipsparseCcsrilu02__funptr,"hipsparseCcsrilu02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsrilu02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseZcsrilu02__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrilu02(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csrilu02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsrilu02__funptr
    __init_symbol(&_hipsparseZcsrilu02__funptr,"hipsparseZcsrilu02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csrilu02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsrilu02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseXbsric02_zeroPivot__funptr = NULL
# \ingroup precond_module
# \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
# storage format
# 
# \details
# \p hipsparseXbsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
# structural or numerical zero has been found during hipsparseXbsric02_analysis() or
# hipsparseXbsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
# stored in \p position, using same index base as the BSR matrix.
# 
# \p position can be in host or device memory. If no zero pivot has been found,
# \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
# \note
# If a zero pivot is found, \p position=j means that either the diagonal block \p A(j,j)
# is missing (structural zero) or the diagonal block \p A(j,j) is not positive definite
# (numerical zero).
# 
# \note \p hipsparseXbsric02_zeroPivot is a blocking function. It might influence
# performance negatively.
cdef hipsparseStatus_t hipsparseXbsric02_zeroPivot(void * handle,bsric02Info_t info,int * position) nogil:
    global _hipsparseXbsric02_zeroPivot__funptr
    __init_symbol(&_hipsparseXbsric02_zeroPivot__funptr,"hipsparseXbsric02_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,bsric02Info_t,int *) nogil> _hipsparseXbsric02_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseSbsric02_bufferSize__funptr = NULL
#  \ingroup precond_module
#  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
#  storage format
# 
#  \details
#  \p hipsparseXbsric02_bufferSize returns the size of the temporary storage buffer
#  that is required by hipsparseXbsric02_analysis() and hipsparseXbsric02(). The
#  temporary storage buffer must be allocated by the user.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseSbsric02_bufferSize__funptr
    __init_symbol(&_hipsparseSbsric02_bufferSize__funptr,"hipsparseSbsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float *,const int *,const int *,int,bsric02Info_t,int *) nogil> _hipsparseSbsric02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseDbsric02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDbsric02_bufferSize__funptr
    __init_symbol(&_hipsparseDbsric02_bufferSize__funptr,"hipsparseDbsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double *,const int *,const int *,int,bsric02Info_t,int *) nogil> _hipsparseDbsric02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseCbsric02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCbsric02_bufferSize__funptr
    __init_symbol(&_hipsparseCbsric02_bufferSize__funptr,"hipsparseCbsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,bsric02Info_t,int *) nogil> _hipsparseCbsric02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseZbsric02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsric02_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZbsric02_bufferSize__funptr
    __init_symbol(&_hipsparseZbsric02_bufferSize__funptr,"hipsparseZbsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,bsric02Info_t,int *) nogil> _hipsparseZbsric02_bufferSize__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,pBufferSizeInBytes)


cdef void* _hipsparseSbsric02_analysis__funptr = NULL
#  \ingroup precond_module
#  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
#  storage format
# 
#  \details
#  \p hipsparseXbsric02_analysis performs the analysis step for hipsparseXbsric02().
# 
#  \note
#  If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#  \note
#  This function is non blocking and executed asynchronously with respect to the host.
#  It may return before the actual computation has finished.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsric02_analysis__funptr
    __init_symbol(&_hipsparseSbsric02_analysis__funptr,"hipsparseSbsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsric02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseDbsric02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsric02_analysis__funptr
    __init_symbol(&_hipsparseDbsric02_analysis__funptr,"hipsparseDbsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsric02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseCbsric02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsric02_analysis__funptr
    __init_symbol(&_hipsparseCbsric02_analysis__funptr,"hipsparseCbsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsric02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseZbsric02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsric02_analysis(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsric02_analysis__funptr
    __init_symbol(&_hipsparseZbsric02_analysis__funptr,"hipsparseZbsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsric02_analysis__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseSbsric02__funptr = NULL
#  \ingroup precond_module
#  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
#  storage format
# 
#  \details
#  \p hipsparseXbsric02 computes the incomplete Cholesky factorization with 0 fill-ins
#  and no pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
#  \f[
#    A \approx LL^T
#  \f]
# 
#  \p hipsparseXbsric02 requires a user allocated temporary buffer. Its size is returned
#  by hipsparseXbsric02_bufferSize(). Furthermore, analysis meta data is required. It
#  can be obtained by hipsparseXbsric02_analysis(). \p hipsparseXbsric02 reports the
#  first zero pivot (either numerical or structural zero). The zero pivot status can be
#  obtained by calling hipsparseXbsric02_zeroPivot().
# 
#  \note
#  This function is non blocking and executed asynchronously with respect to the host.
#  It may return before the actual computation has finished.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseSbsric02__funptr
    __init_symbol(&_hipsparseSbsric02__funptr,"hipsparseSbsric02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseSbsric02__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseDbsric02__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDbsric02__funptr
    __init_symbol(&_hipsparseDbsric02__funptr,"hipsparseDbsric02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDbsric02__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseCbsric02__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCbsric02__funptr
    __init_symbol(&_hipsparseCbsric02__funptr,"hipsparseCbsric02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCbsric02__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseZbsric02__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsric02(void * handle,hipsparseDirection_t dirA,int mb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,bsric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZbsric02__funptr
    __init_symbol(&_hipsparseZbsric02__funptr,"hipsparseZbsric02")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,bsric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZbsric02__funptr)(handle,dirA,mb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,info,policy,pBuffer)


cdef void* _hipsparseXcsric02_zeroPivot__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
#   structural or numerical zero has been found during hipsparseXcsric02_analysis() or
#   hipsparseXcsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
#   is stored in \p position, using same index base as the CSR matrix.
# 
#   \p position can be in host or device memory. If no zero pivot has been found,
#   \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
# 
#   \note \p hipsparseXcsric02_zeroPivot is a blocking function. It might influence
#   performance negatively.
# /
cdef hipsparseStatus_t hipsparseXcsric02_zeroPivot(void * handle,csric02Info_t info,int * position) nogil:
    global _hipsparseXcsric02_zeroPivot__funptr
    __init_symbol(&_hipsparseXcsric02_zeroPivot__funptr,"hipsparseXcsric02_zeroPivot")
    return (<hipsparseStatus_t (*)(void *,csric02Info_t,int *) nogil> _hipsparseXcsric02_zeroPivot__funptr)(handle,info,position)


cdef void* _hipsparseScsric02_bufferSize__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsric02_bufferSize returns the size of the temporary storage buffer
#   that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseScsric02_bufferSize__funptr
    __init_symbol(&_hipsparseScsric02_bufferSize__funptr,"hipsparseScsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float *,const int *,const int *,csric02Info_t,int *) nogil> _hipsparseScsric02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseDcsric02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseDcsric02_bufferSize__funptr
    __init_symbol(&_hipsparseDcsric02_bufferSize__funptr,"hipsparseDcsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double *,const int *,const int *,csric02Info_t,int *) nogil> _hipsparseDcsric02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseCcsric02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseCcsric02_bufferSize__funptr
    __init_symbol(&_hipsparseCcsric02_bufferSize__funptr,"hipsparseCcsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csric02Info_t,int *) nogil> _hipsparseCcsric02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseZcsric02_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsric02_bufferSize(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,int * pBufferSizeInBytes) nogil:
    global _hipsparseZcsric02_bufferSize__funptr
    __init_symbol(&_hipsparseZcsric02_bufferSize__funptr,"hipsparseZcsric02_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csric02Info_t,int *) nogil> _hipsparseZcsric02_bufferSize__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSizeInBytes)


cdef void* _hipsparseScsric02_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsric02_bufferSizeExt returns the size of the temporary storage buffer
#   that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseScsric02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsric02_bufferSizeExt__funptr,"hipsparseScsric02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float *,const int *,const int *,csric02Info_t,unsigned long *) nogil> _hipsparseScsric02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseDcsric02_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseDcsric02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsric02_bufferSizeExt__funptr,"hipsparseDcsric02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double *,const int *,const int *,csric02Info_t,unsigned long *) nogil> _hipsparseDcsric02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseCcsric02_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseCcsric02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsric02_bufferSizeExt__funptr,"hipsparseCcsric02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csric02Info_t,unsigned long *) nogil> _hipsparseCcsric02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseZcsric02_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,unsigned long * pBufferSize) nogil:
    global _hipsparseZcsric02_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsric02_bufferSizeExt__funptr,"hipsparseZcsric02_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csric02Info_t,unsigned long *) nogil> _hipsparseZcsric02_bufferSizeExt__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,pBufferSize)


cdef void* _hipsparseScsric02_analysis__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsric02_analysis performs the analysis step for hipsparseXcsric02().
# 
#   \note
#   If the matrix sparsity pattern changes, the gathered information will become invalid.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsric02_analysis(void * handle,int m,int nnz,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsric02_analysis__funptr
    __init_symbol(&_hipsparseScsric02_analysis__funptr,"hipsparseScsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsric02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseDcsric02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsric02_analysis(void * handle,int m,int nnz,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsric02_analysis__funptr
    __init_symbol(&_hipsparseDcsric02_analysis__funptr,"hipsparseDcsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsric02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseCcsric02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsric02_analysis(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsric02_analysis__funptr
    __init_symbol(&_hipsparseCcsric02_analysis__funptr,"hipsparseCcsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsric02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseZcsric02_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsric02_analysis(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsric02_analysis__funptr
    __init_symbol(&_hipsparseZcsric02_analysis__funptr,"hipsparseZcsric02_analysis")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsric02_analysis__funptr)(handle,m,nnz,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseScsric02__funptr = NULL
#    \ingroup precond_module
#   \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
#   storage format
# 
#   \details
#   \p hipsparseXcsric02 computes the incomplete Cholesky factorization with 0 fill-ins
#   and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
#   \f[
#     A \approx LL^T
#   \f]
# 
#   \p hipsparseXcsric02 requires a user allocated temporary buffer. Its size is returned
#   by hipsparseXcsric02_bufferSize() or hipsparseXcsric02_bufferSizeExt(). Furthermore,
#   analysis meta data is required. It can be obtained by hipsparseXcsric02_analysis().
#   \p hipsparseXcsric02 reports the first zero pivot (either numerical or structural
#   zero). The zero pivot status can be obtained by calling hipsparseXcsric02_zeroPivot().
# 
#   \note
#   The sparse CSR matrix has to be sorted. This can be achieved by calling
#   hipsparseXcsrsort().
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsric02(void * handle,int m,int nnz,void *const descrA,float * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseScsric02__funptr
    __init_symbol(&_hipsparseScsric02__funptr,"hipsparseScsric02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseScsric02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseDcsric02__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsric02(void * handle,int m,int nnz,void *const descrA,double * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseDcsric02__funptr
    __init_symbol(&_hipsparseDcsric02__funptr,"hipsparseDcsric02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseDcsric02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseCcsric02__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsric02(void * handle,int m,int nnz,void *const descrA,float2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseCcsric02__funptr
    __init_symbol(&_hipsparseCcsric02__funptr,"hipsparseCcsric02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseCcsric02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseZcsric02__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsric02(void * handle,int m,int nnz,void *const descrA,double2 * csrSortedValA_valM,const int * csrSortedRowPtrA,const int * csrSortedColIndA,csric02Info_t info,hipsparseSolvePolicy_t policy,void * pBuffer) nogil:
    global _hipsparseZcsric02__funptr
    __init_symbol(&_hipsparseZcsric02__funptr,"hipsparseZcsric02")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,csric02Info_t,hipsparseSolvePolicy_t,void *) nogil> _hipsparseZcsric02__funptr)(handle,m,nnz,descrA,csrSortedValA_valM,csrSortedRowPtrA,csrSortedColIndA,info,policy,pBuffer)


cdef void* _hipsparseSgtsv2_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Tridiagonal solver with pivoting
# 
#   \details
#   \p hipsparseXgtsv2_bufferSize returns the size of the temporary storage buffer
#   that is required by hipsparseXgtsv2(). The temporary storage buffer must be
#   allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(void * handle,int m,int n,const float * dl,const float * d,const float * du,const float * B,int ldb,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseSgtsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSgtsv2_bufferSizeExt__funptr,"hipsparseSgtsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,const float *,const float *,const float *,int,unsigned long *) nogil> _hipsparseSgtsv2_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,ldb,pBufferSizeInBytes)


cdef void* _hipsparseDgtsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(void * handle,int m,int n,const double * dl,const double * d,const double * du,const double * B,int db,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDgtsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDgtsv2_bufferSizeExt__funptr,"hipsparseDgtsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,const double *,const double *,const double *,int,unsigned long *) nogil> _hipsparseDgtsv2_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,db,pBufferSizeInBytes)


cdef void* _hipsparseCgtsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCgtsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCgtsv2_bufferSizeExt__funptr,"hipsparseCgtsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,int,unsigned long *) nogil> _hipsparseCgtsv2_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,ldb,pBufferSizeInBytes)


cdef void* _hipsparseZgtsv2_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZgtsv2_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZgtsv2_bufferSizeExt__funptr,"hipsparseZgtsv2_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,int,unsigned long *) nogil> _hipsparseZgtsv2_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,ldb,pBufferSizeInBytes)


cdef void* _hipsparseSgtsv2__funptr = NULL
#    \ingroup precond_module
#   \brief Tridiagonal solver with pivoting
# 
#   \details
#   \p hipsparseXgtsv2 solves a tridiagonal system for multiple right hand sides using pivoting.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsv2(void * handle,int m,int n,const float * dl,const float * d,const float * du,float * B,int ldb,void * pBuffer) nogil:
    global _hipsparseSgtsv2__funptr
    __init_symbol(&_hipsparseSgtsv2__funptr,"hipsparseSgtsv2")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,const float *,const float *,float *,int,void *) nogil> _hipsparseSgtsv2__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseDgtsv2__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsv2(void * handle,int m,int n,const double * dl,const double * d,const double * du,double * B,int ldb,void * pBuffer) nogil:
    global _hipsparseDgtsv2__funptr
    __init_symbol(&_hipsparseDgtsv2__funptr,"hipsparseDgtsv2")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,const double *,const double *,double *,int,void *) nogil> _hipsparseDgtsv2__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseCgtsv2__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsv2(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,void * pBuffer) nogil:
    global _hipsparseCgtsv2__funptr
    __init_symbol(&_hipsparseCgtsv2__funptr,"hipsparseCgtsv2")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,int,void *) nogil> _hipsparseCgtsv2__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseZgtsv2__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsv2(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,void * pBuffer) nogil:
    global _hipsparseZgtsv2__funptr
    __init_symbol(&_hipsparseZgtsv2__funptr,"hipsparseZgtsv2")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,int,void *) nogil> _hipsparseZgtsv2__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseSgtsv2_nopivot_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Tridiagonal solver (no pivoting)
# 
#   \details
#   \p hipsparseXgtsv2_nopivot_bufferSizeExt returns the size of the temporary storage
#   buffer that is required by hipsparseXgtsv2_nopivot(). The temporary storage buffer
#   must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,const float * dl,const float * d,const float * du,const float * B,int ldb,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseSgtsv2_nopivot_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSgtsv2_nopivot_bufferSizeExt__funptr,"hipsparseSgtsv2_nopivot_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,const float *,const float *,const float *,int,unsigned long *) nogil> _hipsparseSgtsv2_nopivot_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,ldb,pBufferSizeInBytes)


cdef void* _hipsparseDgtsv2_nopivot_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,const double * dl,const double * d,const double * du,const double * B,int db,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDgtsv2_nopivot_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDgtsv2_nopivot_bufferSizeExt__funptr,"hipsparseDgtsv2_nopivot_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,const double *,const double *,const double *,int,unsigned long *) nogil> _hipsparseDgtsv2_nopivot_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,db,pBufferSizeInBytes)


cdef void* _hipsparseCgtsv2_nopivot_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCgtsv2_nopivot_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCgtsv2_nopivot_bufferSizeExt__funptr,"hipsparseCgtsv2_nopivot_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,int,unsigned long *) nogil> _hipsparseCgtsv2_nopivot_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,ldb,pBufferSizeInBytes)


cdef void* _hipsparseZgtsv2_nopivot_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZgtsv2_nopivot_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZgtsv2_nopivot_bufferSizeExt__funptr,"hipsparseZgtsv2_nopivot_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,int,unsigned long *) nogil> _hipsparseZgtsv2_nopivot_bufferSizeExt__funptr)(handle,m,n,dl,d,du,B,ldb,pBufferSizeInBytes)


cdef void* _hipsparseSgtsv2_nopivot__funptr = NULL
#    \ingroup precond_module
#   \brief Tridiagonal solver (no pivoting)
# 
#   \details
#   \p hipsparseXgtsv2_nopivot solves a tridiagonal linear system for multiple right-hand sides
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsv2_nopivot(void * handle,int m,int n,const float * dl,const float * d,const float * du,float * B,int ldb,void * pBuffer) nogil:
    global _hipsparseSgtsv2_nopivot__funptr
    __init_symbol(&_hipsparseSgtsv2_nopivot__funptr,"hipsparseSgtsv2_nopivot")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,const float *,const float *,float *,int,void *) nogil> _hipsparseSgtsv2_nopivot__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseDgtsv2_nopivot__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsv2_nopivot(void * handle,int m,int n,const double * dl,const double * d,const double * du,double * B,int ldb,void * pBuffer) nogil:
    global _hipsparseDgtsv2_nopivot__funptr
    __init_symbol(&_hipsparseDgtsv2_nopivot__funptr,"hipsparseDgtsv2_nopivot")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,const double *,const double *,double *,int,void *) nogil> _hipsparseDgtsv2_nopivot__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseCgtsv2_nopivot__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsv2_nopivot(void * handle,int m,int n,float2 * dl,float2 * d,float2 * du,float2 * B,int ldb,void * pBuffer) nogil:
    global _hipsparseCgtsv2_nopivot__funptr
    __init_symbol(&_hipsparseCgtsv2_nopivot__funptr,"hipsparseCgtsv2_nopivot")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,int,void *) nogil> _hipsparseCgtsv2_nopivot__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseZgtsv2_nopivot__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsv2_nopivot(void * handle,int m,int n,double2 * dl,double2 * d,double2 * du,double2 * B,int ldb,void * pBuffer) nogil:
    global _hipsparseZgtsv2_nopivot__funptr
    __init_symbol(&_hipsparseZgtsv2_nopivot__funptr,"hipsparseZgtsv2_nopivot")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,int,void *) nogil> _hipsparseZgtsv2_nopivot__funptr)(handle,m,n,dl,d,du,B,ldb,pBuffer)


cdef void* _hipsparseSgtsv2StridedBatch_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Strided Batch tridiagonal solver (no pivoting)
# 
#   \details
#   \p hipsparseXgtsv2StridedBatch_bufferSizeExt returns the size of the temporary storage
#   buffer that is required by hipsparseXgtsv2StridedBatch(). The temporary storage buffer
#   must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(void * handle,int m,const float * dl,const float * d,const float * du,const float * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseSgtsv2StridedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSgtsv2StridedBatch_bufferSizeExt__funptr,"hipsparseSgtsv2StridedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,const float *,const float *,const float *,const float *,int,int,unsigned long *) nogil> _hipsparseSgtsv2StridedBatch_bufferSizeExt__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBufferSizeInBytes)


cdef void* _hipsparseDgtsv2StridedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(void * handle,int m,const double * dl,const double * d,const double * du,const double * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDgtsv2StridedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDgtsv2StridedBatch_bufferSizeExt__funptr,"hipsparseDgtsv2StridedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,const double *,const double *,const double *,const double *,int,int,unsigned long *) nogil> _hipsparseDgtsv2StridedBatch_bufferSizeExt__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBufferSizeInBytes)


cdef void* _hipsparseCgtsv2StridedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(void * handle,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCgtsv2StridedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCgtsv2StridedBatch_bufferSizeExt__funptr,"hipsparseCgtsv2StridedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,float2 *,float2 *,float2 *,int,int,unsigned long *) nogil> _hipsparseCgtsv2StridedBatch_bufferSizeExt__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBufferSizeInBytes)


cdef void* _hipsparseZgtsv2StridedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(void * handle,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,int batchStride,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZgtsv2StridedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZgtsv2StridedBatch_bufferSizeExt__funptr,"hipsparseZgtsv2StridedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,double2 *,double2 *,double2 *,int,int,unsigned long *) nogil> _hipsparseZgtsv2StridedBatch_bufferSizeExt__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBufferSizeInBytes)


cdef void* _hipsparseSgtsv2StridedBatch__funptr = NULL
#    \ingroup precond_module
#   \brief Strided Batch tridiagonal solver (no pivoting)
# 
#   \details
#   \p hipsparseXgtsv2StridedBatch solves a batched tridiagonal linear system
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsv2StridedBatch(void * handle,int m,const float * dl,const float * d,const float * du,float * x,int batchCount,int batchStride,void * pBuffer) nogil:
    global _hipsparseSgtsv2StridedBatch__funptr
    __init_symbol(&_hipsparseSgtsv2StridedBatch__funptr,"hipsparseSgtsv2StridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,const float *,const float *,const float *,float *,int,int,void *) nogil> _hipsparseSgtsv2StridedBatch__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBuffer)


cdef void* _hipsparseDgtsv2StridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsv2StridedBatch(void * handle,int m,const double * dl,const double * d,const double * du,double * x,int batchCount,int batchStride,void * pBuffer) nogil:
    global _hipsparseDgtsv2StridedBatch__funptr
    __init_symbol(&_hipsparseDgtsv2StridedBatch__funptr,"hipsparseDgtsv2StridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,const double *,const double *,const double *,double *,int,int,void *) nogil> _hipsparseDgtsv2StridedBatch__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBuffer)


cdef void* _hipsparseCgtsv2StridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsv2StridedBatch(void * handle,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,int batchStride,void * pBuffer) nogil:
    global _hipsparseCgtsv2StridedBatch__funptr
    __init_symbol(&_hipsparseCgtsv2StridedBatch__funptr,"hipsparseCgtsv2StridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,float2 *,float2 *,float2 *,float2 *,int,int,void *) nogil> _hipsparseCgtsv2StridedBatch__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBuffer)


cdef void* _hipsparseZgtsv2StridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsv2StridedBatch(void * handle,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,int batchStride,void * pBuffer) nogil:
    global _hipsparseZgtsv2StridedBatch__funptr
    __init_symbol(&_hipsparseZgtsv2StridedBatch__funptr,"hipsparseZgtsv2StridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,double2 *,double2 *,double2 *,double2 *,int,int,void *) nogil> _hipsparseZgtsv2StridedBatch__funptr)(handle,m,dl,d,du,x,batchCount,batchStride,pBuffer)


cdef void* _hipsparseSgtsvInterleavedBatch_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Interleaved Batch tridiagonal solver
# 
#   \details
#   \p hipsparseXgtsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
#   buffer that is required by hipsparseXgtsvInterleavedBatch(). The temporary storage buffer
#   must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const float * dl,const float * d,const float * du,const float * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseSgtsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSgtsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseSgtsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,const float *,const float *,const float *,int,unsigned long *) nogil> _hipsparseSgtsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseDgtsvInterleavedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const double * dl,const double * d,const double * du,const double * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDgtsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDgtsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseDgtsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,const double *,const double *,const double *,int,unsigned long *) nogil> _hipsparseDgtsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseCgtsvInterleavedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCgtsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCgtsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseCgtsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,int,unsigned long *) nogil> _hipsparseCgtsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseZgtsvInterleavedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZgtsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZgtsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseZgtsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,int,unsigned long *) nogil> _hipsparseZgtsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseSgtsvInterleavedBatch__funptr = NULL
#    \ingroup precond_module
#   \brief Interleaved Batch tridiagonal solver
# 
#   \details
#   \p hipsparseXgtsvInterleavedBatch solves a batched tridiagonal linear system
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgtsvInterleavedBatch(void * handle,int algo,int m,float * dl,float * d,float * du,float * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseSgtsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseSgtsvInterleavedBatch__funptr,"hipsparseSgtsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,float *,float *,float *,float *,int,void *) nogil> _hipsparseSgtsvInterleavedBatch__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBuffer)


cdef void* _hipsparseDgtsvInterleavedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseDgtsvInterleavedBatch(void * handle,int algo,int m,double * dl,double * d,double * du,double * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseDgtsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseDgtsvInterleavedBatch__funptr,"hipsparseDgtsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,double *,double *,double *,double *,int,void *) nogil> _hipsparseDgtsvInterleavedBatch__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBuffer)


cdef void* _hipsparseCgtsvInterleavedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseCgtsvInterleavedBatch(void * handle,int algo,int m,float2 * dl,float2 * d,float2 * du,float2 * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseCgtsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseCgtsvInterleavedBatch__funptr,"hipsparseCgtsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,int,void *) nogil> _hipsparseCgtsvInterleavedBatch__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBuffer)


cdef void* _hipsparseZgtsvInterleavedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseZgtsvInterleavedBatch(void * handle,int algo,int m,double2 * dl,double2 * d,double2 * du,double2 * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseZgtsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseZgtsvInterleavedBatch__funptr,"hipsparseZgtsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,int,void *) nogil> _hipsparseZgtsvInterleavedBatch__funptr)(handle,algo,m,dl,d,du,x,batchCount,pBuffer)


cdef void* _hipsparseSgpsvInterleavedBatch_bufferSizeExt__funptr = NULL
#    \ingroup precond_module
#   \brief Interleaved Batch pentadiagonal solver
# 
#   \details
#   \p hipsparseXgpsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
#   buffer that is required by hipsparseXgpsvInterleavedBatch(). The temporary storage buffer
#   must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const float * ds,const float * dl,const float * d,const float * du,const float * dw,const float * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseSgpsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSgpsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseSgpsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,const float *,const float *,const float *,const float *,const float *,int,unsigned long *) nogil> _hipsparseSgpsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseDgpsvInterleavedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,const double * ds,const double * dl,const double * d,const double * du,const double * dw,const double * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDgpsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDgpsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseDgpsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,const double *,const double *,const double *,const double *,const double *,int,unsigned long *) nogil> _hipsparseDgpsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseCgpsvInterleavedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,float2 * ds,float2 * dl,float2 * d,float2 * du,float2 * dw,float2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCgpsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCgpsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseCgpsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,float2 *,float2 *,int,unsigned long *) nogil> _hipsparseCgpsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseZgpsvInterleavedBatch_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(void * handle,int algo,int m,double2 * ds,double2 * dl,double2 * d,double2 * du,double2 * dw,double2 * x,int batchCount,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZgpsvInterleavedBatch_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZgpsvInterleavedBatch_bufferSizeExt__funptr,"hipsparseZgpsvInterleavedBatch_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,double2 *,double2 *,int,unsigned long *) nogil> _hipsparseZgpsvInterleavedBatch_bufferSizeExt__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBufferSizeInBytes)


cdef void* _hipsparseSgpsvInterleavedBatch__funptr = NULL
#    \ingroup precond_module
#   \brief Interleaved Batch pentadiagonal solver
# 
#   \details
#   \p hipsparseXgpsvInterleavedBatch solves a batched pentadiagonal linear system
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgpsvInterleavedBatch(void * handle,int algo,int m,float * ds,float * dl,float * d,float * du,float * dw,float * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseSgpsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseSgpsvInterleavedBatch__funptr,"hipsparseSgpsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,float *,float *,float *,float *,float *,float *,int,void *) nogil> _hipsparseSgpsvInterleavedBatch__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBuffer)


cdef void* _hipsparseDgpsvInterleavedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseDgpsvInterleavedBatch(void * handle,int algo,int m,double * ds,double * dl,double * d,double * du,double * dw,double * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseDgpsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseDgpsvInterleavedBatch__funptr,"hipsparseDgpsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,double *,double *,double *,double *,double *,double *,int,void *) nogil> _hipsparseDgpsvInterleavedBatch__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBuffer)


cdef void* _hipsparseCgpsvInterleavedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseCgpsvInterleavedBatch(void * handle,int algo,int m,float2 * ds,float2 * dl,float2 * d,float2 * du,float2 * dw,float2 * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseCgpsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseCgpsvInterleavedBatch__funptr,"hipsparseCgpsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,float2 *,float2 *,float2 *,float2 *,float2 *,float2 *,int,void *) nogil> _hipsparseCgpsvInterleavedBatch__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBuffer)


cdef void* _hipsparseZgpsvInterleavedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseZgpsvInterleavedBatch(void * handle,int algo,int m,double2 * ds,double2 * dl,double2 * d,double2 * du,double2 * dw,double2 * x,int batchCount,void * pBuffer) nogil:
    global _hipsparseZgpsvInterleavedBatch__funptr
    __init_symbol(&_hipsparseZgpsvInterleavedBatch__funptr,"hipsparseZgpsvInterleavedBatch")
    return (<hipsparseStatus_t (*)(void *,int,int,double2 *,double2 *,double2 *,double2 *,double2 *,double2 *,int,void *) nogil> _hipsparseZgpsvInterleavedBatch__funptr)(handle,algo,m,ds,dl,d,du,dw,x,batchCount,pBuffer)


cdef void* _hipsparseSnnz__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the number of nonzero elements per row or column and the total
#   number of nonzero elements in a dense matrix.
# 
#   \details
#   The routine does support asynchronous execution if the pointer mode is set to device.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const float * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil:
    global _hipsparseSnnz__funptr
    __init_symbol(&_hipsparseSnnz__funptr,"hipsparseSnnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,int,int *,int *) nogil> _hipsparseSnnz__funptr)(handle,dirA,m,n,descrA,A,lda,nnzPerRowColumn,nnzTotalDevHostPtr)


cdef void* _hipsparseDnnz__funptr = NULL
cdef hipsparseStatus_t hipsparseDnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const double * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil:
    global _hipsparseDnnz__funptr
    __init_symbol(&_hipsparseDnnz__funptr,"hipsparseDnnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,int,int *,int *) nogil> _hipsparseDnnz__funptr)(handle,dirA,m,n,descrA,A,lda,nnzPerRowColumn,nnzTotalDevHostPtr)


cdef void* _hipsparseCnnz__funptr = NULL
cdef hipsparseStatus_t hipsparseCnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,float2 * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil:
    global _hipsparseCnnz__funptr
    __init_symbol(&_hipsparseCnnz__funptr,"hipsparseCnnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,int,int *,int *) nogil> _hipsparseCnnz__funptr)(handle,dirA,m,n,descrA,A,lda,nnzPerRowColumn,nnzTotalDevHostPtr)


cdef void* _hipsparseZnnz__funptr = NULL
cdef hipsparseStatus_t hipsparseZnnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,double2 * A,int lda,int * nnzPerRowColumn,int * nnzTotalDevHostPtr) nogil:
    global _hipsparseZnnz__funptr
    __init_symbol(&_hipsparseZnnz__funptr,"hipsparseZnnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,int,int *,int *) nogil> _hipsparseZnnz__funptr)(handle,dirA,m,n,descrA,A,lda,nnzPerRowColumn,nnzTotalDevHostPtr)


cdef void* _hipsparseSdense2csr__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts the matrix A in dense format into a sparse matrix in CSR format.
#   All the parameters are assumed to have been pre-allocated by the user and the arrays
#   are filled in based on nnz_per_row, which can be pre-computed with hipsparseXnnz().
#   It is executed asynchronously with respect to the host and may return control to the
#   application on the host before the entire result is ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSdense2csr(void * handle,int m,int n,void *const descr,const float * A,int ld,const int * nnz_per_rows,float * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil:
    global _hipsparseSdense2csr__funptr
    __init_symbol(&_hipsparseSdense2csr__funptr,"hipsparseSdense2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,int,const int *,float *,int *,int *) nogil> _hipsparseSdense2csr__funptr)(handle,m,n,descr,A,ld,nnz_per_rows,csr_val,csr_row_ptr,csr_col_ind)


cdef void* _hipsparseDdense2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDdense2csr(void * handle,int m,int n,void *const descr,const double * A,int ld,const int * nnz_per_rows,double * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil:
    global _hipsparseDdense2csr__funptr
    __init_symbol(&_hipsparseDdense2csr__funptr,"hipsparseDdense2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,int,const int *,double *,int *,int *) nogil> _hipsparseDdense2csr__funptr)(handle,m,n,descr,A,ld,nnz_per_rows,csr_val,csr_row_ptr,csr_col_ind)


cdef void* _hipsparseCdense2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseCdense2csr(void * handle,int m,int n,void *const descr,float2 * A,int ld,const int * nnz_per_rows,float2 * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil:
    global _hipsparseCdense2csr__funptr
    __init_symbol(&_hipsparseCdense2csr__funptr,"hipsparseCdense2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,int,const int *,float2 *,int *,int *) nogil> _hipsparseCdense2csr__funptr)(handle,m,n,descr,A,ld,nnz_per_rows,csr_val,csr_row_ptr,csr_col_ind)


cdef void* _hipsparseZdense2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseZdense2csr(void * handle,int m,int n,void *const descr,double2 * A,int ld,const int * nnz_per_rows,double2 * csr_val,int * csr_row_ptr,int * csr_col_ind) nogil:
    global _hipsparseZdense2csr__funptr
    __init_symbol(&_hipsparseZdense2csr__funptr,"hipsparseZdense2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,int,const int *,double2 *,int *,int *) nogil> _hipsparseZdense2csr__funptr)(handle,m,n,descr,A,ld,nnz_per_rows,csr_val,csr_row_ptr,csr_col_ind)


cdef void* _hipsparseSpruneDense2csr_bufferSize__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the the size of the user allocated temporary storage buffer used when converting and pruning
#   a dense matrix to a CSR matrix.
# 
#   \details
#   \p hipsparseXpruneDense2csr_bufferSizeExt returns the size of the temporary storage buffer
#   that is required by hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr(). The
#   temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneDense2csr_bufferSize__funptr
    __init_symbol(&_hipsparseSpruneDense2csr_bufferSize__funptr,"hipsparseSpruneDense2csr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,const float *,void *const,const float *,const int *,const int *,unsigned long *) nogil> _hipsparseSpruneDense2csr_bufferSize__funptr)(handle,m,n,A,lda,threshold,descr,csrVal,csrRowPtr,csrColInd,bufferSize)


cdef void* _hipsparseDpruneDense2csr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneDense2csr_bufferSize__funptr
    __init_symbol(&_hipsparseDpruneDense2csr_bufferSize__funptr,"hipsparseDpruneDense2csr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,const double *,void *const,const double *,const int *,const int *,unsigned long *) nogil> _hipsparseDpruneDense2csr_bufferSize__funptr)(handle,m,n,A,lda,threshold,descr,csrVal,csrRowPtr,csrColInd,bufferSize)


cdef void* _hipsparseSpruneDense2csr_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneDense2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSpruneDense2csr_bufferSizeExt__funptr,"hipsparseSpruneDense2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,const float *,void *const,const float *,const int *,const int *,unsigned long *) nogil> _hipsparseSpruneDense2csr_bufferSizeExt__funptr)(handle,m,n,A,lda,threshold,descr,csrVal,csrRowPtr,csrColInd,bufferSize)


cdef void* _hipsparseDpruneDense2csr_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneDense2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDpruneDense2csr_bufferSizeExt__funptr,"hipsparseDpruneDense2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,const double *,void *const,const double *,const int *,const int *,unsigned long *) nogil> _hipsparseDpruneDense2csr_bufferSizeExt__funptr)(handle,m,n,A,lda,threshold,descr,csrVal,csrRowPtr,csrColInd,bufferSize)


cdef void* _hipsparseSpruneDense2csrNnz__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the number of nonzero elements per row and the total number of
#   nonzero elements in a dense matrix once elements less than the threshold are pruned
#   from the matrix.
# 
#   \details
#   The routine does support asynchronous execution if the pointer mode is set to device.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csrNnz(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,void * buffer) nogil:
    global _hipsparseSpruneDense2csrNnz__funptr
    __init_symbol(&_hipsparseSpruneDense2csrNnz__funptr,"hipsparseSpruneDense2csrNnz")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,const float *,void *const,int *,int *,void *) nogil> _hipsparseSpruneDense2csrNnz__funptr)(handle,m,n,A,lda,threshold,descr,csrRowPtr,nnzTotalDevHostPtr,buffer)


cdef void* _hipsparseDpruneDense2csrNnz__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csrNnz(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,void * buffer) nogil:
    global _hipsparseDpruneDense2csrNnz__funptr
    __init_symbol(&_hipsparseDpruneDense2csrNnz__funptr,"hipsparseDpruneDense2csrNnz")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,const double *,void *const,int *,int *,void *) nogil> _hipsparseDpruneDense2csrNnz__funptr)(handle,m,n,A,lda,threshold,descr,csrRowPtr,nnzTotalDevHostPtr,buffer)


cdef void* _hipsparseSpruneDense2csr__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts the matrix A in dense format into a sparse matrix in CSR format
#   while pruning values that are less than the threshold. All the parameters are assumed
#   to have been pre-allocated by the user.
# 
#   \details
#   The user first allocates \p csrRowPtr to have \p m+1 elements and then calls
#   hipsparseXpruneDense2csrNnz() which fills in the \p csrRowPtr array and stores the
#   number of elements that are larger than the pruning threshold in \p nnzTotalDevHostPtr.
#   The user then allocates \p csrColInd and \p csrVal to have size \p nnzTotalDevHostPtr
#   and completes the conversion by calling hipsparseXpruneDense2csr(). A temporary storage
#   buffer is used by both hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr() and
#   must be allocated by the user and whose size is determined by
#   hipsparseXpruneDense2csr_bufferSizeExt(). The routine hipsparseXpruneDense2csr() is
#   executed asynchronously with respect to the host and may return control to the
#   application on the host before the entire result is ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csr(void * handle,int m,int n,const float * A,int lda,const float * threshold,void *const descr,float * csrVal,const int * csrRowPtr,int * csrColInd,void * buffer) nogil:
    global _hipsparseSpruneDense2csr__funptr
    __init_symbol(&_hipsparseSpruneDense2csr__funptr,"hipsparseSpruneDense2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,const float *,void *const,float *,const int *,int *,void *) nogil> _hipsparseSpruneDense2csr__funptr)(handle,m,n,A,lda,threshold,descr,csrVal,csrRowPtr,csrColInd,buffer)


cdef void* _hipsparseDpruneDense2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csr(void * handle,int m,int n,const double * A,int lda,const double * threshold,void *const descr,double * csrVal,const int * csrRowPtr,int * csrColInd,void * buffer) nogil:
    global _hipsparseDpruneDense2csr__funptr
    __init_symbol(&_hipsparseDpruneDense2csr__funptr,"hipsparseDpruneDense2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,const double *,void *const,double *,const int *,int *,void *) nogil> _hipsparseDpruneDense2csr__funptr)(handle,m,n,A,lda,threshold,descr,csrVal,csrRowPtr,csrColInd,buffer)


cdef void* _hipsparseSpruneDense2csrByPercentage_bufferSize__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the size of the user allocated temporary storage buffer used
#   when converting and pruning by percentage a dense matrix to a CSR matrix.
# 
#   \details
#   When converting and pruning a dense matrix A to a CSR matrix by percentage the
#   following steps are performed. First the user calls
#   \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
#   temporary storage buffer. Once determined, this buffer must be allocated by the user.
#   Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
#   \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
#   by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
#   at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
#   The pruning by percentage works by first sorting the absolute values of the dense
#   matrix \p A. We then determine a position in this sorted array by
#   \f[
#     pos = ceil(m*n*(percentage/100)) - 1
#     pos = min(pos, m*n-1)
#     pos = max(pos, 0)
#     threshold = sorted_A[pos]
#   \f]
#   Once we have this threshold we prune values in the dense matrix \p A as in
#   \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
#   and may return control to the application on the host before the entire result is
#   ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneDense2csrByPercentage_bufferSize__funptr
    __init_symbol(&_hipsparseSpruneDense2csrByPercentage_bufferSize__funptr,"hipsparseSpruneDense2csrByPercentage_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,float,void *const,const float *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseSpruneDense2csrByPercentage_bufferSize__funptr)(handle,m,n,A,lda,percentage,descr,csrVal,csrRowPtr,csrColInd,info,bufferSize)


cdef void* _hipsparseDpruneDense2csrByPercentage_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneDense2csrByPercentage_bufferSize__funptr
    __init_symbol(&_hipsparseDpruneDense2csrByPercentage_bufferSize__funptr,"hipsparseDpruneDense2csrByPercentage_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,double,void *const,const double *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseDpruneDense2csrByPercentage_bufferSize__funptr)(handle,m,n,A,lda,percentage,descr,csrVal,csrRowPtr,csrColInd,info,bufferSize)


cdef void* _hipsparseSpruneDense2csrByPercentage_bufferSizeExt__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the size of the user allocated temporary storage buffer used
#   when converting and pruning by percentage a dense matrix to a CSR matrix.
# 
#   \details
#   When converting and pruning a dense matrix A to a CSR matrix by percentage the
#   following steps are performed. First the user calls
#   \p hipsparseXpruneDense2csrByPercentage_bufferSizeExt which determines the size of the
#   temporary storage buffer. Once determined, this buffer must be allocated by the user.
#   Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
#   \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
#   by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
#   at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
#   The pruning by percentage works by first sorting the absolute values of the dense
#   matrix \p A. We then determine a position in this sorted array by
#   \f[
#     pos = ceil(m*n*(percentage/100)) - 1
#     pos = min(pos, m*n-1)
#     pos = max(pos, 0)
#     threshold = sorted_A[pos]
#   \f]
#   Once we have this threshold we prune values in the dense matrix \p A as in
#   \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
#   and may return control to the application on the host before the entire result is
#   ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSizeExt(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,const float * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneDense2csrByPercentage_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSpruneDense2csrByPercentage_bufferSizeExt__funptr,"hipsparseSpruneDense2csrByPercentage_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,float,void *const,const float *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseSpruneDense2csrByPercentage_bufferSizeExt__funptr)(handle,m,n,A,lda,percentage,descr,csrVal,csrRowPtr,csrColInd,info,bufferSize)


cdef void* _hipsparseDpruneDense2csrByPercentage_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSizeExt(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,const double * csrVal,const int * csrRowPtr,const int * csrColInd,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneDense2csrByPercentage_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDpruneDense2csrByPercentage_bufferSizeExt__funptr,"hipsparseDpruneDense2csrByPercentage_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,double,void *const,const double *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseDpruneDense2csrByPercentage_bufferSizeExt__funptr)(handle,m,n,A,lda,percentage,descr,csrVal,csrRowPtr,csrColInd,info,bufferSize)


cdef void* _hipsparseSpruneDense2csrNnzByPercentage__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the number of nonzero elements per row and the total number of
#   nonzero elements in a dense matrix when converting and pruning by percentage a dense
#   matrix to a CSR matrix.
# 
#   \details
#   When converting and pruning a dense matrix A to a CSR matrix by percentage the
#   following steps are performed. First the user calls
#   \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
#   temporary storage buffer. Once determined, this buffer must be allocated by the user.
#   Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
#   \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
#   by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
#   at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
#   The pruning by percentage works by first sorting the absolute values of the dense
#   matrix \p A. We then determine a position in this sorted array by
#   \f[
#     pos = ceil(m*n*(percentage/100)) - 1
#     pos = min(pos, m*n-1)
#     pos = max(pos, 0)
#     threshold = sorted_A[pos]
#   \f]
#   Once we have this threshold we prune values in the dense matrix \p A as in
#   \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
#   pointer mode is set to device.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseSpruneDense2csrNnzByPercentage__funptr
    __init_symbol(&_hipsparseSpruneDense2csrNnzByPercentage__funptr,"hipsparseSpruneDense2csrNnzByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,float,void *const,int *,int *,pruneInfo_t,void *) nogil> _hipsparseSpruneDense2csrNnzByPercentage__funptr)(handle,m,n,A,lda,percentage,descr,csrRowPtr,nnzTotalDevHostPtr,info,buffer)


cdef void* _hipsparseDpruneDense2csrNnzByPercentage__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,int * csrRowPtr,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseDpruneDense2csrNnzByPercentage__funptr
    __init_symbol(&_hipsparseDpruneDense2csrNnzByPercentage__funptr,"hipsparseDpruneDense2csrNnzByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,double,void *const,int *,int *,pruneInfo_t,void *) nogil> _hipsparseDpruneDense2csrNnzByPercentage__funptr)(handle,m,n,A,lda,percentage,descr,csrRowPtr,nnzTotalDevHostPtr,info,buffer)


cdef void* _hipsparseSpruneDense2csrByPercentage__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the number of nonzero elements per row and the total number of
#   nonzero elements in a dense matrix when converting and pruning by percentage a dense
#   matrix to a CSR matrix.
# 
#   \details
#   When converting and pruning a dense matrix A to a CSR matrix by percentage the
#   following steps are performed. First the user calls
#   \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
#   temporary storage buffer. Once determined, this buffer must be allocated by the user.
#   Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
#   \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
#   by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
#   at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
# 
#   The pruning by percentage works by first sorting the absolute values of the dense
#   matrix \p A. We then determine a position in this sorted array by
#   \f[
#     pos = ceil(m*n*(percentage/100)) - 1
#     pos = min(pos, m*n-1)
#     pos = max(pos, 0)
#     threshold = sorted_A[pos]
#   \f]
#   Once we have this threshold we prune values in the dense matrix \p A as in
#   \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
#   pointer mode is set to device.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(void * handle,int m,int n,const float * A,int lda,float percentage,void *const descr,float * csrVal,const int * csrRowPtr,int * csrColInd,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseSpruneDense2csrByPercentage__funptr
    __init_symbol(&_hipsparseSpruneDense2csrByPercentage__funptr,"hipsparseSpruneDense2csrByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,const float *,int,float,void *const,float *,const int *,int *,pruneInfo_t,void *) nogil> _hipsparseSpruneDense2csrByPercentage__funptr)(handle,m,n,A,lda,percentage,descr,csrVal,csrRowPtr,csrColInd,info,buffer)


cdef void* _hipsparseDpruneDense2csrByPercentage__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(void * handle,int m,int n,const double * A,int lda,double percentage,void *const descr,double * csrVal,const int * csrRowPtr,int * csrColInd,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseDpruneDense2csrByPercentage__funptr
    __init_symbol(&_hipsparseDpruneDense2csrByPercentage__funptr,"hipsparseDpruneDense2csrByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,const double *,int,double,void *const,double *,const int *,int *,pruneInfo_t,void *) nogil> _hipsparseDpruneDense2csrByPercentage__funptr)(handle,m,n,A,lda,percentage,descr,csrVal,csrRowPtr,csrColInd,info,buffer)


cdef void* _hipsparseSdense2csc__funptr = NULL
#    \ingroup conv_module
#   \brief
# 
#   This function converts the matrix A in dense format into a sparse matrix in CSC format.
#   All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_columns, which can be pre-computed with hipsparseXnnz().
#   It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSdense2csc(void * handle,int m,int n,void *const descr,const float * A,int ld,const int * nnz_per_columns,float * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil:
    global _hipsparseSdense2csc__funptr
    __init_symbol(&_hipsparseSdense2csc__funptr,"hipsparseSdense2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,int,const int *,float *,int *,int *) nogil> _hipsparseSdense2csc__funptr)(handle,m,n,descr,A,ld,nnz_per_columns,csc_val,csc_row_ind,csc_col_ptr)


cdef void* _hipsparseDdense2csc__funptr = NULL
cdef hipsparseStatus_t hipsparseDdense2csc(void * handle,int m,int n,void *const descr,const double * A,int ld,const int * nnz_per_columns,double * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil:
    global _hipsparseDdense2csc__funptr
    __init_symbol(&_hipsparseDdense2csc__funptr,"hipsparseDdense2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,int,const int *,double *,int *,int *) nogil> _hipsparseDdense2csc__funptr)(handle,m,n,descr,A,ld,nnz_per_columns,csc_val,csc_row_ind,csc_col_ptr)


cdef void* _hipsparseCdense2csc__funptr = NULL
cdef hipsparseStatus_t hipsparseCdense2csc(void * handle,int m,int n,void *const descr,float2 * A,int ld,const int * nnz_per_columns,float2 * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil:
    global _hipsparseCdense2csc__funptr
    __init_symbol(&_hipsparseCdense2csc__funptr,"hipsparseCdense2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,int,const int *,float2 *,int *,int *) nogil> _hipsparseCdense2csc__funptr)(handle,m,n,descr,A,ld,nnz_per_columns,csc_val,csc_row_ind,csc_col_ptr)


cdef void* _hipsparseZdense2csc__funptr = NULL
cdef hipsparseStatus_t hipsparseZdense2csc(void * handle,int m,int n,void *const descr,double2 * A,int ld,const int * nnz_per_columns,double2 * csc_val,int * csc_row_ind,int * csc_col_ptr) nogil:
    global _hipsparseZdense2csc__funptr
    __init_symbol(&_hipsparseZdense2csc__funptr,"hipsparseZdense2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,int,const int *,double2 *,int *,int *) nogil> _hipsparseZdense2csc__funptr)(handle,m,n,descr,A,ld,nnz_per_columns,csc_val,csc_row_ind,csc_col_ptr)


cdef void* _hipsparseScsr2dense__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts the sparse matrix in CSR format into a dense matrix.
#   It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2dense(void * handle,int m,int n,void *const descr,const float * csr_val,const int * csr_row_ptr,const int * csr_col_ind,float * A,int ld) nogil:
    global _hipsparseScsr2dense__funptr
    __init_symbol(&_hipsparseScsr2dense__funptr,"hipsparseScsr2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,float *,int) nogil> _hipsparseScsr2dense__funptr)(handle,m,n,descr,csr_val,csr_row_ptr,csr_col_ind,A,ld)


cdef void* _hipsparseDcsr2dense__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2dense(void * handle,int m,int n,void *const descr,const double * csr_val,const int * csr_row_ptr,const int * csr_col_ind,double * A,int ld) nogil:
    global _hipsparseDcsr2dense__funptr
    __init_symbol(&_hipsparseDcsr2dense__funptr,"hipsparseDcsr2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,double *,int) nogil> _hipsparseDcsr2dense__funptr)(handle,m,n,descr,csr_val,csr_row_ptr,csr_col_ind,A,ld)


cdef void* _hipsparseCcsr2dense__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2dense(void * handle,int m,int n,void *const descr,float2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,float2 * A,int ld) nogil:
    global _hipsparseCcsr2dense__funptr
    __init_symbol(&_hipsparseCcsr2dense__funptr,"hipsparseCcsr2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,float2 *,int) nogil> _hipsparseCcsr2dense__funptr)(handle,m,n,descr,csr_val,csr_row_ptr,csr_col_ind,A,ld)


cdef void* _hipsparseZcsr2dense__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2dense(void * handle,int m,int n,void *const descr,double2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,double2 * A,int ld) nogil:
    global _hipsparseZcsr2dense__funptr
    __init_symbol(&_hipsparseZcsr2dense__funptr,"hipsparseZcsr2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,double2 *,int) nogil> _hipsparseZcsr2dense__funptr)(handle,m,n,descr,csr_val,csr_row_ptr,csr_col_ind,A,ld)


cdef void* _hipsparseScsc2dense__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts the sparse matrix in CSC format into a dense matrix.
#   It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsc2dense(void * handle,int m,int n,void *const descr,const float * csc_val,const int * csc_row_ind,const int * csc_col_ptr,float * A,int ld) nogil:
    global _hipsparseScsc2dense__funptr
    __init_symbol(&_hipsparseScsc2dense__funptr,"hipsparseScsc2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,float *,int) nogil> _hipsparseScsc2dense__funptr)(handle,m,n,descr,csc_val,csc_row_ind,csc_col_ptr,A,ld)


cdef void* _hipsparseDcsc2dense__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsc2dense(void * handle,int m,int n,void *const descr,const double * csc_val,const int * csc_row_ind,const int * csc_col_ptr,double * A,int ld) nogil:
    global _hipsparseDcsc2dense__funptr
    __init_symbol(&_hipsparseDcsc2dense__funptr,"hipsparseDcsc2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,double *,int) nogil> _hipsparseDcsc2dense__funptr)(handle,m,n,descr,csc_val,csc_row_ind,csc_col_ptr,A,ld)


cdef void* _hipsparseCcsc2dense__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsc2dense(void * handle,int m,int n,void *const descr,float2 * csc_val,const int * csc_row_ind,const int * csc_col_ptr,float2 * A,int ld) nogil:
    global _hipsparseCcsc2dense__funptr
    __init_symbol(&_hipsparseCcsc2dense__funptr,"hipsparseCcsc2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,float2 *,int) nogil> _hipsparseCcsc2dense__funptr)(handle,m,n,descr,csc_val,csc_row_ind,csc_col_ptr,A,ld)


cdef void* _hipsparseZcsc2dense__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsc2dense(void * handle,int m,int n,void *const descr,double2 * csc_val,const int * csc_row_ind,const int * csc_col_ptr,double2 * A,int ld) nogil:
    global _hipsparseZcsc2dense__funptr
    __init_symbol(&_hipsparseZcsc2dense__funptr,"hipsparseZcsc2dense")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,double2 *,int) nogil> _hipsparseZcsc2dense__funptr)(handle,m,n,descr,csc_val,csc_row_ind,csc_col_ptr,A,ld)


cdef void* _hipsparseXcsr2bsrNnz__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
#   BSR matrix given a sparse CSR matrix as input.
# 
#   \details
#   The routine does support asynchronous execution if the pointer mode is set to device.
# /
cdef hipsparseStatus_t hipsparseXcsr2bsrNnz(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,int * bsrRowPtrC,int * bsrNnzb) nogil:
    global _hipsparseXcsr2bsrNnz__funptr
    __init_symbol(&_hipsparseXcsr2bsrNnz__funptr,"hipsparseXcsr2bsrNnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const int *,const int *,int,void *const,int *,int *) nogil> _hipsparseXcsr2bsrNnz__funptr)(handle,dirA,m,n,descrA,csrRowPtrA,csrColIndA,blockDim,descrC,bsrRowPtrC,bsrNnzb)


cdef void* _hipsparseSnnz_compress__funptr = NULL
#    \ingroup conv_module
#   Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
#   in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
#   of remaining elements in the matrix.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSnnz_compress(void * handle,int m,void *const descrA,const float * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,float tol) nogil:
    global _hipsparseSnnz_compress__funptr
    __init_symbol(&_hipsparseSnnz_compress__funptr,"hipsparseSnnz_compress")
    return (<hipsparseStatus_t (*)(void *,int,void *const,const float *,const int *,int *,int *,float) nogil> _hipsparseSnnz_compress__funptr)(handle,m,descrA,csrValA,csrRowPtrA,nnzPerRow,nnzC,tol)


cdef void* _hipsparseDnnz_compress__funptr = NULL
cdef hipsparseStatus_t hipsparseDnnz_compress(void * handle,int m,void *const descrA,const double * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,double tol) nogil:
    global _hipsparseDnnz_compress__funptr
    __init_symbol(&_hipsparseDnnz_compress__funptr,"hipsparseDnnz_compress")
    return (<hipsparseStatus_t (*)(void *,int,void *const,const double *,const int *,int *,int *,double) nogil> _hipsparseDnnz_compress__funptr)(handle,m,descrA,csrValA,csrRowPtrA,nnzPerRow,nnzC,tol)


cdef void* _hipsparseCnnz_compress__funptr = NULL
cdef hipsparseStatus_t hipsparseCnnz_compress(void * handle,int m,void *const descrA,float2 * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,float2 tol) nogil:
    global _hipsparseCnnz_compress__funptr
    __init_symbol(&_hipsparseCnnz_compress__funptr,"hipsparseCnnz_compress")
    return (<hipsparseStatus_t (*)(void *,int,void *const,float2 *,const int *,int *,int *,float2) nogil> _hipsparseCnnz_compress__funptr)(handle,m,descrA,csrValA,csrRowPtrA,nnzPerRow,nnzC,tol)


cdef void* _hipsparseZnnz_compress__funptr = NULL
cdef hipsparseStatus_t hipsparseZnnz_compress(void * handle,int m,void *const descrA,double2 * csrValA,const int * csrRowPtrA,int * nnzPerRow,int * nnzC,double2 tol) nogil:
    global _hipsparseZnnz_compress__funptr
    __init_symbol(&_hipsparseZnnz_compress__funptr,"hipsparseZnnz_compress")
    return (<hipsparseStatus_t (*)(void *,int,void *const,double2 *,const int *,int *,int *,double2) nogil> _hipsparseZnnz_compress__funptr)(handle,m,descrA,csrValA,csrRowPtrA,nnzPerRow,nnzC,tol)


cdef void* _hipsparseXcsr2coo__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse CSR matrix into a sparse COO matrix
# 
#   \details
#   \p hipsparseXcsr2coo converts the CSR array containing the row offsets, that point
#   to the start of every row, into a COO array of row indices.
# 
#   \note
#   It can also be used to convert a CSC array containing the column offsets into a COO
#   array of column indices.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseXcsr2coo(void * handle,const int * csrRowPtr,int nnz,int m,int * cooRowInd,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseXcsr2coo__funptr
    __init_symbol(&_hipsparseXcsr2coo__funptr,"hipsparseXcsr2coo")
    return (<hipsparseStatus_t (*)(void *,const int *,int,int,int *,hipsparseIndexBase_t) nogil> _hipsparseXcsr2coo__funptr)(handle,csrRowPtr,nnz,m,cooRowInd,idxBase)


cdef void* _hipsparseScsr2csc__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse CSR matrix into a sparse CSC matrix
# 
#   \details
#   \p hipsparseXcsr2csc converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2csc
#   can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
#   whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
#   or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
#   \note
#   The resulting matrix can also be seen as the transpose of the input matrix.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2csc(void * handle,int m,int n,int nnz,const float * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,float * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseScsr2csc__funptr
    __init_symbol(&_hipsparseScsr2csc__funptr,"hipsparseScsr2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const float *,const int *,const int *,float *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t) nogil> _hipsparseScsr2csc__funptr)(handle,m,n,nnz,csrSortedVal,csrSortedRowPtr,csrSortedColInd,cscSortedVal,cscSortedRowInd,cscSortedColPtr,copyValues,idxBase)


cdef void* _hipsparseDcsr2csc__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2csc(void * handle,int m,int n,int nnz,const double * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,double * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseDcsr2csc__funptr
    __init_symbol(&_hipsparseDcsr2csc__funptr,"hipsparseDcsr2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const double *,const int *,const int *,double *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t) nogil> _hipsparseDcsr2csc__funptr)(handle,m,n,nnz,csrSortedVal,csrSortedRowPtr,csrSortedColInd,cscSortedVal,cscSortedRowInd,cscSortedColPtr,copyValues,idxBase)


cdef void* _hipsparseCcsr2csc__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2csc(void * handle,int m,int n,int nnz,float2 * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,float2 * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseCcsr2csc__funptr
    __init_symbol(&_hipsparseCcsr2csc__funptr,"hipsparseCcsr2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float2 *,const int *,const int *,float2 *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t) nogil> _hipsparseCcsr2csc__funptr)(handle,m,n,nnz,csrSortedVal,csrSortedRowPtr,csrSortedColInd,cscSortedVal,cscSortedRowInd,cscSortedColPtr,copyValues,idxBase)


cdef void* _hipsparseZcsr2csc__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2csc(void * handle,int m,int n,int nnz,double2 * csrSortedVal,const int * csrSortedRowPtr,const int * csrSortedColInd,double2 * cscSortedVal,int * cscSortedRowInd,int * cscSortedColPtr,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseZcsr2csc__funptr
    __init_symbol(&_hipsparseZcsr2csc__funptr,"hipsparseZcsr2csc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double2 *,const int *,const int *,double2 *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t) nogil> _hipsparseZcsr2csc__funptr)(handle,m,n,nnz,csrSortedVal,csrSortedRowPtr,csrSortedColInd,cscSortedVal,cscSortedRowInd,cscSortedColPtr,copyValues,idxBase)


cdef void* _hipsparseCsr2cscEx2_bufferSize__funptr = NULL
#    \ingroup conv_module
#   \brief This function computes the size of the user allocated temporary storage buffer used
#   when converting a sparse CSR matrix into a sparse CSC matrix.
# 
#   \details
#   \p hipsparseXcsr2cscEx2_bufferSize calculates the required user allocated temporary buffer needed
#   by \p hipsparseXcsr2cscEx2 to convert a CSR matrix into a CSC matrix. \p hipsparseXcsr2cscEx2
#   can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
#   whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
#   or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
#   \note
#   The resulting matrix can also be seen as the transpose of the input matrix.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(void * handle,int m,int n,int nnz,const void * csrVal,const int * csrRowPtr,const int * csrColInd,void * cscVal,int * cscColPtr,int * cscRowInd,hipDataType valType,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase,hipsparseCsr2CscAlg_t alg,unsigned long * bufferSize) nogil:
    global _hipsparseCsr2cscEx2_bufferSize__funptr
    __init_symbol(&_hipsparseCsr2cscEx2_bufferSize__funptr,"hipsparseCsr2cscEx2_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const void *,const int *,const int *,void *,int *,int *,hipDataType,hipsparseAction_t,hipsparseIndexBase_t,hipsparseCsr2CscAlg_t,unsigned long *) nogil> _hipsparseCsr2cscEx2_bufferSize__funptr)(handle,m,n,nnz,csrVal,csrRowPtr,csrColInd,cscVal,cscColPtr,cscRowInd,valType,copyValues,idxBase,alg,bufferSize)


cdef void* _hipsparseCsr2cscEx2__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse CSR matrix into a sparse CSC matrix
# 
#   \details
#   \p hipsparseXcsr2cscEx2 converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2cscEx2
#   can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
#   whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
#   or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
#   \note
#   The resulting matrix can also be seen as the transpose of the input matrix.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseCsr2cscEx2(void * handle,int m,int n,int nnz,const void * csrVal,const int * csrRowPtr,const int * csrColInd,void * cscVal,int * cscColPtr,int * cscRowInd,hipDataType valType,hipsparseAction_t copyValues,hipsparseIndexBase_t idxBase,hipsparseCsr2CscAlg_t alg,void * buffer) nogil:
    global _hipsparseCsr2cscEx2__funptr
    __init_symbol(&_hipsparseCsr2cscEx2__funptr,"hipsparseCsr2cscEx2")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const void *,const int *,const int *,void *,int *,int *,hipDataType,hipsparseAction_t,hipsparseIndexBase_t,hipsparseCsr2CscAlg_t,void *) nogil> _hipsparseCsr2cscEx2__funptr)(handle,m,n,nnz,csrVal,csrRowPtr,csrColInd,cscVal,cscColPtr,cscRowInd,valType,copyValues,idxBase,alg,buffer)


cdef void* _hipsparseScsr2hyb__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse CSR matrix into a sparse HYB matrix
# 
#   \details
#   \p hipsparseXcsr2hyb converts a CSR matrix into a HYB matrix. It is assumed
#   that \p hyb has been initialized with hipsparseCreateHybMat().
# 
#   \note
#   This function requires a significant amount of storage for the HYB matrix,
#   depending on the matrix structure.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2hyb(void * handle,int m,int n,void *const descrA,const float * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil:
    global _hipsparseScsr2hyb__funptr
    __init_symbol(&_hipsparseScsr2hyb__funptr,"hipsparseScsr2hyb")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,void *,int,hipsparseHybPartition_t) nogil> _hipsparseScsr2hyb__funptr)(handle,m,n,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,hybA,userEllWidth,partitionType)


cdef void* _hipsparseDcsr2hyb__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2hyb(void * handle,int m,int n,void *const descrA,const double * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil:
    global _hipsparseDcsr2hyb__funptr
    __init_symbol(&_hipsparseDcsr2hyb__funptr,"hipsparseDcsr2hyb")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,void *,int,hipsparseHybPartition_t) nogil> _hipsparseDcsr2hyb__funptr)(handle,m,n,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,hybA,userEllWidth,partitionType)


cdef void* _hipsparseCcsr2hyb__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2hyb(void * handle,int m,int n,void *const descrA,float2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil:
    global _hipsparseCcsr2hyb__funptr
    __init_symbol(&_hipsparseCcsr2hyb__funptr,"hipsparseCcsr2hyb")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,void *,int,hipsparseHybPartition_t) nogil> _hipsparseCcsr2hyb__funptr)(handle,m,n,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,hybA,userEllWidth,partitionType)


cdef void* _hipsparseZcsr2hyb__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2hyb(void * handle,int m,int n,void *const descrA,double2 * csrSortedValA,const int * csrSortedRowPtrA,const int * csrSortedColIndA,void * hybA,int userEllWidth,hipsparseHybPartition_t partitionType) nogil:
    global _hipsparseZcsr2hyb__funptr
    __init_symbol(&_hipsparseZcsr2hyb__funptr,"hipsparseZcsr2hyb")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,void *,int,hipsparseHybPartition_t) nogil> _hipsparseZcsr2hyb__funptr)(handle,m,n,descrA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA,hybA,userEllWidth,partitionType)


cdef void* _hipsparseSgebsr2gebsc_bufferSize__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
# 
#   \details
#   \p hipsparseXgebsr2gebsc_bufferSize returns the size of the temporary storage buffer
#   required by hipsparseXgebsr2gebsc().
#   The temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,const float * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseSgebsr2gebsc_bufferSize__funptr
    __init_symbol(&_hipsparseSgebsr2gebsc_bufferSize__funptr,"hipsparseSgebsr2gebsc_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const float *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseSgebsr2gebsc_bufferSize__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseDgebsr2gebsc_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,const double * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseDgebsr2gebsc_bufferSize__funptr
    __init_symbol(&_hipsparseDgebsr2gebsc_bufferSize__funptr,"hipsparseDgebsr2gebsc_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const double *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseDgebsr2gebsc_bufferSize__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseCgebsr2gebsc_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,float2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseCgebsr2gebsc_bufferSize__funptr
    __init_symbol(&_hipsparseCgebsr2gebsc_bufferSize__funptr,"hipsparseCgebsr2gebsc_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float2 *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseCgebsr2gebsc_bufferSize__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseZgebsr2gebsc_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(void * handle,int mb,int nb,int nnzb,double2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseZgebsr2gebsc_bufferSize__funptr
    __init_symbol(&_hipsparseZgebsr2gebsc_bufferSize__funptr,"hipsparseZgebsr2gebsc_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double2 *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseZgebsr2gebsc_bufferSize__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseSgebsr2gebsc__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
# 
#   \details
#   \p hipsparseXgebsr2gebsc converts a GEneral BSR matrix into a GEneral BSC matrix. \p hipsparseXgebsr2gebsc
#   can also be used to convert a GEneral BSC matrix into a GEneral BSR matrix. \p copy_values decides
#   whether \p bsc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
#   or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
# 
#   \p hipsparseXgebsr2gebsc requires extra temporary storage buffer that has to be allocated
#   by the user. Storage buffer size can be determined by hipsparseXgebsr2gebsc_bufferSize().
# 
#   \note
#   The resulting matrix can also be seen as the transpose of the input matrix.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgebsr2gebsc(void * handle,int mb,int nb,int nnzb,const float * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,float * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil:
    global _hipsparseSgebsr2gebsc__funptr
    __init_symbol(&_hipsparseSgebsr2gebsc__funptr,"hipsparseSgebsr2gebsc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const float *,const int *,const int *,int,int,float *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t,void *) nogil> _hipsparseSgebsr2gebsc__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,bsc_val,bsc_row_ind,bsc_col_ptr,copy_values,idx_base,temp_buffer)


cdef void* _hipsparseDgebsr2gebsc__funptr = NULL
cdef hipsparseStatus_t hipsparseDgebsr2gebsc(void * handle,int mb,int nb,int nnzb,const double * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,double * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil:
    global _hipsparseDgebsr2gebsc__funptr
    __init_symbol(&_hipsparseDgebsr2gebsc__funptr,"hipsparseDgebsr2gebsc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const double *,const int *,const int *,int,int,double *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t,void *) nogil> _hipsparseDgebsr2gebsc__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,bsc_val,bsc_row_ind,bsc_col_ptr,copy_values,idx_base,temp_buffer)


cdef void* _hipsparseCgebsr2gebsc__funptr = NULL
cdef hipsparseStatus_t hipsparseCgebsr2gebsc(void * handle,int mb,int nb,int nnzb,float2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,float2 * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil:
    global _hipsparseCgebsr2gebsc__funptr
    __init_symbol(&_hipsparseCgebsr2gebsc__funptr,"hipsparseCgebsr2gebsc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float2 *,const int *,const int *,int,int,float2 *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t,void *) nogil> _hipsparseCgebsr2gebsc__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,bsc_val,bsc_row_ind,bsc_col_ptr,copy_values,idx_base,temp_buffer)


cdef void* _hipsparseZgebsr2gebsc__funptr = NULL
cdef hipsparseStatus_t hipsparseZgebsr2gebsc(void * handle,int mb,int nb,int nnzb,double2 * bsr_val,const int * bsr_row_ptr,const int * bsr_col_ind,int row_block_dim,int col_block_dim,double2 * bsc_val,int * bsc_row_ind,int * bsc_col_ptr,hipsparseAction_t copy_values,hipsparseIndexBase_t idx_base,void * temp_buffer) nogil:
    global _hipsparseZgebsr2gebsc__funptr
    __init_symbol(&_hipsparseZgebsr2gebsc__funptr,"hipsparseZgebsr2gebsc")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double2 *,const int *,const int *,int,int,double2 *,int *,int *,hipsparseAction_t,hipsparseIndexBase_t,void *) nogil> _hipsparseZgebsr2gebsc__funptr)(handle,mb,nb,nnzb,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,bsc_val,bsc_row_ind,bsc_col_ptr,copy_values,idx_base,temp_buffer)


cdef void* _hipsparseScsr2gebsr_bufferSize__funptr = NULL
#    \ingroup conv_module
#   \brief
#    \details
#    \p hipsparseXcsr2gebsr_bufferSize returns the size of the temporary buffer that
#    is required by \p hipsparseXcsr2gebcsrNnz and \p hipsparseXcsr2gebcsr.
#    The temporary storage buffer must be allocated by the user.
# 
#   This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
#   GEneral BSR matrix given a sparse CSR matrix as input.
# 
#   \details
#   The routine does support asynchronous execution if the pointer mode is set to device.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const float * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseScsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseScsr2gebsr_bufferSize__funptr,"hipsparseScsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseScsr2gebsr_bufferSize__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseDcsr2gebsr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const double * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseDcsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseDcsr2gebsr_bufferSize__funptr,"hipsparseDcsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseDcsr2gebsr_bufferSize__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseCcsr2gebsr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,float2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseCcsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseCcsr2gebsr_bufferSize__funptr,"hipsparseCcsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseCcsr2gebsr_bufferSize__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseZcsr2gebsr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,double2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,int row_block_dim,int col_block_dim,unsigned long * p_buffer_size) nogil:
    global _hipsparseZcsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseZcsr2gebsr_bufferSize__funptr,"hipsparseZcsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,int,unsigned long *) nogil> _hipsparseZcsr2gebsr_bufferSize__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,row_block_dim,col_block_dim,p_buffer_size)


cdef void* _hipsparseXcsr2gebsrNnz__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
#   GEneral BSR matrix given a sparse CSR matrix as input.
# 
# /
cdef hipsparseStatus_t hipsparseXcsr2gebsrNnz(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,int * bsr_row_ptr,int row_block_dim,int col_block_dim,int * bsr_nnz_devhost,void * p_buffer) nogil:
    global _hipsparseXcsr2gebsrNnz__funptr
    __init_symbol(&_hipsparseXcsr2gebsrNnz__funptr,"hipsparseXcsr2gebsrNnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const int *,const int *,void *const,int *,int,int,int *,void *) nogil> _hipsparseXcsr2gebsrNnz__funptr)(handle,dir,m,n,csr_descr,csr_row_ptr,csr_col_ind,bsr_descr,bsr_row_ptr,row_block_dim,col_block_dim,bsr_nnz_devhost,p_buffer)


cdef void* _hipsparseScsr2gebsr__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse CSR matrix into a sparse GEneral BSR matrix
# 
#   \details
#   \p hipsparseXcsr2gebsr converts a CSR matrix into a GEneral BSR matrix. It is assumed,
#   that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
#   for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
#   the GEneral BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
#   \p csr2gebsr_nnz() which also fills in \p bsr_row_ptr.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const float * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,float * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil:
    global _hipsparseScsr2gebsr__funptr
    __init_symbol(&_hipsparseScsr2gebsr__funptr,"hipsparseScsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,const int *,const int *,void *const,float *,int *,int *,int,int,void *) nogil> _hipsparseScsr2gebsr__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,bsr_descr,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer)


cdef void* _hipsparseDcsr2gebsr__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,const double * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,double * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil:
    global _hipsparseDcsr2gebsr__funptr
    __init_symbol(&_hipsparseDcsr2gebsr__funptr,"hipsparseDcsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,const int *,const int *,void *const,double *,int *,int *,int,int,void *) nogil> _hipsparseDcsr2gebsr__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,bsr_descr,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer)


cdef void* _hipsparseCcsr2gebsr__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,float2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,float2 * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil:
    global _hipsparseCcsr2gebsr__funptr
    __init_symbol(&_hipsparseCcsr2gebsr__funptr,"hipsparseCcsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,void *const,float2 *,int *,int *,int,int,void *) nogil> _hipsparseCcsr2gebsr__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,bsr_descr,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer)


cdef void* _hipsparseZcsr2gebsr__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2gebsr(void * handle,hipsparseDirection_t dir,int m,int n,void *const csr_descr,double2 * csr_val,const int * csr_row_ptr,const int * csr_col_ind,void *const bsr_descr,double2 * bsr_val,int * bsr_row_ptr,int * bsr_col_ind,int row_block_dim,int col_block_dim,void * p_buffer) nogil:
    global _hipsparseZcsr2gebsr__funptr
    __init_symbol(&_hipsparseZcsr2gebsr__funptr,"hipsparseZcsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,void *const,double2 *,int *,int *,int,int,void *) nogil> _hipsparseZcsr2gebsr__funptr)(handle,dir,m,n,csr_descr,csr_val,csr_row_ptr,csr_col_ind,bsr_descr,bsr_val,bsr_row_ptr,bsr_col_ind,row_block_dim,col_block_dim,p_buffer)


cdef void* _hipsparseScsr2bsr__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse CSR matrix into a sparse BSR matrix
# 
#   \details
#   \p hipsparseXcsr2bsr converts a CSR matrix into a BSR matrix. It is assumed,
#   that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
#   for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
#   the BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
#   \p csr2bsr_nnz() which also fills in \p bsr_row_ptr.
# 
#   \p hipsparseXcsr2bsr requires extra temporary storage that is allocated internally if
#   \p block_dim>16
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,float * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil:
    global _hipsparseScsr2bsr__funptr
    __init_symbol(&_hipsparseScsr2bsr__funptr,"hipsparseScsr2bsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,const int *,const int *,int,void *const,float *,int *,int *) nogil> _hipsparseScsr2bsr__funptr)(handle,dirA,m,n,descrA,csrValA,csrRowPtrA,csrColIndA,blockDim,descrC,bsrValC,bsrRowPtrC,bsrColIndC)


cdef void* _hipsparseDcsr2bsr__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,double * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil:
    global _hipsparseDcsr2bsr__funptr
    __init_symbol(&_hipsparseDcsr2bsr__funptr,"hipsparseDcsr2bsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,const int *,const int *,int,void *const,double *,int *,int *) nogil> _hipsparseDcsr2bsr__funptr)(handle,dirA,m,n,descrA,csrValA,csrRowPtrA,csrColIndA,blockDim,descrC,bsrValC,bsrRowPtrC,bsrColIndC)


cdef void* _hipsparseCcsr2bsr__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,float2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil:
    global _hipsparseCcsr2bsr__funptr
    __init_symbol(&_hipsparseCcsr2bsr__funptr,"hipsparseCcsr2bsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,void *const,float2 *,int *,int *) nogil> _hipsparseCcsr2bsr__funptr)(handle,dirA,m,n,descrA,csrValA,csrRowPtrA,csrColIndA,blockDim,descrC,bsrValC,bsrRowPtrC,bsrColIndC)


cdef void* _hipsparseZcsr2bsr__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2bsr(void * handle,hipsparseDirection_t dirA,int m,int n,void *const descrA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,int blockDim,void *const descrC,double2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC) nogil:
    global _hipsparseZcsr2bsr__funptr
    __init_symbol(&_hipsparseZcsr2bsr__funptr,"hipsparseZcsr2bsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,void *const,double2 *,int *,int *) nogil> _hipsparseZcsr2bsr__funptr)(handle,dirA,m,n,descrA,csrValA,csrRowPtrA,csrColIndA,blockDim,descrC,bsrValC,bsrRowPtrC,bsrColIndC)


cdef void* _hipsparseSbsr2csr__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse BSR matrix into a sparse CSR matrix
# 
#   \details
#   \p hipsparseXbsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
#   that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
#   for \p csr_row_ptr is computed by the number of block rows multiplied by the block
#   dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
#   the number of blocks in the BSR matrix multiplied by the block dimension squared.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,float * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseSbsr2csr__funptr
    __init_symbol(&_hipsparseSbsr2csr__funptr,"hipsparseSbsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,const int *,const int *,int,void *const,float *,int *,int *) nogil> _hipsparseSbsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseDbsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,double * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseDbsr2csr__funptr
    __init_symbol(&_hipsparseDbsr2csr__funptr,"hipsparseDbsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,const int *,const int *,int,void *const,double *,int *,int *) nogil> _hipsparseDbsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseCbsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseCbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,float2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseCbsr2csr__funptr
    __init_symbol(&_hipsparseCbsr2csr__funptr,"hipsparseCbsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,void *const,float2 *,int *,int *) nogil> _hipsparseCbsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseZbsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseZbsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int blockDim,void *const descrC,double2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseZbsr2csr__funptr
    __init_symbol(&_hipsparseZbsr2csr__funptr,"hipsparseZbsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,void *const,double2 *,int *,int *) nogil> _hipsparseZbsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,blockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseSgebsr2csr__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse general BSR matrix into a sparse CSR matrix
# 
#   \details
#   \p hipsparseXgebsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
#   that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
#   for \p csr_row_ptr is computed by the number of block rows multiplied by the block
#   dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
#   the number of blocks in the BSR matrix multiplied by the product of the block dimensions.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,float * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseSgebsr2csr__funptr
    __init_symbol(&_hipsparseSgebsr2csr__funptr,"hipsparseSgebsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const float *,const int *,const int *,int,int,void *const,float *,int *,int *) nogil> _hipsparseSgebsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDim,colBlockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseDgebsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,double * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseDgebsr2csr__funptr
    __init_symbol(&_hipsparseDgebsr2csr__funptr,"hipsparseDgebsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,const double *,const int *,const int *,int,int,void *const,double *,int *,int *) nogil> _hipsparseDgebsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDim,colBlockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseCgebsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseCgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,float2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseCgebsr2csr__funptr
    __init_symbol(&_hipsparseCgebsr2csr__funptr,"hipsparseCgebsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,float2 *,const int *,const int *,int,int,void *const,float2 *,int *,int *) nogil> _hipsparseCgebsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDim,colBlockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseZgebsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseZgebsr2csr(void * handle,hipsparseDirection_t dirA,int mb,int nb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDim,int colBlockDim,void *const descrC,double2 * csrValC,int * csrRowPtrC,int * csrColIndC) nogil:
    global _hipsparseZgebsr2csr__funptr
    __init_symbol(&_hipsparseZgebsr2csr__funptr,"hipsparseZgebsr2csr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,void *const,double2 *,const int *,const int *,int,int,void *const,double2 *,int *,int *) nogil> _hipsparseZgebsr2csr__funptr)(handle,dirA,mb,nb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDim,colBlockDim,descrC,csrValC,csrRowPtrC,csrColIndC)


cdef void* _hipsparseScsr2csr_compress__funptr = NULL
#  \ingroup conv_module
#  \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
# 
#  \details
#  \p hipsparseXcsr2csr_compress converts a CSR matrix into a compressed CSR matrix by
#  removing entries in the input CSR matrix that are below a non-negative threshold \p tol
# 
#  \note
#  In the case of complex matrices only the magnitude of the real part of \p tol is used.
# 
# @{*/
cdef hipsparseStatus_t hipsparseScsr2csr_compress(void * handle,int m,int n,void *const descrA,const float * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,float * csrValC,int * csrColIndC,int * csrRowPtrC,float tol) nogil:
    global _hipsparseScsr2csr_compress__funptr
    __init_symbol(&_hipsparseScsr2csr_compress__funptr,"hipsparseScsr2csr_compress")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,int,const int *,float *,int *,int *,float) nogil> _hipsparseScsr2csr_compress__funptr)(handle,m,n,descrA,csrValA,csrColIndA,csrRowPtrA,nnzA,nnzPerRow,csrValC,csrColIndC,csrRowPtrC,tol)


cdef void* _hipsparseDcsr2csr_compress__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2csr_compress(void * handle,int m,int n,void *const descrA,const double * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,double * csrValC,int * csrColIndC,int * csrRowPtrC,double tol) nogil:
    global _hipsparseDcsr2csr_compress__funptr
    __init_symbol(&_hipsparseDcsr2csr_compress__funptr,"hipsparseDcsr2csr_compress")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,int,const int *,double *,int *,int *,double) nogil> _hipsparseDcsr2csr_compress__funptr)(handle,m,n,descrA,csrValA,csrColIndA,csrRowPtrA,nnzA,nnzPerRow,csrValC,csrColIndC,csrRowPtrC,tol)


cdef void* _hipsparseCcsr2csr_compress__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2csr_compress(void * handle,int m,int n,void *const descrA,float2 * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,float2 * csrValC,int * csrColIndC,int * csrRowPtrC,float2 tol) nogil:
    global _hipsparseCcsr2csr_compress__funptr
    __init_symbol(&_hipsparseCcsr2csr_compress__funptr,"hipsparseCcsr2csr_compress")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,int,const int *,float2 *,int *,int *,float2) nogil> _hipsparseCcsr2csr_compress__funptr)(handle,m,n,descrA,csrValA,csrColIndA,csrRowPtrA,nnzA,nnzPerRow,csrValC,csrColIndC,csrRowPtrC,tol)


cdef void* _hipsparseZcsr2csr_compress__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2csr_compress(void * handle,int m,int n,void *const descrA,double2 * csrValA,const int * csrColIndA,const int * csrRowPtrA,int nnzA,const int * nnzPerRow,double2 * csrValC,int * csrColIndC,int * csrRowPtrC,double2 tol) nogil:
    global _hipsparseZcsr2csr_compress__funptr
    __init_symbol(&_hipsparseZcsr2csr_compress__funptr,"hipsparseZcsr2csr_compress")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,int,const int *,double2 *,int *,int *,double2) nogil> _hipsparseZcsr2csr_compress__funptr)(handle,m,n,descrA,csrValA,csrColIndA,csrRowPtrA,nnzA,nnzPerRow,csrValC,csrColIndC,csrRowPtrC,tol)


cdef void* _hipsparseSpruneCsr2csr_bufferSize__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  \p hipsparseXpruneCsr2csr_bufferSize returns the size of the temporary buffer that
#  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
#  temporary storage buffer must be allocated by the user.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneCsr2csr_bufferSize__funptr
    __init_symbol(&_hipsparseSpruneCsr2csr_bufferSize__funptr,"hipsparseSpruneCsr2csr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,const float *,void *const,const float *,const int *,const int *,unsigned long *) nogil> _hipsparseSpruneCsr2csr_bufferSize__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrValC,csrRowPtrC,csrColIndC,bufferSize)


cdef void* _hipsparseDpruneCsr2csr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneCsr2csr_bufferSize__funptr
    __init_symbol(&_hipsparseDpruneCsr2csr_bufferSize__funptr,"hipsparseDpruneCsr2csr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,const double *,void *const,const double *,const int *,const int *,unsigned long *) nogil> _hipsparseDpruneCsr2csr_bufferSize__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrValC,csrRowPtrC,csrColIndC,bufferSize)


cdef void* _hipsparseSpruneCsr2csr_bufferSizeExt__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  \p hipsparseXpruneCsr2csr_bufferSizeExt returns the size of the temporary buffer that
#  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
#  temporary storage buffer must be allocated by the user.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneCsr2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSpruneCsr2csr_bufferSizeExt__funptr,"hipsparseSpruneCsr2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,const float *,void *const,const float *,const int *,const int *,unsigned long *) nogil> _hipsparseSpruneCsr2csr_bufferSizeExt__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrValC,csrRowPtrC,csrColIndC,bufferSize)


cdef void* _hipsparseDpruneCsr2csr_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneCsr2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDpruneCsr2csr_bufferSizeExt__funptr,"hipsparseDpruneCsr2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,const double *,void *const,const double *,const int *,const int *,unsigned long *) nogil> _hipsparseDpruneCsr2csr_bufferSizeExt__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrValC,csrRowPtrC,csrColIndC,bufferSize)


cdef void* _hipsparseSpruneCsr2csrNnz__funptr = NULL
#    \ingroup conv_module
#    \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
#    \details
#    \p hipsparseXpruneCsr2csrNnz computes the number of nonzero elements per row and the total
#    number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
#    pruned from the matrix.
# 
#    \note The routine does support asynchronous execution if the pointer mode is set to device.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csrNnz(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,void * buffer) nogil:
    global _hipsparseSpruneCsr2csrNnz__funptr
    __init_symbol(&_hipsparseSpruneCsr2csrNnz__funptr,"hipsparseSpruneCsr2csrNnz")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,const float *,void *const,int *,int *,void *) nogil> _hipsparseSpruneCsr2csrNnz__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrRowPtrC,nnzTotalDevHostPtr,buffer)


cdef void* _hipsparseDpruneCsr2csrNnz__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csrNnz(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,void * buffer) nogil:
    global _hipsparseDpruneCsr2csrNnz__funptr
    __init_symbol(&_hipsparseDpruneCsr2csrNnz__funptr,"hipsparseDpruneCsr2csrNnz")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,const double *,void *const,int *,int *,void *) nogil> _hipsparseDpruneCsr2csrNnz__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrRowPtrC,nnzTotalDevHostPtr,buffer)


cdef void* _hipsparseSpruneCsr2csr__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
#  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
#  The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
#  by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
#  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
#  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
#  in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
#  is executed asynchronously with respect to the host and may return control to the application on the host
#  before the entire result is ready.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csr(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * threshold,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC,void * buffer) nogil:
    global _hipsparseSpruneCsr2csr__funptr
    __init_symbol(&_hipsparseSpruneCsr2csr__funptr,"hipsparseSpruneCsr2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,const float *,void *const,float *,const int *,int *,void *) nogil> _hipsparseSpruneCsr2csr__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrValC,csrRowPtrC,csrColIndC,buffer)


cdef void* _hipsparseDpruneCsr2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csr(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * threshold,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC,void * buffer) nogil:
    global _hipsparseDpruneCsr2csr__funptr
    __init_symbol(&_hipsparseDpruneCsr2csr__funptr,"hipsparseDpruneCsr2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,const double *,void *const,double *,const int *,int *,void *) nogil> _hipsparseDpruneCsr2csr__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,threshold,descrC,csrValC,csrRowPtrC,csrColIndC,buffer)


cdef void* _hipsparseSpruneCsr2csrByPercentage_bufferSize__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  \p hipsparseXpruneCsr2csrByPercentage_bufferSize returns the size of the temporary buffer that
#  is required by \p hipsparseXpruneCsr2csrNnzByPercentage.
#  The temporary storage buffer must be allocated by the user.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneCsr2csrByPercentage_bufferSize__funptr
    __init_symbol(&_hipsparseSpruneCsr2csrByPercentage_bufferSize__funptr,"hipsparseSpruneCsr2csrByPercentage_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,float,void *const,const float *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseSpruneCsr2csrByPercentage_bufferSize__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrValC,csrRowPtrC,csrColIndC,info,bufferSize)


cdef void* _hipsparseDpruneCsr2csrByPercentage_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneCsr2csrByPercentage_bufferSize__funptr
    __init_symbol(&_hipsparseDpruneCsr2csrByPercentage_bufferSize__funptr,"hipsparseDpruneCsr2csrByPercentage_bufferSize")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,double,void *const,const double *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseDpruneCsr2csrByPercentage_bufferSize__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrValC,csrRowPtrC,csrColIndC,info,bufferSize)


cdef void* _hipsparseSpruneCsr2csrByPercentage_bufferSizeExt__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  \p hipsparseXpruneCsr2csrByPercentage_bufferSizeExt returns the size of the temporary buffer that
#  is required by \p hipsparseXpruneCsr2csrNnzByPercentage.
#  The temporary storage buffer must be allocated by the user.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,const float * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseSpruneCsr2csrByPercentage_bufferSizeExt__funptr
    __init_symbol(&_hipsparseSpruneCsr2csrByPercentage_bufferSizeExt__funptr,"hipsparseSpruneCsr2csrByPercentage_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,float,void *const,const float *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseSpruneCsr2csrByPercentage_bufferSizeExt__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrValC,csrRowPtrC,csrColIndC,info,bufferSize)


cdef void* _hipsparseDpruneCsr2csrByPercentage_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,const double * csrValC,const int * csrRowPtrC,const int * csrColIndC,pruneInfo_t info,unsigned long * bufferSize) nogil:
    global _hipsparseDpruneCsr2csrByPercentage_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDpruneCsr2csrByPercentage_bufferSizeExt__funptr,"hipsparseDpruneCsr2csrByPercentage_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,double,void *const,const double *,const int *,const int *,pruneInfo_t,unsigned long *) nogil> _hipsparseDpruneCsr2csrByPercentage_bufferSizeExt__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrValC,csrRowPtrC,csrColIndC,info,bufferSize)


cdef void* _hipsparseSpruneCsr2csrNnzByPercentage__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  \p hipsparseXpruneCsr2csrNnzByPercentage computes the number of nonzero elements per row and the total
#  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
#  pruned from the matrix.
# 
#  \note The routine does support asynchronous execution if the pointer mode is set to device.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseSpruneCsr2csrNnzByPercentage__funptr
    __init_symbol(&_hipsparseSpruneCsr2csrNnzByPercentage__funptr,"hipsparseSpruneCsr2csrNnzByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,float,void *const,int *,int *,pruneInfo_t,void *) nogil> _hipsparseSpruneCsr2csrNnzByPercentage__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrRowPtrC,nnzTotalDevHostPtr,info,buffer)


cdef void* _hipsparseDpruneCsr2csrNnzByPercentage__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,int * csrRowPtrC,int * nnzTotalDevHostPtr,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseDpruneCsr2csrNnzByPercentage__funptr
    __init_symbol(&_hipsparseDpruneCsr2csrNnzByPercentage__funptr,"hipsparseDpruneCsr2csrNnzByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,double,void *const,int *,int *,pruneInfo_t,void *) nogil> _hipsparseDpruneCsr2csrNnzByPercentage__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrRowPtrC,nnzTotalDevHostPtr,info,buffer)


cdef void* _hipsparseSpruneCsr2csrByPercentage__funptr = NULL
#  \ingroup conv_module
#  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
# 
#  \details
#  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
#  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
#  The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
#  by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
#  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
#  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
#  in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
#  is executed asynchronously with respect to the host and may return control to the application on the host
#  before the entire result is ready.
# 
# @{*/
cdef hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,float percentage,void *const descrC,float * csrValC,const int * csrRowPtrC,int * csrColIndC,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseSpruneCsr2csrByPercentage__funptr
    __init_symbol(&_hipsparseSpruneCsr2csrByPercentage__funptr,"hipsparseSpruneCsr2csrByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const float *,const int *,const int *,float,void *const,float *,const int *,int *,pruneInfo_t,void *) nogil> _hipsparseSpruneCsr2csrByPercentage__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrValC,csrRowPtrC,csrColIndC,info,buffer)


cdef void* _hipsparseDpruneCsr2csrByPercentage__funptr = NULL
cdef hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(void * handle,int m,int n,int nnzA,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,double percentage,void *const descrC,double * csrValC,const int * csrRowPtrC,int * csrColIndC,pruneInfo_t info,void * buffer) nogil:
    global _hipsparseDpruneCsr2csrByPercentage__funptr
    __init_symbol(&_hipsparseDpruneCsr2csrByPercentage__funptr,"hipsparseDpruneCsr2csrByPercentage")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const double *,const int *,const int *,double,void *const,double *,const int *,int *,pruneInfo_t,void *) nogil> _hipsparseDpruneCsr2csrByPercentage__funptr)(handle,m,n,nnzA,descrA,csrValA,csrRowPtrA,csrColIndA,percentage,descrC,csrValC,csrRowPtrC,csrColIndC,info,buffer)


cdef void* _hipsparseShyb2csr__funptr = NULL
#    \ingroup conv_module
#   \brief Convert a sparse HYB matrix into a sparse CSR matrix
# 
#   \details
#   \p hipsparseXhyb2csr converts a HYB matrix into a CSR matrix.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseShyb2csr(void * handle,void *const descrA,void *const hybA,float * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil:
    global _hipsparseShyb2csr__funptr
    __init_symbol(&_hipsparseShyb2csr__funptr,"hipsparseShyb2csr")
    return (<hipsparseStatus_t (*)(void *,void *const,void *const,float *,int *,int *) nogil> _hipsparseShyb2csr__funptr)(handle,descrA,hybA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA)


cdef void* _hipsparseDhyb2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDhyb2csr(void * handle,void *const descrA,void *const hybA,double * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil:
    global _hipsparseDhyb2csr__funptr
    __init_symbol(&_hipsparseDhyb2csr__funptr,"hipsparseDhyb2csr")
    return (<hipsparseStatus_t (*)(void *,void *const,void *const,double *,int *,int *) nogil> _hipsparseDhyb2csr__funptr)(handle,descrA,hybA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA)


cdef void* _hipsparseChyb2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseChyb2csr(void * handle,void *const descrA,void *const hybA,float2 * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil:
    global _hipsparseChyb2csr__funptr
    __init_symbol(&_hipsparseChyb2csr__funptr,"hipsparseChyb2csr")
    return (<hipsparseStatus_t (*)(void *,void *const,void *const,float2 *,int *,int *) nogil> _hipsparseChyb2csr__funptr)(handle,descrA,hybA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA)


cdef void* _hipsparseZhyb2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseZhyb2csr(void * handle,void *const descrA,void *const hybA,double2 * csrSortedValA,int * csrSortedRowPtrA,int * csrSortedColIndA) nogil:
    global _hipsparseZhyb2csr__funptr
    __init_symbol(&_hipsparseZhyb2csr__funptr,"hipsparseZhyb2csr")
    return (<hipsparseStatus_t (*)(void *,void *const,void *const,double2 *,int *,int *) nogil> _hipsparseZhyb2csr__funptr)(handle,descrA,hybA,csrSortedValA,csrSortedRowPtrA,csrSortedColIndA)


cdef void* _hipsparseXcoo2csr__funptr = NULL
#    \ingroup conv_module
#    \brief Convert a sparse COO matrix into a sparse CSR matrix
# 
#    \details
#    \p hipsparseXcoo2csr converts the COO array containing the row indices into a
#    CSR array of row offsets, that point to the start of every row.
#    It is assumed that the COO row index array is sorted.
# 
#    \note It can also be used, to convert a COO array containing the column indices into
#    a CSC array of column offsets, that point to the start of every column. Then, it is
#    assumed that the COO column index array is sorted, instead.
# 
#    \note
#    This function is non blocking and executed asynchronously with respect to the host.
#    It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseXcoo2csr(void * handle,const int * cooRowInd,int nnz,int m,int * csrRowPtr,hipsparseIndexBase_t idxBase) nogil:
    global _hipsparseXcoo2csr__funptr
    __init_symbol(&_hipsparseXcoo2csr__funptr,"hipsparseXcoo2csr")
    return (<hipsparseStatus_t (*)(void *,const int *,int,int,int *,hipsparseIndexBase_t) nogil> _hipsparseXcoo2csr__funptr)(handle,cooRowInd,nnz,m,csrRowPtr,idxBase)


cdef void* _hipsparseCreateIdentityPermutation__funptr = NULL
#    \ingroup conv_module
#   \brief Create the identity map
# 
#   \details
#   \p hipsparseCreateIdentityPermutation stores the identity map in \p p, such that
#   \f$p = 0:1:(n-1)\f$.
# 
#   \code{.c}
#       for(i = 0; i < n; ++i)
#       {
#           p[i] = i;
#       }
#   \endcode
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseCreateIdentityPermutation(void * handle,int n,int * p) nogil:
    global _hipsparseCreateIdentityPermutation__funptr
    __init_symbol(&_hipsparseCreateIdentityPermutation__funptr,"hipsparseCreateIdentityPermutation")
    return (<hipsparseStatus_t (*)(void *,int,int *) nogil> _hipsparseCreateIdentityPermutation__funptr)(handle,n,p)


cdef void* _hipsparseXcsrsort_bufferSizeExt__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse CSR matrix
# 
#   \details
#   \p hipsparseXcsrsort_bufferSizeExt returns the size of the temporary storage buffer
#   required by hipsparseXcsrsort(). The temporary storage buffer must be allocated by
#   the user.
# /
cdef hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(void * handle,int m,int n,int nnz,const int * csrRowPtr,const int * csrColInd,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseXcsrsort_bufferSizeExt__funptr
    __init_symbol(&_hipsparseXcsrsort_bufferSizeExt__funptr,"hipsparseXcsrsort_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const int *,const int *,unsigned long *) nogil> _hipsparseXcsrsort_bufferSizeExt__funptr)(handle,m,n,nnz,csrRowPtr,csrColInd,pBufferSizeInBytes)


cdef void* _hipsparseXcsrsort__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse CSR matrix
# 
#   \details
#   \p hipsparseXcsrsort sorts a matrix in CSR format. The sorted permutation vector
#   \p perm can be used to obtain sorted \p csr_val array. In this case, \p perm must be
#   initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
# 
#   \p hipsparseXcsrsort requires extra temporary storage buffer that has to be allocated by
#   the user. Storage buffer size can be determined by hipsparseXcsrsort_bufferSizeExt().
# 
#   \note
#   \p perm can be \p NULL if a sorted permutation vector is not required.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseXcsrsort(void * handle,int m,int n,int nnz,void *const descrA,const int * csrRowPtr,int * csrColInd,int * P,void * pBuffer) nogil:
    global _hipsparseXcsrsort__funptr
    __init_symbol(&_hipsparseXcsrsort__funptr,"hipsparseXcsrsort")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const int *,int *,int *,void *) nogil> _hipsparseXcsrsort__funptr)(handle,m,n,nnz,descrA,csrRowPtr,csrColInd,P,pBuffer)


cdef void* _hipsparseXcscsort_bufferSizeExt__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse CSC matrix
# 
#   \details
#   \p hipsparseXcscsort_bufferSizeExt returns the size of the temporary storage buffer
#   required by hipsparseXcscsort(). The temporary storage buffer must be allocated by
#   the user.
# /
cdef hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(void * handle,int m,int n,int nnz,const int * cscColPtr,const int * cscRowInd,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseXcscsort_bufferSizeExt__funptr
    __init_symbol(&_hipsparseXcscsort_bufferSizeExt__funptr,"hipsparseXcscsort_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const int *,const int *,unsigned long *) nogil> _hipsparseXcscsort_bufferSizeExt__funptr)(handle,m,n,nnz,cscColPtr,cscRowInd,pBufferSizeInBytes)


cdef void* _hipsparseXcscsort__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse CSC matrix
# 
#   \details
#   \p hipsparseXcscsort sorts a matrix in CSC format. The sorted permutation vector
#   \p perm can be used to obtain sorted \p csc_val array. In this case, \p perm must be
#   initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
# 
#   \p hipsparseXcscsort requires extra temporary storage buffer that has to be allocated by
#   the user. Storage buffer size can be determined by hipsparseXcscsort_bufferSizeExt().
# 
#   \note
#   \p perm can be \p NULL if a sorted permutation vector is not required.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseXcscsort(void * handle,int m,int n,int nnz,void *const descrA,const int * cscColPtr,int * cscRowInd,int * P,void * pBuffer) nogil:
    global _hipsparseXcscsort__funptr
    __init_symbol(&_hipsparseXcscsort__funptr,"hipsparseXcscsort")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,const int *,int *,int *,void *) nogil> _hipsparseXcscsort__funptr)(handle,m,n,nnz,descrA,cscColPtr,cscRowInd,P,pBuffer)


cdef void* _hipsparseXcoosort_bufferSizeExt__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse COO matrix
# 
#   \details
#   \p hipsparseXcoosort_bufferSizeExt returns the size of the temporary storage buffer
#   required by hipsparseXcoosort(). The temporary storage buffer must be allocated by
#   the user.
# /
cdef hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(void * handle,int m,int n,int nnz,const int * cooRows,const int * cooCols,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseXcoosort_bufferSizeExt__funptr
    __init_symbol(&_hipsparseXcoosort_bufferSizeExt__funptr,"hipsparseXcoosort_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,const int *,const int *,unsigned long *) nogil> _hipsparseXcoosort_bufferSizeExt__funptr)(handle,m,n,nnz,cooRows,cooCols,pBufferSizeInBytes)


cdef void* _hipsparseXcoosortByRow__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse COO matrix by row
# 
#   \details
#   \p hipsparseXcoosortByRow sorts a matrix in COO format by row. The sorted
#   permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
#   case, \p perm must be initialized as the identity permutation, see
#   hipsparseCreateIdentityPermutation().
# 
#   \p hipsparseXcoosortByRow requires extra temporary storage buffer that has to be
#   allocated by the user. Storage buffer size can be determined by
#   hipsparseXcoosort_bufferSizeExt().
# 
#   \note
#   \p perm can be \p NULL if a sorted permutation vector is not required.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseXcoosortByRow(void * handle,int m,int n,int nnz,int * cooRows,int * cooCols,int * P,void * pBuffer) nogil:
    global _hipsparseXcoosortByRow__funptr
    __init_symbol(&_hipsparseXcoosortByRow__funptr,"hipsparseXcoosortByRow")
    return (<hipsparseStatus_t (*)(void *,int,int,int,int *,int *,int *,void *) nogil> _hipsparseXcoosortByRow__funptr)(handle,m,n,nnz,cooRows,cooCols,P,pBuffer)


cdef void* _hipsparseXcoosortByColumn__funptr = NULL
#    \ingroup conv_module
#   \brief Sort a sparse COO matrix by column
# 
#   \details
#   \p hipsparseXcoosortByColumn sorts a matrix in COO format by column. The sorted
#   permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
#   case, \p perm must be initialized as the identity permutation, see
#   hipsparseCreateIdentityPermutation().
# 
#   \p hipsparseXcoosortByColumn requires extra temporary storage buffer that has to be
#   allocated by the user. Storage buffer size can be determined by
#   hipsparseXcoosort_bufferSizeExt().
# 
#   \note
#   \p perm can be \p NULL if a sorted permutation vector is not required.
# 
#   \note
#   This function is non blocking and executed asynchronously with respect to the host.
#   It may return before the actual computation has finished.
# /
cdef hipsparseStatus_t hipsparseXcoosortByColumn(void * handle,int m,int n,int nnz,int * cooRows,int * cooCols,int * P,void * pBuffer) nogil:
    global _hipsparseXcoosortByColumn__funptr
    __init_symbol(&_hipsparseXcoosortByColumn__funptr,"hipsparseXcoosortByColumn")
    return (<hipsparseStatus_t (*)(void *,int,int,int,int *,int *,int *,void *) nogil> _hipsparseXcoosortByColumn__funptr)(handle,m,n,nnz,cooRows,cooCols,P,pBuffer)


cdef void* _hipsparseSgebsr2gebsr_bufferSize__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
#   general BSR matrix to another sparse general BSR matrix.
# 
#   \details
#   \p hipsparseXgebsr2gebsr_bufferSize returns the size of the temporary storage buffer
#   that is required by hipsparseXgebsr2gebsrNnz() and hipsparseXgebsr2gebsr().
#   The temporary storage buffer must be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil:
    global _hipsparseSgebsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseSgebsr2gebsr_bufferSize__funptr,"hipsparseSgebsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,const float *,const int *,const int *,int,int,int,int,int *) nogil> _hipsparseSgebsr2gebsr_bufferSize__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,bufferSize)


cdef void* _hipsparseDgebsr2gebsr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil:
    global _hipsparseDgebsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseDgebsr2gebsr_bufferSize__funptr,"hipsparseDgebsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,const double *,const int *,const int *,int,int,int,int,int *) nogil> _hipsparseDgebsr2gebsr_bufferSize__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,bufferSize)


cdef void* _hipsparseCgebsr2gebsr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil:
    global _hipsparseCgebsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseCgebsr2gebsr_bufferSize__funptr,"hipsparseCgebsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,float2 *,const int *,const int *,int,int,int,int,int *) nogil> _hipsparseCgebsr2gebsr_bufferSize__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,bufferSize)


cdef void* _hipsparseZgebsr2gebsr_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,int rowBlockDimC,int colBlockDimC,int * bufferSize) nogil:
    global _hipsparseZgebsr2gebsr_bufferSize__funptr
    __init_symbol(&_hipsparseZgebsr2gebsr_bufferSize__funptr,"hipsparseZgebsr2gebsr_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,double2 *,const int *,const int *,int,int,int,int,int *) nogil> _hipsparseZgebsr2gebsr_bufferSize__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,rowBlockDimC,colBlockDimC,bufferSize)


cdef void* _hipsparseXgebsr2gebsrNnz__funptr = NULL
#    \ingroup conv_module
#   \brief This function is used when converting a general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
#   Specifically, this function determines the number of non-zero blocks that will exist in \p C (stored using either a host
#   or device pointer), and computes the row pointer array for \p C.
# 
#   \details
#   The routine does support asynchronous execution.
# /
cdef hipsparseStatus_t hipsparseXgebsr2gebsrNnz(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,int * bsrRowPtrC,int rowBlockDimC,int colBlockDimC,int * nnzTotalDevHostPtr,void * buffer) nogil:
    global _hipsparseXgebsr2gebsrNnz__funptr
    __init_symbol(&_hipsparseXgebsr2gebsrNnz__funptr,"hipsparseXgebsr2gebsrNnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,const int *,const int *,int,int,void *const,int *,int,int,int *,void *) nogil> _hipsparseXgebsr2gebsrNnz__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,descrC,bsrRowPtrC,rowBlockDimC,colBlockDimC,nnzTotalDevHostPtr,buffer)


cdef void* _hipsparseSgebsr2gebsr__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts the general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
# 
#   \details
#   The conversion uses three steps. First, the user calls hipsparseXgebsr2gebsr_bufferSize() to determine the size of
#   the required temporary storage buffer. The user then allocates this buffer. Secondly, the user then allocates \p mb_C+1
#   integers for the row pointer array for \p C where \p mb_C=(m+row_block_dim_C-1)/row_block_dim_C. The user then calls
#   hipsparseXgebsr2gebsrNnz() to fill in the row pointer array for \p C ( \p bsr_row_ptr_C ) and determine the number of
#   non-zero blocks that will exist in \p C. Finally, the user allocates space for the colimn indices array of \p C to have
#   \p nnzb_C elements and space for the values array of \p C to have \p nnzb_C*roc_block_dim_C*col_block_dim_C and then calls
#   hipsparseXgebsr2gebsr() to complete the conversion.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseSgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const float * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,float * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil:
    global _hipsparseSgebsr2gebsr__funptr
    __init_symbol(&_hipsparseSgebsr2gebsr__funptr,"hipsparseSgebsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,const float *,const int *,const int *,int,int,void *const,float *,int *,int *,int,int,void *) nogil> _hipsparseSgebsr2gebsr__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,descrC,bsrValC,bsrRowPtrC,bsrColIndC,rowBlockDimC,colBlockDimC,buffer)


cdef void* _hipsparseDgebsr2gebsr__funptr = NULL
cdef hipsparseStatus_t hipsparseDgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,const double * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,double * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil:
    global _hipsparseDgebsr2gebsr__funptr
    __init_symbol(&_hipsparseDgebsr2gebsr__funptr,"hipsparseDgebsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,const double *,const int *,const int *,int,int,void *const,double *,int *,int *,int,int,void *) nogil> _hipsparseDgebsr2gebsr__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,descrC,bsrValC,bsrRowPtrC,bsrColIndC,rowBlockDimC,colBlockDimC,buffer)


cdef void* _hipsparseCgebsr2gebsr__funptr = NULL
cdef hipsparseStatus_t hipsparseCgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,float2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,float2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil:
    global _hipsparseCgebsr2gebsr__funptr
    __init_symbol(&_hipsparseCgebsr2gebsr__funptr,"hipsparseCgebsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,float2 *,const int *,const int *,int,int,void *const,float2 *,int *,int *,int,int,void *) nogil> _hipsparseCgebsr2gebsr__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,descrC,bsrValC,bsrRowPtrC,bsrColIndC,rowBlockDimC,colBlockDimC,buffer)


cdef void* _hipsparseZgebsr2gebsr__funptr = NULL
cdef hipsparseStatus_t hipsparseZgebsr2gebsr(void * handle,hipsparseDirection_t dirA,int mb,int nb,int nnzb,void *const descrA,double2 * bsrValA,const int * bsrRowPtrA,const int * bsrColIndA,int rowBlockDimA,int colBlockDimA,void *const descrC,double2 * bsrValC,int * bsrRowPtrC,int * bsrColIndC,int rowBlockDimC,int colBlockDimC,void * buffer) nogil:
    global _hipsparseZgebsr2gebsr__funptr
    __init_symbol(&_hipsparseZgebsr2gebsr__funptr,"hipsparseZgebsr2gebsr")
    return (<hipsparseStatus_t (*)(void *,hipsparseDirection_t,int,int,int,void *const,double2 *,const int *,const int *,int,int,void *const,double2 *,int *,int *,int,int,void *) nogil> _hipsparseZgebsr2gebsr__funptr)(handle,dirA,mb,nb,nnzb,descrA,bsrValA,bsrRowPtrA,bsrColIndA,rowBlockDimA,colBlockDimA,descrC,bsrValC,bsrRowPtrC,bsrColIndC,rowBlockDimC,colBlockDimC,buffer)


cdef void* _hipsparseScsru2csr_bufferSizeExt__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function calculates the amount of temporary storage required for
#   hipsparseXcsru2csr() and hipsparseXcsr2csru().
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,float * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseScsru2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseScsru2csr_bufferSizeExt__funptr,"hipsparseScsru2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float *,const int *,int *,csru2csrInfo_t,unsigned long *) nogil> _hipsparseScsru2csr_bufferSizeExt__funptr)(handle,m,n,nnz,csrVal,csrRowPtr,csrColInd,info,pBufferSizeInBytes)


cdef void* _hipsparseDcsru2csr_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,double * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseDcsru2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseDcsru2csr_bufferSizeExt__funptr,"hipsparseDcsru2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double *,const int *,int *,csru2csrInfo_t,unsigned long *) nogil> _hipsparseDcsru2csr_bufferSizeExt__funptr)(handle,m,n,nnz,csrVal,csrRowPtr,csrColInd,info,pBufferSizeInBytes)


cdef void* _hipsparseCcsru2csr_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,float2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseCcsru2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseCcsru2csr_bufferSizeExt__funptr,"hipsparseCcsru2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,float2 *,const int *,int *,csru2csrInfo_t,unsigned long *) nogil> _hipsparseCcsru2csr_bufferSizeExt__funptr)(handle,m,n,nnz,csrVal,csrRowPtr,csrColInd,info,pBufferSizeInBytes)


cdef void* _hipsparseZcsru2csr_bufferSizeExt__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(void * handle,int m,int n,int nnz,double2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,unsigned long * pBufferSizeInBytes) nogil:
    global _hipsparseZcsru2csr_bufferSizeExt__funptr
    __init_symbol(&_hipsparseZcsru2csr_bufferSizeExt__funptr,"hipsparseZcsru2csr_bufferSizeExt")
    return (<hipsparseStatus_t (*)(void *,int,int,int,double2 *,const int *,int *,csru2csrInfo_t,unsigned long *) nogil> _hipsparseZcsru2csr_bufferSizeExt__funptr)(handle,m,n,nnz,csrVal,csrRowPtr,csrColInd,info,pBufferSizeInBytes)


cdef void* _hipsparseScsru2csr__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts unsorted CSR format to sorted CSR format. The required
#   temporary storage has to be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsru2csr(void * handle,int m,int n,int nnz,void *const descrA,float * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseScsru2csr__funptr
    __init_symbol(&_hipsparseScsru2csr__funptr,"hipsparseScsru2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,float *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseScsru2csr__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseDcsru2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsru2csr(void * handle,int m,int n,int nnz,void *const descrA,double * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseDcsru2csr__funptr
    __init_symbol(&_hipsparseDcsru2csr__funptr,"hipsparseDcsru2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,double *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseDcsru2csr__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseCcsru2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsru2csr(void * handle,int m,int n,int nnz,void *const descrA,float2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseCcsru2csr__funptr
    __init_symbol(&_hipsparseCcsru2csr__funptr,"hipsparseCcsru2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,float2 *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseCcsru2csr__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseZcsru2csr__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsru2csr(void * handle,int m,int n,int nnz,void *const descrA,double2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseZcsru2csr__funptr
    __init_symbol(&_hipsparseZcsru2csr__funptr,"hipsparseZcsru2csr")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,double2 *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseZcsru2csr__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseScsr2csru__funptr = NULL
#    \ingroup conv_module
#   \brief
#   This function converts sorted CSR format to unsorted CSR format. The required
#   temporary storage has to be allocated by the user.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsr2csru(void * handle,int m,int n,int nnz,void *const descrA,float * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseScsr2csru__funptr
    __init_symbol(&_hipsparseScsr2csru__funptr,"hipsparseScsr2csru")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,float *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseScsr2csru__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseDcsr2csru__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsr2csru(void * handle,int m,int n,int nnz,void *const descrA,double * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseDcsr2csru__funptr
    __init_symbol(&_hipsparseDcsr2csru__funptr,"hipsparseDcsr2csru")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,double *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseDcsr2csru__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseCcsr2csru__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsr2csru(void * handle,int m,int n,int nnz,void *const descrA,float2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseCcsr2csru__funptr
    __init_symbol(&_hipsparseCcsr2csru__funptr,"hipsparseCcsr2csru")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,float2 *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseCcsr2csru__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseZcsr2csru__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsr2csru(void * handle,int m,int n,int nnz,void *const descrA,double2 * csrVal,const int * csrRowPtr,int * csrColInd,csru2csrInfo_t info,void * pBuffer) nogil:
    global _hipsparseZcsr2csru__funptr
    __init_symbol(&_hipsparseZcsr2csru__funptr,"hipsparseZcsr2csru")
    return (<hipsparseStatus_t (*)(void *,int,int,int,void *const,double2 *,const int *,int *,csru2csrInfo_t,void *) nogil> _hipsparseZcsr2csru__funptr)(handle,m,n,nnz,descrA,csrVal,csrRowPtr,csrColInd,info,pBuffer)


cdef void* _hipsparseScsrcolor__funptr = NULL
#    \ingroup reordering_module
#   \brief Coloring of the adjacency graph of the matrix \f$A\f$ stored in the CSR format.
# 
#   \details
#   \p hipsparseXcsrcolor performs the coloring of the undirected graph represented by the (symmetric) sparsity pattern of the matrix \f$A\f$ stored in CSR format. Graph coloring is a way of coloring the nodes of a graph such that no two adjacent nodes are of the same color. The \p fraction_to_color is a parameter to only color a given percentage of the graph nodes, the remaining uncolored nodes receive distinct new colors. The optional \p reordering array is a permutation array such that unknowns of the same color are grouped. The matrix \f$A\f$ must be stored as a general matrix with a symmetric sparsity pattern, and if the matrix \f$A\f$ is non-symmetric then the user is responsible to provide the symmetric part \f$\frac{A+A^T}{2}\f$.
# /
#   @{*/
cdef hipsparseStatus_t hipsparseScsrcolor(void * handle,int m,int nnz,void *const descrA,const float * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil:
    global _hipsparseScsrcolor__funptr
    __init_symbol(&_hipsparseScsrcolor__funptr,"hipsparseScsrcolor")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const float *,const int *,const int *,const float *,int *,int *,int *,void *) nogil> _hipsparseScsrcolor__funptr)(handle,m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,fractionToColor,ncolors,coloring,reordering,info)


cdef void* _hipsparseDcsrcolor__funptr = NULL
cdef hipsparseStatus_t hipsparseDcsrcolor(void * handle,int m,int nnz,void *const descrA,const double * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil:
    global _hipsparseDcsrcolor__funptr
    __init_symbol(&_hipsparseDcsrcolor__funptr,"hipsparseDcsrcolor")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,const double *,const int *,const int *,const double *,int *,int *,int *,void *) nogil> _hipsparseDcsrcolor__funptr)(handle,m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,fractionToColor,ncolors,coloring,reordering,info)


cdef void* _hipsparseCcsrcolor__funptr = NULL
cdef hipsparseStatus_t hipsparseCcsrcolor(void * handle,int m,int nnz,void *const descrA,float2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,const float * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil:
    global _hipsparseCcsrcolor__funptr
    __init_symbol(&_hipsparseCcsrcolor__funptr,"hipsparseCcsrcolor")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,float2 *,const int *,const int *,const float *,int *,int *,int *,void *) nogil> _hipsparseCcsrcolor__funptr)(handle,m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,fractionToColor,ncolors,coloring,reordering,info)


cdef void* _hipsparseZcsrcolor__funptr = NULL
cdef hipsparseStatus_t hipsparseZcsrcolor(void * handle,int m,int nnz,void *const descrA,double2 * csrValA,const int * csrRowPtrA,const int * csrColIndA,const double * fractionToColor,int * ncolors,int * coloring,int * reordering,void * info) nogil:
    global _hipsparseZcsrcolor__funptr
    __init_symbol(&_hipsparseZcsrcolor__funptr,"hipsparseZcsrcolor")
    return (<hipsparseStatus_t (*)(void *,int,int,void *const,double2 *,const int *,const int *,const double *,int *,int *,int *,void *) nogil> _hipsparseZcsrcolor__funptr)(handle,m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,fractionToColor,ncolors,coloring,reordering,info)


cdef void* _hipsparseCreateSpVec__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateSpVec(void ** spVecDescr,long size,long nnz,void * indices,void * values,hipsparseIndexType_t idxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil:
    global _hipsparseCreateSpVec__funptr
    __init_symbol(&_hipsparseCreateSpVec__funptr,"hipsparseCreateSpVec")
    return (<hipsparseStatus_t (*)(void **,long,long,void *,void *,hipsparseIndexType_t,hipsparseIndexBase_t,hipDataType) nogil> _hipsparseCreateSpVec__funptr)(spVecDescr,size,nnz,indices,values,idxType,idxBase,valueType)


cdef void* _hipsparseDestroySpVec__funptr = NULL
cdef hipsparseStatus_t hipsparseDestroySpVec(void * spVecDescr) nogil:
    global _hipsparseDestroySpVec__funptr
    __init_symbol(&_hipsparseDestroySpVec__funptr,"hipsparseDestroySpVec")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroySpVec__funptr)(spVecDescr)


cdef void* _hipsparseSpVecGet__funptr = NULL
cdef hipsparseStatus_t hipsparseSpVecGet(void *const spVecDescr,long * size,long * nnz,void ** indices,void ** values,hipsparseIndexType_t * idxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil:
    global _hipsparseSpVecGet__funptr
    __init_symbol(&_hipsparseSpVecGet__funptr,"hipsparseSpVecGet")
    return (<hipsparseStatus_t (*)(void *const,long *,long *,void **,void **,hipsparseIndexType_t *,hipsparseIndexBase_t *,hipDataType *) nogil> _hipsparseSpVecGet__funptr)(spVecDescr,size,nnz,indices,values,idxType,idxBase,valueType)


cdef void* _hipsparseSpVecGetIndexBase__funptr = NULL
cdef hipsparseStatus_t hipsparseSpVecGetIndexBase(void *const spVecDescr,hipsparseIndexBase_t * idxBase) nogil:
    global _hipsparseSpVecGetIndexBase__funptr
    __init_symbol(&_hipsparseSpVecGetIndexBase__funptr,"hipsparseSpVecGetIndexBase")
    return (<hipsparseStatus_t (*)(void *const,hipsparseIndexBase_t *) nogil> _hipsparseSpVecGetIndexBase__funptr)(spVecDescr,idxBase)


cdef void* _hipsparseSpVecGetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseSpVecGetValues(void *const spVecDescr,void ** values) nogil:
    global _hipsparseSpVecGetValues__funptr
    __init_symbol(&_hipsparseSpVecGetValues__funptr,"hipsparseSpVecGetValues")
    return (<hipsparseStatus_t (*)(void *const,void **) nogil> _hipsparseSpVecGetValues__funptr)(spVecDescr,values)


cdef void* _hipsparseSpVecSetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseSpVecSetValues(void * spVecDescr,void * values) nogil:
    global _hipsparseSpVecSetValues__funptr
    __init_symbol(&_hipsparseSpVecSetValues__funptr,"hipsparseSpVecSetValues")
    return (<hipsparseStatus_t (*)(void *,void *) nogil> _hipsparseSpVecSetValues__funptr)(spVecDescr,values)


cdef void* _hipsparseCreateCoo__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateCoo(void ** spMatDescr,long rows,long cols,long nnz,void * cooRowInd,void * cooColInd,void * cooValues,hipsparseIndexType_t cooIdxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil:
    global _hipsparseCreateCoo__funptr
    __init_symbol(&_hipsparseCreateCoo__funptr,"hipsparseCreateCoo")
    return (<hipsparseStatus_t (*)(void **,long,long,long,void *,void *,void *,hipsparseIndexType_t,hipsparseIndexBase_t,hipDataType) nogil> _hipsparseCreateCoo__funptr)(spMatDescr,rows,cols,nnz,cooRowInd,cooColInd,cooValues,cooIdxType,idxBase,valueType)


cdef void* _hipsparseCreateCooAoS__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateCooAoS(void ** spMatDescr,long rows,long cols,long nnz,void * cooInd,void * cooValues,hipsparseIndexType_t cooIdxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil:
    global _hipsparseCreateCooAoS__funptr
    __init_symbol(&_hipsparseCreateCooAoS__funptr,"hipsparseCreateCooAoS")
    return (<hipsparseStatus_t (*)(void **,long,long,long,void *,void *,hipsparseIndexType_t,hipsparseIndexBase_t,hipDataType) nogil> _hipsparseCreateCooAoS__funptr)(spMatDescr,rows,cols,nnz,cooInd,cooValues,cooIdxType,idxBase,valueType)


cdef void* _hipsparseCreateCsr__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateCsr(void ** spMatDescr,long rows,long cols,long nnz,void * csrRowOffsets,void * csrColInd,void * csrValues,hipsparseIndexType_t csrRowOffsetsType,hipsparseIndexType_t csrColIndType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil:
    global _hipsparseCreateCsr__funptr
    __init_symbol(&_hipsparseCreateCsr__funptr,"hipsparseCreateCsr")
    return (<hipsparseStatus_t (*)(void **,long,long,long,void *,void *,void *,hipsparseIndexType_t,hipsparseIndexType_t,hipsparseIndexBase_t,hipDataType) nogil> _hipsparseCreateCsr__funptr)(spMatDescr,rows,cols,nnz,csrRowOffsets,csrColInd,csrValues,csrRowOffsetsType,csrColIndType,idxBase,valueType)


cdef void* _hipsparseCreateCsc__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateCsc(void ** spMatDescr,long rows,long cols,long nnz,void * cscColOffsets,void * cscRowInd,void * cscValues,hipsparseIndexType_t cscColOffsetsType,hipsparseIndexType_t cscRowIndType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil:
    global _hipsparseCreateCsc__funptr
    __init_symbol(&_hipsparseCreateCsc__funptr,"hipsparseCreateCsc")
    return (<hipsparseStatus_t (*)(void **,long,long,long,void *,void *,void *,hipsparseIndexType_t,hipsparseIndexType_t,hipsparseIndexBase_t,hipDataType) nogil> _hipsparseCreateCsc__funptr)(spMatDescr,rows,cols,nnz,cscColOffsets,cscRowInd,cscValues,cscColOffsetsType,cscRowIndType,idxBase,valueType)


cdef void* _hipsparseCreateBlockedEll__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateBlockedEll(void ** spMatDescr,long rows,long cols,long ellBlockSize,long ellCols,void * ellColInd,void * ellValue,hipsparseIndexType_t ellIdxType,hipsparseIndexBase_t idxBase,hipDataType valueType) nogil:
    global _hipsparseCreateBlockedEll__funptr
    __init_symbol(&_hipsparseCreateBlockedEll__funptr,"hipsparseCreateBlockedEll")
    return (<hipsparseStatus_t (*)(void **,long,long,long,long,void *,void *,hipsparseIndexType_t,hipsparseIndexBase_t,hipDataType) nogil> _hipsparseCreateBlockedEll__funptr)(spMatDescr,rows,cols,ellBlockSize,ellCols,ellColInd,ellValue,ellIdxType,idxBase,valueType)


cdef void* _hipsparseDestroySpMat__funptr = NULL
cdef hipsparseStatus_t hipsparseDestroySpMat(void * spMatDescr) nogil:
    global _hipsparseDestroySpMat__funptr
    __init_symbol(&_hipsparseDestroySpMat__funptr,"hipsparseDestroySpMat")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroySpMat__funptr)(spMatDescr)


cdef void* _hipsparseCooGet__funptr = NULL
cdef hipsparseStatus_t hipsparseCooGet(void *const spMatDescr,long * rows,long * cols,long * nnz,void ** cooRowInd,void ** cooColInd,void ** cooValues,hipsparseIndexType_t * idxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil:
    global _hipsparseCooGet__funptr
    __init_symbol(&_hipsparseCooGet__funptr,"hipsparseCooGet")
    return (<hipsparseStatus_t (*)(void *const,long *,long *,long *,void **,void **,void **,hipsparseIndexType_t *,hipsparseIndexBase_t *,hipDataType *) nogil> _hipsparseCooGet__funptr)(spMatDescr,rows,cols,nnz,cooRowInd,cooColInd,cooValues,idxType,idxBase,valueType)


cdef void* _hipsparseCooAoSGet__funptr = NULL
cdef hipsparseStatus_t hipsparseCooAoSGet(void *const spMatDescr,long * rows,long * cols,long * nnz,void ** cooInd,void ** cooValues,hipsparseIndexType_t * idxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil:
    global _hipsparseCooAoSGet__funptr
    __init_symbol(&_hipsparseCooAoSGet__funptr,"hipsparseCooAoSGet")
    return (<hipsparseStatus_t (*)(void *const,long *,long *,long *,void **,void **,hipsparseIndexType_t *,hipsparseIndexBase_t *,hipDataType *) nogil> _hipsparseCooAoSGet__funptr)(spMatDescr,rows,cols,nnz,cooInd,cooValues,idxType,idxBase,valueType)


cdef void* _hipsparseCsrGet__funptr = NULL
cdef hipsparseStatus_t hipsparseCsrGet(void *const spMatDescr,long * rows,long * cols,long * nnz,void ** csrRowOffsets,void ** csrColInd,void ** csrValues,hipsparseIndexType_t * csrRowOffsetsType,hipsparseIndexType_t * csrColIndType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil:
    global _hipsparseCsrGet__funptr
    __init_symbol(&_hipsparseCsrGet__funptr,"hipsparseCsrGet")
    return (<hipsparseStatus_t (*)(void *const,long *,long *,long *,void **,void **,void **,hipsparseIndexType_t *,hipsparseIndexType_t *,hipsparseIndexBase_t *,hipDataType *) nogil> _hipsparseCsrGet__funptr)(spMatDescr,rows,cols,nnz,csrRowOffsets,csrColInd,csrValues,csrRowOffsetsType,csrColIndType,idxBase,valueType)


cdef void* _hipsparseBlockedEllGet__funptr = NULL
cdef hipsparseStatus_t hipsparseBlockedEllGet(void *const spMatDescr,long * rows,long * cols,long * ellBlockSize,long * ellCols,void ** ellColInd,void ** ellValue,hipsparseIndexType_t * ellIdxType,hipsparseIndexBase_t * idxBase,hipDataType * valueType) nogil:
    global _hipsparseBlockedEllGet__funptr
    __init_symbol(&_hipsparseBlockedEllGet__funptr,"hipsparseBlockedEllGet")
    return (<hipsparseStatus_t (*)(void *const,long *,long *,long *,long *,void **,void **,hipsparseIndexType_t *,hipsparseIndexBase_t *,hipDataType *) nogil> _hipsparseBlockedEllGet__funptr)(spMatDescr,rows,cols,ellBlockSize,ellCols,ellColInd,ellValue,ellIdxType,idxBase,valueType)


cdef void* _hipsparseCsrSetPointers__funptr = NULL
cdef hipsparseStatus_t hipsparseCsrSetPointers(void * spMatDescr,void * csrRowOffsets,void * csrColInd,void * csrValues) nogil:
    global _hipsparseCsrSetPointers__funptr
    __init_symbol(&_hipsparseCsrSetPointers__funptr,"hipsparseCsrSetPointers")
    return (<hipsparseStatus_t (*)(void *,void *,void *,void *) nogil> _hipsparseCsrSetPointers__funptr)(spMatDescr,csrRowOffsets,csrColInd,csrValues)


cdef void* _hipsparseCscSetPointers__funptr = NULL
cdef hipsparseStatus_t hipsparseCscSetPointers(void * spMatDescr,void * cscColOffsets,void * cscRowInd,void * cscValues) nogil:
    global _hipsparseCscSetPointers__funptr
    __init_symbol(&_hipsparseCscSetPointers__funptr,"hipsparseCscSetPointers")
    return (<hipsparseStatus_t (*)(void *,void *,void *,void *) nogil> _hipsparseCscSetPointers__funptr)(spMatDescr,cscColOffsets,cscRowInd,cscValues)


cdef void* _hipsparseCooSetPointers__funptr = NULL
cdef hipsparseStatus_t hipsparseCooSetPointers(void * spMatDescr,void * cooRowInd,void * cooColInd,void * cooValues) nogil:
    global _hipsparseCooSetPointers__funptr
    __init_symbol(&_hipsparseCooSetPointers__funptr,"hipsparseCooSetPointers")
    return (<hipsparseStatus_t (*)(void *,void *,void *,void *) nogil> _hipsparseCooSetPointers__funptr)(spMatDescr,cooRowInd,cooColInd,cooValues)


cdef void* _hipsparseSpMatGetSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatGetSize(void * spMatDescr,long * rows,long * cols,long * nnz) nogil:
    global _hipsparseSpMatGetSize__funptr
    __init_symbol(&_hipsparseSpMatGetSize__funptr,"hipsparseSpMatGetSize")
    return (<hipsparseStatus_t (*)(void *,long *,long *,long *) nogil> _hipsparseSpMatGetSize__funptr)(spMatDescr,rows,cols,nnz)


cdef void* _hipsparseSpMatGetFormat__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatGetFormat(void *const spMatDescr,hipsparseFormat_t * format) nogil:
    global _hipsparseSpMatGetFormat__funptr
    __init_symbol(&_hipsparseSpMatGetFormat__funptr,"hipsparseSpMatGetFormat")
    return (<hipsparseStatus_t (*)(void *const,hipsparseFormat_t *) nogil> _hipsparseSpMatGetFormat__funptr)(spMatDescr,format)


cdef void* _hipsparseSpMatGetIndexBase__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatGetIndexBase(void *const spMatDescr,hipsparseIndexBase_t * idxBase) nogil:
    global _hipsparseSpMatGetIndexBase__funptr
    __init_symbol(&_hipsparseSpMatGetIndexBase__funptr,"hipsparseSpMatGetIndexBase")
    return (<hipsparseStatus_t (*)(void *const,hipsparseIndexBase_t *) nogil> _hipsparseSpMatGetIndexBase__funptr)(spMatDescr,idxBase)


cdef void* _hipsparseSpMatGetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatGetValues(void * spMatDescr,void ** values) nogil:
    global _hipsparseSpMatGetValues__funptr
    __init_symbol(&_hipsparseSpMatGetValues__funptr,"hipsparseSpMatGetValues")
    return (<hipsparseStatus_t (*)(void *,void **) nogil> _hipsparseSpMatGetValues__funptr)(spMatDescr,values)


cdef void* _hipsparseSpMatSetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatSetValues(void * spMatDescr,void * values) nogil:
    global _hipsparseSpMatSetValues__funptr
    __init_symbol(&_hipsparseSpMatSetValues__funptr,"hipsparseSpMatSetValues")
    return (<hipsparseStatus_t (*)(void *,void *) nogil> _hipsparseSpMatSetValues__funptr)(spMatDescr,values)


cdef void* _hipsparseSpMatGetStridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatGetStridedBatch(void * spMatDescr,int * batchCount) nogil:
    global _hipsparseSpMatGetStridedBatch__funptr
    __init_symbol(&_hipsparseSpMatGetStridedBatch__funptr,"hipsparseSpMatGetStridedBatch")
    return (<hipsparseStatus_t (*)(void *,int *) nogil> _hipsparseSpMatGetStridedBatch__funptr)(spMatDescr,batchCount)


cdef void* _hipsparseSpMatSetStridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatSetStridedBatch(void * spMatDescr,int batchCount) nogil:
    global _hipsparseSpMatSetStridedBatch__funptr
    __init_symbol(&_hipsparseSpMatSetStridedBatch__funptr,"hipsparseSpMatSetStridedBatch")
    return (<hipsparseStatus_t (*)(void *,int) nogil> _hipsparseSpMatSetStridedBatch__funptr)(spMatDescr,batchCount)


cdef void* _hipsparseCooSetStridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseCooSetStridedBatch(void * spMatDescr,int batchCount,long batchStride) nogil:
    global _hipsparseCooSetStridedBatch__funptr
    __init_symbol(&_hipsparseCooSetStridedBatch__funptr,"hipsparseCooSetStridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,long) nogil> _hipsparseCooSetStridedBatch__funptr)(spMatDescr,batchCount,batchStride)


cdef void* _hipsparseCsrSetStridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseCsrSetStridedBatch(void * spMatDescr,int batchCount,long offsetsBatchStride,long columnsValuesBatchStride) nogil:
    global _hipsparseCsrSetStridedBatch__funptr
    __init_symbol(&_hipsparseCsrSetStridedBatch__funptr,"hipsparseCsrSetStridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,long,long) nogil> _hipsparseCsrSetStridedBatch__funptr)(spMatDescr,batchCount,offsetsBatchStride,columnsValuesBatchStride)


cdef void* _hipsparseSpMatGetAttribute__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatGetAttribute(void * spMatDescr,hipsparseSpMatAttribute_t attribute,void * data,unsigned long dataSize) nogil:
    global _hipsparseSpMatGetAttribute__funptr
    __init_symbol(&_hipsparseSpMatGetAttribute__funptr,"hipsparseSpMatGetAttribute")
    return (<hipsparseStatus_t (*)(void *,hipsparseSpMatAttribute_t,void *,unsigned long) nogil> _hipsparseSpMatGetAttribute__funptr)(spMatDescr,attribute,data,dataSize)


cdef void* _hipsparseSpMatSetAttribute__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMatSetAttribute(void * spMatDescr,hipsparseSpMatAttribute_t attribute,const void * data,unsigned long dataSize) nogil:
    global _hipsparseSpMatSetAttribute__funptr
    __init_symbol(&_hipsparseSpMatSetAttribute__funptr,"hipsparseSpMatSetAttribute")
    return (<hipsparseStatus_t (*)(void *,hipsparseSpMatAttribute_t,const void *,unsigned long) nogil> _hipsparseSpMatSetAttribute__funptr)(spMatDescr,attribute,data,dataSize)


cdef void* _hipsparseCreateDnVec__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateDnVec(void ** dnVecDescr,long size,void * values,hipDataType valueType) nogil:
    global _hipsparseCreateDnVec__funptr
    __init_symbol(&_hipsparseCreateDnVec__funptr,"hipsparseCreateDnVec")
    return (<hipsparseStatus_t (*)(void **,long,void *,hipDataType) nogil> _hipsparseCreateDnVec__funptr)(dnVecDescr,size,values,valueType)


cdef void* _hipsparseDestroyDnVec__funptr = NULL
cdef hipsparseStatus_t hipsparseDestroyDnVec(void * dnVecDescr) nogil:
    global _hipsparseDestroyDnVec__funptr
    __init_symbol(&_hipsparseDestroyDnVec__funptr,"hipsparseDestroyDnVec")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroyDnVec__funptr)(dnVecDescr)


cdef void* _hipsparseDnVecGet__funptr = NULL
cdef hipsparseStatus_t hipsparseDnVecGet(void *const dnVecDescr,long * size,void ** values,hipDataType * valueType) nogil:
    global _hipsparseDnVecGet__funptr
    __init_symbol(&_hipsparseDnVecGet__funptr,"hipsparseDnVecGet")
    return (<hipsparseStatus_t (*)(void *const,long *,void **,hipDataType *) nogil> _hipsparseDnVecGet__funptr)(dnVecDescr,size,values,valueType)


cdef void* _hipsparseDnVecGetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseDnVecGetValues(void *const dnVecDescr,void ** values) nogil:
    global _hipsparseDnVecGetValues__funptr
    __init_symbol(&_hipsparseDnVecGetValues__funptr,"hipsparseDnVecGetValues")
    return (<hipsparseStatus_t (*)(void *const,void **) nogil> _hipsparseDnVecGetValues__funptr)(dnVecDescr,values)


cdef void* _hipsparseDnVecSetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseDnVecSetValues(void * dnVecDescr,void * values) nogil:
    global _hipsparseDnVecSetValues__funptr
    __init_symbol(&_hipsparseDnVecSetValues__funptr,"hipsparseDnVecSetValues")
    return (<hipsparseStatus_t (*)(void *,void *) nogil> _hipsparseDnVecSetValues__funptr)(dnVecDescr,values)


cdef void* _hipsparseCreateDnMat__funptr = NULL
cdef hipsparseStatus_t hipsparseCreateDnMat(void ** dnMatDescr,long rows,long cols,long ld,void * values,hipDataType valueType,hipsparseOrder_t order) nogil:
    global _hipsparseCreateDnMat__funptr
    __init_symbol(&_hipsparseCreateDnMat__funptr,"hipsparseCreateDnMat")
    return (<hipsparseStatus_t (*)(void **,long,long,long,void *,hipDataType,hipsparseOrder_t) nogil> _hipsparseCreateDnMat__funptr)(dnMatDescr,rows,cols,ld,values,valueType,order)


cdef void* _hipsparseDestroyDnMat__funptr = NULL
cdef hipsparseStatus_t hipsparseDestroyDnMat(void * dnMatDescr) nogil:
    global _hipsparseDestroyDnMat__funptr
    __init_symbol(&_hipsparseDestroyDnMat__funptr,"hipsparseDestroyDnMat")
    return (<hipsparseStatus_t (*)(void *) nogil> _hipsparseDestroyDnMat__funptr)(dnMatDescr)


cdef void* _hipsparseDnMatGet__funptr = NULL
cdef hipsparseStatus_t hipsparseDnMatGet(void *const dnMatDescr,long * rows,long * cols,long * ld,void ** values,hipDataType * valueType,hipsparseOrder_t * order) nogil:
    global _hipsparseDnMatGet__funptr
    __init_symbol(&_hipsparseDnMatGet__funptr,"hipsparseDnMatGet")
    return (<hipsparseStatus_t (*)(void *const,long *,long *,long *,void **,hipDataType *,hipsparseOrder_t *) nogil> _hipsparseDnMatGet__funptr)(dnMatDescr,rows,cols,ld,values,valueType,order)


cdef void* _hipsparseDnMatGetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseDnMatGetValues(void *const dnMatDescr,void ** values) nogil:
    global _hipsparseDnMatGetValues__funptr
    __init_symbol(&_hipsparseDnMatGetValues__funptr,"hipsparseDnMatGetValues")
    return (<hipsparseStatus_t (*)(void *const,void **) nogil> _hipsparseDnMatGetValues__funptr)(dnMatDescr,values)


cdef void* _hipsparseDnMatSetValues__funptr = NULL
cdef hipsparseStatus_t hipsparseDnMatSetValues(void * dnMatDescr,void * values) nogil:
    global _hipsparseDnMatSetValues__funptr
    __init_symbol(&_hipsparseDnMatSetValues__funptr,"hipsparseDnMatSetValues")
    return (<hipsparseStatus_t (*)(void *,void *) nogil> _hipsparseDnMatSetValues__funptr)(dnMatDescr,values)


cdef void* _hipsparseDnMatGetStridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseDnMatGetStridedBatch(void * dnMatDescr,int * batchCount,long * batchStride) nogil:
    global _hipsparseDnMatGetStridedBatch__funptr
    __init_symbol(&_hipsparseDnMatGetStridedBatch__funptr,"hipsparseDnMatGetStridedBatch")
    return (<hipsparseStatus_t (*)(void *,int *,long *) nogil> _hipsparseDnMatGetStridedBatch__funptr)(dnMatDescr,batchCount,batchStride)


cdef void* _hipsparseDnMatSetStridedBatch__funptr = NULL
cdef hipsparseStatus_t hipsparseDnMatSetStridedBatch(void * dnMatDescr,int batchCount,long batchStride) nogil:
    global _hipsparseDnMatSetStridedBatch__funptr
    __init_symbol(&_hipsparseDnMatSetStridedBatch__funptr,"hipsparseDnMatSetStridedBatch")
    return (<hipsparseStatus_t (*)(void *,int,long) nogil> _hipsparseDnMatSetStridedBatch__funptr)(dnMatDescr,batchCount,batchStride)


cdef void* _hipsparseAxpby__funptr = NULL
cdef hipsparseStatus_t hipsparseAxpby(void * handle,const void * alpha,void * vecX,const void * beta,void * vecY) nogil:
    global _hipsparseAxpby__funptr
    __init_symbol(&_hipsparseAxpby__funptr,"hipsparseAxpby")
    return (<hipsparseStatus_t (*)(void *,const void *,void *,const void *,void *) nogil> _hipsparseAxpby__funptr)(handle,alpha,vecX,beta,vecY)


cdef void* _hipsparseGather__funptr = NULL
cdef hipsparseStatus_t hipsparseGather(void * handle,void * vecY,void * vecX) nogil:
    global _hipsparseGather__funptr
    __init_symbol(&_hipsparseGather__funptr,"hipsparseGather")
    return (<hipsparseStatus_t (*)(void *,void *,void *) nogil> _hipsparseGather__funptr)(handle,vecY,vecX)


cdef void* _hipsparseScatter__funptr = NULL
cdef hipsparseStatus_t hipsparseScatter(void * handle,void * vecX,void * vecY) nogil:
    global _hipsparseScatter__funptr
    __init_symbol(&_hipsparseScatter__funptr,"hipsparseScatter")
    return (<hipsparseStatus_t (*)(void *,void *,void *) nogil> _hipsparseScatter__funptr)(handle,vecX,vecY)


cdef void* _hipsparseRot__funptr = NULL
cdef hipsparseStatus_t hipsparseRot(void * handle,const void * c_coeff,const void * s_coeff,void * vecX,void * vecY) nogil:
    global _hipsparseRot__funptr
    __init_symbol(&_hipsparseRot__funptr,"hipsparseRot")
    return (<hipsparseStatus_t (*)(void *,const void *,const void *,void *,void *) nogil> _hipsparseRot__funptr)(handle,c_coeff,s_coeff,vecX,vecY)


cdef void* _hipsparseSparseToDense_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSparseToDense_bufferSize(void * handle,void * matA,void * matB,hipsparseSparseToDenseAlg_t alg,unsigned long * bufferSize) nogil:
    global _hipsparseSparseToDense_bufferSize__funptr
    __init_symbol(&_hipsparseSparseToDense_bufferSize__funptr,"hipsparseSparseToDense_bufferSize")
    return (<hipsparseStatus_t (*)(void *,void *,void *,hipsparseSparseToDenseAlg_t,unsigned long *) nogil> _hipsparseSparseToDense_bufferSize__funptr)(handle,matA,matB,alg,bufferSize)


cdef void* _hipsparseSparseToDense__funptr = NULL
cdef hipsparseStatus_t hipsparseSparseToDense(void * handle,void * matA,void * matB,hipsparseSparseToDenseAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseSparseToDense__funptr
    __init_symbol(&_hipsparseSparseToDense__funptr,"hipsparseSparseToDense")
    return (<hipsparseStatus_t (*)(void *,void *,void *,hipsparseSparseToDenseAlg_t,void *) nogil> _hipsparseSparseToDense__funptr)(handle,matA,matB,alg,externalBuffer)


cdef void* _hipsparseDenseToSparse_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseDenseToSparse_bufferSize(void * handle,void * matA,void * matB,hipsparseDenseToSparseAlg_t alg,unsigned long * bufferSize) nogil:
    global _hipsparseDenseToSparse_bufferSize__funptr
    __init_symbol(&_hipsparseDenseToSparse_bufferSize__funptr,"hipsparseDenseToSparse_bufferSize")
    return (<hipsparseStatus_t (*)(void *,void *,void *,hipsparseDenseToSparseAlg_t,unsigned long *) nogil> _hipsparseDenseToSparse_bufferSize__funptr)(handle,matA,matB,alg,bufferSize)


cdef void* _hipsparseDenseToSparse_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseDenseToSparse_analysis(void * handle,void * matA,void * matB,hipsparseDenseToSparseAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseDenseToSparse_analysis__funptr
    __init_symbol(&_hipsparseDenseToSparse_analysis__funptr,"hipsparseDenseToSparse_analysis")
    return (<hipsparseStatus_t (*)(void *,void *,void *,hipsparseDenseToSparseAlg_t,void *) nogil> _hipsparseDenseToSparse_analysis__funptr)(handle,matA,matB,alg,externalBuffer)


cdef void* _hipsparseDenseToSparse_convert__funptr = NULL
cdef hipsparseStatus_t hipsparseDenseToSparse_convert(void * handle,void * matA,void * matB,hipsparseDenseToSparseAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseDenseToSparse_convert__funptr
    __init_symbol(&_hipsparseDenseToSparse_convert__funptr,"hipsparseDenseToSparse_convert")
    return (<hipsparseStatus_t (*)(void *,void *,void *,hipsparseDenseToSparseAlg_t,void *) nogil> _hipsparseDenseToSparse_convert__funptr)(handle,matA,matB,alg,externalBuffer)


cdef void* _hipsparseSpVV_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSpVV_bufferSize(void * handle,hipsparseOperation_t opX,void * vecX,void * vecY,void * result,hipDataType computeType,unsigned long * bufferSize) nogil:
    global _hipsparseSpVV_bufferSize__funptr
    __init_symbol(&_hipsparseSpVV_bufferSize__funptr,"hipsparseSpVV_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,void *,void *,void *,hipDataType,unsigned long *) nogil> _hipsparseSpVV_bufferSize__funptr)(handle,opX,vecX,vecY,result,computeType,bufferSize)


cdef void* _hipsparseSpVV__funptr = NULL
cdef hipsparseStatus_t hipsparseSpVV(void * handle,hipsparseOperation_t opX,void * vecX,void * vecY,void * result,hipDataType computeType,void * externalBuffer) nogil:
    global _hipsparseSpVV__funptr
    __init_symbol(&_hipsparseSpVV__funptr,"hipsparseSpVV")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,void *,void *,void *,hipDataType,void *) nogil> _hipsparseSpVV__funptr)(handle,opX,vecX,vecY,result,computeType,externalBuffer)


cdef void* _hipsparseSpMV_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMV_bufferSize(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const vecX,const void * beta,void *const vecY,hipDataType computeType,hipsparseSpMVAlg_t alg,unsigned long * bufferSize) nogil:
    global _hipsparseSpMV_bufferSize__funptr
    __init_symbol(&_hipsparseSpMV_bufferSize__funptr,"hipsparseSpMV_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *const,hipDataType,hipsparseSpMVAlg_t,unsigned long *) nogil> _hipsparseSpMV_bufferSize__funptr)(handle,opA,alpha,matA,vecX,beta,vecY,computeType,alg,bufferSize)


cdef void* _hipsparseSpMV_preprocess__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMV_preprocess(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const vecX,const void * beta,void *const vecY,hipDataType computeType,hipsparseSpMVAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseSpMV_preprocess__funptr
    __init_symbol(&_hipsparseSpMV_preprocess__funptr,"hipsparseSpMV_preprocess")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *const,hipDataType,hipsparseSpMVAlg_t,void *) nogil> _hipsparseSpMV_preprocess__funptr)(handle,opA,alpha,matA,vecX,beta,vecY,computeType,alg,externalBuffer)


cdef void* _hipsparseSpMV__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMV(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const vecX,const void * beta,void *const vecY,hipDataType computeType,hipsparseSpMVAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseSpMV__funptr
    __init_symbol(&_hipsparseSpMV__funptr,"hipsparseSpMV")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *const,hipDataType,hipsparseSpMVAlg_t,void *) nogil> _hipsparseSpMV__funptr)(handle,opA,alpha,matA,vecX,beta,vecY,computeType,alg,externalBuffer)


cdef void* _hipsparseSpMM_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMM_bufferSize(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,const void * beta,void *const matC,hipDataType computeType,hipsparseSpMMAlg_t alg,unsigned long * bufferSize) nogil:
    global _hipsparseSpMM_bufferSize__funptr
    __init_symbol(&_hipsparseSpMM_bufferSize__funptr,"hipsparseSpMM_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *const,hipDataType,hipsparseSpMMAlg_t,unsigned long *) nogil> _hipsparseSpMM_bufferSize__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,bufferSize)


cdef void* _hipsparseSpMM_preprocess__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMM_preprocess(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,const void * beta,void *const matC,hipDataType computeType,hipsparseSpMMAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseSpMM_preprocess__funptr
    __init_symbol(&_hipsparseSpMM_preprocess__funptr,"hipsparseSpMM_preprocess")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *const,hipDataType,hipsparseSpMMAlg_t,void *) nogil> _hipsparseSpMM_preprocess__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,externalBuffer)


cdef void* _hipsparseSpMM__funptr = NULL
cdef hipsparseStatus_t hipsparseSpMM(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,const void * beta,void *const matC,hipDataType computeType,hipsparseSpMMAlg_t alg,void * externalBuffer) nogil:
    global _hipsparseSpMM__funptr
    __init_symbol(&_hipsparseSpMM__funptr,"hipsparseSpMM")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *const,hipDataType,hipsparseSpMMAlg_t,void *) nogil> _hipsparseSpMM__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,externalBuffer)


cdef void* _hipsparseSpGEMM_createDescr__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr) nogil:
    global _hipsparseSpGEMM_createDescr__funptr
    __init_symbol(&_hipsparseSpGEMM_createDescr__funptr,"hipsparseSpGEMM_createDescr")
    return (<hipsparseStatus_t (*)(hipsparseSpGEMMDescr_t*) nogil> _hipsparseSpGEMM_createDescr__funptr)(descr)


cdef void* _hipsparseSpGEMM_destroyDescr__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr) nogil:
    global _hipsparseSpGEMM_destroyDescr__funptr
    __init_symbol(&_hipsparseSpGEMM_destroyDescr__funptr,"hipsparseSpGEMM_destroyDescr")
    return (<hipsparseStatus_t (*)(hipsparseSpGEMMDescr_t) nogil> _hipsparseSpGEMM_destroyDescr__funptr)(descr)


cdef void* _hipsparseSpGEMM_workEstimation__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMM_workEstimation(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize1,void * externalBuffer1) nogil:
    global _hipsparseSpGEMM_workEstimation__funptr
    __init_symbol(&_hipsparseSpGEMM_workEstimation__funptr,"hipsparseSpGEMM_workEstimation")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *,void *,const void *,void *,hipDataType,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t,unsigned long *,void *) nogil> _hipsparseSpGEMM_workEstimation__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,spgemmDescr,bufferSize1,externalBuffer1)


cdef void* _hipsparseSpGEMM_compute__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMM_compute(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize2,void * externalBuffer2) nogil:
    global _hipsparseSpGEMM_compute__funptr
    __init_symbol(&_hipsparseSpGEMM_compute__funptr,"hipsparseSpGEMM_compute")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *,void *,const void *,void *,hipDataType,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t,unsigned long *,void *) nogil> _hipsparseSpGEMM_compute__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,spgemmDescr,bufferSize2,externalBuffer2)


cdef void* _hipsparseSpGEMM_copy__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMM_copy(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr) nogil:
    global _hipsparseSpGEMM_copy__funptr
    __init_symbol(&_hipsparseSpGEMM_copy__funptr,"hipsparseSpGEMM_copy")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *,void *,const void *,void *,hipDataType,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t) nogil> _hipsparseSpGEMM_copy__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,spgemmDescr)


cdef void* _hipsparseSpGEMMreuse_workEstimation__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,void * matA,void * matB,void * matC,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize1,void * externalBuffer1) nogil:
    global _hipsparseSpGEMMreuse_workEstimation__funptr
    __init_symbol(&_hipsparseSpGEMMreuse_workEstimation__funptr,"hipsparseSpGEMMreuse_workEstimation")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,void *,void *,void *,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t,unsigned long *,void *) nogil> _hipsparseSpGEMMreuse_workEstimation__funptr)(handle,opA,opB,matA,matB,matC,alg,spgemmDescr,bufferSize1,externalBuffer1)


cdef void* _hipsparseSpGEMMreuse_nnz__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMMreuse_nnz(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,void * matA,void * matB,void * matC,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize2,void * externalBuffer2,unsigned long * bufferSize3,void * externalBuffer3,unsigned long * bufferSize4,void * externalBuffer4) nogil:
    global _hipsparseSpGEMMreuse_nnz__funptr
    __init_symbol(&_hipsparseSpGEMMreuse_nnz__funptr,"hipsparseSpGEMMreuse_nnz")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,void *,void *,void *,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t,unsigned long *,void *,unsigned long *,void *,unsigned long *,void *) nogil> _hipsparseSpGEMMreuse_nnz__funptr)(handle,opA,opB,matA,matB,matC,alg,spgemmDescr,bufferSize2,externalBuffer2,bufferSize3,externalBuffer3,bufferSize4,externalBuffer4)


cdef void* _hipsparseSpGEMMreuse_compute__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMMreuse_compute(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void * matA,void * matB,const void * beta,void * matC,hipDataType computeType,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr) nogil:
    global _hipsparseSpGEMMreuse_compute__funptr
    __init_symbol(&_hipsparseSpGEMMreuse_compute__funptr,"hipsparseSpGEMMreuse_compute")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *,void *,const void *,void *,hipDataType,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t) nogil> _hipsparseSpGEMMreuse_compute__funptr)(handle,opA,opB,alpha,matA,matB,beta,matC,computeType,alg,spgemmDescr)


cdef void* _hipsparseSpGEMMreuse_copy__funptr = NULL
cdef hipsparseStatus_t hipsparseSpGEMMreuse_copy(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,void * matA,void * matB,void * matC,hipsparseSpGEMMAlg_t alg,hipsparseSpGEMMDescr_t spgemmDescr,unsigned long * bufferSize5,void * externalBuffer5) nogil:
    global _hipsparseSpGEMMreuse_copy__funptr
    __init_symbol(&_hipsparseSpGEMMreuse_copy__funptr,"hipsparseSpGEMMreuse_copy")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,void *,void *,void *,hipsparseSpGEMMAlg_t,hipsparseSpGEMMDescr_t,unsigned long *,void *) nogil> _hipsparseSpGEMMreuse_copy__funptr)(handle,opA,opB,matA,matB,matC,alg,spgemmDescr,bufferSize5,externalBuffer5)


cdef void* _hipsparseSDDMM__funptr = NULL
cdef hipsparseStatus_t hipsparseSDDMM(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const A,void *const B,const void * beta,void * C,hipDataType computeType,hipsparseSDDMMAlg_t alg,void * tempBuffer) nogil:
    global _hipsparseSDDMM__funptr
    __init_symbol(&_hipsparseSDDMM__funptr,"hipsparseSDDMM")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *,hipDataType,hipsparseSDDMMAlg_t,void *) nogil> _hipsparseSDDMM__funptr)(handle,opA,opB,alpha,A,B,beta,C,computeType,alg,tempBuffer)


cdef void* _hipsparseSDDMM_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSDDMM_bufferSize(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const A,void *const B,const void * beta,void * C,hipDataType computeType,hipsparseSDDMMAlg_t alg,unsigned long * bufferSize) nogil:
    global _hipsparseSDDMM_bufferSize__funptr
    __init_symbol(&_hipsparseSDDMM_bufferSize__funptr,"hipsparseSDDMM_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *,hipDataType,hipsparseSDDMMAlg_t,unsigned long *) nogil> _hipsparseSDDMM_bufferSize__funptr)(handle,opA,opB,alpha,A,B,beta,C,computeType,alg,bufferSize)


cdef void* _hipsparseSDDMM_preprocess__funptr = NULL
cdef hipsparseStatus_t hipsparseSDDMM_preprocess(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const A,void *const B,const void * beta,void * C,hipDataType computeType,hipsparseSDDMMAlg_t alg,void * tempBuffer) nogil:
    global _hipsparseSDDMM_preprocess__funptr
    __init_symbol(&_hipsparseSDDMM_preprocess__funptr,"hipsparseSDDMM_preprocess")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,const void *,void *,hipDataType,hipsparseSDDMMAlg_t,void *) nogil> _hipsparseSDDMM_preprocess__funptr)(handle,opA,opB,alpha,A,B,beta,C,computeType,alg,tempBuffer)


cdef void* _hipsparseSpSV_createDescr__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr) nogil:
    global _hipsparseSpSV_createDescr__funptr
    __init_symbol(&_hipsparseSpSV_createDescr__funptr,"hipsparseSpSV_createDescr")
    return (<hipsparseStatus_t (*)(hipsparseSpSVDescr_t*) nogil> _hipsparseSpSV_createDescr__funptr)(descr)


cdef void* _hipsparseSpSV_destroyDescr__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr) nogil:
    global _hipsparseSpSV_destroyDescr__funptr
    __init_symbol(&_hipsparseSpSV_destroyDescr__funptr,"hipsparseSpSV_destroyDescr")
    return (<hipsparseStatus_t (*)(hipsparseSpSVDescr_t) nogil> _hipsparseSpSV_destroyDescr__funptr)(descr)


cdef void* _hipsparseSpSV_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSV_bufferSize(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const x,void *const y,hipDataType computeType,hipsparseSpSVAlg_t alg,hipsparseSpSVDescr_t spsvDescr,unsigned long * bufferSize) nogil:
    global _hipsparseSpSV_bufferSize__funptr
    __init_symbol(&_hipsparseSpSV_bufferSize__funptr,"hipsparseSpSV_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const void *,void *const,void *const,void *const,hipDataType,hipsparseSpSVAlg_t,hipsparseSpSVDescr_t,unsigned long *) nogil> _hipsparseSpSV_bufferSize__funptr)(handle,opA,alpha,matA,x,y,computeType,alg,spsvDescr,bufferSize)


cdef void* _hipsparseSpSV_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSV_analysis(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const x,void *const y,hipDataType computeType,hipsparseSpSVAlg_t alg,hipsparseSpSVDescr_t spsvDescr,void * externalBuffer) nogil:
    global _hipsparseSpSV_analysis__funptr
    __init_symbol(&_hipsparseSpSV_analysis__funptr,"hipsparseSpSV_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const void *,void *const,void *const,void *const,hipDataType,hipsparseSpSVAlg_t,hipsparseSpSVDescr_t,void *) nogil> _hipsparseSpSV_analysis__funptr)(handle,opA,alpha,matA,x,y,computeType,alg,spsvDescr,externalBuffer)


cdef void* _hipsparseSpSV_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSV_solve(void * handle,hipsparseOperation_t opA,const void * alpha,void *const matA,void *const x,void *const y,hipDataType computeType,hipsparseSpSVAlg_t alg,hipsparseSpSVDescr_t spsvDescr,void * externalBuffer) nogil:
    global _hipsparseSpSV_solve__funptr
    __init_symbol(&_hipsparseSpSV_solve__funptr,"hipsparseSpSV_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,const void *,void *const,void *const,void *const,hipDataType,hipsparseSpSVAlg_t,hipsparseSpSVDescr_t,void *) nogil> _hipsparseSpSV_solve__funptr)(handle,opA,alpha,matA,x,y,computeType,alg,spsvDescr,externalBuffer)


cdef void* _hipsparseSpSM_createDescr__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr) nogil:
    global _hipsparseSpSM_createDescr__funptr
    __init_symbol(&_hipsparseSpSM_createDescr__funptr,"hipsparseSpSM_createDescr")
    return (<hipsparseStatus_t (*)(hipsparseSpSMDescr_t*) nogil> _hipsparseSpSM_createDescr__funptr)(descr)


cdef void* _hipsparseSpSM_destroyDescr__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr) nogil:
    global _hipsparseSpSM_destroyDescr__funptr
    __init_symbol(&_hipsparseSpSM_destroyDescr__funptr,"hipsparseSpSM_destroyDescr")
    return (<hipsparseStatus_t (*)(hipsparseSpSMDescr_t) nogil> _hipsparseSpSM_destroyDescr__funptr)(descr)


cdef void* _hipsparseSpSM_bufferSize__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSM_bufferSize(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,void *const matC,hipDataType computeType,hipsparseSpSMAlg_t alg,hipsparseSpSMDescr_t spsmDescr,unsigned long * bufferSize) nogil:
    global _hipsparseSpSM_bufferSize__funptr
    __init_symbol(&_hipsparseSpSM_bufferSize__funptr,"hipsparseSpSM_bufferSize")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,void *const,hipDataType,hipsparseSpSMAlg_t,hipsparseSpSMDescr_t,unsigned long *) nogil> _hipsparseSpSM_bufferSize__funptr)(handle,opA,opB,alpha,matA,matB,matC,computeType,alg,spsmDescr,bufferSize)


cdef void* _hipsparseSpSM_analysis__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSM_analysis(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,void *const matC,hipDataType computeType,hipsparseSpSMAlg_t alg,hipsparseSpSMDescr_t spsmDescr,void * externalBuffer) nogil:
    global _hipsparseSpSM_analysis__funptr
    __init_symbol(&_hipsparseSpSM_analysis__funptr,"hipsparseSpSM_analysis")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,void *const,hipDataType,hipsparseSpSMAlg_t,hipsparseSpSMDescr_t,void *) nogil> _hipsparseSpSM_analysis__funptr)(handle,opA,opB,alpha,matA,matB,matC,computeType,alg,spsmDescr,externalBuffer)


cdef void* _hipsparseSpSM_solve__funptr = NULL
cdef hipsparseStatus_t hipsparseSpSM_solve(void * handle,hipsparseOperation_t opA,hipsparseOperation_t opB,const void * alpha,void *const matA,void *const matB,void *const matC,hipDataType computeType,hipsparseSpSMAlg_t alg,hipsparseSpSMDescr_t spsmDescr,void * externalBuffer) nogil:
    global _hipsparseSpSM_solve__funptr
    __init_symbol(&_hipsparseSpSM_solve__funptr,"hipsparseSpSM_solve")
    return (<hipsparseStatus_t (*)(void *,hipsparseOperation_t,hipsparseOperation_t,const void *,void *const,void *const,void *const,hipDataType,hipsparseSpSMAlg_t,hipsparseSpSMDescr_t,void *) nogil> _hipsparseSpSM_solve__funptr)(handle,opA,opB,alpha,matA,matB,matC,computeType,alg,spsmDescr,externalBuffer)
