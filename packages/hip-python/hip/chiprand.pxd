# AMD_COPYRIGHT
from libc.stdint cimport *
ctypedef bint _Bool # bool is not a reserved keyword in C, _Bool is
from .chip cimport hipStream_t
cdef extern from "hiprand/hiprand.h":

    cdef int HIPRAND_VERSION

    cdef int HIPRAND_DEFAULT_MAX_BLOCK_SIZE

    cdef int HIPRAND_DEFAULT_MIN_WARPS_PER_EU

    ctypedef struct uint4:
        unsigned int x
        unsigned int y
        unsigned int z
        unsigned int w

    cdef struct rocrand_discrete_distribution_st:
        unsigned int size
        unsigned int offset
        unsigned int * alias
        double * probability
        double * cdf

    ctypedef rocrand_discrete_distribution_st * rocrand_discrete_distribution

    cdef struct rocrand_generator_base_type:
        pass

    ctypedef rocrand_generator_base_type * rocrand_generator

    cdef enum rocrand_status:
        ROCRAND_STATUS_SUCCESS
        ROCRAND_STATUS_VERSION_MISMATCH
        ROCRAND_STATUS_NOT_CREATED
        ROCRAND_STATUS_ALLOCATION_FAILED
        ROCRAND_STATUS_TYPE_ERROR
        ROCRAND_STATUS_OUT_OF_RANGE
        ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
        ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED
        ROCRAND_STATUS_LAUNCH_FAILURE
        ROCRAND_STATUS_INTERNAL_ERROR

    cdef enum rocrand_rng_type:
        ROCRAND_RNG_PSEUDO_DEFAULT
        ROCRAND_RNG_PSEUDO_XORWOW
        ROCRAND_RNG_PSEUDO_MRG32K3A
        ROCRAND_RNG_PSEUDO_MTGP32
        ROCRAND_RNG_PSEUDO_PHILOX4_32_10
        ROCRAND_RNG_PSEUDO_MRG31K3P
        ROCRAND_RNG_PSEUDO_LFSR113
        ROCRAND_RNG_QUASI_DEFAULT
        ROCRAND_RNG_QUASI_SOBOL32
        ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
        ROCRAND_RNG_QUASI_SOBOL64
        ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64

    ctypedef rocrand_generator_base_type hiprandGenerator_st

    ctypedef rocrand_discrete_distribution_st hiprandDiscreteDistribution_st

    ctypedef rocrand_generator_base_type * hiprandGenerator_t

    ctypedef rocrand_discrete_distribution_st * hiprandDiscreteDistribution_t

    cdef enum hiprandStatus:
        HIPRAND_STATUS_SUCCESS
        HIPRAND_STATUS_VERSION_MISMATCH
        HIPRAND_STATUS_NOT_INITIALIZED
        HIPRAND_STATUS_ALLOCATION_FAILED
        HIPRAND_STATUS_TYPE_ERROR
        HIPRAND_STATUS_OUT_OF_RANGE
        HIPRAND_STATUS_LENGTH_NOT_MULTIPLE
        HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED
        HIPRAND_STATUS_LAUNCH_FAILURE
        HIPRAND_STATUS_PREEXISTING_FAILURE
        HIPRAND_STATUS_INITIALIZATION_FAILED
        HIPRAND_STATUS_ARCH_MISMATCH
        HIPRAND_STATUS_INTERNAL_ERROR
        HIPRAND_STATUS_NOT_IMPLEMENTED

    ctypedef hiprandStatus hiprandStatus_t

    cdef enum hiprandRngType:
        HIPRAND_RNG_TEST
        HIPRAND_RNG_PSEUDO_DEFAULT
        HIPRAND_RNG_PSEUDO_XORWOW
        HIPRAND_RNG_PSEUDO_MRG32K3A
        HIPRAND_RNG_PSEUDO_MTGP32
        HIPRAND_RNG_PSEUDO_MT19937
        HIPRAND_RNG_PSEUDO_PHILOX4_32_10
        HIPRAND_RNG_QUASI_DEFAULT
        HIPRAND_RNG_QUASI_SOBOL32
        HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
        HIPRAND_RNG_QUASI_SOBOL64
        HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64

    ctypedef hiprandRngType hiprandRngType_t

# 
# \brief Creates a new random number generator.
# 
# Creates a new random number generator of type \p rng_type,
# and returns it in \p generator. That generator will use
# GPU to create random numbers.
# 
# Values for \p rng_type are:
# - HIPRAND_RNG_PSEUDO_DEFAULT
# - HIPRAND_RNG_PSEUDO_XORWOW
# - HIPRAND_RNG_PSEUDO_MRG32K3A
# - HIPRAND_RNG_PSEUDO_MTGP32
# - HIPRAND_RNG_PSEUDO_MT19937
# - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
# - HIPRAND_RNG_QUASI_DEFAULT
# - HIPRAND_RNG_QUASI_SOBOL32
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
# - HIPRAND_RNG_QUASI_SOBOL64
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
# 
# \param generator - Pointer to generator
# \param rng_type - Type of random number generator to create
# 
# \return
# - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
# - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
# - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
#   dynamically linked library version \n
# - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
# - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type \p rng_type is not implemented yet \n
# - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
#
cdef hiprandStatus hiprandCreateGenerator(hiprandGenerator_t* generator,hiprandRngType rng_type) nogil


# 
# \brief Creates a new random number generator on host.
# 
# Creates a new host random number generator of type \p rng_type
# and returns it in \p generator. Created generator will use
# host CPU to generate random numbers.
# 
# Values for \p rng_type are:
# - HIPRAND_RNG_PSEUDO_DEFAULT
# - HIPRAND_RNG_PSEUDO_XORWOW
# - HIPRAND_RNG_PSEUDO_MRG32K3A
# - HIPRAND_RNG_PSEUDO_MTGP32
# - HIPRAND_RNG_PSEUDO_MT19937
# - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
# - HIPRAND_RNG_QUASI_DEFAULT
# - HIPRAND_RNG_QUASI_SOBOL32
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
# - HIPRAND_RNG_QUASI_SOBOL64
# - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
# 
# \param generator - Pointer to generator
# \param rng_type - Type of random number generator to create
# 
# \return
# - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
# - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
#   dynamically linked library version \n
# - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
# - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type \p rng_type is not implemented yet \n
# - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
cdef hiprandStatus hiprandCreateGeneratorHost(hiprandGenerator_t* generator,hiprandRngType rng_type) nogil


# 
# \brief Destroys random number generator.
# 
# Destroys random number generator and frees related memory.
# 
# \param generator - Generator to be destroyed
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully \n
cdef hiprandStatus hiprandDestroyGenerator(hiprandGenerator_t generator) nogil


# 
# \brief Generates uniformly distributed 32-bit unsigned integers.
# 
# Generates \p n uniformly distributed 32-bit unsigned integers and
# saves them to \p output_data.
# 
# Generated numbers are between \p 0 and \p 2^32, including \p 0 and
# excluding \p 2^32.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 32-bit unsigned integers to generate
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerate(hiprandGenerator_t generator,unsigned int * output_data,unsigned long n) nogil


# 
# \brief Generates uniformly distributed 8-bit unsigned integers.
# 
# Generates \p n uniformly distributed 8-bit unsigned integers and
# saves them to \p output_data.
# 
# Generated numbers are between \p 0 and \p 2^8, including \p 0 and
# excluding \p 2^8.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 8-bit unsigned integers to generate
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateChar(hiprandGenerator_t generator,unsigned char * output_data,unsigned long n) nogil


# 
# \brief Generates uniformly distributed 16-bit unsigned integers.
# 
# Generates \p n uniformly distributed 16-bit unsigned integers and
# saves them to \p output_data.
# 
# Generated numbers are between \p 0 and \p 2^16, including \p 0 and
# excluding \p 2^16.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 16-bit unsigned integers to generate
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateShort(hiprandGenerator_t generator,unsigned short * output_data,unsigned long n) nogil


# 
# \brief Generates uniformly distributed floats.
# 
# Generates \p n uniformly distributed 32-bit floating-point values
# and saves them to \p output_data.
# 
# Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
# including \p 1.0f.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateUniform(hiprandGenerator_t generator,float * output_data,unsigned long n) nogil


# 
# \brief Generates uniformly distributed double-precision floating-point values.
# 
# Generates \p n uniformly distributed 64-bit double-precision floating-point
# values and saves them to \p output_data.
# 
# Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
# including \p 1.0.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# 
# Note: When \p generator is of type: \p HIPRAND_RNG_PSEUDO_MRG32K3A,
# \p HIPRAND_RNG_PSEUDO_MTGP32, or \p HIPRAND_RNG_QUASI_SOBOL32,
# then the returned \p double values are generated from only 32 random bits
# each (one <tt>unsigned int</tt> value per one generated \p double).
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateUniformDouble(hiprandGenerator_t generator,double * output_data,unsigned long n) nogil


# 
# \brief Generates uniformly distributed half-precision floating-point values.
# 
# Generates \p n uniformly distributed 16-bit half-precision floating-point
# values and saves them to \p output_data.
# 
# Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
# including \p 1.0.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of halfs to generate
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateUniformHalf(hiprandGenerator_t generator,int * output_data,unsigned long n) nogil


# 
# \brief Generates normally distributed floats.
# 
# Generates \p n normally distributed 32-bit floating-point
# values and saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateNormal(hiprandGenerator_t generator,float * output_data,unsigned long n,float mean,float stddev) nogil


# 
# \brief Generates normally distributed doubles.
# 
# Generates \p n normally distributed 64-bit double-precision floating-point
# numbers and saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of doubles to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateNormalDouble(hiprandGenerator_t generator,double * output_data,unsigned long n,double mean,double stddev) nogil


# 
# \brief Generates normally distributed halfs.
# 
# Generates \p n normally distributed 16-bit half-precision floating-point
# numbers and saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of halfs to generate
# \param mean - Mean value of normal distribution
# \param stddev - Standard deviation value of normal distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateNormalHalf(hiprandGenerator_t generator,int * output_data,unsigned long n,int mean,int stddev) nogil


# 
# \brief Generates log-normally distributed floats.
# 
# Generates \p n log-normally distributed 32-bit floating-point values
# and saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of floats to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateLogNormal(hiprandGenerator_t generator,float * output_data,unsigned long n,float mean,float stddev) nogil


# 
# \brief Generates log-normally distributed doubles.
# 
# Generates \p n log-normally distributed 64-bit double-precision floating-point
# values and saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of doubles to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateLogNormalDouble(hiprandGenerator_t generator,double * output_data,unsigned long n,double mean,double stddev) nogil


# 
# \brief Generates log-normally distributed halfs.
# 
# Generates \p n log-normally distributed 16-bit half-precision floating-point
# values and saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of halfs to generate
# \param mean - Mean value of log normal distribution
# \param stddev - Standard deviation value of log normal distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
# aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGenerateLogNormalHalf(hiprandGenerator_t generator,int * output_data,unsigned long n,int mean,int stddev) nogil


# 
# \brief Generates Poisson-distributed 32-bit unsigned integers.
# 
# Generates \p n Poisson-distributed 32-bit unsigned integers and
# saves them to \p output_data.
# 
# \param generator - Generator to use
# \param output_data - Pointer to memory to store generated numbers
# \param n - Number of 32-bit unsigned integers to generate
# \param lambda - lambda for the Poisson distribution
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
# - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
# - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
# of used quasi-random generator \n
# - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
cdef hiprandStatus hiprandGeneratePoisson(hiprandGenerator_t generator,unsigned int * output_data,unsigned long n,double lambda_) nogil


# 
# \brief Initializes the generator's state on GPU or host.
# 
# Initializes the generator's state on GPU or host.
# 
# If hiprandGenerateSeeds() was not called for a generator, it will be
# automatically called by functions which generates random numbers like
# hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.
# 
# \param generator - Generator to initialize
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
# - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
#   a previous kernel launch \n
# - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
# - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully \n
cdef hiprandStatus hiprandGenerateSeeds(hiprandGenerator_t generator) nogil


# 
# \brief Sets the current stream for kernel launches.
# 
# Sets the current stream for all kernel launches of the generator.
# All functions will use this stream.
# 
# \param generator - Generator to modify
# \param stream - Stream to use or NULL for default stream
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_SUCCESS if stream was set successfully \n
cdef hiprandStatus hiprandSetStream(hiprandGenerator_t generator,hipStream_t stream) nogil


# 
# \brief Sets the seed of a pseudo-random number generator.
# 
# Sets the seed of the pseudo-random number generator.
# 
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# 
# \param generator - Pseudo-random number generator
# \param seed - New seed value
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator \n
# - HIPRAND_STATUS_SUCCESS if seed was set successfully \n
cdef hiprandStatus hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator,unsigned long long seed) nogil


# 
# \brief Sets the offset of a random number generator.
# 
# Sets the absolute offset of the random number generator.
# 
# - This operation resets the generator's internal state.
# - This operation does not change the generator's seed.
# 
# Absolute offset cannot be set if generator's type is
# HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.
# 
# \param generator - Random number generator
# \param offset - New absolute offset
# 
# \return
# - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
# - HIPRAND_STATUS_SUCCESS if offset was successfully set \n
# - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
# or HIPRAND_RNG_PSEUDO_MT19937 \n
cdef hiprandStatus hiprandSetGeneratorOffset(hiprandGenerator_t generator,unsigned long long offset) nogil


# 
# \brief Set the number of dimensions of a quasi-random number generator.
# 
# Set the number of dimensions of a quasi-random number generator.
# Supported values of \p dimensions are 1 to 20000.
# 
# - This operation resets the generator's internal state.
# - This operation does not change the generator's offset.
# 
# \param generator - Quasi-random number generator
# \param dimensions - Number of dimensions
# 
# \return
# - HIPRAND_STATUS_NOT_CREATED if the generator wasn't created \n
# - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
# - HIPRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
# - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
cdef hiprandStatus hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator,unsigned int dimensions) nogil


# 
# \brief Returns the version number of the cuRAND or rocRAND library.
# 
# Returns in \p version the version number of the underlying cuRAND or
# rocRAND library.
# 
# \param version - Version of the library
# 
# \return
# - HIPRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
# - HIPRAND_STATUS_SUCCESS if the version number was successfully returned \n
cdef hiprandStatus hiprandGetVersion(int * version) nogil


# 
# \brief Construct the histogram for a Poisson distribution.
# 
# Construct the histogram for the Poisson distribution with lambda \p lambda.
# 
# \param lambda - lambda for the Poisson distribution
# \param discrete_distribution - pointer to the histogram in device memory
# 
# \return
# - HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
# - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
# - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
# - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
cdef hiprandStatus hiprandCreatePoissonDistribution(double lambda_,hiprandDiscreteDistribution_t* discrete_distribution) nogil


# 
# \brief Destroy the histogram array for a discrete distribution.
# 
# Destroy the histogram array for a discrete distribution created by
# hiprandCreatePoissonDistribution.
# 
# \param discrete_distribution - pointer to the histogram in device memory
# 
# \return
# - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
# - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
cdef hiprandStatus hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution) nogil
