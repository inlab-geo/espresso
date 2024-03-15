import numpy as np
cimport numpy as cnp

cdef extern from "knt.h":
    int partial_modified(   int, int, int, const float*, const float*, const float*,
                            float, float, float, float, float, float, float*)

def rf_calc(ps, thik, beta, kapa, p, duration, dt, gauss=5., shft=0., db=0., dh=0.):
    cdef int   c_ps       = ps
    cdef int   c_m        = len(thik)
    cdef float c_p        = p
    cdef float c_duration = duration
    cdef float c_dt       = dt
    cdef float c_gauss    = gauss
    cdef float c_shft     = shft
    cdef float c_db       = db
    cdef float c_dh       = dh

    # c_nft has to be a power of 2
    nft_valid = (int) (c_duration / c_dt)
    cdef int c_nft = 1
    while c_nft < nft_valid: c_nft *= 2

    # make sure the three are float32 ndarray
    thik = np.array(thik, dtype=np.float32)
    beta = np.array(beta, dtype=np.float32)
    kapa = np.array(kapa, dtype=np.float32)
    cdef float[:] c_thik = thik # Note, this does not create a new array or allocate a new memory partition.
    cdef float[:] c_beta = beta # Instead, it declares a Cython memory view (float[:])  and binds it to the
    cdef float[:] c_kapa = kapa # same memory as the NumPy ndarray.
    cdef float* c_thik_ptr = &c_thik[0] # Then, we can access the pointer pointing to the memory.
    cdef float* c_beta_ptr = &c_beta[0]
    cdef float* c_kapa_ptr = &c_kapa[0]

    # buf for outputing the results
    result = np.zeros(c_nft, dtype=np.float32)
    cdef float[:] c_result   = result
    cdef float* c_result_ptr = &c_result[0]

    partial_modified(c_ps, c_nft, c_m, c_thik_ptr, c_beta_ptr, c_kapa_ptr, c_p, c_dt, c_gauss, c_shft, c_db, c_dh, c_result_ptr)
    return result[:nft_valid]
