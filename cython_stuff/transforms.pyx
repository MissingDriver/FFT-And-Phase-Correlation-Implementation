import numpy as np
cimport numpy as np
import math

DTYPE = np.complex
ctypedef np.complex128_t DTYPE_t
ctypedef np.uint8_t ODTYPE_t


def forward_transform(np.ndarray[DTYPE_t, ndim=2] matrix): #This is the function that should be called for 2dfft
    cdef int M=matrix.shape[0]
    cdef int N=matrix.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] ans = np.zeros([M,N], dtype=DTYPE)

    cdef int U
    cdef int V

    for U in range(0,M): #calculates the 1dfft for each row, and then each column of the result. This creates the 2dfft 
        ans[U, :] =  fft(matrix[U, :]) 
    for V in range(0,N):
        ans[:, V] =  fft(ans[:, V])
    return ans

def fft(np.ndarray[DTYPE_t, ndim=1]  matrix): #This function calculates the 1dfft that is necesary for the 2d. It should not be called outside of forward_transform
    cdef np.ndarray[DTYPE_t, ndim=1] even
    cdef np.ndarray[DTYPE_t, ndim=1] odd
    cdef int powerTwo = matrix.size%2
    cdef int N = matrix.shape[0]

    if powerTwo == 1:
        bad = np.zeros(matrix.shape[0], dtype=DTYPE)
        return bad
    if N > 2:
        even =  fft(matrix[::2])
        odd =  fft(matrix[1::2])
    else:
        even = matrix[::2]
        odd = matrix[1::2]

    cdef np.ndarray[DTYPE_t, ndim=1] ans = np.zeros(N, dtype=DTYPE)
    cdef int k

    for k in range(0, int(N/2)):
        ans[k] = even[k] + odd[k] * np.exp(-2j*np.pi*k/N)
        ans[k+int(N/2)] = even[k] + odd[k] * np.exp(-2j*np.pi*(k+(N/2))/N)

    return ans
   

def ifft(np.ndarray[DTYPE_t, ndim=1]  matrix): #This function calculates the 1d inverse fft that is necesary for the 2d. It should not be called outside of inverse_transform
    cdef np.ndarray[DTYPE_t, ndim=1] even
    cdef np.ndarray[DTYPE_t, ndim=1] odd
    cdef int powerTwo = matrix.size%2
    cdef int N = matrix.shape[0]

    if powerTwo == 1:
        bad = np.zeros(matrix.shape[0], dtype=DTYPE)
        return bad
    if N > 2:
        even =  ifft(matrix[::2])
        odd =  ifft(matrix[1::2])
    else:
        even = matrix[::2]
        odd = matrix[1::2]

    cdef np.ndarray[DTYPE_t, ndim=1] ans = np.zeros(N, dtype=DTYPE)
    cdef int k

    for k in range(0, int(N/2)):
        ans[k] = even[k] + odd[k] * np.exp(2j*np.pi*k/N)
        ans[k+int(N/2)] = even[k] + odd[k] * np.exp(2j*np.pi*(k+(N/2))/N)
    return ans

def inverse_transform(np.ndarray[DTYPE_t, ndim=2]  matrix): #This is the function for the inverse 2d fft
    cdef int M=matrix.shape[0]
    cdef int N=matrix.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] ans = np.zeros([M,N], dtype=DTYPE)

    cdef int U
    cdef int V

    for U in range(0,M):
        ans[U, :] =  ifft(matrix[U, :]) 
    for V in range(0,N):
        ans[:, V] =  ifft(ans[:, V])
    return ((ans/(N*M)) + .5)
