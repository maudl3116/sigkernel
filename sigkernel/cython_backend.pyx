# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport exp
import numpy as np

def sigkernel_cython(double[:,:,:] G_static, bint _naive_solver=False):

    cdef int A = G_static.shape[0]
    cdef int M = G_static.shape[1]
    cdef int N = G_static.shape[2]
    cdef int i, j, l

    cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)

    for l in range(A):
        for i in range(M+1):
            K[l,i,0] = 1.

        for j in range(N+1):
            K[l,0,j] = 1.

        for i in range(M):
            for j in range(N):

                if _naive_solver:
                    K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1. + 0.5*G_static[l,i,j]) - K[l,i,j]
                    #K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] + K[l,i,j]*(G_static[l,i,j] - 1.)
                else:
                    K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1.+0.5*G_static[l,i,j]+(1./12)*G_static[l,i,j]**2) - K[l,i,j]*(1. - (1./12)*G_static[l,i,j]**2)
                    #K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] - K[l,i,j] + (exp(0.5*G_static[l,i,j])-1.)*(K[l,i+1,j] + K[l,i,j+1])

    return np.array(K)

def sigkernel_from_abelian_cython(double[:,:,:] G_static, double[:,:,:] X, double[:,:,:] Y, double[:,:,:,:] M_x, double[:,:,:,:] M_y, double[:,:,:,:] Mxvy, double[:,:,:,:] Myvx, double[:,:,:,:,:] adj, bint _naive_solver=False):

    cdef int A = G_static.shape[0]
    cdef int M = G_static.shape[1]
    cdef int N = G_static.shape[2]
    cdef int D = X.shape[2]
    cdef int i, j, l, p, q
    cdef double X_ik, Y_jk

    cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)
    cdef double[:,:,:] eval_adj = np.zeros((A,M+1,N+1), dtype=np.float64)

    for l in range(A):
        for i in range(M+1):
            K[l,i,0] = 1.

        for j in range(N+1):
            K[l,0,j] = 1.

        for i in range(M):
            for j in range(N):
                
                for p in range(D):
                
                    eval_adj[l, i, j] = eval_adj[l, i, j] + adj[l, 0, i, j, p]*Myvx[l, i, j, p] + adj[l, 1, i, j, p]*Mxvy[l, i, j, p]
        
                    adj[l, 0, i+1, j+1, p] =  adj[l, 0, i, j+1, p] + K[l,i,j] * X[l, i, p] 
                    adj[l, 1, i+1, j+1, p] =  adj[l, 1, i+1, j, p] + K[l,i,j] * Y[l, j, p]
                
                    for q in range(D):
                        adj[l, 0, i+1, j+1, p] = adj[l, 0, i+1, j+1, p]  + adj[l, 1, i, j+1, q] * M_x[l, i, q, p]
                        adj[l, 1, i+1, j+1, p] = adj[l, 1, i+1, j+1, p] + adj[l, 0, i+1, j, q] * M_y[l, j, q, p]
                

                K[l,i+1,j+1] = K[l,i+1,j] + K[l,i,j+1] - K[l,i,j] + K[l,i,j]*G_static[l,i,j] + eval_adj[l,i,j]

    return np.array(K)


def sigkernel_derivatives_cython(double[:,:,:] G_static, double[:,:,:] G_static_direction):

    cdef int A = G_static.shape[0]
    cdef int M = G_static.shape[1]
    cdef int N = G_static.shape[2]
    cdef int i, j, l

    cdef double[:,:,:] K = np.zeros((A,M+1,N+1), dtype=np.float64)
    cdef double[:,:,:] K_diff = np.zeros((A,M+1,N+1), dtype=np.float64)
    cdef double[:,:,:] K_diffdiff = np.zeros((A,M+1,N+1), dtype=np.float64)

    for l in range(A):

        for i in range(M+1):
            K[l,i,0] = 1.

        for j in range(N+1):
            K[l,0,j] = 1.

        for i in range(M):
            for j in range(N):
                K[l,i+1,j+1] = (K[l,i+1,j] + K[l,i,j+1])*(1. + .5*G_static[l,i,j]) - K[l,i,j]
                K_diff[l,i+1,j+1] = (K_diff[l,i+1,j] + K_diff[l,i,j+1])*(1. + .5*G_static[l,i,j]) - K_diff[l,i,j] + (K[l,i+1,j] + K[l,i,j+1])*.5*G_static_direction[l,i,j]
                K_diffdiff[l,i+1,j+1] = (K_diffdiff[l,i+1,j] + K_diffdiff[l,i,j+1])*(1. + .5*G_static[l,i,j]) - K_diffdiff[l,i,j] + (K_diff[l,i+1,j] + K_diff[l,i,j+1] + K[l,i+1,j] + K[l,i,j+1])*.5*G_static_direction[l,i,j]

    return np.array(K), np.array(K_diff), np.array(K_diffdiff)


def sigkernel_Gram_cython(double[:,:,:,:] G_static, bint sym=False, bint _naive_solver=False):

    cdef int A = G_static.shape[0]
    cdef int B = G_static.shape[1]
    cdef int M = G_static.shape[2]
    cdef int N = G_static.shape[3]
    cdef int i, j, l, m

    cdef double[:,:,:,:] K = np.zeros((A,B,M+1,N+1), dtype=np.float64)

    if sym:
        # for l in prange(A,nogil=True):
        for l in range(A):
            for m in range(l,A):

                for i in range(M+1):
                    K[l,m,i,0] = 1.
                    K[m,l,i,0] = 1.

                for j in range(N+1):
                    K[l,m,0,j] = 1.
                    K[m,l,0,j] = 1.

                for i in range(M):
                    for j in range(N):

                        if _naive_solver:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]) - K[l,m,i,j]
                            #K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] + K[l,m,i,j]*(G_static[l,m,i,j]-1.)
                        else:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1.+0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1.-(1./12)*G_static[l,m,i,j]**2)
                            #K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])

                        K[m,l,j+1,i+1] = K[l,m,i+1,j+1]

    else:
        # for l in prange(A,nogil=True):
        for l in range(A):
            for m in range(B):

                for i in range(M+1):
                    K[l,m,i,0] = 1.

                for j in range(N+1):
                    K[l,m,0,j] = 1.

                for i in range(M):
                    for j in range(N):

                        if _naive_solver:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]) - K[l,m,i,j]
                        else:
                            K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]+(1./12)*G_static[l,m,i,j]**2) - K[l,m,i,j]*(1. - (1./12)*G_static[l,m,i,j]**2)
                            #K[l,m,i+1,j+1] = K[l,m,i+1,j] + K[l,m,i,j+1] - K[l,m,i,j] + (exp(0.5*G_static[l,m,i,j])-1.)*(K[l,m,i+1,j] + K[l,m,i,j+1])

    return np.array(K)


def sigkernel_derivatives_Gram_cython(double[:,:,:,:] G_static, double[:,:,:,:] G_static_direction, bint sym=False, bint _naive_solver=False):

    cdef int A = G_static.shape[0]
    cdef int B = G_static.shape[1]
    cdef int M = G_static.shape[2]
    cdef int N = G_static.shape[3]
    cdef int i, j, l, m

    cdef double[:,:,:,:] K = np.zeros((A,B,M+1,N+1), dtype=np.float64)
    cdef double[:,:,:,:] K_diff = np.zeros((A,B,M+1,N+1), dtype=np.float64)
    cdef double[:,:,:,:] K_diffdiff = np.zeros((A,B,M+1,N+1), dtype=np.float64)

    if sym:
        for l in range(A):
            for m in range(l,A):

                for i in range(M+1):
                    K[l,m,i,0] = 1.
                    K[m,l,i,0] = 1.

                for j in range(N+1):
                    K[l,m,0,j] = 1.
                    K[m,l,0,j] = 1.

                for i in range(M):
                    for j in range(N):

                        K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]) - K[l,m,i,j]
                        K_diff[l,m,i+1,j+1] = (K_diff[l,m,i+1,j] + K_diff[l,m,i,j+1])*(1. + .5*G_static[l,m,i,j]) - K_diff[l,m,i,j] + (K[l,m,i+1,j] + K[l,m,i,j+1])*.5*G_static_direction[l,m,i,j]
                        K_diffdiff[l,m,i+1,j+1] = (K_diffdiff[l,m,i+1,j] + K_diffdiff[l,m,i,j+1])*(1. + .5*G_static[l,m,i,j]) - K_diffdiff[l,m,i,j] + (K_diff[l,m,i+1,j] + K_diff[l,m,i,j+1] + K[l,m,i+1,j] + K[l,m,i,j+1])*.5*G_static_direction[l,m,i,j]


                        K[m,l,j+1,i+1] = K[l,m,i+1,j+1]
                        K_diff[m,l,j+1,i+1] = K_diff[l,m,i+1,j+1]
                        K_diffdiff[m,l,j+1,i+1] = K_diffdiff[l,m,i+1,j+1]

    else:
        for l in range(A):
            for m in range(B):

                for i in range(M+1):
                    K[l,m,i,0] = 1.

                for j in range(N+1):
                    K[l,m,0,j] = 1.

                for i in range(M):
                    for j in range(N):

                        K[l,m,i+1,j+1] = (K[l,m,i+1,j] + K[l,m,i,j+1])*(1. + 0.5*G_static[l,m,i,j]) - K[l,m,i,j]
                        K_diff[l,m,i+1,j+1] = (K_diff[l,m,i+1,j] + K_diff[l,m,i,j+1])*(1. + .5*G_static[l,m,i,j]) - K_diff[l,m,i,j] + (K[l,m,i+1,j] + K[l,m,i,j+1])*.5*G_static_direction[l,m,i,j]
                        K_diffdiff[l,m,i+1,j+1] = (K_diffdiff[l,m,i+1,j] + K_diffdiff[l,m,i,j+1])*(1. + .5*G_static[l,m,i,j]) - K_diffdiff[l,m,i,j] + (K_diff[l,m,i+1,j] + K_diff[l,m,i,j+1] + K[l,m,i+1,j] + K[l,m,i,j+1])*.5*G_static_direction[l,m,i,j]


    return np.array(K)
