import torch.cuda
from numba import cuda
import math

# ===========================================================================================================
@cuda.jit
def sigkernel_cuda(M_inc, len_x, len_y, n_anti_diagonals, M_sol, _naive_solver=False):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc[block_id, i-1, j-1]

            k_01 = M_sol[block_id, i-1, j]
            k_10 = M_sol[block_id, i, j-1]
            k_00 = M_sol[block_id, i-1, j-1]

            if _naive_solver:
                M_sol[block_id, i, j] = (k_01 + k_10) * (1. + 0.5*inc) - k_00
                #M_sol[block_id, i, j] = k_01 + k_10 + k_00*(inc-1.)
            else:
                M_sol[block_id, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)
                #M_sol[block_id, i, j] = k_01 + k_10 - k_00 + (math.exp(0.5*inc) - 1.)*(k_01 + k_10)

        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================
# ===========================================================================================================
@cuda.jit
def sigkernel_from_abelian_cuda(M_inc_area, X, Y, M_x, M_y, Mxvy, Myvx, eval_adj, dim, len_x, len_y, n_anti_diagonals, adj, M_sol, _naive_solver=False):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol: a 3-tensor storing the solutions of the PDEs.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id
    
    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1
        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc_area[block_id, i-1, j-1]

            k_01 = M_sol[block_id, i-1, j]
            k_10 = M_sol[block_id, i, j-1]
            k_00 = M_sol[block_id, i-1, j-1]    
            
            # update the adjoint states
            for k in range(dim):
                
                cuda.atomic.add(eval_adj, (block_id, i-1, j-1), adj[block_id, 0, i-1, j-1, k]*Myvx[block_id, i-1, j-1, k] +adj[block_id, 1, i-1, j-1, k]*Mxvy[block_id, i-1, j-1, k])
                
                X_ik = X[block_id, i-1, k] 
                Y_jk = Y[block_id, j-1, k]
                adj[block_id, 0, i, j, k] =  adj[block_id, 0, i-1, j, k] + k_00 * X_ik 
                adj[block_id, 1, i, j, k] =  adj[block_id, 1, i, j-1, k] + k_00 * Y_jk 
                
                for l in range(dim):
                    cuda.atomic.add(adj, (block_id, 0, i, j, k), adj[block_id, 1, i-1, j, l] * M_x[block_id, i-1, l, k])
                    cuda.atomic.add(adj, (block_id, 1, i, j, k), adj[block_id, 0, i, j-1, l] * M_y[block_id, j-1, l, k])

            
            # update the main state
            M_sol[block_id, i, j] = k_01 + k_10 + k_00*(inc-1.) + eval_adj[block_id, i-1, j-1]
            
        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================


# ===========================================================================================================
@cuda.jit
def sigkernel_derivatives_cuda(M_inc, M_inc_diff, len_x, len_y, n_anti_diagonals, M_sol, M_sol_diff, M_sol_diffdiff):
    """
    We start from a list of pairs of paths [(x^1,y^1), ..., (x^n, y^n)]
    M_inc: a 3-tensor D[i,j,k] = <x^i_j, y^i_k>.
    M_inc_diff: a 3-tensor D[i,j,k] = <gamma^i_j, y^i_k>.
    n_anti_diagonals = 2 * max(len_x, len_y) - 1
    M_sol, M_sol_diff, M_sol_diffdiff: two 3-tensor storing the solutions of the PDEs for the kernel and its first and second derivative resp.
    """

    # Each block corresponds to a pair (x_i,y_i).
    block_id = cuda.blockIdx.x
    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc[block_id, i-1, j-1]
            inc_diff = M_inc_diff[block_id, i-1, j-1]

            k_01 = M_sol[block_id, i-1, j]
            k_10 = M_sol[block_id, i, j-1]
            k_00 = M_sol[block_id, i-1, j-1]

            k_01_diff = M_sol_diff[block_id, i-1, j]
            k_10_diff = M_sol_diff[block_id, i, j-1]
            k_00_diff = M_sol_diff[block_id, i-1, j-1]

            k_01_diffdiff = M_sol_diffdiff[block_id, i - 1, j]
            k_10_diffdiff = M_sol_diffdiff[block_id, i, j - 1]
            k_00_diffdiff = M_sol_diffdiff[block_id, i - 1, j - 1]

            M_sol[block_id, i, j] = (k_01 + k_10) * (1. + .5*inc) - k_00
            M_sol_diff[block_id, i, j] = (k_01_diff + k_10_diff) * (1. + .5*inc) - k_00_diff + .5*inc_diff*(k_01 + k_10)
            M_sol_diffdiff[block_id, i, j] = (k_01_diffdiff + k_10_diffdiff) * (1. + .5*inc) - k_00_diffdiff + .5*inc_diff*(k_01_diff + k_10_diff + k_01 + k_10)

        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================

# ===========================================================================================================
@cuda.jit
def sigkernel_Gram_cuda(M_inc, len_x, len_y, n_anti_diagonals, M_sol, _naive_solver=False):

    block_id_x = cuda.blockIdx.x
    block_id_y = cuda.blockIdx.y

    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc[block_id_x, block_id_y, i-1, j-1]

            k_01 = M_sol[block_id_x, block_id_y, i-1, j]
            k_10 = M_sol[block_id_x, block_id_y, i, j-1]
            k_00 = M_sol[block_id_x, block_id_y, i-1, j-1]

            # vanilla scheme
            if _naive_solver:
                M_sol[block_id_x, block_id_y, i, j] = (k_01 + k_10) * (1. + 0.5 * inc) - k_00
                #M_sol[block_id_x, block_id_y, i, j] = k_01 + k_10 + k_00*(inc-1.)
            else:
                M_sol[block_id_x, block_id_y, i, j] = (k_01 + k_10)*(1. + 0.5*inc + (1./12)*inc**2) - k_00*(1. - (1./12)*inc**2)
                #M_sol[block_id_x, block_id_y, i, j] = k_01 + k_10 - k_00 + (math.exp(0.5*inc) - 1.)*(k_01 + k_10)

        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================


# ===========================================================================================================
@cuda.jit
def sigkernel_derivatives_Gram_cuda(M_inc, M_inc_diff, len_x, len_y, n_anti_diagonals, M_sol, M_sol_diff, M_sol_diffdiff):

    block_id_x = cuda.blockIdx.x
    block_id_y = cuda.blockIdx.y

    # Each thread works on a node of a diagonal.
    thread_id = cuda.threadIdx.x

    I = thread_id

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_anti_diagonals):

        # The index is actually 'p - thread_id' but need to force it in-bounds
        J = max(0, min(p - thread_id, len_y - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal
        if I + J == p and (I < len_x and J < len_y):

            inc = M_inc[block_id_x, block_id_y, i-1, j-1]
            inc_diff = M_inc_diff[block_id_x, block_id_y, i-1, j-1]

            k_01 = M_sol[block_id_x, block_id_y, i-1, j]
            k_10 = M_sol[block_id_x, block_id_y, i, j-1]
            k_00 = M_sol[block_id_x, block_id_y, i-1, j-1]

            k_01_diff = M_sol_diff[block_id_x, block_id_y, i-1, j]
            k_10_diff = M_sol_diff[block_id_x, block_id_y, i, j-1]
            k_00_diff = M_sol_diff[block_id_x, block_id_y, i-1, j-1]

            k_01_diffdiff = M_sol_diffdiff[block_id_x, block_id_y, i-1, j]
            k_10_diffdiff = M_sol_diffdiff[block_id_x, block_id_y, i, j-1]
            k_00_diffdiff = M_sol_diffdiff[block_id_x, block_id_y, i-1, j-1]

            M_sol[block_id_x, block_id_y, i, j] = (k_01 + k_10) * (1. + 0.5 * inc) - k_00
            M_sol_diff[block_id_x, block_id_y, i, j] = (k_01_diff + k_10_diff) * (1. + .5 * inc) - k_00_diff + .5 * inc_diff * (k_01 + k_10)
            M_sol_diffdiff[block_id_x, block_id_y, i, j] = (k_01_diffdiff + k_10_diffdiff) * (1. + .5 * inc) - k_00_diffdiff + .5 * inc_diff * (k_01_diff + k_10_diff + k_01 + k_10)

        # Wait for other threads in this block
        cuda.syncthreads()
# ===========================================================================================================