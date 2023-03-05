import numpy as np


def blob_size(width, max_degree_included_in_blob=-1):
    if max_degree_included_in_blob >= 0:
        if width == 0:
            return 2
        if width == 1:
            return max_degree_included_in_blob + 2
        return int(
            1 + int((-1 + width ** (1 + max_degree_included_in_blob)) / (-1 + width))
        )
    else:
        return int(1)

def tensor_shape(degree, width):
    return tuple([width for i in range(degree)])

def layers(blobsz, width):
    return next((k for k in range(-1, blobsz) if blob_size(width, k) >= blobsz), None)


def one(width, depth=0):
    ans = np.zeros(blob_size(width, depth), dtype=np.float64)
    ans[0:2] = np.array([np.float64(width), 1.0])
    return ans
    
def tensor_multiply(lhs, rhs, depth):
    """
    >>> print(tensor_multiply(arange(3,2),arange(3,2),2))
    [ 3.  1.  4.  6.  8. 14. 18. 22. 22. 27. 32. 30. 36. 42.]
    >>> print(tensor_multiply(arange(3,2),ones(3,2),2))
    [ 3.  1.  3.  4.  5.  8.  9. 10. 12. 13. 14. 16. 17. 18.]
    >>> print(tensor_multiply(arange(3,2),one(3,2),2))
    [ 3.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13.]
    >>> 
    """
    # lhs and rhs same width
    if int(rhs[0:1]) != int(lhs[0:1]):
        raise ValueError(
            "different width tensors cannot be combined:", lhs[0], "!=", rhs[0]
        )
    # extract width
    width = int(lhs[0])
    lhs_layers = layers(lhs.size, width)
    rhs_layers = layers(rhs.size, width)
    out_depth = min(depth, lhs_layers + rhs_layers)
    ans = zero(int(lhs[0]), depth)
    for i in range(
        min(out_depth, lhs_layers + rhs_layers) + 1
    ):  ## i is the total degree ## j is the degree of the rhs term
        for j in range(max(i - lhs_layers, 0), min(i, rhs_layers) + 1):
            ## nxt row the tensors must be shaped before multiplicaton and flattened before assignment
            ansb = blob_size(width, i - 1)
            anse = blob_size(width, i)
            lhsb = blob_size(width, (i - j) - 1)
            lhse = blob_size(width, (i - j))
            rhsb = blob_size(width, j - 1)
            rhse = blob_size(width, j)
            ans[ansb:anse] += np.tensordot(
                np.reshape(lhs[lhsb:lhse], tensor_shape(i - j, width)),
                np.reshape(rhs[rhsb:rhse], tensor_shape(j, width)),
                axes=0,
            ).flatten()
    return ans

def tensor_exp(arg, depth):
    """"
    >>> d = 7
    >>> s = stream2sigtensor(brownian(100,2),d)
    >>> t = tensor_log(s,d)
    >>> np.sum(tensor_sub(s, tensor_exp(tensor_log(s,d), d))[blob_size(2):]**2) < 1e-25
    True
    >>> 
    >>> # Computes the truncated exponential of arg
    >>> #     1 + arg + arg^2/2! + ... + arg^n/n! where n = depth
    """
    width = int(arg[0])
    result = np.array(one(width))
    if arg.size > blob_size(width):
        top = np.array(arg[0 : blob_size(width) + 1])
        scalar = top[-1]
        top[-1] = 0.0
        x = np.array(arg)
        x[blob_size(width)] = 0.0
        for i in range(depth, 0, -1):
            xx = rescale(
                arg, 1.0 / np.float64(i), top
            )  # top resets the shape and here is extended to set the scalar coefficient to zero
            result = tensor_multiply(result, xx, depth)
            result[blob_size(width)] += 1.0
        result = np.tensordot(math.exp(scalar), result, axes=0)
        result[: blob_size(width)] = top[: blob_size(width)]
    return result