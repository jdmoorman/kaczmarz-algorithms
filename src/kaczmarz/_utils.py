from scipy import sparse


def scale_rows(A, v):
    return sparse.spdiags(v, 0, len(v), len(v)) @ A


def scale_cols(A, v):
    return A @ sparse.spdiags(v, 0, len(v), len(v))


def square(A):
    if sparse.issparse(A):
        return A.power(2)
    return A ** 2
