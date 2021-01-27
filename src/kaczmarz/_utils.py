from scipy import sparse


def scale_rows(A, v):
    """Scale each row of A by the corresponding entry of v and return the
    result."""
    return sparse.spdiags(v, 0, len(v), len(v)) @ A


def scale_cols(A, v):
    """Scale each column of A by the corresponding entry of v and return the
    result."""
    return A @ sparse.spdiags(v, 0, len(v), len(v))


def square(A):
    """Return the product of A with itself."""
    if sparse.issparse(A):
        return A.power(2)
    return A ** 2
