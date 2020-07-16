import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def compute_row_norms(A):
    """Compute the norm of each row of a matrix.

    Parameters
    ----------
    A : (m, n) spmatrx or array_like

    Returns
    -------
    row_norms : (m,) array
    """
    if sp.issparse(A):
        return spla.norm(A, axis=1)

    return np.linalg.norm(A, axis=1)


def normalize_matrix(A, row_norms):
    """Normalize a matrix to have rows with norm 1.

    Parameters
    ----------
    A : (m, n) spmatrx or array
    row_norms : (m,) array

    Returns
    -------
    A_normalized : (m, n) spmatrx or array
    """

    # Be careful! Do not try ``A / row_norms[:, None]`` with a sparse matrix!
    # You will end up with a np.matrix rather than a sparse matrix.

    normalization_matrix = sp.diags(1 / row_norms)
    return normalization_matrix @ A


def normalize_system(A, b):
    """Scale the system ``A @ x = b`` so that the rows of ``A`` have norm 1.

    Parameters
    ----------
    A : (m, n) spmatrix or array_like
    b : (m,) or (m, 1) array_like

    Returns
    -------
    A_normalized : (m, n) array or spmatrx
        Copy of ``A`` with rows scaled to have norm ``1``.
    b_normalized : (m,) or (m, 1) array
        Copy of ``b`` with entries divided by the row norms of ``A``.
    """
    if not sp.issparse(A):
        A = np.array(A)

    row_norms = compute_row_norms(A)
    A = normalize_matrix(A, row_norms=row_norms)
    b = np.array(b).ravel() / row_norms

    return A, b, row_norms
