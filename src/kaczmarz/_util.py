import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


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
        return sp.linalg.norm(A, axis=1)

    return np.linalg.norm(A, axis=1)


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
    row_norms = compute_row_norms(A)
    A = normalize(A, norm="l2")
    b = np.array(b).ravel() / row_norms

    return A, b
