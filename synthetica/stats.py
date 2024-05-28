import numpy as np


def _is_positive_definite(symmetric_matrix: np.array) -> bool:
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(symmetric_matrix)
        return True

    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(matrix: np.array) -> np.ndarray:
    """Find the nearest positive-definite matrix to input.

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], 
    which credits [2]:
        * [1] [Matlab](https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd)
        * [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): [Source](https://doi.org/10.1016/0024-3795(88)90223-6)

    Notes
    -----
        Other sources:
        * [Stackoverflow](https://stackoverflow.com/.../python-convert-matrix-to-positive-semi-definite)
        * [Gist](https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd)

    Parameters
    ----------
    matrix : np.array
        The input matrix.

    Returns
    -------
    np.ndarray
        The nearest positive-definite matrix.
    """
    # Computes the symmetric component of a covariance matrix.
    # It does this by adding the covariance matrix with its transpose and
    # dividing the result by 2, which produces a new matrix that is symmetric
    # about its main diagonal. This operation ensures that the resulting matrix
    # can be used in further mathematical operations that require a symmetric
    # matrix.
    symmetric_matrix = (matrix + matrix.T) / 2
    # Performs a Singular Value Decomposition (SVD) on a symmetric matrix.
    # Decomposes a matrix into three matrices: left singular vectors, singular
    # values, and right singular vectors.
    # Returns principal components and their corresponding variances
    _, variance, eigenvectors = np.linalg.svd(symmetric_matrix)
    # Computes the covariance matrix of the data in a new coordinate system
    # defined by the eigenvectors of the original covariance matrix
    transformed_matrix = np.dot(
        eigenvectors.T, np.dot(np.diag(variance), eigenvectors))
    # Symmetric matrix that blends the original symmetric matrix of the data
    # with the covariance matrix of the data in the transformed coordinate
    # system
    weighted_matrix = (symmetric_matrix + transformed_matrix) / 2
    weighted_symetric_matrix = (weighted_matrix + weighted_matrix.T) / 2
    if _is_positive_definite(weighted_symetric_matrix):
        return weighted_symetric_matrix
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(min_eigenvalues)`, since
    # `min_eigenvalues` is usually on the order of 1e-16, and `eps(1e-16)` is
    # on the order of 1e-34, whereas `spacing` will, for Gaussian random matrixes
    # of small dimension, be on other order of 1e-16. In practice, both ways
    # converge, as the unit test below suggests.
    spacing = np.spacing(np.linalg.norm(matrix))
    # Process known as "shrinking" a matrix towards the identity matrix to
    # ensure that it is positive definite
    identity_matrix = np.eye(matrix.shape[0])
    k = 1  # shrinkage parameter
    # Loop until the weighted symmetric matrix is positive definite, which is
    # determined by the is_positive_definite() function. This function likely
    # checks that all eigenvalues of the matrix are positive.
    while not _is_positive_definite(weighted_symetric_matrix):
        # Find the minimum eigenvalue of the matrix using linalg.eigvals().
        # The np.real() function is used to discard any imaginary components
        # that may be returned due to numerical errors.
        min_eigenvalues = np.min(
            np.real(np.linalg.eigvals(weighted_symetric_matrix)))
        # Shrink the matrix towards the identity matrix by adding a multiple of
        # the identity matrix to the matrix. The multiple is calculated using a
        # formula involving the minimum eigenvalue, the shrinkage parameter k,
        # and a constant spacing.
        weighted_symetric_matrix += identity_matrix * \
            (- min_eigenvalues * k**2 + spacing)
        k += 1  # Increment the shrinkage parameter k.
    return weighted_symetric_matrix
