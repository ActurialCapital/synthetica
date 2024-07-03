import numpy as np

from synthetica.stats import _is_positive_definite, nearest_positive_definite


def test_is_positive_definite():
    positive_definite_matrix = np.array([[2, 1], [1, 2]])
    assert _is_positive_definite(
        positive_definite_matrix) == True, "Expected the matrix to be positive definite"

    non_positive_definite_matrix = np.array([[1, 2], [2, 1]])
    assert _is_positive_definite(
        non_positive_definite_matrix) == False, "Expected the matrix to be non-positive definite"


def test_nearest_positive_definite():
    non_positive_definite_matrix = np.array([[1, 2], [2, 1]])
    result = nearest_positive_definite(non_positive_definite_matrix)
    assert _is_positive_definite(result) == True, \
        "Expected the result to be positive definite"
    assert result.shape == non_positive_definite_matrix.shape, \
        "Expected the result to have the same shape as the input"


def test_nearest_positive_definite_identity():
    identity_matrix = np.eye(3)
    result = nearest_positive_definite(identity_matrix)
    assert np.allclose(result, identity_matrix), \
        "Expected the result to be the same as the input identity matrix"


def test_nearest_positive_definite_almost_positive_definite():
    almost_positive_definite_matrix = np.array([[1, 0.99], [0.99, 1]])
    result = nearest_positive_definite(almost_positive_definite_matrix)
    assert _is_positive_definite(result) == True, \
        "Expected the result to be positive definite"
    assert result.shape == almost_positive_definite_matrix.shape, \
        "Expected the result to have the same shape as the input"


def test_nearest_positive_definite_large_matrix():
    large_matrix = np.random.rand(100, 100)
    large_matrix = (large_matrix + large_matrix.T) / 2  # Make it symmetric
    result = nearest_positive_definite(large_matrix)
    assert _is_positive_definite(result) == True, \
        "Expected the result to be positive definite"
    assert result.shape == large_matrix.shape, \
        "Expected the result to have the same shape as the input"
