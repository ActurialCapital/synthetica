import pytest 
import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal

from synthetica import GeometricBrownianMotion, nearest_positive_definite

Model = GeometricBrownianMotion
index = pd.date_range("2020-01-01", "2020-02-01", inclusive="left")


def test_random_seed():
    arr1 = Model(seed=123).white_noise
    arr2 = Model(seed=123).white_noise
    assert np.array_equal(arr1, arr2)

    arr1 = Model(seed=None).white_noise
    arr2 = Model(seed=None).white_noise
    assert not np.array_equal(arr1, arr2)


def test_repr():
    model = Model()
    assert model.__class__.__name__ == 'GeometricBrownianMotion'


def test_datetime_index():
    # Before transformation
    model = Model(length=index)
    assert_index_equal(model.index, index)

    # After transformation
    simulated = model.transform()
    assert_index_equal(simulated.index, index)


def test_cholesky_transform_positive_definite():
    # Test with a positive definite matrix
    rvs = np.array([[1, 2], [3, 4]])
    matrix = np.array([[2, 1], [1, 2]])
    expected_output = np.linalg.cholesky(matrix) @ rvs.T
    expected_output = expected_output.T

    result = Model().cholesky_transform(rvs, matrix)
    assert np.allclose(result, expected_output), \
        f"Expected {expected_output}, but got {result}"


def test_cholesky_transform_non_positive_definite():
    # Test with a non-positive definite matrix
    rvs = np.array([[1, 2], [3, 4]])
    matrix = np.array([[1, 2], [2, 1]])  # This is not positive definite

    # Calculate the nearest positive definite matrix
    positive_definite_matrix = nearest_positive_definite(matrix)
    expected_output = np.linalg.cholesky(positive_definite_matrix) @ rvs.T
    expected_output = expected_output.T

    result = Model().cholesky_transform(rvs, matrix)
    assert np.allclose(result, expected_output), \
        f"Expected {expected_output}, but got {result}"


def test_create_corr_returns_positive_definite():
    num_paths = 2
    length = 10
    mean = 0
    delta = 0.01
    sigma = 0.2
    seed = 123
    
    model = Model(length, num_paths, mean, delta, sigma, seed=seed)
    matrix = np.array([[2, 1], [1, 2]])  # Positive definite matrix
    res = model.create_corr_returns(matrix)

    assert isinstance(res, pd.DataFrame), \
        "Expected result to be a pandas DataFrame"
    assert res.shape == (length, num_paths), \
        f"Expected shape {(length, num_paths)}, but got {res.shape}"

def test_create_corr_returns_non_positive_definite():
    num_paths = 2
    length = 10
    mean = 0
    delta = 0.01
    sigma = 0.2
    seed = 123
    
    model = Model(length, num_paths, mean, delta, sigma, seed=seed)
    matrix = np.array([[1, -2], [2, 1]])  # Not positive definite matrix
    
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(matrix)
        
    # Calculate the nearest positive definite matrix
    positive_definite_matrix = nearest_positive_definite(matrix)
    res = model.create_corr_returns(positive_definite_matrix)

    assert isinstance(res, pd.DataFrame), \
        "Expected result to be a pandas DataFrame"
    assert res.shape == (length, num_paths), \
        f"Expected shape {(length, num_paths)}, but got {res.shape}"


def test_red_noise():
    pass


def test_white_noise():
    num_paths = 1
    length = 10
    mean = 0
    delta = 0.01
    sigma = 0.2
    seed = 123
    
    model = Model(length, num_paths, mean, delta, sigma, seed=seed)
    res = model.white_noise

    assert isinstance(res, np.ndarray), \
        "Expected result to be a numpy ndarray"
    assert res.shape == (length, num_paths), \
        f"Expected shape {(length, num_paths)}, but got {res.shape}"

    # Validate mean
    actual_mean = res.mean()
    assert np.isclose(actual_mean, mean, atol=0.1), \
        f"Expected mean close to {mean}, but got {actual_mean}"

    # Validate mean setter
    new_mean = 1
    model.mean = new_mean
    actual_mean = model.white_noise.mean()
    assert np.isclose(actual_mean, new_mean, rtol=0.1), \
        f"Expected variance close to {new_mean}, but got {actual_mean}"
        
    # Validate variance
    expected_variance = np.sqrt(delta) * sigma
    actual_variance = model.white_noise.std()
    assert np.isclose(actual_variance, expected_variance, rtol=0.2), \
        f"Expected variance close to {expected_variance}, but got {actual_variance}"
        
    # Validate variance setter
    new_sigma = 1
    model.sigma = new_sigma
    expected_variance = np.sqrt(delta) * new_sigma
    actual_variance = model.white_noise.std()
    assert np.isclose(actual_variance, expected_variance, rtol=0.2), \
        f"Expected variance close to {expected_variance}, but got {actual_variance}"

    # Validate delta setter
    new_delta = 1
    model.delta = new_delta
    expected_variance = np.sqrt(new_delta) * new_sigma
    actual_variance = model.white_noise.std()
    assert np.isclose(actual_variance, expected_variance, rtol=0.2), \
        f"Expected variance close to {expected_variance}, but got {actual_variance}"


def test_transform():
    num_paths = 2
    length = 10
    
    model = Model(length, num_paths)
    res = model.transform()
    # Check if the output data is a pandas Series or DataFrame
    assert isinstance(res, (pd.Series, pd.DataFrame)), \
        "Not a pd.Series or pd.DataFrame."
    # Validate that the dimensions of the data match the specified length and
    # number of paths
    assert pd.DataFrame(res).shape == (length, num_paths), \
        "Shape does not match settings."
