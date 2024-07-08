import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal

from synthetica import GeometricBrownianMotion, nearest_positive_definite

Model = GeometricBrownianMotion
index = pd.date_range("2010-01-01", "2020-02-01", inclusive="left")
num_paths = 3
length = len(index)
mean = 1
delta = 0.01
sigma = 0.2
seed = None

# Positive definite matrix
matrix = np.array([
    [1,     .7,     -.7],   # path_1
    [.7,     1,      .0],   # path_2
    [-.7,   .0,       1]])  # path_3


def test_random_seed():
    model = Model(length, num_paths)
    # cached
    arr1 = model.white_noise
    arr2 = model.white_noise
    assert np.array_equal(arr1, arr2)

    model.seed = 123
    arr1 = model.white_noise
    arr2 = model.white_noise
    assert np.array_equal(arr1, arr2)

    model.seed = 432
    arr3 = model.white_noise
    arr4 = model.white_noise
    assert np.array_equal(arr3, arr4)

    assert not np.array_equal(arr1, arr4)

    model = Model(length, num_paths)
    arr1 = model.create_corr_returns(matrix).to_numpy()
    arr2 = model.create_corr_returns(matrix).to_numpy()
    assert not np.array_equal(arr1, arr2)

    model.seed = 432
    arr1 = model.create_corr_returns(matrix).to_numpy()
    arr2 = model.create_corr_returns(matrix).to_numpy()
    assert np.array_equal(arr1, arr2)


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


@pytest.mark.parametrize("random_seed", [123, 234, 495, 221, 330, 23, 982])
def test_cholesky_transform(random_seed):
    model = Model(
        length=length,
        num_paths=num_paths,
        mean=mean,
        delta=delta,
        sigma=sigma,
        seed=random_seed
    )

    rvs = model.white_noise

    arr = model.cholesky_transform(rvs, matrix)
    df = model.to_pandas(arr)

    # Test standard deviation
    calculated_std = df.std()
    expected_std = np.sqrt(delta) * sigma
    assert np.allclose(calculated_std, expected_std, atol=0.01), \
        f"Expected std deviation {expected_std}, but got {calculated_std}"

    calculated_corr = df.corr()
    expected_corr = matrix / \
        np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    assert np.allclose(calculated_corr, expected_corr, atol=0.05), \
        f"Expected correlation matrix {expected_corr}, but got {calculated_corr}"


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


@pytest.mark.parametrize("random_seed", [123, 234, 495, 221, 330, 23, 982])
def test_create_corr_returns(random_seed):
    model = Model(
        length=length,
        num_paths=num_paths,
        mean=mean,
        delta=delta,
        sigma=sigma,
        seed=random_seed
    )
    df = model.create_corr_returns(matrix)

    calculated_mean = df.mean()
    assert np.allclose(calculated_mean, 0, atol=0.01), \
        f"Expected mean {mean}, but got {calculated_mean}"

    # Test standard deviation
    calculated_std = df.std()
    expected_std = np.sqrt(delta) * sigma
    assert np.allclose(calculated_std, expected_std, atol=0.01), \
        f"Expected std deviation {expected_std}, but got {calculated_std}"

    calculated_corr = df.corr()
    expected_corr = matrix / \
        np.sqrt(np.outer(np.diag(matrix), np.diag(matrix)))
    assert np.allclose(calculated_corr, expected_corr, atol=0.05), \
        f"Expected correlation matrix {expected_corr}, but got {calculated_corr}"


def test_create_corr_returns_positive_definite():
    num_paths = 2

    matrix = np.array([[2, 1], [1, 2]])  # Positive definite matrix

    model = Model(length, num_paths, mean, delta, sigma, seed=123)
    res = model.create_corr_returns(matrix)

    assert isinstance(res, pd.DataFrame), \
        "Expected result to be a pandas DataFrame"
    assert res.shape == (length, num_paths), \
        f"Expected shape {(length, num_paths)}, but got {res.shape}"


def test_create_corr_returns_non_positive_definite():
    num_paths = 2

    matrix = np.array([[1, -2], [2, 1]])  # Not positive definite matrix

    model = Model(length, num_paths, mean, delta, sigma, seed=seed)

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
    model = Model(length, num_paths, mean, delta, sigma, seed=123)
    with pytest.raises(AttributeError):
        model.red_noise


def test_white_noise():
    model = Model(length, num_paths, mean, delta, sigma, seed=123)
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


def test_to_pandas_dataframe():
    model = Model(length, num_paths)
    res = model.transform()
    # Check if the output data is a pandas Series or DataFrame
    assert isinstance(res, pd.DataFrame), \
        "Not a pd.DataFrame."
    # Validate that the dimensions of the data match the specified length and
    # number of paths
    assert pd.DataFrame(res).shape == (length, num_paths), \
        "Shape does not match settings."


def test_to_pandas_series():
    num_paths = 1

    model = Model(length, num_paths)
    res = model.transform()
    # Check if the output data is a pandas Series or DataFrame
    assert isinstance(res, pd.Series), \
        "Not a pd.Series."
    # Validate that the dimensions of the data match the specified length and
    # number of paths
    assert pd.DataFrame(res).shape == (length, num_paths), \
        "Shape does not match settings."
