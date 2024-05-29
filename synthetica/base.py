from abc import ABC, abstractmethod
from typing import Iterable
from functools import cached_property
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import synthetica as sth


class BaseSynthetic(ABC):
    """
    A base class for synthetic data generation methods.

    This mixin class provides methods for synthetic data generation using 
    Cholesky decomposition and other related calculations.

    Attributes
    ----------
    length : int | pd.DatetimeIndex
        The length of the time series or a pandas DatetimeIndex.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic processes. Default is 0.125.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        freq: str = 'D',
        seed: int = None
    ):
        # Generic
        self._length = length
        self._freq = freq
        self._num_paths = num_paths
        if seed is not None:
            np.random.seed(seed)
        # White noise params
        self._mean = mean
        self._delta = delta
        self._sigma = sigma

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def freq(self) -> str:
        return self._freq

    @property
    def index(self) -> pd.DatetimeIndex:
        if isinstance(self._length, pd.DatetimeIndex):
            return self._length
        else:
            # Calculate the nt ago from today
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self._length - 1)
            return pd.date_range(start=start_date, end=end_date, freq=self.freq)

    @property
    def length(self) -> int:
        return len(self.index)

    @property
    def num_paths(self) -> int:
        return self._num_paths

    @abstractmethod
    def transform(self, *args, **kwargs) -> pd.Series | pd.DataFrame:
        pass

    @cached_property
    def white_noise(self) -> np.ndarray:
        """
        Generate white noise (Wiener process).

        Note
        ----
        White noise has zero mean, constant variance, and is uncorrelated in 
        time. As its name suggests, white noise has a power spectrum which is 
        uniformly spread across all allowable frequencies.

        Returns
        -------
        np.ndarray
            White noise (Wiener process) paths.
        """
        return np.random.normal(
            loc=self.mean,
            scale=np.sqrt(self.delta) * self.sigma,
            size=(self.length, self.num_paths)
        )

    @property
    def mean(self) -> float:
        """Mean value"""
        return self._mean

    @mean.setter
    @sth.callback('white_noise')
    def mean(self, value) -> float:
        """Mean value update"""
        if value != self._mean:
            self._mean = value

    @property
    def delta(self) -> float:
        """Delta value"""
        return self._delta

    @delta.setter
    @sth.callback('white_noise')
    def delta(self, value) -> float:
        """Delta value update"""
        if value != self._delta:
            self._delta = value

    @property
    def sigma(self) -> float:
        """Sigma value"""
        return self._sigma

    @sigma.setter
    @sth.callback('white_noise')
    def sigma(self, value) -> float:
        """Sigma value update"""
        if value != self._sigma:
            self._sigma = value

    @cached_property
    def red_noise(self) -> np.ndarray:
        """
        Generate red noise (correlated noise).

        Note
        ----
        Red noise has zero mean, constant variance, and is serially correlated 
        in time, such that the lag-1 autocorrelation between two successive 
        time samples has correlation coefficient 0 < r < 1. Red noise has a
        power spectrum weighted toward low frequencies, but has no single 
        preferred period.

        Returns
        -------
        np.ndarray
            Red noise paths.
        """
        if not hasattr(self, 'tau'):
            raise AttributeError(f"{self} does not integrate red noise.")

        # Initialize the red noise array
        red_noise = np.zeros((self.length, self.num_paths))
        previous_value = None

        for i, noise in enumerate(self.white_noise):
            time_diff = self.delta  # Assuming constant time intervals

            if previous_value is None:
                previous_value = noise
            else:
                red_noise[i] = (
                    (self.tau / (self.tau + time_diff)) *
                    (time_diff * noise + previous_value)
                )

            previous_value = red_noise[i]

        return red_noise

    # #### Cholesky #### #

    def create_corr_returns(self, matrix) -> pd.Series | pd.DataFrame:
        """
        This method can construct a basket of correlated asset paths using the 
        Cholesky decomposition method.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array
            The matrix applied in Cholesky decomposition.

        Returns
        -------
        pd.Series or pd.DataFrame:
            Data representing correlated log returns.
        """
        try:
            decomposition = np.linalg.cholesky(matrix)

        except:
            updated_matrix = sth.nearest_positive_definite(matrix)
            decomposition = np.linalg.cholesky(updated_matrix)

        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma

        # Construct uncorrelated paths to convert into correlated paths
        uncorrelated_paths = [
            np.array(
                # Uncorrelated random numbers
                [np.random.normal(0, sqrt_delta_sigma)
                 for _ in range(self.num_paths)]
            )
            for _ in range(self.length + 1)
        ]

        uncorrelated_matrix = np.matrix(uncorrelated_paths)
        correlated_matrix = uncorrelated_matrix * decomposition
        extracted_paths = [[] for _ in range(1, self.num_paths + 1)]
        for j in range(0, len(correlated_matrix) * self.num_paths - self.num_paths, self.num_paths):
            for i in range(self.num_paths):
                extracted_paths[i].append(correlated_matrix.item(j + i))

        return self.to_pandas(np.array(extracted_paths).T)

    def cholesky_transform(self, rvs: np.array, matrix: np.array) -> np.ndarray:
        """
        Perform Cholesky transformation on random variables.

        Parameters
        ----------
        rvs : np.array
            Random variables to transform.
        matrix : np.array
            The matrix for Cholesky decomposition.

        Returns
        -------
        np.ndarray
            Transformed random variables.
        """
        try:
            decomposition = np.linalg.cholesky(matrix)

        except:
            # raise Warning('Covariance is not positive definite.')
            positive_definite_matrix = sth.nearest_positive_definite(matrix)
            decomposition = np.linalg.cholesky(positive_definite_matrix)

        # return (decomposition @ rvs.T).T
        return np.matmul(decomposition, rvs.T).T

    # #### Converter #### #

    @staticmethod
    def to_returns(log_returns: Iterable) -> np.ndarray:
        """
        This method exponentiates a sequence of log-returns to returns.

        Parameters
        ----------
        log_returns : Iterable
            An iterable containing log returns.

        Returns
        -------
        np.ndarray
            An array of returns.
        """
        return np.exp(log_returns)

    def to_prices(
        self,
        log_returns: Iterable,
        start_value: float = 100.0
    ) -> np.ndarray:
        """
       This method converts a sequence of log returns into normal returns 
       (exponentiation) and then computes a price sequence given a starting 
       price, start_value.

       Parameters
       ----------
       log_returns : Iterable
           An iterable containing log returns.
       start_value : float, optional
           The starting value of the price sequence. Default is 100.0.

       Returns
       -------
       np.ndarray
           An array of price sequences.
       """

        log_returns = np.cumsum(log_returns, axis=0)

        returns = self.to_returns(log_returns)
        # A sequence of prices starting with start_value
        return returns * start_value

    def to_pandas(self, output) -> pd.Series | pd.DataFrame:
        """
        Convert synthetic output to a pandas DataFrame or Series.

        Parameters
        ----------
        output : np.ndarray
            The synthetic output to convert.

        Returns
        -------
        pd.DataFrame or pd.Series
            A pandas DataFrame or Series containing the converted data.
        """
        return (
            pd.Series(output[:, 0], index=self.index, name='symbol')
            if output.shape[1] == 1
            else pd.DataFrame(output, index=self.index, columns=['path_' + str(i+1) for i in range(self.num_paths)])
            .rename_axis('Date', axis=0)
            .rename_axis('symbol', axis=1)
        )
