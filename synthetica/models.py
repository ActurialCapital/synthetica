import numpy as np
import pandas as pd
from scipy.stats import levy_stable, norm

from synthetica import BaseSynthetic


class GeometricBrownianMotion(BaseSynthetic):
    """
    A class for generating synthetic data using the Geometric Brownian Motion 
    (GBM) model.

    GBM is commonly used for modeling asset prices and is a fundamental 
    component of the Black-Scholes options pricing formula.

    Attributes
    ----------
    length : int | pd.DatetimeIndex, optional
        The length of the time series or a date index. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic process. Default is 0.125.
    mu : float, optional
        The expected return rate (drift) for the GBM. Default is 0.058.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Note
    ----
    This method constructs a sequence of log returns which, when exponentiated, 
    produce a random Geometric Brownian Motion. GBM is the stochastic 
    process underlying the Black Scholes options pricing formula.

    Example
    -------
    >>> gbm_model = GeometricBrownianMotion(
    ...     length=252, 
    ...     num_paths=1, 
    ...     delta=1/252, 
    ...     sigma=0.125, 
    ...     mu=0.058
    ... )
    >>> synthetic_data = gbm_model.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        mu: float = 0.058,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.mu = mu

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic Geometric Brownian Motion (GBM) data.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic GBM data.
        """
        sigma_pow_mu_delta = (
            self.mu - 0.5 * np.power(self.sigma, 2.0)
        ) * self.delta

        noise = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )

        paths = np.array(noise) + sigma_pow_mu_delta
        prices = self.to_prices(paths)

        return self.to_pandas(prices)


class Heston(BaseSynthetic):
    """
    A class for generating synthetic data using the Heston model.

    This class generates synthetic data based on the Heston model, which is 
    commonly used for modeling stochastic volatility. It combines a stochastic 
    volatility process with geometric Brownian motion.

    Attributes
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic processes. Default is 0.125.
    matrix : pd.DataFrame or np.array, optional
        The matrix applied in Cholesky decomposition (optional).
    start_value : float, optional
        The starting value for the process. Default is 100.0.
    rho : float, optional
        The correlation coefficient between two Brownian motions. Default 
        is 0.7.
    vol0 : float, optional
        The initial volatility value. Default is 0.25^2.
    rf : float, optional
        The risk-free rate. Default is 0.02.
    kappa : float, optional
        The rate of mean reversion for the volatility. Default is 3.
    theta : float, optional
        The long-run average volatility. Default is 0.20^2.
    nu : float, optional
        The volatility of the volatility. Default is 0.6.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Example
    -------
    >>> heston_model = Heston(
    ...     length=252, 
    ...     num_paths=1, d
    ...     elta=1/252, 
    ...     sigma=0.125, 
    ...     start_value=100.0, 
    ...     rho=0.7, 
    ...     vol0=0.25^2, 
    ...     rf=0.02, kappa=3, 
    ...     theta=0.20^2, 
    ...     nu=0.6
    ... )
    >>> synthetic_data = heston_model.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        start_value: float = 100.0,
        rho: float = 0.7,
        vol0: float = 0.25**2,
        rf: float = 0.02,
        kappa: float = 3,
        theta: float = 0.20**2,
        nu: float = 0.6,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.start_value = start_value
        self.rho = rho
        self.vol0 = vol0
        self.rf = rf
        self.kappa = kappa
        self.theta = theta
        self.nu = nu

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic Heston model data.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic Heston model data.
        """
        bm_volatility = np.random.multivariate_normal(
            mean=np.array([0, 0]),
            cov=np.array([[1, self.rho], [self.rho, 1]]),
            size=(self.length, self.num_paths)
        )

        if matrix is not None:
            bm_volatility = self.cholesky_transform(bm_volatility, matrix)

        # arrays for storing prices and variances
        prices = np.full(
            shape=((self.length+1, self.num_paths)),
            fill_value=self.start_value,
            dtype=np.float64
        )
        volatility = np.full(
            shape=((self.length+1, self.num_paths)),
            fill_value=self.vol0,
            dtype=np.float64
        )

        for length in range(1, self.length + 1):
            prices[length] = (
                prices[length-1]
                * np.exp(
                    (self.rf - 0.5 * volatility[length-1])
                    * self.delta
                    + np.sqrt(volatility[length-1] * self.delta)
                    * bm_volatility[length-1, :, 0]
                )
            )
            prev_vol = (
                volatility[length-1] + self.kappa *
                (self.theta - volatility[length-1]) *
                self.delta + self.nu *
                np.sqrt(volatility[length-1] * self.delta) *
                bm_volatility[length-1, :, 1]
            )
            volatility[length] = np.maximum(prev_vol, 0)

        return self.to_pandas(prices[1:,])


class Merton(BaseSynthetic):
    """
    A class for generating synthetic data using the Merton model.

    This class generates synthetic data based on the Merton model, which is 
    used to model asset prices in the presence of jumps. It combines a jump 
    diffusion process with geometric Brownian motion.

    Attributes
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic process. Default is 0.125.
    lmbda : float, optional
        The jump intensity parameter. Default is 0.00025.
    var : float, optional
        The variance of jump sizes. Default is 0.001.
    mu : float, optional
        The expected return rate (drift) for the Merton model. Default is 0.2.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Note
    ------
    This method produces a sequence of Jump Sizes which represent a jump 
    diffusion process. These jumps are combined with a geometric brownian 
    motion (log returns) to produce the Merton model.

    Example
    -------
    >>> merton_model = Merton(
    ...     length=252, 
    ...     num_paths=1, 
    ...     delta=1/252, 
    ...     sigma=0.125, 
    ...     lmbda=0.00025, 
    ...     var=0.001, 
    ...     mu=0.2
    ... )
    synthetic_data = merton_model.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        lmbda: float = 0.00025,
        var: float = 0.001,
        mu: float = 0.2,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.lmbda = lmbda
        self.var = var
        self.mu = mu

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic Merton model data.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic Merton model data.
        """
        small_lmbda = - (1.0 / self.lmbda)
        jump = np.zeros((self.length, self.num_paths), dtype=np.float64)

        for path in range(self.num_paths):
            s_n = time = 0

            while s_n < self.length:
                s_n += small_lmbda * np.log(np.random.uniform(0, 1))

                for length in range(self.length):
                    if time * self.delta <= s_n * self.delta <= (length + 1) * self.delta:
                        rand = np.random.normal(self.mu, self.var)
                        jump[length, path] += rand
                        break

                time += 1

        noise = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )

        paths = np.add(jump, noise)
        prices = self.to_prices(paths)

        return self.to_pandas(prices)


class Poisson(BaseSynthetic):
    """
    A class for generating synthetic data using the poisson (jump) process 
    model.

    This class generates synthetic data using the poisson model, which 
    combines a jump diffusion process with a Geometric Brownian Motion (GBM).

    Attributes
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the GBM component. Default is 0.125.
    lmbda : float, optional
        The jump intensity parameter. Default is 0.00025.
    var : float, optional
        The jump size variance. Default is 0.001.
    mu : float, optional
        The mean of jump sizes. Default is 0.2.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Note
    ----
    This method produces a sequence of jump sizes which represent a jump 
    diffusion process. These jumps are combined with a geometric brownian 
    motion (log returns). This model is an alternative to the Merton model.

    Example
    -------
    >>> poisson = Poisson(
    ...     length=252, 
    ...     num_paths=1, 
    ...     delta=1/252, 
    ...     sigma=0.125, 
    ...     lmbda=0.00025, 
    ...     var=0.001, 
    ...     mu=0.2
    ... )
    synthetic_data = poisson.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        lmbda: float = 0.00025,
        var: float = 0.001,
        mu: float = 0.2,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.lmbda = lmbda
        self.var = var
        self.mu = mu

    @property
    def lambda_poisson(self) -> float:
        return 2 * (1 / self.length)

    @lambda_poisson.setter
    def lambda_poisson(self, l: int | float = 2) -> float:
        self.lambda_poisson = l * (1 / self.length)

    def _inverse_poisson(self) -> int:
        """
        Calculate the inverse of the Poisson cumulative distribution function.

        Parameters:
        ----------
        u : float
            Random uniform variable.
        lmbda : float
            Lambda parameter for Poisson distribution.

        Returns
        -------
        int
            Inverse Poisson random variable.
        """
        poisson = np.exp(-self.lambda_poisson)
        F = poisson
        k = 0
        u = np.random.rand()
        while u > F:
            k = k + 1
            poisson = poisson * (self.lambda_poisson / k)
            F = F + poisson

        return k

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic data using the Poisson model.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic data following the poisson process model.
        """
        jump = np.zeros((self.length, self.num_paths))
        for path in range(self.num_paths):
            for length in range(self.length):
                alea_poisson = self._inverse_poisson()
                if alea_poisson != 0:
                    for j in range(0, alea_poisson):
                        jump[length, path] += norm.ppf(
                            np.random.rand(),
                            loc=self.mu,
                            scale=self.var
                        )

        noise = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )

        paths = np.add(jump, noise)
        prices = self.to_prices(paths)

        return self.to_pandas(prices)


class LevyStable(BaseSynthetic):
    """
    A class for generating synthetic data using the Levy Stable Process model.

    This class generates synthetic data using the Levy Stable Process model, 
    which generalizes several distributions including left-skewed Levy, Levy, 
    Cauchy, and Normal distributions.

    Attributes
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The scale parameter of the Levy Stable distribution. Default is 0.125.
    alpha : float, optional
        The alpha parameter of the Levy Stable distribution. Should be in the 
        range (0, 2].
        Default is 1.68.
    beta : float, optional
        The beta parameter of the Levy Stable distribution. Should be in the 
        range [-1, 1].
        Default is 0.01.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Note
    ----
    This method returns a Levy stable process. levy_stable generalizes 
    several distributions: 0 < alpha <= 2 and -1 <= beta <= 1, such as:

    | Alpha | Beta | Equivalent | Description        |
    |-------|------|------------|--------------------|
    | 0.5   | -1   | levy_l     | Left-skewed Levy   |
    | 0.5   | 1    | levy       | Levy               |
    | 1     | 0    | cauchy     | Cauchy             |
    | 2     | Any  | norm       | Normal             |

    Example
    -------
    >>> levy_stable_model = LevyStable(
    ...     length=252, 
    ...     num_paths=1, 
    ...     delta=1/252, 
    ...     sigma=0.125, 
    ...     alpha=1.68, 
    ...     beta=0.01
    ... )
    >>> synthetic_data = levy_stable_model.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        alpha: float = 1.68,
        beta: float = 0.01,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.alpha = alpha
        self.beta = beta

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic data using the Levy Stable Process model.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic data following the Levy Stable Process model.
        """
        paths = levy_stable.rvs(
            self.alpha,
            self.beta,
            loc=self.mean,
            scale=np.sqrt(self.delta) * self.sigma,
            size=(self.length, self.num_paths)
        )

        if matrix is not None:
            paths = self.cholesky_transform(paths, matrix)

        prices = self.to_prices(paths)

        return self.to_pandas(prices)


class CIR(BaseSynthetic):
    """
    A class for generating synthetic data using the Cox-Ingersoll-Ross (CIR) 
    model.

    This class generates synthetic interest rate data based on the 
    Cox-Ingersoll-Ross (CIR) model.

    The CIR model is a mean-reverting model commonly used for modeling interest 
    rates.

    Attributes
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic processes. Default is 0.125.
    start_value : float, optional
        The starting value. Default is 0.5.
    kappa : float, optional
        The rate of mean reversion for the CIR model. Default is 3.0.
    mu : float, optional
        The long-run average interest rate for the CIR model. Default is 0.5.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Note
    ----
    This method returns the rate levels of a mean-reverting cox ingersoll 
    ross process. It is used to model interest rates as well as stochastic 
    volatility in the Heston model. Because the returns between the 
    underlying and the stochastic volatility should be correlated we pass a 
    correlated Brownian motion process into the method from which the 
    interest rate levels are constructed. The other correlated process is 
    used in the Heston model.

    Example
    --------
    >>> cir_model = CIR(
    ...     length=252, 
    ...     num_paths=1, 
    ...     delta=1/252, 
    ...     sigma=0.125, 
    ...     start_value=0.5, 
    ...     kappa=3.0, 
    ...     mu=0.5
    ... )
    >>> synthetic_data = cir_model.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        start_value: float = 0.5,
        kappa: float = 6.0,
        mu: float = 0.5,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.start_value = start_value
        self.kappa = kappa
        self.mu = mu

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic interest rate data using the Cox-Ingersoll-Ross 
        (CIR) model.

        Note
        ----
        The main difference between this and the Ornstein Uhlenbeck model
        is that we multiply the 'random' component by the square-root of
        the previous level i.e. the process has level dependent interest
        rates.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic interest rate data.
        """
        paths = np.full((self.length+1, self.num_paths), self.start_value)

        noise = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )

        for length in range(1, self.length + 1):
            drift = (
                self.kappa * (self.mu - paths[length-1]) * self.delta
            )
            deviation = (
                np.sqrt(paths[length-1]) * noise[length-1]
            )
            paths[length] = paths[length-1] + drift + deviation

        return self.to_pandas(paths[1:,])


class MeanReverting(BaseSynthetic):
    """
    A class for generating synthetic data using a mean-reverting 
    Ornstein-Uhlenbeck model.

    Note
    ----
    This class generates synthetic data using a mean-reverting
    Ornstein-Uhlenbeck model. The Ornstein-Uhlenbeck model is commonly used for 
    modeling mean-reverting processes.

    Attributes
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic processes. Default is 0.125.
    start_value : float, optional
        The starting value. Default is 0.5.
    kappa : float, optional
        The rate of mean reversion for the Ornstein-Uhlenbeck model. Default 
        is 3.0.
    mu : float, optional
        The long-run average interest rate for the Ornstein-Uhlenbeck model.
        Default is 0.5.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.

    Example
    -------
    >>> ou_model = MeanReverting(
    ...     length=252, 
    ...     num_paths=1, 
    ...     delta=1/252, 
    ...     sigma=0.125, 
    ...     start_value=0.5, 
    ...     kappa=3.0, 
    ...     mu=0.5
    ... )
    >>> synthetic_data = ou_model.transform()
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        mean: float = 0,
        delta: float = 1/252,
        sigma: float = 0.125,
        start_value: float = 0.5,
        kappa: float = 6.0,
        mu: float = 0.5,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.start_value = start_value
        self.kappa = kappa
        self.mu = mu

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic mean-reverting data using the Ornstein-Uhlenbeck
        model.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic mean-reverting data.
        """
        paths = np.full(
            (self.length+1, self.num_paths), self.start_value, dtype=np.float64
        )

        noise = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )

        for length in range(1, self.length + 1):
            drift = (
                self.kappa * (self.mu - paths[length-1]) * self.delta
            )
            deviation = noise[length - 1]

            paths[length] = paths[length-1] + drift + deviation

        return self.to_pandas(paths[1:,])


class AR(BaseSynthetic):
    """
    A class for generating synthetic autoregressive (AR) model.

    Generates time series with an autogressive lag defined by the number of 
    parameters.

    Parameters:
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic processes. Default is 0.125.
    start_value : float, optional
        The starting value. Default is 0.5.
    ar : list of float, optional
        Coefficients for the AR model. The number of coefficients defines the 
        order. E.g., [0.8, -0.2] for an AR(2) process. IF None, it defaults to
        [0.8]. Dafaults to None.
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
        ar=None,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.ar = ar or [0.8]
        self.order = len(self.ar)

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic autoregressive (AR) data.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing synthetic autoregressive (AR) data.

        """
        paths = np.zeros((self.length+1, self.num_paths), dtype=np.float64)

        noise = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )

        for length in range(1, self.length + 1):
            deviation = noise[length - 1]
            paths[length] = (
                sum([self.ar[j] * paths[length - j - 1] for j in range(self.order)]) + deviation
            )

        return self.to_pandas(paths[1:,])


class NARMA(BaseSynthetic):
    """
    A class for generating synthetic non-linear Autoregressive Moving Average 
    model.

    An n-th order signal generator of the NARMA class, as defined in the 
    Harvard University official documentation[^1]:

    $$ 
    y(k+1) = a_0 y(k) + a_1 y(k) \sum_{i=0}^{n-1} y(k-i) + a_2 u(k-(n-1)) u(k) + a_3
    $$

    where $u$ is generated from $U(0, 0.5)$.

    [^1]: Harvard University [official documentation](http://ieeexplore.ieee.org.ezp-prod1.hul.harvard.edu/stamp/stamp.jsp?arnumber=846741)


    Parameters
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    start_value : float, optional
        The starting value. Default is 0.5.
    n : int, optional
        The order of the NARMA process. Default is 10.
    a : list, optional
        Coefficients for the NARMA model. When None, we use reference 
        coefficients [0.3, 0.05, 1.5, 0.1]. Default is None.
    freq : str, optional
        The frequency of the data. Default is 'D'.
    seed : int, optional
        The random seed for reproducibility. Default is None.
    """

    def __init__(
        self,
        length: int | pd.DatetimeIndex = 252,
        num_paths: int = 1,
        start_value: float = 0.5,
        n: int = 10,
        a: list | None = None,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            freq=freq,
            seed=seed
        )
        self.n = n
        self.a = [0.3, 0.05, 1.5, 0.1] or a
        self.start_value = start_value

        if len(self.a) != 4:
            raise ValueError(
                "We need 4 Coefficients for the NARMA model. Number of coeff is "
                f"{len(self.a)}."
            )

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic Non-linear Autoregressive Moving Average (NARMA) 
        data.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing non-linear Autoregressive Moving Average (NARMA) data.
        """
        paths = np.full(
            (self.length+1, self.num_paths),
            self.start_value,
            dtype=np.float64
        )
        # Uniform data
        u = np.random.uniform(
            low=0,
            high=0.5,
            size=(self.length + 1, self.num_paths)
        )

        for length in range(1, self.length + 1):
            paths[length] = (
                self.a[0] * paths[length - 1] +
                self.a[1] * paths[length - 1] * sum(paths[length - self.n:length]) +
                self.a[2] * u[length - self.n] * u[length] +
                self.a[3]
            )

        if matrix is not None:
            paths = self.cholesky_transform(paths, matrix)

        return self.to_pandas(paths[1:,])


class Seasonal(BaseSynthetic):
    """
    A class for generating seasonal patterns model. The seasonal pattern is 
    generated using the sinusoidal function

    Note
    ----
    The data generation process compute: a sin(2π ft + p).
    where t is a time vector from 0 to length - 1, a is the random amplitude 
    values with a normal distribution (mean=1, std=0.1), f is the random 
    frequency values with a normal distribution (mean=0.1, std=0.01). We then 
    adjust the mean and std based on the desired seasonality. Finally, p is the 
    random phase shifts uniformly distributed between 0 and 2π.

    Parameters
    ----------
    length : int, optional
        The length of the time series. Default is 252.
    num_paths : int, optional
        The number of paths to generate. Default is 1.
    mean : float, optional
        Mean (“centre”) of the distribution. Defaults to 0.
    delta : float, optional
        The time step size (e.g., 1/252 for daily data). Default is 1/252.
    sigma : float, optional
        The volatility of the amplitude process. Default is 0.125.
    omega : float, optional
        How often the wave oscillates or repeats its cycle per unit time. It 
        represents the rate of change of the phase of a sinusoidal waveform.
        Defaults to 1.0.
    phi : float, optional
        Represents the phase angle, the duration of one cycle of the wave or 
        the variation of the frequency (omega)
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
        omega: float = 1.0,
        phi: float = 0.4,
        freq: str = 'D',
        seed: int = None
    ):
        super().__init__(
            length=length,
            num_paths=num_paths,
            mean=mean,
            delta=delta,
            sigma=sigma,
            freq=freq,
            seed=seed
        )
        self.omega = omega
        self.phi = phi

    def transform(self, matrix: pd.DataFrame | np.ndarray = None) -> pd.Series | pd.DataFrame:
        """
        Generate synthetic seasonal patterns data.

        Parameters
        ----------
        matrix : pd.DataFrame or np.array, optional
            The matrix applied in Cholesky decomposition (optional).

        Returns
        -------
        pd.Series | pd.DataFrame
            Data containing seasonal patterns.

        """
        # A time vector from 0 to length - 1.
        t = np.arange(self.length).reshape(-1, 1)
        
        # Random amplitude values
        amplitude = (
            self.cholesky_transform(self.white_noise, matrix)
            if matrix is not None
            else self.white_noise
        )
       
        #  Random frequency values
        frequency = np.random.normal(
            loc=self.omega,
            scale=self.phi,
            size=(self.length, self.num_paths)
        )
        
        # Random phase shifts uniformly distributed between 0 and 2pi.
        phase = np.random.uniform(
            low=0,
            high=2*np.pi,
            size=(self.length, self.num_paths)
        )
        
        # Generate the seasonal patterns
        paths = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        return self.to_pandas(paths)
