from dataclasses import dataclass


@dataclass
class SyntheticConfig:
    """
    A data class for configuration parameters used in synthetic data 
    generation.

    This data class defines various configuration parameters that can be used 
    in synthetic data generation methods.

    Attributes:
    ----------
    start_value : float, optional
        The starting asset value. Default is 100.0.
    start_rate : float, optional
        The starting interest rate value. Default is 0.5.
    num_paths : int, optional
        The number of paths to simulate. Default is 1.
    length : int, optional
        The length of the simulation. Default is 252.
    delta : float, optional
        The rate of time (e.g., 1/252 for daily). Default is 1/252.
    sigma : float, optional
        The volatility of the stochastic processes. Default is 0.125.
    matrix : np.ndarray or pd.DataFrame, optional
        The matrix applied in Cholesky decomposition (optional).
    gbm_mu : float, optional
        The annual drift factor for geometric Brownian motion. Default is 
        0.058.
    jumps_lambda : float, optional
        The probability of a jump happening at each point in time for Merton 
        model. Default is 0.00025.
    jumps_sigma : float, optional
        The volatility of the jump size for Merton model. Default is 0.001.
    jumps_mu : float, optional
        The average jump size for Merton model. Default is 0.2.
    cir_kappa : float, optional
        The rate of mean reversion for Cox Ingersoll Ross model. 
        Default is 3.0.
    cir_mu : float, optional
        The long-run average interest rate for Cox Ingersoll Ross model. 
        Default is 0.5.
    cir_rho : float, optional
        The correlation between the Wiener processes of the Heston model. 
        Default is 0.5.
    ou_kappa : float, optional
        The rate of mean reversion for Ornstein Uhlenbeck model. 
        Default is 3.0.
    ou_mu : float, optional
        The long-run average interest rate for Ornstein Uhlenbeck model. 
        Default is 0.5.
    heston_kappa : float, optional
        The rate of mean reversion for volatility in the Heston model. 
        Default is 3.
    heston_theta : float, optional
        The long-run average volatility for the Heston model. 
        Default is 0.20^2.
    heston_vol0 : float, optional
        The starting volatility value for the Heston model. 
        Default is 0.25^2.
    heston_sigma : float, optional
        The vol of vol / volatility of variance process for the Heston model. 
        Default is 0.6.
    heston_rf : float, optional
        The risk-free rate for the Heston model. Default is 0.02.
    heston_rho : float, optional
        The correlation between asset returns and variance for the Heston 
        model. Default is 0.7.
    levy_alpha : float, optional
        The alpha parameter in the Levy Stable diffusion. Default is 1.68.
    levy_beta : float, optional
        The beta parameter in the Levy Stable diffusion. Default is 0.01.

    """
    # General params

    # This is the starting asset value
    start_value: float = 100.0
    # This is the starting interest rate value
    start_rate: float = 0.5
    # This is the amount of time to simulate for
    num_paths: int = 1
    # This is the time decay to iterate through
    length: int = 252
    # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
    delta: float = 1/252
    # This is the volatility of the stochastic processes
    sigma: float = 0.125
    # This is the matrix applied on the cholesky decomposition (optional)
    matrix = None

    # GBM
    # ---
    # This is the annual drift factor for geometric brownian motion
    gbm_mu: float = 0.058

    # Merton
    # ------
    # This is the probability of a jump happening at each point in time
    jumps_lambda: float = 0.00025
    # This is the volatility of the jump size
    jumps_sigma: float = 0.001
    # This is the average jump size
    jumps_mu: float = 0.2

    # CIR
    # ---
    # This is the rate of mean reversion for Cox Ingersoll Ross
    cir_kappa: float = 3.0
    # This is the long run average interest rate for Cox Ingersoll Ross
    cir_mu: float = 0.5
    # This is the correlation between the wiener processes of the Heston model
    cir_rho: float = 0.5

    # Ornstein Uhlenbeck
    # ------------------
    # This is the rate of mean reversion for Ornstein Uhlenbeck
    ou_kappa: float = 3.0
    # This is the long run average interest rate for Ornstein Uhlenbeck
    ou_mu: float = 0.5

    # Heston
    # ------
    # This is the rate of mean reversion for volatility in the Heston model
    heston_kappa: float = 3
    # This is the long run average volatility for the Heston model
    heston_theta: float = 0.20**2
    # This is the starting volatility value for the Heston model
    heston_vol0: float = 0.25**2
    # This is the vol of vol / volatility of variance process for the Heston model
    heston_sigma: float = 0.6
    # This is the risk-free rate for the Heston model
    heston_rf: float = 0.02
    # This is the correlation between asset returns and variance for the Heston model
    heston_rho: float = 0.7

    # Levy Stable
    # -----------
    # This is the alpha parameter in the Levy Stable diffusion
    levy_alpha: float = 1.68
    # This is the beta parameter in the Levy Stable diffusion
    levy_beta: float = 0.01
