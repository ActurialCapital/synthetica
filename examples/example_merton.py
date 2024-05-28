import pandas as pd
import synthetica as sth


estimator = sth.Merton


if __name__ == "__main__":

# =============================================================================
#     # Model
# =============================================================================
    model = estimator(length=252, num_paths=1, seed=124)
    print(model)
    # GeometricBrownianMotion()

# =============================================================================
#     # Output
# =============================================================================
    df = model.transform()
    print(df)
    # 2023-09-20    100.247377
    # 2023-09-21     99.902493
    # 2023-09-22     98.875156
    # 2023-09-23    100.715725
    # 2023-09-24     99.578532

    # 2024-05-24    124.738211
    # 2024-05-25    123.043536
    # 2024-05-26    122.088517
    # 2024-05-27    121.475601
    # 2024-05-28    121.292057
    # Freq: D, Name: symbol, Length: 252, dtype: float64

    df.plot()
    # --matplotlib plt--

# =============================================================================
#     # Underlying white noise
# =============================================================================
    noise = model.white_noise
    print(noise)
    # [[ 2.27155909e-03]
    #  [-3.64542074e-03]
    #  [-1.05357945e-02]
    #  [ 1.82447835e-02]
    #  [-1.15544994e-02]

    # ...

    pd.DataFrame(noise).plot()
    # --matplotlib plt--

# =============================================================================
#     # Testing callback
# =============================================================================
    # Mean
    # ----
    mean_value = model.mean
    # 0
    model.mean = 1
    noise = model.white_noise
    print(noise)
    # [[0.99862152]
    #  [1.00205968]
    #  [0.98505675]
    #  [1.00049552]
    #  [0.98925715]

    # ...

    pd.DataFrame(noise).plot()
    # --matplotlib plt--

    # Delta
    # -----
    model.delta
    # 0.003968253968253968
    model.delta = 1/12
    noise = model.white_noise
    print(noise)
    # [[0.96191091]
    #  [1.05444961]
    #  [0.9932223 ]
    #  [0.96212298]
    #  [0.96547194]

    # ...

    pd.DataFrame(noise).plot()
    # --matplotlib plt--

    # Sigma
    # -----
    model.sigma
    # 0.125
    model.sigma = 0.4
    noise = model.white_noise
    print(noise)
    # [[1.11282455]
    #  [0.97786102]
    #  [1.09209923]
    #  [1.03305848]
    #  [0.94427619]

    # ...

    pd.DataFrame(noise).plot()
    # --matplotlib plt--

# =============================================================================
#     # Red noise if any
# =============================================================================
    try:
        model.red_noise
    except AttributeError as e:
        print(e)
    # GeometricBrownianMotion does not integrate red noise.

# =============================================================================
#     # Cholesky
# =============================================================================

    seed = 92
    num_paths = 2

    # Create matrix for illustration purposes
    model = estimator(num_paths=num_paths, seed=seed)
    df1 = model.transform()
    print(df1)
    df1.plot()
    # --matplotlib plt--

    # # With correlated variables
    model = estimator(num_paths=num_paths, matrix=df1.cov(), seed=seed)
    df2 = model.transform()
    print(df2)
    df2.plot()
    # --matplotlib plt--

    print(df1.corr())
    print(df2.corr())
