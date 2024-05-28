import pandas as pd
import synthetica as sth


estimator = sth.Seasonal


if __name__ == "__main__":

# =============================================================================
#     # Model
# =============================================================================
    model = estimator(length=252, num_paths=1, seed=124)
    print(model)
    # Seasonal

# =============================================================================
#     # Output
# =============================================================================
    df = model.transform()
    print(df)
    # 2023-09-20    0.000786
    # 2023-09-21   -0.002465
    # 2023-09-22   -0.008799
    # 2023-09-23    0.005338
    # 2023-09-24    0.003796
      
    # 2024-05-24    0.009035
    # 2024-05-25   -0.013374
    # 2024-05-26    0.004433
    # 2024-05-27    0.002473
    # 2024-05-28    0.001302
    # Freq: D, Name: symbol, Length: 252, dtype: float64

    df.plot()
    # --matplotlib plt--

# =============================================================================
#     # Underlying white noise
# =============================================================================
    noise = model.white_noise
    print(noise)
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
    # ...

    pd.DataFrame(noise).plot()
    # --matplotlib plt--

# =============================================================================
#     # Cholesky
# =============================================================================

    seed = 9
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

