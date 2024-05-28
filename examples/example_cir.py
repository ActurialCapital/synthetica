import pandas as pd
import synthetica as sth


estimator = sth.CIR


if __name__ == "__main__":

# =============================================================================
#     # Model
# =============================================================================
    model = estimator(length=252, num_paths=1, seed=124)
    print(model)

# =============================================================================
#     # Output
# =============================================================================
    df = model.transform()
    print(df)
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
