from pathlib import Path
import pandas as pd
import numpy as np
import synthetica as sth

# Plot theme
pd.options.plotting.backend = "plotly"
template = 'simple_white'

# Path
PATH_FOLDER = Path(__file__).parent.absolute().parent / 'docs' / 'static'


def save_to_png(
    png_name: str,
    model: sth.BaseSynthetic,
    func: str,
    cumulative: bool = False,
    **func_kwargs
):
    df = getattr(model, func)(**func_kwargs)
    if cumulative:
        df = np.cumprod(1 + df)
    fig = df.plot(template=template)
    fig.write_image(PATH_FOLDER / png_name)


if __name__ == "__main__":

    params = dict(length=252, num_paths=5, seed=123)
    # matrix = np.array([[1, .8], [.8, 1]])
    matrix = np.array([[1, 0.8, 0.6, 0.4, 0.2],
                       [0.8, 1, 0.8, 0.6, 0.4],
                       [0.6, 0.8, 1, 0.8, 0.6],
                       [0.4, 0.6, 0.8, 1, 0.8],
                       [0.2, 0.4, 0.6, 0.8, 1]])

    configs = [
        
        # example_cholesky
        
        dict(
            png_name='cholesky_corr_rets.png',
            model=sth.GeometricBrownianMotion(**params),
            func='create_corr_returns',
            cumulative=True,
            matrix=matrix
        ),
        dict(
            png_name='cholesky_generic_transform.png',
            model=sth.GeometricBrownianMotion(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        dict(
            png_name='cholesky_random_transform.png',
            model=sth.GeometricBrownianMotion(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        # example_ar
        
        dict(
            png_name='ar_random_transform.png',
            model=sth.AR(**params),
            func='transform',
            cumulative=True,
            matrix=None
        ),
        
        dict(
            png_name='ar_corr_transform.png',
            model=sth.AR(**params),
            func='transform',
            cumulative=True,
            matrix=matrix
        ),
        
        # example_cir
        
        dict(
            png_name='cir_random_transform.png',
            model=sth.CIR(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='cir_corr_transform.png',
            model=sth.CIR(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_gbm
        
        dict(
            png_name='gbm_random_transform.png',
            model=sth.GeometricBrownianMotion(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='gbm_corr_transform.png',
            model=sth.GeometricBrownianMotion(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_heston
        
        dict(
            png_name='heston_random_transform.png',
            model=sth.Heston(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='heston_corr_transform.png',
            model=sth.Heston(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_levy_stable
        
        dict(
            png_name='levy_stable_random_transform.png',
            model=sth.LevyStable(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='levy_stable_corr_transform.png',
            model=sth.LevyStable(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_mean_reverting
        
        dict(
            png_name='mean_reverting_random_transform.png',
            model=sth.MeanReverting(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='mean_reverting_corr_transform.png',
            model=sth.MeanReverting(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_merton
        
        dict(
            png_name='merton_random_transform.png',
            model=sth.Merton(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='merton_corr_transform.png',
            model=sth.Merton(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_narma
        
        dict(
            png_name='narma_random_transform.png',
            model=sth.NARMA(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='narma_corr_transform.png',
            model=sth.NARMA(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_poisson
        
        dict(
            png_name='poisson_random_transform.png',
            model=sth.Poisson(**params),
            func='transform',
            cumulative=False,
            matrix=None
        ),
        
        dict(
            png_name='poisson_corr_transform.png',
            model=sth.Poisson(**params),
            func='transform',
            cumulative=False,
            matrix=matrix
        ),
        
        # example_seasonal
        
        dict(
            png_name='seasonal_random_transform.png',
            model=sth.Seasonal(**params),
            func='transform',
            cumulative=True,
            matrix=None
        ),
        
        dict(
            png_name='seasonal_corr_transform.png',
            model=sth.Seasonal(**params),
            func='transform',
            cumulative=True,
            matrix=matrix
        ),
    ]

    for config in configs:
        save_to_png(**config)
