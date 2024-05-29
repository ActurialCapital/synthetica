import numpy as np
import synthetica as sth


if __name__ == "__main__":

    model = sth.GeometricBrownianMotion(length=252, num_paths=2, seed=123)
    matrix = np.array([[1, .8], [.8, 1]])
    
    # Correlated returns
    df1 = model.create_corr_returns(matrix)
    df1.corr()
    # ...
    np.cumprod(1 + df1).plot()
    
    # Generic transform
    
    df2 = model.transform(matrix)
    df2.corr()
    # ...
    df2.plot()
    
    # Random
    
    df3 = model.transform()
    df3.corr()
    # ...
    df3.plot()
