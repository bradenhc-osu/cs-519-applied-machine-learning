import numpy as np

def rmsle(ypred, yacc):
    assert len(ypred) == len(yacc)
    return np.sqrt(np.mean((np.log1p(ypred) - np.log1p(yacc)) ** 2))

def rmse(ypred, yacc):
    assert len(ypred) == len(yacc)
    return np.sqrt(np.mean(ypred - yacc) ** 2)

