from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def emd(X, Y):
    d = cdist(X, Y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(X), len(Y))