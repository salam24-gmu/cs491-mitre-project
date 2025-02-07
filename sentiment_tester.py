import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_correlation_test(x: np.array, y: np.array, alpha=0.05) -> bool:
    """
    Checks whether the observed correlation in an (X, Y) scatterplot is statistically significant.
    """
    r, p_value = stats.pearsonr(x, y)
    return p_value < alpha