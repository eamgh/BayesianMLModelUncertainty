from scipy.stats import dirichlet
from sklearn.utils import resample
import numpy as np


def bayesian_bootstrap(samples: np.array, num_samples: int):
    """
    Returns Bayesian bootstrapped (Rubin's) samples.

    :param samples: np array of original samples. Must be 1d.
    :param num_samples: number of dirichlet samples to draw
    """

    dirichlet_samples = dirichlet([1] * len(samples)).rvs(num_samples)
    bootstrapped_samples = (samples * dirichlet_samples).sum(axis=1)

    return bootstrapped_samples


def classic_bootstrap(samples, num_samples):
    """
    Returns bootstrapped samples.

    :param samples: np array of samples. Must be 1d.
    :param num_samples: number of samples to create
    """

    return [resample(samples).mean() for _ in range(num_samples)]


if __name__ == "__main__":
    test_sample = np.array([1.865, 3.053, 1.401, 0.569, 4.132])
    q = bayesian_bootstrap(test_sample, 100) (5,)
    print(q)