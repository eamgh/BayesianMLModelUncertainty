from scipy.stats import norm, uniform
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from random import randint
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod
from typing import List
from joblib import Parallel, delayed
from uncertainty_analysis.bootstrap import bayesian_bootstrap
import numpy as np


class PBFPrior(ABC):
    """Parent class for prior distribution creation."""

    @abstractmethod
    def __init__(self, values: np.array = None):
        pass

    def sample_from_prior(self):
        return self.dist.rvs(1)


class NormalPrior(PBFPrior):
    def __init__(self, values: np.array = None, mean=None, std=None):
        if mean is not None and std is not None:
            self.dist = norm(mean, std)
        else:
            mean, std = norm.fit(values)
            self.dist = norm(mean, std)


class UniformPrior(PBFPrior):
    def __init__(self, values: np.array = None, min=None, max=None):
        if min is not None and max is not None:
            self.dist = uniform(loc=min, scale=max - min)
        else:
            min = np.min(values)
            self.dist = uniform(loc=min, scale=np.max(values) - min)


class ProperBayesianForest:

    def __init__(self, X: np.array, y: np.array, f0s: List[PBFPrior], k_values: np.array, n_galvani_samples: int,
                 k_nearest: int = 5, verbose: bool = False, **kwargs):
        """
        a new vector of covariates x is generated from the prior distributions F0 defined for the covariates, then
        the new value of the response variable is associated on the basis of the prior distribution chosen to
        model the relation between the target variable and the covariates.

        :param X: array of shape (n_samples, n_features)
        :param y: array of shape (n_samples) for target values
        :param f0s: list of prior distributions for the features
        :param k_values: array of confidence parameters for the priors (f0s)
        :param n_galvani_samples: int of number of bootstrap samples to create when generating the forest
        :param k_nearest: int of the k nearest to use to get the target value for each bootstrapped vector of covariates
        :param verbose: boolean value of additional information output
        :param **kwargs: args used to create each RandomForestRegressor during model fitting
        """

        self.X = X
        self.y = y
        self.f0s = f0s
        self.k_values = k_values
        self.n_galvani_samples = n_galvani_samples
        self.k_nearest = k_nearest
        self.verbose = verbose

        # check data is in a valid form
        self.check_input_validity()

        # create random forest
        self.proper_bayesian_forest = RandomForestRegressor(**kwargs)

    def check_input_validity(self):
        if len(self.f0s) != self.X.shape[1]:
            raise ValueError(f"Unequal number of prior distributions specified: {self.f0s.shape[0]} priors " +
                             f"to {self.X.shape[1]} features")

        if len(self.k_values) != self.X.shape[1]:
            raise ValueError(f"Unequal number of k values specified: {self.f0s.shape[0]} to " +
                             f"{self.X.shape[1]} features")

    def fit(self):
        """"Create bootstrapped dataset and use them to create the forest estimators"""

        b_X, b_y = self.bootstrap_dataset(self.k_nearest)
        self.proper_bayesian_forest.fit(b_X, b_y)

    def bootstrap_dataset(self, k_nearest: int):
        """
        Create new samples using the extension of Rubin's bootstrap.

        :param k_nearest: number of k nearest neighbours to use to create new target values for bootstrapped samples
        :return (bootstrapped_X, bootstrapped_Y): arrays of shape (n_samples, n_features) and (n_samples) respectively,
        containing new training set & target values
        """

        # arrays to hold new samples
        bootstrapped_X = np.empty((self.n_galvani_samples, self.X.shape[1]))
        bootstrapped_y = np.empty((self.n_galvani_samples))

        # KNN model to get new target values for the new covariate vectors generated
        knn = KNeighborsRegressor(n_neighbors=k_nearest, weights='distance').fit(self.X, self.y)

        for i in range(0, self.n_galvani_samples):
            # get new vector of covariates using prior distribution
            prior_covariates = np.ndarray.flatten(np.array([p.sample_from_prior() for p in self.f0s]))

            # draw a sample of an existing covariate vector
            n = self.X.shape[0]
            existing_covariates = self.X[randint(0, n - 1)]

            # calculate (k+n)^-1(k*F0 + n*Fn): combines the prior and the existing vector of covariates
            for j in range(0, len(prior_covariates)):
                bootstrapped_X[i][j] = ((self.k_values[j] * prior_covariates[j]) + (n * existing_covariates[j])) / (
                        self.k_values[j] + n)

            # calculate new y value using KNN model
            bootstrapped_y[i] = knn.predict(bootstrapped_X[i].reshape(1, -1))

        return bootstrapped_X, bootstrapped_y

    def predict(self, X: np.array, n_jobs: int = None, n_rubin_samples: int = 100):
        """
        Get point estimate prediction of the forest.

        :param X: array of shape (n_samples, n_features) used to make predictions
        :param n_jobs: number of jobs to run in parallel when making predictions
        :param n_rubin_samples: number of samples to use in the Bayesian bootstrap after having made predictions

        :return: point estimate of forest prediction
        """
        bootstrapped_predictions = self.get_bootstrapped_predictions(X, n_jobs, n_rubin_samples)

        return np.mean(bootstrapped_predictions, axis=1)

    def get_prediction_distribution(self, X, n_jobs=None, n_rubin_samples=100):
        """
        Get the prediction distribution of the forest (after Bayesian bootstrap applied).

        :param X: array of shape (n_samples, n_features) used to make predictions
        :param n_jobs: number of jobs to run in parallel when making predictions
        :param n_rubin_samples: number of samples to use in the Bayesian bootstrap after having made predictions

        :return result: array of bootstrapped predicitions
        """

        predictions = self.get_raw_predictions(X, n_jobs)

        result = np.empty((predictions.shape[0], n_rubin_samples), dtype=np.float64)
        for i, p_set in enumerate(predictions):
            result[i] = bayesian_bootstrap(p_set, n_rubin_samples)

        return result

    def get_raw_predictions(self, X, n_jobs=None):
        """
        Get the predictions of the forest (before Bayesian bootstrap applied).

        :param X: array of shape (n_samples, n_features) used to make predictions
        :param n_jobs: number of jobs to run in parallel when making predictions
        :param n_rubin_samples: number of samples to use in the Bayesian bootstrap after having made predictions

        :return result: array of predictions of each tree in forest
        """

        predictions = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(tree.predict)(X) for tree in self.proper_bayesian_forest.estimators_)

        predictions = np.transpose(predictions)

        return predictions


def get_k(w, n):
    """Create k for a given weight (refer to Marta Galvani et al. 'A Bayesian nonparametric learning approach to
    ensemble models using the proper Bayesian bootstrap'. In: Algorithms 14.1 (2021), p. 11."""

    return (w * n) / (1 - w)


def hist(data: np.array):
    df = np.transpose(data)
    n_features = df.shape[0]
    for i in range(n_features):
        plt.subplot(int(n_features // 5) + 1, 5, i + 1)
        plt.title(f"Feature: {i}")
        plt.tight_layout()
        plt.hist(df[i])
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    from sklearn.model_selection import train_test_split

    tx, vx, ty, vy = train_test_split(data, target, test_size=0.3, random_state=0)
    from uncertainty_analysis import proper_bayesian_forest as pbf

    l = tx.shape[1]
    priors = np.empty(l, dtype=pbf.UniformPrior)
    for i in range(0, l):
        priors[i] = pbf.UniformPrior(tx[:, i])

    pbrgr = pbf.ProperBayesianForest(tx, ty,
                                     priors,
                                     k_values=np.array([200] * 13, dtype=np.int32),
                                     n_samples=10).fit()
    pbrgr.fit()

    print(pbrgr.get_bootstrapped_predictions(vx[0].reshape(1, -1)))
