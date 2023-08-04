from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from numpy import array


class RandomForestRegressorEnsemble:
    """
    Wrapper for multiple RandomForestRegressors, intended for use as an ensemble of RandomForestRegressors.
    Note that arguments to the RandomForestRegressor are passed to the function as kwargs.
    """

    def __init__(self, n_regressors=1, n_jobs=1, **params):
        self.n_jobs = n_jobs
        self.regressors = Parallel(n_jobs=self.n_jobs)(delayed(RandomForestRegressor)(**params)
                                                       for _ in range(n_regressors))

    def fit(self, X, y):
        self.regressors = Parallel(self.n_jobs)(delayed(r.fit)(X, y) for r in self.regressors)
        return self

    def get_params(self, deep=True):
        return self.regressors[0].get_params(deep)

    def partial_fit(self, X, y):
        return Parallel(self.n_jobs)(delayed(r.partial_fit)(X, y) for r in self.regressors)

    def predict(self, X):
        p = Parallel(self.n_jobs)(delayed(r.predict)(X) for r in self.regressors)
        return array(p)

    def score(self, X, y, sample_weight=None):
        return Parallel(self.n_jobs)(delayed(r.score)(X, y, sample_weight) for r in self.regressors)

    def set_params(self, **params):
        return Parallel(self.n_jobs)(delayed(r.set_params)(**params) for r in self.regressors)


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=200, random_state=1, n_features=2)  # note 100 features by default
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=10)

    regrs = RandomForestRegressorEnsemble(2, 100, max_iter=20).fit(X_train, y_train)
    p = regrs.predict(X_test)
    print(p)

