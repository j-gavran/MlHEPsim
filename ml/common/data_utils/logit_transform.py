import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogitTransform(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1e-6, validate=True):
        """Transforms the input to the logit space.

        Parameters
        ----------
        alpha: float
            Small constant that is used to scale the original input, by default 1e-6.
        validate: bool
            If True, input values are checked to be in the range [0, 1], by default True.

        Note
        ----
        Input values must be in the range [0, 1].

        References
        ----------
        [1] - https://en.wikipedia.org/wiki/Logit#Definition

        """
        super().__init__()
        self.alpha = alpha
        self.validate = validate

    def validate_data(self, X):
        if self.validate:
            test = (X + self.alpha >= 0) & (X - self.alpha <= 1)
            assert test.all(), "values must be in range [0, 1]"

    def logistic_transform(self, x):
        x = (x - 0.5 * self.alpha) / (1 - self.alpha)
        return 1 / (1 + np.exp(-x))

    def logit_transform(self, x):
        x = x * (1 - self.alpha) + 0.5 * self.alpha
        return np.log(x / (1 - x))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        self.validate_data(X)
        return self.logit_transform(X)

    def inverse_transform(self, X, y=None):
        return self.logistic_transform(X)
