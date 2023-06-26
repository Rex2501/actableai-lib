import numpy as np
from autogluon.core import space
from autogluon.core.models.abstract.abstract_model import AbstractModel
from autogluon.core.constants import QUANTILE

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer


def ag_quantile_hyperparameters():
    """Returns a dictionary of Quantile Regressor Model for AutoGluon hyperparameters.

    Returns:
        dictionnary: Models for AutoGluon hyperparameters.
    """
    return {
        GradientBoostQuantileRegressor: {
            "max_depth": space.Int(3, 32),
        },
    }


class GradientBoostQuantileRegressor(AbstractModel):
    def _get_model_type(self):
        if self.problem_type == QUANTILE:
            return GradientBoostQuantileRegressor
        else:
            raise ValueError(
                f"GradientBoostQuantileRegressor does not support problem_type={self.problem_type}"
            )

    def __init__(self, quantile_levels: list, **kwargs):
        super().__init__(**kwargs)
        self.quantile_levels = quantile_levels
        for level in quantile_levels:
            self.model["%.2f" % level] = GradientBoostingRegressor(
                loss="quantile",
                alpha=level / 100.0,
                **kwargs,
            )

    def _fit(self, X, y, **kwargs):
        X = self.preprocess(X, is_train=True)
        for m in self.model.values():
            m.fit(X, y, **kwargs)
        return self.model

    def _predict_proba(self, X, **kwargs):
        y_pred = []
        for m in self.model.values():
            y_pred.append(m.predict(X, **kwargs))
        return np.asarray(y_pred).T

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        self.na_imputer = SimpleImputer(strategy="median")
        X = self.na_imputer.fit_transform(X)
        return X
