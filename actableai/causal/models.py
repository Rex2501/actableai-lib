import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from econml.dml import LinearDML
from econml.iv.nnet import DeepIV
from econml.metalearners import DomainAdaptationLearner
from hyperopt import hp
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from actableai.causal import autogluon_hyperparameters
from actableai.causal.predictors import SKLearnTabularWrapper
from actableai.regression import PolynomialLinearPredictor
from actableai.utils import random_directory
from actableai.utils.sklearn import sklearn_canonical_pipeline


class UntrainedModelException(ValueError):
    pass


class UnsupportedTargetUnitsMultipleTreatments(ValueError):
    pass


class UnsupportedTargetUnitsIVMethods(ValueError):
    pass


class AAICausalEstimator:
    def __init__(
        self,
        model_params: Optional[dict] = None,
        scorer: Optional[list] = None,
        has_categorical_treatment: bool = False,
        has_binary_outcome: bool = False,
    ):
        """Construct an AAICausalEstimator object

        Args:
            model_params: List of BaseCausalEstimatorParams objects. Defaults to None.
            scorer: _description_. Defaults to None.
            has_categorical_treatment: Whether the treatment is categorical.
                Defaults to False.
            has_binary_outcome: Whether the outcome is binary. Defaults to False.
        """
        self.model_params = (
            {params.MODEL_NAME: params for params in model_params}
            if model_params is not None
            else {}
        )
        self.scorer = scorer
        self.estimator = None
        self.tune_results_df = None
        self.is_multiple_treatments = False
        self.is_multiple_outcomes = False
        self.has_categorical_treatment = has_categorical_treatment
        self.has_binary_outcome = has_binary_outcome

    def fit(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray] = None,
        W: np.ndarray = None,
        Z: np.ndarray = None,
        label_t: str = "t",
        label_y: str = "y",
        target_units: str = "ate",
        trials: int = 3,
        tune_params=None,
        cv: Union[str, int] = "auto",
        feature_importance: bool = False,
        model_directory: Optional[str] = None,
        hyperparameters: Union[dict, str] = "auto",
        presets: str = "medium_quality_faster_train",
        random_state: Optional[int] = None,
        mc_iters: str = "auto",
        remove_outliers: bool = True,
        contamination: float = 0.05,
        num_gpus: int = 0,
        drop_unique: bool = True,
        drop_useless_features: bool = True,
    ):
        """Function fits a causal model with a single deterministic model.

        Args:
            Y: (n × d_y) matrix or vector of length n. Outcomes for each sample
            T: (n × d_t) matrix or vector of length n. Treatments for each sample
            X: (n × d_x) matrix. Defaults to None. Features for each sample
            W: (n × d_w) matrix. Defaults to None. Controls for each sample
            Z: (n × d_z) matrix. Defaults to None. Instruments for each sample
            label_t: Treatment for the causal inference
            label_y: Outcome for the causal inference
            target_units: either "ate", "att" or "atc"
            trials: number of trials for hyperparameter tuning experiment
            tune_params: dictionary of tune parameters
            max_concurrent: max concurcent
            scheduler: tune scheduler object
            stopper: tune stopper object
            cv: Number of cross validation fold
            feature_importance: Whether the feature importance are computed.
            model_directory: Directory to save AutoGluon model
                See
                https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit
            hyperparameters: Hyperparameters for AutoGluon model.
                See
                https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.fit
            presets: Presets for Autogluon Model.
            random_state: Random State for LinearDML. See
                https://econml.azurewebsites.net/_autosummary/econml.dml.LinearDML.html#econml.dml.LinearDML
            mc_iters: Random State for LinearDML. See
                https://econml.azurewebsites.net/_autosummary/econml.dml.LinearDML.html#econml.dml.LinearDML
            remove_outliers: Whether we remove outliers.
            contamination: Contamination parameter for removing outliers.
                See
                https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html.
            num_gpus: Number of GPUs to use
            drop_unique: Wether to drop columns with only unique values as preprocessing step.
            drop_useless_features: Whether to drop columns with only unique values at fit time.
        """
        start = time.time()

        automl_pipeline_feature_parameters = {}
        if X is None:
            drop_unique = False
        if not drop_useless_features:
            automl_pipeline_feature_parameters["pre_drop_useless"] = False
            automl_pipeline_feature_parameters["post_generators"] = []

        try:
            if T.shape[1] > 1:
                self.is_multiple_treatments = True
        except IndexError:
            pass
        try:
            if Y.shape[1] > 1:
                self.is_multiple_outcomes = True
        except IndexError:
            pass

        if hyperparameters == "auto":
            if Y.shape[0] <= 200:
                if mc_iters == "auto":
                    mc_iters = 10
                if cv == "auto":
                    cv = 10
                hyperparameters = {
                    "RF": {},
                    "LR": {"penalty": "l1"},
                    PolynomialLinearPredictor: [{"degree": 2}],
                }
            else:
                hyperparameters = autogluon_hyperparameters()

        if mc_iters == "auto":
            mc_iters = 5
        if cv == "auto":
            cv = 5
            if self.has_categorical_treatment or self.has_binary_outcome:
                for col in T.columns:
                    cv = min(T[col].value_counts().min(), cv)
                for col in Y.columns:
                    cv = min(Y[col].value_counts().min(), cv)

        xw_col = []
        if X is not None:
            xw_col += list(X.columns)
        if W is not None:
            xw_col += list(W.columns)

        model_t = TabularPredictor(
            path=random_directory(model_directory),
            label=label_t,
            problem_type="multiclass"
            if self.has_categorical_treatment
            else "regression",
        )
        model_t = SKLearnTabularWrapper(
            model_t,
            xw_col,
            hyperparameters=hyperparameters,
            presets=presets,
            ag_args_fit={"num_gpus": num_gpus, "drop_unique": drop_unique},
            feature_generator=AutoMLPipelineFeatureGenerator(
                **automl_pipeline_feature_parameters
            ),
        )
        model_y = TabularPredictor(
            path=random_directory(model_directory),
            label=label_y,
            problem_type="binary" if self.has_binary_outcome else "regression",
        )
        model_y = SKLearnTabularWrapper(
            model_y,
            xw_col,
            hyperparameters=hyperparameters,
            presets=presets,
            ag_args_fit={"num_gpus": num_gpus, "drop_unique": drop_unique},
            feature_generator=AutoMLPipelineFeatureGenerator(
                **automl_pipeline_feature_parameters
            ),
        )
        self.estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            linear_first_stages=False,
            cv=cv,
            random_state=random_state,
            mc_iters=mc_iters,
            discrete_treatment=self.has_categorical_treatment,
        )

        # Remove outliers
        if (
            remove_outliers
            and not self.has_categorical_treatment
            and not self.has_binary_outcome
        ):
            df_ = np.hstack([Y, T])
            if X is not None:
                df_ = np.hstack([df_, X])
            if W is not None:
                df_ = np.hstack([df_, W])
            df_ = pd.DataFrame(df_)
            outlier_clf = sklearn_canonical_pipeline(
                df_, IsolationForest(contamination=contamination)
            )
            is_inline = outlier_clf.fit_predict(df_) == 1
            Y, T = Y[is_inline], T[is_inline]
            if X is not None:
                X = X[is_inline]
            if W is not None:
                W = W[is_inline]

        self.estimator.fit(Y, T, X=X, W=W, cache_values=True)

        if feature_importance and (X is not None or W is not None):
            importances = []
            # Only run feature importance for first mc_iter to speed it up
            for _, m in enumerate(self.estimator.models_t[0]):
                importances.append(m.feature_importance())
            self.model_t_feature_importances = sum(importances) / cv
            self.model_t_feature_importances[
                "stderr"
            ] = self.model_t_feature_importances["stddev"] / np.sqrt(cv)
            self.model_t_feature_importances.sort_values(
                ["importance"], ascending=False, inplace=True
            )

            importances = []
            for _, m in enumerate(self.estimator.models_y[0]):
                importances.append(m.feature_importance())
            self.model_y_feature_importances = sum(importances) / cv
            self.model_y_feature_importances[
                "stderr"
            ] = self.model_y_feature_importances["stddev"] / np.sqrt(cv)
            self.model_y_feature_importances.sort_values(
                ["importance"], ascending=False, inplace=True
            )

        self.total_trial_time = time.time() - start

    def fit_search(
        self,
        Y,
        T,
        X=None,
        W=None,
        Z=None,
        target_units="ate",
        validation_ratio=0.2,
        trials=3,
        tune_params=None,
        max_concurrent=None,
        scheduler=None,
        stopper=None,
        cv=10,
    ):
        """Fit casaul model with model and hyper-parameter search.

        Args:
            Y (np.ndarray): (n × d_y) matrix or vector of length n. Outcomes for each sample
            T (np.ndarray): (n × d_t) matrix or vector of length n. Treatments for each sample
            X (np.ndarray, optional): (n × d_x) matrix. Defaults to None. Features for each sample
            W (np.ndarray, optional): (n × d_w) matrix. Defaults to None. Controls for each sample
            Z (np.ndarray, optional): (n × d_z) matrix. Defaults to None. Instruments for each sample
            target_units (str, optional): either "ate", "att" or "atc". Default to "ate"
            test_size (float): test size in percent. Defaults to 20
            trials (int, optional): number of trials for hyperparameter tuning experiment
            tune_params (dict, optional): dictionary of tune parameters
            max_concurrent (int, optional): max concurcent
            scheduler (object, optional): tune scheduler object
            stopper (object, optional): tune stopper object

        """
        # fit different causal estimators with train data
        id_train, id_val = train_test_split(
            np.arange(Y.shape[0]), test_size=validation_ratio
        )
        if X is not None:
            X_train, X_val = X[id_train], X[id_val]
        else:
            X_train, X_val = None, None
        if W is not None:
            W_train, W_val = W[id_train], W[id_val]
        else:
            W_train, W_val = None, None
        if Z is not None:
            Z_train, _ = Z[id_train], Z[id_val]
        else:
            Z_train, _ = None, None
        Y_train, Y_val = Y[id_train], Y[id_val]
        T_train, T_val = T[id_train], T[id_val]

        try:
            if T.shape[1] > 1:
                self.is_multiple_treatments = True
        except IndexError:
            pass
        try:
            if Y.shape[1] > 1:
                self.is_multiple_outcomes = True
        except IndexError:
            pass

        def causal_estimation_score(params):
            """Function calculates CATE estimation score on validation features X_val

            Args:
                X_val (np.ndarray): (n_val x d_x) matrix. Validation features
                params (BaseCausalEstimatorParams): causal estimator parameter object
            """
            estimator = self.model_params[params["name"]].build_estimator(params=params)
            estimator_type = type(estimator)
            if estimator_type in [DomainAdaptationLearner]:
                estimator.fit(Y_train, T_train, X=X_train)
            elif estimator_type in [DeepIV]:
                estimator.fit(Y_train, T_train, X=X_train, Z=Z_train)
            else:
                estimator.fit(Y_train, T_train, X=X_train, W=W_train)
            try:
                self.scorer.fit(Y_val, T_val, X=X_val, W=W_val)
                if estimator_type not in [DeepIV]:
                    score = self.scorer.score(estimator)
                else:
                    Y_pred = estimator.predict(T_val, X_val)
                    score = np.sqrt(mean_squared_error(Y_pred, Y_val))
            except ValueError:
                # RScorer is not supported when X is None
                score = 0
            return score

        def trainable(params):
            """Function that is a trainable for tune

            Args:
                models (list): List of BaseCausalEstimatorParams objects
            """
            score = causal_estimation_score(params["model"])
            tune.report(score=score)

        # construct the config dict for HyperOptSearch
        models = []
        for name, p in self.model_params.items():
            if p is not None:
                params = p.tune_config()
                params["name"] = p.MODEL_NAME
                models.append(params)
        config = {"model": hp.choice("model", models)}
        algo = HyperOptSearch(config, metric="score", mode="min")
        if max_concurrent is not None:
            algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)
        if tune_params is None:
            tune_params = {}

        analysis = tune.run(
            trainable,
            search_alg=algo,
            num_samples=trials,
            scheduler=scheduler,
            stop=stopper,
            **tune_params,
        )
        self.tune_results_df = analysis.results_df

        # get time of each trial
        time_total_s = 0
        for _, result in analysis.results.items():
            if result is not None and "time_total_s" in result:
                time_total_s += result["time_total_s"]

        start = time.time()

        best_config = analysis.get_best_config(metric="score", mode="max")
        temp_estimator = self.model_params[
            best_config["model"]["name"]
        ].build_estimator(params=best_config["model"])

        estimator_type = type(temp_estimator)
        if estimator_type in [DeepIV]:
            self.estimator = temp_estimator
            if target_units != "ate":
                raise UnsupportedTargetUnitsIVMethods()
            self.estimator.fit(Y, T, X=X, Z=Z)
        else:
            # DoWhy wrapper only support single treatment case
            if (
                (not self.is_multiple_treatments)
                and (X is not None)
                and (W is not None)
            ):
                self.estimator = temp_estimator.dowhy
                # refit with entire dataset
                self.estimator.fit(Y, T, X=X, W=W, target_units=target_units)
            else:
                self.estimator = temp_estimator
                if target_units != "ate":
                    raise UnsupportedTargetUnitsMultipleTreatments()
                self.estimator.fit(Y, T, X=X, W=W)

            _ = self.estimator.models_t

        self.total_trial_time = time.time() - start + time_total_s

    def effect(
        self,
        X: Optional[np.ndarray] = None,
        T0: Optional[np.ndarray] = None,
        T1: Optional[np.ndarray] = None,
        alpha: float = 0.05,
    ):
        """Compute heterogeneous treatment effect

        Args:
            X: (m, d_x) matrix. Features for each sample.
                Default to None.
            T0: (m, d_t) maxtrix or vector of length m.
                Base treatment for each sample. Default to None.
            T1: (m, d_t) maxtrix or vector of length m.
                Target treatment for each sample. Default to None.

        Returns: Tuple
            np.ndarray: (m, d_t) matrix. Estimated treatment effect.
            np.ndarray: Lower bound of the confidence interval.
            np.ndarray: Upper bound of the confidence interval.
        """
        if self.estimator is None:
            raise UntrainedModelException()

        # DeepIV inference not working, this is an open issue:
        # https://github.com/microsoft/EconML/issues/367
        lb, ub = None, None

        if (~self.is_multiple_treatments) and (~self.is_multiple_outcomes):
            if (T0 is not None) and (T1 is not None):
                effect = self.estimator.effect(X=X, T0=T0, T1=T1)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.effect_interval(
                        X=X, T0=T0, T1=T1, alpha=alpha
                    )
            else:
                effect = self.estimator.effect(X=X)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.effect_interval(X=X, alpha=alpha)
        else:
            if (T0 is not None) and (T1 is not None):
                effect = self.estimator.const_marginal_effect(X=X, T0=T0, T1=T1)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.const_marginal_effect_interval(
                        X=X, T0=T0, T1=T1, alpha=alpha
                    )
            else:
                effect = self.estimator.const_marginal_effect(X=X)
                if type(self.estimator) not in [DeepIV]:
                    lb, ub = self.estimator.const_marginal_effect_interval(
                        X=X, alpha=alpha
                    )

        return effect, lb, ub
