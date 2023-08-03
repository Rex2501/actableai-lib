from enum import Enum, unique

from .association_rules import AAIAssociationRulesTask
from .bayesian_regression import AAIBayesianRegressionTask
from .causal_inference import AAICausalInferenceTask
from .classification import AAIClassificationTask
from .clustering import AAIClusteringTask
from .correlation import AAICorrelationTask
from .data_imputation import AAIDataImputationTask
from .forecast import AAIForecastTask
from .intervention import AAIInterventionTask
from .regression import AAIRegressionTask
from .sentiment_analysis import AAISentimentAnalysisTask
from .causal_discovery import AAICausalDiscoveryTask
from .ocr import AAIOCRTask
from .text_extraction import AAITextExtractionTask


@unique
class TaskType(str, Enum):
    """
    Enum representing the different tasks available
    """

    CAUSAL_INFERENCE = "causal_inference"
    DIRECT_CAUSAL_FEATURE_SELECTION = "direct_causal_feature_selection"
    CLASSIFICATION = "classification"
    CLASSIFICATION_TRAIN = "classification_train"
    CLUSTERING = "clustering"
    DEC_ANCHOR_CLUSTERING = "dec_anchor_clustering"
    CORRELATION = "correlation"
    DATA_IMPUTATION = "data_imputation"
    FORECAST = "forecast"
    REGRESSION = "regression"
    REGRESSION_TRAIN = "regression_train"
    BAYESIAN_REGRESSION = "bayesian_regression"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    INTERVENTION = "intervention"
    ASSOCIATION_RULES = "association_rules"
    CAUSAL_DISCOVERY = "causal_discovery"
    OCR = "ocr"
    TEXT_EXTRACTION = "text_extraction"
