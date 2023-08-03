from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import pandas as pd
from pydantic import BaseModel, root_validator

from actableai.utils import get_type_special_no_ag


class Constraints(BaseModel):
    causes: List[str]
    effects: List[str]
    forbiddenRelationships: List[Tuple[str, str]]
    potentialRelationships: Optional[List[Tuple[str, str]]] = None


class Dataset(BaseModel):
    data: Dict[str, List[Any]]


class CausalVariableNature(str, Enum):
    Discrete = "Discrete"
    Continuous = "Continuous"
    CategoricalOrdinal = "Categorical Ordinal"
    CategoricalNominal = "Categorical Nominal"
    Binary = "Binary"
    Excluded = "Excluded"


class CausalVariable(BaseModel):
    name: str
    nature: Optional[CausalVariableNature] = None

    @classmethod
    def from_dataset(cls, column: str, dataset: Dataset) -> "CausalVariable":
        column_data = pd.Series(dataset.data[column])
        column_type = get_type_special_no_ag(column_data)
        unique_values = column_data.nunique()

        nature = None
        if column_type == "integer":
            if unique_values <= 10:
                nature = CausalVariableNature.Discrete
            else:
                nature = CausalVariableNature.Continuous
        elif column_type == "numeric":
            nature = CausalVariableNature.Continuous
        else:
            if unique_values == 2:
                nature = CausalVariableNature.Binary
            else:
                nature = CausalVariableNature.CategoricalNominal

        return cls(
            name=column,
            nature=nature,
        )


class NormalizationOptions(BaseModel):
    with_mean: bool = True
    with_std: bool = True


class CausalDiscoveryPayload(BaseModel):
    dataset: Dataset
    normalization: NormalizationOptions = NormalizationOptions()
    constraints: Constraints
    causal_variables: List[CausalVariable] = []

    class Config:
        arbitrary_types_allowed = True

    @root_validator(allow_reuse=True)
    def set_causal_variables(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        causal_variables = values["causal_variables"]
        set_variables = {causal_variable.name for causal_variable in causal_variables}

        for column in set(values["dataset"].data.keys()).difference(set_variables):
            causal_variables.append(
                CausalVariable.from_dataset(
                    column=column,
                    dataset=values["dataset"],
                )
            )

        values["causal_variables"] = causal_variables
        return values


class ATEDetails(BaseModel):
    reference: Optional[Union[float, str]] = None
    intervention: Optional[Union[float, str]] = None
    nature: Optional[CausalVariableNature] = None


# TODO: UPDATE DEFAULTS
class DeciModelOptions(BaseModel):
    base_distribution_type: Literal["gaussian", "spline"] = "spline"
    spline_bins: int = 8
    imputation: bool = False
    lambda_dag: float = 100.0
    lambda_sparse: float = 5.0
    tau_gumbel: float = 1.0
    var_dist_A_mode: Literal["simple", "enco", "true", "three"] = "three"
    imputer_layer_sizes: Optional[List[int]] = None
    mode_adjacency: Literal["upper", "lower", "learn"] = "learn"
    norm_layers: bool = True
    res_connection: bool = True
    encoder_layer_sizes: Optional[List[int]] = [32, 32]
    decoder_layer_sizes: Optional[List[int]] = [32, 32]
    cate_rff_n_features: int = 3000
    cate_rff_lengthscale: Union[int, float, List[float], Tuple[float, float]] = 1


# To speed up training you can try:
#  increasing learning_rate
#  increasing batch_size (reduces noise when using higher learning rate)
#  decreasing max_steps_auglag (go as low as you can and still get a DAG)
#  decreasing max_auglag_inner_epochs
# TODO: UPDATE DEFAULTS
class DeciTrainingOptions(BaseModel):
    learning_rate: float = 3e-2
    batch_size: int = 512
    standardize_data_mean: bool = False
    standardize_data_std: bool = False
    rho: float = 10.0
    safety_rho: float = 1e13
    alpha: float = 0.0
    safety_alpha: float = 1e13
    tol_dag: float = 1e-3
    progress_rate: float = 0.25
    max_steps_auglag: int = 20
    max_auglag_inner_epochs: int = 1000
    max_p_train_dropout: float = 0.25
    reconstruction_loss_factor: float = 1.0
    anneal_entropy: Literal["linear", "noanneal"] = "noanneal"


class DeciAteOptions(BaseModel):
    Ngraphs: Optional[int] = 1
    Nsamples_per_graph: Optional[int] = 5000
    most_likely_graph: Optional[int] = True
    ate_details_by_name: Dict[str, ATEDetails] = dict()


class DeciPayload(CausalDiscoveryPayload):
    model_options: DeciModelOptions = DeciModelOptions()
    training_options: DeciTrainingOptions = DeciTrainingOptions()
    ate_options: DeciAteOptions = DeciAteOptions()
    model_save_dir: str = None


class DirectLiNGAMPayload(CausalDiscoveryPayload):
    pass


class NotearsPayload(CausalDiscoveryPayload):
    max_iter: int = 100


class PCPayload(CausalDiscoveryPayload):
    variant: Literal["original", "stable"] = "original"
    alpha: float = 0.05
    ci_test: Literal["gauss", "g2", "chi2"] = "gauss"
