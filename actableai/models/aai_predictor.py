from typing import Dict, List, Optional
from autogluon.tabular import TabularPredictor
from econml.dml import DML
import pandas as pd

from actableai.intervention import custom_intervention_effect


class AAIModel:
    model_version = 1

    def __init__(self, version: int = model_version) -> None:
        self.version = version


class AAITabularModel(AAIModel):
    def __init__(
        self,
        version: int,
        predictor: TabularPredictor,
    ) -> None:
        super().__init__(version)
        self.predictor = predictor


class AAITabularModelInterventional(AAITabularModel):
    def __init__(
        self,
        version: int,
        predictor: TabularPredictor,
        causal_model: DML,
        intervened_column: str,
        common_causes: Optional[List[str]],
        discrete_treatment: Optional[bool],
    ) -> None:
        super().__init__(
            version,
            predictor,
        )
        self.causal_model = causal_model
        self.intervened_column = intervened_column
        self.common_causes = common_causes
        self.discrete_treatment = discrete_treatment

    def intervention_effect(self, df: pd.DataFrame, pred: Dict) -> pd.DataFrame:
        return custom_intervention_effect(
            df=df,
            pred=pred,
            causal_model=self.causal_model,
            intervened_column=self.intervened_column,
            common_causes=self.common_causes,
            predictor=self.predictor,
            discrete_treatment=self.discrete_treatment,
        )