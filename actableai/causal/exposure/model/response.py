#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project.
#

from typing import List, Optional, Union

from pydantic import BaseModel

from actableai.causal.exposure.model.confidence_interval_models import ConfidenceIntervalResult
from actableai.causal.exposure.model.estimate_effect_models import EstimateResult
from actableai.causal.exposure.model.refute_estimate_models import RefuterResult
from actableai.causal.exposure.model.shap_interpreter_models import ListShapInterpreterResult
from actableai.causal.exposure.model.significance_test_models import SignificanceTestResult


class StatusModel(BaseModel):
    status: str
    completed: int
    pending: int
    failed: int
    results: Optional[
        Union[
            List[
                Union[
                    RefuterResult,
                    EstimateResult,
                    ConfidenceIntervalResult,
                    ListShapInterpreterResult,
                ]
            ],
            SignificanceTestResult,
        ]
    ]

    def to_dict(self):
        return {
            "status": self.status,
            "completed": self.completed,
            "pending": self.pending,
            "failed": self.failed,
            "results": [result.to_dict() for result in self.results]
            if isinstance(self.results, list)
            else self.results.to_dict(),
        }


class NumberOfExecutionsResult(BaseModel):
    count: int
