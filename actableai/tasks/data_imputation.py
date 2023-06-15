from typing import Dict, List, Tuple

from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


# We cannot use RawRules directly
def construct_rules(data):
    def construct_validation_rules(rows) -> List[str]:
        rules = []
        if len(rows) < 1:
            return rules

        for row in rows:
            when = " & ".join(
                [
                    f"{when['column']['value']}{when['operator']['value']}{when['comparedColumn']['value']}"
                    for when in row["when"]
                ]
            )
            then = " & ".join(
                [
                    f"{when['column']['value']}{when['operator']['value']}{when['comparedColumn']['value']}"
                    for when in row["then"]
                ]
            )
            if not when or not then:
                continue
            else:
                rules.append(f"{when} -> {then}")

        return rules

    def construct_misplaced_rules(rows) -> List[str]:
        rules = []
        if len(rows) < 1:
            return rules

        for row in rows:
            if row["isRegex"]:
                rules.append(
                    f'{row["column"]["value"]}{row["operator"]["value"]}s/{row["value"]}/g'
                )
            else:
                rules.append(
                    f'{row["column"]["value"]}{row["operator"]["value"]}{row["value"]}'
                )
        return rules

    if data is None:
        return "", ""

    validation_rules = []
    misplaced_rules = []
    for i in range(len(data)):
        validation_rows = data[i]["validations"]
        misplaced_rows = data[i]["misplaced"]

        validation_rules.extend(construct_validation_rules(validation_rows))
        misplaced_rules.extend(construct_misplaced_rules(misplaced_rows))

    return " OR ".join(validation_rules), " OR ".join(misplaced_rules)


class AAIDataImputationTask(AAITask):
    @AAITask.run_with_ray_remote(TaskType.DATA_IMPUTATION)
    def run(
        self,
        df,
        rules: Tuple[str, str] = ("", ""),
        impute_nulls: bool = True,
        override_column_types: Dict = {},
    ) -> Dict:
        """Impute the DataFrame df

        Args:
            df: DataFrame to impute
            rules: Set of rules for imputation. Defaults to ("", "").
            impute_nulls: Whether null, None and nan values should be imputed.
                Defaults to True.
            override_column_types: Columns overriden by a special type . Defaults to {}.

        Examples:
            >>> df = pd.read_csv("path/to/dataframe")
            >>> result = AAIDataImputationTask().run(df)
            >>> result

        Returns:
            Dict: Dictionnary of results
                - "status": "SUCCESS" if the task successfully ran else "FAILURE"
                - "messenger": Message returned with the task
                - "validations": List of validations on the data.
                    non-empty if the data presents a problem for the task
                - "runtime": Execution time of the task
                - "data": Dictionnary containing the data for the task
                    - "columns": Columns of the new table
                    - "records": Records containing the new values after imputation
        """
        from actableai.data_imputation.error_detector import (
            NullDetector,
            ValidationDetector,
            MisplacedDetector,
        )
        import time
        import pandas as pd
        from actableai.data_imputation.data.data_frame import DataFrame
        from actableai.data_imputation.error_detector.rule_parser import (
            RulesBuilder,
        )
        from actableai.data_imputation.error_detector.rule_parser import RulesRaw
        from actableai.data_validation.params import DataImputationDataValidator
        from actableai.data_validation.base import CheckLevels
        from actableai.utils.sanitize import sanitize_timezone

        pd.set_option("chained_assignment", "warn")
        start = time.time()

        # To resolve any issues of acces rights make a copy
        df = df.copy()
        df = sanitize_timezone(df)

        data_validation_results = DataImputationDataValidator().validate(df)
        failed_checks = [x for x in data_validation_results if x is not None]
        if CheckLevels.CRITICAL in [x.level for x in failed_checks]:
            return {
                "status": "FAILURE",
                "data": {},
                "validations": [
                    {"name": x.name, "level": x.level, "message": x.message}
                    for x in failed_checks
                ],
                "runtime": time.time() - start,
            }

        raw_df = DataFrame(df.copy())
        custom_rules = RulesBuilder.parse(raw_df.column_types, RulesRaw(*rules))
        detectors = [
            ValidationDetector(custom_rules.constraints),
            MisplacedDetector(customize_rules=custom_rules.match_rules),
        ]
        if impute_nulls:
            detectors.append(NullDetector())

        raw_df = DataFrame(df.copy())
        for column, dtype in override_column_types.items():
            raw_df.override_column_type(column, dtype)

        errors = raw_df.detect_error(*detectors)
        if not errors:
            return {
                "messenger": "No errors detected",
                "status": "SUCCESS",
                "runtime": time.time() - start,
                "data": {},
                "validations": [
                    {
                        "name": "NoErrorDetected",
                        "level": CheckLevels.WARNING,
                        "message": "No errors detected",
                    }
                ],
            }

        def diff(x):
            raw_v = x[0]
            fixed_v = x[1]
            if (
                str(raw_v) == str(fixed_v)
                or (
                    isinstance(raw_v, int)
                    and isinstance(fixed_v, str)
                    and raw_v == fixed_v
                )
                or (isinstance(raw_v, float) and pd.isna(raw_v) and fixed_v == "_nan_")
                or (
                    pd.notna(raw_v)
                    and (
                        isinstance(raw_v, float)
                        and (isinstance(fixed_v, str))
                        and int(raw_v) == int(fixed_v)
                    )
                )
            ):
                return ""
            else:
                return "highlight"

        fixed_df = raw_df.auto_fix(errors, *detectors)
        df_highlight = pd.concat(
            [df, fixed_df],
            axis="columns",
            keys=["raw_df", "fixed_df"],
            join="outer",
        )
        df_highlight = df_highlight.swaplevel(axis="columns")[df.columns]
        df_highlight = df_highlight.groupby(level=0, axis=1).apply(
            lambda frame: frame.apply(diff, axis=1)
        )

        records = []
        date_columns = fixed_df.select_dtypes(include=["datetime"]).columns.tolist()
        df[date_columns] = df[date_columns].astype(str)
        fixed_df[date_columns] = fixed_df[date_columns].astype(str)
        for idx, (o, t, c) in enumerate(
            zip(
                df.to_dict("record"),
                fixed_df.to_dict("record"),
                df_highlight.to_dict("record"),
            )
        ):
            records.append({"oldtext": o, "text": t, "class": c, "index": idx})

        data = {"columns": list(df.columns), "records": records}
        runtime = time.time() - start
        return {
            "status": "SUCCESS",
            "messenger": "",
            "data": data,
            "runtime": runtime,
            "validations": [
                {"name": x.name, "level": x.level, "message": x.message}
                for x in failed_checks
            ],
        }
