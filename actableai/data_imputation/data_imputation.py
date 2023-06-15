def data_imputation(df, rules="", impute_nulls=True):
    from actableai.data_imputation.error_detector.null_detector import NullDetector
    from actableai.data_imputation.error_detector.validation_detector import (
        ValidationDetector,
    )
    import time
    import pandas as pd
    import numpy as np
    from actableai.data_imputation.data.data_frame import DataFrame

    start = time.time()

    detectors = [ValidationDetector.from_constraints(rules)]
    if impute_nulls:
        detectors.append(NullDetector())

    raw_df = DataFrame(df.copy())
    errors = raw_df.detect_error(*detectors)
    if not errors:
        return {
            "messenger": "No errors detected.",
            "status": "SUCCESS",
            "runtime": time.time() - start,
            "data": {},
            "validation": [],
        }

    def diff(x):
        raw_v = x[0]
        fixed_v = x[1]
        if (
            str(raw_v) == str(fixed_v)
            or (
                isinstance(raw_v, int)
                and isinstance(fixed_v, float)
                and raw_v == fixed_v
            )
            or (isinstance(raw_v, float) and np.isnan(raw_v) and fixed_v == "_nan_")
        ):
            return ""
        else:
            return "highlight"

    fixed_df = raw_df.auto_fix(errors, *detectors)
    df_highlight = pd.concat(
        [raw_df, fixed_df], axis="columns", keys=["raw_df", "fixed_df"], join="outer"
    )
    df_highlight = df_highlight.swaplevel(axis="columns")[raw_df.columns[1:]]
    df_highlight = df_highlight.groupby(level=0, axis=1).apply(
        lambda frame: frame.apply(diff, axis=1)
    )

    records = []
    for idx, (o, t, c) in enumerate(
        zip(
            raw_df.to_dict("record"),
            fixed_df.to_dict("record"),
            df_highlight.to_dict("record"),
        )
    ):
        records.append({"oldtext": o, "text": t, "class": c, "index": idx})

    data = {"columns": raw_df.columns.tolist(), "records": records}
    runtime = time.time() - start
    return {
        "status": "SUCCESS",
        "messenger": "",
        "data": data,
        "runtime": runtime,
        "validations": [],
    }
