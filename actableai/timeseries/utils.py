import re
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


def interpolate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Interpolate missing values in time series.

    Args:
        df: Time series DataFrame.
        freq: Frequency to use when interpolating

    Returns:
        Interpolated new DataFrame
    """
    res = df.resample(freq).interpolate(method="linear")
    # In some cases, the above operation can return NaNs. If so, use an
    # alternate method. Note that the resultant values may differ from the
    # original ones
    if res.isnull().values.any():
        oidx = df.index
        nidx = pd.date_range(oidx.min(), oidx.max(), freq=freq)
        res = df.reindex(oidx.union(nidx)).interpolate("index").reindex(nidx)
    return res


def find_gluonts_freq(pd_date: pd.Series, freq: str) -> str:
    """Convert pandas frequency to GluonTS frequency.

    Args:
        pd_date: List of datetime as a pandas Series.
        freq: pandas frequency.

    Returns:
        GluonTS frequency.
    """
    if pd_date.dt.freq is None:
        if re.findall("\d*W", freq):
            return re.findall("\d*W", freq)[0]
        elif re.findall("\d*M", freq):
            return re.findall("\d*M", freq)[0]
        return freq

    return pd_date.dt.to_period().iloc[0].freqstr


def find_freq(pd_date: pd.Series, period: int = 10) -> Optional[str]:
    """Find the frequency from a list of datetime.
    Args:
        pd_date: List of datetime as a pandas Series, needs to be sorted.
        period: Window to look for when computing the frequency.
    Returns:
        The frequency or None if not found.
    """

    pd_date = pd.DatetimeIndex(pd_date)

    if len(pd_date) < 3:
        return None

    pd_date = pd_date.sort_values()
    freq = pd.infer_freq(pd_date)

    if freq:
        return freq

    infer_list = {}
    data_len = len(pd_date)
    for i in range(0, data_len - period, period):
        freq = pd.infer_freq(pd_date[i : i + period])
        if freq is not None:
            if freq not in infer_list:
                infer_list[str(freq)] = 0
            infer_list[str(freq)] += 1

    if len(infer_list) == 0:
        # Trick to find the freq when infer_freq is failing
        # https://stackoverflow.com/questions/68931854/pandas-infer-freq-returns-none
        # https://stackoverflow.com/questions/70771611/is-there-a-way-to-convert-timedelta-to-pandas-freq-string
        return to_offset(pd.to_timedelta(np.diff(pd_date).min())).freqstr

    most_freq = max(infer_list, key=lambda freq: infer_list[freq])
    return most_freq


def get_satisfied_formats(row: pd.Series, unique_formats: List[str]) -> List[str]:
    """Find the datetime formats compatible with a list of datetime.

    Args:
        row: List of datetime as a pandas Series.
        unique_formats: List of formats to try.

    Returns:
        List of compatible formats.
    """
    satisfied_formats = []
    for dt_format in unique_formats:
        try:
            pd.to_datetime(row, format=dt_format)
            satisfied_formats.append(dt_format)
        except:
            continue

    return satisfied_formats


def parse_datetime(dt_str: pd.Series, formats: List[str]) -> Optional[pd.Series]:
    """Try to parse datetime using a list of formats. Returns the first working format.

    Args:
        dt_str: List of datetime as pandas Series.
        formats: List of datetime formats to try.

    Returns:
        Parsed list of datetime or None if no formats are compatible.
    """
    for fm in formats:
        try:
            result = pd.to_datetime(dt_str, format=fm)
            return result
        except Exception:
            pass
    return None


def parse_by_format_with_valid_frequency(
    series: pd.Series, formats: List[str]
) -> pd.Series:
    """Parse datetime using a list of formats. Returns the first working format.

    Args:
        series: List of datetime as pandas Series.
        formats: List of datetime formats to try.

    Returns:
        Parsed list of datetime.
    """
    for fm in formats:
        try:
            result = pd.to_datetime(series.astype(str), format=fm)
            if pd.infer_freq(result):
                return result
        except Exception:
            pass
    return pd.to_datetime(series, format=formats[0])


def handle_datetime_column(
    series: pd.Series, min_parsed_rate: float = 0.5
) -> Tuple[pd.Series, str]:
    """Parse datetime from a list of datetime.

    Args:
        series: List of datetime to parse as a pandas Series.
        min_parsed_rate: Ratio of minimum parsed dates.

    Returns:
        - Parsed pandas series.
        - Data type of the parsed series ["datetime", "others"].
    """
    from pandas._libs.tslibs.parsing import guess_datetime_format

    series = series[series.notna()]

    if pd.api.types.is_datetime64_ns_dtype(series):
        return series, "datetime"

    parsed_rate_check = lambda x: x.isna().sum() >= min_parsed_rate * len(series)
    unique_formats = (
        pd.concat(
            [
                series.astype(str).apply(guess_datetime_format, dayfirst=False),
                series.astype(str).apply(guess_datetime_format, dayfirst=True),
            ]
        )
        .value_counts()
        .index.to_list()
    )

    if len(unique_formats) < 1:
        try:
            parsed_dt = pd.to_datetime(series, errors="coerce")
            if parsed_rate_check(parsed_dt):
                return series, "others"
            return parsed_dt, "datetime"
        except Exception:
            pass

        column_dtype = "others"
        return series, column_dtype
    else:
        column_dtype = "datetime"

    satisfied_formats = series.apply(
        get_satisfied_formats, unique_formats=unique_formats
    )
    unique_satisfied_formats_sorted = pd.Series(
        sum(satisfied_formats.values.tolist(), [])
    ).value_counts()
    satisfied_formats_sorted = satisfied_formats.apply(
        lambda x: sorted(x, key=unique_satisfied_formats_sorted.get, reverse=True)
    )

    if satisfied_formats_sorted.astype(str).nunique() == 1:
        parsed_dt = parse_by_format_with_valid_frequency(
            series, satisfied_formats_sorted.values[0]
        )
        column_dtype = "datetime"
    else:
        parsed_dt = pd.Series(
            [
                parse_datetime(series[i], satisfied_formats_sorted[i])
                for i in range(len(series))
            ]
        )

    if parsed_rate_check(parsed_dt):
        return series, "others"

    return parsed_dt, column_dtype
