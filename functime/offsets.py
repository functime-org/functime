from typing import List, Tuple, Union

OFFSET_ALIASES = {"s", "m", "h", "d", "w", "mo", "y", "i"}


def _strip_freq_alias(freq: str) -> Tuple[int, str]:
    """Return (index count, offset string) given Polars offset alias.

    For example, `freq = "3mo"` returns `(3, "mo")`.
    """
    freq = freq.lower()
    for x in OFFSET_ALIASES:
        if freq.endswith(x):
            offset_n = int(freq.rstrip(x))
            offset_alias = x
            return offset_n, offset_alias


def freq_to_sp(freq: str) -> Union[List[int], List[float]]:
    """Return seasonal periods given offset alias.

    Reference: https://robjhyndman.com/hyndsight/seasonal-periods/

    Parameters
    ----------
    freq : str
        Supported offset aliases:\n
        - 1s (1 second)
        - 1m (1 minute)
        - 30m (30 minute)
        - 1h (1 hour)
        - 1d (1 day)
        - 1w (1 week)
        - 1mo (1 calendar month)
        - 3mo (1 calendar quarter)
        - 1y (1 calendar year)

    Returns
    -------
    sp : list of int
    """

    seasonal_periods = {
        "1s": [60, 3_600, 86_400, 604_800, 31_557_600],
        "1m": [60, 1_440, 10_080, 525_960],
        "30m": [48, 336, 17_532],
        "1h": [24, 168, 8_766],
        "1d": [7, 365],
        "1w": [52],
        "1mo": [12],
        "3mo": [4],
        "1y": [1],
    }

    try:
        sp = seasonal_periods[freq]
    except KeyError as exc:
        raise ValueError(f"Offset {freq!r} not supported") from exc

    return sp
