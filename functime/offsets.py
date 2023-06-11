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


def freq_to_sp(freq: str, include_dec: bool = False) -> Union[List[int], List[float]]:
    """Return seasonal periods given offset alias.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.

        The offset is dictated by the following string language:\n
        - 1ns (1 nanosecond)
        - 1us (1 microsecond)
        - 1ms (1 millisecond)
        - 1s (1 second)
        - 1m (1 minute)
        - 1h (1 hour)
        - 1d (1 day)
        - 1w (1 week)
        - 1mo (1 calendar month)
        - 1q (1 calendar quarter)
        - 1y (1 calendar year)
        - 1i (1 index count)
    include_dec : bool
        If True, return floating point seasonal periods.
        Otherwise, all seasonal periods are rounded down
        to the nearest integer.

    Returns
    -------
    sp : list of int, list of float
    """

    # Seasonal periods for given frequency as suggested by:
    # https://robjhyndman.com/hyndsight/seasonal-periods/

    alias_to_sp = {
        "s": [7.0, 365.25],
        "m": [48.0, 336.0, 17532.0],
        "30m": [60.0, 1440.0, 10080.0, 525960.0],
        "h": [24.0, 168.0, 8766.0],
        "d": [7.0, 365.25],
        # 365.25/7 = 52.18 on average, allowing
        # for a leap year every fourth year
        "w": [52.18],
        "y": [1.0],
        "mo": [12.0],
        "3mo": [4.0],
    }
    n, alias = _strip_freq_alias(freq)
    alias = (
        f"{n}{alias}"
        if ((alias.endswith("m") and n == 30) or (alias.endswith("mo") and n == 3))
        else alias
    )

    try:
        sp = alias_to_sp[alias]
    except KeyError as exc:
        raise ValueError(f"Offset {freq!r} not supported") from exc

    if not include_dec:
        sp = list(map(int, sp))

    return sp
