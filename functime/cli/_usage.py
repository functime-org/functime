import httpx
from rich.console import Console
from rich.table import Table
from typing import Union

from functime.config import API_CALL_TIMEOUT, FUNCTIME_SERVER_URL
from functime.io.auth import require_token


@require_token
def _get_usage_response(token, **params):
    with httpx.Client(http2=True) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = client.get(
            FUNCTIME_SERVER_URL + "/usage",
            headers=headers,
            params=params,
            timeout=API_CALL_TIMEOUT,
        )
        response.raise_for_status()
    return response.json()


def usage_cli():
    res = _get_usage_response()
    """
   response = {
        "Data Used": {
            "unit": "MB",
            "used": usage["mb_used"],
            "limit": mb_limit,
        },
        "Forecasts Used": {
            "unit": "preds",
            "used": usage["forecasts_used"],
            "limit": forecast_limit,
        },
        "Single Request Limit": {
            "unit": "MB",
            "used": None,
            "limit": single_req_limit,
        },
    }

    """

    console = Console()
    table = Table("Metric", "Usage")
    for title, info in res.items():
        table.add_row(*format_usage_line(**info, title=title))
    console.print(table)


def format_usage_line(
    used: Union[int, float, None],
    limit: Union[int, float],
    unit: str,
    title: str,
):
    limit_str = f"{limit:.2f}" if isinstance(limit, float) else str(limit)
    if used is None:
        return (f"{title} ({unit})", f"{limit_str}")
    used_str = f"{used:.2f}" if isinstance(used, float) else str(used)
    return (
        f"{title} ({unit})",
        f"{used_str} / {limit_str} ({used / limit * 100:.2f}%)",
    )
