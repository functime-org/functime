from typing import Union

import httpx
from rich.console import Console
from rich.table import Table

from functime.cli.utils import apply_color, format_url
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
    usage = res["usage"]
    tier = res["tier"]
    console = Console()
    console.print(
        f"\nData and forecast usage limits reset at the start of each month."
        f"\nYou are currently on the {format_tier(tier)} tier."
        f"\nVisit {format_url('https://functime.ai/')} to upgrade."
    )
    table = Table(
        "Metric",
        "Limit",
        "Used",
    )
    for title, info in usage.items():
        table.add_row(*format_usage_line(**info, title=title))
    console.print(table)


def format_tier(tier: str):
    tier = tier.lower()
    if tier == "free":
        res = apply_color(tier, "blue")
    elif tier == "pro":
        res = apply_color(tier, "yellow")
    elif tier == "enterprise":
        res = apply_color(tier, "magenta")
    return f"[bold]{res}[/bold]"


def format_usage_pct(used: Union[int, float], limit: Union[int, float]):
    val = used / limit * 100
    if val < 70:
        color = "green"
    elif val < 90:
        color = "yellow"
    else:
        color = "red"
    return apply_color(f"{val:.2f}%", color)


def format_usage_line(
    used: Union[int, float, None],
    limit: Union[int, float],
    unit: str,
    title: str,
):
    limit_str = f"{limit:.2f}" if isinstance(limit, float) else str(limit)
    limit_str = apply_color(limit_str, "white")
    if used is not None:
        used_str = f"{used:.2f}" if isinstance(used, float) else str(used)
        used_str = (
            apply_color(used_str, "white") + f" ({format_usage_pct(used, limit)})"
        )
    else:
        used_str = "-"
    return (
        f"{title} ({unit})",
        limit_str,
        used_str,
    )
