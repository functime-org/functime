from typing import Union

from rich.console import Console
from rich.table import Table

from functime.cli._styling import apply_color, format_url
from functime.io.client import FunctimeH2Client


def _get_usage_response(**params):
    with FunctimeH2Client() as client:
        response = client.get(
            "/usage",
            headers={
                "Content-Type": "application/json",
            },
            params=params,
        )
    return response.json()


def usage_cli():
    res = _get_usage_response()
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
