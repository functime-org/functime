from typing import List, Optional

import httpx
import typer
from rich.console import Console

from functime.config import API_CALL_TIMEOUT, FUNCTIME_SERVER_URL
from functime.io.auth import require_token

deploy_cli = typer.Typer(
    name="deploy", help="Manage deployed models.", no_args_is_help=True
)


@require_token
def _remove_api_call(token, endpoint: str, params: dict):
    with httpx.Client(http2=True) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = client.post(
            FUNCTIME_SERVER_URL + endpoint,
            headers=headers,
            json=params,
            timeout=API_CALL_TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


@deploy_cli.command(help="Remove a model deployment.", no_args_is_help=True)
def remove(
    estimator_ids: Optional[List[str]] = typer.Argument(
        None, help="The stub ID(s) of the estimator(s) to remove."
    ),
    all: bool = typer.Option(False, help="Remove all estimators."),
):
    console = Console()
    confirm_msg = "Do you really want to remove {}?".format(
        "all estimators" if all else estimator_ids
    )
    if not typer.confirm(confirm_msg):
        console.print("[blue]Stopped.[/blue]")
        return
    with console.status("[blue]Removing estimators...[/blue]", spinner="dots"):
        params = {
            "all": all,
            "estimator_ids": estimator_ids,
        }
        removed = _remove_api_call("/deploy/remove", params)
    if removed:
        console.print(f"[green]Successfully removed {removed}![/green]")
    else:
        console.print("[yellow]No estimators removed.[/yellow]")
