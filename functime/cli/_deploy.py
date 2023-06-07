from typing import List, Optional

import typer
from rich.console import Console

from functime.io.client import FunctimeH2Client

deploy_cli = typer.Typer(
    name="deploy", help="Manage deployed models.", no_args_is_help=True
)


def _remove_api_call(params: dict, msg: str):
    with FunctimeH2Client(msg=msg) as client:
        response = client.post(
            "/deploy/remove",
            headers={
                "Content-Type": "application/json",
            },
            json=params,
        )
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
    # with console.status("[blue]Removing estimators...[/blue]", spinner="dots"):
    params = {
        "all": all,
        "estimator_ids": estimator_ids,
    }
    removed = _remove_api_call(params, "Removing estimators")
    if removed:
        console.print(f"[green]Successfully removed {removed}![/green]")
    else:
        console.print("[yellow]No estimators removed.[/yellow]")
