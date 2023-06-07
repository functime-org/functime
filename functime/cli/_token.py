import getpass
from typing import Optional

import rich
import typer

from functime.config import USER_CONFIG_PATH, _store_user_config
from functime.io.client import FunctimeH2Client

token_cli = typer.Typer(name="token", help="Manage tokens.", no_args_is_help=True)


@token_cli.command(help="Manage tokens.", no_args_is_help=True)
def set(
    token_id: Optional[str] = typer.Option(None, help="Account token ID."),
    token_secret: Optional[str] = typer.Option(None, help="Account token secret."),
    no_verify: bool = False,
):
    if token_id is None:
        token_id = getpass.getpass("Token ID:")
    if token_secret is None:
        token_secret = getpass.getpass("Token secret:")

    if not no_verify:
        credentials = {"token_id": token_id, "token_secret": token_secret}
        with FunctimeH2Client(msg="Verifying token", credentials=credentials) as client:
            client.get("/verify")
        rich.print("[green]Token verified successfully![/green]")

    _store_user_config({"token_id": token_id, "token_secret": token_secret})
    rich.print(f"Token written to {USER_CONFIG_PATH}")
