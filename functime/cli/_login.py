import time
import webbrowser

import httpx
import rich
import typer
from rich.console import Console

from functime.config import (
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
    AUTH_FLOW_TIMEOUT,
    FUNCTIME_SERVER_URL,
    _store_user_config,
)


def login_cli():
    console = Console()

    with httpx.Client(http2=True) as client:
        response = client.get(FUNCTIME_SERVER_URL)
        if response.status_code != 200:
            console.print("[red]Error connecting to functime server[/red]")
            raise typer.Exit(code=1)

        device_code_payload = {"client_id": AUTH0_CLIENT_ID, "scope": "openid profile"}
        device_code_response = client.post(
            f"https://{AUTH0_DOMAIN}/oauth/device/code", data=device_code_payload
        )

        if device_code_response.status_code != 200:
            console.print("[red]Error generating the device code[/red]")
            raise typer.Exit(code=1)

    console.print("[green]Generated device code![/green]")
    device_code_data = device_code_response.json()
    web_url = device_code_data["verification_uri_complete"]
    device_code_data["created_at"] = int(time.time())
    with console.status(
        "[blue]Waiting for authentication in the web browser...[/blue]", spinner="dots"
    ):
        console.print("[blue]Launching login page in your browser window...[/blue]")
        if webbrowser.open_new_tab(web_url):
            console.print(
                "[blue]If this is not showing up, please copy this URL into your web browser manually:[/blue]"
            )
        else:
            console.print(
                "[red]Was not able to launch web browser[/red]"
                " - please go to this URL manually and complete the flow:"
            )
        console.print(f"\n[link={web_url}]{web_url}[/link]\n")
    with Console().status(
        "[blue]Waiting for authentication from functime server...[/blue]",
        spinner="dots",
    ):
        response = finish_token_flow(device_code_data)

    if "error" in response:
        console.print(f"[red]Error: {response['error']}[/red]")
        raise typer.Exit(code=1)

    status = response.get("status")
    if status == "success":
        console.print(f"[green]{response.get('message')}[/green]")
        return

    _store_user_config(
        {
            "token_id": response["token_id"],
            "token_secret": response["token_secret"],
        }
    )
    console.print("[green]Login successful![/green]")


def finish_token_flow(device_code_data: dict):
    try:
        with httpx.Client(http2=True) as client:
            response = client.post(
                FUNCTIME_SERVER_URL + "/signup_flow",
                json=device_code_data,
                timeout=AUTH_FLOW_TIMEOUT,
            )
            response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail")
        status = e.response.status_code
        rich.print(f"\n[red]Authentication Error: {status} \nReason: {detail}[/red]")
        raise typer.Exit(code=1) from e
