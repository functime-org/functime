import datetime

import httpx
import typer
from rich.console import Console
from rich.table import Table

from functime.config import API_CALL_TIMEOUT, FUNCTIME_SERVER_URL
from functime.io.auth import require_token


@require_token
def _get_list_response(token, **params):
    with httpx.Client(http2=True) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = client.get(
            FUNCTIME_SERVER_URL + "/list_models",
            headers=headers,
            params=params,
            timeout=API_CALL_TIMEOUT,
        )
        response.raise_for_status()
    return response.json()


def timestamp_to_local(ts: str):
    dt = datetime.datetime.fromisoformat(ts)
    local_dt = dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_dict(d):
    return "\n".join([f"{str(k)}={str(v)}" for k, v in d.items()])


def list_cli(
    id: bool = typer.Option(False, help="Show the IDs only."),
):
    res = _get_list_response(id=id)
    console = Console()
    if id:
        table = Table("Model ID")
        for id in res:
            table.add_row(id)
    else:
        table = Table(
            "Model ID", "Created At", "Last Used", "Parameters", "Model Kwargs"
        )
        for id, est in res.items():
            table.add_row(
                id,
                timestamp_to_local(est["created_at"]),
                timestamp_to_local(est["last_used"]),
                format_dict(est["params"]),
                format_dict(est["model_kwargs"]),
            )

    console.print(table)
