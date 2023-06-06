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
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


def format_dict(
    d, sep="\n", kv_sep="=", kv_sep_color="white", k_color="blue", v_color="green"
):
    items = []
    for k, v in d.items():
        key = apply_color(k, k_color)
        kv_sep = apply_color(kv_sep, kv_sep_color)
        val = apply_color(v, v_color)
        items.append(f"{key}{kv_sep}{val}")
    return sep.join(items)


def format_inner_stats(inner):
    return {k.replace("_", " ").capitalize(): v for k, v in inner.items()}


def format_stats(input_stats):
    vals = []
    for key, inner_stat in input_stats.items():
        # e.g. X_future_stats -> X future
        name = key.replace("_", " ").rsplit(" ", 1)[0]
        inner = format_inner_stats(inner_stat)
        vals.append(
            f"Dataframe: {apply_color(name, 'yellow')}\n{format_dict(inner, kv_sep=' = ')}"
        )
    return "\n".join(vals)


def format_time(t, color="white"):
    return apply_color(timestamp_to_local(t).replace(" ", "\n"), color)


def apply_color(s, color="white"):
    return f"[{color}]{str(s)}[/{color}]"


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
            "Stub ID",
            "Model ID",
            "Model Params",
            "Stats",
            "Created At",
            "Last Used",
        )
        for id, est in res.items():
            model_id = est["params"]["model_id"]
            table.add_row(
                id + "\n",
                f"[yellow]{model_id}[/yellow]\n",
                format_dict(est["model_kwargs"], kv_sep=" = ") + "\n",
                format_stats(est["input_stats"]) + "\n",
                format_time(est["created_at"]) + "\n",
                format_time(est["last_used"], color="green") + "\n",
            )

    console.print(table)
