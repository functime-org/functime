import os
from typing import Any, Mapping

import toml

USER_CONFIG_PATH: str = os.environ.get("FUNCTIME_CONFIG_PATH") or os.path.expanduser(
    "~/.functime.toml"
)

FUNCTIME_SERVER_URL = (
    os.environ.get("FUNCTIME_SERVER_URL")
    or "https://functional-analytics--prod-functime-api-endpoint.modal.run"
)
AUTH0_DOMAIN = "functime.us.auth0.com"
AUTH0_CLIENT_ID = "8NNUOaC3yI1PBdpsZdvxBwuKvVq03RpK"
AUTH_FLOW_TIMEOUT = 120
API_CALL_TIMEOUT = 60
API_CALL_MAX_RETRIES = 3


def _read_user_config():
    if os.path.exists(USER_CONFIG_PATH):
        with open(USER_CONFIG_PATH) as f:
            return toml.load(f)
    else:
        return {}


def _store_user_config(new_settings: Mapping[str, Any]) -> None:
    """Internal method, used by the CLI to set tokens."""
    user_config = _read_user_config()
    for key, value in new_settings.items():
        if value is None:
            del user_config[key]
        else:
            user_config[key] = value
    _write_user_config(user_config)


def _write_user_config(user_config: Mapping[str, Any]) -> None:
    with open(USER_CONFIG_PATH, "w") as f:
        toml.dump(user_config, f)
