import httpx

from functime.config import FUNCTIME_SERVER_URL, _read_user_config, _store_user_config


def get_access_token(use_cache: bool = True):
    config = _read_user_config()
    if use_cache and "auth_token" in config:
        return config["auth_token"]

    with httpx.Client(http2=True) as client:
        response = client.post(
            FUNCTIME_SERVER_URL + "/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": config["token_id"],
                "password": config["token_secret"],
            },
        )
        response.raise_for_status()
    access_token = response.json()["access_token"]
    _store_user_config({"auth_token": access_token})
    return access_token
