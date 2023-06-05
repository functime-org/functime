import time
from functools import wraps

import httpx

from functime.config import FUNCTIME_SERVER_URL, _read_user_config, _store_user_config
from functime.log import get_logger

MAX_RETRIES = 3

logger = get_logger(__name__)


def require_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_access_token()
        for retry in range(MAX_RETRIES):
            try:
                return f(token, *args, **kwargs)
            except httpx.ReadTimeout:
                # exponential backoff
                timeout = 2**retry
                logger.info(f"Retrying in {timeout} seconds...")
                time.sleep(timeout)
            except httpx.HTTPError as e:
                if e.response.status_code != 401:
                    logger.error(e)
                    raise e
                token = get_access_token(use_cache=False)

        raise httpx.HTTPError(f"Failed to authenticate after {MAX_RETRIES} tries.")

    return decorated


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
