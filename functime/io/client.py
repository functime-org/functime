import time
from typing import Mapping, Optional

import httpx
import typer
from rich.console import Console

from functime.config import (
    API_CALL_MAX_RETRIES,
    API_CALL_TIMEOUT,
    FUNCTIME_SERVER_URL,
    USER_CONFIG_PATH,
    _read_user_config,
    _store_user_config,
)


class FunctimeH2Client:
    def __init__(
        self,
        n_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        msg: Optional[str] = None,
        credentials: Optional[Mapping[str, str]] = None,
    ):
        self.console = Console()
        self.client = httpx.Client(base_url=FUNCTIME_SERVER_URL, http2=True)
        self.n_retries = n_retries or API_CALL_MAX_RETRIES
        self.timeout = timeout or API_CALL_TIMEOUT
        credentials = credentials or {}
        self.auth_token = None
        self.token_id = credentials.get("token_id")
        self.token_secret = credentials.get("token_secret")
        self.loading_msg = msg or "Loading"

    def __enter__(self):
        self.client.__enter__()
        self._get_access_token(
            token_id=self.token_id,
            token_secret=self.token_secret,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.__exit__(exc_type, exc_value, traceback)

    def get(
        self,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        return self.request("GET", endpoint, headers=headers, **kwargs)

    def post(
        self,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        return self.request("POST", endpoint, headers=headers, **kwargs)

    def request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        for retry in range(self.n_retries):
            headers = headers or {}
            headers.update({"Authorization": f"Bearer {self.auth_token}"})
            try:
                with self.console.status(
                    f"[blue]{self.loading_msg}...[/blue]", spinner="dots"
                ):
                    response = self.client.request(
                        method=method,
                        url=endpoint,
                        headers=headers,
                        timeout=self.timeout,
                        **kwargs,
                    )
                response.raise_for_status()
                return response
            except httpx.ReadTimeout:
                timeout = 2**retry
                with self.console.status(
                    f"[yellow]Retrying in {timeout} seconds...[/yellow]", spinner="dots"
                ):
                    time.sleep(timeout)
            except httpx.HTTPError as e:
                if e.response.status_code == 401:
                    with self.console.status(
                        "[blue]Re-authenticating...[/blue]", spinner="dots"
                    ):
                        self._get_access_token(force_refresh=True)
                    continue
                if e.response.status_code == 400:
                    detail = e.response.json().get("detail")
                    raise ValueError(detail) from e
                self.console.print(f"[red]{e}[/red]")
                raise e

        raise httpx.HTTPError(f"Failed to authenticate after {self.n_retries} tries.")

    def _get_access_token(
        self,
        *,
        token_id: Optional[str] = None,
        token_secret: Optional[str] = None,
        force_refresh: bool = False,
    ) -> None:
        """Get access token from server.

        Behavior
        --------
        - If either `token_id` or `token_secret` is not provided, then they are read from the config file.
        - If both are provided, then they are used instead of the config file.
        - If `force_refresh` is True, then the token is refreshed even if it is cached.
        """
        use_config = token_id is None or token_secret is None
        if use_config:
            config = _read_user_config()
            if not force_refresh and "auth_token" in config:
                # Use cached token when not forcing refresh
                self.auth_token = config["auth_token"]
                return
            token_id, token_secret = config.get("token_id"), config.get("token_secret")
        # Here the credentials should be set.
        # If not then something is wrong.
        if token_id is None or token_secret is None:
            self.console.print(
                f"\n[red]Missing credentials{f' in {USER_CONFIG_PATH}' if use_config else ''}. Please login first or set your tokens.[/red]"
            )
            raise typer.Exit(code=1)

        try:
            response = self.client.post(
                "/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "username": token_id,
                    "password": token_secret,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.console.print(
                    "\n[red]Invalid token credentials. Please login again.[/red]"
                    "\n[red]If this error persists please contact us at [/red][magenta]team@functime.ai[/magenta]"
                )
            raise
        self.auth_token = response.json()["access_token"]
        _store_user_config({"auth_token": self.auth_token})
