import time
from typing import Mapping, Optional

import httpx
from rich.console import Console

from functime.config import API_CALL_MAX_RETRIES, API_CALL_TIMEOUT, FUNCTIME_SERVER_URL
from functime.io.auth import get_access_token


class FunctimeH2Client:
    def __init__(
        self,
        n_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        msg: Optional[str] = None,
    ):
        self.client = httpx.Client(base_url=FUNCTIME_SERVER_URL, http2=True)
        self.n_retries = n_retries or API_CALL_MAX_RETRIES
        self.timeout = timeout or API_CALL_TIMEOUT
        self.token = get_access_token()
        self.console = Console()
        self.loading_msg = msg or "Loading"

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.__exit__(exc_type, exc_value, traceback)

    def get(
        self,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        return self.request("GET", endpoint, headers=headers, **kwargs)

    def post(
        self,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        return self.request("POST", endpoint, headers=headers, **kwargs)

    def request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        for retry in range(self.n_retries):
            headers = headers or {}
            headers.update({"Authorization": f"Bearer {self.token}"})
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
                if e.response.status_code != 401:
                    self.console.print(f"[red]{e}[/red]")
                    raise e
                # Current token is invalid, get a new one
                with self.console.status(
                    "[yellow]Retrying with new token...[/yellow]", spinner="dots"
                ):
                    self.token = get_access_token(use_cache=False)
        raise httpx.HTTPError(f"Failed to authenticate after {self.n_retries} tries.")
