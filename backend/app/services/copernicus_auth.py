import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AccessToken:
    value: str
    expires_at: float

    @property
    def valid(self) -> bool:
        return bool(self.value) and time.time() < self.expires_at - 60


class CopernicusAuth:
    """OAuth2 client with token reuse and secret-safe errors."""

    def __init__(self, client_id: str, client_secret: str, token_url: str) -> None:
        self.client_id, self.client_secret, self.token_url = client_id, client_secret, token_url
        self._token: AccessToken | None = None

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def get_token(self) -> str:
        if self._token and self._token.valid:
            return self._token.value
        if not self.configured:
            raise RuntimeError("Copernicus credentials are not configured")
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(self.token_url, data={"grant_type": "client_credentials"}, auth=(self.client_id, self.client_secret))
            response.raise_for_status()
            payload = response.json()
            self._token = AccessToken(payload["access_token"], time.time() + int(payload.get("expires_in", 300)))
            return self._token.value
        except httpx.HTTPStatusError as exc:
            logger.error("Copernicus authentication failed with status %s", exc.response.status_code)
            raise RuntimeError("Copernicus authentication failed") from exc
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.error("Copernicus authentication request failed: %s", type(exc).__name__)
            raise RuntimeError("Copernicus authentication request failed") from exc
