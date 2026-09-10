"""Narrow, asynchronous adapter for the Black Forest Labs FLUX.2 Pro API.

The rest of the application should deal only with the DTOs in this module.  In
particular, it must not retain an API key or a signed BFL delivery URL in an
error message.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

BFL_BASE_URL = "https://api.bfl.ai"
FLUX_2_PRO_PATH = "/v1/flux-2-pro"
FLUX_2_WIDTH = 1088
FLUX_2_HEIGHT = 1920
FLUX_2_OUTPUT_FORMAT = "jpeg"
MAX_REFERENCES = 7
DEFAULT_DOWNLOAD_LIMIT_BYTES = 25 * 1024 * 1024
DEFAULT_REDIRECT_LIMIT = 3
_TERMINAL_FAILURE_STATUSES = frozenset(
    {
        "task not found",
        "request moderated",
        "content moderated",
        "error",
    }
)
_NONTERMINAL_STATUSES = frozenset({"pending", "reasoning", "generating"})
_API_CLUSTER_HOST = re.compile(r"api\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.bfl\.ai\Z")


class BFLProviderError(RuntimeError):
    """A sanitized error returned by, or encountered while calling, BFL."""


class BFLProtocolError(BFLProviderError):
    """BFL responded successfully but did not honour its response contract."""


class BFLDownloadError(BFLProviderError):
    """A generated result could not be downloaded safely."""


class BFLRetryableDownloadError(BFLDownloadError):
    """Result retrieval can be attempted again using the same provider job."""


class BFLSubmissionRejected(BFLProviderError):
    """BFL definitively rejected a submission; it must be recorded as failed."""


class BFLSubmissionUncertain(BFLProviderError):
    """Transport failed, so BFL may have accepted a billable submission."""


class BFLJobState(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Flux2ProRequest:
    """The already-normalized images to submit to FLUX.2 Pro.

    ``reference_images`` are data URIs or provider-reachable image strings.
    They deliberately do not include local filesystem paths: callers must
    prepare provider upload variants before crossing this boundary.
    """

    prompt: str
    reference_images: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("A non-empty FLUX prompt is required.")
        if not self.reference_images:
            raise ValueError("At least one reference image is required.")
        if len(self.reference_images) > MAX_REFERENCES:
            raise ValueError(
                f"FLUX.2 Pro accepts at most {MAX_REFERENCES} references for this application."
            )
        if any(not image.strip() for image in self.reference_images):
            raise ValueError("Reference images must be non-empty strings.")

    def payload(self) -> dict[str, object]:
        """Return BFL's wire format without exposing any local app details."""

        payload: dict[str, object] = {
            "prompt": self.prompt,
            "width": FLUX_2_WIDTH,
            "height": FLUX_2_HEIGHT,
            "output_format": FLUX_2_OUTPUT_FORMAT,
            "input_image": self.reference_images[0],
        }
        for position, image in enumerate(self.reference_images[1:], start=2):
            payload[f"input_image_{position}"] = image
        return payload


@dataclass(frozen=True, slots=True)
class BFLSubmission:
    request_id: str
    polling_url: str


@dataclass(frozen=True, slots=True)
class BFLPollResult:
    state: BFLJobState
    result_url: str | None = None
    error: str | None = None


def _safe_provider_detail(value: str, *, api_key: str) -> str:
    """Keep a short provider diagnostic without retaining credentials or URLs."""

    # Provider errors are not a trusted data source.  Redact entire URLs rather
    # than attempting to enumerate the signed query parameter names BFL may use.
    without_key = re.sub(re.escape(api_key), "[redacted API key]", value, flags=re.I)
    without_urls = re.sub(r"https?://[^\s'\"]+", "[redacted URL]", without_key, flags=re.I)
    return without_urls[:500]


def _terminal_failure_detail(data: Mapping[str, Any], *, status: str, api_key: str) -> str:
    """Return a safe terminal diagnostic without inspecting result-bearing fields."""

    for field in ("error", "message"):
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            return _safe_provider_detail(value, api_key=api_key)

    details = data.get("details")
    if isinstance(details, Mapping):
        for field in ("error", "message", "reason"):
            value = details.get(field)
            if isinstance(value, str) and value.strip():
                return _safe_provider_detail(value, api_key=api_key)

    # The recognized provider status is a useful, safe fallback.  Do not
    # inspect result or preview fields: they may contain signed delivery URLs.
    return status.strip()


def _url_parts(value: object, *, field: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise BFLProtocolError(f"BFL response omitted a valid {field}.")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BFLProtocolError(f"BFL response contained an invalid {field}.") from exc
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise BFLProtocolError(f"BFL response contained a non-HTTPS {field}.")
    return value, hostname.casefold()


def _is_delivery_host(hostname: str) -> bool:
    return hostname == "delivery.bfl.ai" or (
        hostname.startswith("delivery.") and hostname.endswith(".bfl.ai")
    )


def _is_api_host(hostname: str) -> bool:
    return hostname == "api.bfl.ai" or _API_CLUSTER_HOST.fullmatch(hostname) is not None


def _response_error(response: httpx.Response, *, action: str) -> BFLProviderError:
    # Do not include body text: it can contain submitted prompt/image values and
    # potentially signed URLs.  The status still makes the error actionable.
    return BFLProviderError(f"BFL {action} failed with HTTP {response.status_code}.")


class BFLClient:
    """HTTP client for Umkleide's FLUX.2 Pro integration.

    A caller supplies an already-resolved API key.  This class intentionally
    does not inspect environment variables, making it straightforward to test
    and ensuring environment ownership remains in the application layer.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: httpx.Timeout | float = 30.0,
        max_download_bytes: int = DEFAULT_DOWNLOAD_LIMIT_BYTES,
        max_redirects: int = DEFAULT_REDIRECT_LIMIT,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty BFL API key is required.")
        if max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive.")
        if max_redirects < 0:
            raise ValueError("max_redirects cannot be negative.")

        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        self._owns_client = client is None
        self._max_download_bytes = max_download_bytes
        self._max_redirects = max_redirects

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def _headers(self) -> dict[str, str]:
        return {"accept": "application/json", "x-key": self._api_key}

    async def submit_flux_2_pro(self, request: Flux2ProRequest) -> BFLSubmission:
        """Submit one asynchronous FLUX.2 Pro request."""

        url = f"{BFL_BASE_URL}{FLUX_2_PRO_PATH}"
        try:
            response = await self._client.post(url, headers=self._headers, json=request.payload())
        except httpx.HTTPError as exc:
            raise BFLSubmissionUncertain("BFL submission could not be completed.") from exc
        if 400 <= response.status_code < 500:
            raise BFLSubmissionRejected(
                f"BFL submission was rejected with HTTP {response.status_code}."
            )
        if not 200 <= response.status_code < 300:
            raise BFLSubmissionUncertain(
                f"BFL submission received an unexpected HTTP {response.status_code} response."
            )

        try:
            data = self._json_object(response, action="submission")
        except BFLProtocolError as exc:
            raise BFLSubmissionUncertain(
                "BFL submission response did not satisfy the expected protocol."
            ) from exc
        request_id = data.get("id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise BFLSubmissionUncertain("BFL submission response omitted a valid id.")
        # Preserve the provider's receipt before applying the outbound credential policy.
        # ``poll`` validates the URL immediately before transmitting the API key.
        polling_url = data.get("polling_url")
        if not isinstance(polling_url, str) or not polling_url.strip():
            raise BFLSubmissionUncertain("BFL submission response omitted a valid polling URL.")
        return BFLSubmission(request_id=request_id, polling_url=polling_url)

    async def poll(self, polling_url: str) -> BFLPollResult:
        """Make exactly one status request to BFL's returned polling URL."""

        polling_url = self._validated_polling_url(polling_url)
        try:
            response = await self._client.get(polling_url, headers=self._headers)
        except httpx.HTTPError as exc:
            raise BFLProviderError("BFL status check could not be completed.") from exc
        if response.is_error:
            raise _response_error(response, action="status check")

        data = self._json_object(response, action="status check")
        provider_status = data.get("status")
        if not isinstance(provider_status, str) or not provider_status.strip():
            raise BFLProtocolError("BFL status response omitted a valid status.")

        normalized_status = provider_status.casefold()
        if normalized_status == "ready":
            result = data.get("result")
            if not isinstance(result, Mapping):
                raise BFLProtocolError("BFL ready response omitted a result object.")
            result_url = self._validated_result_url(result.get("sample"), field="result sample URL")
            return BFLPollResult(BFLJobState.READY, result_url=result_url)
        if normalized_status in _TERMINAL_FAILURE_STATUSES:
            return BFLPollResult(
                BFLJobState.FAILED,
                error=_terminal_failure_detail(data, status=provider_status, api_key=self._api_key),
            )
        if normalized_status in _NONTERMINAL_STATUSES:
            return BFLPollResult(BFLJobState.PROCESSING)
        raise BFLProtocolError("BFL status response contained an unknown status.")

    async def download_result(self, result_url: str) -> bytes:
        """Download an HTTPS result with a byte bound and HTTPS-only redirects."""

        current_url = self._validated_result_url(result_url, field="result URL")
        for redirect_count in range(self._max_redirects + 1):
            try:
                async with self._client.stream(
                    "GET", current_url, headers={"accept": "image/*"}, follow_redirects=False
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if location is None:
                            raise BFLDownloadError("BFL result download redirect omitted Location.")
                        if redirect_count == self._max_redirects:
                            raise BFLDownloadError(
                                "BFL result download exceeded the redirect limit."
                            )
                        next_url = urljoin(current_url, location)
                        current_url = self._validated_result_url(
                            next_url, field="result download redirect"
                        )
                        continue
                    if response.status_code in {408, 429} or 500 <= response.status_code < 600:
                        raise BFLRetryableDownloadError(
                            f"BFL result download is unavailable (HTTP {response.status_code})."
                        )
                    if response.is_error:
                        raise _response_error(response, action="result download")
                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            if int(content_length) > self._max_download_bytes:
                                raise BFLDownloadError(
                                    "BFL result exceeds the configured download limit."
                                )
                        except ValueError as exc:
                            raise BFLDownloadError(
                                "BFL result had an invalid Content-Length."
                            ) from exc

                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > self._max_download_bytes:
                            raise BFLDownloadError(
                                "BFL result exceeds the configured download limit."
                            )
                    return bytes(content)
            except BFLProviderError:
                raise
            except httpx.HTTPError as exc:
                raise BFLRetryableDownloadError(
                    "BFL result download could not be completed."
                ) from exc

        raise BFLDownloadError("BFL result download exceeded the redirect limit.")

    @staticmethod
    def _json_object(response: httpx.Response, *, action: str) -> Mapping[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise BFLProtocolError(f"BFL {action} response was not valid JSON.") from exc
        if not isinstance(data, Mapping):
            raise BFLProtocolError(f"BFL {action} response was not a JSON object.")
        return data

    def _validated_polling_url(self, value: object) -> str:
        url, hostname = _url_parts(value, field="polling_url")
        if not _is_api_host(hostname):
            raise BFLProtocolError("BFL response contained an untrusted polling_url.")
        return url

    def _validated_result_url(self, value: object, *, field: str) -> str:
        url, hostname = _url_parts(value, field=field)
        if not _is_delivery_host(hostname):
            raise BFLProtocolError(f"BFL response contained an untrusted {field}.")
        return url


__all__ = [
    "BFL_BASE_URL",
    "BFLDownloadError",
    "BFLClient",
    "BFLJobState",
    "BFLPollResult",
    "BFLProtocolError",
    "BFLProviderError",
    "BFLSubmissionRejected",
    "BFLSubmissionUncertain",
    "BFLSubmission",
    "DEFAULT_DOWNLOAD_LIMIT_BYTES",
    "FLUX_2_HEIGHT",
    "FLUX_2_OUTPUT_FORMAT",
    "FLUX_2_PRO_PATH",
    "FLUX_2_WIDTH",
    "Flux2ProRequest",
    "MAX_REFERENCES",
]
