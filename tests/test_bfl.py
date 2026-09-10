from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from umkleide.bfl import (
    BFL_BASE_URL,
    DEFAULT_DOWNLOAD_LIMIT_BYTES,
    FLUX_2_HEIGHT,
    FLUX_2_OUTPUT_FORMAT,
    FLUX_2_PRO_PATH,
    FLUX_2_WIDTH,
    MAX_REFERENCES,
    BFLClient,
    BFLDownloadError,
    BFLJobState,
    BFLPollResult,
    BFLProtocolError,
    BFLProviderError,
    BFLRetryableDownloadError,
    BFLSubmission,
    BFLSubmissionRejected,
    BFLSubmissionUncertain,
    Flux2ProRequest,
    _safe_provider_detail,
)

Handler = Callable[[httpx.Request], httpx.Response]
TEST_API_KEY = "test-key"
REFERENCE_IMAGE = "data:image/jpeg;base64,one"
DEFAULT_POLLING_URL = f"{BFL_BASE_URL}/v1/get_result?id=1"
DELIVERY_RESULT_URL = "https://delivery.bfl.ai/result"
DELIVERY_IMAGE_URL = f"{DELIVERY_RESULT_URL}.jpg"


def _connect_error(_: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection lost")


async def _submit(handler: Handler) -> BFLSubmission:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        return await BFLClient(TEST_API_KEY, client=transport).submit_flux_2_pro(
            Flux2ProRequest("prompt", (REFERENCE_IMAGE,))
        )


async def _poll(handler: Handler, polling_url: str = DEFAULT_POLLING_URL) -> BFLPollResult:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        return await BFLClient(TEST_API_KEY, client=transport).poll(polling_url)


async def _download(
    handler: Handler, result_url: str, *, limit: int = DEFAULT_DOWNLOAD_LIMIT_BYTES
) -> bytes:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        return await BFLClient(
            TEST_API_KEY, client=transport, max_download_bytes=limit
        ).download_result(result_url)


async def test_submit_flux_2_pro_uses_fixed_endpoint_headers_and_payload() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"id": "request-123", "polling_url": f"{BFL_BASE_URL}/v1/get_result?id=123"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        submission = await BFLClient(TEST_API_KEY, client=transport).submit_flux_2_pro(
            Flux2ProRequest("wear reference garments", (REFERENCE_IMAGE, "two", "three"))
        )

    assert submission.request_id == "request-123"
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == BFL_BASE_URL + FLUX_2_PRO_PATH
    assert request.headers["x-key"] == TEST_API_KEY
    assert json.loads(request.content) == {
        "prompt": "wear reference garments",
        "width": FLUX_2_WIDTH,
        "height": FLUX_2_HEIGHT,
        "output_format": FLUX_2_OUTPUT_FORMAT,
        "input_image": REFERENCE_IMAGE,
        "input_image_2": "two",
        "input_image_3": "three",
    }


@pytest.mark.parametrize(
    ("prompt", "references", "message"),
    [
        ("", ("image",), "non-empty FLUX prompt"),
        ("prompt", (), "At least one reference"),
        ("prompt", tuple("image" for _ in range(MAX_REFERENCES + 1)), str(MAX_REFERENCES)),
        ("prompt", (" ",), "non-empty strings"),
    ],
)
def test_request_validates_prompt_and_reference_count(
    prompt: str, references: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Flux2ProRequest(prompt, references)


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (lambda _: httpx.Response(422, json={"detail": "bad request"}), BFLSubmissionRejected),
        (_connect_error, BFLSubmissionUncertain),
        (lambda _: httpx.Response(503, json={}), BFLSubmissionUncertain),
        (lambda _: httpx.Response(200, content=b"not-json"), BFLSubmissionUncertain),
        (lambda _: httpx.Response(200, json={"id": "request-1"}), BFLSubmissionUncertain),
    ],
    ids=[
        "rejected",
        "transport-outcome-uncertain",
        "server-outcome-uncertain",
        "malformed-success",
        "missing-polling-url",
    ],
)
async def test_submission_distinguishes_rejection_from_uncertain_outcomes(
    handler: Handler, error: type[Exception]
) -> None:
    with pytest.raises(error):
        await _submit(handler)


async def test_successful_submission_retains_polling_url_before_use_validation() -> None:
    polling_url = "https://future.example/get_result?id=request-1&token=secret"

    submission = await _submit(
        lambda _: httpx.Response(
            200,
            json={"id": "request-1", "polling_url": polling_url},
        )
    )

    assert submission == BFLSubmission(request_id="request-1", polling_url=polling_url)


@pytest.mark.parametrize(
    ("response", "state", "result_url", "error"),
    [
        (
            {"status": "Ready", "result": {"sample": DELIVERY_IMAGE_URL}},
            BFLJobState.READY,
            DELIVERY_IMAGE_URL,
            None,
        ),
        ({"status": "Pending"}, BFLJobState.PROCESSING, None, None),
        ({"status": "Reasoning"}, BFLJobState.PROCESSING, None, None),
        ({"status": "Generating"}, BFLJobState.PROCESSING, None, None),
        ({"status": "Request Moderated"}, BFLJobState.FAILED, None, "Request Moderated"),
        ({"status": "Content Moderated"}, BFLJobState.FAILED, None, "Content Moderated"),
        ({"status": "Task not found"}, BFLJobState.FAILED, None, "Task not found"),
        ({"status": "Error"}, BFLJobState.FAILED, None, "Error"),
    ],
)
async def test_poll_maps_documented_statuses(
    response: dict[str, object], state: BFLJobState, result_url: str | None, error: str | None
) -> None:
    result = await _poll(lambda _: httpx.Response(200, json=response))

    assert result.state is state
    assert result.result_url == result_url
    assert result.error == error


async def test_poll_returns_a_safe_terminal_error_without_leaking_provider_secrets() -> None:
    api_key = "bfl-live-reflected-secret"
    detail = f"credential {api_key}; see {DELIVERY_IMAGE_URL}?signature=secret"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"status": "Error", "details": {"message": detail}})
        )
    ) as transport:
        result = await BFLClient(api_key, client=transport).poll(DEFAULT_POLLING_URL)

    assert result.state is BFLJobState.FAILED
    assert result.error == "credential [redacted API key]; see [redacted URL]"
    assert api_key not in result.error
    assert "signature=secret" not in result.error


@pytest.mark.parametrize(
    ("detail", "api_key", "expected"),
    [
        (
            "BFL-LIVE-KEY blocked by HTTPS://delivery.bfl.ai/file?signature=secret",
            "bfl-live-key",
            "[redacted API key] blocked by [redacted URL]",
        ),
        ("x" * 600, "key", "x" * 500),
    ],
)
def test_sanitizer_redacts_keys_and_mixed_case_urls(
    detail: str, api_key: str, expected: str
) -> None:
    assert _safe_provider_detail(detail, api_key=api_key) == expected


@pytest.mark.parametrize(
    "polling_url",
    [
        DEFAULT_POLLING_URL,
        "https://api.eu.bfl.ai/v1/get_result?id=1",
        "https://api.us.bfl.ai/v1/get_result?id=1",
        "https://api.us1.bfl.ai/v1/get_result?id=1",
    ],
)
async def test_poll_accepts_bfl_global_regional_and_cluster_hosts(polling_url: str) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "Pending"})

    result = await _poll(handler, polling_url)

    assert result.state is BFLJobState.PROCESSING
    assert [str(request.url) for request in requests] == [polling_url]
    assert requests[0].headers["x-key"] == TEST_API_KEY


@pytest.mark.parametrize(
    "polling_url",
    [
        "https://attacker.example/collect",
        "https://api.attacker.example.bfl.ai/collect",
        "https://not-api.us1.bfl.ai/collect",
    ],
)
async def test_poll_rejects_untrusted_host_before_transmitting_api_key(
    polling_url: str,
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"status": "Pending"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        with pytest.raises(BFLProtocolError, match="untrusted polling_url"):
            await BFLClient("secret-key", client=transport).poll(polling_url)

    assert calls == []


@pytest.mark.parametrize(
    "polling_url",
    [
        "http://api.us1.bfl.ai/collect",
        "https://user@api.us1.bfl.ai/collect",
        "https://api.us1.bfl.ai:444/collect",
    ],
)
async def test_poll_rejects_unsafe_api_url_before_transmitting_api_key(
    polling_url: str,
) -> None:
    calls: list[httpx.Request] = []

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200))
    ) as transport:
        with pytest.raises(BFLProtocolError, match="non-HTTPS polling_url"):
            await BFLClient("secret-key", client=transport).poll(polling_url)

    assert calls == []


@pytest.mark.parametrize(
    ("result_url", "location", "message", "expected_requests"),
    [
        ("http://delivery.bfl.ai/result", None, "non-HTTPS result URL", []),
        ("https://attacker.example/result", None, "untrusted result URL", []),
        (
            DELIVERY_RESULT_URL,
            "http://delivery.bfl.ai/file",
            "non-HTTPS result download redirect",
            [DELIVERY_RESULT_URL],
        ),
        (
            DELIVERY_RESULT_URL,
            "https://attacker.example/file",
            "untrusted result download redirect",
            [DELIVERY_RESULT_URL],
        ),
    ],
)
async def test_download_enforces_result_and_redirect_host_https_policy(
    result_url: str, location: str | None, message: str, expected_requests: list[str]
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={} if location is None else {"location": location})

    with pytest.raises(BFLProtocolError, match=message):
        await _download(handler, result_url)

    assert [str(request.url) for request in requests] == expected_requests


async def test_download_follows_trusted_redirects_and_bounds_streamed_bytes() -> None:
    requests: list[httpx.Request] = []

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/first":
            return httpx.Response(302, headers={"location": "/result?signature=secret"})
        return httpx.Response(200, content=b"jpeg-bytes")

    assert await _download(redirect_handler, "https://delivery.bfl.ai/first") == b"jpeg-bytes"
    assert [str(request.url) for request in requests] == [
        "https://delivery.bfl.ai/first",
        f"{DELIVERY_RESULT_URL}?signature=secret",
    ]

    def oversized_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"12345"))

    with pytest.raises(BFLDownloadError, match="download limit"):
        await _download(oversized_handler, DELIVERY_RESULT_URL, limit=4)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 599])
async def test_download_identifies_retryable_http_errors_without_response_content(status: int):
    with pytest.raises(BFLRetryableDownloadError) as error:
        await _download(
            lambda _: httpx.Response(status, text=f"{TEST_API_KEY} {DELIVERY_RESULT_URL}"),
            DELIVERY_RESULT_URL,
        )
    assert str(status) in str(error.value)
    assert TEST_API_KEY not in str(error.value)
    assert DELIVERY_RESULT_URL not in str(error.value)


async def test_download_transport_failure_is_retryable_and_sanitized():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"{TEST_API_KEY} {request.url}")

    with pytest.raises(BFLRetryableDownloadError) as error:
        await _download(fail, DELIVERY_RESULT_URL)
    assert TEST_API_KEY not in str(error.value)
    assert DELIVERY_RESULT_URL not in str(error.value)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_download_other_http_errors_are_terminal(status: int):
    with pytest.raises(BFLProviderError) as error:
        await _download(lambda _: httpx.Response(status), DELIVERY_RESULT_URL)
    assert not isinstance(error.value, BFLRetryableDownloadError)


async def test_aclose_only_closes_the_client_owned_by_the_adapter() -> None:
    borrowed_transport = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    )
    borrowed = BFLClient(TEST_API_KEY, client=borrowed_transport)
    owned = BFLClient(TEST_API_KEY)

    await borrowed.aclose()
    await owned.aclose()

    assert not borrowed_transport.is_closed
    assert owned._client.is_closed
    await borrowed_transport.aclose()
