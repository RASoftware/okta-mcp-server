# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2025-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Raw JSON GETs through the Okta SDK's request executor (RA Capital fork).

The SDK's typed client parses every response into generated pydantic models,
and those models reject real tenant data whenever the spec drifts (upstream
okta-mcp-server issue 48).  For the read-only audit tools added in this fork we
skip the models entirely: the executor still handles auth headers, the org URL
and the SDK's built-in retry, and we just ``json.loads`` the body.

Pagination follows the ``Link: rel="next"`` header like the rest of the API.
A 429 is retried with a backoff that honours ``X-Rate-Limit-Reset`` when Okta
sends it.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from loguru import logger

from okta_mcp_server.utils.pagination import extract_after_cursor

#: How many times a single GET is attempted when Okta answers 429.
_MAX_ATTEMPTS = 4

#: Never sleep longer than this on a rate-limit reset, even if the header says so.
_MAX_BACKOFF_SECONDS = 60.0


def build_url(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Join an API path and a params dict, skipping None/empty values."""
    clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    return path + (f"?{urlencode(clean)}" if clean else "")


def _is_rate_limited(response: Any, err: Any) -> bool:
    status = getattr(response, "status", None)
    # E0000047 is Okta's "API call exceeded rate limit" error code.
    return status == 429 or "E0000047" in str(err)


def _backoff_seconds(response: Any, attempt: int) -> float:
    """Seconds to wait before retrying a 429.

    Okta's ``X-Rate-Limit-Reset`` is an epoch timestamp for when the window
    resets; fall back to exponential backoff when it is missing or odd.
    """
    headers = getattr(response, "headers", None)
    reset = None
    if headers is not None:
        try:
            reset = headers.get("X-Rate-Limit-Reset") or headers.get("x-rate-limit-reset")
        except Exception:
            reset = None
    if reset:
        try:
            wait = float(reset) - time.time()
            if wait > 0:
                return min(wait + 0.5, _MAX_BACKOFF_SECONDS)
        except (TypeError, ValueError):
            pass
    return min(float(2 ** attempt), _MAX_BACKOFF_SECONDS)


async def raw_get(client: Any, url: str) -> Tuple[Any, Any, Any]:
    """GET ``url`` and return ``(parsed_json, response, error)``.

    ``parsed_json`` is the decoded body (``None`` when the body is empty or on
    error).  ``response`` is the transport response (has ``.status`` and
    ``.headers``); it is returned even on error when Okta answered at all.
    """
    executor = client.get_request_executor()
    request, err = await executor.create_request(method="GET", url=url, body={}, headers={}, oauth=False)
    if err:
        return None, None, err

    response = None
    err = None
    for attempt in range(_MAX_ATTEMPTS):
        response, body, err = await executor.execute(request)
        if err is None:
            return (json.loads(body) if body else None), response, None
        if not _is_rate_limited(response, err) or attempt == _MAX_ATTEMPTS - 1:
            break
        wait = _backoff_seconds(response, attempt)
        logger.warning(f"[raw_request] 429 from Okta on {url}; retrying in {wait:.1f}s (attempt {attempt + 1})")
        await asyncio.sleep(wait)

    return None, response, err


async def raw_get_pages(
    client: Any,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    fetch_all: bool = True,
    max_pages: int = 50,
    delay_between_requests: float = 0.1,
) -> Tuple[List[Any], Dict[str, Any], Any]:
    """GET a list endpoint and optionally follow ``Link: rel="next"``.

    Returns ``(items, pagination_info, first_page_error)``.  If the first page
    fails there is nothing to return and ``first_page_error`` is set.  A later
    page failing does not discard what was already fetched: the error lands in
    ``pagination_info["errors"]`` and ``pagination_info["resume_cursor"]`` is
    the ``after`` value to continue from.
    """
    params = dict(params or {})
    items: List[Any] = []
    info: Dict[str, Any] = {
        "pages_fetched": 0,
        "total_items": 0,
        "stopped_early": False,
        "stop_reason": None,
        "errors": [],
        "has_more": False,
        "next_cursor": None,
    }

    page, response, err = await raw_get(client, build_url(path, params))
    if err:
        return [], info, err

    items.extend(page if isinstance(page, list) else [])
    info["pages_fetched"] = 1
    cursor = extract_after_cursor(response)

    if not fetch_all:
        info["has_more"] = cursor is not None
        info["next_cursor"] = cursor
        info["total_items"] = len(items)
        return items, info, None

    while cursor and info["pages_fetched"] < max_pages:
        if delay_between_requests > 0:
            await asyncio.sleep(delay_between_requests)
        next_params = dict(params)
        next_params["after"] = cursor
        page, response, err = await raw_get(client, build_url(path, next_params))
        if err:
            info["errors"].append({"page": info["pages_fetched"] + 1, "after": cursor, "error": str(err)})
            info["stopped_early"] = True
            info["stop_reason"] = f"API error: {err}"
            info["resume_cursor"] = cursor
            info["has_more"] = True
            info["next_cursor"] = cursor
            break
        items.extend(page if isinstance(page, list) else [])
        info["pages_fetched"] += 1
        cursor = extract_after_cursor(response)

    if cursor and info["pages_fetched"] >= max_pages and not info["stopped_early"]:
        info["stopped_early"] = True
        info["stop_reason"] = f"Reached maximum page limit ({max_pages})"
        info["resume_cursor"] = cursor
        info["has_more"] = True
        info["next_cursor"] = cursor

    info["total_items"] = len(items)
    return items, info, None
