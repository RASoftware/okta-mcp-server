# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2026-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Per-item tolerance for Okta SDK **list** responses.

Why this exists
---------------
:mod:`okta_mcp_server.utils.okta_compat` fixes four *known* over-strict models,
but those models live inside a pinned dependency generated from a spec that keeps
drifting; the next attribute Okta adds will break deserialization again.  For
audit and inventory work — "list every app in the tenant" — partial results with
an explicit warning are strictly more useful than a hard failure that returns
nothing at all.

Where it hooks in
-----------------
``okta/api_client.py`` maps a 200 response to a type string such as
``"List[Application]"`` and expands it with::

    return [self.__deserialize(sub_data, sub_kls) for sub_data in data]

There is no per-element guard, so one bad record aborts the whole page.  This
module replaces ``ApiClient``'s name-mangled ``_ApiClient__deserialize`` with a
wrapper that, for ``List[...]`` targets, parses each element in its own
``try``/``except``.  Assigning to the mangled attribute is sufficient: the SDK's
internal ``self.__deserialize(...)`` calls compile to
``self._ApiClient__deserialize(...)`` and therefore resolve to the replacement.

Hooking the SDK's own deserialization seam — rather than bypassing the typed
client and re-issuing raw requests — means we do not have to re-implement the
SDK's snake_case→camelCase query-parameter mapping, and the tolerance
generalizes to policies, policy rules, groups and users rather than only apps.

Scope: **list responses only**
------------------------------
Single-object endpoints (``get_application``, ``get_policy``,
``get_policy_rule``) are deliberately **not** made tolerant.  There is no partial
result to salvage from a single object — dropping it would turn a hard error into
a silent ``None`` — and those paths are already covered by the targeted model
patches in :mod:`okta_mcp_server.utils.okta_compat`.

RA Capital fork: bad items are kept, not dropped
------------------------------------------------
Upstream PR 108 drops an unparseable item from the list and only reports it in
the warnings array.  For audit work ("is Wispr Flow still assigned to anyone?")
a silently shorter inventory is a trap, so this fork keeps the item in the list
as a **trimmed raw dict** built by :func:`_fallback_item`: the handful of fields
every inventory needs (``id``, ``label``, ``name``, ``signOnMode``, ``status``,
``features``, ``created``, ``lastUpdated``, plus ``profile``/``type`` for groups
and users) and a ``parse_error`` string saying why the SDK rejected it.  The
warning entry is still recorded so callers can tell the two apart.

Warnings and redaction
----------------------
Failures are collected in a :class:`contextvars.ContextVar`, so they are
async-safe and scoped per request rather than shared process-wide.  Each entry
carries the target model name and a one-line summary of the validation error.

RA Capital fork: the item's **raw payload dict** is only included (as
``raw_item``) when the tool opted in via :func:`set_include_raw_items`, which
the list tools expose as ``include_raw=True``.  By default it is left out: the
trimmed item in ``items`` already carries ``parse_error``, and 23 raw app
payloads (credentials, ``_links``, settings) were more than doubling the size
of a list response.  The raw payload is real tenant data and is **never**
written to the logs at INFO or above; it appears solely in DEBUG output.

Strict mode
-----------
Set ``OKTA_MCP_STRICT_DESERIALIZATION`` to ``1``/``true``/``yes``
(case-insensitive) to disable tolerance entirely and let ``ValidationError``
propagate exactly as it does without this module.  The flag is read at call time,
not import time.
"""

from __future__ import annotations

import contextvars
import os
import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pydantic import ValidationError

__all__ = [
    "STRICT_ENV_VAR",
    "get_deserialization_warnings",
    "install_tolerant_deserialization",
    "reset_deserialization_warnings",
    "set_include_raw_items",
    "strict_deserialization_enabled",
]


#: Environment variable that restores the SDK's original hard-fail behavior.
STRICT_ENV_VAR = "OKTA_MCP_STRICT_DESERIALIZATION"

#: Maximum characters of the error summary kept in a warning entry / parse_error.
#: 120 is enough for "<field>: <message>" plus the start of pydantic's enum list.
_ERROR_SUMMARY_LIMIT = 120

#: Matches the SDK's own ``List[...]`` response-type spelling.
_LIST_TYPE_RE = re.compile(r"^List\[(.*)]$")

#: Per-request collection of dropped items.  ``None`` means "nothing collected
#: in this context yet"; the list is created lazily so the ContextVar never
#: shares a mutable default across contexts.
_warnings: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "okta_mcp_deserialization_warnings", default=None
)

#: Per-request switch: should warnings carry the raw payload (``raw_item``)?
#: Set by the tool via :func:`set_include_raw_items`; reset with the warnings.
_include_raw: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "okta_mcp_deserialization_include_raw", default=False
)

#: Set once :func:`install_tolerant_deserialization` has patched ``ApiClient``.
_installed = False


def strict_deserialization_enabled() -> bool:
    """Return True when the operator has opted out of per-item tolerance.

    Truthy values: ``1``, ``true``, ``yes`` (case-insensitive).  Read on every
    call rather than cached at import, so tests and operators can toggle it.
    """
    return os.environ.get(STRICT_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def reset_deserialization_warnings() -> None:
    """Start a fresh, empty warning collection for the current context.

    Called by :func:`okta_mcp_server.utils.serialization.json_response` before it
    invokes a tool, so warnings never leak between requests.
    """
    _warnings.set([])
    _include_raw.set(False)


def set_include_raw_items(include_raw: bool) -> None:
    """Choose whether this request's warnings carry the raw payload.

    A list tool calls this with its ``include_raw`` argument before talking to
    Okta.  ``json_response`` resets it to False at the start of every call.
    """
    _include_raw.set(bool(include_raw))


def get_deserialization_warnings() -> List[Dict[str, Any]]:
    """Return a copy of the warnings collected in the current context.

    ``raw_item`` is stripped from each entry unless :func:`set_include_raw_items`
    was called with True in this context.
    """
    collected = _warnings.get()
    if not collected:
        return []
    if _include_raw.get():
        return [dict(entry) for entry in collected]
    return [{k: v for k, v in entry.items() if k != "raw_item"} for entry in collected]


#: Top-level keys kept on an item the SDK could not parse.  Covers apps
#: (label, signOnMode, features), groups/users (profile, type) and the common
#: audit fields.  Anything else is left out so a broken item never bloats a
#: compact listing.
_FALLBACK_KEYS = (
    "id", "label", "name", "signOnMode", "status", "features",
    "created", "lastUpdated", "type", "profile",
)

#: Inside ``profile`` (groups and users) keep only the identifying fields.
_FALLBACK_PROFILE_KEYS = ("name", "description", "login", "email", "firstName", "lastName")


def _error_summary(exc: BaseException) -> str:
    """One short line describing why an item failed to parse.

    Pydantic's ``str(ValidationError)`` runs to several hundred characters per
    item (every failing field, its input value, a docs URL).  Twenty of those
    in one list response is more text than the items themselves, so keep only
    the first failing field and its message plus a count of the rest.
    Anything that is not a ValidationError is cut to its first line.
    """
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(part) for part in first.get("loc", ()))
            summary = f"{loc}: {first['msg']}" if loc else str(first.get("msg", ""))
            if len(errors) > 1:
                more = len(errors) - 1
                summary += f" (+{more} more validation error{'s' if more > 1 else ''})"
            return summary[:_ERROR_SUMMARY_LIMIT]
    text = str(exc).strip()
    return (text.splitlines()[0] if text else type(exc).__name__)[:_ERROR_SUMMARY_LIMIT]


def _fallback_item(model_name: str, exc: BaseException, payload: Any) -> Dict[str, Any]:
    """Build the trimmed raw dict that stands in for an unparseable list item.

    The dict is JSON-native already, so ``to_jsonable`` passes it through
    unchanged, and it sits in ``items`` next to the parsed SDK models.
    """
    item: Dict[str, Any] = {}
    if isinstance(payload, dict):
        for key in _FALLBACK_KEYS:
            if key in payload:
                item[key] = payload[key]
        profile = item.get("profile")
        if isinstance(profile, dict):
            item["profile"] = {k: v for k, v in profile.items() if k in _FALLBACK_PROFILE_KEYS}
        # Match the compact application shape: visibility (hide flags) for
        # every app, and settings.app.url for bookmarks since that is the only
        # setting that says where a bookmark points.  Nothing else from
        # settings or credentials is kept.
        visibility = payload.get("visibility")
        if isinstance(visibility, dict):
            item["visibility"] = visibility
        if payload.get("signOnMode") == "BOOKMARK":
            app_settings = (payload.get("settings") or {}).get("app") if isinstance(payload.get("settings"), dict) else None
            if isinstance(app_settings, dict) and "url" in app_settings:
                item["settings"] = {"app": {"url": app_settings["url"]}}
    item["parse_error"] = f"{model_name}: {_error_summary(exc)}"
    return item


def _record_warning(model_name: str, exc: BaseException, payload: Any) -> None:
    """Record one unparseable item, and log it **without** the raw payload above DEBUG."""
    summary = _error_summary(exc)
    entry: Dict[str, Any] = {
        "type": "deserialization_error",
        "model": model_name,
        "error": summary,
        "message": "Kept in 'items' as a trimmed dict with 'parse_error'; include_raw=True adds the raw payload.",
        # Stored on every entry; get_deserialization_warnings() strips it unless
        # the tool asked for it, so the big payload never leaks out by accident.
        "raw_item": payload,
    }
    collected = _warnings.get()
    if collected is None:
        collected = []
        _warnings.set(collected)
    collected.append(entry)

    # The raw payload is real tenant data — DEBUG only, never INFO or above.
    logger.warning(f"[tolerant_deserialization] kept one unparseable {model_name} as raw dict: {summary}")
    logger.debug(f"[tolerant_deserialization] unparseable {model_name} raw payload: {payload!r}")


def install_tolerant_deserialization() -> bool:
    """Patch ``okta.api_client.ApiClient`` for per-item list tolerance.

    Idempotent: a second call is a no-op and returns ``False``.  Returns ``True``
    when the patch was installed by this call.

    Called at the top of :mod:`okta_mcp_server.server` alongside
    :func:`okta_mcp_server.utils.okta_compat.apply_okta_model_compat`.
    """
    global _installed
    if _installed:
        return False

    try:
        from okta.api_client import ApiClient
    except ImportError as exc:  # pragma: no cover — SDK is a hard dependency
        logger.warning(f"[tolerant_deserialization] okta.api_client unavailable: {exc}")
        return False

    original: Optional[Callable] = getattr(ApiClient, "_ApiClient__deserialize", None)
    if original is None:  # pragma: no cover — future SDK renamed the method
        logger.warning(
            "[tolerant_deserialization] ApiClient.__deserialize not found; "
            "list responses keep the SDK's all-or-nothing behavior"
        )
        return False

    def tolerant_deserialize(self, data, klass):
        """Deserialize ``data`` as ``klass``, keeping unparseable list elements as raw dicts.

        Signature intentionally mirrors the SDK's own ``__deserialize`` — it is
        installed in its place and must accept the same positional arguments.

        Only ``List[...]`` targets are handled specially; everything else is
        delegated to the SDK implementation untouched, which keeps single-object
        endpoints strict.
        """
        if (
            not strict_deserialization_enabled()
            and isinstance(klass, str)
            and isinstance(data, list)
        ):
            match = _LIST_TYPE_RE.match(klass)
            if match is not None:
                sub_kls = match.group(1)
                parsed: List[Any] = []
                for item in data:
                    try:
                        parsed.append(original(self, item, sub_kls))
                    # ValidationError subclasses ValueError in Pydantic v2, but both
                    # are named: the SDK's own ``from_dict`` raises a bare ValueError
                    # for a payload it cannot route (e.g. an unrecognized subtype
                    # discriminator), and that is equally a per-item problem.
                    except (ValidationError, ValueError) as exc:
                        _record_warning(sub_kls, exc, item)
                        # RA Capital fork: keep the item so the inventory stays
                        # complete; the caller sees `parse_error` on it.
                        parsed.append(_fallback_item(sub_kls, exc, item))
                return parsed

        return original(self, data, klass)

    ApiClient._ApiClient__deserialize = tolerant_deserialize
    _installed = True
    logger.debug("[tolerant_deserialization] installed per-item tolerance on ApiClient")
    return True
