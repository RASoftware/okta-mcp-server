# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2025-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Compact views of Okta objects for list tools (RA Capital fork).

A full Okta application object is several KB of settings, credentials and
``_links``.  An LLM listing 150 apps does not need any of that, and the full
payload blows through context budgets.  The ``compact=True`` default on the
list tools trims each item to the fields an access audit actually reads; the
full object is one ``get_application`` / ``get_group`` call away, or
``compact=False`` on the list tool.

Every function here accepts either a parsed SDK model or a plain dict (the
tolerant deserializer leaves unparseable items as dicts), normalizes it through
:func:`okta_mcp_server.utils.serialization.to_jsonable` first, and keeps the
nested paths of the full object (``visibility.hide`` stays nested) so a reader
switching between compact and full output sees the same field names.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from okta_mcp_server.utils.serialization import to_jsonable

# Fields kept for every application in a compact listing.
_APP_KEYS = ("id", "label", "name", "signOnMode", "status", "features", "created", "lastUpdated")

# Fields kept for every group in a compact listing.
_GROUP_KEYS = ("id", "type", "created", "lastUpdated", "lastMembershipUpdated")
_GROUP_PROFILE_KEYS = ("name", "description")

# Fields kept for an app-user assignment (GET /api/v1/apps/{id}/users).
_APP_USER_KEYS = ("id", "scope", "status", "syncState", "created", "lastUpdated", "statusChanged", "passwordChanged")

# Fields kept for an app-group assignment (GET /api/v1/apps/{id}/groups).
_APP_GROUP_KEYS = ("id", "priority", "lastUpdated")


def _as_dict(obj: Any) -> Dict[str, Any]:
    data = to_jsonable(obj)
    return data if isinstance(data, dict) else {}


def _pick(data: Dict[str, Any], keys) -> Dict[str, Any]:
    """Return the subset of ``data`` for ``keys`` that are actually present."""
    return {k: data[k] for k in keys if k in data}


def compact_application(app: Any) -> Dict[str, Any]:
    """Trim an application to its inventory fields.

    Keeps ``visibility.hide`` (is it on the end-user dashboard) and, for
    BOOKMARK apps, ``settings.app.url`` (the only setting that tells you what
    a bookmark actually points at).  A ``parse_error`` left by the tolerant
    deserializer is carried through so the caller knows the item is partial.
    """
    data = _as_dict(app)
    out = _pick(data, _APP_KEYS)

    visibility = data.get("visibility")
    if isinstance(visibility, dict) and "hide" in visibility:
        out["visibility"] = {"hide": visibility["hide"]}

    if data.get("signOnMode") == "BOOKMARK":
        settings = data.get("settings")
        app_settings = settings.get("app") if isinstance(settings, dict) else None
        if isinstance(app_settings, dict) and "url" in app_settings:
            out["settings"] = {"app": {"url": app_settings["url"]}}

    if "parse_error" in data:
        out["parse_error"] = data["parse_error"]
    return out


def compact_group(group: Any) -> Dict[str, Any]:
    """Trim a group to id, type, timestamps and profile name/description."""
    data = _as_dict(group)
    out = _pick(data, _GROUP_KEYS)
    profile = data.get("profile")
    # The SDK models Group.profile as an anyOf wrapper, which model_dump renders
    # as {"actual_instance": {...}, "any_of_schemas": [...]}; unwrap it.
    if isinstance(profile, dict) and isinstance(profile.get("actual_instance"), dict):
        profile = profile["actual_instance"]
    if isinstance(profile, dict):
        out["profile"] = _pick(profile, _GROUP_PROFILE_KEYS)
    if "parse_error" in data:
        out["parse_error"] = data["parse_error"]
    return out


def compact_app_user(app_user: Any) -> Dict[str, Any]:
    """Trim an app-user assignment.

    ``id`` is the Okta user ID.  ``scope`` is the field audits care about:
    ``USER`` means a direct assignment, ``GROUP`` means the user got the app
    through a group.  ``credentials.userName`` is the app-side username when
    present.  When the call used ``expand=user`` the embedded user's login is
    surfaced as ``login``.
    """
    data = _as_dict(app_user)
    out = _pick(data, _APP_USER_KEYS)
    credentials = data.get("credentials")
    if isinstance(credentials, dict) and "userName" in credentials:
        out["credentials"] = {"userName": credentials["userName"]}
    login = _embedded_path(data, "user", "profile", "login")
    if login is not None:
        out["login"] = login
    return out


def compact_app_group(app_group: Any) -> Dict[str, Any]:
    """Trim an app-group assignment.

    ``id`` is the Okta group ID.  When the call used ``expand=group`` the
    embedded group's name is surfaced as ``groupName``.
    """
    data = _as_dict(app_group)
    out = _pick(data, _APP_GROUP_KEYS)
    name = _embedded_path(data, "group", "profile", "name")
    if name is not None:
        out["groupName"] = name
    return out


def _embedded_path(data: Dict[str, Any], *path: str) -> Optional[Any]:
    """Walk ``data["_embedded"][path...]`` and return the value or None."""
    node: Any = data.get("_embedded")
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def compact_list(items: List[Any], fn) -> List[Dict[str, Any]]:
    """Apply one of the ``compact_*`` functions to every item in ``items``."""
    return [fn(item) for item in items]
