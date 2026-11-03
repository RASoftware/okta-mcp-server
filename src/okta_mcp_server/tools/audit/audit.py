# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2025-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Read-only access-audit tools (RA Capital fork).

Four tools that answer the questions an app-access review keeps asking:

* ``list_user_app_links``      — what does this user see on their dashboard?
* ``list_application_users``   — who is assigned to this app, directly or via group?
* ``list_application_groups``  — which groups are assigned to this app?
* ``app_last_sso``             — when did anyone last sign in to this app?

All four go through :mod:`okta_mcp_server.utils.raw_request` (plain JSON via
the SDK's request executor) instead of the generated SDK models, so a spec
mismatch on one app can never fail the whole call.  Every tool is gated on an
existing read scope in ``TOOL_SCOPE_REGISTRY``; none of them can write.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from loguru import logger
from mcp.server.fastmcp import Context

from okta_mcp_server.server import mcp
from okta_mcp_server.utils.client import get_okta_client
from okta_mcp_server.utils.compact import compact_app_group, compact_app_user, compact_list
from okta_mcp_server.utils.raw_request import build_url, raw_get, raw_get_pages
from okta_mcp_server.utils.scope_guard import require_scopes
from okta_mcp_server.utils.serialization import json_response
from okta_mcp_server.utils.validation import validate_ids

#: Okta's maximum page sizes for these endpoints.
_APP_USERS_PAGE_SIZE = 500
_APP_GROUPS_PAGE_SIZE = 200

#: Safety cap on pages per call (500 users x 50 pages = 25,000 assignments).
_MAX_PAGES = 50


def _paged_result(items: list, info: Dict[str, Any], fetch_all: bool) -> Dict[str, Any]:
    """Shape a raw paginated fetch like the rest of the server's list tools."""
    result: Dict[str, Any] = {
        "items": items,
        "total_fetched": len(items),
        "has_more": info.get("has_more", False),
        "next_cursor": info.get("next_cursor"),
        "fetch_all_used": fetch_all,
        "pagination_info": info,
    }
    if info.get("errors"):
        result["errors"] = info["errors"]
    return result


@mcp.tool()
@require_scopes("okta.users.read")
@validate_ids("user_id", error_return_type="dict")
@json_response
async def list_user_app_links(user_id: str, ctx: Context = None) -> dict:
    """List the app links a user sees on their Okta End-User Dashboard.

    This is GET /api/v1/users/{id}/appLinks and matches the Applications tab
    on the user's page in the Admin Console: one entry per assigned app
    instance the user can launch, including apps assigned through groups.
    The endpoint returns everything in one response (no pagination).

    Parameters:
        user_id (str, required): The user's Okta ID or login (email).

    Returns:
        Dict containing:
        - items: List of app links. Each has id, label, appName, appInstanceId
          (the application ID to use with other app tools), appAssignmentId,
          linkUrl, hidden, sortOrder and credentialsSetup.
        - total_fetched: Number of app links
        - user_id: The user this was fetched for
    """
    logger.info(f"Listing app links for user: {user_id}")
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    try:
        client = await get_okta_client(manager)
        links, _, err = await raw_get(client, f"/api/v1/users/{user_id}/appLinks")
        if err:
            logger.error(f"Okta API error while listing app links for {user_id}: {err}")
            return {"error": str(err)}
        links = links if isinstance(links, list) else []
        logger.info(f"Retrieved {len(links)} app links for user {user_id}")
        return {"items": links, "total_fetched": len(links), "user_id": user_id}
    except Exception as e:
        logger.error(f"Exception while listing app links for {user_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}


@mcp.tool()
@require_scopes("okta.apps.read")
@validate_ids("app_id", error_return_type="dict")
@json_response
async def list_application_users(
    app_id: str,
    ctx: Context = None,
    fetch_all: bool = True,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    expand: Optional[str] = None,
    compact: bool = True,
) -> dict:
    """List the users assigned to an application, with how each one got it.

    This is GET /api/v1/apps/{id}/users. Every entry's ``scope`` says whether
    the assignment is direct (``USER``) or inherited from a group (``GROUP``);
    that is the field to read when deciding whether removing a group would
    actually take the app away from someone.

    Parameters:
        app_id (str, required): The application ID (starts with 0oa).
        fetch_all (bool, optional): Follow all pages. Default: True, because an
            assignment review needs the whole list.
        after (str, optional): Pagination cursor when fetch_all=False.
        limit (int, optional): Page size, max 500. Default: 500.
        expand (str, optional): Pass "user" to embed each user's profile; compact
            output then adds a ``login`` field.
        compact (bool, optional): If True (default) each item carries only id (the
            Okta user ID), scope, status, username (the app-side credentials.userName),
            created, lastUpdated and login (with expand=user). Pass compact=False for
            the full app-user objects (syncState, statusChanged, passwordChanged, ...).

    Returns:
        Dict containing items, total_fetched, has_more, next_cursor, fetch_all_used,
        pagination_info and a ``scope_counts`` dict ({"USER": n, "GROUP": m}).
        ``errors`` appears only when a later page failed; earlier pages are kept.
    """
    logger.info(f"Listing users assigned to application: {app_id}")
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    page_size = min(limit or _APP_USERS_PAGE_SIZE, _APP_USERS_PAGE_SIZE)
    params = {"limit": page_size, "after": after, "expand": expand}

    try:
        client = await get_okta_client(manager)
        items, info, err = await raw_get_pages(
            client, f"/api/v1/apps/{app_id}/users", params, fetch_all=fetch_all, max_pages=_MAX_PAGES
        )
        if err:
            logger.error(f"Okta API error while listing users for app {app_id}: {err}")
            return {"error": str(err)}

        # Count direct vs group assignments before trimming; both shapes keep `scope`.
        scope_counts: Dict[str, int] = {}
        for item in items:
            scope = item.get("scope", "UNKNOWN") if isinstance(item, dict) else "UNKNOWN"
            scope_counts[scope] = scope_counts.get(scope, 0) + 1

        if compact:
            items = compact_list(items, compact_app_user)

        logger.info(f"Retrieved {len(items)} app users for {app_id}: {scope_counts}")
        result = _paged_result(items, info, fetch_all)
        result["scope_counts"] = scope_counts
        result["app_id"] = app_id
        return result
    except Exception as e:
        logger.error(f"Exception while listing users for app {app_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}


@mcp.tool()
@require_scopes("okta.apps.read")
@validate_ids("app_id", error_return_type="dict")
@json_response
async def list_application_groups(
    app_id: str,
    ctx: Context = None,
    fetch_all: bool = True,
    after: Optional[str] = None,
    limit: Optional[int] = None,
    expand: Optional[str] = "group",
    compact: bool = True,
) -> dict:
    """List the groups assigned to an application, with each group's name and type.

    This is GET /api/v1/apps/{id}/groups?expand=group. An app with no group
    assignments and no user assignments is a candidate for cleanup; an inactive
    app that still has groups here will re-grant access to everyone in them if
    reactivated.

    Parameters:
        app_id (str, required): The application ID (starts with 0oa).
        fetch_all (bool, optional): Follow all pages. Default: True.
        after (str, optional): Pagination cursor when fetch_all=False.
        limit (int, optional): Page size, max 200. Default: 200.
        expand (str, optional): Default "group", which makes Okta embed each group
            in the same response so names need no extra calls. Pass None to skip it;
            compact output then resolves missing names with one GET per group.
        compact (bool, optional): If True (default) each item carries id (the Okta
            group ID), name, type (OKTA_GROUP, APP_GROUP, BUILT_IN), priority and
            lastUpdated. Pass compact=False for the full app-group objects, including
            the embedded group.

    Returns:
        Dict containing items, total_fetched, has_more, next_cursor, fetch_all_used
        and pagination_info. ``errors`` appears only when a later page failed, or when
        a group's name could not be resolved.
    """
    logger.info(f"Listing groups assigned to application: {app_id}")
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    page_size = min(limit or _APP_GROUPS_PAGE_SIZE, _APP_GROUPS_PAGE_SIZE)
    params = {"limit": page_size, "after": after, "expand": expand}

    try:
        client = await get_okta_client(manager)
        items, info, err = await raw_get_pages(
            client, f"/api/v1/apps/{app_id}/groups", params, fetch_all=fetch_all, max_pages=_MAX_PAGES
        )
        if err:
            logger.error(f"Okta API error while listing groups for app {app_id}: {err}")
            return {"error": str(err)}

        if compact:
            items = compact_list(items, compact_app_group)
            # Fallback when the embedded group is missing (expand=None, or Okta
            # left it out): one read-only GET /api/v1/groups/{id} per unresolved
            # row. Normally expand=group covers everything and this loop is idle.
            lookup_errors = await _fill_missing_group_names(client, items)
            if lookup_errors:
                info.setdefault("errors", []).extend(lookup_errors)

        logger.info(f"Retrieved {len(items)} app groups for {app_id}")
        result = _paged_result(items, info, fetch_all)
        result["app_id"] = app_id
        return result
    except Exception as e:
        logger.error(f"Exception while listing groups for app {app_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}


async def _fill_missing_group_names(client: Any, items: list) -> list:
    """Add ``name``/``type`` to compact app-group rows that lack them, via get-group calls.

    Returns a list of error dicts for groups that could not be fetched; the row
    is left without a name rather than failing the whole call.
    """
    errors = []
    for row in items:
        if not isinstance(row, dict) or "name" in row or not row.get("id"):
            continue
        group, _, err = await raw_get(client, f"/api/v1/groups/{row['id']}")
        if err or not isinstance(group, dict):
            errors.append({"group_id": row["id"], "error": str(err or "empty response")})
            continue
        profile = group.get("profile") if isinstance(group.get("profile"), dict) else {}
        if profile.get("name") is not None:
            row["name"] = profile["name"]
        if group.get("type") is not None:
            row["type"] = group["type"]
    return errors


def _iso_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _compact_log_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the handful of System Log fields that identify a sign-in."""
    actor = event.get("actor") if isinstance(event.get("actor"), dict) else {}
    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
    client = event.get("client") if isinstance(event.get("client"), dict) else {}
    return {
        "published": event.get("published"),
        "eventType": event.get("eventType"),
        "actor": {"alternateId": actor.get("alternateId"), "displayName": actor.get("displayName")},
        "outcome": {"result": outcome.get("result")},
        "client": {"ipAddress": client.get("ipAddress")},
    }


@mcp.tool()
@require_scopes("okta.logs.read")
@validate_ids("app_id", error_return_type="dict")
@json_response
async def app_last_sso(app_id: str, ctx: Context = None, days: int = 90) -> dict:
    """Find the most recent SSO sign-in to an application.

    One System Log query for ``user.authentication.sso`` events targeting the
    app in the last ``days`` days, newest first, limit 1. Use it to tell an
    app nobody has touched in months from one that is quietly in daily use.
    A 429 from Okta is retried with backoff.

    Parameters:
        app_id (str, required): The application ID (starts with 0oa).
        days (int, optional): How far back to look. Default: 90. Okta keeps
            System Log data for 90 days, so larger values add nothing.

    Returns:
        Dict containing:
        - app_id, days, since: the query that was run
        - last_sso: ISO 8601 timestamp of the newest SSO event, or null if none
          in the window
        - event: compact details of that event (actor, outcome, client IP), or null
    """
    logger.info(f"Looking up last SSO event for application {app_id} over {days} days")
    manager = ctx.request_context.lifespan_context.okta_auth_manager

    if days < 1:
        return {"error": "days must be at least 1"}

    since = _iso_utc(datetime.now(timezone.utc) - timedelta(days=days))
    params = {
        "since": since,
        "filter": f'eventType eq "user.authentication.sso" and target.id eq "{app_id}"',
        "limit": 1,
        # System Log defaults to oldest-first; we want the newest event.
        "sortOrder": "DESCENDING",
    }

    try:
        client = await get_okta_client(manager)
        events, _, err = await raw_get(client, build_url("/api/v1/logs", params))
        if err:
            logger.error(f"Okta API error while querying SSO events for {app_id}: {err}")
            return {"error": str(err)}

        events = events if isinstance(events, list) else []
        if not events or not isinstance(events[0], dict):
            logger.info(f"No SSO events for {app_id} since {since}")
            return {"app_id": app_id, "days": days, "since": since, "last_sso": None, "event": None}

        newest = events[0]
        logger.info(f"Last SSO for {app_id}: {newest.get('published')}")
        return {
            "app_id": app_id,
            "days": days,
            "since": since,
            "last_sso": newest.get("published"),
            "event": _compact_log_event(newest),
        }
    except Exception as e:
        logger.error(f"Exception while querying SSO events for {app_id}: {type(e).__name__}: {e}")
        return {"error": str(e)}
