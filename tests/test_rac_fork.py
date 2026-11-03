# The Okta software accompanied by this notice is provided pursuant to the following terms:
# Copyright © 2025-Present, Okta, Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Tests for the RA Capital fork additions.

Covers the compact serializers, the raw-request pagination helper, the four
audit tools, the ``compact`` flag on the existing list tools, and the partial
fetch_all result shape.  All Okta traffic is mocked; nothing here touches a
real org.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from okta_mcp_server.utils.compact import (
    compact_app_group,
    compact_app_user,
    compact_application,
    compact_group,
)
from okta_mcp_server.utils.pagination import create_paginated_response, paginate_all_results
from okta_mcp_server.utils.raw_request import build_url, raw_get, raw_get_pages
from okta_mcp_server.utils.scope_registry import TOOL_SCOPE_REGISTRY
from tests.conftest import FakeLifespanContext, FakeOktaAuthManager, load_fixture

APP_ID = "0oats7mabxlQsIbxH4x7"


def _ctx():
    request_context = MagicMock()
    request_context.lifespan_context = FakeLifespanContext(okta_auth_manager=FakeOktaAuthManager())
    ctx = MagicMock()
    ctx.request_context = request_context
    return ctx


def _response(status=200, link_next=None, headers=None):
    """Fake aiohttp-style response: ``.status`` and a ``.headers`` with ``.get``."""
    # spec keeps MagicMock from inventing a has_next(), which would route
    # paginate_all_results down the SDK v2 branch.
    resp = MagicMock(spec=["status", "headers"])
    resp.status = status
    hdrs = dict(headers or {})
    if link_next:
        hdrs["Link"] = f'<{link_next}>; rel="next"'
    resp.headers = hdrs
    return resp


def _client_with_pages(pages):
    """Build a fake Okta client whose executor serves ``pages`` in order.

    Each entry is ``(status, body_json_str, link_next_url_or_None, error_or_None)``.
    """
    import json

    executor = MagicMock()
    executor.create_request = AsyncMock(return_value=({"url": "x"}, None))
    results = []
    for status, body, link_next, err in pages:
        results.append((_response(status, link_next), body if isinstance(body, str) else json.dumps(body), err))
    executor.execute = AsyncMock(side_effect=results)
    client = MagicMock()
    client.get_request_executor.return_value = executor
    return client, executor


# ---------------------------------------------------------------------------
# compact.py
# ---------------------------------------------------------------------------

def test_compact_application_keeps_only_inventory_fields():
    app = load_fixture("saml_app_signon_fully_populated.json")
    app["visibility"] = {"hide": {"iOS": False, "web": True}, "autoLaunch": False}

    out = compact_application(app)

    assert out["id"] == app["id"]
    assert out["label"] == app["label"]
    assert out["signOnMode"] == "SAML_2_0"
    assert out["visibility"] == {"hide": {"iOS": False, "web": True}}
    assert "settings" not in out  # only bookmarks carry settings
    assert "_links" not in out and "credentials" not in out


def test_compact_application_bookmark_keeps_url_and_parse_error():
    app = {
        "id": "0oa1", "label": "Payroll", "name": "bookmark", "signOnMode": "BOOKMARK",
        "status": "ACTIVE", "settings": {"app": {"url": "https://payroll.example.invalid", "requestIntegration": False}},
        "parse_error": "Application: boom",
    }
    out = compact_application(app)
    assert out["settings"] == {"app": {"url": "https://payroll.example.invalid"}}
    assert out["parse_error"] == "Application: boom"


def test_compact_group_trims_profile():
    out = compact_group({"id": "00g1", "type": "OKTA_GROUP", "profile": {"name": "IT", "description": "d", "samAccountName": "x"},
                         "_links": {}, "created": "2020-01-01T00:00:00.000Z"})
    assert out == {"id": "00g1", "type": "OKTA_GROUP", "created": "2020-01-01T00:00:00.000Z",
                   "profile": {"name": "IT", "description": "d"}}


def test_compact_app_user_and_group_surface_embedded_names():
    user = {"id": "00u1", "scope": "GROUP", "status": "ACTIVE", "credentials": {"userName": "j@example.invalid", "password": {}},
            "profile": {"big": "blob"}, "_embedded": {"user": {"profile": {"login": "j@example.invalid"}}}}
    out = compact_app_user(user)
    assert out["scope"] == "GROUP"
    assert out["username"] == "j@example.invalid"
    assert out["login"] == "j@example.invalid"
    assert "profile" not in out and "credentials" not in out

    group = {"id": "00g1", "priority": 0, "profile": {},
             "_embedded": {"group": {"type": "OKTA_GROUP", "profile": {"name": "Everyone"}}}}
    assert compact_app_group(group) == {"id": "00g1", "priority": 0, "name": "Everyone", "type": "OKTA_GROUP"}
    # No embedded group: name/type are simply absent so the caller can tell.
    assert compact_app_group({"id": "00g2", "priority": 1}) == {"id": "00g2", "priority": 1}


def test_compact_app_user_is_small():
    """rac.2: only id, scope, status, username, created, lastUpdated survive."""
    user = {"id": "00u1", "scope": "USER", "status": "ACTIVE", "syncState": "DISABLED",
            "created": "2025-01-01T00:00:00.000Z", "lastUpdated": "2025-02-01T00:00:00.000Z",
            "statusChanged": "2025-01-01T00:00:00.000Z", "passwordChanged": None,
            "credentials": {"userName": "a@example.invalid"}, "profile": {}, "_links": {"x": 1}}
    assert compact_app_user(user) == {
        "id": "00u1", "scope": "USER", "status": "ACTIVE", "username": "a@example.invalid",
        "created": "2025-01-01T00:00:00.000Z", "lastUpdated": "2025-02-01T00:00:00.000Z",
    }


# ---------------------------------------------------------------------------
# raw_request.py
# ---------------------------------------------------------------------------

def test_build_url_skips_empty_params():
    assert build_url("/api/v1/x", {"limit": 200, "after": None, "expand": ""}) == "/api/v1/x?limit=200"
    assert build_url("/api/v1/x") == "/api/v1/x"


@pytest.mark.asyncio
async def test_raw_get_pages_follows_link_header_and_keeps_partial_on_error():
    client, executor = _client_with_pages([
        (200, [{"id": "1"}], "https://org.okta.com/api/v1/apps/x/users?after=c1", None),
        (200, [{"id": "2"}], "https://org.okta.com/api/v1/apps/x/users?after=c2", None),
        (500, "", None, Exception("boom")),
    ])

    items, info, err = await raw_get_pages(client, "/api/v1/apps/x/users", {"limit": 500}, delay_between_requests=0)

    assert err is None
    assert [i["id"] for i in items] == ["1", "2"]
    assert info["pages_fetched"] == 2
    assert info["stopped_early"] is True
    assert info["errors"][0]["after"] == "c2"
    assert info["resume_cursor"] == "c2" and info["next_cursor"] == "c2" and info["has_more"] is True


@pytest.mark.asyncio
async def test_raw_get_pages_first_page_error_is_returned():
    client, _ = _client_with_pages([(403, "", None, Exception("forbidden"))])
    items, info, err = await raw_get_pages(client, "/api/v1/apps/x/users")
    assert items == [] and str(err) == "forbidden"


@pytest.mark.asyncio
async def test_raw_get_retries_on_429(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("okta_mcp_server.utils.raw_request.asyncio.sleep", fake_sleep)
    client, executor = _client_with_pages([
        (429, "", None, Exception("E0000047 rate limit")),
        (200, [{"published": "2026-10-01T00:00:00.000Z"}], None, None),
    ])

    body, resp, err = await raw_get(client, "/api/v1/logs?limit=1")

    assert err is None and body[0]["published"].startswith("2026-10")
    assert executor.execute.await_count == 2
    assert sleeps == [1.0]  # 2**0 fallback when no X-Rate-Limit-Reset header


# ---------------------------------------------------------------------------
# pagination.py: partial fetch_all
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paginate_all_results_records_errors_and_resume_cursor():
    first = _response(link_next="https://org.okta.com/api/v1/apps?after=c1")

    async def next_page(cursor):
        return None, None, Exception("page blew up")

    items, info = await paginate_all_results(first, [{"id": "a"}], next_page_fn=next_page, delay_between_requests=0)

    assert items == [{"id": "a"}]
    assert info["errors"] == [{"page": 2, "error": "API error: page blew up", "after": "c1"}]
    assert info["resume_cursor"] == "c1"

    result = create_paginated_response(items, first, fetch_all_used=True, pagination_info=info)
    assert result["errors"] == info["errors"]
    assert result["has_more"] is True and result["next_cursor"] == "c1"


@pytest.mark.asyncio
async def test_paginate_all_results_clean_walk_has_empty_errors():
    first = _response(link_next="https://org.okta.com/api/v1/apps?after=c1")

    async def next_page(cursor):
        return [{"id": "b"}], _response(), None

    items, info = await paginate_all_results(first, [{"id": "a"}], next_page_fn=next_page, delay_between_requests=0)
    assert len(items) == 2 and info["errors"] == [] and "resume_cursor" not in info
    result = create_paginated_response(items, first, fetch_all_used=True, pagination_info=info)
    assert "errors" not in result and result["has_more"] is False


# ---------------------------------------------------------------------------
# audit tools
# ---------------------------------------------------------------------------

def test_audit_tools_are_in_scope_registry_with_read_scopes():
    assert TOOL_SCOPE_REGISTRY["list_user_app_links"] == "okta.users.read"
    assert TOOL_SCOPE_REGISTRY["list_application_users"] == "okta.apps.read"
    assert TOOL_SCOPE_REGISTRY["list_application_groups"] == "okta.apps.read"
    assert TOOL_SCOPE_REGISTRY["app_last_sso"] == "okta.logs.read"
    # Nothing in the fork may require a manage scope.
    for name in ("list_user_app_links", "list_application_users", "list_application_groups", "app_last_sso"):
        assert TOOL_SCOPE_REGISTRY[name].endswith(".read")


@pytest.mark.asyncio
async def test_list_user_app_links_returns_items():
    from okta_mcp_server.tools.audit.audit import list_user_app_links

    links = [{"id": "l1", "label": "Slack", "appInstanceId": APP_ID, "appName": "slack", "hidden": False}]
    client, executor = _client_with_pages([(200, links, None, None)])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await list_user_app_links(user_id="jdemaio@example.invalid", ctx=_ctx())

    assert result["total_fetched"] == 1 and result["items"][0]["appInstanceId"] == APP_ID
    url = executor.create_request.await_args.kwargs["url"]
    assert url == "/api/v1/users/jdemaio@example.invalid/appLinks"


@pytest.mark.asyncio
async def test_list_application_users_counts_scopes_and_compacts():
    from okta_mcp_server.tools.audit.audit import list_application_users

    users = [
        {"id": "00u1", "scope": "GROUP", "status": "ACTIVE", "profile": {"x": 1}, "credentials": {"userName": "a"}},
        {"id": "00u2", "scope": "USER", "status": "ACTIVE", "profile": {"x": 2}},
        {"id": "00u3", "scope": "GROUP", "status": "ACTIVE"},
    ]
    client, executor = _client_with_pages([(200, users, None, None)])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await list_application_users(app_id=APP_ID, ctx=_ctx())

    assert result["scope_counts"] == {"GROUP": 2, "USER": 1}
    assert result["total_fetched"] == 3
    assert result["items"][0] == {"id": "00u1", "scope": "GROUP", "status": "ACTIVE", "username": "a"}
    assert "profile" not in result["items"][1]
    assert executor.create_request.await_args.kwargs["url"] == f"/api/v1/apps/{APP_ID}/users?limit=500"


@pytest.mark.asyncio
async def test_list_application_groups_full_objects_when_not_compact():
    from okta_mcp_server.tools.audit.audit import list_application_groups

    groups = [{"id": "00g1", "priority": 0, "profile": {}, "_links": {"self": {"href": "x"}}}]
    client, executor = _client_with_pages([(200, groups, None, None)])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await list_application_groups(app_id=APP_ID, ctx=_ctx(), compact=False, expand="group")

    assert result["items"] == groups
    assert executor.create_request.await_args.kwargs["url"] == f"/api/v1/apps/{APP_ID}/groups?limit=200&expand=group"


@pytest.mark.asyncio
async def test_app_last_sso_returns_newest_event_and_null_when_none():
    from okta_mcp_server.tools.audit.audit import app_last_sso

    event = {"published": "2026-10-29T12:00:00.000Z", "eventType": "user.authentication.sso",
             "actor": {"alternateId": "j@example.invalid", "displayName": "J"}, "outcome": {"result": "SUCCESS"},
             "client": {"ipAddress": "203.0.113.5"}, "target": [{"id": APP_ID}]}
    client, executor = _client_with_pages([(200, [event], None, None)])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await app_last_sso(app_id=APP_ID, ctx=_ctx(), days=30)

    assert result["last_sso"] == "2026-10-29T12:00:00.000Z"
    assert result["event"]["actor"]["alternateId"] == "j@example.invalid"
    url = executor.create_request.await_args.kwargs["url"]
    assert url.startswith("/api/v1/logs?since=")
    assert "limit=1" in url and "sortOrder=DESCENDING" in url
    assert "user.authentication.sso" in url and APP_ID in url

    client, _ = _client_with_pages([(200, [], None, None)])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await app_last_sso(app_id=APP_ID, ctx=_ctx())
    assert result["last_sso"] is None and result["event"] is None and result["days"] == 90


# ---------------------------------------------------------------------------
# compact flag on existing list tools
# ---------------------------------------------------------------------------

def _app_model(app_id, label):
    app = MagicMock()
    app.to_dict.return_value = {"id": app_id, "label": label, "signOnMode": "SAML_2_0", "status": "ACTIVE",
                                "settings": {"signOn": {"big": "blob"}}, "_links": {}}
    return app


@pytest.mark.asyncio
async def test_list_applications_compact_default_and_opt_out():
    from okta_mcp_server.tools.applications.applications import list_applications

    apps = [_app_model("0oa1", "One"), {"id": "0oa2", "label": "Broken", "parse_error": "Application: nope",
                                        "settings": {"x": 1}}]
    client = AsyncMock()
    client.list_applications.return_value = (apps, _response(), None)
    with patch("okta_mcp_server.tools.applications.applications.get_okta_client", AsyncMock(return_value=client)):
        compact = await list_applications(ctx=_ctx())
        full = await list_applications(ctx=_ctx(), compact=False)

    assert compact["items"][0] == {"id": "0oa1", "label": "One", "signOnMode": "SAML_2_0", "status": "ACTIVE"}
    assert compact["items"][1] == {"id": "0oa2", "label": "Broken", "parse_error": "Application: nope"}
    assert full["items"][0]["settings"] == {"signOn": {"big": "blob"}}


@pytest.mark.asyncio
async def test_get_application_falls_back_to_raw_json_on_validation_error():
    from pydantic import BaseModel

    from okta_mcp_server.tools.applications.applications import get_application

    class _M(BaseModel):
        label: str

    try:
        _M()  # missing label -> real pydantic ValidationError
    except Exception as exc:  # noqa: BLE001
        validation_error = exc

    client, executor = _client_with_pages([(200, {"id": APP_ID, "label": "Wisprflow", "features": ["SCIM_PROVISIONING"]}, None, None)])
    client.get_application = AsyncMock(side_effect=validation_error)
    with patch("okta_mcp_server.tools.applications.applications.get_okta_client", AsyncMock(return_value=client)):
        result = await get_application(ctx=_ctx(), app_id=APP_ID)

    assert result["label"] == "Wisprflow" and result["features"] == ["SCIM_PROVISIONING"]
    assert result["parse_error"].startswith("Application:") and "label" in result["parse_error"]
    assert executor.create_request.await_args.kwargs["url"] == f"/api/v1/apps/{APP_ID}"


def test_compact_group_unwraps_sdk_any_of_profile():
    dumped = {"id": "00g1", "type": "BUILT_IN",
              "profile": {"actual_instance": {"name": "Everyone", "description": "All", "additional_properties": {}},
                          "any_of_schemas": ["OktaActiveDirectoryGroupProfile", "OktaUserGroupProfile"]}}
    assert compact_group(dumped)["profile"] == {"name": "Everyone", "description": "All"}


@pytest.mark.asyncio
async def test_list_user_groups_compact_default():
    from okta_mcp_server.tools.user_resources.user_resources import list_user_groups

    group = MagicMock()
    group.to_dict.return_value = {"id": "00g1", "type": "OKTA_GROUP", "profile": {"name": "IT", "extra": 1}, "_links": {}}
    client = AsyncMock()
    client.list_user_groups.return_value = ([group], MagicMock(), None)
    with patch("okta_mcp_server.tools.user_resources.user_resources.get_okta_client", AsyncMock(return_value=client)):
        result = await list_user_groups(user_id="00u1", ctx=_ctx())

    assert result == [{"id": "00g1", "type": "OKTA_GROUP", "profile": {"name": "IT"}}]


# ---------------------------------------------------------------------------
# rac.2 fixes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_applications_limit_flows_through_to_okta():
    """limit=2 must reach the Okta query as 2 (upstream raised anything under 20 to 20)."""
    from okta_mcp_server.tools.applications.applications import list_applications

    client = AsyncMock()
    client.list_applications.return_value = ([_app_model("0oa1", "One"), _app_model("0oa2", "Two")], _response(), None)
    with patch("okta_mcp_server.tools.applications.applications.get_okta_client", AsyncMock(return_value=client)):
        result = await list_applications(ctx=_ctx(), limit=2)
        assert client.list_applications.await_args.kwargs["limit"] == 2
        assert result["total_fetched"] == 2

        await list_applications(ctx=_ctx(), limit=999)
        assert client.list_applications.await_args.kwargs["limit"] == 200

        # No limit + fetch_all: biggest page so the walk takes the fewest calls.
        await list_applications(ctx=_ctx(), fetch_all=True)
        assert client.list_applications.await_args.kwargs["limit"] == 200

        # No limit, single page: leave it to Okta's default.
        await list_applications(ctx=_ctx())
        assert "limit" not in client.list_applications.await_args.kwargs


@pytest.mark.asyncio
async def test_list_applications_include_raw_switches_warning_payloads():
    from okta_mcp_server.tools.applications import applications as apps_mod

    client = AsyncMock()
    client.list_applications.return_value = ([_app_model("0oa1", "One")], _response(), None)
    with patch.object(apps_mod, "get_okta_client", AsyncMock(return_value=client)), \
         patch.object(apps_mod, "set_include_raw_items") as switch:
        await apps_mod.list_applications(ctx=_ctx())
        switch.assert_called_with(False)
        await apps_mod.list_applications(ctx=_ctx(), include_raw=True)
        switch.assert_called_with(True)


@pytest.mark.asyncio
async def test_list_group_apps_include_raw_switch():
    from okta_mcp_server.tools.groups import groups as groups_mod

    client = AsyncMock()
    client.list_assigned_applications_for_group.return_value = ([_app_model("0oa1", "One")], _response(), None)
    with patch.object(groups_mod, "get_okta_client", AsyncMock(return_value=client)), \
         patch.object(groups_mod, "set_include_raw_items") as switch:
        await groups_mod.list_group_apps(group_id="00g00000000000000001", ctx=_ctx(), include_raw=True)
        switch.assert_called_with(True)


@pytest.mark.asyncio
async def test_list_application_groups_resolves_names_from_embedded_group():
    """Default expand=group: names and types come back in the same response, no extra calls."""
    from okta_mcp_server.tools.audit.audit import list_application_groups

    groups = [
        {"id": "00g1", "priority": 0, "lastUpdated": "2025-01-01T00:00:00.000Z", "profile": {},
         "_embedded": {"group": {"id": "00g1", "type": "OKTA_GROUP", "profile": {"name": "Slack Users"}, "_links": {}}}},
        {"id": "00g2", "priority": 1, "lastUpdated": "2025-01-02T00:00:00.000Z", "profile": {},
         "_embedded": {"group": {"id": "00g2", "type": "APP_GROUP", "profile": {"name": "Tech Atlas"}}}},
    ]
    client, executor = _client_with_pages([(200, groups, None, None)])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await list_application_groups(app_id=APP_ID, ctx=_ctx())

    assert executor.create_request.await_args.kwargs["url"] == f"/api/v1/apps/{APP_ID}/groups?limit=200&expand=group"
    assert executor.create_request.await_count == 1
    assert result["items"] == [
        {"id": "00g1", "priority": 0, "lastUpdated": "2025-01-01T00:00:00.000Z", "name": "Slack Users", "type": "OKTA_GROUP"},
        {"id": "00g2", "priority": 1, "lastUpdated": "2025-01-02T00:00:00.000Z", "name": "Tech Atlas", "type": "APP_GROUP"},
    ]
    assert "errors" not in result


@pytest.mark.asyncio
async def test_list_application_groups_falls_back_to_get_group_per_id():
    """Without an embedded group, each unresolved row costs one GET /groups/{id}; failures are reported, not fatal."""
    from okta_mcp_server.tools.audit.audit import list_application_groups

    groups = [{"id": "00g1", "priority": 0}, {"id": "00g2", "priority": 1}]
    client, executor = _client_with_pages([
        (200, groups, None, None),
        (200, {"id": "00g1", "type": "OKTA_GROUP", "profile": {"name": "Structured Capital", "description": ""}}, None, None),
        (404, "", None, "Not found"),
    ])
    with patch("okta_mcp_server.tools.audit.audit.get_okta_client", AsyncMock(return_value=client)):
        result = await list_application_groups(app_id=APP_ID, ctx=_ctx(), expand=None)

    urls = [c.kwargs["url"] for c in executor.create_request.await_args_list]
    assert urls == [f"/api/v1/apps/{APP_ID}/groups?limit=200", "/api/v1/groups/00g1", "/api/v1/groups/00g2"]
    assert result["items"][0] == {"id": "00g1", "priority": 0, "name": "Structured Capital", "type": "OKTA_GROUP"}
    assert result["items"][1] == {"id": "00g2", "priority": 1}
    assert result["errors"] == [{"group_id": "00g2", "error": "Not found"}]


def test_extract_after_cursor_reads_every_link_header():
    """Okta sends self and next as separate Link headers; aiohttp .get() only returns the first."""
    from multidict import CIMultiDict, CIMultiDictProxy

    from okta_mcp_server.utils.pagination import extract_after_cursor

    headers = CIMultiDict()
    headers.add("Link", '<https://example.invalid/api/v1/apps?limit=2>; rel="self"')
    headers.add("Link", '<https://example.invalid/api/v1/apps?after=0oaNEXT&limit=2>; rel="next"')
    resp = MagicMock(spec=["status", "headers"])
    resp.status = 200
    resp.headers = CIMultiDictProxy(headers)

    assert extract_after_cursor(resp) == "0oaNEXT"
