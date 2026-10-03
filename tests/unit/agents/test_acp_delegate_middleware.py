"""Unit tests for the ``acp-delegate`` plugin middleware and manager capability lookup.

The directive-injection middleware ships as a local hook plugin
(``local-plugins/acp-delegate/``); these tests load it directly from the
plugin source.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from octop.config import OctopConfig
from octop.infra.agents.manager import AgentManager
from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.services import build_shared_services
from octop.infra.utils.paths import PathLayout

_PLUGIN_DIR = Path(__file__).resolve().parents[3] / "local-plugins" / "acp-delegate"


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "acp_delegate_plugin_main", _PLUGIN_DIR / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acp_delegate_mw = _load_plugin_module()


def _request(
    *messages: object,
    system: SystemMessage | None = None,
) -> ModelRequest:
    return ModelRequest(
        model=ChatOpenAI(model="gpt-4o-mini", api_key="test"),
        messages=list(messages),  # type: ignore[arg-type]
        system_message=system,
    )


def _marked_human(text: str, runner: str, task: str) -> HumanMessage:
    return HumanMessage(
        content=text,
        additional_kwargs={
            acp_delegate_mw.ACP_DELEGATE_KEY: {"runner": runner, "task": task},
        },
    )


def _capture(request: ModelRequest) -> ModelRequest:
    return request


def test_middleware_injects_directive_for_marked_human_message() -> None:
    mw = acp_delegate_mw.AcpDelegateMiddleware()
    request = _request(_marked_human("@opencode 修复登录bug", "opencode", "修复登录bug"))

    out = mw.wrap_model_call(request, _capture)

    assert out.system_message is not None
    text = str(out.system_message.content)
    assert '"opencode"' in text
    assert 'action="start"' in text
    assert "[permission_required]" in text


async def test_middleware_async_injects_directive() -> None:
    mw = acp_delegate_mw.AcpDelegateMiddleware()
    request = _request(_marked_human("@opencode fix it", "opencode", "fix it"))

    async def acapture(req: ModelRequest) -> ModelRequest:
        return req

    out = await mw.awrap_model_call(request, acapture)

    assert out.system_message is not None


def test_middleware_passthrough_without_marker() -> None:
    mw = acp_delegate_mw.AcpDelegateMiddleware()
    request = _request(HumanMessage(content="plain message"))

    out = mw.wrap_model_call(request, _capture)

    assert out.system_message is None


def test_middleware_ignores_older_marker_when_newest_human_clean() -> None:
    mw = acp_delegate_mw.AcpDelegateMiddleware()
    request = _request(
        _marked_human("@opencode fix it", "opencode", "fix it"),
        AIMessage(content="done"),
        HumanMessage(content="thanks, now do X"),
    )

    out = mw.wrap_model_call(request, _capture)

    assert out.system_message is None


def test_middleware_merges_with_existing_system_message() -> None:
    mw = acp_delegate_mw.AcpDelegateMiddleware()
    request = _request(
        _marked_human("@codex fix it", "codex", "fix it"),
        system=SystemMessage(content="Base prompt."),
    )

    out = mw.wrap_model_call(request, _capture)

    assert out.system_message is not None
    text = str(out.system_message.content)
    assert text.startswith("Base prompt.")
    assert '"codex"' in text


def test_middleware_dedupes_repeated_hint() -> None:
    mw = acp_delegate_mw.AcpDelegateMiddleware()
    request = _request(_marked_human("@codex fix it", "codex", "fix it"))
    once = mw.wrap_model_call(request, _capture)
    assert once.system_message is not None

    twice = mw.wrap_model_call(once, _capture)

    assert str(twice.system_message.content) == str(once.system_message.content)


# ---------------------------------------------------------------------------
# Tool-call guard: ACPSessionError must not crash the turn
# ---------------------------------------------------------------------------


def _tool_request(name: str, call_id: str = "call1") -> Any:
    from langgraph.prebuilt.tool_node import ToolCallRequest

    return ToolCallRequest(
        tool_call={"name": name, "args": {"action": "message"}, "id": call_id},
        tool=None,
        state=None,
        runtime=None,
    )


def test_tool_guard_converts_session_error_to_tool_message() -> None:
    from octop_harness.acp.models import ACPSessionError

    mw = acp_delegate_mw.AcpDelegateMiddleware()

    def boom(req: Any) -> ToolMessage:
        raise ACPSessionError("no bound ACP session found for runner 'x'")

    out = mw.wrap_tool_call(_tool_request("acp_runner"), boom)

    assert isinstance(out, ToolMessage)
    assert out.tool_call_id == "call1"
    assert 'action="start"' in str(out.content)


async def test_tool_guard_async_converts_session_error() -> None:
    from octop_harness.acp.models import ACPSessionError

    mw = acp_delegate_mw.AcpDelegateMiddleware()

    async def aboom(req: Any) -> ToolMessage:
        raise ACPSessionError("no bound ACP session found")

    out = await mw.awrap_tool_call(_tool_request("acp_runner"), aboom)

    assert isinstance(out, ToolMessage)
    assert 'action="start"' in str(out.content)


def test_tool_guard_propagates_for_other_tools() -> None:
    from octop_harness.acp.models import ACPSessionError

    mw = acp_delegate_mw.AcpDelegateMiddleware()

    def boom(req: Any) -> ToolMessage:
        raise ACPSessionError("no bound ACP session found")

    with pytest.raises(ACPSessionError):
        mw.wrap_tool_call(_tool_request("shell"), boom)


def test_directive_warns_about_in_memory_sessions() -> None:
    text = acp_delegate_mw._directive("opencode")
    assert "in-memory" in text


# ---------------------------------------------------------------------------
# AgentManager.acp_delegate_capabilities
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> AgentManager:
    paths = PathLayout(tmp_path / ".octop")
    paths.ensure_root()
    db = SqlitePool(paths.db)
    run_migrations(db)
    services = build_shared_services(db=db, paths=paths, config=OctopConfig())
    return AgentManager(repos=services.repos, paths=services.paths)


def _save_runners(manager: AgentManager, user_id: int) -> None:
    manager.acp_settings.save_runners(
        user_id,
        {
            "opencode": {
                "enabled": True,
                "command": "opencode",
                "args": ["acp"],
                "env": {},
                "trusted": True,
                "tool_parse_mode": "update_detail",
                "stdio_buffer_limit_bytes": 52428800,
            },
            "codex": {
                "enabled": False,
                "command": "npx",
                "args": ["-y", "@zed-industries/codex-acp"],
                "env": {},
                "trusted": False,
                "tool_parse_mode": "update_detail",
                "stdio_buffer_limit_bytes": 52428800,
            },
        },
    )


def test_capabilities_none_for_unknown_agent(manager: AgentManager) -> None:
    assert manager.acp_delegate_capabilities("missing-agent") is None


def test_capabilities_reflect_owner_runners_and_toggle(
    manager: AgentManager,
) -> None:
    uid = manager._repos.user_repo.create(
        username="owner", password_hash="h", role="user"
    )
    _save_runners(manager, uid)
    manager._repos.agent_repo.create(
        agent_id="bot1",
        user_id=uid,
        name="bot",
        config_json=json.dumps({"acp": {"tool_enabled": True}}),
    )

    enabled, tool_enabled = manager.acp_delegate_capabilities("bot1") or ([], False)

    assert enabled == ["opencode"]
    assert tool_enabled is True


def test_capabilities_toggle_off(manager: AgentManager) -> None:
    uid = manager._repos.user_repo.create(
        username="owner", password_hash="h", role="user"
    )
    _save_runners(manager, uid)
    manager._repos.agent_repo.create(
        agent_id="bot2",
        user_id=uid,
        name="bot",
        config_json=json.dumps({"acp": {"tool_enabled": False}}),
    )

    enabled, tool_enabled = manager.acp_delegate_capabilities("bot2") or ([], False)

    assert enabled == ["opencode"]
    assert tool_enabled is False


def test_capabilities_team_agent_zeroed(manager: AgentManager) -> None:
    uid = manager._repos.user_repo.create(
        username="owner", password_hash="h", role="user"
    )
    _save_runners(manager, uid)
    manager._repos.agent_repo.create(
        agent_id="team1",
        user_id=uid,
        name="team",
        kind="team",
        config_json=json.dumps({"acp": {"tool_enabled": True}}),
    )

    assert manager.acp_delegate_capabilities("team1") == ([], False)


# ---------------------------------------------------------------------------
# Plugin loads through the harness plugin loader
# ---------------------------------------------------------------------------


def test_plugin_loads_via_harness_loader() -> None:
    from octop_harness.plugins import PluginRegistry, load_plugin_dir

    PluginRegistry().clear()
    try:
        loaded = load_plugin_dir(_PLUGIN_DIR, install_deps=False)
        assert loaded.manifest.id == "acp-delegate"
        assert loaded.manifest.kind == "hook"
        middleware = PluginRegistry().build_middleware_chain()
        assert any(type(m).__name__ == "AcpDelegateMiddleware" for m in middleware)
    finally:
        PluginRegistry().clear()
