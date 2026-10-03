"""Unit tests for ``@<runner>`` ACP mention parsing / resolution."""

from __future__ import annotations

from octop.infra.gateway.process.acp_delegate import (
    ACP_DELEGATE_KEY,
    parse_acp_mention,
    resolve_acp_delegate,
)


def test_delegate_key_is_stable() -> None:
    assert ACP_DELEGATE_KEY == "octop_acp_delegate"


def test_parse_leading_mention_with_cjk_task() -> None:
    assert parse_acp_mention("@opencode 修复登录bug") == ("opencode", "修复登录bug")


def test_parse_strips_surrounding_whitespace() -> None:
    assert parse_acp_mention("  @claude_code fix the bug  ") == (
        "claude_code",
        "fix the bug",
    )


def test_parse_dotted_runner_name() -> None:
    assert parse_acp_mention("@my.runner-1 do it") == ("my.runner-1", "do it")


def test_parse_rejects_mid_sentence_mention() -> None:
    assert parse_acp_mention("please ask @opencode to help") is None


def test_parse_rejects_bare_mention() -> None:
    assert parse_acp_mention("@opencode") is None
    assert parse_acp_mention("@opencode   ") is None


def test_parse_rejects_empty_and_non_mention() -> None:
    assert parse_acp_mention("") is None
    assert parse_acp_mention("hello world") is None
    assert parse_acp_mention("@") is None


def test_resolve_returns_marker_for_enabled_runner() -> None:
    assert resolve_acp_delegate(
        text="@opencode 修复登录bug",
        enabled_runner_names=["opencode", "codex"],
        tool_enabled=True,
    ) == {"runner": "opencode", "task": "修复登录bug"}


def test_resolve_passes_through_when_tool_disabled() -> None:
    assert (
        resolve_acp_delegate(
            text="@opencode fix it",
            enabled_runner_names=["opencode"],
            tool_enabled=False,
        )
        is None
    )


def test_resolve_passes_through_for_unknown_runner() -> None:
    assert (
        resolve_acp_delegate(
            text="@unknown fix it",
            enabled_runner_names=["opencode"],
            tool_enabled=True,
        )
        is None
    )


def test_resolve_passes_through_for_no_runners() -> None:
    assert (
        resolve_acp_delegate(
            text="@opencode fix it",
            enabled_runner_names=[],
            tool_enabled=True,
        )
        is None
    )


def test_resolve_case_insensitive_fallback_uses_canonical_name() -> None:
    assert resolve_acp_delegate(
        text="@OpenCode fix it",
        enabled_runner_names=["opencode"],
        tool_enabled=True,
    ) == {"runner": "opencode", "task": "fix it"}


def test_resolve_bare_mention_passes_through() -> None:
    assert (
        resolve_acp_delegate(
            text="@opencode",
            enabled_runner_names=["opencode"],
            tool_enabled=True,
        )
        is None
    )
