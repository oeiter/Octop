"""Parse ``@<runner>`` mentions in dashboard chat turns for ACP delegation.

A dashboard message that *starts* with ``@<enabled_runner_name>`` (e.g.
``@opencode 修复登录bug``) is delegated to that ACP runner for this turn
only: :func:`resolve_acp_delegate` produces a marker that travels through
``InboundMessage`` metadata into ``HumanMessage.additional_kwargs`` and is
turned into a system directive by ``AcpDelegateMiddleware`` at model-call
time. Messages without a matching prefix reach the Octop agent untouched.
"""

from __future__ import annotations

import re

__all__ = ["ACP_DELEGATE_KEY", "parse_acp_mention", "resolve_acp_delegate"]

# Persisted on HumanMessage.additional_kwargs; read by AcpDelegateMiddleware
# (the acp-delegate hook plugin) at model-call time to inject the per-turn
# ACP delegation directive.
ACP_DELEGATE_KEY = "octop_acp_delegate"

# ``@name`` must be the first token; name chars mirror runner slug style.
_ACP_MENTION_RE = re.compile(r"^@([A-Za-z0-9_][A-Za-z0-9_.-]*)(?:\s+(.+))?$", re.DOTALL)


def parse_acp_mention(text: str) -> tuple[str, str] | None:
    """Return ``(runner, task)`` when *text* starts with ``@<name> <task>``.

    A bare ``@name`` with no task returns ``None`` — the harness tool would
    send a default ``"hi"`` prompt to the external agent.
    """
    match = _ACP_MENTION_RE.match((text or "").strip())
    if match is None:
        return None
    task = (match.group(2) or "").strip()
    if not task:
        return None
    return match.group(1), task


def resolve_acp_delegate(
    *,
    text: str,
    enabled_runner_names: list[str] | frozenset[str] | tuple[str, ...],
    tool_enabled: bool,
) -> dict[str, str] | None:
    """Resolve an ``@<runner>`` mention against the agent's ACP capabilities.

    Returns ``{"runner": ..., "task": ...}`` or ``None`` to pass the message
    through untouched (unknown/disabled runner, bare mention, tool off).
    """
    if not tool_enabled or not enabled_runner_names:
        return None
    parsed = parse_acp_mention(text)
    if parsed is None:
        return None
    runner, task = parsed
    names = [str(name) for name in enabled_runner_names if str(name)]
    if runner not in names:
        lowered = runner.lower()
        matches = [name for name in names if name.lower() == lowered]
        if len(matches) != 1:
            return None
        runner = matches[0]
    return {"runner": runner, "task": task}
