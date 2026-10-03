"""ACP delegation directive injection (hook plugin).

The dashboard stamps ``octop_acp_delegate`` onto the turn's
``HumanMessage.additional_kwargs`` when the user message starts with
``@<enabled_runner_name>`` (see ``octop.infra.gateway.process.acp_delegate``
on the host side). This plugin turns that marker into a system directive so
the Octop agent delegates the turn to the external ACP runner via the
existing ``acp_runner`` tool. The marker itself is never sent to the
provider; the directive lives only in ``system_message`` for the model
calls of the turn that carries it.

Wire protocol note: the key is a stable contract between host and plugin,
so it is intentionally hardcoded here instead of imported — the plugin
stays self-contained even if the host refactors its internals.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

try:  # optional extra: only available with octop-harness[acp] installed
    from octop_harness.acp.models import ACPSessionError
except ImportError:  # pragma: no cover
    ACPSessionError = None  # type: ignore[assignment, misc]

ACP_DELEGATE_KEY = "octop_acp_delegate"


def _delegate_marker(request: ModelRequest[Any]) -> dict[str, Any] | None:
    """Marker on the newest HumanMessage, or ``None``.

    Only the newest HumanMessage counts — a later turn without a mention
    must not inherit the previous turn's delegation directive.
    """
    for message in reversed(request.messages or []):
        if not isinstance(message, HumanMessage):
            continue
        raw = message.additional_kwargs.get(ACP_DELEGATE_KEY)
        if isinstance(raw, dict) and str(raw.get("runner") or "").strip():
            return raw
        return None
    return None


def _directive(runner: str) -> str:
    return "\n".join(
        [
            f'ACP delegation requested for this turn — the external runner "{runner}" '
            "owns this task:",
            "- Do NOT answer the task yourself, not even partially: call the "
            "`acp_runner` tool FIRST, before emitting any text to the user.",
            f'- Call acp_runner with action="start", runner="{runner}", message=<the '
            f'user\'s request without the "@{runner}" prefix>.',
            "- Wait for the tool result. If it reports an already-open session for "
            "this runner, re-call with "
            'action="message" to continue it instead of starting over.',
            "- Sessions are in-memory and disappear when Octop restarts: if the tool "
            'says no session is bound, call action="start" again — never assume one '
            "is still open.",
            "- If the result contains [permission_required], show the user the listed "
            "options; when they answer, call "
            f'action="respond" with runner="{runner}" and the exact option id they chose.',
            "- Your final message must only relay the runner's answer (a one-line "
            f'lead-in such as "已转给 {runner}" is fine); do not add your own '
            "analysis or redo the delegated work.",
        ]
    )


def _append_system_hint(request: ModelRequest[Any], hint: str) -> ModelRequest[Any]:
    cleaned = hint.strip()
    if not cleaned:
        return request
    existing = request.system_message
    if existing is None:
        return request.override(system_message=SystemMessage(content=cleaned))
    content = existing.content
    if isinstance(content, str) and cleaned in content:
        return request
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("text") == cleaned for block in content
    ):
        return request
    if isinstance(content, str):
        merged: str | list[Any] = f"{content.rstrip()}\n\n{cleaned}"
    elif isinstance(content, list):
        merged = [*content, {"type": "text", "text": cleaned}]
    else:
        merged = cleaned
    return request.override(system_message=SystemMessage(content=merged))


def _tool_name(request: Any) -> str:
    return str((getattr(request, "tool_call", None) or {}).get("name", ""))


def _session_error_tool_message(request: Any, exc: Exception) -> ToolMessage:
    tool_call_id = str((getattr(request, "tool_call", None) or {}).get("id", ""))
    return ToolMessage(
        content=(
            f"Error: {exc}. Recoverable: call acp_runner with action=\"start\" "
            "(runner=<name>, message=<task>) to open a session, then retry."
        ),
        tool_call_id=tool_call_id,
    )


class AcpDelegateMiddleware(AgentMiddleware[Any, Any]):
    """Turn the ``@<runner>`` mention marker into a per-turn system directive.

    Also guards ``acp_runner`` tool calls: the harness does not catch
    ``ACPSessionError`` (e.g. the model skipped ``action="start"`` after an
    Octop restart wiped the in-memory sessions), which would crash the whole
    turn. Here it is converted into a normal tool error the model can recover
    from.
    """

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage],
    ) -> ToolMessage:
        if ACPSessionError is None or _tool_name(request) != "acp_runner":
            return handler(request)
        try:
            return handler(request)
        except ACPSessionError as exc:
            return _session_error_tool_message(request, exc)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        if ACPSessionError is None or _tool_name(request) != "acp_runner":
            return await handler(request)
        try:
            return await handler(request)
        except ACPSessionError as exc:
            return _session_error_tool_message(request, exc)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        marker = _delegate_marker(request)
        if marker is None:
            return handler(request)
        return handler(_append_system_hint(request, _directive(str(marker["runner"]).strip())))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        marker = _delegate_marker(request)
        if marker is None:
            return await handler(request)
        return await handler(
            _append_system_hint(request, _directive(str(marker["runner"]).strip()))
        )


def setup(ctx: Any) -> None:
    """Plugin entry — register the middleware with the harness plugin registry."""
    ctx.middleware(AcpDelegateMiddleware())
