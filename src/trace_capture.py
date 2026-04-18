"""Per-turn trace capture for AgentSim's agentic dataset format.

Records every LLM call made by an agent as one `turn` entry, in the schema
expected by AgentSim's `validation/trace_estimator/convert_agentbench.py`.

Output envelope:

    {
      "version": 2,
      "type": "agentic",
      "model": "<model string>",
      "agents": [
        {
          "success": <bool>,
          "turns": [
            {
              "prompt_tokens": int,
              "completion_tokens": int,
              "reasoning_tokens": int,
              "non_llm_latency": float,
              "role": "actor" | "reflect" | ...,
              "request_messages": [{"role": ..., "content": ...}, ...],
              "response_messages": [{"role": "assistant", "content": ...}]
            },
            ...
          ]
        },
        ...
      ]
    }

`non_llm_latency[i]` is the wall-clock gap between the end of turn i's LLM
call and the start of turn i+1's LLM call (tool + observation-handling
time). It is 0.0 for the final turn of an agent run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


# ---------------------------------------------------------------------------
# HTTP response snoop.
#
# LangChain's ChatOpenAI parses the OpenAI-compat response and drops vendor
# fields like gpt-oss's `message.reasoning`. To recover them, we register an
# httpx response event hook that saves the raw JSON body into a thread-local
# slot *before* LangChain parses it. The callback reads that slot in
# `on_llm_end` and attaches the missing fields to the turn record.
# ---------------------------------------------------------------------------

_snoop_storage = threading.local()


def _parse_sse_stream(body: bytes) -> Optional[Dict[str, Any]]:
    """Reconstruct a synthetic chat-completions dict from an SSE body.

    OpenAI-compatible streams are a series of `data: {...}\\n\\n` events and
    a final `data: [DONE]`. Each event has a `choices[0].delta` partial; we
    concatenate `delta.content` and `delta.reasoning` (gpt-oss vendor
    extension) across events, and capture `usage` from whichever event
    carries it (vLLM typically attaches it to the final non-DONE event).
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    text = body.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if isinstance(delta.get("reasoning"), str):
            reasoning_parts.append(delta["reasoning"])
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    if not content_parts and not reasoning_parts and usage is None:
        return None

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "reasoning": "".join(reasoning_parts),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {},
    }


def _snoop_response_hook(response: httpx.Response) -> None:
    try:
        response.read()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            _snoop_storage.latest = _parse_sse_stream(response.content)
        else:
            _snoop_storage.latest = json.loads(
                response.content.decode("utf-8")
            )
    except Exception:
        _snoop_storage.latest = None


def make_snooping_http_client() -> httpx.Client:
    """Return an httpx.Client with the snoop hook installed.

    Pass this into ChatOpenAI via `http_client=` so every chat-completions
    response (streaming or not) is captured before LangChain parses it.
    """
    return httpx.Client(event_hooks={"response": [_snoop_response_hook]})


def _pop_latest_raw_response() -> Optional[Dict[str, Any]]:
    data = getattr(_snoop_storage, "latest", None)
    _snoop_storage.latest = None
    return data


_LC_TYPE_TO_OPENAI_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}


def _message_to_openai_dict(message: BaseMessage) -> Dict[str, Any]:
    message_type = getattr(message, "type", "user")
    role = _LC_TYPE_TO_OPENAI_ROLE.get(message_type, message_type)
    content = (
        message.content
        if isinstance(message.content, str)
        else str(message.content)
    )
    return {"role": role, "content": content}


class TraceCaptureCallback(BaseCallbackHandler):
    """LangChain callback that records per-LLM-call turns in AgentSim format.

    Attach to a ChatOpenAI (or any chat model) instance via
    ``model.callbacks = [callback]``. The handler subscribes to two trap
    points fired by the chat-model runtime: ``on_chat_model_start`` (sees
    outgoing prompt messages) and ``on_llm_end`` (sees the response + token
    usage). Each pair produces one turn entry.
    """

    def __init__(self, default_role: str = "actor") -> None:
        self.default_role = default_role
        self.role_override: Optional[str] = None
        self.turns: List[Dict[str, Any]] = []
        self._pending: Dict[UUID, Dict[str, Any]] = {}
        self._last_end_time: Optional[float] = None

    def reset(self) -> None:
        self.turns = []
        self._pending.clear()
        self._last_end_time = None
        self.role_override = None

    def set_role(self, role: Optional[str]) -> None:
        """Override role for subsequent LLM calls until cleared (role=None)."""
        self.role_override = role

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        start_time = time.time()
        flat_messages = messages[0] if messages else []
        if self.turns and self._last_end_time is not None:
            self.turns[-1]["non_llm_latency"] = max(
                0.0, start_time - self._last_end_time
            )
        self._pending[run_id] = {
            "request_messages": [
                _message_to_openai_dict(message) for message in flat_messages
            ],
            "role": self.role_override or self.default_role,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        end_time = time.time()
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        content = ""
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0

        if response.generations and response.generations[0]:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            if message is not None:
                content = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )
                usage = getattr(message, "usage_metadata", None) or {}
                prompt_tokens = int(usage.get("input_tokens", 0) or 0)
                completion_tokens = int(usage.get("output_tokens", 0) or 0)
                output_details = usage.get("output_token_details", {}) or {}
                reasoning_tokens = int(output_details.get("reasoning", 0) or 0)
            else:
                content = getattr(generation, "text", "") or ""

        if prompt_tokens == 0 and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {}) or {}
            prompt_tokens = int(
                token_usage.get("prompt_tokens", prompt_tokens) or 0
            )
            completion_tokens = int(
                token_usage.get("completion_tokens", completion_tokens) or 0
            )
            completion_details = (
                token_usage.get("completion_tokens_details", {}) or {}
            )
            reasoning_tokens = int(
                completion_details.get("reasoning_tokens", reasoning_tokens)
                or 0
            )

        # Recover fields that LangChain dropped during parsing.
        reasoning_text = ""
        raw = _pop_latest_raw_response()
        if raw is not None:
            try:
                raw_message = raw["choices"][0]["message"]
                reasoning_text = raw_message.get("reasoning") or ""
                raw_usage = raw.get("usage") or {}
                raw_completion_details = (
                    raw_usage.get("completion_tokens_details") or {}
                )
                reasoning_tokens_from_raw = raw_completion_details.get(
                    "reasoning_tokens"
                )
                if reasoning_tokens_from_raw:
                    reasoning_tokens = int(reasoning_tokens_from_raw)
            except (KeyError, IndexError, TypeError):
                pass

        turn = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "non_llm_latency": 0.0,
            "role": pending["role"],
            "request_messages": pending["request_messages"],
            "response_messages": [
                {"role": "assistant", "content": content}
            ],
        }
        if reasoning_text:
            turn["reasoning_text"] = reasoning_text
        self.turns.append(turn)
        self._last_end_time = end_time

    def snapshot_turns(self) -> List[Dict[str, Any]]:
        return [dict(turn) for turn in self.turns]


def write_agentsim_trace(
    path: str,
    model: str,
    agents: List[Dict[str, Any]],
) -> None:
    """Write an AgentSim agentic-format JSON trace to `path`."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "version": 2,
        "type": "agentic",
        "model": model,
        "agents": agents,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
