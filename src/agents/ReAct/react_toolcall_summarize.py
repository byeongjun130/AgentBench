"""ReAct + auto-compact summarization (variant of `react_toolcall.py`).

Behavior matches `react_toolcall.py` until the prompt that the *next*
`agent` call would send exceeds `summarize_token_threshold` tokens. At
that point, control routes to a `summarizer` node that:

  1. Builds a summarize-only prompt from the existing conversation
     (pinned system / fewshot / first user-query head + everything since)
     and a closing "summarize the work above" instruction.
  2. Invokes the same model (no tools bound) to produce a compact summary.
  3. Replaces `state["messages"]` with
     ``[pinned head] + AIMessage(summary) + HumanMessage("Continue ...")``
     via ``RemoveMessage(REMOVE_ALL_MESSAGES)`` + re-add, so the next
     `agent` call resumes with a much smaller working set.

The graph is:

    agent ─┬─(step limit)──▶ END
           └─▶ tool ─┬─(done)──▶ END
                     ├─(over T)─▶ summarizer ─▶ agent
                     └─(else)──▶ agent

The first `agent` call is never compacted (no entry-point check), so an
agent run with N initial pinned tokens just below T behaves identically
to the unmodified `react_toolcall` graph until enough observations
accumulate.

Token counting uses ``tokenizer.apply_chat_template(..., tools=...,
add_generation_prompt=True)`` so the threshold check sees the same
length vLLM's chat template will produce server-side. The OpenAI-shaped
message dicts are produced by ``trace_capture._message_to_openai_dict``
to keep the routing-time count consistent with what
``TraceCaptureCallback`` records as ``prompt_tokens``.
"""

import json
import re
from typing import Any, Literal, Optional, Sequence

from colorama import Fore, Style
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import Annotated, TypedDict

from src.trace_capture import _message_to_openai_dict


SUMMARIZE_REQUEST = (
    "Summarize the work above into a concise context the assistant can use "
    "to continue the task. Include: (1) Findings — key facts and results "
    "from tool outputs so far. (2) Open questions — what we still need to "
    "determine. (3) Next step — the next concrete tool call to make. "
    "Reply with the summary text only; do not call any tool."
)

RESUME_PROMPT = (
    "Continue the task using the summary above. Issue exactly one tool "
    "call — never produce a final answer as plain text; submit it through "
    "the appropriate tool (for QA tasks, call `finish(answer=...)`)."
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    actor_steps: int


def create_react_agent_toolcall_summarize(
    model: LanguageModelLike,
    tools: Sequence[BaseTool],
    summarize_token_threshold: int,
    pinned_head_count: int,
    tokenizer: Any,
    iteration_limit: int,
    trace_callback: Optional[Any] = None,
    print_log: bool = False,
) -> CompiledStateGraph:
    """Build a ReAct-toolcall graph with a token-budget compaction node.

    Args:
        model: Unbound chat model (we bind tools internally for the agent
            node and call the unbound version for the summarizer node).
        tools: Workload tools, including any terminal `finish` tool.
        summarize_token_threshold: When > 0 and the next agent prompt
            would exceed this length, route to the summarizer first.
        pinned_head_count: Number of leading messages to preserve across
            compaction (system prompt + fewshot block + first user query).
        tokenizer: Loaded HuggingFace tokenizer for the served model. The
            same path passed to `TraceCaptureCallback(tokenizer_path=...)`.
        iteration_limit: Maximum number of actor (agent-node) steps. Only
            agent calls count; summarizer calls are excluded. LangGraph's
            graph-level recursion_limit should be set higher than this to
            leave headroom for summarizer and tool node visits.
        trace_callback: Optional `TraceCaptureCallback`; used to flip
            `set_role("summarizer")` around the summarizer LLM call so
            its turn is labeled correctly in the saved trace.
        print_log: Print per-turn agent thoughts and observations.
    """
    tool_dict = {t.name: t for t in tools}
    bound_model = model.bind_tools(list(tools), tool_choice="any")
    tools_schema = [convert_to_openai_tool(t) for t in tools]

    def _next_prompt_tokens(messages: Sequence[BaseMessage]) -> int:
        """Return the token count `bound_model` would send for these messages."""
        openai_msgs = [_message_to_openai_dict(m) for m in messages]
        ids = tokenizer.apply_chat_template(
            openai_msgs,
            tools=tools_schema,
            add_generation_prompt=True,
            tokenize=True,
        )
        return len(ids)

    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = None
        for chunk in bound_model.stream(state["messages"], config):
            response = chunk if response is None else response + chunk
        if print_log:
            if response.content:
                print(Fore.CYAN + str(response.content) + Style.RESET_ALL)
            for tc in getattr(response, "tool_calls", None) or []:
                print(
                    Fore.MAGENTA
                    + f"{tc['name']}({tc.get('args', {})})"
                    + Style.RESET_ALL
                )
        return {
            "messages": [response],
            "actor_steps": state.get("actor_steps", 0) + 1,
        }

    def execute_tool(state: AgentState) -> AgentState:
        last: AIMessage = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            content = (getattr(last, "content", "") or "").strip()
            match = re.search(
                r"\{[^{}]*\"answer\"\s*:\s*\"([^\"]*)\"[^{}]*\}",
                content,
                re.DOTALL,
            )
            if match and "finish" in tool_dict:
                rescued = tool_dict["finish"].invoke({"answer": match.group(1)})
                return {"messages": [ToolMessage(
                    content=rescued,
                    tool_call_id="fallback_finish",
                    artifact={"done": True},
                )]}
            return {"messages": [ToolMessage(
                content="No tool call issued. Call one of: search, lookup, finish.",
                tool_call_id="no_tool_call",
                artifact={"done": True},
            )]}

        out = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            tc_id = tc.get("id") or f"call_{name}"
            if name not in tool_dict:
                out.append(ToolMessage(
                    content=f"Tool {name} not found.",
                    tool_call_id=tc_id,
                    artifact={"done": False},
                ))
                continue
            try:
                result = tool_dict[name].invoke(args)
                tool_artifact = None
                if isinstance(result, tuple):
                    result, tool_artifact = result
            except Exception as e:
                out.append(ToolMessage(
                    content=f"Tool Execution Error: {e}",
                    tool_call_id=tc_id,
                    artifact={"done": False},
                ))
                return {"messages": out}
            if print_log:
                print(f"Observation: {result}")
            if name == "finish":
                out.append(ToolMessage(
                    content=result,
                    tool_call_id=tc_id,
                    artifact={"done": True},
                ))
                return {"messages": out}
            out.append(ToolMessage(content=result, tool_call_id=tc_id, artifact=tool_artifact))
        return {"messages": out}

    def summarize(state: AgentState, config: RunnableConfig) -> AgentState:
        messages = list(state["messages"])
        pinned = messages[:pinned_head_count]
        # Ask the unbound model to summarize the *full* current conversation.
        # The summarizer call is intentionally not tool-bound: tool_choice is
        # not forced, so the model is free to produce plain text.
        summarize_messages = pinned + messages[pinned_head_count:] + [
            HumanMessage(content=SUMMARIZE_REQUEST),
        ]
        if trace_callback is not None:
            trace_callback.set_role("summarizer")
        try:
            summary_response = model.invoke(summarize_messages, config)
        finally:
            if trace_callback is not None:
                trace_callback.set_role(None)

        summary_text = (
            summary_response.content
            if isinstance(summary_response.content, str)
            else str(summary_response.content)
        ).strip()
        if not summary_text:
            # Defensive: never let an empty summary blank the working state.
            summary_text = "(No summary produced; continue with the task.)"
        if print_log:
            print(
                Fore.YELLOW
                + f"[auto-compact] summarized into {len(summary_text)} chars"
                + Style.RESET_ALL
            )

        # Replace state["messages"] in-place: drop everything, re-add the
        # pinned head, then a single AI/Human pair carrying the summary.
        # `add_messages` interprets RemoveMessage(id=REMOVE_ALL_MESSAGES)
        # as "wipe the list before applying the rest of this update".
        rebuilt = list(pinned) + [
            AIMessage(content=summary_text),
            HumanMessage(content=RESUME_PROMPT),
        ]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)] + rebuilt}

    def route_after_agent(state: AgentState) -> Literal["tool", "__end__"]:
        if state.get("actor_steps", 0) >= iteration_limit:
            return "__end__"
        return "tool"

    def route_after_tool(state: AgentState) -> Literal["agent", "summarizer", "__end__"]:
        last = state["messages"][-1]
        artifact = getattr(last, "artifact", None)
        if artifact and artifact.get("done"):
            return "__end__"
        if summarize_token_threshold > 0:
            projected = _next_prompt_tokens(state["messages"])
            if projected > summarize_token_threshold:
                return "summarizer"
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tool", execute_tool)
    workflow.add_node("summarizer", summarize)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_conditional_edges("tool", route_after_tool)
    workflow.add_edge("summarizer", "agent")
    return workflow.compile()
