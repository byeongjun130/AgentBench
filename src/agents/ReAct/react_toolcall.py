"""ReAct agent using OpenAI-style native tool calling.

Unlike `react.py` (which teaches the model a text-based `Action: tool[arg]`
syntax), this variant binds the workload tools through the chat-completions
`tools` API. The model emits structured `tool_calls` as a first-class field,
so its planning/analysis stays in the reasoning channel and the format-drift
we see with text-based ReAct (preamble leakage, multi-`Action:` bursts,
hallucinated `Observation:` lines) simply cannot occur.

Termination is signalled by the `finish` tool: when the model calls it, we
emit a terminal `ToolMessage` with `artifact={"done": True}` and the graph
exits.
"""

import json
import re
from typing import Literal, Sequence

from colorama import Fore, Style
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    actor_steps: int


def create_react_agent_toolcall(
    model: LanguageModelLike,
    tools: Sequence[BaseTool],
    iteration_limit: int,
    print_log: bool = False,
) -> CompiledStateGraph:
    """Build a tool-calling ReAct graph with an actor-step iteration cap.

    Only `agent` (call_model) visits count toward `iteration_limit`; tool
    visits do not. LangGraph's recursion_limit must be set higher than
    iteration_limit on the caller side (≥ 2 × iteration_limit + headroom)
    to leave room for tool node visits.
    """
    tool_dict = {t.name: t for t in tools}
    # tool_choice="any" forces the model to always emit a structured
    # tool_call; otherwise gpt-oss sometimes drops the finish call into
    # content as plain JSON, which vLLM's parser correctly routes to the
    # `final` channel — losing the tool_call attribution.
    bound_model = model.bind_tools(list(tools), tool_choice="any")

    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = None
        for chunk in bound_model.stream(state["messages"], config):
            response = chunk if response is None else response + chunk
        if print_log:
            if response.content:
                print(Fore.CYAN + str(response.content) + Style.RESET_ALL)
            for tc in getattr(response, "tool_calls", None) or []:
                print(Fore.MAGENTA + f"{tc['name']}({tc.get('args', {})})" + Style.RESET_ALL)
        return {
            "messages": [response],
            "actor_steps": state.get("actor_steps", 0) + 1,
        }

    def execute_tool(state: AgentState) -> AgentState:
        last: AIMessage = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            # gpt-oss on Harmony sometimes emits a "direct answer" on the
            # `final` channel instead of calling `finish()` on the tool
            # channel (`commentary to=functions.finish`). Rescue that case
            # by parsing a `{"answer": "..."}` JSON blob out of content and
            # treating it as a finish call.
            content = (getattr(last, "content", "") or "").strip()
            match = re.search(r"\{[^{}]*\"answer\"\s*:\s*\"([^\"]*)\"[^{}]*\}", content, re.DOTALL)
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
            # webshop's SearchTool/ClickTool return (observation, info); info carries
            # `done: True` once the user clicks Buy Now. Pass it through so
            # `route_after_tool` can exit without a dedicated finish tool.
            out.append(ToolMessage(content=result, tool_call_id=tc_id, artifact=tool_artifact))
        return {"messages": out}

    def route_after_agent(state: AgentState) -> Literal["tool", "__end__"]:
        if state.get("actor_steps", 0) >= iteration_limit:
            return "__end__"
        return "tool"

    def route_after_tool(state: AgentState) -> Literal["agent", "__end__"]:
        last = state["messages"][-1]
        artifact = getattr(last, "artifact", None)
        if artifact and artifact.get("done"):
            return "__end__"
        return "agent"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tool", execute_tool)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_conditional_edges("tool", route_after_tool)
    return workflow.compile()
