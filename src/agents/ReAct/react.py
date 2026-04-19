from typing import Literal, Sequence, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from colorama import Fore, Style
import re

def extract_thoughts_and_actions(input_str):
    stop_boundary = r"(?:\s*Thought:|\s*Action:|\s*Observation:|\Z)"
    thought_pattern = re.compile(r"Thought:\s*(.*?)" + stop_boundary, re.DOTALL)
    action_pattern = re.compile(r"Action:\s*(.*?)" + stop_boundary, re.DOTALL)
    thought_match = thought_pattern.search(input_str)
    action_match = action_pattern.search(input_str)

    result = ""
    if thought_match:
        thought = thought_match.group(1).strip()
        result+="Thought: " + thought + "\n"
    if action_match:
        action = action_match.group(1).strip()
        result+="Action: " + action + "\n"

    if not result:
        return input_str
    return result.strip()

def extract_tool_calls(input_str):
    input_split = input_str.split("Action: ")
    if len(input_split) == 1:
        return []
    input_str = input_split[1]
    pattern = re.compile(r"^(\w+)\[", re.MULTILINE)
    matches = pattern.finditer(input_str)

    results = []
    for match in matches:
        name = match.group(1)
        
        start_idx = match.end()
        stack = 1
        end_idx = start_idx

        while end_idx < len(input_str) and stack > 0:
            if input_str[end_idx] == '[':
                stack += 1
            elif input_str[end_idx] == ']':
                stack -= 1
            end_idx += 1

        if stack == 0:  
            argument = input_str[start_idx:end_idx - 1].strip()
            results.append({"name": name, "argument": argument})
    if not results:
        return []
    return [results[0]]
    
# We create the AgentState that we will pass around
# This simply involves a list of messages
# We want steps to return messages to append to the list
# So we annotate the messages attribute with operator.add
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_react_agent(
    model: LanguageModelLike,
    tools: Union[Sequence[BaseTool]],
    print_log: bool = False,
) -> CompiledStateGraph:

    tool_dict = {}
    for tool in tools:
        tool_dict[tool.name] = tool
    if print_log:
        print(f"Registered tools: {list(tool_dict.keys())}")

    format_error_message = (
        "Observation: Your response must be in this exact format:\n"
        "Thought: <your reasoning>\n"
        "Action: tool_name[argument]\n"
        "Do not include any prefix text, extra Actions, or additional sections."
    )

    def is_valid_react_format(content: str) -> bool:
        stripped = content.strip()
        if not stripped.startswith("Thought:"):
            return False
        if stripped.count("Action:") != 1:
            return False
        return True

    def execute_tool(state: AgentState) -> AgentState:
        last_message = state["messages"][-1]
        if not is_valid_react_format(last_message.content):
            return {"messages": [HumanMessage(content=format_error_message, artifact={"done": False})]}
        tool_calls = extract_tool_calls(last_message.content)
        if not tool_calls:
            return {"messages": [HumanMessage(content=format_error_message, artifact={"done": False})]}
        else:
            messages = []
            for tool in tool_calls:
                if tool["name"] not in tool_dict:
                    if print_log:
                        print(f"Tool {tool['name']} not found.")
                    continue
                try:
                    tool_output = tool_dict[tool["name"]].invoke(tool["argument"])
                    if isinstance(tool_output, tuple):
                        tool_output, artifact = tool_output
                    else:
                        artifact = None
                    if print_log:
                        print(f"Observation: {tool_output}")
                except Exception as e:
                    tool_output = f"Tool Execution Error: {e}"
                    if print_log:
                        print(tool_output)
                    messages.append(HumanMessage(content=tool_output, artifact={"done": False}))
                    return {"messages": messages}
                if tool["name"] == "finish":
                    messages.append(HumanMessage(content=tool_output, artifact={"done": True}))
                    return {"messages": messages}
                else:
                    messages.append(HumanMessage(content=f"Observation: {tool_output}", artifact=artifact))
            if messages:
                return {"messages": messages}
            else:
                return {"messages": [HumanMessage(content=format_error_message, artifact={"done": False})]}

    # Define the function that calls the model
    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = None
        for chunk in model.stream(state["messages"], config):
            if response:
                response += chunk
            else:
                response = chunk
        if print_log:
            print(response.content)
        return {"messages": [response]}

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState) -> Literal["agent", "__end__"]:
        if print_log:
            print(Fore.CYAN+Style.BRIGHT+f"{'-'*30}"+Style.RESET_ALL)
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.artifact and "done" in last_message.artifact and last_message.artifact["done"]:
            return "__end__"
        else:
            return "agent"

    # Define a new graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tool", execute_tool)

    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "tool")
    # We now add a conditional edge
    workflow.add_conditional_edges(
        "tool",
        should_continue,
    )
    
    return workflow.compile()
