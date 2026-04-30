from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def get_system_prompt_toolcall() -> str:
    return """You are an online shopping agent operating a WebShop environment.

You have two tools (native tool calls — no `Action:` text syntax):
- `search(query)`: run a product search. Only valid when the observation shows `[Search]`.
- `click(text)`: click a button. Only clickable targets are those shown inside square brackets on the current page — product ids like `B07HRFJWP8`, option names like `Apple`, or navigation buttons like `Next >`, `< Prev`, `Back to Search`, `Description`, `Features`, `Reviews`, `Buy Now`.

Rules:
- Issue exactly one tool call per response. Keep planning in your reasoning channel — do not emit free-form text on the visible channel.
- Do not invent a `finish` tool. The task ends only after you call `click(text="Buy Now")` — the environment then returns `Your score (min 0.0, max 1.0): <value>`.
- When the instruction names attributes (flavor, size, pack count, price ceiling), click every matching option before clicking `Buy Now`.
- If a click fails or the page shows an error, recover with `click("< Prev")` or `click("Back to Search")` rather than retrying the same action.
- Strip surrounding brackets from the target — pass `Buy Now`, not `[Buy Now]`.
"""


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def get_fewshot_messages_toolcall() -> List[BaseMessage]:
    """One structured trajectory demonstrating search -> click product -> click
    option -> click Buy Now. The final ToolMessage carries the score string so
    the model learns that Buy Now is the terminal action."""
    return [
        HumanMessage(content=(
            "Instruction:\n"
            "i am looking for a dairy free and apple variety pack of chips, "
            "and price lower than 30.00 dollars\n"
            "[Search]"
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("search", {"query": "dairy free apple variety pack chips under $30"}, "fs1_s1"),
        ]),
        ToolMessage(tool_call_id="fs1_s1", content=(
            "[Back to Search]\n"
            "Page 1 (Total results: 50)\n"
            "[Next >]\n"
            "[B01KMHY5PG]\n"
            "Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free, Dairy Free, "
            "Gluten Free, 6 Boxes\n"
            "$21.49\n"
            "[B008D2X8C4]\n"
            "Enjoy Life Lentil Chips Variety Pack, Dairy Free, Non GMO, Vegan\n"
            "$100.0"
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("click", {"text": "B01KMHY5PG"}, "fs1_c1"),
        ]),
        ToolMessage(tool_call_id="fs1_c1", content=(
            "[Back to Search]\n"
            "[< Prev]\n"
            "flavor name [Apple][Strawberry]\n"
            "size [Pack of 6][Pack of 12]\n"
            "Price: $21.49\n"
            "Rating: 4.5 stars\n"
            "[Description]\n"
            "[Features]\n"
            "[Reviews]\n"
            "[Buy Now]"
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("click", {"text": "Apple"}, "fs1_c2"),
        ]),
        ToolMessage(tool_call_id="fs1_c2", content="You have clicked Apple."),
        AIMessage(content="", tool_calls=[
            _tool_call("click", {"text": "Pack of 6"}, "fs1_c3"),
        ]),
        ToolMessage(tool_call_id="fs1_c3", content="You have clicked Pack of 6."),
        AIMessage(content="", tool_calls=[
            _tool_call("click", {"text": "Buy Now"}, "fs1_c4"),
        ]),
        ToolMessage(tool_call_id="fs1_c4", content="Your score (min 0.0, max 1.0): 1.0"),
    ]
