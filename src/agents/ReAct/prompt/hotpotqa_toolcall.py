from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def get_system_prompt_toolcall() -> str:
    return """You are a Wikipedia-grounded question answering agent.

You have three tools (native tool calls — no `Action:` text syntax):
- `search(text)`: Wikipedia search for `text`. Returns the first paragraph of the best-matching article, or a list of similar titles if no exact match.
- `lookup(text)`: On the most-recently searched article, return the next paragraph containing `text`. Use this to drill into the current page without re-searching.
- `finish(answer)`: Submit the final answer and end the task.

Rules:
- Issue exactly one tool call per response. Do not write free-form text in the visible channel; keep planning in your reasoning channel.
- Call `finish` as soon as you can answer. ALWAYS submit the final answer via `finish(answer=...)`, never as plain text.
- `answer` must match the phrasing used in the Wikipedia text — include qualifying words ("seated", "from"), honorifics ("DSC", "Jr."), and the full name as it appears (e.g. "Barton Lee Hazlewood", not just "Lee Hazlewood"). For yes/no questions, use exactly "yes" or "no".
- NEVER claim you don't know — if unclear, issue another `search` or use `lookup`.
- Answer MUST NOT be 'unclear', 'unknown', 'neither', 'unrelated', or 'undetermined'.
"""


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def get_fewshot_messages_toolcall() -> List[BaseMessage]:
    """Return a list of prior-turn messages demonstrating tool use + answer phrasing.

    Each example shows: user question → structured `search`/`lookup` tool
    calls with real `ToolMessage` responses → a `finish` tool call whose
    `answer` exactly matches the Wikipedia phrasing expected by HotpotQA's
    exact-match scorer.
    """
    return [
        # ---------- Example 1: include the qualifier word ("seated") ----------
        HumanMessage(content=(
            "The arena where the Lewiston Maineiacs played their home games can "
            "seat how many people?"
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("search", {"text": "Lewiston Maineiacs home arena"}, "fs1_s1"),
        ]),
        ToolMessage(tool_call_id="fs1_s1", content=(
            "The Lewiston Maineiacs were a junior ice hockey team that played "
            "home games at the Androscoggin Bank Colisée in Lewiston, Maine. "
            "The arena has a seating capacity of 3,677 seated."
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("finish", {"answer": "3,677 seated"}, "fs1_f"),
        ]),
        ToolMessage(tool_call_id="fs1_f", content="Answer: 3,677 seated"),

        # ---------- Example 2: yes/no answer is a single token ----------
        HumanMessage(content="Are Giuseppe Verdi and Ambroise Thomas both opera composers?"),
        AIMessage(content="", tool_calls=[
            _tool_call("search", {"text": "Giuseppe Verdi"}, "fs2_s1"),
        ]),
        ToolMessage(tool_call_id="fs2_s1", content=(
            "Giuseppe Fortunino Francesco Verdi was an Italian composer, best "
            "known for his operas."
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("search", {"text": "Ambroise Thomas"}, "fs2_s2"),
        ]),
        ToolMessage(tool_call_id="fs2_s2", content=(
            "Charles Louis Ambroise Thomas was a French composer, best known "
            "for his operas Mignon (1866) and Hamlet (1868)."
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("finish", {"answer": "yes"}, "fs2_f"),
        ]),
        ToolMessage(tool_call_id="fs2_f", content="Answer: yes"),

        # ---------- Example 3: use the full name as it appears ----------
        HumanMessage(content=(
            "Who was the writer of These Boots Are Made for Walkin' and who died in 2007?"
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("search", {"text": "These Boots Are Made for Walkin'"}, "fs3_s1"),
        ]),
        ToolMessage(tool_call_id="fs3_s1", content=(
            "\"These Boots Are Made for Walkin'\" is a song written by Barton "
            "Lee Hazlewood and recorded by Nancy Sinatra in 1966."
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("lookup", {"text": "Hazlewood"}, "fs3_l1"),
        ]),
        ToolMessage(tool_call_id="fs3_l1", content=(
            "(Result 1/1) Barton Lee Hazlewood (July 9, 1929 – August 4, 2007) "
            "was an American country and pop singer, songwriter, and producer."
        )),
        AIMessage(content="", tool_calls=[
            _tool_call("finish", {"answer": "Barton Lee Hazlewood"}, "fs3_f"),
        ]),
        ToolMessage(tool_call_id="fs3_f", content="Answer: Barton Lee Hazlewood"),
    ]
