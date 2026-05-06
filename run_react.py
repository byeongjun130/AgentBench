import json
import os
import time
import numpy as np
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from colorama import Fore, Style
from src.agents.ReAct.react import create_react_agent
from src.agents.ReAct.react_toolcall import create_react_agent_toolcall
from src.agents.ReAct.react_toolcall_summarize import (
    create_react_agent_toolcall_summarize,
)
from src.trace_capture import (
    TraceCaptureCallback,
    make_snooping_http_client,
    write_agentsim_trace,
)
from src.utils import parse_answer

from dotenv import load_dotenv
from langsmith import traceable, trace

load_dotenv()


def _flush_trace(args, trace_agents, tools_schema, envelope_metadata):
    """Write the current agent list to disk after each completed sample."""
    write_agentsim_trace(
        path=args.trace_path,
        model=args.model,
        agents=trace_agents,
        tools=tools_schema,
        metadata=envelope_metadata,
    )


def main(args):
    print_log = bool(getattr(args, "print_log", False))
    if args.host:
        host_url = f"http://{args.host}:{args.port}/v1"
    else:
        host_url = None

    score_sum = 0
    pass_count = 0
    # Load dataset
    from src.utils import load_dataset, get_evaluation_function
    print(f"Loading dataset for workload: {args.workload}")
    dataset = load_dataset(args.workload)
    evaluator = get_evaluation_function(args.workload)
    sample_indices = getattr(args, "sample_indices", None)
    if sample_indices:
        dataset = [dataset[i] for i in sample_indices]
        samples = len(dataset)
    else:
        sample_start = int(getattr(args, "sample_start", 0) or 0)
        dataset = dataset[sample_start:]
        samples = min(len(dataset), args.samples) if args.samples else len(dataset)
    latencies = []

    def pretty_output(i):
        print(Fore.YELLOW+"=" * 30)
        print(f"Sample {i + 1}/{samples}")
        if args.workload == "webshop":
            print(f"Average score so far: {round(score_sum / (i + 1), 2)}")
        print(f"Accuracy so far: {round(pass_count / (i + 1), 2)}")
        if latencies:
            print(f"Avg. latency: {round(sum(latencies) / len(latencies), 2)} sec")
            print(f"p50 latency: {round(np.percentile(latencies, 50), 2)} sec")
            print(f"p90 latency: {round(np.percentile(latencies, 90), 2)} sec")
            print(f"p95 latency: {round(np.percentile(latencies, 95), 2)} sec") 
            print(f"p99 latency: {round(np.percentile(latencies, 99), 2)} sec")
        print("=" * 30+Style.RESET_ALL)
        print("\n")

    use_tool_calling = bool(getattr(args, "use_tool_calling", False))
    summarize_token_threshold = int(
        getattr(args, "summarize_token_threshold", 0) or 0
    )
    if summarize_token_threshold > 0 and not use_tool_calling:
        raise ValueError(
            "summarize_token_threshold > 0 requires use_tool_calling=true; "
            "auto-compact is only wired for the tool-calling ReAct variant."
        )

    # Load model
    save_trace = bool(getattr(args, "save_trace", False))
    http_client = make_snooping_http_client() if save_trace else None
    model_kwargs = dict(
        model=args.model,
        base_url=host_url,
        stream_usage=True,
        temperature=args.temperature,
        http_client=http_client,
    )
    if not use_tool_calling:
        model_kwargs["stop"] = ["Observation:"]
    model = ChatOpenAI(**model_kwargs)

    trace_callback = (
        TraceCaptureCallback(default_role="actor", tokenizer_path=args.model)
        if save_trace
        else None
    )
    if trace_callback is not None:
        model.callbacks = [trace_callback]
    trace_agents = []

    # Tokenizer used by the summarize-aware graph to count the prompt the
    # *next* agent call would send. Loaded once when needed; matches the
    # tokenizer the trace_callback lazy-loads for reasoning-token counting.
    summarize_tokenizer = None
    if summarize_token_threshold > 0:
        from transformers import AutoTokenizer
        summarize_tokenizer = AutoTokenizer.from_pretrained(args.model)

    envelope_metadata = (
        {"summarize_token_threshold": summarize_token_threshold}
        if summarize_token_threshold > 0 else None
    )

    system_prompt = None
    tools_schema = None
    count = 0
    pass_count = 0

    # Resume: if a partial trace file already exists, reload completed agents
    # and skip those samples so the run picks up where it left off.
    resume_from = 0
    if save_trace and os.path.exists(args.trace_path):
        try:
            with open(args.trace_path) as f:
                existing = json.load(f)
            trace_agents = existing.get("agents", [])
            resume_from = len(trace_agents)
            pass_count = sum(1 for a in trace_agents if a.get("success"))
            count = resume_from
            if resume_from > 0:
                print(
                    f"[resume] Loaded {resume_from} completed agents from "
                    f"{args.trace_path}; resuming from sample "
                    f"{getattr(args, 'sample_start', 0) + resume_from}."
                )
        except Exception as exc:
            print(f"[resume] Failed to load existing trace ({exc}); starting fresh.")
            trace_agents = []
            resume_from = 0
    if args.workload == "hotpotqa":
        from src.tools.hotpotqa_tools.wikipedia import WikipediaTool, LookupTool, FinishTool
        search = WikipediaTool(name="search")
        lookup = LookupTool(name="lookup")
        finish = FinishTool(name="finish")
        tools = [search, lookup, finish]
        if use_tool_calling:
            from src.agents.ReAct.prompt.hotpotqa_toolcall import (
                get_fewshot_messages_toolcall,
                get_system_prompt_toolcall,
            )
            system_prompt = get_system_prompt_toolcall()
            fewshot_messages = get_fewshot_messages_toolcall()
            if summarize_token_threshold > 0:
                pinned_head_count = (
                    (1 if system_prompt else 0) + len(fewshot_messages) + 1
                )
                langgraph_agent_executor = create_react_agent_toolcall_summarize(
                    model,
                    tools=tools,
                    summarize_token_threshold=summarize_token_threshold,
                    pinned_head_count=pinned_head_count,
                    tokenizer=summarize_tokenizer,
                    iteration_limit=args.iteration_limit,
                    trace_callback=trace_callback,
                    print_log=print_log,
                )
            else:
                langgraph_agent_executor = create_react_agent_toolcall(
                    model, tools=tools,
                    iteration_limit=args.iteration_limit,
                    print_log=print_log,
                )
        else:
            from src.agents.ReAct.prompt.hotpotqa import get_system_prompt
            if args.fewshot > 5:
                print(f"Max fewshot examples for {args.workload} is 5. Running with 5 fewshot examples.")
            system_prompt = get_system_prompt(fewshots=min(args.fewshot, 5))
            langgraph_agent_executor = create_react_agent(
                model, tools=tools, print_log=print_log
            )

        if use_tool_calling:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            tools_schema = [convert_to_openai_tool(t) for t in tools]
        else:
            tools_schema = None

        for i in range(resume_from, samples):
            query = dataset[i]["question"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)

            if system_prompt:
                messages = [("system", system_prompt)]
                if use_tool_calling:
                    messages += fewshot_messages
                messages += [("human", query)]
            else:
                messages = [("human", query)]

            count += 1
            if trace_callback is not None:
                trace_callback.reset()
            sample_pass = False
            start_time = time.time()
            try:
                hotpotqa_tags = [args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit)]
                if summarize_token_threshold > 0:
                    hotpotqa_tags.append(f"summarize_T:{summarize_token_threshold}")
                with trace("ReAct_trace", tags=hotpotqa_tags):
                    output_dict = run_agent(
                        args=args,
                        agent=langgraph_agent_executor,
                        messages=messages,
                        label=dataset[i]['answer'],
                        evaluator=evaluator,
                        query=query,
                        extra_state={"actor_steps": 0} if use_tool_calling else None,
                        recursion_limit=args.iteration_limit * 3 if use_tool_calling else None,
                    )
                sample_pass = bool(output_dict["ispass"])
                if sample_pass:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n"+Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec")
            pretty_output(i)
            if trace_callback is not None:
                trace_agents.append(
                    {"success": sample_pass, "turns": trace_callback.snapshot_turns()}
                )
                _flush_trace(args, trace_agents, tools_schema, envelope_metadata)

    elif args.workload == "webshop":
        from src.tools.webshop_tools.webshop_tools import SearchTool, ClickTool, ResetTool, set_webshop_url
        set_webshop_url(args.webshop_url)
        reset = ResetTool()
        search = SearchTool()
        click = ClickTool()
        tools = [search, click]
        fewshot_messages = []
        if use_tool_calling:
            from src.agents.ReAct.prompt.webshop_toolcall import (
                get_fewshot_messages_toolcall,
                get_system_prompt_toolcall,
            )
            system_prompt = get_system_prompt_toolcall()
            fewshot_messages = get_fewshot_messages_toolcall()
            if summarize_token_threshold > 0:
                pinned_head_count = (
                    (1 if system_prompt else 0) + len(fewshot_messages) + 1
                )
                langgraph_agent_executor = create_react_agent_toolcall_summarize(
                    model,
                    tools=tools,
                    summarize_token_threshold=summarize_token_threshold,
                    pinned_head_count=pinned_head_count,
                    tokenizer=summarize_tokenizer,
                    iteration_limit=args.iteration_limit,
                    trace_callback=trace_callback,
                    print_log=print_log,
                )
            else:
                langgraph_agent_executor = create_react_agent_toolcall(
                    model, tools=tools,
                    iteration_limit=args.iteration_limit,
                    print_log=print_log,
                )
        else:
            from src.agents.ReAct.prompt.webshop import get_system_prompt
            if args.fewshot > 5:
                print(f"Max fewshot examples for {args.workload} is 5. Running with 5 fewshot examples.")
            system_prompt = get_system_prompt(fewshots=min(args.fewshot, 5))
            langgraph_agent_executor = create_react_agent(
                model, tools=tools, print_log=print_log
            )

        if use_tool_calling:
            from langchain_core.utils.function_calling import convert_to_openai_tool
            tools_schema = [convert_to_openai_tool(t) for t in tools]
        else:
            tools_schema = None

        for i in range(resume_from, samples):
            session_id = dataset[i]
            query = reset._run(session_id=session_id)
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            if system_prompt:
                messages = [("system", system_prompt)]
                if use_tool_calling:
                    messages += fewshot_messages
                messages += [("human", query)]
            else:
                messages = [("human", query)]

            count += 1
            if trace_callback is not None:
                trace_callback.reset()
            sample_pass = False
            start_time = time.time()
            try:
                webshop_tags = [args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]
                if summarize_token_threshold > 0:
                    webshop_tags.append(f"summarize_T:{summarize_token_threshold}")
                with trace("ReAct_trace", tags=webshop_tags):
                    output_dict = run_agent(
                        args=args,
                        agent=langgraph_agent_executor,
                        messages=messages,
                        label=None,
                        evaluator=evaluator,
                        query=query,
                        extra_state={"actor_steps": 0} if use_tool_calling else None,
                        recursion_limit=args.iteration_limit * 3 if use_tool_calling else None,
                    )
                sample_pass = bool(output_dict["ispass"])
                if sample_pass:
                    pass_count += 1

                score_sum += float(output_dict["score"])
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec\n")
            pretty_output(i)
            if trace_callback is not None:
                trace_agents.append(
                    {"success": sample_pass, "turns": trace_callback.snapshot_turns()}
                )
                _flush_trace(args, trace_agents, tools_schema, envelope_metadata)

    elif args.workload == "math":
        from src.tools.math_tools.math_tools import WolframAlphaTool, CalculatorTool, FinishTool
        from src.tools.math_tools.math_equivalence import extract_boxed_value
        from src.agents.ReAct.prompt.math import get_system_prompt
        
        tools = [WolframAlphaTool(), CalculatorTool(), FinishTool()]
        langgraph_agent_executor = create_react_agent(
            model, tools=tools, print_log=print_log
        )
        if args.fewshot > 2:
            print(f"Max fewshot examples for {args.workload} is 2. Running with 2 fewshot examples.")
        system_prompt = get_system_prompt(min(args.fewshot, 2))
        for i in range(samples):
            query = dataset[i]["problem"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            messages = [("system", system_prompt), ("human", query)]
            count += 1
            start_time = time.time()
            try:
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages,
                                            label=extract_boxed_value(dataset[i]['solution']), 
                                            evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec\n")
            pretty_output(i)

    elif args.workload == "humaneval":
        from src.tools.humaneval_tools.coding_tools import GeneratorTool, ExecutorTool, FinishTool
        from src.agents.ReAct.prompt.humaneval import HUMANEVAL_PROMPT
        language = "python"
        exe = ExecutorTool(language = language, is_leet = False)
        gen = GeneratorTool(name = "generate", llm=model)
        finish = FinishTool()
        tools = [exe, finish]
        langgraph_agent_executor = create_react_agent(
            model, tools=tools, print_log=print_log
        )
        if args.fewshot > 1:
            print(f"Max fewshot examples for {args.workload} is 1. Running with 1 fewshot example.")
        system_prompt = HUMANEVAL_PROMPT

        for i in range(samples):
            query = dataset[i]["prompt"]
            tests = dataset[i]["test"]
            entry_point = dataset[i]["entry_point"]
            print(Fore.CYAN+Style.BRIGHT+f"[Sample {i+1}/{samples}] {query}"+Style.RESET_ALL)
            messages = [("system", system_prompt), ("human", query)]
            count += 1
            start_time = time.time()
            try:
                finish.tests = tests
                finish.entry_point = entry_point
                with trace("ReAct_trace", tags=[args.workload, args.model, "Iteration_limit:"+str(args.iteration_limit), "Index:"+str(i)]):
                    exe.tests_i = gen.invoke(query)
                    output_dict = run_agent(args=args, agent=langgraph_agent_executor, messages=messages, label=None, evaluator=evaluator, query=query)
                if output_dict["ispass"]:
                    pass_count += 1
            except GraphRecursionError:
                print(Fore.RED + f"Error: The agent has reached its maximum iteration limit. Increase the iteration limit to reduce errors.\n" + Style.RESET_ALL)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print(Fore.RED + f"Error: {e}"+Style.RESET_ALL)
            end_time = time.time()
            latencies.append(end_time-start_time)
            print(f"Latency: {round(end_time-start_time, 2)} sec\n")
            pretty_output(i)

    if save_trace and trace_agents:
        _flush_trace(args, trace_agents, tools_schema, envelope_metadata)
        print(f"Saved AgentSim trace to {args.trace_path} ({len(trace_agents)} agents)")

@traceable()
def run_agent(args, agent, messages, label=None, evaluator=None, query=None,
              extra_state=None, recursion_limit=None):
    score_output = ""
    initial_input = {"messages": messages}
    if extra_state:
        initial_input.update(extra_state)
    effective_recursion_limit = (
        recursion_limit if recursion_limit is not None else args.iteration_limit
    )
    for num, chunk in enumerate(
        agent.stream(
            initial_input,
            stream_mode="values",
            config={"recursion_limit": effective_recursion_limit}
        )
    ):
        final_output = chunk
        if args.workload == "webshop":
            # Track the last purchase
            if "Your score (min 0.0, max 1.0): " in chunk['messages'][-1].content:
                score_output = chunk['messages'][-1].content
            
    
    output = parse_answer(final_output['messages'][-1].content)
    print(f'Output: {Fore.CYAN+Style.BRIGHT+output+Style.RESET_ALL}')

    score = 0.0      
    if args.workload == "webshop":
        ispass, score = evaluator(score_output)
        if ispass:
            output = score_output
            print(Fore.GREEN+f'Score: {str(score)}'+Style.RESET_ALL)
        else:
            print(Fore.RED+f'Score: {str(score)}'+Style.RESET_ALL)
    else:
        if args.workload != "humaneval":
            print(f'Label: {Fore.CYAN+Style.BRIGHT+label+Style.RESET_ALL}')
        ispass, _ = evaluator(output, label)

    if ispass:
        print(Fore.GREEN + "PASS" + Style.RESET_ALL)
    else:
        print(Fore.RED + "FAIL" + Style.RESET_ALL)
    return {"output": output, "ispass": ispass, "score": score}

