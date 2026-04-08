"""
Inference Script for KGQA Environment
===================================
MANDATORY Environment Variables:
    API_BASE_URL       The API endpoint for the LLM (default: https://router.huggingface.co/v1)
    MODEL_NAME         The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN           Your Hugging Face / API key
    LOCAL_IMAGE_NAME   Docker image name (optional, if using from_docker_image)

STDOUT FORMAT:
    [START] task=<task_name> env=kgqa model=<model_name>
    [STEP]  step=<n> action=<tool_call> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openenv.core.env_server.mcp_types import CallToolAction

from client import KGQAEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ALL_TASKS = ["triple_completion", "inconsistency_repair", "multi_hop_qa"]
BENCHMARK = "kgqa"
MAX_STEPS = 25
TEMPERATURE = 0.2
MAX_TOKENS = 500

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent interacting with a Knowledge Graph QA environment.
    You have access to tools to explore and modify a knowledge graph.

    IMPORTANT: Respond with ONLY a JSON object, no other text. The format must be:
    {"tool_name": "<tool>", "arguments": {<args>}}

    Strategy by task type:

    FOR triple_completion:
    1. Call get_task_info — read the text carefully, it describes ALL relationships
    2. Call get_schema — learn the entity types and relation types
    3. Call query_entities to see what entities exist and their IDs
    4. Compare text against graph to identify missing triples
    5. Call add_triple for EACH missing relationship
    6. Call submit_answer ONLY AFTER adding all triples

    FOR inconsistency_repair:
    1. Call get_task_info — the text describes the CORRECT relationships
    2. Explore graph with query_entities, get_entity, get_neighbors
    3. Compare each triple in the graph against the text
    4. Call remove_triple for wrong triples, add_triple for corrections
    5. Call submit_answer ONLY AFTER all fixes

    FOR multi_hop_qa:
    1. Call get_task_info — read the question
    2. Use get_entity and get_neighbors to traverse the graph
    3. Call submit_text_answer with your final answer

    CRITICAL: You MUST add/remove triples BEFORE submitting. Submitting with 0 changes = 0 reward.
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def parse_tool_call(text: str) -> Dict[str, Any]:
    """Extract JSON tool call from LLM response, stripping markdown fences."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    parsed = json.loads(clean)
    return {"tool_name": parsed["tool_name"], "arguments": parsed.get("arguments", {})}


def get_llm_tool_call(
    client: OpenAI,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Ask the LLM which tool to call next. Returns parsed tool call dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    text = (response.choices[0].message.content or "").strip()
    return text, parse_tool_call(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SERVER_URL = os.getenv("KGQA_SERVER_URL", "https://tan291994-openenv-kgqa.hf.space")


async def make_env() -> KGQAEnv:
    """Create a fresh connected env for each episode."""
    if LOCAL_IMAGE_NAME:
        return await KGQAEnv.from_docker_image(LOCAL_IMAGE_NAME)
    env = KGQAEnv(base_url=SERVER_URL)
    await env.connect()
    return env


async def run_episode(llm: OpenAI, task_name: str) -> None:
    """Run one episode with a fresh env connection. Always emits [START]/[STEP]/[END]."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        env = await make_env()

        reset_result = await env.reset(task_type=task_name)
        obs = reset_result.observation
        obs_meta = getattr(obs, "metadata", {}) or {}

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Episode started. Task type: {task_name}\n"
                f"Observation: {json.dumps(obs_meta, default=str)}\n\n"
                f"Start by calling get_task_info to understand the task."
            )},
        ]

        done = reset_result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Ask LLM which tool to call
            try:
                raw_text, tool_call = get_llm_tool_call(llm, messages)
                messages.append({"role": "assistant", "content": raw_text})
            except Exception as e:
                err = str(e)[:120]
                log_step(step=step, action="parse_error", reward=0.0, done=False, error=err)
                rewards.append(0.0)
                steps_taken = step
                messages.append({
                    "role": "user",
                    "content": (
                        f"Error: {err}. "
                        f'Respond with ONLY: {{"tool_name": "...", "arguments": {{...}}}}'
                    ),
                })
                continue

            tool_name = tool_call["tool_name"]
            arguments = tool_call["arguments"]

            try:
                action = CallToolAction(tool_name=tool_name, arguments=arguments)
                result = await env.step(action)
                reward = result.reward or 0.0
                done = result.done
                error_msg = None
                obs = result.observation
                if hasattr(obs, "error") and obs.error:
                    error_msg = str(obs.error)
            except Exception as e:
                reward = 0.0
                done = False
                error_msg = str(e)[:120]

            rewards.append(reward)
            steps_taken = step
            action_str = f"{tool_name}({json.dumps(arguments)})" if arguments else f"{tool_name}()"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

            # Feed observation back to LLM
            obs_data: Any = {}
            try:
                if hasattr(obs, "result"):
                    obs_data = obs.result if isinstance(obs.result, dict) else {"result": str(obs.result)}
                elif hasattr(obs, "metadata"):
                    obs_data = obs.metadata or {}
            except Exception:
                obs_data = {}

            remaining = MAX_STEPS - step
            urgency = ""
            if remaining <= 3:
                urgency = " ALMOST OUT OF STEPS — submit NOW!"
            elif remaining <= 8:
                urgency = " Running low — start adding/removing triples."

            messages.append({
                "role": "user",
                "content": (
                    f"Tool '{tool_name}' returned:\n"
                    f"{json.dumps(obs_data, indent=2, default=str)}\n\n"
                    f"Step {step}/{MAX_STEPS} ({remaining} remaining).{urgency} "
                    f"What tool do you want to call next?"
                ),
            })

        # Force submit if agent ran out of steps
        if not done:
            try:
                submit_tool = "submit_text_answer" if task_name == "multi_hop_qa" else "submit_answer"
                submit_args = {"answer": "unknown"} if task_name == "multi_hop_qa" else {}
                action = CallToolAction(tool_name=submit_tool, arguments=submit_args)
                result = await env.step(action)
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken += 1
                log_step(step=steps_taken, action=f"{submit_tool}()", reward=reward, done=True, error=None)
            except Exception as e:
                print(f"[DEBUG] force-submit error: {e}", flush=True)

        score = max(rewards) if rewards else 0.01
        score = min(max(score, 0.01), 0.99)
        success = score > 0.01

    except Exception as e:
        print(f"[DEBUG] run_episode({task_name}) error: {e}", flush=True)
        score = 0.01
        success = False

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards if rewards else [0.01])


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    for task_name in ALL_TASKS:
        await run_episode(llm, task_name)


if __name__ == "__main__":
    asyncio.run(main())
