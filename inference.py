"""
Baseline inference script for the KGQA environment.

Uses Azure OpenAI API to run an LLM agent that plays all 3 task types.
The agent reads task info, explores the knowledge graph, and submits answers.

Usage:
    # Against local server
    AZURE_OPENAI_URL=... AZURE_OPENAI_KEY=... python -m baseline.inference

    # Against HF Spaces
    AZURE_OPENAI_URL=... AZURE_OPENAI_KEY=... python -m baseline.inference --url https://tan291994-openenv-kgqa.hf.space

Environment Variables:
    AZURE_OPENAI_URL: Required. Your Azure OpenAI endpoint URL.
    AZURE_OPENAI_KEY: Required. Your Azure OpenAI subscription key.
"""

import argparse
import json
import os
import sys
from typing import Any

# Load .env file if present (so you don't need to export vars manually)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars work fine without it

# Azure OpenAI client
try:
    from openai import AzureOpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# WebSocket client for persistent sessions
try:
    import websockets.sync.client as ws_client
except ImportError:
    print("ERROR: websockets package not installed. Run: pip install websockets")
    sys.exit(1)


# -------------------------------------------------------------------
# Azure OpenAI config
# -------------------------------------------------------------------

MODEL_NAME = "gpt-4o"
DEPLOYMENT = "azure-gpt-4o"
API_VERSION = "2024-12-01-preview"


# -------------------------------------------------------------------
# Environment client — thin wrapper around the WebSocket protocol
# -------------------------------------------------------------------

class KGQAClient:
    """
    Connects to the KGQA environment via WebSocket.

    Why WebSocket instead of HTTP?
      HTTP /reset and /step are stateless — each request creates a fresh
      environment, so state from reset is lost by the time step is called.
      WebSocket maintains a persistent session across the full episode.

    Protocol:
      Send: {"type": "reset", "data": {"task_type": "..."}}
      Send: {"type": "step",  "data": {"type": "call_tool", "tool_name": "...", "arguments": {...}}}
      Recv: {"type": "observation", "data": {"observation": {...}, "reward": ..., "done": ...}}
    """

    def __init__(self, base_url: str):
        ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
        self.ws_url = f"{ws_url}/ws"
        self.ws = None

    def connect(self):
        self.ws = ws_client.connect(self.ws_url)

    def close(self):
        if self.ws:
            self.ws.close()

    def reset(self, task_type: str | None = None) -> dict:
        """Start a new episode. Returns the initial observation."""
        data = {"task_type": task_type} if task_type else {}
        self.ws.send(json.dumps({"type": "reset", "data": data}))
        response = json.loads(self.ws.recv())
        return response["data"]

    def call_tool(self, tool_name: str, arguments: dict | None = None) -> dict:
        """Call an MCP tool. Returns the full response data."""
        action = {
            "type": "call_tool",
            "tool_name": tool_name,
            "arguments": arguments or {},
        }
        self.ws.send(json.dumps({"type": "step", "data": action}))
        response = json.loads(self.ws.recv())
        return response["data"]


# -------------------------------------------------------------------
# LLM agent — uses Azure OpenAI to decide what tool to call
# -------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI agent interacting with a Knowledge Graph QA environment.
You have access to tools to explore and modify a knowledge graph.

IMPORTANT: Respond with ONLY a JSON object, no other text. The format must be:
{"tool_name": "<tool>", "arguments": {<args>}}

Strategy by task type:

FOR triple_completion:
1. Call get_task_info — read the text carefully, it describes ALL relationships including missing ones
2. Call get_schema — learn the entity types and relation types
3. Call query_entities to see what entities exist and their IDs
4. Compare the text against the graph to identify missing triples
5. Call add_triple for EACH missing relationship you find in the text
6. Call submit_answer ONLY AFTER you have added all triples you found

FOR inconsistency_repair:
1. Call get_task_info — the text describes the CORRECT relationships
2. Explore the graph with query_entities, get_entity, get_neighbors
3. Compare each triple in the graph against the text
4. Call remove_triple for triples that contradict the text
5. Call add_triple for the correct versions
6. Call submit_answer ONLY AFTER removing wrong triples and adding corrections

FOR multi_hop_qa:
1. Call get_task_info — read the question
2. Use get_entity and get_neighbors to traverse the graph step by step
3. Call submit_text_answer with your final answer

CRITICAL RULES:
- You MUST add/remove triples BEFORE submitting. Submitting with 0 changes = 0 reward.
- Do NOT submit early. Spend most of your steps adding triples, not just exploring.
- Keep exploration brief (3-5 calls), then act on what you learned from the text."""


def run_llm_agent(
    client: KGQAClient,
    openai_client: AzureOpenAI,
    task_type: str,
    max_steps: int = 25,
) -> dict:
    """
    Run one episode with the LLM agent.

    Loop:
      1. Reset environment for the given task type
      2. Show the LLM the observation
      3. LLM picks a tool call (as JSON)
      4. Execute it via WebSocket
      5. Feed result back to LLM
      6. Repeat until done or max_steps
    """
    reset_data = client.reset(task_type=task_type)
    obs = reset_data["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Episode started. Task type: {obs.get('task_type', 'unknown')}\n"
            f"Description: {obs.get('description', '')}\n"
            f"Graph summary: {json.dumps(obs.get('graph_summary', {}))}\n"
            f"Available tools: {json.dumps(obs.get('available_tools', []))}\n\n"
            f"Start by calling get_task_info to understand the task."
        )},
    ]

    result = None

    for step in range(max_steps):
        # Ask the LLM what tool to call
        response = openai_client.chat.completions.create(
            model=DEPLOYMENT,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )

        assistant_msg = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_msg})

        # Parse the LLM's JSON tool call
        try:
            clean = assistant_msg
            # Strip markdown code fences if the LLM wraps its response
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            tool_call = json.loads(clean)
            tool_name = tool_call["tool_name"]
            arguments = tool_call.get("arguments", {})
        except (json.JSONDecodeError, KeyError) as e:
            messages.append({
                "role": "user",
                "content": (
                    f"Error parsing your response: {e}. "
                    f"Respond with ONLY a JSON object: "
                    f'{{\"tool_name\": \"...\", \"arguments\": {{...}}}}'
                ),
            })
            continue

        # Execute the tool call against the environment
        result = client.call_tool(tool_name, arguments)

        # Check if episode is done (agent submitted or max env steps)
        if result.get("done", False):
            return result

        # Feed the tool output back to the LLM for the next decision
        tool_output = result.get("observation", {})
        remaining = max_steps - step - 1
        urgency = ""
        if remaining <= 3:
            urgency = " ⚠️ ALMOST OUT OF STEPS — submit NOW if you haven't already!"
        elif remaining <= 8:
            urgency = " ⚠️ Running low on steps — start adding/removing triples if you haven't."

        messages.append({
            "role": "user",
            "content": (
                f"Tool '{tool_name}' returned:\n"
                f"{json.dumps(tool_output, indent=2)}\n\n"
                f"Step {step + 1}/{max_steps} ({remaining} remaining).{urgency} "
                f"What tool do you want to call next?"
            ),
        })

    # Ran out of steps — force submit so we get a score
    if task_type == "multi_hop_qa":
        result = client.call_tool("submit_text_answer", {"answer": "unknown"})
    else:
        result = client.call_tool("submit_answer", {})
    return result or {"done": True, "reward": 0.0, "observation": {}}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KGQA baseline agent (Azure OpenAI)")
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL of the KGQA environment server",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per task (default: 3)",
    )
    args = parser.parse_args()

    # Verify Azure credentials
    endpoint = os.getenv("AZURE_OPENAI_URL")
    key = os.getenv("AZURE_OPENAI_KEY")
    if not endpoint or not key:
        print("ERROR: Set AZURE_OPENAI_URL and AZURE_OPENAI_KEY environment variables")
        sys.exit(1)

    openai_client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=API_VERSION,
    )

    print(f"Model: {MODEL_NAME} (deployment: {DEPLOYMENT})")
    print(f"Server: {args.url}")
    print(f"Episodes per task: {args.episodes}")
    print()

    task_types = ["triple_completion", "inconsistency_repair", "multi_hop_qa"]
    all_results = {}

    for task_type in task_types:
        print(f"=== Task: {task_type} ===")
        scores = []

        for ep in range(args.episodes):
            client = KGQAClient(args.url)
            try:
                client.connect()
                result = run_llm_agent(client, openai_client, task_type)
                reward = result.get("reward") or 0.0  # coerce None → 0.0
                scores.append(reward)

                breakdown = result.get("observation", {}).get("reward_breakdown", {})
                print(f"  Episode {ep + 1}: reward={reward:.4f}  {breakdown}")
            except Exception as e:
                print(f"  Episode {ep + 1}: ERROR — {e}")
                scores.append(0.0)
            finally:
                client.close()

        mean = sum(scores) / len(scores) if scores else 0.0
        all_results[task_type] = {
            "scores": scores,
            "mean": round(mean, 4),
        }
        print(f"  Mean: {mean:.4f}\n")

    # Summary
    print("=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    for task_type, data in all_results.items():
        print(f"  {task_type}: mean={data['mean']:.4f}  scores={data['scores']}")

    output = {
        "model": MODEL_NAME,
        "deployment": DEPLOYMENT,
        "server": args.url,
        "episodes_per_task": args.episodes,
        "results": all_results,
    }
    print(f"\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
