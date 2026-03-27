"""
FastAPI server for the KGQA environment.

Environment Variables:
    KGQA_DATA_PATH: Path to data directory (default: ./data)
    KGQA_MAX_STEPS: Maximum tool calls per episode (default: 30)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from pydantic import field_validator

from models import TASK1_TOOLS, TASK2_TOOLS, TASK3_TOOLS
from .kgqa_environment import KGQAEnvironment

logger = logging.getLogger(__name__)

# Default data path: kgqa_env/data/ (sibling of server/)
_DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "data")
DATA_PATH = os.environ.get("KGQA_DATA_PATH", _DEFAULT_DATA)
MAX_STEPS = int(os.environ.get("KGQA_MAX_STEPS", "30"))


def _env_factory():
    """Create a new KGQAEnvironment for each session."""
    return KGQAEnvironment(data_path=DATA_PATH, max_steps=MAX_STEPS)


class KGQACallToolAction(CallToolAction):
    """Handles JSON strings from web UI (same pattern as finqa_env)."""

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Dict[str, Any]:
        if isinstance(v, str):
            return json.loads(v)
        return v


# Use our KGQACallToolAction so the Gradio web UI can send arguments
# as JSON strings (text fields). WebSocket clients send proper dicts
# which pass through the validator unchanged. list_tools routing breaks
# but it already times out via MCP — agents get tools from reset obs.
app = create_app(
    _env_factory, KGQACallToolAction, CallToolObservation, env_name="kgqa_env"
)


# ---------------------------------------------------------------
# Hackathon-required endpoints: /tasks, /grader, /baseline
# These are custom routes added AFTER create_app(). The hackathon's
# automated checker hits these to verify the environment is valid.
# ---------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    """
    GET /tasks — List available tasks and their metadata.

    The hackathon checker uses this to discover what tasks exist,
    their difficulty, and what tools the agent can use. Nemotron
    (the Phase 2 LLM judge) also reads this to understand the env.
    """
    return {
        "tasks": [
            {
                "task_type": "triple_completion",
                "difficulty": "easy",
                "description": (
                    "Some relationships are missing from the knowledge graph. "
                    "Read the text, explore the graph, and add the missing triples."
                ),
                "available_tools": TASK1_TOOLS,
                "submission_tool": "submit_answer",
                "scoring": "reward = recall * 0.7 + precision * 0.3",
                "num_instances": 25,
            },
            {
                "task_type": "inconsistency_repair",
                "difficulty": "medium",
                "description": (
                    "The knowledge graph contains incorrect relationships. "
                    "Read the text, find wrong triples, remove them, and add corrections."
                ),
                "available_tools": TASK2_TOOLS,
                "submission_tool": "submit_answer",
                "scoring": "reward = removal_score * 0.6 + replacement_score * 0.4",
                "num_instances": 25,
            },
            {
                "task_type": "multi_hop_qa",
                "difficulty": "hard",
                "description": (
                    "Answer a question by traversing 2-3 hops in the knowledge graph. "
                    "No text provided — pure graph reasoning."
                ),
                "available_tools": TASK3_TOOLS,
                "submission_tool": "submit_text_answer",
                "scoring": "Layered text matching: exact=1.0, substring=0.8, partial=0.3-0.7",
                "num_instances": 25,
            },
        ],
        "action_schema": {
            "type": "call_tool",
            "tool_name": "string (one of the available_tools)",
            "arguments": "dict (tool-specific arguments)",
        },
    }


@app.get("/grader")
def get_grader_info():
    """
    GET /grader — Describe how grading works for each task.

    Grading happens automatically when the agent calls submit_answer
    or submit_text_answer during step(). The reward (0.0-1.0) is
    returned in the observation's "reward" field with a full breakdown
    in "reward_breakdown".

    This endpoint documents the scoring methodology so the checker
    can verify graders exist and produce valid 0.0-1.0 scores.
    """
    return {
        "grading_method": "automatic",
        "grading_trigger": "Agent calls submit_answer or submit_text_answer",
        "score_range": [0.0, 1.0],
        "reward_location": "observation.reward (float) + observation.reward_breakdown (dict)",
        "tasks": {
            "triple_completion": {
                "difficulty": "easy",
                "formula": "recall * 0.7 + precision * 0.3",
                "metrics": {
                    "precision": "correct_triples / agent_added_triples",
                    "recall": "correct_triples / gold_triples",
                },
                "rationale": (
                    "Recall weighted higher because finding missing triples "
                    "is the primary objective. Precision penalizes noise."
                ),
            },
            "inconsistency_repair": {
                "difficulty": "medium",
                "formula": "removal_score * 0.6 + replacement_score * 0.4",
                "metrics": {
                    "removal_score": "removal_recall * 0.7 + removal_precision * 0.3",
                    "replacement_score": "replacement_recall * 0.7 + replacement_precision * 0.3",
                },
                "rationale": (
                    "Removal weighted higher because identifying errors is "
                    "harder than adding corrections (text makes corrections obvious)."
                ),
            },
            "multi_hop_qa": {
                "difficulty": "hard",
                "formula": "Layered matching: exact > entity_id > substring > token_overlap",
                "metrics": {
                    "exact_match": "1.0 if answer matches gold exactly",
                    "entity_id_match": "1.0 if answer matches gold entity ID",
                    "substring_match": "0.8 if one contains the other",
                    "token_overlap": "0.3 + 0.4 * jaccard_similarity",
                    "no_match": "0.0",
                },
                "rationale": (
                    "Layered matching handles answer variations (e.g. "
                    "'Stanford' vs 'Stanford University') without an LLM judge."
                ),
            },
        },
    }


@app.post("/baseline")
def run_baseline():
    """
    POST /baseline — Run the baseline agent and return scores.

    Creates a temporary environment, runs 3 episodes per task using
    a simple heuristic agent (no LLM needed), and returns the scores.

    A separate baseline/inference.py uses OpenAI API for a smarter
    baseline. This endpoint uses a rule-based agent for fast, reliable
    scoring that doesn't require an API key.
    """
    from .rewards import (
        compute_triple_completion_reward,
        compute_inconsistency_repair_reward,
        compute_multi_hop_qa_reward,
    )

    env = _env_factory()
    results = {}
    episodes_per_task = 3

    for task_type in ["triple_completion", "inconsistency_repair", "multi_hop_qa"]:
        scores = []
        for _ in range(episodes_per_task):
            obs = env.reset(task_type=task_type)

            # Simple heuristic agent: call get_task_info, then submit immediately.
            # This gets a low but non-zero score — proves the pipeline works.
            from openenv.core.env_server.mcp_types import CallToolAction as CTA

            # Step 1: get_task_info
            action = CTA(type="call_tool", tool_name="get_task_info", arguments={})
            env.step(action)

            # Step 2: get_schema
            action = CTA(type="call_tool", tool_name="get_schema", arguments={})
            env.step(action)

            # Step 3: submit (baseline just submits without doing real work)
            if task_type == "multi_hop_qa":
                action = CTA(
                    type="call_tool",
                    tool_name="submit_text_answer",
                    arguments={"answer": "unknown"},
                )
            else:
                action = CTA(type="call_tool", tool_name="submit_answer", arguments={})

            result = env.step(action)
            scores.append(result.reward)

        results[task_type] = {
            "difficulty": {"triple_completion": "easy", "inconsistency_repair": "medium", "multi_hop_qa": "hard"}[task_type],
            "episodes_run": episodes_per_task,
            "scores": scores,
            "mean_score": round(sum(scores) / len(scores), 4),
        }

    return {
        "baseline_type": "heuristic (rule-based, no LLM)",
        "note": "See baseline/inference.py for OpenAI API-powered baseline with higher scores",
        "results": results,
    }
