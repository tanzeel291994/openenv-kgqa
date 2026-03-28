"""
State and observation types for the KGQA environment.

KGQA tests an agent's ability to complete a knowledge graph by reading text
and using MCP tools to query/modify the graph.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server import State
from openenv.core.env_server.types import Observation


# All tool names
AVAILABLE_TOOLS = [
    "get_task_info",
    "get_schema",
    "query_entities",
    "get_entity",
    "get_neighbors",
    "add_triple",
    "remove_triple",
    "submit_answer",
    "submit_text_answer",
]

# Task-specific tool subsets — returned in reset observation
TASK1_TOOLS = [
    "get_task_info", "get_schema", "query_entities", "get_entity",
    "get_neighbors", "add_triple", "submit_answer",
]
TASK2_TOOLS = [
    "get_task_info", "get_schema", "query_entities", "get_entity",
    "get_neighbors", "add_triple", "remove_triple", "submit_answer",
]
TASK3_TOOLS = [
    "get_task_info", "get_schema", "query_entities", "get_entity",
    "get_neighbors", "submit_text_answer",
]


class KGQAObservation(Observation):
    """
    Observation returned by the KGQA environment.

    Fields added here (beyond done/reward/metadata) get serialized
    into the HTTP response's "observation" dict.
    """

    instance_id: str = ""
    task_type: str = ""
    description: str = ""
    graph_summary: Dict[str, Any] = {}
    step_count: int = 0
    available_tools: List[str] = []
    # For step responses (tool results flow through CallToolObservation,
    # but these fields appear on submit/termination)
    reward_breakdown: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class KGQAState(State):
    """
    Internal environment state for tracking the current episode.

    Inherited from State: episode_id, step_count
    """

    task_type: str = ""              # "triple_completion", "inconsistency_repair", "multi_hop_qa"
    current_text: str = ""           # Natural language text describing the graph
    instance_id: str = ""            # Which instance we're on

    # Task 1: triple completion
    gold_triples: list = []          # Triples the agent must find
    agent_added_triples: list = []   # Triples the agent has added

    # Task 2: inconsistency repair
    agent_removed_triples: list = [] # Triples the agent has removed
    injected_triples: list = []      # Incorrect triples in the graph (gold: what to remove)
    correct_replacements: list = []  # Correct triples that should replace the injected ones

    # Task 3: multi-hop QA
    question: str = ""               # Natural language question
    gold_answer: str = ""            # Expected text answer
    gold_answer_entity_id: str = ""  # Entity ID of the answer
    agent_text_answer: str = ""      # What the agent submitted
