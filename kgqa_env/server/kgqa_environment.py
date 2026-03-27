"""
KGQA Environment — Knowledge Graph QA for RL training.

Three task types:
  1. Triple completion — add missing triples
  2. Inconsistency repair — find/remove wrong triples, add corrections
  3. Multi-hop QA — answer questions by graph traversal
"""

import json
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.env_server.types import Action, Observation

from models import KGQAObservation, KGQAState, TASK1_TOOLS, TASK2_TOOLS, TASK3_TOOLS
from .graph import KnowledgeGraph
from .rewards import (
    compute_triple_completion_reward,
    compute_inconsistency_repair_reward,
    compute_multi_hop_qa_reward,
)

logger = logging.getLogger(__name__)

_TOOLS_BY_TASK = {
    "triple_completion": TASK1_TOOLS,
    "inconsistency_repair": TASK2_TOOLS,
    "multi_hop_qa": TASK3_TOOLS,
}


class KGQAEnvironment(MCPEnvironment):
    """
    Knowledge Graph QA environment.

    Args:
        data_path: Path to directory containing task instance JSON files.
        max_steps: Maximum tool calls per episode (default: 30).
    """

    def __init__(self, data_path: str = "./data", max_steps: int = 30):
        mcp = FastMCP("kgqa_env")
        super().__init__(mcp)

        self.data_path = data_path
        self.max_steps = max_steps

        self._instances = self._load_instances()
        logger.info(f"Loaded {len(self._instances)} instances")

        self._shuffled = self._instances.copy()
        random.shuffle(self._shuffled)
        self._index = 0

        self._graph = KnowledgeGraph()
        self._state = KGQAState()
        self._schema: dict = {}

        # ---- Register MCP tools ----
        # Using self.tool() (not @mcp.tool) so the framework calls them
        # as direct Python functions, bypassing the MCP async client.

        @self.tool()
        def get_task_info() -> str:
            """Get the current task description, text, and/or question.

            Returns:
                JSON with task_type, description, and task-specific content.
            """
            info: dict[str, Any] = {"task_type": self._state.task_type}

            if self._state.task_type == "triple_completion":
                info["description"] = (
                    "Some relationships are missing from this knowledge graph. "
                    "Read the text below, explore the graph using the available tools, "
                    "and add the missing triples using add_triple. "
                    "When you're done, call submit_answer."
                )
                info["text"] = self._state.current_text

            elif self._state.task_type == "inconsistency_repair":
                info["description"] = (
                    "This knowledge graph contains some INCORRECT relationships. "
                    "Read the text below (which describes the correct relationships), "
                    "explore the graph to find triples that contradict the text, "
                    "remove incorrect triples using remove_triple, "
                    "and optionally add correct replacements using add_triple. "
                    "When you're done, call submit_answer."
                )
                info["text"] = self._state.current_text

            elif self._state.task_type == "multi_hop_qa":
                info["description"] = (
                    "Answer the following question by exploring the knowledge graph. "
                    "Use get_entity and get_neighbors to traverse relationships. "
                    "When you have the answer, call submit_text_answer with your answer."
                )
                info["question"] = self._state.question

            return json.dumps(info)

        @self.tool()
        def get_schema() -> str:
            """Get the knowledge graph schema: entity types and relation types.

            Returns:
                JSON with entity_types and relation_types lists.
            """
            return json.dumps(self._schema)

        @self.tool()
        def query_entities(entity_type: str = "", property_filter: str = "") -> str:
            """Search for entities by type and/or property values.

            Args:
                entity_type: Filter by type (e.g. "Company", "Person"). Empty = all.
                property_filter: JSON string of property filters, e.g. '{"name": "NovaTech"}'. Empty = no filter.

            Returns:
                JSON list of matching entities.
            """
            pf = json.loads(property_filter) if property_filter else None
            et = entity_type if entity_type else None
            results = self._graph.query_entities(entity_type=et, property_filter=pf)
            return json.dumps(results)

        @self.tool()
        def get_entity(entity_id: str) -> str:
            """Get full details of an entity by its ID.

            Args:
                entity_id: The entity ID (e.g. "c1", "p2").

            Returns:
                JSON with entity details and all its triples.
            """
            entity = self._graph.get_entity(entity_id)
            if not entity:
                return json.dumps({"error": f"Entity '{entity_id}' not found"})

            triples = self._graph.get_triples(subject_id=entity_id)
            triples += self._graph.get_triples(object_id=entity_id)

            return json.dumps({
                "entity": entity,
                "triples": [
                    {"subject": s, "predicate": p, "object": o}
                    for s, p, o in triples
                ],
            })

        @self.tool()
        def get_neighbors(entity_id: str, relation_type: str = "") -> str:
            """Get all entities connected to the given entity.

            Args:
                entity_id: The entity to find neighbors for.
                relation_type: Filter by relation type (e.g. "works_at"). Empty = all.

            Returns:
                JSON list of neighbors with relation and direction.
            """
            rt = relation_type if relation_type else None
            neighbors = self._graph.get_neighbors(entity_id, relation_type=rt)
            return json.dumps(neighbors)

        @self.tool()
        def add_triple(subject_id: str, predicate: str, object_id: str) -> str:
            """Add a relationship (triple) to the knowledge graph.

            Args:
                subject_id: The source entity ID (e.g. "p1").
                predicate: The relation type (e.g. "ceo_of", "works_at").
                object_id: The target entity ID (e.g. "c1").

            Returns:
                JSON with success status.
            """
            if not self._graph.get_entity(subject_id):
                return json.dumps({"success": False, "error": f"Entity '{subject_id}' not found"})
            if not self._graph.get_entity(object_id):
                return json.dumps({"success": False, "error": f"Entity '{object_id}' not found"})

            added = self._graph.add_triple(subject_id, predicate, object_id)
            if added:
                self._state.agent_added_triples.append([subject_id, predicate, object_id])

            return json.dumps({
                "success": added,
                "message": "Triple added" if added else "Triple already exists",
            })

        @self.tool()
        def remove_triple(subject_id: str, predicate: str, object_id: str) -> str:
            """Remove a relationship (triple) from the knowledge graph.

            Args:
                subject_id: The source entity ID (e.g. "p1").
                predicate: The relation type (e.g. "ceo_of", "works_at").
                object_id: The target entity ID (e.g. "c1").

            Returns:
                JSON with success status.
            """
            removed = self._graph.remove_triple(subject_id, predicate, object_id)
            if removed:
                self._state.agent_removed_triples.append([subject_id, predicate, object_id])

            return json.dumps({
                "success": removed,
                "message": "Triple removed" if removed else "Triple not found",
            })

        @self.tool()
        def submit_answer() -> str:
            """Submit your answer — signals that you're done modifying the graph.
            The environment will score your work.

            Returns:
                Confirmation message.
            """
            return json.dumps({
                "status": "submitted",
                "triples_added": len(self._state.agent_added_triples),
                "triples_removed": len(self._state.agent_removed_triples),
            })

        @self.tool()
        def submit_text_answer(answer: str) -> str:
            """Submit a text answer to a question (for multi-hop QA tasks).

            Args:
                answer: Your text answer to the question.

            Returns:
                Confirmation message.
            """
            self._state.agent_text_answer = answer
            return json.dumps({"status": "submitted", "answer": answer})

    def _load_instances(self) -> list[dict]:
        """Load task instances from all JSON files in the data directory."""
        all_instances = []
        for filename in ["task1_instances.json", "task2_instances.json", "task3_instances.json"]:
            path = Path(self.data_path) / filename
            if path.exists():
                with open(path) as f:
                    all_instances.extend(json.load(f))
                logger.info(f"Loaded {filename}")
        if not all_instances:
            raise FileNotFoundError(f"No instance files found in {self.data_path}")
        return all_instances

    def _get_next_instance(self, task_type: str | None = None) -> dict:
        """Get next instance, optionally filtered by task type."""
        if task_type:
            candidates = [i for i in self._instances if i["task_type"] == task_type]
            if not candidates:
                raise ValueError(f"No instances for task_type={task_type}")
            return random.choice(candidates)

        if self._index >= len(self._shuffled):
            random.shuffle(self._shuffled)
            self._index = 0
        instance = self._shuffled[self._index]
        self._index += 1
        return instance

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset for a new episode. Pass task_type in kwargs to filter."""
        task_type = kwargs.get("task_type")
        instance = self._get_next_instance(task_type)

        # Build knowledge graph
        self._graph = KnowledgeGraph()
        for entity in instance["entities"]:
            self._graph.add_entity(entity["id"], entity["type"], entity["properties"])
        for s, p, o in instance["initial_triples"]:
            self._graph.add_triple(s, p, o)

        self._schema = instance["schema"]

        self._state = KGQAState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_type=instance["task_type"],
            current_text=instance.get("text", ""),
            instance_id=instance["id"],
            # Task 1
            gold_triples=instance.get("gold_triples", []),
            agent_added_triples=[],
            # Task 2
            agent_removed_triples=[],
            injected_triples=instance.get("injected_triples", []),
            correct_replacements=instance.get("correct_replacements", []),
            # Task 3
            question=instance.get("question", ""),
            gold_answer=instance.get("gold_answer", ""),
            gold_answer_entity_id=instance.get("gold_answer_entity_id", ""),
            agent_text_answer="",
        )

        tools = _TOOLS_BY_TASK.get(instance["task_type"], TASK1_TOOLS)

        logger.info(
            f"Reset episode {self._state.episode_id}: "
            f"instance={instance['id']}, task={instance['task_type']}, "
            f"{len(instance['initial_triples'])} initial triples"
        )

        return KGQAObservation(
            done=False,
            reward=0.0,
            instance_id=instance["id"],
            task_type=instance["task_type"],
            description=instance["description"],
            graph_summary=self._graph.summary(),
            step_count=0,
            available_tools=tools.copy(),
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (error — this env is MCP-only)."""
        return KGQAObservation(
            done=False,
            reward=0.0,
            error=f"Unknown action type: {type(action).__name__}. Use CallToolAction.",
        )

    def _compute_reward(self) -> dict:
        """Route to the correct reward function based on task type."""
        task = self._state.task_type

        if task == "triple_completion":
            return compute_triple_completion_reward(
                [tuple(t) for t in self._state.agent_added_triples],
                [tuple(t) for t in self._state.gold_triples],
            )

        elif task == "inconsistency_repair":
            return compute_inconsistency_repair_reward(
                [tuple(t) for t in self._state.agent_removed_triples],
                [tuple(t) for t in self._state.agent_added_triples],
                [tuple(t) for t in self._state.injected_triples],
                [tuple(t) for t in self._state.correct_replacements],
            )

        elif task == "multi_hop_qa":
            return compute_multi_hop_qa_reward(
                self._state.agent_text_answer,
                self._state.gold_answer,
                self._state.gold_answer_entity_id,
            )

        return {"reward": 0.0, "error": f"Unknown task type: {task}"}

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute one step."""
        self._state.step_count += 1

        # Let base class handle the MCP tool call
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Check if agent submitted
        if isinstance(action, CallToolAction) and action.tool_name in (
            "submit_answer", "submit_text_answer"
        ):
            result = self._compute_reward()

            logger.info(
                f"Episode {self._state.episode_id} ended: "
                f"task={self._state.task_type}, reward={result.get('reward', 0)}"
            )

            return KGQAObservation(
                done=True,
                reward=result["reward"],
                instance_id=self._state.instance_id,
                task_type=self._state.task_type,
                step_count=self._state.step_count,
                reward_breakdown=result,
            )

        # Check max steps
        if self._state.step_count >= self.max_steps:
            logger.info(f"Episode {self._state.episode_id}: max steps reached")
            return KGQAObservation(
                done=True,
                reward=0.0,
                error=f"Max steps ({self.max_steps}) reached. Submit your answer earlier.",
            )

        return obs

    @property
    def state(self) -> KGQAState:
        """Current environment state."""
        return self._state
