---
title: KGQA Environment
emoji: 🔗
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Knowledge Graph QA Environment

OpenEnv environment for Knowledge Graph construction, repair, and reasoning.
Agents read text, explore a knowledge graph via MCP tools, and solve 3 task types.

## Tasks

| Task | Difficulty | Goal |
|------|-----------|------|
| `triple_completion` | Easy | Read text, find and add missing triples |
| `inconsistency_repair` | Medium | Find incorrect triples, remove them, add corrections |
| `multi_hop_qa` | Hard | Answer a question by traversing 2-3 graph hops |

25 instances per task. Rewards are 0.0-1.0 with partial credit.

## API

- `GET /health` — health check
- `GET /docs` — interactive API docs
- `GET /tasks` — list available tasks and action schemas
- `GET /grader` — scoring methodology for each task
- `POST /baseline` — run heuristic baseline, returns scores
- `POST /reset` — start new episode
- `POST /step` — execute tool call
- `WS /ws` — WebSocket for persistent sessions
- `GET /web/` — interactive Gradio playground

## MCP Tools

| Tool | Purpose |
|------|---------|
| `get_task_info` | Get task description, text, or question |
| `get_schema` | Entity types and relation types |
| `query_entities` | Search entities by type/properties |
| `get_entity` | Full entity details + triples |
| `get_neighbors` | Connected entities |
| `add_triple` | Add a relationship |
| `remove_triple` | Remove a relationship |
| `submit_answer` | Submit (triple_completion, inconsistency_repair) |
| `submit_text_answer` | Submit text answer (multi_hop_qa) |

## Baseline

Run the LLM-powered baseline agent:

```bash
pip install openai websockets python-dotenv
AZURE_OPENAI_URL=... AZURE_OPENAI_KEY=... python -m baseline.inference --url https://tan291994-openenv-kgqa.hf.space
```

## Action/Observation Spaces

**Action** (via MCP tool calls):
```json
{"type": "call_tool", "tool_name": "add_triple", "arguments": {"subject_id": "p1", "predicate": "ceo_of", "object_id": "c1"}}
```

**Observation** (on reset):
```json
{"instance_id": "task1_instance_0", "task_type": "triple_completion", "description": "...", "graph_summary": {"num_entities": 24, "num_triples": 27}, "available_tools": [...]}
```

**Observation** (on submit):
```json
{"done": true, "reward": 0.645, "reward_breakdown": {"precision": 0.75, "recall": 0.6, "correct": 3, "agent_total": 4, "gold_total": 5}}
```
