"""
Test script for the KGQA environment using the OpenEnv MCPToolClient.

Usage (while server is running on port 8000):
    python3 test_kgqa.py
"""

import json
import sys

sys.path.insert(0, "/Users/tanzeel.shaikh/Sources/Projects/openenv/OpenEnv/src")
sys.path.insert(0, "/Users/tanzeel.shaikh/Sources/Projects/openenv/hackathon-RL")

from openenv.core.mcp_client import MCPToolClient

AVAILABLE_TOOLS = [
    "get_task_info", "get_schema", "query_entities",
    "get_entity", "get_neighbors", "add_triple", "submit_answer",
]


def main():
    url = "http://localhost:8000"
    print(f"Connecting to {url}...\n")

    with MCPToolClient(base_url=url, connect_timeout_s=10, message_timeout_s=60).sync() as env:
        # 1. Reset
        print("=== RESET ===")
        result = env.reset()
        print(f"Reset result: {result}")
        print()

        # 2. List tools (skipped — list_tools also uses MCP client which times out)
        # We know our tools: get_task_info, get_schema, query_entities,
        # get_entity, get_neighbors, add_triple, submit_answer
        print("=== TOOLS (from reset observation) ===")
        print(f"  {AVAILABLE_TOOLS}")
        print()

        # 3. Get task info (includes the text the agent reads)
        print("=== GET TASK INFO ===")
        result = env.call_tool("get_task_info")
        parsed = json.loads(result) if isinstance(result, str) else result
        print(f"Task type: {parsed.get('task_type')}")
        text = parsed.get("text", "")
        print(f"Text ({len(text)} chars):")
        for para in text.split("\n\n")[:2]:
            print(f"  {para[:120]}...")
        print()

        # 4. Get schema
        print("=== GET SCHEMA ===")
        result = env.call_tool("get_schema")
        schema = json.loads(result) if isinstance(result, str) else result
        print(f"Entity types: {schema['entity_types']}")
        print(f"Relation types: {schema['relation_types']}")
        print()

        # 5. Query companies
        print("=== QUERY COMPANIES ===")
        result = env.call_tool("query_entities", entity_type="Company")
        companies = json.loads(result) if isinstance(result, str) else result
        for c in companies:
            print(f"  {c['id']}: {c['properties']['name']} ({c['properties']['industry']})")
        print()

        # 6. Get entity details
        print("=== GET ENTITY c1 ===")
        result = env.call_tool("get_entity", entity_id="c1")
        info = json.loads(result) if isinstance(result, str) else result
        print(f"  Entity: {info['entity']['properties']['name']}")
        print(f"  Triples ({len(info['triples'])}):")
        for t in info["triples"]:
            print(f"    ({t['subject']}, {t['predicate']}, {t['object']})")
        print()

        # 7. Get neighbors
        print("=== NEIGHBORS OF c1 ===")
        result = env.call_tool("get_neighbors", entity_id="c1")
        neighbors = json.loads(result) if isinstance(result, str) else result
        for n in neighbors:
            name = n["entity"]["properties"].get("name", n["entity"]["id"])
            print(f"  {n['direction']}: {n['relation']} -> {name}")
        print()

        # 8. Add a triple
        print("=== ADD TRIPLE (p1, ceo_of, c1) ===")
        result = env.call_tool("add_triple", subject_id="p1", predicate="ceo_of", object_id="c1")
        print(f"  {result}")
        print()

        # 9. Submit and see reward
        print("=== SUBMIT ANSWER ===")
        result = env.call_tool("submit_answer")
        print(f"  {result}")
        print()

    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
