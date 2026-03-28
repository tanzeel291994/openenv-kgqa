"""
KnowledgeGraph — in-memory graph data structure.

This is the core data structure for the KGQA environment.
It stores entities (nodes) and triples (edges) and provides
CRUD + query operations that MCP tools will call.

No OpenEnv dependencies — pure Python + dicts/lists.
"""

from __future__ import annotations

from typing import Any


class KnowledgeGraph:
    """
    A simple knowledge graph stored as:
    - entities: dict mapping entity_id -> {id, type, properties}
    - triples: list of (subject_id, predicate, object_id) tuples

    Example:
        >>> kg = KnowledgeGraph()
        >>> kg.add_entity("c1", "Company", {"name": "NovaTech", "founded": 2015})
        >>> kg.add_entity("p1", "Person", {"name": "Alice Chen", "role": "CEO"})
        >>> kg.add_triple("p1", "ceo_of", "c1")
        >>> kg.get_entity("p1")
        {'id': 'p1', 'type': 'Person', 'properties': {'name': 'Alice Chen', 'role': 'CEO'}}
    """

    def __init__(self) -> None:
        # entity_id -> {"id": str, "type": str, "properties": dict}
        self.entities: dict[str, dict[str, Any]] = {}
        # List of (subject_id, predicate, object_id)
        self.triples: list[tuple[str, str, str]] = []

    # ---- Entity operations ----

    def add_entity(self, entity_id: str, entity_type: str, properties: dict[str, Any]) -> None:
        """Add an entity (node) to the graph."""
        self.entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "properties": properties,
        }

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Get an entity by ID, or None if not found."""
        return self.entities.get(entity_id)

    def query_entities(
        self, entity_type: str | None = None, property_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Search entities by type and/or property values.

        Args:
            entity_type: Filter by type (e.g. "Company"). None = all types.
            property_filter: Filter by property values (e.g. {"name": "NovaTech"}).
                             None = no property filter.

        Returns:
            List of matching entity dicts.
        """
        results = []
        for entity in self.entities.values():
            # Filter by type
            if entity_type and entity["type"] != entity_type:
                continue
            # Filter by properties
            if property_filter:
                props = entity["properties"]
                if not all(props.get(k) == v for k, v in property_filter.items()):
                    continue
            results.append(entity)
        return results

    # ---- Triple operations ----

    def add_triple(self, subject_id: str, predicate: str, object_id: str) -> bool:
        """
        Add a triple (edge) to the graph.

        Returns True if added, False if it already exists.
        """
        triple = (subject_id, predicate, object_id)
        if triple in self.triples:
            return False
        self.triples.append(triple)
        return True

    def remove_triple(self, subject_id: str, predicate: str, object_id: str) -> bool:
        """
        Remove a triple from the graph.

        Returns True if removed, False if not found.
        """
        triple = (subject_id, predicate, object_id)
        if triple in self.triples:
            self.triples.remove(triple)
            return True
        return False

    def get_triples(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Get triples matching any combination of filters.
        None means "any value" for that position.

        Examples:
            get_triples(subject_id="p1")           → all triples where p1 is subject
            get_triples(predicate="ceo_of")        → all ceo_of triples
            get_triples(subject_id="p1", predicate="works_at") → p1's workplace
        """
        results = []
        for s, p, o in self.triples:
            if subject_id and s != subject_id:
                continue
            if predicate and p != predicate:
                continue
            if object_id and o != object_id:
                continue
            results.append((s, p, o))
        return results

    def get_neighbors(
        self, entity_id: str, relation_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get all entities connected to entity_id.

        Checks both directions:
        - entity_id as subject → returns objects
        - entity_id as object → returns subjects

        Args:
            entity_id: The entity to find neighbors for.
            relation_type: Filter by relation type. None = all relations.

        Returns:
            List of dicts: {"entity": entity_dict, "relation": predicate, "direction": "outgoing"|"incoming"}
        """
        neighbors = []

        for s, p, o in self.triples:
            if relation_type and p != relation_type:
                continue

            if s == entity_id:
                entity = self.entities.get(o)
                if entity:
                    neighbors.append({"entity": entity, "relation": p, "direction": "outgoing"})
            elif o == entity_id:
                entity = self.entities.get(s)
                if entity:
                    neighbors.append({"entity": entity, "relation": p, "direction": "incoming"})

        return neighbors

    # ---- Utility ----

    def summary(self) -> dict[str, Any]:
        """Quick stats about the graph."""
        return {
            "num_entities": len(self.entities),
            "num_triples": len(self.triples),
            "entity_types": list({e["type"] for e in self.entities.values()}),
            "relation_types": list({p for _, p, _ in self.triples}),
        }
