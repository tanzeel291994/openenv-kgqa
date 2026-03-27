"""
Reward functions for KGQA environment.

Three task types:
  1. Triple completion — add missing triples (precision/recall on additions)
  2. Inconsistency repair — remove wrong triples + add corrections
  3. Multi-hop QA — answer a question via graph traversal (text matching)

Partial credit via precision/recall ensures GRPO gets meaningful
gradient signal even from strong models.
"""

from __future__ import annotations


def compute_triple_completion_reward(
    agent_triples: list[tuple[str, str, str]],
    gold_triples: list[tuple[str, str, str]],
) -> dict[str, float]:
    """
    Score how well the agent filled in missing triples.

    Args:
        agent_triples: Triples the agent added during the episode.
        gold_triples: The ground-truth missing triples.

    Returns:
        Dict with breakdown:
        {
            "reward": float 0.0-1.0 (final score),
            "precision": float (what fraction of agent's triples are correct),
            "recall": float (what fraction of gold triples were found),
            "correct": int,
            "agent_total": int,
            "gold_total": int,
        }

    How scoring works:
        reward = recall * 0.7 + precision * 0.3

    Why this split?
        - Recall (70%): Finding missing triples is the main task
        - Precision (30%): Penalizes adding wrong triples (noise)
        - This creates a tension: adding more triples helps recall but can hurt precision
        - That tension = variance between rollouts = GRPO training signal
    """
    # Convert to sets for fast lookup
    agent_set = set(agent_triples)
    gold_set = set(gold_triples)

    # How many of the agent's triples are actually correct?
    correct = len(agent_set & gold_set)

    # Precision: of everything the agent added, how much was right?
    precision = correct / len(agent_set) if agent_set else 0.0

    # Recall: of everything that was missing, how much did the agent find?
    recall = correct / len(gold_set) if gold_set else 0.0

    # Weighted combination
    reward = recall * 0.7 + precision * 0.3

    return {
        "reward": round(reward, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "correct": correct,
        "agent_total": len(agent_set),
        "gold_total": len(gold_set),
    }


def compute_inconsistency_repair_reward(
    agent_removed: list[tuple[str, str, str]],
    agent_added: list[tuple[str, str, str]],
    injected_triples: list[tuple[str, str, str]],
    correct_replacements: list[tuple[str, str, str]],
) -> dict[str, float]:
    """
    Score how well the agent repaired inconsistencies.

    Two components:
      - Removal (60%): Did the agent find and remove the bad triples?
      - Replacement (40%): Did the agent add back the correct ones?

    Args:
        agent_removed: Triples the agent removed during the episode.
        agent_added: Triples the agent added during the episode.
        injected_triples: The incorrect triples that were in the graph.
        correct_replacements: The correct triples that should replace them.
    """
    removed_set = set(agent_removed)
    injected_set = set(injected_triples)
    added_set = set(agent_added)
    replacement_set = set(correct_replacements)

    # Removal scoring
    correctly_removed = len(removed_set & injected_set)
    removal_precision = correctly_removed / len(removed_set) if removed_set else 0.0
    removal_recall = correctly_removed / len(injected_set) if injected_set else 0.0
    removal_score = removal_recall * 0.7 + removal_precision * 0.3

    # Replacement scoring
    correctly_replaced = len(added_set & replacement_set)
    replacement_precision = correctly_replaced / len(added_set) if added_set else 0.0
    replacement_recall = correctly_replaced / len(replacement_set) if replacement_set else 0.0
    replacement_score = replacement_recall * 0.7 + replacement_precision * 0.3

    reward = removal_score * 0.6 + replacement_score * 0.4

    return {
        "reward": round(reward, 4),
        "removal_precision": round(removal_precision, 4),
        "removal_recall": round(removal_recall, 4),
        "removal_score": round(removal_score, 4),
        "replacement_precision": round(replacement_precision, 4),
        "replacement_recall": round(replacement_recall, 4),
        "replacement_score": round(replacement_score, 4),
        "correctly_removed": correctly_removed,
        "correctly_replaced": correctly_replaced,
        "injected_total": len(injected_set),
        "replacement_total": len(replacement_set),
    }


def compute_multi_hop_qa_reward(
    agent_answer: str,
    gold_answer: str,
    gold_answer_entity_id: str,
) -> dict[str, float]:
    """
    Score a text answer against the gold answer.

    Layered matching (highest match wins):
      1. Exact match → 1.0
      2. Entity ID match → 1.0
      3. Substring containment → 0.8
      4. Token overlap (Jaccard) → 0.3-0.7
      5. No match → 0.0

    Args:
        agent_answer: The text answer submitted by the agent.
        gold_answer: The expected text answer (e.g. "Stanford University").
        gold_answer_entity_id: The entity ID of the answer (e.g. "u1").
    """
    agent_clean = agent_answer.strip().lower()
    gold_clean = gold_answer.strip().lower()

    # 1. Exact match
    if agent_clean == gold_clean:
        return {"reward": 1.0, "match_type": "exact",
                "gold_answer": gold_answer, "agent_answer": agent_answer}

    # 2. Entity ID match
    if agent_clean == gold_answer_entity_id.strip().lower():
        return {"reward": 1.0, "match_type": "entity_id",
                "gold_answer": gold_answer, "agent_answer": agent_answer}

    # 3. Substring containment
    if gold_clean in agent_clean or agent_clean in gold_clean:
        return {"reward": 0.8, "match_type": "substring",
                "gold_answer": gold_answer, "agent_answer": agent_answer}

    # 4. Token overlap (Jaccard similarity)
    agent_tokens = set(agent_clean.split())
    gold_tokens = set(gold_clean.split())
    if agent_tokens and gold_tokens:
        intersection = len(agent_tokens & gold_tokens)
        union = len(agent_tokens | gold_tokens)
        jaccard = intersection / union
        if jaccard > 0:
            reward = round(0.3 + 0.4 * jaccard, 4)
            return {"reward": reward, "match_type": "partial",
                    "gold_answer": gold_answer, "agent_answer": agent_answer}

    # 5. No match
    return {"reward": 0.0, "match_type": "none",
            "gold_answer": gold_answer, "agent_answer": agent_answer}
