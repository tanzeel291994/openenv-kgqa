"""
Data generator for KGQA environment.

Creates task instances for three task types:
  1. Triple completion — find and add missing triples
  2. Inconsistency repair — find wrong triples, remove + replace them
  3. Multi-hop QA — answer questions by traversing the graph

Run directly to generate all instance files:
    PYTHONPATH=. python -m data.generator
"""

from __future__ import annotations

import json
import random
from pathlib import Path


# ---- Schema constants ----

ENTITY_TYPES = ["Company", "Person", "Product", "Location", "University"]

RELATION_TYPES = [
    "ceo_of", "works_at", "founded_by", "makes_product",
    "headquartered_in", "located_in", "acquired", "invested_in",
    "attended", "born_in", "has_revenue", "partner_of",
]

SCHEMA = {
    "entity_types": ENTITY_TYPES,
    "relation_types": RELATION_TYPES,
}


# ---- Gold graph ----

def build_gold_graph() -> dict:
    """
    Build a complete knowledge graph about fictional tech companies.

    ~21 entities, ~30 triples — enough combinatorial variety for 25 instances
    per task type.

    Returns dict with:
        entities: list of {id, type, properties}
        triples: list of [subject_id, predicate, object_id]
    """
    entities = [
        # Companies
        {"id": "c1", "type": "Company", "properties": {"name": "NovaTech", "founded": 2015, "industry": "AI"}},
        {"id": "c2", "type": "Company", "properties": {"name": "DataFlow", "founded": 2018, "industry": "Data Analytics"}},
        {"id": "c3", "type": "Company", "properties": {"name": "CloudPeak", "founded": 2012, "industry": "Cloud Infrastructure"}},
        {"id": "c4", "type": "Company", "properties": {"name": "QuantumLeap", "founded": 2020, "industry": "Quantum Computing"}},
        {"id": "c5", "type": "Company", "properties": {"name": "BioSynth", "founded": 2016, "industry": "Biotech"}},
        # People
        {"id": "p1", "type": "Person", "properties": {"name": "Alice Chen", "role": "CEO"}},
        {"id": "p2", "type": "Person", "properties": {"name": "Bob Rao", "role": "CTO"}},
        {"id": "p3", "type": "Person", "properties": {"name": "Carol Zhang", "role": "VP Engineering"}},
        {"id": "p4", "type": "Person", "properties": {"name": "David Kim", "role": "Founder"}},
        {"id": "p5", "type": "Person", "properties": {"name": "Elena Ruiz", "role": "Lead Scientist"}},
        {"id": "p6", "type": "Person", "properties": {"name": "Frank Liu", "role": "CEO"}},
        {"id": "p7", "type": "Person", "properties": {"name": "Grace Patel", "role": "CTO"}},
        {"id": "p8", "type": "Person", "properties": {"name": "Henry Okafor", "role": "Research Director"}},
        # Products
        {"id": "pr1", "type": "Product", "properties": {"name": "NovaDB", "category": "Database"}},
        {"id": "pr2", "type": "Product", "properties": {"name": "FlowEngine", "category": "ETL Pipeline"}},
        {"id": "pr3", "type": "Product", "properties": {"name": "PeakCloud", "category": "Cloud Platform"}},
        {"id": "pr4", "type": "Product", "properties": {"name": "QuantumSDK", "category": "Development Kit"}},
        {"id": "pr5", "type": "Product", "properties": {"name": "BioAnalyzer", "category": "Analysis Tool"}},
        # Locations
        {"id": "loc1", "type": "Location", "properties": {"name": "San Francisco", "country": "USA"}},
        {"id": "loc2", "type": "Location", "properties": {"name": "Austin", "country": "USA"}},
        {"id": "loc3", "type": "Location", "properties": {"name": "Seattle", "country": "USA"}},
        # Universities
        {"id": "u1", "type": "University", "properties": {"name": "Stanford University"}},
        {"id": "u2", "type": "University", "properties": {"name": "MIT"}},
        {"id": "u3", "type": "University", "properties": {"name": "UC Berkeley"}},
    ]

    triples = [
        # Leadership & employment
        ["p1", "ceo_of", "c1"],           # Alice is CEO of NovaTech
        ["p6", "ceo_of", "c4"],           # Frank is CEO of QuantumLeap
        ["p4", "founded_by", "c2"],       # David founded DataFlow
        ["p2", "works_at", "c1"],         # Bob works at NovaTech
        ["p3", "works_at", "c2"],         # Carol works at DataFlow
        ["p5", "works_at", "c1"],         # Elena works at NovaTech
        ["p7", "works_at", "c4"],         # Grace works at QuantumLeap
        ["p8", "works_at", "c5"],         # Henry works at BioSynth
        # Products
        ["c1", "makes_product", "pr1"],   # NovaTech makes NovaDB
        ["c2", "makes_product", "pr2"],   # DataFlow makes FlowEngine
        ["c3", "makes_product", "pr3"],   # CloudPeak makes PeakCloud
        ["c4", "makes_product", "pr4"],   # QuantumLeap makes QuantumSDK
        ["c5", "makes_product", "pr5"],   # BioSynth makes BioAnalyzer
        # Locations
        ["c1", "headquartered_in", "loc1"],  # NovaTech in SF
        ["c2", "headquartered_in", "loc2"],  # DataFlow in Austin
        ["c3", "headquartered_in", "loc1"],  # CloudPeak in SF
        ["c4", "headquartered_in", "loc3"],  # QuantumLeap in Seattle
        ["c5", "headquartered_in", "loc2"],  # BioSynth in Austin
        # Business relations
        ["c1", "acquired", "c2"],            # NovaTech acquired DataFlow
        ["c3", "invested_in", "c1"],         # CloudPeak invested in NovaTech
        ["c1", "partner_of", "c3"],          # NovaTech partners with CloudPeak
        ["c4", "invested_in", "c5"],         # QuantumLeap invested in BioSynth
        ["c3", "partner_of", "c4"],          # CloudPeak partners with QuantumLeap
        # Education
        ["p1", "attended", "u1"],    # Alice → Stanford
        ["p2", "attended", "u2"],    # Bob → MIT
        ["p4", "attended", "u1"],    # David → Stanford
        ["p6", "attended", "u3"],    # Frank → UC Berkeley
        ["p7", "attended", "u2"],    # Grace → MIT
        ["p8", "attended", "u3"],    # Henry → UC Berkeley
        # Birthplace
        ["p1", "born_in", "loc1"],   # Alice born in SF
        ["p5", "born_in", "loc2"],   # Elena born in Austin
        ["p6", "born_in", "loc3"],   # Frank born in Seattle
    ]

    return {"entities": entities, "triples": triples}


# ---- Text generation ----

def generate_text_from_graph(entities: list[dict], triples: list[list[str]]) -> str:
    """
    Generate natural language text that describes the full graph.
    Covers ALL triples so the agent can reconstruct removed/corrupted info.
    """
    names = {e["id"]: e["properties"].get("name", e["id"]) for e in entities}

    paragraphs = []

    paragraphs.append(
        f"{names['c1']} is an AI company founded in 2015, headquartered in {names['loc1']}. "
        f"The company's flagship product is {names['pr1']}, a high-performance database. "
        f"{names['p1']} serves as CEO, while {names['p2']} is the CTO. "
        f"{names['p5']} works at {names['c1']} as Lead Scientist."
    )

    paragraphs.append(
        f"{names['c2']} specializes in data analytics and was founded by {names['p4']} in 2018. "
        f"The company is headquartered in {names['loc2']} and develops {names['pr2']}, "
        f"an ETL pipeline tool. {names['p3']} serves as VP of Engineering at {names['c2']}."
    )

    paragraphs.append(
        f"{names['c3']}, a cloud infrastructure provider founded in 2012, "
        f"is headquartered in {names['loc1']}. Their main product is {names['pr3']}. "
        f"{names['c3']} made an early investment in {names['c1']}, "
        f"and the two companies maintain a strategic partnership."
    )

    paragraphs.append(
        f"{names['c4']} is a quantum computing startup founded in 2020, "
        f"headquartered in {names['loc3']}. {names['p6']} is the CEO and "
        f"{names['p7']} serves as CTO. Their product {names['pr4']} is a development kit "
        f"for quantum applications. {names['c4']} invested in {names['c5']}, "
        f"and {names['c3']} and {names['c4']} have a strategic partnership."
    )

    paragraphs.append(
        f"{names['c5']} is a biotech company founded in 2016, headquartered in {names['loc2']}. "
        f"{names['p8']} is the Research Director and leads development of {names['pr5']}, "
        f"their flagship analysis tool."
    )

    paragraphs.append(
        f"In a major industry move, {names['c1']} acquired {names['c2']}, "
        f"bringing {names['c2']}'s analytics capabilities in-house."
    )

    paragraphs.append(
        f"{names['p1']}, who was born in {names['loc1']}, attended {names['u1']} "
        f"before building her career in tech. {names['p2']} is an {names['u2']} graduate. "
        f"{names['p4']} also attended {names['u1']} before founding {names['c2']}. "
        f"{names['p6']}, born in {names['loc3']}, attended {names['u3']}. "
        f"{names['p7']} graduated from {names['u2']}. "
        f"{names['p8']} studied at {names['u3']}. "
        f"{names['p5']} was born in {names['loc2']}."
    )

    return "\n\n".join(paragraphs)


# ---- Task 1: Triple Completion ----

def create_task1_instances(num_instances: int = 25, seed: int = 42) -> list[dict]:
    """Create triple-completion instances. Remove 3-6 triples per instance."""
    rng = random.Random(seed)
    gold = build_gold_graph()
    text = generate_text_from_graph(gold["entities"], gold["triples"])

    instances = []
    all_triples = gold["triples"]

    for i in range(num_instances):
        num_to_remove = rng.randint(3, 6)
        shuffled = all_triples.copy()
        rng.shuffle(shuffled)
        removed = shuffled[:num_to_remove]
        remaining = [t for t in all_triples if t not in removed]

        instances.append({
            "id": f"task1_instance_{i}",
            "task_type": "triple_completion",
            "description": (
                "Some relationships are missing from this knowledge graph. "
                "Read the text, explore the graph, and add the missing triples."
            ),
            "text": text,
            "entities": gold["entities"],
            "initial_triples": remaining,
            "gold_triples": removed,
            "schema": SCHEMA,
        })

    return instances


# ---- Task 2: Inconsistency Repair ----

def _corrupt_triple(
    triple: list[str],
    all_triples: list[list[str]],
    entities: list[dict],
    rng: random.Random,
) -> list[str]:
    """
    Create an incorrect version of a triple via one of:
      - Wrong predicate
      - Wrong object (same type)
      - Wrong subject (same type)
    """
    s, p, o = triple
    entity_map = {e["id"]: e for e in entities}
    strategy = rng.choice(["wrong_predicate", "wrong_object", "wrong_subject"])

    if strategy == "wrong_predicate":
        other_preds = [r for r in RELATION_TYPES if r != p]
        new_p = rng.choice(other_preds)
        return [s, new_p, o]

    elif strategy == "wrong_object":
        o_type = entity_map[o]["type"]
        same_type = [e["id"] for e in entities if e["type"] == o_type and e["id"] != o]
        if same_type:
            return [s, p, rng.choice(same_type)]
        # Fallback to wrong predicate
        other_preds = [r for r in RELATION_TYPES if r != p]
        return [s, rng.choice(other_preds), o]

    else:  # wrong_subject
        s_type = entity_map[s]["type"]
        same_type = [e["id"] for e in entities if e["type"] == s_type and e["id"] != s]
        if same_type:
            return [rng.choice(same_type), p, o]
        other_preds = [r for r in RELATION_TYPES if r != p]
        return [s, rng.choice(other_preds), o]


def create_task2_instances(num_instances: int = 25, seed: int = 100) -> list[dict]:
    """
    Create inconsistency-repair instances.

    For each instance, corrupt 2-4 triples in the graph. The agent must
    find the incorrect triples, remove them, and add the correct ones.
    """
    rng = random.Random(seed)
    gold = build_gold_graph()
    text = generate_text_from_graph(gold["entities"], gold["triples"])

    instances = []
    all_triples = gold["triples"]

    for i in range(num_instances):
        num_to_corrupt = rng.randint(2, 4)
        shuffled = all_triples.copy()
        rng.shuffle(shuffled)
        originals = shuffled[:num_to_corrupt]

        # Create corrupted versions
        corrupted = []
        for orig in originals:
            bad = _corrupt_triple(orig, all_triples, gold["entities"], rng)
            # Ensure the corrupted triple is actually different
            while bad == orig:
                bad = _corrupt_triple(orig, all_triples, gold["entities"], rng)
            corrupted.append(bad)

        # Build initial triples: replace originals with corrupted versions
        initial = [t for t in all_triples if t not in originals] + corrupted

        instances.append({
            "id": f"task2_instance_{i}",
            "task_type": "inconsistency_repair",
            "description": (
                "This knowledge graph contains some INCORRECT relationships. "
                "Read the text, find the wrong triples, remove them, and add corrections."
            ),
            "text": text,
            "entities": gold["entities"],
            "initial_triples": initial,
            "injected_triples": corrupted,         # Bad triples (what to remove)
            "correct_replacements": originals,     # Correct triples (what to add back)
            "gold_triples": [],                    # Not used for Task 2
            "schema": SCHEMA,
        })

    return instances


# ---- Task 3: Multi-hop QA ----

def _build_question_templates(gold: dict) -> list[dict]:
    """
    Build concrete question instances from templates + gold graph.
    Each question requires 2-3 hops to answer.
    """
    entities = {e["id"]: e for e in gold["entities"]}
    names = {eid: e["properties"].get("name", eid) for eid, e in entities.items()}

    # Index triples for lookup: (subject, predicate) -> object
    sp_to_o = {}
    for s, p, o in gold["triples"]:
        sp_to_o[(s, p)] = o
    # Reverse index: (predicate, object) -> subject
    po_to_s = {}
    for s, p, o in gold["triples"]:
        po_to_s.setdefault((p, o), []).append(s)

    questions = []

    # Template 1: "Which university did the CEO of {company} attend?"
    # Hop 1: find CEO → Hop 2: find university
    for person_id in ["p1", "p6"]:
        company_id = sp_to_o.get((person_id, "ceo_of"))
        uni_id = sp_to_o.get((person_id, "attended"))
        if company_id and uni_id:
            questions.append({
                "question": f"Which university did the CEO of {names[company_id]} attend?",
                "gold_answer": names[uni_id],
                "gold_answer_entity_id": uni_id,
                "hop_count": 2,
            })

    # Template 2: "Where is the company that {person} works at headquartered?"
    # Hop 1: find company → Hop 2: find HQ
    for person_id in ["p2", "p3", "p5", "p7", "p8"]:
        company_id = sp_to_o.get((person_id, "works_at"))
        if company_id:
            loc_id = sp_to_o.get((company_id, "headquartered_in"))
            if loc_id:
                questions.append({
                    "question": f"Where is the company that {names[person_id]} works at headquartered?",
                    "gold_answer": names[loc_id],
                    "gold_answer_entity_id": loc_id,
                    "hop_count": 2,
                })

    # Template 3: "What product does the company headquartered in {location} that works in {industry} make?"
    for company in gold["entities"]:
        if company["type"] != "Company":
            continue
        cid = company["id"]
        loc_id = sp_to_o.get((cid, "headquartered_in"))
        prod_id = sp_to_o.get((cid, "makes_product"))
        if loc_id and prod_id:
            industry = company["properties"].get("industry", "")
            questions.append({
                "question": f"What product does the {industry} company headquartered in {names[loc_id]} make?",
                "gold_answer": names[prod_id],
                "gold_answer_entity_id": prod_id,
                "hop_count": 2,
            })

    # Template 4: "Who is the CEO of the company that acquired {company}?"
    # Hop 1: find acquirer → Hop 2: find CEO
    for s, p, o in gold["triples"]:
        if p == "acquired":
            acquirer = s
            acquired = o
            ceo_candidates = po_to_s.get(("ceo_of", acquirer), [])
            for ceo_id in ceo_candidates:
                questions.append({
                    "question": f"Who is the CEO of the company that acquired {names[acquired]}?",
                    "gold_answer": names[ceo_id],
                    "gold_answer_entity_id": ceo_id,
                    "hop_count": 2,
                })

    # Template 5: "Which university did the founder of {company} attend?"
    for s, p, o in gold["triples"]:
        if p == "founded_by":
            founder_id = s
            company_id = o
            uni_id = sp_to_o.get((founder_id, "attended"))
            if uni_id:
                questions.append({
                    "question": f"Which university did the founder of {names[company_id]} attend?",
                    "gold_answer": names[uni_id],
                    "gold_answer_entity_id": uni_id,
                    "hop_count": 2,
                })

    # Template 6: "Where was the CTO of {company} born?" (if birthplace exists)
    # Not all CTOs have born_in, but try
    for person_id in ["p2", "p7"]:
        company_id = sp_to_o.get((person_id, "works_at"))
        birth_loc = sp_to_o.get((person_id, "born_in"))
        if company_id and birth_loc:
            questions.append({
                "question": f"Where was the CTO of {names[company_id]} born?",
                "gold_answer": names[birth_loc],
                "gold_answer_entity_id": birth_loc,
                "hop_count": 2,
            })

    # Template 7 (3-hop): "Which university did the CEO of the company that invested in {company} attend?"
    for s, p, o in gold["triples"]:
        if p == "invested_in":
            investor = s
            investee = o
            ceo_candidates = po_to_s.get(("ceo_of", investor), [])
            for ceo_id in ceo_candidates:
                uni_id = sp_to_o.get((ceo_id, "attended"))
                if uni_id:
                    questions.append({
                        "question": (
                            f"Which university did the CEO of the company "
                            f"that invested in {names[investee]} attend?"
                        ),
                        "gold_answer": names[uni_id],
                        "gold_answer_entity_id": uni_id,
                        "hop_count": 3,
                    })

    # Template 8 (3-hop): "What product does the company that acquired the company founded by {person} make?"
    for s, p, o in gold["triples"]:
        if p == "founded_by":
            founder = s
            founded_co = o
            # Find who acquired this company
            for s2, p2, o2 in gold["triples"]:
                if p2 == "acquired" and o2 == founded_co:
                    acquirer = s2
                    prod_id = sp_to_o.get((acquirer, "makes_product"))
                    if prod_id:
                        questions.append({
                            "question": (
                                f"What product does the company that acquired "
                                f"the company founded by {names[founder]} make?"
                            ),
                            "gold_answer": names[prod_id],
                            "gold_answer_entity_id": prod_id,
                            "hop_count": 3,
                        })

    # Template 9: "Where is the company that makes {product} headquartered?"
    for s, p, o in gold["triples"]:
        if p == "makes_product":
            company_id = s
            product_id = o
            loc_id = sp_to_o.get((company_id, "headquartered_in"))
            if loc_id:
                questions.append({
                    "question": f"Where is the company that makes {names[product_id]} headquartered?",
                    "gold_answer": names[loc_id],
                    "gold_answer_entity_id": loc_id,
                    "hop_count": 2,
                })

    # Template 10: "Who works at the company that is partnered with {company}?"
    for s, p, o in gold["triples"]:
        if p == "partner_of":
            partner_a = s
            partner_b = o
            workers = po_to_s.get(("works_at", partner_b), [])
            for w in workers:
                questions.append({
                    "question": f"Name someone who works at the company partnered with {names[partner_a]}.",
                    "gold_answer": names[w],
                    "gold_answer_entity_id": w,
                    "hop_count": 2,
                })

    return questions


def create_task3_instances(num_instances: int = 25, seed: int = 200) -> list[dict]:
    """
    Create multi-hop QA instances.

    Each instance provides the complete graph (no removals) and a question
    that requires 2-3 hops to answer. No text — pure graph traversal.
    """
    rng = random.Random(seed)
    gold = build_gold_graph()

    all_questions = _build_question_templates(gold)
    rng.shuffle(all_questions)

    # If we have fewer unique questions than needed, cycle through them
    instances = []
    for i in range(num_instances):
        q = all_questions[i % len(all_questions)]
        instances.append({
            "id": f"task3_instance_{i}",
            "task_type": "multi_hop_qa",
            "description": (
                "Answer the question by exploring the knowledge graph. "
                "Use get_entity and get_neighbors to traverse relationships."
            ),
            "text": "",  # No text for Task 3
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "gold_answer_entity_id": q["gold_answer_entity_id"],
            "hop_count": q["hop_count"],
            "entities": gold["entities"],
            "initial_triples": gold["triples"],  # Complete graph
            "gold_triples": [],                   # Not used for Task 3
            "schema": SCHEMA,
        })

    return instances


# ---- Main ----

if __name__ == "__main__":
    output_dir = Path(__file__).parent

    for task_num, creator, filename in [
        (1, create_task1_instances, "task1_instances.json"),
        (2, create_task2_instances, "task2_instances.json"),
        (3, create_task3_instances, "task3_instances.json"),
    ]:
        instances = creator()
        path = output_dir / filename
        with open(path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"Task {task_num}: {len(instances)} instances → {path}")

        for inst in instances[:3]:  # Show first 3
            print(f"  {inst['id']}: task_type={inst['task_type']}")
            if inst.get("gold_triples"):
                print(f"    gold_triples: {len(inst['gold_triples'])}")
            if inst.get("injected_triples"):
                print(f"    injected: {len(inst['injected_triples'])}")
            if inst.get("question"):
                print(f"    Q: {inst['question']}")
                print(f"    A: {inst['gold_answer']}")
        if len(instances) > 3:
            print(f"  ... and {len(instances) - 3} more")
