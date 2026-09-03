#!/usr/bin/env python3
"""
synthetic_networks.py -- Hand-built diagnostic networks for Graph2NL
validation.

This module defines six small synthetic networks, one for each evaluation
competency. The networks use abstract variable descriptions and simple
structures so that the target ground truth is easy to inspect and does not
depend on real-world domain knowledge.

Each test case contains:
  - id and competency identifiers
  - a short description of the test
  - a network dictionary matching the network_for_llm.json schema
  - structured ground truth used by scorer.py

The networks are constructed directly rather than estimated from data, so
their edge values are exact by construction.


Only the network itself is sent to the LLM. Evaluation metadata such as the
test id, competency name, description, and ground truth remain scorer-side
and are not included in the model input.

Usage:
    from graph2nl_core.validation.synthetic_networks import get_network
    test_case = get_network("calibration")

    from graph2nl.validation.synthetic_networks import SYNTHETIC_NETWORKS
    for test_case in SYNTHETIC_NETWORKS:
        ...

The full validation suite is normally run through:

    python3 -m graph2nl.validation.run_validation --llm-config ...

Running this file directly only prints a short preview of the available
test cases. It does not call an LLM or perform scoring.
"""

from collections import defaultdict

import numpy as np


def _min_eig_partial_corr(node_ids, edges):
    """Minimum eigenvalue of the matrix with 1s on the diagonal and
    -weight off-diagonal (0 for pairs with no edge).."""
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    p = np.eye(n)
    for s, t, w in edges:
        i, j = idx[s], idx[t]
        p[i, j] = -w
        p[j, i] = -w
    return float(np.linalg.eigvalsh(p).min())


def _assert_valid_partial_corr_network(node_ids, edges):
    min_eig = _min_eig_partial_corr(node_ids, edges)
    if min_eig <= 0:
        raise ValueError(
            f"Edge weights {edges} do not correspond to a valid "
            f"(positive-definite) partial-correlation network -- implied "
            f"precision-matrix min eigenvalue = {min_eig:.4f}. This "
            f"combination of edges cannot arise from any Gaussian Graphical "
            f"Model; reduce edge density or magnitude for these nodes."
        )


def _build_network(node_defs, edges, note_extra="", include_causal_instruction=True):
    """node_defs: list of (id, description, community). edges: list of
    (source, target, weight).
    """
    _assert_valid_partial_corr_network([nid for nid, _, _ in node_defs], edges)

    strength = defaultdict(float)
    ei = defaultdict(float)
    for s, t, w in edges:
        strength[s] += abs(w)
        strength[t] += abs(w)
        ei[s] += w
        ei[t] += w

    nodes = [
        {
            "id": nid,
            "description": desc,
            "theme": "synthetic",
            "community": comm,
            "strength_centrality": round(strength[nid], 3),
            "expected_influence": round(ei[nid], 3),
        }
        for nid, desc, comm in node_defs
    ]
    edge_objs = sorted(
        (
            {"source": s, "target": t, "weight": round(w, 3),
             "sign": "positive" if w > 0 else "negative"}
            for s, t, w in edges
        ),
        key=lambda e: -abs(e["weight"]),
    )

    note_parts = [
        "This is a cross-sectional association network (all variables measured "
        "at the same time point).",
        "Edges are partial correlations (association net of all other nodes in the "
        "network), not raw correlations. This network was constructed directly at "
        "these exact values, not estimated via regularization -- any pair not listed "
        "in the edges above has an association of exactly zero by construction, not "
        "an unreported or unmeasured association.",
    ]
    if include_causal_instruction:
        note_parts.append("No causal interpretation is warranted.")
    if note_extra:
        note_parts.append(note_extra)

    meta = {
        "dataset": "SYNTHETIC TEST NETWORK -- not real survey data",
        "run_name": "synthetic_validation_network",
        "sample_description": "Synthetic network constructed for interpretation-layer validation.",
        "method": "Hand-constructed ground-truth network (values fixed directly, no estimation involved)",
        "modularity": None,
        "note": " ".join(note_parts),
    }

    return {
        "meta": meta,
        "nodes": nodes,
        "edges": edge_objs,
    }


def _abstract(letter, note=""):
    return f"Synthetic survey item {letter} (no real-world referent; numbers only). {note}".strip()


SYNTHETIC_NETWORKS = []

# --- 1. Calibration: one edge in each of the 4 defined effect-size bands ---
SYNTHETIC_NETWORKS.append({
    "id": "calibration",
    "competency": "magnitude_label_consistency",
    "description": (
        "Star network from hub A: A-B negligible (0.02), A-C weak (0.10), A-D moderate (0.22), "
        "A-E strong (0.45). All other pairs have NO edge. Tests whether the LLM applies the "
        "prompt's own calibration thresholds (<0.05 negligible, 0.05-0.15 weak, 0.15-0.30 "
        "moderate, >0.30 strong) correctly and doesn't overstate small edges or understate large ones."
    ),
    "network": _build_network(
        [("synA", _abstract("synA"), 1), ("synB", _abstract("synB"), 1), ("synC", _abstract("synC"), 1),
         ("synD", _abstract("synD"), 1), ("synE", _abstract("synE"), 1)],
        [("synA", "synB", 0.02), ("synA", "synC", 0.10), ("synA", "synD", 0.22), ("synA", "synE", 0.45)],
    ),
    "ground_truth": {
        "type": "calibration",
        "edges": [
            {"pair": ("synA", "synB"), "weight": 0.02, "band": "negligible"},
            {"pair": ("synA", "synC"), "weight": 0.10, "band": "weak"},
            {"pair": ("synA", "synD"), "weight": 0.22, "band": "moderate"},
            {"pair": ("synA", "synE"), "weight": 0.45, "band": "strong"},
        ],
    },
})

# --- 2. Sign/direction: equal-magnitude positive vs negative edge ---
SYNTHETIC_NETWORKS.append({
    "id": "sign_direction",
    "competency": "sign_direction",
    "description": (
        "F-G = +0.30 (positive), F-H = -0.30 (negative), equal magnitude, opposite sign. "
        "No other edges. Tests whether the LLM correctly states which pair moves together and "
        "which moves in opposite directions, without flipping or hedging the sign."
    ),
    "network": _build_network(
        [("synF", _abstract("synF"), 1), ("synG", _abstract("synG"), 1), ("synH", _abstract("synH"), 1)],
        [("synF", "synG", 0.30), ("synF", "synH", -0.30)],
    ),
    "ground_truth": {
        "type": "sign",
        "edges": [
            {"pair": ("synF", "synG"), "weight": 0.30, "expected_sign": "positive"},
            {"pair": ("synF", "synH"), "weight": -0.30, "expected_sign": "negative"},
        ],
    },
})

# --- 3. Hallucination resistance: only one real edge, four zero pairs ---
SYNTHETIC_NETWORKS.append({
    "id": "hallucination",
    "competency": "unsupported_association_avoidance",
    "description": (
        "Only J-K (0.35) is a real edge among 4 nodes; J-L, J-M, K-L, K-M, L-M are all "
        "genuinely absent (zero, and NOT included in the edges list, matching how the real "
        "pipeline only exports nonzero edges). Tests whether the LLM invents an association "
        "for pairs it has literally no data on, especially L and M which have no edges at all."
    ),
    "network": _build_network(
        [("synJ", _abstract("synJ"), 1), ("synK", _abstract("synK"), 1),
         ("synL", _abstract("synL"), 2), ("synM", _abstract("synM"), 2)],
        [("synJ", "synK", 0.35)],
    ),
    "ground_truth": {
        "type": "hallucination",
        "real_edge": {"pair": ("synJ", "synK"), "weight": 0.35},
        "must_not_claim_association_between": [("synJ", "synL"), ("synJ", "synM"), ("synK", "synL"), ("synK", "synM"), ("synL", "synM")],
        # node -> community, matching the "community" field each node carries in "network"
        # above (and the numbering the LLM sees in network_for_llm.json, so it reliably
        # echoes "Community 1"/"Community 2" using these same numbers). Lets score_hallucination
        # recognize a general cross-community denial ("No edges connect Community 1 to
        # Community 2") as covering every cross-community pair, even when that sentence
        # never repeats the individual node names -- see scorer.py's _community_label_denial.
        "communities": {1: ["synJ", "synK"], 2: ["synL", "synM"]},
    },
})

# --- 4. Community/grounding narration: two clean blocks + one weak bridge ---
SYNTHETIC_NETWORKS.append({
    "id": "community_grounding",
    "competency": "community_grounding",
    "description": (
        "Two tightly-connected 3-node blocks (Alpha1/2/3 all strongly linked to each other; "
        "Beta1/2/3 all strongly linked to each other), joined by one weak bridge edge "
        "Alpha1-Beta1 (0.06). Community labels are pre-assigned in the metadata (as they would "
        "be by the real pipeline's community-detection step). Tests whether the LLM correctly "
        "reports which nodes belong to which community rather than mixing them up."
    ),
    "network": _build_network(
        [("Alpha1", _abstract("Alpha1"), 1), ("Alpha2", _abstract("Alpha2"), 1), ("Alpha3", _abstract("Alpha3"), 1),
         ("Beta1", _abstract("Beta1"), 2), ("Beta2", _abstract("Beta2"), 2), ("Beta3", _abstract("Beta3"), 2)],
        [("Alpha1", "Alpha2", 0.45), ("Alpha1", "Alpha3", 0.42), ("Alpha2", "Alpha3", 0.40),
         ("Beta1", "Beta2", 0.44), ("Beta1", "Beta3", 0.41), ("Beta2", "Beta3", 0.43),
         ("Alpha1", "Beta1", 0.06)],
    ),
    "ground_truth": {
        "type": "community",
        "communities": {1: ["Alpha1", "Alpha2", "Alpha3"], 2: ["Beta1", "Beta2", "Beta3"]},
        "bridge_edge": {"pair": ("Alpha1", "Beta1"), "weight": 0.06, "band": "weak"},
    },
})

# --- 5. Centrality nuance: high strength / near-zero expected influence ---
SYNTHETIC_NETWORKS.append({
    "id": "centrality_nuance",
    "competency": "centrality_nuance",
    "description": (
        "Hub P has 4 edges of equal magnitude (0.30) but offsetting signs (+,+,-,-), giving it "
        "the highest strength centrality (1.20) in the network but expected influence of "
        "exactly 0.0. Node Q has two edges (P-Q=0.30, U-Q=0.05) and correspondingly high "
        "expected influence relative to its strength. Tests whether the LLM explains that high "
        "connectivity is not the same as consistent directional influence, rather than "
        "conflating 'most central' with 'most positively associated with everything'."
    ),
    "network": _build_network(
        [("synP", _abstract("synP"), 1), ("synQ", _abstract("synQ"), 1), ("synR", _abstract("synR"), 1),
         ("synS", _abstract("synS"), 1), ("synT", _abstract("synT"), 1), ("synU", _abstract("synU"), 1)],
        [("synP", "synQ", 0.30), ("synP", "synR", 0.30), ("synP", "synS", -0.30), ("synP", "synT", -0.30),
         ("synU", "synQ", 0.05)],
    ),
    "ground_truth": {
        "type": "centrality",
        "high_strength_low_influence_node": {"id": "synP", "strength": 1.20, "expected_influence": 0.0},
    },
})

# --- 6. Causal-language avoidance: one very strong edge ---
SYNTHETIC_NETWORKS.append({
    "id": "causal_avoidance",
    "competency": "causal_language_avoidance",
    "description": (
        "A single very strong edge (V-W = 0.70) -- strong enough that a model might be tempted "
        "to slip into causal phrasing ('V causes W', 'V drives W'). This network's meta.note "
        "still states the FACT that the data is cross-sectional (like every other network here), "
        "but deliberately withholds the explicit 'no causal interpretation warranted' INSTRUCTION "
        "that the other five networks include as harmless generic context -- so this test measures "
        "whether the model itself recognizes that cross-sectional association doesn't license "
        "causal language, not just whether it follows a rule handed to it in its own input."
    ),
    "network": _build_network(
        [("synV", _abstract("synV"), 1), ("synW", _abstract("synW"), 1)],
        [("synV", "synW", 0.70)],
        include_causal_instruction=False,
    ),
    "ground_truth": {
        "type": "causal_language",
        "banned_terms": ["causes", "cause of", "leads to", "leading to", "drives", "driving",
                          "results in", "resulting in", "because of", "due to", "brings about"],
    },
})


def get_network(test_id):
    for t in SYNTHETIC_NETWORKS:
        if t["id"] == test_id:
            return t
    raise KeyError(f"No synthetic test network with id={test_id!r}")


if __name__ == "__main__":
    import json
    for t in SYNTHETIC_NETWORKS:
        print(f"--- {t['id']} ({t['competency']}) ---")
        print(t["description"])
        print(json.dumps(t["network"], indent=2)[:300], "...\n")
