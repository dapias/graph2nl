#!/usr/bin/env python3
"""
Generate validation networks for the six graph2nl competencies.

These cases vary node counts, edges, weights, and community structure.

Each generated test case contains:
  - id and competency identifiers
  - a short description of the test
  - a network dictionary matching the network_for_llm.json schema
  - structured ground truth used by scorer.py

Ground truth comes from the construction parameters; edges are not estimated
from data. Only the network is sent to the LLM. Descriptions and ground truth
remain with the scorer.

Usage:
    from graph2nl.validation.procedural_networks import \
        generate_procedural_networks

    networks = generate_procedural_networks(
        n_per_competency=15,
        seed=...,
    )

The procedural validation suite is normally run through:

    python3 -m graph2nl.validation.run_validation \
        --llm-config ... --source procedural

Running this file directly previews or exports cases without LLM scoring.
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np

CALIBRATION_BANDS = {
    "negligible": (0.01, 0.04),
    "weak": (0.06, 0.14),
    "moderate": (0.16, 0.29),
    "strong": (0.32, 0.60),
}


def _abstract(letter, note=""):
    return f"Synthetic survey item {letter} (no real-world referent; numbers only). {note}".strip()


def _min_eig_partial_corr(node_ids, edges):
    """Minimum eigenvalue of the implied precision matrix.

    The diagonal is 1, and each off-diagonal entry is minus the edge weight
    (or 0 for an absent edge). Positivity makes the partial-correlation
    network compatible with a Gaussian graphical model.
    """
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    p = np.eye(n)
    for s, t, w in edges:
        i, j = idx[s], idx[t]
        p[i, j] = -w
        p[j, i] = -w
    return float(np.linalg.eigvalsh(p).min())


def _is_valid_partial_corr_network(node_ids, edges):
    return _min_eig_partial_corr(node_ids, edges) > 0


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
    """Build a valid network without depending on synthetic_networks.py.

    Causal tests omit the explicit causal warning, but retain the
    cross-sectional description. Invalid edge weights raise ValueError.
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
        "dataset": "SYNTHETIC TEST NETWORK (procedurally generated) -- not real survey data",
        "run_name": "synthetic_validation_network",
        "sample_description": "Synthetic network (procedurally generated) constructed for interpretation-layer validation.",
        "method": "Procedurally generated ground-truth network (values fixed directly, no estimation involved)",
        "modularity": None,
        "note": " ".join(note_parts),
    }
    # Sample size is inapplicable to constructed networks, so omit meta.n.

    return {
        "meta": meta,
        "nodes": nodes,
        "edges": edge_objs,
    }


def _node_label(idx):
    """Use short IDs that the scorer can match in the model's response."""
    return str(idx)


# 1. Calibration: a star with 2-4 edges drawn from distinct magnitude bands.
def _gen_calibration(rng, idx):
    n_edges = rng.choice([2, 3, 3, 4])  # weight towards 3-4 for a fuller test
    bands = rng.sample(list(CALIBRATION_BANDS.keys()), k=n_edges)
    hub = _node_label("A")
    node_defs = [(hub, _abstract(hub), 1)]
    edges = []
    gt_edges = []
    for i, band in enumerate(bands):
        spoke = _node_label(chr(ord("B") + i))
        node_defs.append((spoke, _abstract(spoke), 1))
        lo, hi = CALIBRATION_BANDS[band]
        w = round(rng.uniform(lo, hi), 3)
        edges.append((hub, spoke, w))
        gt_edges.append({"pair": (hub, spoke), "weight": w, "band": band})
    return {
        "id": f"proc_calibration_{idx:03d}",
        "competency": "magnitude_label_consistency",
        "source": "procedural",
        "params": {"n_edges": n_edges, "bands": bands, "seed_draw": idx},
        "description": (
            f"Procedurally generated star network from hub {hub} with {n_edges} spokes "
            f"spanning bands {bands}. Tests magnitude-label consistency language at "
            f"randomized node count/magnitude."
        ),
        "network": _build_network(node_defs, edges),
        "ground_truth": {"type": "calibration", "edges": gt_edges},
    }


# 2. Sign: a star with 2-4 positive and negative edges.
# A star is valid when the sum of squared edge weights is below 1.
# Redraw the whole star if that condition fails.
_MAX_SIGN_RETRIES = 500


def _gen_sign(rng, idx):
    n_pairs = rng.choice([2, 3, 3, 4])
    hub = _node_label("F")
    node_defs = [(hub, _abstract(hub), 1)]
    spokes = [_node_label(chr(ord("G") + i)) for i in range(n_pairs)]
    node_defs += [(spoke, _abstract(spoke), 1) for spoke in spokes]
    node_ids = [hub] + spokes

    for attempt in range(_MAX_SIGN_RETRIES):
        edges = []
        gt_edges = []
        for spoke in spokes:
            sign = rng.choice(["positive", "negative"])
            mag = round(rng.uniform(0.18, 0.55), 3)
            w = mag if sign == "positive" else -mag
            edges.append((hub, spoke, w))
            gt_edges.append({"pair": (hub, spoke), "weight": w, "expected_sign": sign})
        # Ensure both signs appear so the test distinguishes them.
        signs_present = {e["expected_sign"] for e in gt_edges}
        if len(signs_present) == 1 and n_pairs >= 2:
            gt_edges[-1]["expected_sign"] = "negative" if gt_edges[-1]["expected_sign"] == "positive" else "positive"
            flipped_pair = gt_edges[-1]["pair"]
            edges = [(s, t, -w if (s, t) == flipped_pair else w) for s, t, w in edges]
            gt_edges[-1]["weight"] = -gt_edges[-1]["weight"]
        if _is_valid_partial_corr_network(node_ids, edges):
            break
    else:
        raise RuntimeError(
            f"Could not draw a valid partial-correlation star for hub {hub} "
            f"with {n_pairs} spokes after {_MAX_SIGN_RETRIES} attempts."
        )
    return {
        "id": f"proc_sign_{idx:03d}",
        "competency": "sign_direction",
        "source": "procedural",
        "params": {"n_pairs": n_pairs, "seed_draw": idx},
        "description": (
            f"Procedurally generated hub network from {hub} with {n_pairs} spokes of "
            f"randomized sign/magnitude. Tests direction-reporting at randomized scale."
        ),
        "network": _build_network(node_defs, edges),
        "ground_truth": {"type": "sign", "edges": gt_edges},
    }


# 3. Hallucination resistance: one edge; all other pairs have weight zero.
def _gen_hallucination(rng, idx):
    n_nodes = rng.choice([4, 5, 6, 6, 7, 8])
    labels = [_node_label(chr(ord("J") + i)) for i in range(n_nodes)]
    # The linked pair is community 1; the isolated nodes are community 2.
    real_a, real_b = labels[0], labels[1]
    node_defs = [(real_a, _abstract(real_a), 1), (real_b, _abstract(real_b), 1)]
    node_defs += [(l, _abstract(l), 2) for l in labels[2:]]
    w = round(rng.uniform(0.20, 0.55), 3)
    edges = [(real_a, real_b, w)]
    all_pairs = [(labels[i], labels[j]) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    must_not = [p for p in all_pairs if p != (real_a, real_b)]
    return {
        "id": f"proc_hallucination_{idx:03d}",
        "competency": "unsupported_association_avoidance",
        "source": "procedural",
        "params": {"n_nodes": n_nodes, "seed_draw": idx},
        "description": (
            f"Procedurally generated {n_nodes}-node network with a single real edge "
            f"({real_a}-{real_b}) and {len(must_not)} genuinely-absent pairs. Tests "
            f"resistance to inventing unlisted associations at randomized network size."
        ),
        "network": _build_network(node_defs, edges),
        "ground_truth": {
            "type": "hallucination",
            "real_edge": {"pair": (real_a, real_b), "weight": w},
            "must_not_claim_association_between": must_not,
            "communities": {1: [real_a, real_b], 2: labels[2:]},
        },
    }


# 4. Community structure: 2-3 dense blocks with one weak bridge.
# Redraw invalid blocks, then redraw the full network if the bridge makes
# their combined precision matrix non-positive-definite.
_MAX_BLOCK_RETRIES = 500
_MAX_NETWORK_RETRIES = 200


def _gen_community(rng, idx):
    n_blocks = rng.choice([2, 2, 3])
    block_size = rng.choice([3, 4])
    node_defs = []
    blocks = []
    for b in range(n_blocks):
        block_prefix = chr(ord("A") + b)
        members = [_node_label(f"{block_prefix}{i+1}") for i in range(block_size)]
        blocks.append(members)
        for m in members:
            node_defs.append((m, _abstract(m), b + 1))
    node_ids = [nid for nid, _, _ in node_defs]
    bridge_pair = (blocks[0][0], blocks[1][0])
    bridge_lo, bridge_hi = CALIBRATION_BANDS["weak"]

    for network_attempt in range(_MAX_NETWORK_RETRIES):
        edges = []
        for members in blocks:
            for attempt in range(_MAX_BLOCK_RETRIES):
                block_edges = []
                for i in range(block_size):
                    for j in range(i + 1, block_size):
                        lo, hi = CALIBRATION_BANDS["strong"] if rng.random() < 0.5 else CALIBRATION_BANDS["moderate"]
                        w = round(rng.uniform(lo, hi), 3)
                        block_edges.append((members[i], members[j], w))
                if _is_valid_partial_corr_network(members, block_edges):
                    break
            else:
                raise RuntimeError(
                    f"Could not draw a valid partial-correlation block for "
                    f"{members} after {_MAX_BLOCK_RETRIES} attempts -- "
                    f"CALIBRATION_BANDS['strong']/['moderate'] may be too wide "
                    f"for a fully-connected block of size {block_size}."
                )
            edges.extend(block_edges)
        bridge_w = round(rng.uniform(bridge_lo, bridge_hi), 3)
        edges.append((bridge_pair[0], bridge_pair[1], bridge_w))
        if _is_valid_partial_corr_network(node_ids, edges):
            break
    else:
        raise RuntimeError(
            f"Could not draw a jointly-valid partial-correlation network for "
            f"blocks {blocks} after {_MAX_NETWORK_RETRIES} attempts."
        )

    communities = {b + 1: blocks[b] for b in range(n_blocks)}
    return {
        "id": f"proc_community_{idx:03d}",
        "competency": "community_grounding",
        "source": "procedural",
        "params": {"n_blocks": n_blocks, "block_size": block_size, "seed_draw": idx},
        "description": (
            f"Procedurally generated {n_blocks}-block network (block size {block_size}) "
            f"joined by one weak bridge edge {bridge_pair}. Tests accurate community "
            f"narration at randomized block count/size."
        ),
        "network": _build_network(node_defs, edges),
        "ground_truth": {
            "type": "community",
            "communities": communities,
            "bridge_edge": {"pair": bridge_pair, "weight": bridge_w, "band": "weak"},
        },
    }


# 5. Centrality: equal positive and negative edges cancel in influence.
def _gen_centrality(rng, idx):
    k = rng.choice([2, 2, 3])
    hub = _node_label("P")
    mag = round(rng.uniform(0.20, 0.35), 3)
    node_defs = [(hub, _abstract(hub), 1)]
    edges = []
    for i in range(k):
        pos_node = _node_label(f"pos{i+1}")
        node_defs.append((pos_node, _abstract(pos_node), 1))
        edges.append((hub, pos_node, mag))
    for i in range(k):
        neg_node = _node_label(f"neg{i+1}")
        node_defs.append((neg_node, _abstract(neg_node), 1))
        edges.append((hub, neg_node, -mag))
    strength = round(2 * k * mag, 3)
    return {
        "id": f"proc_centrality_{idx:03d}",
        "competency": "centrality_nuance",
        "source": "procedural",
        "params": {"k": k, "magnitude": mag, "seed_draw": idx},
        "description": (
            f"Procedurally generated hub {hub} with {k} positive and {k} negative edges "
            f"of equal magnitude ({mag}) -- strength={strength}, expected influence=0. "
            f"Tests strength-vs-influence distinction at randomized degree/magnitude."
        ),
        "network": _build_network(node_defs, edges),
        "ground_truth": {
            "type": "centrality",
            "high_strength_low_influence_node": {"id": hub, "strength": strength, "expected_influence": 0.0},
        },
    }


# 6. Causal language: vary the V-W weight and omit the explicit causal
# warning from the network note. Node labels and banned terms stay fixed.
_BANNED_TERMS = ["causes", "cause of", "leads to", "leading to", "drives", "driving",
                 "results in", "resulting in", "because of", "due to", "brings about"]


def _gen_causal(rng, idx):
    a = _node_label("V")
    b = _node_label("W")
    w = round(rng.uniform(0.45, 0.75), 3)
    return {
        "id": f"proc_causal_avoidance_{idx:03d}",
        "competency": "causal_language_avoidance",
        "source": "procedural",
        "params": {"magnitude": w, "seed_draw": idx},
        "description": (
            f"Procedurally generated single very-strong edge ({a}-{b} = {w}) testing "
            f"causal-language avoidance at randomized magnitude."
        ),
        "network": _build_network(
            [(a, _abstract(a), 1), (b, _abstract(b), 1)],
            [(a, b, w)],
            include_causal_instruction=False,
        ),
        "ground_truth": {"type": "causal_language", "banned_terms": _BANNED_TERMS},
    }


_GENERATORS = {
    "magnitude_label_consistency": _gen_calibration,
    "sign_direction": _gen_sign,
    "unsupported_association_avoidance": _gen_hallucination,
    "community_grounding": _gen_community,
    "centrality_nuance": _gen_centrality,
    "causal_language_avoidance": _gen_causal,
}


def generate_procedural_networks(n_per_competency=10, seed=2026):
    """Return a reproducible list of cases for a given count and seed."""
    if n_per_competency < 1:
        raise ValueError("n_per_competency must be at least 1")
    rng = random.Random(seed)
    out = []
    for competency, gen_fn in _GENERATORS.items():
        for i in range(n_per_competency):
            out.append(gen_fn(rng, i))
    return out


def get_network(test_id, networks=None):
    networks = networks if networks is not None else generate_procedural_networks()
    for t in networks:
        if t["id"] == test_id:
            return t
    raise KeyError(f"No procedural test network with id={test_id!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-competency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", help="If given, write the full battery as JSON to this path")
    args = ap.parse_args()

    if args.n_per_competency < 1:
        ap.error("--n-per-competency must be at least 1")

    nets = generate_procedural_networks(args.n_per_competency, args.seed)
    print(f"Generated {len(nets)} procedural networks "
          f"({args.n_per_competency} per competency x {len(_GENERATORS)} competencies), seed={args.seed}")
    by_comp = defaultdict(list)
    for t in nets:
        by_comp[t["competency"]].append(t)
    for comp, ts in by_comp.items():
        sizes = [len(t["network"]["nodes"]) for t in ts]
        edges = [len(t["network"]["edges"]) for t in ts]
        print(f"  {comp:28s} n={len(ts):3d}  node counts {min(sizes)}-{max(sizes)}  "
              f"edge counts {min(edges)}-{max(edges)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(nets, f, indent=2)
        print(f"\nWrote {args.out}")
