#!/usr/bin/env python3
"""
baselines.py -- Net2Narratives: deterministic template baseline.

Given only the network dictionary (meta/nodes/edges) that the LLM also
receives, this module renders one sentence per edge ("A and B have a
{band} {sign} partial correlation (r = {weight})."), using the protocol's
magnitude bands; one sentence per community (including single-node
communities); and one sentence per node whose positive and negative edges
largely offset, calling it the most connected node only if it has the
highest strength. It never reads a test's ground truth. Absent edges are
described as zero by construction for constructed networks and as "not
retained" for estimated ones (those with meta.n).

Because every statement is computed directly from the supplied numbers, the
template cannot invent edges, misstate signs or magnitude bands, or use
causal language. Scoring it with the same evaluator therefore checks that
each competency's criterion can be satisfied from the supplied network,
i.e. that a perfect score is attainable by construction (paper,
Appendix A.2). No API access is needed.

Usage (matching the paper's benchmark settings):
    python3 -m net2narratives.validation.baselines --source both \
        --n-per-competency 100 --seed 4242 --out-dir baseline_results
"""
import argparse
import datetime
import json
import pathlib

# Magnitude bands exactly as defined in the full protocol:
# negligible |w| < 0.05; weak 0.05 <= |w| < 0.15;
# moderate 0.15 <= |w| <= 0.30; strong |w| > 0.30.
def _band_for(weight):
    aw = abs(weight)
    if aw < 0.05:
        return "negligible"
    if aw < 0.15:
        return "weak"
    if aw <= 0.30:
        return "moderate"
    return "strong"


# A node's positive and negative edges "largely offset" when its expected
# influence is small relative to its strength.
OFFSET_RATIO = 0.15


def _is_estimated(network):
    """True for estimated networks (sample size given), False for networks
    whose values are fixed by construction. Only affects how absent edges
    are described."""
    return network.get("meta", {}).get("n") is not None


def template_interpretation(network):
    """Render a deterministic interpretation from the network alone.

    Uses only meta/nodes/edges -- never a test's ground truth. The same
    input always produces the same text.
    """
    nodes = network["nodes"]
    edges = network["edges"]
    lines = []

    if _is_estimated(network):
        absent = ("pairs not listed below have no retained edge in this estimated "
                  "network, which does not by itself show that they are unrelated")
    else:
        absent = "any pair not listed below has no direct association in this network"
    lines.append(
        f"This network has {len(nodes)} variables and {len(edges)} nonzero partial "
        f"correlations (edges); {absent}."
    )

    # 1. One sentence per edge, in the order given.
    for e in edges:
        band = _band_for(e["weight"])
        sign = "positive" if e["weight"] > 0 else "negative"
        lines.append(
            f"{e['source']} and {e['target']} have a {band} {sign} partial "
            f"correlation (r = {e['weight']:.3f})."
        )

    # 2. One sentence per community, including single-node communities.
    communities = {}
    for n in nodes:
        c = n.get("community")
        if c is not None:
            communities.setdefault(c, []).append(n["id"])
    for cid, members in sorted(communities.items(), key=lambda kv: str(kv[0])):
        if len(members) == 1:
            lines.append(f"Community {cid} consists of a single variable: {members[0]}.")
        else:
            lines.append(f"Community {cid} consists of: {', '.join(members)}.")

    # 3. One sentence per node whose positive and negative edges largely
    # offset. "Most connected" is stated only if the node has the highest
    # strength in the network.
    strengths = [n.get("strength_centrality") for n in nodes
                 if n.get("strength_centrality") is not None]
    max_strength = max(strengths) if strengths else None
    for n in nodes:
        s = n.get("strength_centrality")
        ei = n.get("expected_influence")
        if s is None or ei is None or s == 0 or abs(ei) >= OFFSET_RATIO * s:
            continue
        if s == max_strength:
            rank = "has the highest strength centrality in the network"
        else:
            rank = "has a strength centrality"
        lines.append(
            f"{n['id']} {rank} (strength = {s:.3f}), but its positive and negative "
            f"associations largely offset, giving it a near-zero expected influence "
            f"({ei:.3f}); its connections do not add up to a consistent net "
            f"association with higher or lower values."
        )

    lines.append(
        "This is a cross-sectional partial-correlation network, so these are "
        "associations, not evidence of causation."
    )
    return "\n".join(lines)


def run_baselines(source="both", n_per_competency=100, seed=4242, out_dir="baseline_results"):
    from net2narratives.validation.synthetic_networks import SYNTHETIC_NETWORKS
    from net2narratives.validation.procedural_networks import generate_procedural_networks
    from net2narratives.validation.scorer import score

    tests = []
    if source in ("synthetic", "both"):
        tests += list(SYNTHETIC_NETWORKS)
    if source in ("procedural", "both"):
        tests += generate_procedural_networks(n_per_competency, seed)

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    for test_case in tests:
        text = template_interpretation(test_case["network"])
        result = score(test_case, text)
        result["baseline"] = "deterministic_template"
        result["source"] = test_case.get("source", "synthetic")
        result["timestamp"] = "N/A (deterministic, no wall-clock dependency)"
        all_results.append(result)

        case_dir = out_path / test_case["id"]
        case_dir.mkdir(exist_ok=True)
        with open(case_dir / "output.md", "w") as f:
            f.write(text)
        with open(case_dir / "score.json", "w") as f:
            json.dump(result, f, indent=2)

    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    by_comp = {}
    for r in all_results:
        by_comp.setdefault(r["competency"], []).append(r)
    table = []
    for comp, results in by_comp.items():
        pass_rate = sum(r["passed"] for r in results) / len(results)
        avg_score = sum(r["score"] for r in results) / len(results)
        table.append({"competency": comp, "n": len(results), "pass_rate": pass_rate, "avg_score": avg_score})

    return all_results, table, summary_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "procedural", "both"], default="both")
    ap.add_argument("--n-per-competency", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out-dir", default="baseline_results")
    args = ap.parse_args()

    all_results, table, summary_path = run_baselines(
        args.source, args.n_per_competency, args.seed, args.out_dir
    )

    print(f"{'='*76}\nDeterministic template baseline -- {len(all_results)} networks scored "
          f"(source={args.source})\n{'='*76}")
    for row in table:
        print(f"  {row['competency']:28s} n={row['n']:3d}  pass_rate={row['pass_rate']:.0%}  "
              f"avg_score={row['avg_score']:.3f}")
    print(f"\nWrote {summary_path} and per-test output/score under {args.out_dir}/")


if __name__ == "__main__":
    main()
