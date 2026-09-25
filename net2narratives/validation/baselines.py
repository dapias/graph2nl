#!/usr/bin/env python3
"""Deterministic template baseline for network interpretation.

Builds an interpretation from the same network data given to the LLM,
without using test ground truth or an API. Scoring this output checks
whether the benchmark criteria can be met from the supplied network
(paper, Appendix A.2).

Usage:
    python3 -m net2narratives.validation.baselines --source both \
        --n-per-competency 100 --seed 4242 --out-dir baseline_results
"""
import argparse
import json
import pathlib

CALIBRATION_BAND_THRESHOLDS = [
    (0.05, "negligible"),
    (0.15, "weak"),
    (0.30, "moderate"),
    (float("inf"), "strong"),
]


def _band_for(weight):
    aw = abs(weight)
    for threshold, band in CALIBRATION_BAND_THRESHOLDS:
        if aw < threshold:
            return band
    return "strong"


def _node_lookup(network):
    return {n["id"]: n for n in network["nodes"]}


def template_interpretation(network):
    """Return the same interpretation for the same network, using no ground truth."""
    nodes = _node_lookup(network)
    lines = []

    lines.append(
        f"This network has {len(network['nodes'])} variables and "
        f"{len(network['edges'])} nonzero partial correlations (edges); "
        f"any pair not listed below has no direct association in this network."
    )

    # Describe each edge.
    for e in network["edges"]:
        a, b = e["source"], e["target"]
        band = _band_for(e["weight"])
        sign = "positive" if e["weight"] > 0 else "negative"
        lines.append(
            f"{a} and {b} have a {band} {sign} partial correlation (r = {e['weight']:.3f})."
        )

    # Describe each assigned community.
    communities = {}
    for nid, n in nodes.items():
        c = n.get("community")
        if c is not None:
            communities.setdefault(c, []).append(nid)
    for cid, members in sorted(communities.items(), key=lambda kv: str(kv[0])):
        if len(members) > 1:
            lines.append(f"Community {cid} consists of: {', '.join(members)}.")

    # Identify nodes whose positive and negative edges largely offset.
    for nid, n in nodes.items():
        s = n.get("strength_centrality")
        ei = n.get("expected_influence")
        if s is None or ei is None or s == 0:
            continue
        if abs(ei) < 0.15 * s:
            lines.append(
                f"{nid} is among the most connected (central) nodes in the network "
                f"(strength centrality = {s:.3f}), but its positive and negative "
                f"associations largely offset, giving it a near-zero net expected "
                f"influence ({ei:.3f}) -- it is not consistently associated with higher "
                f"or lower values overall, despite its high connectivity."
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
