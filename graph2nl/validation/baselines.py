#!/usr/bin/env python3
"""
baselines.py -- Graph2NL: baselines for the LLM interpretation layer,
added in response to external review ("what does an LLM add over a
template, and over an unengineered prompt?").

Two baselines live here:

1. Deterministic template baseline (this module builds and runs it -- no
   API access needed, so it's fully reproducible in any environment,
   including a sandboxed one with no LLM access). Given only a
   network_for_llm.json-shaped dict (the exact same input interpret.py
   hands the LLM), mechanically renders one sentence per edge
   ("A and B have a {band} {sign} partial correlation (r = {weight})."),
   one sentence per community, and one sentence per node flagged as
   high-strength/low-influence -- using the network's own numeric fields,
   never anything derived from ground_truth (which the template has no
   access to, exactly like the LLM). This should score near-perfect on
   every synthetic/procedural competency BY CONSTRUCTION (it cannot
   hallucinate an edge that isn't in the input, cannot misstate a sign or
   band since it computes them directly from the numbers, and never uses
   causal language) -- that's the point: it establishes the factual-
   fidelity ceiling the LLM's richer, more readable prose is being traded
   off against. See the paper's Results section for the fidelity-vs-
   richness discussion this baseline is meant to support.

2. Naive-prompt baseline: NOT implemented here, since it requires an
   actual LLM call. See naive_prompt.py for the prompt template and
   run_validation.py's --prompt-override (or a dedicated small script)
   for how to run it once an API key is available -- this module only
   provides the template baseline, which needs none.

Usage:
    python3 -m graph2nl_core.validation.baselines --source both --out-dir baseline_results
"""
import argparse
import datetime
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
    """Pure function of the network dict alone (meta/nodes/edges) -- no
    access to any test's ground_truth, matching exactly what an LLM
    interpreter is given. Deterministic: same input always produces the
    same text, byte for byte."""
    nodes = _node_lookup(network)
    lines = []

    lines.append(
        f"This network has {len(network['nodes'])} variables and "
        f"{len(network['edges'])} nonzero partial correlations (edges); "
        f"any pair not listed below has no direct association in this network."
    )

    # 1. one sentence per edge, in the order given
    for e in network["edges"]:
        a, b = e["source"], e["target"]
        band = _band_for(e["weight"])
        sign = "positive" if e["weight"] > 0 else "negative"
        lines.append(
            f"{a} and {b} have a {band} {sign} partial correlation (r = {e['weight']:.3f})."
        )

    # 2. one sentence per community (by node["community"], skipping unassigned)
    communities = {}
    for nid, n in nodes.items():
        c = n.get("community")
        if c is not None:
            communities.setdefault(c, []).append(nid)
    for cid, members in sorted(communities.items(), key=lambda kv: str(kv[0])):
        if len(members) > 1:
            lines.append(f"Community {cid} consists of: {', '.join(members)}.")

    # 3. centrality: flag any node whose expected influence is small relative
    # to its strength (i.e. positive and negative associations largely
    # offset) -- purely numeric, computed from the node's own
    # strength_centrality/expected_influence fields, same threshold logic a
    # human reading a centrality table would apply.
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


def run_baselines(source="both", n_per_competency=10, seed=2026, out_dir="baseline_results"):
    from graph2nl_core.validation.synthetic_networks import SYNTHETIC_NETWORKS
    from graph2nl_core.validation.procedural_networks import generate_procedural_networks
    from graph2nl_core.validation.scorer import score

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
    ap.add_argument("--n-per-competency", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
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
