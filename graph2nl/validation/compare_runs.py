#!/usr/bin/env python3
"""
compare_runs.py -- Graph2NL: aggregate several run_validation.py summary.json
files (different models, and/or the naive-prompt baseline) into one
comparison table, for the cross-model and naive-vs-engineered-prompt
comparisons requested in external review.

Does NOT call any LLM itself -- it only reads summary.json files that
run_validation.py has already written. Run run_validation.py once per
config first, e.g.:

    graph2nl-validate --llm-config examples/cross_model/llm_config_gpt_oss_120b.yaml \\
        --source both --out-dir results_gpt_oss_120b
    graph2nl-validate --llm-config examples/cross_model/llm_config_qwen.yaml \\
        --source both --out-dir results_qwen
    graph2nl-validate --llm-config examples/cross_model/llm_config_glm.yaml \\
        --source both --out-dir results_glm
    graph2nl-validate --llm-config examples/cross_model/llm_config_naive_prompt.yaml \\
        --source both --out-dir results_naive_prompt

Then:
    python3 -m graph2nl_core.validation.compare_runs \\
        --run gpt-oss-120b=results_gpt_oss_120b/summary.json \\
        --run qwen3.5-397b=results_qwen/summary.json \\
        --run glm-4.7=results_glm/summary.json \\
        --run naive-prompt=results_naive_prompt/summary.json \\
        --out comparison_table.md

This produces the paper's Results-section comparison table (one row per
competency, one column per run) plus an overall row, in Markdown, ready to
paste into the paper.
"""
import argparse
import json
from collections import defaultdict


def load_run(path):
    with open(path) as f:
        results = json.load(f)
    by_comp = defaultdict(list)
    for r in results:
        by_comp[r["competency"]].append(r)
    return by_comp


def summarize(by_comp):
    out = {}
    for comp, results in by_comp.items():
        pass_rate = sum(r["passed"] for r in results) / len(results)
        avg_score = sum(r["score"] for r in results) / len(results)
        out[comp] = {"n": len(results), "pass_rate": pass_rate, "avg_score": avg_score}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                     help="One run to include, e.g. --run gpt-oss-120b=results/summary.json "
                          "-- repeat for each model/config to compare")
    ap.add_argument("--out", help="Write the Markdown table here (default: print to stdout)")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = summarize(load_run(path))

    all_comps = sorted({c for s in runs.values() for c in s})
    labels = list(runs.keys())

    lines = []
    header = "| Competency | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join("---" for _ in labels) + "|"
    lines.append(header)
    lines.append(sep)
    for comp in all_comps:
        cells = []
        for label in labels:
            s = runs[label].get(comp)
            cells.append(f"{s['pass_rate']:.0%} / {s['avg_score']:.2f} (n={s['n']})" if s else "—")
        lines.append(f"| {comp} | " + " | ".join(cells) + " |")

    # overall row -- micro-average across all competencies' individual results is more
    # informative than a macro-average of pass rates, but macro is simpler and matches how
    # the per-competency rows already read; both reported so the reader isn't misled by n
    # differences across competencies (e.g. hallucination and causal-language each contribute
    # one pass/fail per network, while calibration/sign contribute one per EDGE).
    overall_cells = []
    for label in labels:
        rates = [runs[label][c]["pass_rate"] for c in all_comps if c in runs[label]]
        scores = [runs[label][c]["avg_score"] for c in all_comps if c in runs[label]]
        overall_cells.append(f"{sum(rates)/len(rates):.0%} / {sum(scores)/len(scores):.2f}" if rates else "—")
    lines.append(f"| **Macro-average across competencies** | " + " | ".join(overall_cells) + " |")

    table = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"Wrote {args.out}")
    else:
        print(table)


if __name__ == "__main__":
    main()
