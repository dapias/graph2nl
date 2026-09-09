#!/usr/bin/env python3
"""
rescore.py -- graph2nl-core: re-score already-collected LLM outputs under a
results directory with the current scorer.py, WITHOUT calling the LLM
again. Use this after fixing/tuning the scorer (as opposed to
run_validation.py, which always makes fresh API calls) -- the raw model
output doesn't change, only how it's judged, so there's no reason to spend
API budget re-collecting it.

Rescores both hand-built (synthetic_networks.py) and procedural
(procedural_networks.py) results. A procedural test_id (e.g.
"proc_calibration_003") is not present in synthetic_networks.get_network(),
so procedural results are reconstructed via procedural_networks.py
instead. Since procedural network generation is deterministic given
(n_per_competency, seed) -- see procedural_networks.py -- passing the SAME
values used for the original run_validation.py call reconstructs the
identical ground truth needed to rescore those results; --n-per-competency/
--proc-seed below default to run_validation.py's own defaults (10, 2026),
so pass whatever
non-default values you actually used (e.g. --n-per-competency 15) or the
lookup won't find matching test_ids.

Usage:
    python3 -m graph2nl_core.validation.rescore                       # rescore ./validation_results
    python3 -m graph2nl_core.validation.rescore --results-dir my_dir
    python3 -m graph2nl_core.validation.rescore --results-dir results_gpt_oss_120b_procedural \\
        --n-per-competency 15                                         # for a procedural results dir
"""
import argparse
import json
import pathlib
import sys

from graph2nl_core.validation.synthetic_networks import get_network
from graph2nl_core.validation.procedural_networks import generate_procedural_networks
from graph2nl_core.validation.scorer import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="validation_results")
    ap.add_argument("--n-per-competency", type=int, default=10,
                     help="Must match what --n-per-competency was set to on the original "
                          "run_validation.py call that produced this --results-dir, if it "
                          "contains procedural results (default 10, run_validation.py's own "
                          "default). Ignored if the directory has no procedural test_ids.")
    ap.add_argument("--proc-seed", type=int, default=2026,
                     help="Must match --proc-seed from the original run (default 2026, "
                          "run_validation.py's own default).")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"{results_dir} doesn't exist -- nothing to rescore. Run run_validation.py first.")

    # Build one lookup covering both synthetic and procedural test_ids, so a
    # results-dir containing either (or, via --source both, a mix of both)
    # rescores correctly either way.
    procedural_by_id = {t["id"]: t for t in generate_procedural_networks(args.n_per_competency, args.proc_seed)}

    def get_test_case(test_id):
        try:
            return get_network(test_id)
        except KeyError:
            pass
        if test_id in procedural_by_id:
            return procedural_by_id[test_id]
        raise KeyError(test_id)

    skipped = []
    all_results = []
    for test_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        test_id = test_dir.name
        try:
            test_case = get_test_case(test_id)
        except KeyError:
            skipped.append(test_id)
            continue  # not a recognized synthetic or procedural test_id, skip

        for output_path in sorted(test_dir.glob("output*.md")):
            suffix = output_path.stem[len("output"):]  # "" or "_rep2" etc.
            score_path = test_dir / f"score{suffix}.json"

            old_result = None
            if score_path.exists():
                with open(score_path) as f:
                    old_result = json.load(f)

            llm_text = output_path.read_text()
            new_result = score(test_case, llm_text)
            # carry over run metadata (model, timestamp, repeat) from the old score file if present
            if old_result:
                for k in ("repeat", "model", "timestamp"):
                    if k in old_result:
                        new_result[k] = old_result[k]

            with open(score_path, "w") as f:
                json.dump(new_result, f, indent=2)

            old_passed = old_result["passed"] if old_result else None
            changed = " <-- CHANGED" if old_result and old_passed != new_result["passed"] else ""
            print(f"{test_id}{suffix}: passed={new_result['passed']} score={new_result['score']:.2f}"
                  + (f" (was passed={old_passed})" if old_result else "") + changed)
            all_results.append(new_result)

    if skipped:
        print(f"NOTE: skipped {len(skipped)} subfolder(s) not recognized as a synthetic or "
              f"procedural test_id: {skipped}. If these are procedural results and this list "
              f"isn't empty when you expected it to be, check --n-per-competency/--proc-seed "
              f"match the values the original run_validation.py call used.\n")

    if not all_results:
        sys.exit(f"No output*.md files found under {results_dir}/<test_id>/ -- nothing to rescore."
                  + (f" ({len(skipped)} subfolder(s) were skipped as unrecognized -- see NOTE above.)" if skipped else ""))

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}\nRescored summary ({len(all_results)} outputs)\n{'='*70}")
    by_test = {}
    by_competency = {}
    for r in all_results:
        by_test.setdefault(r["test_id"], []).append(r)
        by_competency.setdefault(r["competency"], []).append(r)
    for test_id, results in by_test.items():
        pass_rate = sum(r["passed"] for r in results) / len(results)
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"  {test_id:22s} pass_rate={pass_rate:.0%}  avg_score={avg_score:.2f}  (n={len(results)})")

    if len(by_competency) != len(by_test):
        # more than one test_id shares a competency (e.g. many procedural
        # networks per competency) -- the per-test_id breakdown above is
        # each individual network; this rolls it up the way compare_runs.py
        # will read it (grouped by competency), which is usually the number
        # actually worth looking at for a procedural results-dir.
        print(f"\n{'-'*70}\nBy competency (matches how compare_runs.py aggregates)\n{'-'*70}")
        for competency, results in by_competency.items():
            pass_rate = sum(r["passed"] for r in results) / len(results)
            avg_score = sum(r["score"] for r in results) / len(results)
            print(f"  {competency:26s} pass_rate={pass_rate:.0%}  avg_score={avg_score:.2f}  (n={len(results)})")

    print(f"\nWrote {summary_path} (overwritten) and per-test score*.json files under {results_dir}/")


if __name__ == "__main__":
    main()
