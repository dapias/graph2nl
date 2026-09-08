#!/usr/bin/env python3
"""
run_validation.py -- graph2nl: run the synthetic ground-truth test
suite against a real LLM endpoint and score the results.

For each network in synthetic_networks.SYNTHETIC_NETWORKS: builds the same
kind of prompt the real interpret.py pipeline would, sends it to the configured LLM,
scores the response with scorer.py, and writes both the raw output and a
summary table.

Supports repeating each test N times (--repeats) to check run-to-run
consistency, since LLM output is not deterministic even at low temperature.

Requires the same setup as graph2nl.interpret: `pip install
graph2nl` (which pulls in openai/pyyaml), and GWDG_API_KEY (or your
configured api_key_env) set, e.g. via a .env in your working directory.

Usage:
    graph2nl-validate --llm-config examples/llm_config.example.yaml
    graph2nl-validate --llm-config my_llm_config.yaml --repeats 3
    graph2nl-validate --llm-config my_llm_config.yaml --test-id calibration

    # equivalently, without installing:
    python3 -m graph2nl.validation.run_validation --llm-config ...
"""
import argparse
import datetime
import json
import pathlib
import time

import yaml
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from graph2nl.interpret import load_prompt_template, call_llm, resolve_prompt_template, LLMEmptyResponseError
from graph2nl.validation.synthetic_networks import SYNTHETIC_NETWORKS, get_network
from graph2nl.validation.procedural_networks import generate_procedural_networks
from graph2nl.validation.scorer import score

# call_llm() already retries these with its own exponential backoff (see
# interpret.py); they're caught again here in case that internal backoff
# still isn't enough (e.g. a sustained rate-limit window) -- same
# retry-once-then-record-and-continue policy as LLMEmptyResponseError, so a
# persistent 429 degrades one repeat to an "error": true result instead of
# crashing the whole --repeats 21 / multi-model batch.
_LLM_RETRYABLE_ERRORS = (LLMEmptyResponseError, RateLimitError, APIConnectionError,
                          APITimeoutError, InternalServerError)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-config", required=True, help="Path to a small YAML with LLM settings")
    ap.add_argument("--test-id", help="Run only this synthetic/procedural test id (default: all)")
    ap.add_argument("--repeats", type=int, default=1, help="Repeat each test N times (default 1)")
    ap.add_argument("--out-dir", default="validation_results", help="Where to write results")
    ap.add_argument("--source", choices=["synthetic", "procedural", "both"], default="synthetic",
                     help="synthetic = the 6 hand-built unit-test networks (default, matches the "
                          "original validation run); procedural = the larger procedurally "
                          "generated battery (see procedural_networks.py); both = run all of them. "
                          "Kept as an explicit opt-in rather than the new default so re-running "
                          "this script without --source reproduces the original n=6-per-repeat "
                          "results exactly.")
    ap.add_argument("--n-per-competency", type=int, default=10,
                     help="Only used when --source includes procedural (default 10 -> 60 networks total)")
    ap.add_argument("--proc-seed", type=int, default=2026,
                     help="Only used when --source includes procedural -- see procedural_networks.py")
    ap.add_argument("--call-delay", type=float, default=2.0,
                     help="Seconds to sleep between successive LLM calls (default 2.0), to stay "
                          "under provider per-minute rate limits in the first place rather than "
                          "relying only on retry-after-429. Set to 0 to disable.")
    args = ap.parse_args()

    with open(args.llm_config) as f:
        llm_cfg = yaml.safe_load(f)

    prompt_path = resolve_prompt_template(llm_cfg["prompt_template"])
    system_prompt, user_template = load_prompt_template(prompt_path)

    if args.test_id:
        all_nets = list(SYNTHETIC_NETWORKS) + generate_procedural_networks(args.n_per_competency, args.proc_seed)
        tests = [next(t for t in all_nets if t["id"] == args.test_id)]
    else:
        tests = []
        if args.source in ("synthetic", "both"):
            tests += list(SYNTHETIC_NETWORKS)
        if args.source in ("procedural", "both"):
            tests += generate_procedural_networks(args.n_per_competency, args.proc_seed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"

    # Resume support: if this --out-dir already has a summary.json (e.g.
    # from a run that crashed on a persistent rate-limit error before this
    # fix), pick up where it left off instead of silently overwriting
    # everything from scratch. Only non-error entries count as "done" --
    # an "error": true repeat from a prior run is deliberately re-attempted
    # rather than kept, since it represents a call that never produced a
    # scoreable result.
    all_results = []
    completed = set()
    if summary_path.exists():
        with open(summary_path) as f:
            try:
                previous = json.load(f)
            except json.JSONDecodeError:
                previous = []
        for r in previous:
            if not r.get("error"):
                all_results.append(r)
                completed.add((r["test_id"], r["repeat"]))
        if all_results:
            print(f"Resuming {summary_path}: {len(all_results)} already-completed repeat(s) "
                  f"found and will be skipped; any prior error entries will be retried.")

    first_call = True
    for test_case in tests:
        network_json = json.dumps(test_case["network"], indent=2)
        user_prompt = user_template.replace("{{network_json}}", network_json)

        for rep in range(args.repeats):
            if (test_case["id"], rep) in completed:
                continue

            print(f"Running {test_case['id']} (competency={test_case['competency']}) "
                  f"rep {rep + 1}/{args.repeats} with model={llm_cfg['model']} ...")

            if args.call_delay and not first_call:
                time.sleep(args.call_delay)
            first_call = False

            # A single flaky repeat (e.g. a reasoning model returning an
            # empty completion, or a rate-limit/connection error that
            # outlasts call_llm's own internal backoff -- see
            # _LLM_RETRYABLE_ERRORS above) should not abort an entire
            # --repeats 21 / multi-competency batch and lose every
            # already-completed result. Retry once more here (these
            # failure modes are typically transient), then record it as an
            # explicit error result and move on rather than crashing --
            # and since results are resumable now (see above), a
            # persistent outage can just be re-run later against the same
            # --out-dir to pick up only what's still missing.
            llm_text = None
            error_message = None
            for attempt in range(2):
                try:
                    llm_text = call_llm(llm_cfg, system_prompt, user_prompt)
                    error_message = None
                    break
                except _LLM_RETRYABLE_ERRORS as e:
                    error_message = str(e)
                    print(f"  -> WARNING: {type(e).__name__}: {error_message}"
                          f"{' (retrying once)' if attempt == 0 else ' (giving up on this repeat)'}")

            if llm_text is None:
                result = {
                    "test_id": test_case["id"],
                    "competency": test_case["competency"],
                    "passed": False,
                    "score": 0.0,
                    "details": [f"LLM call error, not a scored content failure: {error_message}"],
                    "error": True,
                }
            else:
                result = score(test_case, llm_text)
            result["repeat"] = rep
            result["model"] = llm_cfg["model"]
            result["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
            # Carry the generation parameters through into summary.json --
            # test_case["params"] (e.g. block_size/n_blocks for
            # community_grounding, n_edges/bands for calibration) only
            # exists on procedural test cases (see procedural_networks.py's
            # generator functions); hand-built SYNTHETIC_NETWORKS entries
            # have neither "params" nor "source", hence the defaults below.
            # Without this, checking whether pass rate depends on structural
            # complexity (e.g. does community_grounding degrade as block
            # size grows) requires re-deriving each network's parameters
            # from its test_id after the fact instead of reading them
            # straight out of the results file.
            result["source"] = test_case.get("source", "synthetic")
            result["params"] = test_case.get("params", {})
            all_results.append(result)

            case_dir = out_dir / test_case["id"]
            case_dir.mkdir(exist_ok=True)
            suffix = f"_rep{rep}" if args.repeats > 1 else ""
            with open(case_dir / f"output{suffix}.md", "w") as f:
                f.write(llm_text if llm_text is not None else f"[LLM call error: {error_message}]")
            with open(case_dir / f"score{suffix}.json", "w") as f:
                json.dump(result, f, indent=2)

            # Write the running summary after every repeat (not just at the
            # end) so a later crash -- this repeat's error handling covers
            # the known failure mode, but not every possible one -- doesn't
            # discard already-completed, API-metered results.
            with open(summary_path, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"  -> passed={result['passed']} score={result['score']:.2f}"
                  f"{'  [ERROR]' if result.get('error') else ''}")

    print(f"\n{'='*70}\nSummary ({len(all_results)} runs across {len(tests)} test(s))\n{'='*70}")
    by_test = {}
    for r in all_results:
        by_test.setdefault(r["test_id"], []).append(r)
    for test_id, results in by_test.items():
        scores = [r["score"] for r in results]
        pass_rate = sum(r["passed"] for r in results) / len(results)
        avg_score = sum(scores) / len(scores)
        n_errors = sum(1 for r in results if r.get("error"))
        error_note = f"  ({n_errors} LLM call error(s), scored as fail)" if n_errors else ""
        print(f"  {test_id:22s} pass_rate={pass_rate:.0%}  avg_score={avg_score:.2f}  "
              f"(n={len(results)} run(s)){error_note}")

    total_errors = sum(1 for r in all_results if r.get("error"))
    if total_errors:
        print(f"\nNOTE: {total_errors} repeat(s) had an LLM call error (empty response, rate "
              f"limit, or connection/timeout error that outlasted retries) rather than a scored "
              f"content failure -- these are counted as failing the competency above, but are "
              f"distinguishable in summary.json via \"error\": true. Re-run this exact command "
              f"against the same --out-dir to retry just those (everything else is skipped as "
              f"already-completed) before reporting pass rates.")

    print(f"\nWrote {summary_path} and per-test output/scores under {out_dir}/")


if __name__ == "__main__":
    main()
