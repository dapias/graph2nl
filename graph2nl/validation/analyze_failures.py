#!/usr/bin/env python3
"""
analyze_failures.py -- graph2nl-core: aggregate failure/near-miss patterns
across a big batch of already-scored validation runs (e.g. after a
`--repeats 20` run), without having to hand-paste every score*.json into a
chat.

Doesn't call the LLM or re-score anything -- just reads existing score*.json
files under a results directory's <test_id>/ subfolders and buckets each
non-perfect result by a coarse, test-specific failure category, so you can
see at a glance which failure MODE is common (worth investigating) vs. rare
(probably noise) before hand-reading individual cases.

Usage:
    python3 -m graph2nl_core.validation.analyze_failures                      # all tests
    python3 -m graph2nl_core.validation.analyze_failures --test-id calibration
    python3 -m graph2nl_core.validation.analyze_failures --test-id hallucination --show 5
"""
import argparse
import glob
import json
import pathlib
import re
from collections import Counter, defaultdict


def categorize_calibration(detail):
    if "NOT MENTIONED" in detail:
        return "not mentioned together at all"
    if "AMBIGUOUS" in detail:
        return "AMBIGUOUS (correct band + at least one other band both matched)"
    if "WRONG" in detail:
        return "WRONG (correct band term absent / different band matched)"
    return None  # "correct" -- not a failure, skip


def categorize_hallucination(detail):
    # "correctly reported as unrelated" covers every clean path score_hallucination
    # has (explicit denial language, no-relational-language overview sentences, and
    # the community-label denial check) -- matching on this shared suffix instead of
    # each path's own wording separately avoids this script itself going stale every
    # time scorer.py gains a new way to mark something clean.
    if ("not mentioned together" in detail or "no relational language" in detail
            or "correctly reported as unrelated" in detail):
        return None  # clean -- not a failure, skip
    if "CO-OCCUR" in detail:
        # heuristic: does the flagged window text look like a community/group-label
        # denial that scorer.py's community-label check doesn't recognize -- e.g.
        # unfamiliar wording, or a group reference it doesn't catch -- as opposed to
        # ordinary pair-specific contamination?
        if re.search(r'\bcommunity\b|\bgroup\b|\bcluster\b|\bblock\b', detail, re.IGNORECASE):
            return "flagged window mentions community/group/cluster/block label NOT caught by the community-label check -- needs a fresh look"
        return "flagged window is pair-specific text, no group-label wording (needs a fresh look)"
    return "other"


def categorize_generic(result):
    return f"score={result['score']:.2f}"


CATEGORIZERS = {
    "calibration": ("details", categorize_calibration),
    "hallucination": ("details", categorize_hallucination),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="validation_results")
    ap.add_argument("--test-id", default=None, help="only analyze this test (default: all)")
    ap.add_argument("--show", type=int, default=2, help="how many example detail lines to print per category")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    test_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())
    if args.test_id:
        test_dirs = [p for p in test_dirs if p.name == args.test_id]

    for test_dir in test_dirs:
        test_id = test_dir.name
        score_files = sorted(test_dir.glob("score*.json"))
        if not score_files:
            continue

        results = []
        for f in score_files:
            with open(f) as fh:
                results.append((f.name, json.load(fh)))

        n = len(results)
        n_perfect = sum(1 for _, r in results if r["score"] == 1.0)
        print(f"\n{'='*70}\n{test_id}  ({n} runs, {n_perfect} perfect, {n - n_perfect} non-perfect)\n{'='*70}")

        if test_id not in CATEGORIZERS:
            # generic: just bucket by score value
            counts = Counter(categorize_generic(r) for _, r in results if r["score"] < 1.0)
            for cat, cnt in counts.most_common():
                print(f"  {cnt:3d}x  {cat}")
            continue

        field, categorize = CATEGORIZERS[test_id]
        bucket_examples = defaultdict(list)
        bucket_counts = Counter()
        for fname, r in results:
            for detail in r.get(field, []):
                cat = categorize(detail)
                if cat is None:
                    continue
                bucket_counts[cat] += 1
                if len(bucket_examples[cat]) < args.show:
                    bucket_examples[cat].append((fname, detail))

        if not bucket_counts:
            print("  (no non-clean detail lines found)")
            continue

        for cat, cnt in bucket_counts.most_common():
            print(f"\n  {cnt:3d}x  {cat}")
            for fname, detail in bucket_examples[cat]:
                short = detail if len(detail) <= 220 else detail[:220] + "..."
                print(f"        [{fname}] {short}")


if __name__ == "__main__":
    main()
