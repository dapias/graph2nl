"""Synthetic ground-truth validation suite for graph2nl-core's LLM
interpretation layer. See synthetic_networks.py for the six competency
tests, scorer.py for automated scoring, run_validation.py for the harness
that calls a real LLM endpoint, and rescore.py to re-judge already-collected
output against an updated scorer without new API calls."""
