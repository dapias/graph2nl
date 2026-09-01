"""graph2nl: validated LLM interpretation of statistical association
networks (partial-correlation matrices) as natural language.

Dataset-agnostic by design -- this package only ever consumes a network
description in the schema documented in schema/network_for_llm.md (nodes,
edges, weights, community/centrality metadata) and produces a natural-
language interpretation of it via an LLM. How that network was built
(which survey, which correlation method, which nodes) is entirely the
concern of an upstream, dataset-specific pipeline.

The interpretation layer's validity claim rests on validation/, a suite of
hand-built and procedurally generated networks with machine-checkable
ground truth for six interpretive competencies (magnitude-label
consistency, sign/direction, unsupported-association avoidance, community
narration, centrality nuance, causal-language avoidance)
"""

__version__ = "0.1.0"