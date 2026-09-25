# `network_for_llm.json` schema

This is the input format net2narratives expects. Any upstream pipeline that
produces a JSON file in this shape -- regardless of domain, correlation
method, or estimation software -- can pass it to `net2narratives-interpret`
(or `python3 -m net2narratives.interpret`) to obtain a natural-language
interpretation.

`net2narratives/validation/synthetic_networks.py` contains hand-built examples
in this format (the validation suite runs against them); reading one or
two is the fastest way to see the schema in practice.

## Top-level shape

```json
{
  "meta": { ... },
  "nodes": [ { ... }, ... ],
  "edges": [ { ... }, ... ]
}
```

The entire object is passed to the model as JSON, so any additional
fields are shown to the model verbatim.

### `meta` (object; all fields optional but recommended)

| field | type | meaning |
|---|---|---|
| `dataset` | string | Human-readable name/citation of the source data. |
| `run_name` | string | Identifier for this specific run/configuration. |
| `sample_description` | string | One-line description of the sample/scope. |
| `n` | integer | Sample size used for estimation. Omit for constructed (non-estimated) networks. |
| `method` | string | How the correlations and network were obtained (e.g. `"Polychoric correlations -> EBICglasso regularized partial-correlation network (gamma=0.5)"`). For constructed test networks, state that values are exact by construction; the bundled prompts use this field to decide how to interpret absent edges. |
| `modularity` | number or null | Modularity of the supplied community partition, if any. (The empirical networks in the accompanying paper name this field `community_detection_modularity`; either name is passed through to the model unchanged.) |
| `note` | string | Caveats the model should be told up front (e.g. that the data are cross-sectional). |

### `nodes` (array of objects)

| field | type | meaning |
|---|---|---|
| `id` | string | Unique node identifier, referenced by `edges.source`/`edges.target`. Use distinctive tokens (e.g. `N1`, `burnout_exh`), not single letters or common words, so that identifiers can be matched reliably in the generated text. |
| `description` | string | What this node measures, in plain language. |
| `theme` | string | Optional researcher-assigned grouping label (not the same as `community`). |
| `community` | integer | Supplied community membership. This may come from data-driven community detection (e.g. walktrap, spin-glass) or from a theoretical grouping; say which in `meta.method` or `meta.note`. |
| `strength_centrality` | number | Sum of \|edge weight\| over edges touching this node. |
| `expected_influence` | number | Sum of signed edge weights over edges touching this node. |

### `edges` (array of objects)

| field | type | meaning |
|---|---|---|
| `source` | string | A node id. |
| `target` | string | A node id. |
| `weight` | number | Partial correlation / regularized edge weight. |
| `sign` | string | `"positive"` or `"negative"` (redundant with the sign of `weight`, kept explicit so the model does not have to infer it). |

Include only nonzero edges: edges removed by regularization are absent,
not listed with weight 0. List edges in descending order of \|weight\|;
the bundled prompts tell the model that edges are sorted this way.

## Conventions assumed by the bundled prompts

- **Higher node values = more of the named construct.** If raw items are
  reverse-coded, do this upstream and describe the recoded direction in
  `description`; the prompts take `description` at face value.
- **Edges are partial correlations** (associations conditional on all other
  nodes in the network), not zero-order correlations. If your network is
  of a different kind, say so explicitly in `meta.method`; the bundled
  prompts are written for partial-correlation networks.
- **Cross-sectional, non-causal by default.** The full-protocol prompt
  instructs the model to avoid causal language. You may additionally state
  the design in `meta.note`. (In the validation suite, the
  causal-language test networks deliberately omit an explicit "no causal
  interpretation" instruction, so that the test measures whether the model
  avoids causal language without being told to in its input.)
