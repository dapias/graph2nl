# graph2nl

Validated LLM interpretation of statistical association networks
(partial-correlation matrices) as natural language.

`graph2nl` takes an already-estimated cross-sectional network -- nodes,
edges, weights, community/centrality metadata -- and produces a
scientifically appropriate natural-language interpretation of it via an
LLM, following a reporting-constrained protocol grounded in psychological-
network methodology. It is dataset-agnostic: how the network was built
(which survey, which correlation method, which nodes) is the concern of an
upstream pipeline, not this package.

This package also includes the validation suite (hand-built diagnostic
networks + a deterministic scorer) used to evaluate the interpretation
layer's reliability across six competencies: edge-sign fidelity,
unsupported-association avoidance, magnitude-label consistency,
community-membership grounding, centrality interpretation, and
causal-language avoidance.

## Install

```bash
git clone https://github.com/dapias/graph2nl.git
cd graph2nl
pip install -e .
```

Requires Python >= 3.9.

## Quickstart

1. **Set your API key.** Any OpenAI-compatible chat completions endpoint
   works. Either export the variable your config names, or drop it in a
   `.env` file in your working directory:

   ```bash
   export GWDG_API_KEY=...   # or OPENAI_API_KEY, or whatever your llm_config uses
   ```

2. **Copy the example LLM config** and edit `base_url` / `model` if you're
   not using GWDG's Academic Cloud AI service:

   ```bash
   cp examples/llm_config.example.yaml my_llm_config.yaml
   ```

3. **Run the interpreter** on the bundled example network (a small
   hand-built network with two clean communities and one weak bridge --
   see `examples/network_for_llm.example.json`):

   ```bash
   graph2nl-interpret --network examples/network_for_llm.example.json \
                       --llm-config my_llm_config.yaml
   ```

   This writes `examples/network_for_llm.example_interpretation.md` (the
   model's response) and a matching `.audit.json` (exact system/user
   prompt, model, timestamp -- so any reported interpretation is
   traceable to the exact call that produced it).

4. **Point it at your own network** once you have one, by supplying a
   JSON file in the same schema (nodes with `id`/`description`/
   `community`/`strength_centrality`/`expected_influence`; edges with
   `source`/`target`/`weight`/`sign`; see
   `examples/network_for_llm.example.json` or
   `graph2nl/validation/synthetic_networks.py` for further worked
   examples).

## Prompt templates

Three bundled prompt templates live in `graph2nl/prompts/`, in order of
increasing methodological scaffolding:

- `naive.md` -- minimal context, general-audience interpretation, no
  grounding or methodological constraints.
- `scientific_minimal.md` -- basic source-grounding instruction and a
  cross-sectional causal-language caution, for a research audience.
- `full_protocol.md` -- the full reporting-constrained protocol used for
  this project's confirmatory results: magnitude-label calibration,
  sparsity/absent-edge caveats, network-relative centrality interpretation,
  sign/reverse-coding handling, and a two-section (plain-language +
  technical) output format.

Select one via `prompt_template` in your `llm_config.yaml`, or point it at
the path to your own custom template (must contain `## System prompt` and
`## User prompt` headings, each on its own line).

## Validation suite

```bash
graph2nl-validate --llm-config my_llm_config.yaml
```

Runs the six hand-built diagnostic networks (`graph2nl/validation/
synthetic_networks.py`) against your configured model and scores each
response with the deterministic evaluator (`graph2nl/validation/
scorer.py`). Useful flags:

```bash
graph2nl-validate --llm-config my_llm_config.yaml --repeats 21   # check run-to-run consistency
graph2nl-validate --llm-config my_llm_config.yaml --test-id calibration
graph2nl-validate --llm-config my_llm_config.yaml --source procedural  # larger procedurally generated battery
```

## Package layout

```
graph2nl/
  interpret.py            # graph2nl-interpret: single-network interpretation
  prompts/                 # bundled prompt templates (see above)
  validation/
    synthetic_networks.py  # 6 hand-built diagnostic networks, one per competency
    procedural_networks.py # procedurally generated battery (structural variation)
    scorer.py               # deterministic, rule-based evaluator
    run_validation.py       # graph2nl-validate: runs + scores the suite
examples/
  llm_config.example.yaml
  network_for_llm.example.json
```

## Citation

If you use this package, please cite:

> Tapias, D. (2026). From Networks to Narratives: Evaluating LLM-Based
> Interpretation of Psychological Networks.

## License

MIT
