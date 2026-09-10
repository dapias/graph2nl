#!/usr/bin/env python3
"""
interpret.py -- graph2nl-core: LLM interpretation of a partial-correlation
network, via an OpenAI-compatible endpoint.

Consumes a network description in the schema documented in
schema/network_for_llm.md (nodes/edges/meta -- see that file, or
graph2nl/validation/synthetic_networks.py for hand-built examples) and
an LLM config (small YAML: base_url, api_key_env, model, prompt_template,
etc. -- see examples/llm_config.example.yaml), and writes a natural-language
interpretation.

This is intentionally dataset-agnostic: the network can come from any
correlation method, any domain, any upstream pipeline -- as long as it
matches the schema. Tested against GWDG's Academic Cloud AI service
(https://chat-ai.academiccloud.de/v1), and works with any other
OpenAI-compatible provider by changing llm_config.base_url / llm_config.model.

Requires:
    pip install graph2nl        # installs the `graph2nl-interpret` CLI
    export GWDG_API_KEY=...          # or whatever env var llm_config.api_key_env names

Usage:
    graph2nl-interpret --network network_for_llm.json --llm-config llm_config.yaml
    graph2nl-interpret --network network_for_llm.json --llm-config llm_config.yaml --out interpretation.md

    # equivalently, without installing:
    python3 -m graph2nl.interpret --network ... --llm-config ...
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import sys
import time

import yaml

try:
    from dotenv import load_dotenv
    # No explicit path: python-dotenv's default search walks up from the
    # current working directory, which is the right behavior here since
    # this package is typically invoked from an upstream pipeline's own
    # directory (e.g. graph2nl-ewcs), not from inside this package.
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed: fall back to whatever's already in the environment

BUNDLED_PROMPTS_DIR = pathlib.Path(__file__).resolve().parent / "prompts"


def resolve_prompt_template(spec):
    """Resolve a prompt_template value from an llm_config to an actual file
    path. Accepts either a path (absolute, or relative to the current
    working directory) to a custom prompt template, or the bare name of one
    of this package's bundled templates (e.g. "full_protocol", with or
    without the .md extension) under graph2nl_core/prompts/."""
    p = pathlib.Path(spec)
    if p.exists():
        return p
    for candidate in (BUNDLED_PROMPTS_DIR / spec, BUNDLED_PROMPTS_DIR / f"{spec}.md"):
        if candidate.exists():
            return candidate
    sys.exit(f"Prompt template '{spec}' not found as a path, nor bundled in {BUNDLED_PROMPTS_DIR}/ "
              f"(bundled options: {[p.stem for p in BUNDLED_PROMPTS_DIR.glob('*.md')]})")


def load_prompt_template(path):
    """Split a prompt template file into (system_prompt, user_prompt).

    Matches '## System prompt' / '## User prompt' only when they appear as a
    heading on their own line (start of line, optional trailing whitespace),
    so mentioning those words elsewhere in the file (e.g. in a comment
    explaining the format) can't be mistaken for the real section marker.
    """
    with open(path) as f:
        text = f.read()

    sys_match = re.search(r"^##\s*System prompt\s*$", text, re.MULTILINE)
    user_match = re.search(r"^##\s*User prompt\s*$", text, re.MULTILINE)
    if not sys_match or not user_match:
        sys.exit(f"Prompt template {path} must contain '## System prompt' and "
                  f"'## User prompt' headings, each on their own line.")
    if user_match.start() <= sys_match.end():
        sys.exit(f"Prompt template {path}: '## User prompt' must come after '## System prompt'.")

    system_prompt = text[sys_match.end():user_match.start()].strip()
    user_prompt = text[user_match.end():].strip()
    return system_prompt, user_prompt


class LLMEmptyResponseError(RuntimeError):
    """Raised when the configured endpoint returns a response that isn't
    usable as a complete interpretation. Covers three distinct observed
    failure modes, all surfaced through this one exception so existing
    callers (interpret.py's main(), run_validation.py's retry-and-continue
    wrapper) don't need separate except clauses for each:

    1. message.content is None -- some OpenAI-compatible providers, in
       particular reasoning models, return this if the model only emitted
       a reasoning/thinking segment and no final answer before hitting
       max_tokens, or if a provider-side safety filter suppressed the
       reply.
    2. finish_reason == 'length' -- the response is non-empty but was cut
       off mid-generation. For a reasoning model, max_tokens covers the
       hidden reasoning trace as well as the visible answer, so this can
       happen even when the visible text alone is far short of the
       configured max_tokens; raising max_tokens is the usual fix.
    3. finish_reason == 'stop' but the response is missing a required
       section heading -- observed with full_protocol.md's two-section
       format: the model produces a complete, well-formed Plain-language
       summary and then stops on its own (a genuine stop token, not a
       length cutoff) without ever starting the Technical interpretation
       section. This looks like call-to-call stochastic behavior rather
       than a systematic prompt problem, since most calls with the same
       config succeed -- so it's handled the same way as a transient
       provider error: retry with backoff.

    Without this check, a case-1 response fails loudly elsewhere (e.g.
    validation/scorer.py's _normalize, which expects a string and would
    otherwise raise an opaque 'NoneType has no attribute replace'), but
    cases 2 and 3 previously failed silently: the partial or incomplete
    text was written to disk as if it were a normal, complete response."""


def call_llm(llm_cfg, system_prompt, user_prompt, max_retries=5, base_delay=15,
             required_headings=None):
    """Call the configured OpenAI-compatible endpoint and return the raw
    text response. Shared by this module's normal run and by
    validation/run_validation.py's synthetic-network test harness, so both
    paths exercise the exact same request-construction code

    Retries with exponential backoff (base_delay, doubling, capped at 300s)
    on BOTH transient provider errors (rate limiting, connection drops,
    timeouts, 5xx responses) AND unusable-but-not-erroring responses (see
    LLMEmptyResponseError) -- both are typically resolved by trying again
    rather than indicating a bad request, so they share one retry loop.
    GWDG's Academic Cloud AI service in particular rate-limits (429 'API
    rate limit exceeded') under the kind of tight-loop repeated calling
    run_validation.py's --repeats does, so this retry logic lives here, at
    the single shared call site, rather than in each caller -- callers only
    need to catch LLMEmptyResponseError for the case where every retry
    was exhausted.

    required_headings: optional tuple of heading strings (e.g.
    ("## Plain-language summary", "## Technical interpretation")) that
    must each appear on their own line in the response. Pass this when the
    prompt template's system prompt mandates specific output sections --
    full_protocol.md does, naive.md/scientific_minimal.md may not, so this
    is opt-in per call rather than hardcoded here. Leave as None to skip
    this check.

    Raises LLMEmptyResponseError if no attempt produces a usable response
    within max_retries. Re-raises the underlying openai exception if a
    transient error specifically (not an unusable-response case) persists
    past max_retries."""
    try:
        from openai import (
            OpenAI,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:
        sys.exit("Missing dependency: pip install openai")

    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        sys.exit(f"Set {api_key_env} in your environment before running this script.")

    client = OpenAI(api_key=api_key, base_url=llm_cfg["base_url"])

    transient_errors = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
    delay = base_delay
    last_unusable_reason = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=llm_cfg["model"],
                max_tokens=llm_cfg.get("max_tokens", 4000),
                temperature=llm_cfg.get("temperature", 0.3),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except transient_errors as e:
            if attempt == max_retries:
                raise
            print(f"  -> {type(e).__name__}: {e} "
                  f"(attempt {attempt + 1}/{max_retries + 1}, retrying in {delay}s)")
            time.sleep(delay)
            delay = min(delay * 2, 300)
            continue

        content = response.choices[0].message.content
        finish_reason = getattr(response.choices[0], "finish_reason", None)

        if content is None:
            last_unusable_reason = (
                f"no content returned (finish_reason={finish_reason!r}). Typically a reasoning "
                "model that only emitted a reasoning/thinking segment before hitting max_tokens, "
                "or a safety filter suppressing the reply."
            )
        elif finish_reason == "length":
            last_unusable_reason = (
                f"response truncated by max_tokens (finish_reason='length'), "
                f"{len(content)} chars received before cutoff. For a reasoning model, max_tokens "
                "covers the hidden reasoning trace as well as the visible answer, so this can "
                "happen even when the visible text looks short -- raise max_tokens in this "
                "model's llm_config."
            )
        elif required_headings and (
            missing := [h for h in required_headings
                        if not re.search(rf"^{re.escape(h)}\s*$", content, re.MULTILINE)]
        ):
            last_unusable_reason = (
                f"response is missing required section heading(s) {missing} "
                f"(finish_reason={finish_reason!r}, {len(content)} chars, "
                f"{len(content.split())} words). The model stopped cleanly rather than being cut "
                "off mid-sentence -- this looks like call-to-call variability rather than a "
                "systematic prompt problem, so retrying usually succeeds."
            )
        else:
            return content  # usable response

        if attempt == max_retries:
            break

        print(f"  -> Unusable response: {last_unusable_reason} "
              f"(attempt {attempt + 1}/{max_retries + 1}, retrying in {delay}s)")
        time.sleep(delay)
        delay = min(delay * 2, 300)

    raise LLMEmptyResponseError(
        f"Model '{llm_cfg['model']}' did not return a usable response after "
        f"{max_retries + 1} attempts. Last failure: {last_unusable_reason} "
        "If this recurs consistently for one model/config rather than intermittently, try "
        "raising max_tokens, tightening the prompt's section instructions, or checking the "
        "provider's status -- intermittent failures across otherwise-successful calls are "
        "expected to be resolved by the retries above."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True,
                     help="Path to a network_for_llm.json-shaped file (nodes/edges/meta) -- "
                          "see schema/network_for_llm.md")
    ap.add_argument("--llm-config", required=True,
                     help="Path to a small YAML with base_url/api_key_env/model/prompt_template/etc. "
                          "-- see examples/llm_config.example.yaml")
    ap.add_argument("--out", help="Output markdown path (default: <network>_interpretation.md "
                                   "next to --network)")
    args = ap.parse_args()

    with open(args.llm_config) as f:
        llm_cfg = yaml.safe_load(f)

    network_path = pathlib.Path(args.network)
    with open(network_path) as f:
        network_json = f.read()

    prompt_path = resolve_prompt_template(llm_cfg["prompt_template"])
    system_prompt, user_template = load_prompt_template(prompt_path)
    user_prompt = user_template.replace("{{network_json}}", network_json)

    # full_protocol.md's system prompt mandates these two exact section
    # headings; other bundled templates (naive.md, scientific_minimal.md)
    # may not impose this structure, so only require it when the prompt
    # actually declares it. Auto-detected from the system prompt itself
    # (it states each required heading in backticks, e.g. `## Technical
    # interpretation`) so this doesn't need updating by hand if a template
    # adds/renames a section -- new templates just need to follow the same
    # backtick-heading convention to opt in automatically.
    required_headings = tuple(re.findall(r"`(##\s+[^`]+)`", system_prompt)) or None

    try:
        interpretation = call_llm(llm_cfg, system_prompt, user_prompt,
                                   required_headings=required_headings)
    except LLMEmptyResponseError as e:
        sys.exit(str(e))

    out_path = pathlib.Path(args.out) if args.out else \
        network_path.with_name(network_path.stem + "_interpretation.md")
    out_path.write_text(interpretation, encoding="utf-8")

    audit = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "base_url": llm_cfg["base_url"],
        "model": llm_cfg["model"],
        "temperature": llm_cfg.get("temperature", 0.3),
        "max_tokens": llm_cfg.get("max_tokens", 4000),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "input_file": str(network_path),
        "output_file": str(out_path),
    }
    with open(str(out_path) + ".audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"Wrote {out_path}.audit.json")

    

if __name__ == "__main__":
    main()
