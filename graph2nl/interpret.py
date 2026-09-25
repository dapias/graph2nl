#!/usr/bin/env python3
"""Generate a natural-language interpretation of a network through an LLM.

Reads a network in the schema/network_for_llm.md format and a YAML LLM
configuration. The network may come from any upstream pipeline that follows
the schema. Responses are requested from an OpenAI-compatible endpoint.

Usage:
    graph2nl-interpret --network network_for_llm.json --llm-config llm_config.yaml
    graph2nl-interpret --network network_for_llm.json --llm-config llm_config.yaml --out interpretation.md
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
    # Load an available .env file before reading the configured API key.
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed: fall back to whatever's already in the environment

BUNDLED_PROMPTS_DIR = pathlib.Path(__file__).resolve().parent / "prompts"


def resolve_prompt_template(spec):
    """Find a custom path or a bundled prompt name, with or without .md."""
    p = pathlib.Path(spec)
    if p.exists():
        return p
    for candidate in (BUNDLED_PROMPTS_DIR / spec, BUNDLED_PROMPTS_DIR / f"{spec}.md"):
        if candidate.exists():
            return candidate
    sys.exit(f"Prompt template '{spec}' not found as a path, nor bundled in {BUNDLED_PROMPTS_DIR}/ "
              f"(bundled options: {[p.stem for p in BUNDLED_PROMPTS_DIR.glob('*.md')]})")


def load_prompt_template(path):
    """Read the system and user sections marked by standalone headings."""
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
    """Raised after retries for empty, truncated, or incomplete responses."""


def call_llm(llm_cfg, system_prompt, user_prompt, max_retries=5, base_delay=15,
             required_headings=None):
    """Return a usable response from the configured endpoint.

    Retry transient provider errors and unusable responses with exponential
    backoff. If supplied, required_headings must each occupy a complete line.
    Raise LLMEmptyResponseError after unusable responses exhaust retries.
    """
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

    # Require only headings named in the system prompt, whether quoted in
    # backticks or written as literal Markdown headings.
    backtick_style = re.findall(r"`(##\s+[^`]+)`", system_prompt)
    literal_style = re.findall(r"^(##\s+\S.*?)\s*$", system_prompt, re.MULTILINE)
    required_headings = tuple(dict.fromkeys(backtick_style + literal_style)) or None

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
