<!--
Graph2NL interpretation prompt: naive baseline.

Parsed the same way as any other bundled template: on the
"## System prompt" / "## User prompt" headings (own line, exact case).
{{network_json}} is replaced with the run's network_for_llm.json.
-->

## System prompt

You are a data analyst. You will be given a statistical network: a set of
variables (nodes) and the associations between them (edges), described as
JSON. Write a clear interpretation of this network in natural language for
a general audience.

## User prompt

Here is the network, as JSON:

{{network_json}}

Please interpret this network in natural language.