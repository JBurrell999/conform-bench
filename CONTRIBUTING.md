# Contributing

## Ways to help

1. **Run the suite against an engine/version/model combo that isn't in the
   matrix yet** (or re-run one that is, on a newer version) and open a PR
   with the updated `results/<engine>-latest.json`.
2. **Add a test case.** More schema features and tool-calling shapes means
   more real-world bugs caught.
3. **Add an engine adapter** for something not yet supported (TGI, LM
   Studio, a cloud provider's self-hosted option, etc).
4. **Fix a bug** in the harness itself -- see `tests/` for how the pieces
   fit together; `tests/test_runner_integration.py` in particular is the
   one to run after touching anything under `testcases/`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

All tests run offline against either `httpx.MockTransport` or the built-in
`llm_conform.mock_server` -- no GPU, API key, or real engine required to
contribute.

## Adding a test case

Drop a new YAML file in `testcases/structured_output/` or
`testcases/tool_calling/`. Run `pytest tests/test_runner_integration.py -k
<your_test_id>` immediately after -- it synthesizes the response a
perfectly-conformant engine *should* give (from your own schema/checks) and
asserts it scores PASS. If that fails, your test case has a bug (not a real
engine), usually one of:

- A `checks` path that doesn't exist in your own schema (typo, or wrong
  nesting depth).
- A schema that's self-contradictory or unsatisfiable.
- A recursive schema (`$ref`/`$defs`) with no terminating hint -- add a
  `checks` entry for the path where recursion should stop, with the empty
  value (`[]` for an array, `{}` for an object). See
  `testcases/structured_output/recursive_tree_schema.yaml` for an example.

### Structured output test fields

```yaml
id: unique_snake_case_id
description: One line, shown in the matrix and `list-tests`.
difficulty: basic | medium | advanced
schema: { ... }            # a JSON Schema
system_prompt: optional string
prompt: |
  The user prompt. Be unambiguous about what value each field should hold --
  these tests check exact values, not just "valid JSON."
checks:                    # optional; dotted-path == exact value assertions
  - path: user.name
    equals: Priya
  - path: books.0.year
    equals: 1965
```

### Tool-calling test fields

```yaml
id: unique_snake_case_id
description: One line.
difficulty: basic | medium | advanced
tools:
  - name: get_weather
    description: ...
    parameters: { ... }    # a JSON Schema for the arguments object
tool_choice: auto | required | { name: tool_name }
messages:
  - role: user
    content: "..."
expect:
  calls_tool: true                 # false for "shouldn't call anything" tests
  tool_name: get_weather            # or:
  tool_name_any_of: [a, b]          # "any one of these is acceptable"
  min_calls: 1
  max_calls: 1                      # omit (null) for "no upper bound"
  args_checks:
    - path: city
      equals: Rome
```

A gap that's *expected* (e.g. a native API with no `tool_choice` equivalent)
is still a legitimate FAIL in the matrix -- don't special-case it in the
test. The point is to report reality, including gaps that are "working as
designed."

## Adding an engine adapter

Most engines just speak OpenAI's `/v1/chat/completions` -- subclass
`llm_conform.engines.openai_compat.OpenAICompatAdapter` and override
`get_version()` and, if needed, `_structured_payload_extra`/
`_tools_payload_extra` for quirks. If the engine has its own native wire
format (like Ollama), implement `llm_conform.engines.base.EngineAdapter`
directly -- see `llm_conform/engines/ollama.py`.

Register it in `llm_conform/engines/__init__.py`'s `ADAPTERS` dict, add a
sample entry to `engines.example.yaml`, and add unit tests mirroring
`tests/test_engines.py` (payload shape via `httpx.MockTransport` -- no real
server needed).

## Submitting real results

```bash
cp engines.example.yaml engines.yaml   # edit base_url/model for your setup
llm-conform run --config engines.yaml --latest
llm-conform report results/*-latest.json --update-readme README.md
```

Commit the updated `results/<engine>-latest.json` and `README.md` together.
Please include in your PR description: the engine's exact version/commit,
the model used, and any non-default server flags (e.g.
`--enable-auto-tool-choice --tool-call-parser hermes` for vLLM) -- those
flags are often the difference between a real gap and a misconfiguration,
and matrix footnotes exist to capture that.
