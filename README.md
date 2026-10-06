# llm-conform

**Does JSON-schema-constrained output and tool-call parsing actually work on
your engine?** `llm-conform` is a conformance test suite that runs the same
battery of structured-output and tool-calling tests against vLLM, SGLang,
llama.cpp, Ollama (or anything else that speaks a compatible API), and
publishes a compatibility matrix from the results.

These features are implemented by every serving engine, "work" in the demo,
and then break in specific, boring ways: a recursive schema that hangs the
grammar compiler, `additionalProperties: false` that's accepted but not
enforced, a `tool_choice` that's silently ignored because of a missing
server flag, parallel tool calls that only one engine actually supports.
This project exists to turn "I think X broke in version Y" into a
reproducible test and a line in a table.

<!-- LLM-CONFORM:MATRIX:START -->
### Summary

| Engine | Version | Model | Structured Output | Tool Calling | Tested |
|---|---|---|---|---|---|
| **llama.cpp** | b11429-d81235049 | `Qwen2.5-1.5B-Instruct-GGUF` | 9/9 (100%) | 6/8 (75%) | 2026-10-06 |
| **ollama** | 0.35.1 | `qwen2.5:1.5b` | 9/9 (100%) | 6/8 (75%) | 2026-10-06 |

### Structured output (JSON-schema-constrained generation)

| Test | llama.cpp | ollama |
|---|---|---|
| `array_of_objects` (Array of objects, each with its own required fields, minItems enforced) | ✅ | ✅ |
| `enum_field` (String field constrained to a fixed enum of allowed values) | ✅ | ✅ |
| `flat_object_basic` (Flat object, two required fields, additionalProperties false) | ✅ | ✅ |
| `nested_object_two_levels` (Object nested two levels deep, with required fields at each level) | ✅ | ✅ |
| `nullable_field` (Field that is explicitly nullable (type [string, "null"]) rather than omitted) | ✅ | ✅ |
| `numeric_range_constraint` (Integer field constrained by minimum and maximum (a 1-5 rating)) | ✅ | ✅ |
| `pattern_regex_field` (String field constrained by a regex pattern (5-digit zip code)) | ✅ | ✅ |
| `recursive_tree_schema` (Recursive schema via $defs/$ref (a self-referencing category tree node) -- schema-to-grammar compilers (outlines, xgrammar, GBNF translation, etc.) historically vary a lot in whether they support self-referencing $ref at all.) | ✅ | ✅ |
| `strict_additional_properties` (additionalProperties: false, paired with a prompt that invites extra detail -- tests whether the engine's constrained decoding actually enforces "no extra keys" rather than just the model loosely following instructions.) | ✅ | ✅ |

### Tool calling

| Test | llama.cpp | ollama |
|---|---|---|
| `array_argument` (Tool parameter is an array of strings) | ✅ | ❌ |
| `enum_argument` (Tool parameter constrained to an enum; checks the model selects the exact matching value) | ❌ | ✅ |
| `forced_tool_choice` (tool_choice forces a specific named tool even though the prompt alone wouldn't obviously call for it. Native APIs without a tool_choice equivalent (e.g. Ollama) are expected to fail this one -- that's a real, reportable gap, not a harness bug.) | ❌ | ❌ |
| `multi_tool_selection` (Two similarly-named tools available; the model must pick the correct one) | ✅ | ✅ |
| `nested_object_argument` (Tool parameter is itself an object with sub-fields) | ✅ | ✅ |
| `no_tool_needed` (Prompt doesn't need any tool; the model should just answer directly) | ✅ | ✅ |
| `parallel_tool_calls` (Prompt naturally needs the same tool called twice with different arguments in one turn. Parallel/multiple tool calls are a feature that varies a lot across engines and server flags.) | ✅ | ✅ |
| `single_tool_basic` (One tool available, prompt clearly needs it; checks correct name and arguments) | ✅ | ✅ |

**Legend:** ✅ pass &nbsp;·&nbsp; ❌ fail (ran, output didn't conform) &nbsp;·&nbsp; ⚠️ error (request/transport failure) &nbsp;·&nbsp; ➖ unsupported (engine rejected the request, e.g. HTTP 400) &nbsp;·&nbsp; ⏱️ timeout &nbsp;·&nbsp; *(blank)* not run

<details>
<summary>Why each non-passing cell failed</summary>

<details>
<summary>llama.cpp: 2 non-passing test(s)</summary>

- `enum_argument` (tool_calling): **fail** -- expected a tool call but got none; content was: "Sure, I'll set the thermostat to heat mode. Please confirm if you want to keep the heat on all day or just for a specific period."
- `forced_tool_choice` (tool_calling): **fail** -- expected a tool call but got none; content was: "That's great to hear! Is there anything specific you'd like to know or discuss?"

</details>

<details>
<summary>ollama: 2 non-passing test(s)</summary>

- `array_argument` (tool_calling): **fail** -- expected a tool call but got none; content was: ''
- `forced_tool_choice` (tool_calling): **fail** -- expected a tool call but got none; content was: "That's great! What can I help you with today?"

</details>

</details>
<!-- LLM-CONFORM:MATRIX:END -->

*(Run `llm-conform report results/*-latest.json --update-readme README.md`
to regenerate the table above from `results/*-latest.json`. An engine with
no row yet (vLLM, SGLang) hasn't been run in this environment -- see
[Known gaps / what isn't tested here](#known-gaps--what-isnt-tested-here).)*

## Quickstart

```bash
git clone <this repo> && cd llm-conform
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Try it against the built-in fake engine first -- no GPU or model needed.
python -m llm_conform.mock_server --port 8800 --behavior schema_violator &
cat > engines.yaml <<EOF
engines:
  - name: demo
    type: openai_compat
    base_url: http://localhost:8800
    model: mock
EOF
llm-conform run --config engines.yaml
```

You should see a handful of `fail`s -- that behavior deliberately breaks
schema conformance, to show what a real detected regression looks like.

Then point it at a real engine:

```bash
cp engines.example.yaml engines.yaml   # edit base_url / model for your setup
llm-conform run --config engines.yaml --latest
llm-conform report results/*-latest.json --update-readme README.md
```

## What it actually checks

**Structured output** (`testcases/structured_output/`): the engine is asked
for JSON constrained to a schema (OpenAI's `response_format: json_schema`
contract, or Ollama's native `format`), and the response is checked for:
exact JSON validity (no markdown fences, no trailing prose), full
`jsonschema` conformance (types, `enum`, `required`, numeric bounds, regex
`pattern`, nested objects, arrays with `minItems`), `additionalProperties:
false` actually being enforced rather than just requested, recursive
schemas via local `$ref`/`$defs`, and that the *values* match what the
prompt asked for (not just "valid shape with made-up data").

**Tool calling** (`testcases/tool_calling/`): the engine is given one or
more tool definitions and checked for: calling the right tool (not a
similarly-named one), *not* calling a tool when none is needed, arguments
that parse as JSON and conform to that tool's own parameter schema,
honoring (or, if unsupported, honestly failing) a forced `tool_choice`, and
parallel/multiple tool calls in one turn.

Run `llm-conform list-tests` for the full, current catalog with
descriptions.

Every outcome is one of:

| Outcome | Meaning |
|---|---|
| ✅ `pass` | Ran, and conformed. |
| ❌ `fail` | Ran, but didn't conform (wrong JSON, wrong tool, wrong args). |
| ⚠️ `error` | Request/transport failure (HTTP 5xx, network error, timeout). |
| ➖ `unsupported` | Engine rejected the request outright (HTTP 4xx) -- usually "this feature isn't implemented here." |

## Architecture

```
testcases/*.yaml  --loader-->  StructuredTestCase / ToolTestCase
                                        |
                                        v
engines.yaml --config--> EngineConfig --+--> EngineAdapter (vllm/sglang/llamacpp/ollama)
                                        |              |
                                        |     complete_structured() / complete_tools()
                                        |              |
                                        v              v
                                   validators.py  <-- RawResponse (normalized)
                                        |
                                        v
                                   TestResult --> results/*.json --> report.py --> README matrix
```

- **`llm_conform/engines/`** -- one adapter per engine. vLLM, SGLang, and
  llama.cpp all subclass `OpenAICompatAdapter` (they share the
  `/v1/chat/completions` contract); Ollama gets its own adapter for its
  native `/api/chat` wire format. Adding a new OpenAI-compatible engine is
  usually a ~15-line subclass -- see [CONTRIBUTING.md](CONTRIBUTING.md).
- **`llm_conform/validators.py`** -- the only place that decides pass/fail.
  It never trusts "the engine said 200 OK"; it always re-parses and
  re-validates the actual content against the test's schema/checks.
- **`llm_conform/mock_server.py`** -- a dependency-free fake engine (stdlib
  `http.server`) with pluggable misbehavior presets (`schema_violator`,
  `malformed_tool_args`, `http_400`, ...). Used by the test suite and as a
  zero-setup way to try the CLI.
- **`llm_conform/synth.py`** -- generates a schema-conforming value from a
  JSON Schema. Used only to test the harness against itself (see
  `tests/test_runner_integration.py`); it never judges a real engine's
  output -- that's always `jsonschema` via `validators.py`.

## Known gaps / what isn't tested here

- **vLLM and SGLang need a GPU for realistic throughput**, and both
  primarily target CUDA; this repo's own CI/dev results come from Apple
  Silicon (Ollama, native; llama.cpp, native via Metal), so those two rows
  are marked "not run in this environment" until someone with a CUDA box
  contributes results. The adapters are fully implemented and unit-tested
  against the OpenAI wire contract either way -- see
  [CONTRIBUTING.md](CONTRIBUTING.md#submitting-real-results).
- **Tool calling on vLLM/SGLang requires server flags**
  (`--enable-auto-tool-choice --tool-call-parser <name>`) that are easy to
  forget; without them `tool_choice` is silently a no-op and every tool
  test will FAIL, which looks like an engine bug but is a config gap. The
  matrix notes should call this out per run.
- **Ollama's native API has no `tool_choice` equivalent at all** -- the
  adapter deliberately doesn't fake support for it, so `forced_tool_choice`
  failing against Ollama is an accurate result, not a bug.
- This suite tests **conformance, not quality**: whether output validates
  and matches the prompt's facts, not whether the model is *smart*. A model
  too weak to understand the prompt and an engine that mangles valid
  structured output both show up as FAIL; check the per-test `reason` field
  in `results/*.json` to tell them apart.
- Only **local `$ref`** (`#/$defs/...`) is supported by the schemas in this
  suite and by `synth.py`; no remote `$ref` resolution.

## Project layout

```
llm_conform/            the harness (installable package, `llm-conform` CLI)
testcases/              the test catalog (YAML, one file per test case)
tests/                  pytest suite -- runs fully offline
results/                committed *-latest.json per engine (matrix source of truth)
  results/runs/         gitignored, timestamped output from local `run` invocations
engines.example.yaml    copy to engines.yaml and point at your own servers
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) -- adding a test case, adding an
engine adapter, and submitting real results are all independent, small PRs.

## License

MIT -- see [LICENSE](LICENSE).
