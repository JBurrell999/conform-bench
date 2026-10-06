"""A small, generic "give me *a* value that satisfies this JSON Schema"
generator.

This is intentionally not a general-purpose JSON Schema solver -- it covers
the subset of Schema actually used under `testcases/` (object/array/string/
integer/number/boolean/null, `enum`, `const`, `properties`/`required`,
`items`, local `$ref`/`$defs`, `additionalProperties: false`). It exists
purely so we can test *our own harness* deterministically (mock server,
integration tests) without needing a real model to decide what counts as
"a valid answer". It is never used to judge a real engine's output --
that's `validators.py`, which only ever checks real responses against
`jsonschema`.

`hints`, when given, is a dotted-path -> value map (the same shape as
`FieldCheck`) that overrides the generic default at that exact path, so
test-fixture code can get both "schema-valid" and "has the exact value the
test's checks expect" for free. A hint is also the precise way to terminate
a recursive schema (see `testcases/structured_output/recursive_tree_schema.yaml`),
but even with no hints at all this won't recurse forever: past `MAX_DEPTH`
it starts preferring the empty/terminal case (an empty array, the first
`anyOf`/`oneOf` branch) over one that would recurse further, so a
self-referential schema still gets *a* valid instance -- just not a deep one.
"""

from __future__ import annotations

from typing import Any

MAX_DEPTH = 6


def synthesize_value(
    schema: dict[str, Any],
    hints: dict[str, Any] | None = None,
    _path: str = "",
    _root: dict[str, Any] | None = None,
    _depth: int = 0,
) -> Any:
    hints = hints or {}
    root = schema if _root is None else _root

    if _path in hints:
        return hints[_path]

    if "$ref" in schema:
        return synthesize_value(_resolve_ref(schema["$ref"], root), hints, _path, root, _depth + 1)

    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0]

    if schema_type == "object" or ("properties" in schema and schema_type is None):
        result = {}
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties.keys()))
        for name in properties:
            child_path = _prefix(_path, name)
            if name in required or child_path in hints:
                result[name] = synthesize_value(properties[name], hints, child_path, root, _depth + 1)
        return result

    if schema_type == "array":
        item_schema = schema.get("items", {})
        # Beyond MAX_DEPTH, default to the bare minimum (0 unless minItems
        # says otherwise) rather than growing forever -- this is what keeps
        # a self-referential schema (a node whose `children` are more
        # nodes) from recursing infinitely when no hint says where to stop.
        if _depth >= MAX_DEPTH:
            count = schema.get("minItems", 0)
        else:
            count = max(schema.get("minItems", 1), 1)
        # A hint might target an index beyond what minItems implies (e.g.
        # `tags.1` with no `minItems` set) -- grow the array to cover it,
        # regardless of depth, since that's an explicit instruction.
        prefix = f"{_path}." if _path else ""
        for key in hints:
            if key.startswith(prefix):
                index_part = key[len(prefix):].split(".", 1)[0]
                if index_part.isdigit():
                    count = max(count, int(index_part) + 1)
        return [
            synthesize_value(item_schema, hints, _prefix(_path, str(i)), root, _depth + 1)
            for i in range(count)
        ]

    if schema_type == "string":
        if schema.get("format") == "date-time":
            return "2024-01-01T00:00:00Z"
        if schema.get("format") == "date":
            return "2024-01-01"
        return schema.get("default", "example")

    if schema_type == "integer":
        minimum = schema.get("minimum")
        return minimum if minimum is not None else schema.get("default", 1)

    if schema_type == "number":
        minimum = schema.get("minimum")
        return float(minimum) if minimum is not None else schema.get("default", 1.0)

    if schema_type == "boolean":
        return schema.get("default", True)

    if schema_type == "null":
        return None

    if "anyOf" in schema:
        return synthesize_value(schema["anyOf"][0], hints, _path, root, _depth + 1)
    if "oneOf" in schema:
        return synthesize_value(schema["oneOf"][0], hints, _path, root, _depth + 1)

    return schema.get("default", "example")


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"synth.py only resolves local refs (got {ref!r})")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _prefix(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def hints_from_checks(checks: list[Any]) -> dict[str, Any]:
    """Convert a list of `FieldCheck` into the dotted-path hint map
    `synthesize_value` expects."""
    return {c.path: c.equals for c in checks}
