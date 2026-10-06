"""A tiny, dependency-free fake engine server that speaks just enough of
the OpenAI (`/v1/chat/completions`) and Ollama-native (`/api/chat`)
contracts to drive every adapter in this project -- used by the test suite
and as a demo/dev tool so you can see the harness report real FAILs/
UNSUPPORTEDs without needing a GPU or a real model.

Run standalone:

    python -m conform_bench.mock_server --port 8800 --behavior schema_violator

Then point an `engines.yaml` entry of type `openai_compat` (or `ollama`, for
the native endpoint) at `http://127.0.0.1:8800` and run the suite against
it -- it's a good first thing to try after cloning the repo, before you've
set up any real engine.

Behaviors simulate real, observed conformance failure modes:

- conformant          -- always produces a schema-valid answer / correct tool call
- markdown_fence       -- wraps otherwise-valid JSON in a ```json fence
- schema_violator      -- drops a required field / uses the wrong type
- prose_instead_of_json -- ignores the schema and just writes English
- malformed_tool_args  -- tool call arguments string is truncated/invalid JSON
- no_tool_call         -- ignores `tools` entirely and answers in prose
- wrong_tool           -- calls a tool other than the first one offered
- http_400             -- rejects the request outright (simulates "unsupported")
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from conform_bench.synth import synthesize_value

DEFAULT_BEHAVIOR = "conformant"
BEHAVIORS = {
    "conformant",
    "markdown_fence",
    "schema_violator",
    "prose_instead_of_json",
    "malformed_tool_args",
    "no_tool_call",
    "wrong_tool",
    "http_400",
}


def _violate(value: Any) -> Any:
    """Generic, schema-agnostic way to make a previously-valid value invalid:
    drop a key, empty a list, or swap in the wrong type."""
    if isinstance(value, dict) and value:
        violated = dict(value)
        del violated[next(iter(violated))]
        return violated
    if isinstance(value, list):
        return []
    if isinstance(value, str):
        return 12345
    return None


def _chat_content_and_calls(body: dict[str, Any], behavior: str) -> tuple[int, str | None, list[dict] | None]:
    """Shared decision logic for both wire formats. Returns
    (status_code, content, tool_calls) where tool_calls is a list of
    {"name":..., "arguments": <str|dict>} or None."""
    tools = body.get("tools")
    response_format = body.get("response_format")  # OpenAI-compat wire shape
    native_format = body.get("format")  # Ollama native wire shape

    if behavior == "http_400":
        return 400, None, None

    if tools:
        if behavior == "no_tool_call":
            return 200, "I can help, but let me just answer directly.", None
        chosen = tools[1] if (behavior == "wrong_tool" and len(tools) > 1) else tools[0]
        fn = chosen.get("function", chosen)
        args = synthesize_value(fn.get("parameters", {}))
        args_str = json.dumps(args)
        if behavior == "malformed_tool_args":
            args_str = args_str[:-3]
        return 200, None, [{"name": fn.get("name"), "arguments": args_str}]

    schema = None
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = response_format["json_schema"]["schema"]
    elif isinstance(native_format, dict):
        schema = native_format  # Ollama's native `format` is the raw schema itself

    if schema is not None:
        if behavior == "prose_instead_of_json":
            return 200, "Here's a description in plain English instead of JSON.", None
        value = synthesize_value(schema)
        if behavior == "schema_violator":
            value = _violate(value)
        text = json.dumps(value)
        if behavior == "markdown_fence":
            text = f"```json\n{text}\n```"
        return 200, text, None

    return 200, "Hello from the conform-bench mock server.", None


def make_handler(behavior: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # silence request logging
            pass

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming)
            if self.path in ("/version", "/api/version", "/get_server_info"):
                self._send_json(200, {"version": "mock-1.0.0"})
            elif self.path == "/props":
                self._send_json(200, {"build_info": "mock-build", "version": "mock-1.0.0"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            body = self._read_json()
            status, content, tool_calls = _chat_content_and_calls(body, behavior)

            if self.path == "/v1/chat/completions":
                if status >= 400:
                    self._send_json(status, {"error": {"message": "request rejected by mock server (http_400 behavior)"}})
                    return
                raw_calls = None
                if tool_calls:
                    raw_calls = [
                        {"id": "call_1", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in tool_calls
                    ]
                self._send_json(
                    200,
                    {
                        "id": "mock-completion",
                        "object": "chat.completion",
                        "model": body.get("model", "mock"),
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "tool_calls" if raw_calls else "stop",
                                "message": {"role": "assistant", "content": content, "tool_calls": raw_calls},
                            }
                        ],
                    },
                )
            elif self.path == "/api/chat":
                if status >= 400:
                    self._send_json(status, {"error": "request rejected by mock server (http_400 behavior)"})
                    return
                raw_calls = None
                if tool_calls:
                    # Ollama's native wire format hands back arguments already
                    # decoded as an object, except in the malformed-args
                    # behavior where we deliberately keep it a broken string.
                    raw_calls = []
                    for c in tool_calls:
                        args = c["arguments"]
                        if behavior != "malformed_tool_args":
                            args = json.loads(args)
                        raw_calls.append({"function": {"name": c["name"], "arguments": args}})
                self._send_json(
                    200,
                    {
                        "model": body.get("model", "mock"),
                        "done": True,
                        "message": {"role": "assistant", "content": content, "tool_calls": raw_calls},
                    },
                )
            else:
                self._send_json(404, {"error": "not found"})

    return Handler


class MockEngineServer:
    """Context manager wrapping a `ThreadingHTTPServer` bound to an ephemeral
    port, for use directly from Python (tests, scripts) without shelling out
    to the CLI entry point."""

    def __init__(self, behavior: str = DEFAULT_BEHAVIOR, host: str = "127.0.0.1", port: int = 0):
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown behavior {behavior!r}; choices: {sorted(BEHAVIORS)}")
        self._httpd = ThreadingHTTPServer((host, port), make_handler(behavior))
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "MockEngineServer":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--behavior", choices=sorted(BEHAVIORS), default=DEFAULT_BEHAVIOR)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(args.behavior))
    print(f"conform-bench mock server listening on http://{args.host}:{args.port} (behavior={args.behavior})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
