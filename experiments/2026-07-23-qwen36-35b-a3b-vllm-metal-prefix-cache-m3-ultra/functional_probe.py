#!/usr/bin/env python3
"""Correctness gate for Qwen3.6 vLLM-Metal OpenAI-compatible serving."""
from __future__ import annotations
import argparse
import ast
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPLACEMENT = "\ufffd"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_temperature",
        "description": "Return the current temperature for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
}


def get_json(url: str, timeout: int = 60):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, json.load(response)


def post_json(base: str, payload: dict, timeout: int = 600):
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            return response.status, time.perf_counter() - start, body, None
    except Exception as exc:
        return None, time.perf_counter() - start, None, repr(exc)


def base_payload(model: str, messages: list[dict], max_tokens: int = 256):
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def message_from(body):
    try:
        return body["choices"][0]["message"]
    except Exception:
        return {}


def visible(msg):
    return msg.get("content") or ""


def reasoning(msg):
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def common_valid(body):
    msg = message_from(body)
    text = visible(msg)
    return bool(text or msg.get("tool_calls")) and REPLACEMENT not in text and not reasoning(msg)


def extract_code(text: str):
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    return match.group(1).strip() if match else text.strip()


def run_case(base, model, name, messages, validator, *, tools=None, tool_choice=None, max_tokens=256):
    payload = base_payload(model, messages, max_tokens)
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    status, elapsed, body, error = post_json(base, payload)
    passed = bool(status == 200 and body is not None and common_valid(body) and validator(body))
    return {
        "case": name,
        "passed": passed,
        "http_status": status,
        "elapsed_s": elapsed,
        "error": error,
        "request": payload,
        "response": body,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8094/v1")
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    status, models = get_json(args.base.rstrip("/") + "/models")
    model = args.model or models["data"][0]["id"]
    records = [{"case": "models", "passed": status == 200, "http_status": status, "response": models}]

    for index in range(5):
        records.append(run_case(
            args.base, model, f"exact_match_{index + 1}",
            [{"role": "user", "content": "Reply with exactly TOKENIZER_OK and nothing else."}],
            lambda body: visible(message_from(body)).strip() == "TOKENIZER_OK",
            max_tokens=32,
        ))

    records.append(run_case(
        args.base, model, "short_factual",
        [{"role": "user", "content": "What is the capital of France? Reply with exactly PARIS."}],
        lambda body: visible(message_from(body)).strip() == "PARIS",
        max_tokens=32,
    ))

    def code_valid(body):
        code = extract_code(visible(message_from(body)))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "merge_intervals" for node in ast.walk(tree))

    records.append(run_case(
        args.base, model, "python_syntax",
        [{"role": "user", "content": "Write only Python code defining merge_intervals(intervals), which merges overlapping [start, end] intervals."}],
        code_valid,
        max_tokens=256,
    ))

    records.append(run_case(
        args.base, model, "direct_no_tool",
        [{"role": "user", "content": "What is 2+2? Reply with exactly 4. Do not call a tool."}],
        lambda body: visible(message_from(body)).strip() == "4" and not message_from(body).get("tool_calls"),
        tools=[TOOL], tool_choice="none", max_tokens=32,
    ))

    forced = run_case(
        args.base, model, "forced_tool",
        [{"role": "user", "content": "Look up the temperature in Tokyo in Celsius. Use the provided tool."}],
        lambda body: validate_tool(message_from(body)),
        tools=[TOOL],
        tool_choice="required",
        max_tokens=128,
    )
    records.append(forced)

    if forced["passed"]:
        assistant = message_from(forced["response"])
        call = assistant["tool_calls"][0]
        continuation = [
            {"role": "user", "content": "Look up the temperature in Tokyo in Celsius. Use the provided tool."},
            assistant,
            {"role": "tool", "tool_call_id": call["id"], "name": "lookup_temperature", "content": json.dumps({"city": "Tokyo", "unit": "celsius", "temperature": 21})},
        ]
        records.append(run_case(
            args.base, model, "tool_result_continuation", continuation,
            lambda body: "21" in visible(message_from(body)) and not message_from(body).get("tool_calls"),
            tools=[TOOL], tool_choice="none", max_tokens=128,
        ))
    else:
        records.append({"case": "tool_result_continuation", "passed": False, "skipped": True, "reason": "forced_tool failed"})

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    failed = [record["case"] for record in records if not record.get("passed")]
    print(json.dumps({"model": model, "total": len(records), "passed": len(records) - len(failed), "failed": failed, "output": str(output)}, indent=2))
    return 1 if failed else 0


def validate_tool(msg):
    calls = msg.get("tool_calls") or []
    if len(calls) != 1:
        return False
    fn = calls[0].get("function") or {}
    if fn.get("name") != "lookup_temperature":
        return False
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False
    return args == {"city": "Tokyo", "unit": "celsius"}


if __name__ == "__main__":
    sys.exit(main())
