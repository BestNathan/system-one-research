from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx


def auth_values(key: str, style: str) -> list[str]:
    if style == "raw":
        return [key]
    if style == "bearer":
        return [key if key.lower().startswith("bearer ") else f"Bearer {key}"]
    if key.lower().startswith("bearer "):
        return [key]
    # The current Alibaba examples are inconsistent across pages: some show
    # the raw key while the API reference says Bearer. Try the user-provided
    # raw form first, then standards-compatible Bearer only on auth failure.
    return [key, f"Bearer {key}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--auth-env", required=True)
    ap.add_argument("--auth-style", choices=["raw", "bearer", "auto"], default="auto")
    ap.add_argument("--backend-name", required=True)
    args = ap.parse_args()

    key = os.environ.get(args.auth_env)
    if not key:
        raise SystemExit(f"missing required environment variable: {args.auth_env}")

    case = json.loads(args.case.read_text(encoding="utf-8"))
    payload = {
        "model": args.model,
        "state": case["state"],
        "questions": case["questions"],
    }

    attempts = []
    body = None
    elapsed_ms = None
    used_auth = None

    with httpx.Client(timeout=90) as client:
        for auth_index, auth in enumerate(auth_values(key, args.auth_style)):
            t0 = time.perf_counter()
            response = client.post(
                args.url,
                headers={"Authorization": auth, "Content-Type": "application/json"},
                json=payload,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            attempts.append({
                "auth_variant": "bearer" if auth.lower().startswith("bearer ") else "raw",
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
            })
            if response.status_code not in (401, 403):
                response.raise_for_status()
                body = response.json()
                used_auth = attempts[-1]["auth_variant"]
                break
            if auth_index == len(auth_values(key, args.auth_style)) - 1:
                response.raise_for_status()

    if body is None:
        raise RuntimeError("request completed without a response body")

    result = {
        "backend": args.backend_name,
        "model_requested": args.model,
        "model_reported": body.get("model"),
        "case_id": case.get("id"),
        "elapsed_ms": elapsed_ms,
        "auth_variant": used_auth,
        "attempts": attempts,
        "answers": body.get("answers"),
        "raw": body,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": result["backend"],
        "model_requested": result["model_requested"],
        "model_reported": result["model_reported"],
        "case_id": result["case_id"],
        "elapsed_ms": result["elapsed_ms"],
        "auth_variant": result["auth_variant"],
        "answer_keys": sorted((result["answers"] or {}).keys()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
