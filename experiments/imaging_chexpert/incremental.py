"""Cumulative, rate-limited CheXpert execution with API-safe checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import dotenv_values


MODEL = "meta/llama-3.2-90b-vision-instruct"
DEFAULT_TARGETS = (20, 40, 60, 80, 100)
PHASE_FILES = {
    "baseline": "imaging_blind_metric_phase1_baseline.jsonl",
    "blind": "imaging_blind_metric_phase2_blind.jsonl",
    "test_aware": "imaging_blind_metric_phase3_test_aware.jsonl",
}


def load_api_key(env_path: Path) -> tuple[str | None, str | None]:
    """Read the preferred N_2 key without sourcing arbitrary shell text."""
    values = dotenv_values(env_path)
    for name in ("N_2", "NVIDIA_API_KEY"):
        value = values.get(name)
        if value:
            return name, value
    return None, None


def _number(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def build_runner_command(
    *,
    python: str,
    runner: Path,
    manifest: Path,
    image_root: Path,
    model: str,
    cache: Path,
    out: Path,
    target: int,
    rpm: float = 20,
    request_retries: int = 2,
    timeout: float = 60,
) -> list[str]:
    return [
        python,
        str(runner),
        "--manifest",
        str(manifest),
        "--image-root",
        str(image_root),
        "--model",
        model,
        "--cache",
        str(cache),
        "--out",
        str(out),
        "--n",
        str(target),
        "--timeout",
        _number(timeout),
        "--rpm",
        _number(rpm),
        "--request-retries",
        str(request_retries),
        "--max-retries",
        "0",
    ]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def seed_checkpoints(previous_out: Path, current_out: Path, model: str) -> int:
    """Seed a new cumulative block with the previous block's phase checkpoints."""
    previous_dir = previous_out / model.replace("/", "_")
    current_dir = current_out / model.replace("/", "_")
    copied = 0
    for filename in PHASE_FILES.values():
        source = previous_dir / filename
        destination = current_dir / filename
        if source.exists() and not destination.exists():
            current_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied += 1
    return copied


def validate_block(out_root: Path, model: str, target: int, cache_path: Path) -> dict:
    """Validate a completed block without trusting its process exit code alone."""
    model_dir = out_root / model.replace("/", "_")
    errors: list[str] = []
    missing_phases: list[str] = []
    phase_counts: dict[str, int] = {}
    phase_case_ids: dict[str, set[str]] = {}
    for phase, filename in PHASE_FILES.items():
        path = model_dir / filename
        if not path.exists():
            missing_phases.append(phase)
            phase_counts[phase] = 0
            continue
        try:
            rows = _read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{phase}_checkpoint:{type(exc).__name__}")
            phase_counts[phase] = 0
            continue
        phase_counts[phase] = len(rows)
        phase_case_ids[phase] = {row.get("case_id") for row in rows}
        if len(rows) != len({row.get("case_id") for row in rows}):
            errors.append(f"{phase}_duplicate_case_ids")
        if len(rows) != target:
            errors.append(f"{phase}_count={len(rows)}")

    summary_n = None
    summary_path = model_dir / "imaging_blind_metric_summary.json"
    try:
        summary = json.loads(summary_path.read_text())
        summary_n = summary.get("n")
        if summary_n != target:
            errors.append(f"summary_n={summary_n}")
        if summary.get("model") != model:
            errors.append("summary_model_mismatch")
    except (OSError, json.JSONDecodeError):
        errors.append("summary_missing_or_invalid")

    cache_rows = 0
    duplicate_cache_keys: list[str] = []
    cache_conflicts: list[str] = []
    responses: dict[str, object] = {}
    try:
        for row in _read_jsonl(cache_path):
            cache_rows += 1
            key = row["k"]
            response = row["resp"]
            if key in responses:
                if key not in duplicate_cache_keys:
                    duplicate_cache_keys.append(key)
                if responses[key] != response:
                    cache_conflicts.append(key)
            else:
                responses[key] = response
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cache:{type(exc).__name__}")

    if missing_phases:
        errors.append("missing_phases")
    present_phase_ids = [ids for ids in phase_case_ids.values() if ids]
    if present_phase_ids and any(ids != present_phase_ids[0] for ids in present_phase_ids[1:]):
        errors.append("phase_case_ids_mismatch")
    if duplicate_cache_keys:
        errors.append("duplicate_cache_keys")
    if cache_conflicts:
        errors.append("cache_conflicts")

    return {
        "valid": not errors,
        "target": target,
        "phase_counts": phase_counts,
        "missing_phases": missing_phases,
        "summary_n": summary_n,
        "cache_rows": cache_rows,
        "cache_unique_keys": len(responses),
        "duplicate_cache_keys": duplicate_cache_keys,
        "cache_conflicts": cache_conflicts,
        "errors": errors,
    }


def probe_api(model: str, api_key: str, timeout: float = 30) -> dict:
    """Make a minimal inference request; never return or log its response body."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with only OK."}],
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
            ok = response.status == 200 and bool(data.get("choices"))
            return {"ok": ok, "http": response.status, "error": None if ok else "invalid_response"}
    except HTTPError as exc:
        return {"ok": False, "http": exc.code, "error": "http_error"}
    except (TimeoutError, socket.timeout):
        return {"ok": False, "http": 0, "error": "timeout"}
    except (URLError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "http": 0, "error": type(exc).__name__}


def _cache_rows(path: Path) -> int:
    try:
        return len(_read_jsonl(path))
    except (OSError, json.JSONDecodeError):
        return 0


def append_status(
    path: Path,
    *,
    source: str,
    target: int,
    event: str,
    probe: dict,
    cache: Path,
    model: str = MODEL,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "key_source": source,
        "target": target,
        "event": event,
        "http": probe.get("http"),
        "error": probe.get("error"),
        "cache_rows": _cache_rows(cache),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def child_environment(repo_root: Path, api_key: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Run the child against this checkout, even when another copy is installed."""
    child_env = dict(os.environ if env is None else env)
    child_env["NVIDIA_API_KEY"] = api_key
    existing = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing if existing else "")
    return child_env


def run_target(
    command: list[str],
    *,
    api_key: str,
    log_path: Path,
    repo_root: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run one cumulative target, streaming output to terminal and a safe log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = (
        child_environment(repo_root, api_key, env)
        if repo_root is not None
        else child_environment(Path.cwd(), api_key, env)
    )
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def run_incremental(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    env_path = root / args.env_file
    cache = root / args.cache
    output_root = root / args.output_root
    status_log = root / args.status_log
    runner = root / "experiments/imaging_chexpert/imaging_blind_metric.py"
    manifest = root / args.manifest
    image_root = Path(args.image_root).expanduser()

    previous_out: Path | None = None
    for target in args.targets:
        while True:
            source, api_key = load_api_key(env_path)
            if not api_key or not source:
                probe = {"ok": False, "http": 0, "error": "missing_api_key"}
                append_status(status_log, source="none", target=target, event="api_unavailable", probe=probe, cache=cache)
                time.sleep(args.retry_wait)
                continue

            probe = probe_api(args.model, api_key, timeout=args.probe_timeout)
            if not probe["ok"]:
                append_status(status_log, source=source, target=target, event="api_unavailable", probe=probe, cache=cache)
                time.sleep(args.retry_wait)
                continue

            out = output_root / f"phase2_n{target}"
            if previous_out is not None:
                seed_checkpoints(previous_out, out, args.model)
            command = build_runner_command(
                python=args.python,
                runner=runner,
                manifest=manifest,
                image_root=image_root,
                model=args.model,
                cache=cache,
                out=out,
                target=target,
                rpm=args.rpm,
                request_retries=args.request_retries,
                timeout=args.timeout,
            )
            log_path = out / "run.log"
            return_code = run_target(
                command,
                api_key=api_key,
                log_path=log_path,
                repo_root=root,
            )
            report = validate_block(out, args.model, target, cache)
            if return_code == 0 and report["valid"]:
                append_status(
                    status_log,
                    source=source,
                    target=target,
                    event="block_complete",
                    probe=probe,
                    cache=cache,
                    model=args.model,
                )
                print(json.dumps(report, sort_keys=True))
                previous_out = out
                break

            event = "validation_failed" if return_code == 0 else "runner_failed"
            append_status(
                status_log,
                source=source,
                target=target,
                event=event,
                probe=probe,
                cache=cache,
                model=args.model,
            )
            print(json.dumps(report, sort_keys=True), file=sys.stderr)
            time.sleep(args.retry_wait)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--manifest", default="experiments/imaging_chexpert/results/solo_600.csv")
    parser.add_argument("--image-root", default="/Users/yehu/.cache/kagglehub/datasets/ashery/chexpert/versions/1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cache", default="experiments/chexpert/results/600_runs/img_cache_llama_n2.jsonl")
    parser.add_argument("--output-root", default="experiments/chexpert/results")
    parser.add_argument(
        "--status-log",
        default="experiments/chexpert/results/api_watchdog_n2_incremental.jsonl",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--targets", type=int, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--rpm", type=float, default=20)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--probe-timeout", type=float, default=30)
    parser.add_argument("--retry-wait", type=float, default=60)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_incremental(parse_args()))
