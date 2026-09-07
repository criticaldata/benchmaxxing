# Serving vLLM on Modal instead of MIT SLURM GPUs

Written after using [Modal](https://modal.com) as a drop-in replacement for MIT's SLURM GPU
partitions during the BODHI-Medcalc rerun campaign (Sept 2026), when the MIT GPU QOS was maxed
out (`QOSMaxGRESPerUser`). Modal has its own, separate GPU pool, so it let us keep running while
MIT's queue was full — and once it was set up, we could run all 4 model families **in parallel**,
which we couldn't do on the shared MIT partition.

This is not a Modal tutorial — it's the specific pattern we used, plus every gotcha we hit,
so the next person doesn't have to rediscover them.

## The core idea

Split the job in two, and keep them talking over plain HTTP:

- **The GPU server** (vLLM, OpenAI-compatible) runs on Modal.
- **The client** (your existing eval script) runs wherever it already runs (MIT SLURM,
  a laptop, whatever) and just points `OPENAI_COMPATIBLE_BASE_URL` at the Modal endpoint.

If your client code already talks to an OpenAI-compatible server (most of our eval harnesses do),
**zero client-side code changes are needed.** Only the serving side moves.

## Setup (one time)

```bash
pip install modal
modal setup          # opens a browser, links your Modal account/workspace
```

You need a Modal workspace with GPU access (ask whoever owns the org's Modal account to add you).

## The server: one `@app.cls` per model

Don't try to build one generic parameterized class for every model — `@modal.web_server` needs a
fixed port and label at decoration time, so a parameterized class doesn't give you stable,
predictable URLs. Write one small class per model instead. It's more typing but it's boring and
it works.

```python
import subprocess, time, urllib.request, urllib.error
import modal

app = modal.App("my-vllm-servers")
VLLM_PORT = 8000

hf_cache_volume = modal.Volume.from_name("my-hf-cache", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "huggingface_hub[hf_transfer]")
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

def _wait_for_health(port, timeout_s=1800):
    """Block until vLLM's own /v1/models responds. Do NOT rely on @modal.web_server's
    default startup_timeout for this -- a cold weight load (especially a big/quantized
    checkpoint) can take several minutes and the default is too short."""
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM on port {port} never became healthy")

@app.cls(
    gpu="H100",
    image=vllm_image,
    volumes={"/cache/huggingface": hf_cache_volume},
    scaledown_window=600,     # stay warm 10 min between requests
    min_containers=0,         # no idle GPU spend
    max_containers=2,         # SEE "GPU quota" GOTCHA BELOW -- do not skip this
    timeout=3600,
)
class MyModelServer:
    @modal.enter()
    def start_vllm(self):
        cmd = ["vllm", "serve", "org/my-model-id",
               "--served-model-name", "org/my-model-id",
               "--port", str(VLLM_PORT), "--dtype", "auto",
               "--max-model-len", "8192", "--gpu-memory-utilization", "0.90",
               "--tensor-parallel-size", "1"]
        self._proc = subprocess.Popen(cmd)
        _wait_for_health(VLLM_PORT)

    @modal.exit()
    def stop_vllm(self):
        if getattr(self, "_proc", None):
            self._proc.terminate()

    @modal.web_server(VLLM_PORT, startup_timeout=1800, label="my-model")
    def serve(self):
        pass
```

Deploy it:

```bash
modal deploy path/to/vllm_server_app.py
```

This prints a stable HTTPS URL per class, e.g. `https://<workspace>--my-model.modal.run`. Smoke
test before trusting it with real work:

```bash
curl https://<workspace>--my-model.modal.run/v1/models
```

## The client side

Point your existing OpenAI-compatible client at the Modal URL:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://<workspace>--my-model.modal.run/v1"
python your_existing_eval_script.py --provider openai_compatible --model org/my-model-id ...
```

If the client runs on SLURM: **don't request a GPU** for the client job (`--gres` etc.) — the GPU
work happens on Modal, the client is CPU/API-bound only.

## Gotchas (the part worth actually reading)

### 1. GPU quota is silent and per-workspace — always set `max_containers`

If you deploy 3-4 model classes without an explicit `max_containers` cap, whichever ones get
traffic first will auto-scale and eat the whole workspace's GPU quota. A later class trying to
start its first container will just... never come up, with no obvious error — it looks like a
hang, not a quota error. We lost real time misdiagnosing this as a container crash loop before
realizing it was quota exhaustion.

**Fix:** always set `max_containers=N` explicitly on every class, sized so the sum across all
your classes fits your workspace's actual GPU quota. Cap first, ask questions later.

To check what's actually running and eating quota:

```bash
modal container list --json
```

Don't guess which containers belong to which model from timing alone — verify:

```bash
modal container exec <container-id> -- python3 -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/v1/models').read())"
```

### 2. Tensor-parallel + quantized weights can silently never come up

We tried `gpu="H100:2"` + `--tensor-parallel-size 2` for an AWQ-quantized 32B model to speed up
generation. The endpoint never became healthy — no crash, no error, just an eternal 000 from
curl. We didn't chase the root cause under time pressure; reverted to `gpu="H100"` +
`tensor-parallel-size 1`, which worked immediately. If you hit this, don't assume it's your code —
try TP=1 first as a sanity check before debugging further.

### 3. Client venv must have every package the eval script imports — including transitive ones

We had several venvs lying around from different experiments; more than one was missing a package
the harness needs (`sympy` for one math-tool eval, `openai`/`pandas`/`huggingface_hub` for others).
A relaunch using the wrong venv fails fast with `ModuleNotFoundError` — annoying but at least
loud. Keep one blessed venv per project with every dependency installed and reuse it everywhere,
rather than grabbing whichever `.venv312` happens to be nearby.

### 4. If you use content-addressed checkpoints, changing any CLI flag on relaunch silently starts over

If your harness resumes from a checkpoint keyed by a hash of its config (CLI args + script
source), relaunching with even one different flag (e.g. a different token budget passed via an
env var you forgot to re-set) produces a **different hash**, which the harness treats as a brand
new run — it won't error, it'll just quietly start from zero and abandon your real progress. This
bit us once: relaunching several near-finished tasks without re-exporting a token-budget env var
threw away 90%+-complete runs before we caught it (checkpoint file was created fresh instead of
being resumed).

**Fix:** when relaunching/resuming anything, diff the full set of CLI args/env vars against the
original launch before hitting go, not after.

### 5. Match your client-side wall-clock limit (SLURM `--time`, etc.) to the actual model, not a generic default

Per-request latency varies a lot by model even at similar sizes — one model we ran took roughly
3-4x longer per request than another of similar parameter count, for reasons we didn't fully
pin down (not explained by output length or contention alone). A `--time` limit that was generous
for the fast models wasn't enough for the slow one, and several tasks hit SLURM's `TIMEOUT` state
89%+ through a run. Check actual observed per-request latency for each model early, and size the
time limit per model rather than copy-pasting one value everywhere.

### 6. Stagger simultaneous dataset downloads

Launching two or more client jobs back-to-back that all call `load_dataset(...)` on the same HF
dataset at nearly the same instant can trip Hugging Face Hub's rate limiter
(`HfHubHTTPError: 429`) on the revision-tree check. Launch such jobs one at a time with a real
delay between them, not in a tight loop.

### 7. Isolate per-task caches on shared filesystems

If many parallel array tasks call `load_dataset()` against the same on-disk HF cache directory
simultaneously, you can hit `OSError: [Errno 116] Stale file handle` from lock contention on NFS.
Give each task its own `HF_DATASETS_CACHE`/`HF_HOME` (e.g. under `/tmp`, keyed by job/task id)
instead of sharing one across a whole array.

## Quick reference

| Task | Command |
|---|---|
| Deploy/update servers | `modal deploy path/to/vllm_server_app.py` |
| Dev/test one class live | `modal serve path/to/vllm_server_app.py` |
| List running containers | `modal container list --json` |
| Tail a container's logs | `modal container logs <id>` |
| Run something inside a live container | `modal container exec <id> -- <cmd>` |
| Stop a container | `modal container stop <id>` (Modal will restart it if there's still demand and you haven't lowered `max_containers`) |
| Stop the whole app | `modal app stop <app-name>` |

## When to reach for this

Good fit: you're blocked on shared cluster GPU quota, you already have an OpenAI-compatible
client, and you can tolerate a few minutes of cold-start latency the first time a model spins up.

Not a fit: sub-second latency requirements, or workloads that need tight co-location with other
cluster resources (shared filesystem throughput, InfiniBand between GPUs, etc.) that Modal doesn't
give you control over.
