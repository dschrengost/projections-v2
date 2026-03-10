# Inference Server Spec

## Spec Status: LIVING v0.5 (2026-03-10)

---

## 1. Motivation

### 1.1 Current gap

The original live pipeline ran model inference inline within Prefect tasks:

```
Prefect task
    → load GameTransformerV2 bundle (~5-10s cold)
    → build inference examples
    → forward pass + world sampling
    → write artifacts
```

That design had major latency problems:

1. **Cold-start penalty**: Every task invocation pays model load cost.
2. **No GPU utilization**: CPU inference is 3-5x slower than GPU for transformers.
3. **Sequential game processing**: Late-news bursts (3-5 games) queue serially.
4. **No batching across games**: Each game is a separate subprocess invocation.

Current post-cutover bottlenecks (as of 2026-03-10):

1. Python-side sampling/projection hotspots can cap throughput even when GPU is active.
2. Sequential game handling protects VRAM but limits peak throughput by design.
3. Remaining heavy blocks are now minutes projection and contract-check synchronization paths, not row materialization.

### 1.2 What this spec changes

1. Add a dedicated inference server (Triton) for GPU-accelerated scoring.
2. Keep Prefect as the orchestration layer for scraping, feature building,
   finalization, and publish gates.
3. Enable per-game sequential scoring with warm model weights.
4. Define explicit latency budgets and failure modes.

### 1.3 Non-negotiable requirements

- Transformer inference latency < 8s per game for 25k worlds on GPU.
- 3-5 affected games complete in < 60s total.
- Late-breaking news triggers targeted game re-scoring, not full-slate rebuild.
- Inference failures are visible to operators and do not silently publish stale
  data.
- Inference server downtime does not corrupt published artifacts.

Current measured baseline (2026-03-10):

- Single-game worlds request (`25k` worlds, `chunk=5000`) historical:
  - pre-optimization: `~132s`
  - after `JointMinutes` optimization: `~15.6-15.8s`
- Single-game sampler microbenchmark (`sample_worlds_for_batch`, local in-process, `25k`, `chunk=5000`): `~2.20s`
- 5-game sequential replay-mode full flow: `156.8s`
- End-to-end replay flow with the latest sampler optimizations has not yet been rerun/documented.

---

## 2. Design Principles

1. **Separation of concerns**: Prefect owns orchestration; Triton owns inference.

2. **Warm over cold**: Model weights stay loaded; requests pay only forward-pass
   cost.

3. **Game-level granularity**: The inference unit is a single game, not a slate.

4. **Fail-closed**: If inference cannot complete within SLA, the system blocks
   publish and surfaces the failure—never falls back to stale or alternate-model
   outputs.

5. **Incremental optimization**: Start with Python backend (keep PyTorch code),
   then optimize hot paths to TensorRT as needed.

---

## 3. Architecture Overview

### 3.1 High-level topology

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Prefect Flow                                │
│                   (nba_live_pipeline_v3_flow)                       │
│                                                                     │
│  scrape_inputs → freeze_manifest → build_features → [inference] →  │
│  finalize → postflight → publish                                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │ gRPC / HTTP
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Triton Inference Server                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  gtv2_scorer (Python backend)                               │   │
│  │                                                             │   │
│  │  - GameTransformerV2 backbone                               │   │
│  │  - JointActiveSetHead                                       │   │
│  │  - JointMinutesHead                                         │   │
│  │  - JointGameFlow                                            │   │
│  │  - PossessionBackbone (if enabled)                          │   │
│  │                                                             │   │
│  │  Loaded once at startup, kept warm                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  GPU: NVIDIA RTX 3060 12GB                                          │
│  Concurrency: max_batch_size=1, instance_count=1                    │
│  Queue depth: 5-8 games                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Request flow

1. Prefect task builds feature tensor for affected game(s).
2. Prefect task splits features by game and sends one request at a time to
   Triton via HTTP.
3. Triton executes each request on GPU with warm weights.
4. Triton returns world tensors to Prefect task.
5. Prefect task merges per-game outputs, writes run-scoped artifacts, and
   proceeds to finalize.

### 3.3 Concurrency model

The 3060 has 12GB VRAM. Running multiple games truly in parallel would thrash
memory. Instead:

- Prefect submits one game request at a time from each run.
- Triton still keeps queueing semantics if multiple runs overlap.
- Queue depth is bounded (5-8 games) to prevent unbounded backlog.
- Total wall-clock for 5 games ≈ 5 × single-game latency.

---

## 4. Model Serving Configuration

### 4.1 Model repository structure

```
$TRITON_MODEL_REPO/
└── gtv2_scorer/
    ├── config.pbtxt
    └── 1/
        └── model.py          # Python backend
```

### 4.2 Python backend rationale

The GameTransformerV2 stack includes components that are difficult to export to
TorchScript or ONNX:

- `JointGameFlow.sample()` — coupling blocks with dynamic masking
- `JointActiveSetHead` — Gumbel-TopK sampling
- `capped_simplex_projection()` — iterative solver
- `PossessionBackbone` — Student-t reparameterization

Using Triton's Python backend preserves the existing PyTorch code while gaining:

- Warm model weights (no load per request)
- GPU scheduling and health checks
- Metrics and observability
- Path to incremental TensorRT optimization

### 4.3 config.pbtxt template

```protobuf
name: "gtv2_scorer"
backend: "python"

max_batch_size: 0

input [
  {
    name: "request_json"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "response_json"
    data_type: TYPE_STRING
    dims: [1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]

parameters [
  {
    key: "project_root"
    value: { string_value: "/home/daniel/projects/projections-v2" }
  },
  {
    key: "bundle_dir"
    value: { string_value: "/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current" }
  },
  {
    key: "num_worlds"
    value: { string_value: "25000" }
  },
  {
    key: "world_chunk_size"
    value: { string_value: "5000" }
  },
  {
    key: "device"
    value: { string_value: "cuda:0" }
  }
]
```

### 4.4 Model versioning

Triton model versions map to promoted GTV2 bundles:

```
gtv2_scorer/
├── 1/    # bundle_20260301T120000Z
├── 2/    # bundle_20260315T090000Z (candidate)
└── 3/    # bundle_20260320T140000Z (experimental)
```

Production traffic routes to the version specified in
`config/gtv2_inference_current.json`. Canary traffic can target a different
version via request metadata.

---

## 5. Prefect Integration

### 5.1 Inference task

```python
@task(retries=2, retry_delay_seconds=5)
def score_game_triton(
    game_id: str,
    features: np.ndarray,
    game_metadata: dict,
    triton_url: str = "localhost:8001",
    timeout_seconds: float = 30.0,
) -> dict:
    """
    Send inference request to Triton and return world tensors.

    Raises:
        InferenceTimeoutError: If request exceeds timeout.
        InferenceServerError: If Triton returns an error.
    """
    ...
```

### 5.2 Sequential game scoring

```python
@task
def score_affected_games(
    affected_games: list[str],
    features_by_game: dict[str, np.ndarray],
    triton_url: str,
) -> dict[str, dict]:
    """
    Submit affected games sequentially (one request in flight at a time).
    This keeps peak VRAM pressure bounded on 12GB GPUs.
    """
    results = {}
    for game_id in affected_games:
        results[game_id] = score_game_triton(
            game_id=game_id,
            features=features_by_game[game_id],
            triton_url=triton_url,
        )

    return results
```

### 5.3 Failure handling

| Failure mode | Behavior |
|--------------|----------|
| Triton unreachable | Retry 2x with backoff, then fail task |
| Request timeout | Retry 1x, then fail task |
| Model error | Fail task, capture error in run manifest |
| Partial success (some games fail) | Fail entire scoring stage |

Failed scoring stages block publish. The operator sees the failure in Prefect UI
and can retry manually or investigate.

---

## 6. Latency Budget

### 6.1 Per-stage targets

| Stage | Target | Notes |
|-------|--------|-------|
| Feature build (per game) | < 5s | Mostly I/O bound |
| Inference (per game, 25k worlds) | < 8s | GPU target |
| Inference (per game, 25k worlds) | < 45s | CPU fallback |
| Finalize (full slate) | < 10s | Merge + write |
| Postflight + publish | < 5s | Validation + pointer |

### 6.2 End-to-end targets

| Scenario | Target |
|----------|--------|
| Single-game late-news rerun | < 30s |
| 5-game sequential burst rerun | < 60s |
| Full-slate rebuild (8 games) | < 120s |

### 6.3 Measurement

Add structured timing to every inference request:

```json
{
  "game_id": "0022500123",
  "request_queued_ts": "2026-03-03T19:45:00.000Z",
  "inference_start_ts": "2026-03-03T19:45:00.050Z",
  "inference_end_ts": "2026-03-03T19:45:06.200Z",
  "total_latency_ms": 6250,
  "queue_latency_ms": 50,
  "inference_latency_ms": 6150,
  "num_worlds": 25000
}
```

### 6.4 Measured results (2026-03-10)

| Measurement | Result | Notes |
|-------------|--------|-------|
| Single-game Triton worlds (`25k`, `chunk=5000`) pre-optimization | ~132s | Reproduced on game-scoped replay request |
| Single-game Triton worlds (`25k`, `chunk=5000`) post-optimization | ~15.6-15.8s | After vectorized capped-simplex in `joint_minutes.py` |
| Single-game local sampler core (`sample_worlds_for_batch`, `25k`, `chunk=5000`) latest | ~2.20s | After vectorized active-set + world-row materialization; excludes request/IO overhead |
| Full 5-game replay-mode flow (`25k`, sequential) | 156.8s | Flow run `02af2e57-5b65-4882-bd73-41c824626b91` |
| `world_chunk_size=25000` VRAM footprint | ~10.7GB | No meaningful speed gain vs `5000` |
| `world_chunk_size=5000` VRAM footprint | ~2.8-2.9GB | Preferred stable operating point on 12GB 3060 |

---

## 7. Warm Process Strategy

### 7.1 Startup behavior

Triton loads the model at server startup and keeps it resident:

- Model weights loaded to GPU VRAM once.
- CUDA context initialized once.
- No per-request model load overhead.

### 7.2 Health checks

Triton exposes `/v2/health/ready` and `/v2/health/live` endpoints.

Prefect tasks should check health before submitting inference:

```python
def check_triton_health(triton_url: str) -> bool:
    response = requests.get(f"http://{triton_url}/v2/health/ready")
    return response.status_code == 200
```

### 7.3 Keep-warm strategy

During active slates, Triton stays warm naturally from request traffic.

For cold periods (overnight, no games):

- Accept that first request pays CUDA warmup (~1-2s).
- Optionally send a synthetic warm-up request at flow start.

---

## 8. GPU Resource Management

### 8.1 VRAM budget

| Component | Estimated VRAM |
|-----------|----------------|
| Model weights (FP16) | ~200-400 MB |
| Per-game inference tensors | ~500 MB - 1 GB |
| CUDA overhead | ~500 MB |
| **Total per request** | ~1.5-2 GB |
| **Available (3060 12GB)** | ~10 GB usable |

Headroom is sufficient for sequential game processing with margin for spikes.

### 8.2 Mixed precision

Use FP16 inference by default:

```python
model = model.half().cuda()
```

FP16 reduces memory footprint and improves throughput on Ampere GPUs.

### 8.3 Memory cleanup

After each request, explicitly clear intermediate tensors:

```python
del worlds_tensor
torch.cuda.empty_cache()
```

This prevents memory fragmentation during long-running sessions.

---

## 9. Deployment and Operations

### 9.1 Docker-first runbook

```bash
# Build runtime image (includes torch + parquet deps)
./scripts/triton/run_triton_gtv2.sh build

# Start Triton container (defaults: 18000/18001/18002)
./scripts/triton/run_triton_gtv2.sh start

# Readiness/model check
./scripts/triton/run_triton_gtv2.sh smoke

# Inspect status/logs
./scripts/triton/run_triton_gtv2.sh status
./scripts/triton/run_triton_gtv2.sh logs

# Stop
./scripts/triton/run_triton_gtv2.sh stop
```

`run_triton_gtv2.sh` mounts:

- `/home/daniel/projections-data/triton_models` → `/models`
- `/home/daniel/projects/projections-v2` (read-only)
- `/home/daniel/projections-data` (read-write)
- Runs container as host UID/GID to avoid root-owned output artifacts.

This keeps Triton path resolution identical to host paths used by Prefect.

### 9.2 Systemd (optional)

Use [`scripts/triton/triton-inference.service.example`](../../scripts/triton/triton-inference.service.example).

### 9.3 Management commands

```bash
cp scripts/triton/triton-inference.service.example \
  ~/.config/systemd/user/triton-inference.service
systemctl --user daemon-reload
systemctl --user enable --now triton-inference.service
systemctl --user status triton-inference.service
journalctl --user -u triton-inference.service -f
```

### 9.4 Model updates

When promoting a new GTV2 bundle:

1. Copy bundle to new version directory in model repository.
2. Update `config/gtv2_inference_current.json` with new version.
3. Re-run `scripts/triton/setup_gtv2_model_repo.py` to refresh `config.pbtxt`.
4. Restart Triton container/service.
5. Verify health and run smoke test.

Triton supports model reload without full restart, but for simplicity,
restart-based updates are acceptable at current scale.

---

## 10. Observability

### 10.1 Metrics

Triton exposes Prometheus metrics at `:8002/metrics`:

- `nv_inference_request_success` — successful requests
- `nv_inference_request_failure` — failed requests
- `nv_inference_queue_duration_us` — queue wait time
- `nv_inference_compute_infer_duration_us` — inference time
- `nv_gpu_memory_used_bytes` — VRAM usage

### 10.2 Logging

Log every inference request with:

- `game_id`
- `run_id` (from Prefect)
- `latency_ms`
- `num_worlds`
- `success` / `error`

### 10.3 Alerting

Alert on:

- Triton health check failure
- Inference latency > 15s (2x target)
- Request queue depth > 10
- GPU memory > 90%

---

## 11. Failure Modes and Recovery

### 11.1 Triton crashes

| Scenario | Detection | Recovery |
|----------|-----------|----------|
| Process crash | systemd detects exit | Auto-restart via `Restart=on-failure` |
| OOM kill | systemd detects exit | Auto-restart; investigate memory leak |
| CUDA error | Health check fails | Manual investigation required |

### 11.2 Inference timeout

If a game request exceeds timeout:

1. Prefect task retries once.
2. If retry fails, task fails.
3. Scoring stage fails; publish blocked.
4. Operator investigates via Prefect UI and Triton logs.

### 11.3 Model loading failure

If Triton cannot load model at startup:

1. Health endpoint returns not ready.
2. Prefect tasks fail on health check.
3. Operator checks Triton logs for load error.
4. Fix config or bundle, restart Triton.

### 11.4 Degraded mode (no GPU)

If GPU is unavailable (driver issue, hardware failure):

- Triton can fall back to CPU instance if configured.
- CPU inference is 5-6x slower but functional.
- Alert on GPU unavailability.
- Do not silently serve CPU results as if GPU; log clearly.

---

## 12. Incremental Optimization Path

### 12.1 Phase 1: Python backend (MVP)

- Keep all PyTorch code as-is.
- Triton handles serving, health, metrics.
- Target: < 8s per game on GPU.

### 12.2 Phase 2: TensorRT backbone

- Export `GameTransformerV2.forward()` to TorchScript or ONNX.
- Convert to TensorRT for fused transformer kernels.
- Keep sampling heads in Python.
- Target: < 5s per game.

### 12.3 Phase 3: Full optimization

- Profile remaining bottlenecks.
- Consider TensorRT for flow sampling if beneficial.
- Optimize memory layout for better throughput.
- Target: < 3s per game.

---

## 13. Testing Strategy

### 13.1 Unit tests

- Model loading and forward pass.
- World sampling output shapes and constraints.
- Input validation and error handling.

### 13.2 Integration tests

- Prefect task → Triton → response roundtrip.
- Sequential per-game submission.
- Timeout and retry behavior.

### 13.3 Load tests

- 5-game sequential burst, measure total latency.
- 10 sequential games, measure throughput.
- Memory usage over extended session.

### 13.4 Smoke test script

```bash
# Verify Triton is serving and model responds
uv run python scripts/triton/smoke_test_gtv2.py \
    --triton-endpoint localhost:18000 \
    --game-date 2026-03-10 \
    --num-worlds 256
```

---

## 14. Implementation Roadmap

### Phase 1: Infrastructure setup

- [x] Install Triton Inference Server (Docker runtime + NVIDIA toolkit).
- [x] Configure model repository structure.
- [x] Write Python backend wrapper for GTV2.
- [x] Add systemd service template for Triton.
- [x] Verify health endpoints and basic metrics.

### Phase 2: Prefect integration

- [x] Add `score_game_triton` task.
- [x] Add Triton health check to scoring stage.
- [x] Wire per-game sequential submission.
- [x] Add timeout and retry logic.
- [x] Update `live_nba_pipeline_v3.py` to use Triton path.

### Phase 3: Validation

- [x] Benchmark single-game latency (GPU vs CPU baseline).
- [x] Benchmark 5-game sequential burst latency.
- [x] Run replay-mode validation flow with runtime stamp capture.
- [ ] Compare world outputs to CPU baseline for parity.
- [x] Run Triton smoke test for both actions (`score`, `worlds`) on live features.

### Phase 4: Production cutover

- [ ] Document rollback procedure.
- [ ] Enable Triton path for one slate with monitoring.
- [ ] Promote to default path after validation.
- [ ] Remove CPU fallback code (or keep as explicit degraded mode).

### Phase 5: Optimization (post-cutover)

- [x] Profile inference bottlenecks.
- [x] Optimize `JointMinutes` capped-simplex projection (vectorized; removed scalar sync hot path).
- [x] Optimize active-set sampling hotspot (`joint_active_set`) with batched top-k selection.
- [x] Optimize world-row materialization (`_build_world_rows`) with vectorized DataFrame construction.
- [x] Optimize make-sampling path by replacing distribution-object sampling with tensor-native beta/binomial sampling.
- [ ] Add structured per-request timing telemetry into run artifacts.
- [ ] Reduce contract-check synchronization overhead (move off critical path or batch host sync).
- [ ] Export backbone to TensorRT.
- [ ] Re-benchmark and document gains.

---

## 15. Configuration Files

### 15.1 Inference server config

`config/gtv2_inference_server.json`:

```json
{
  "enabled": true,
  "backend": "triton",
  "triton_endpoint": "localhost:18000",
  "model_name": "gtv2_scorer",
  "model_version": "1",
  "timeout_seconds": 90.0,
  "healthcheck_timeout_seconds": 3.0
}
```

### 15.2 Current bundle pointer

`config/gtv2_inference_current.json`:

```json
{
  "bundle_dir": "/home/daniel/projections-data/artifacts/game_transformer_v2/bundle_current",
  "bundle_hash": null,
  "promoted_at": null,
  "model_version": "1"
}
```

---

## 16. Relation to Other Specs

### 16.1 GAME_TRANSFORMER_SPEC.md

That spec defines the model architecture, training, and output contracts.
This spec defines how that model is served in production.

### 16.2 LIVE_PIPELINE_PRODUCTION_SPEC.md

That spec defines orchestration, freshness gates, and publish contracts.
This spec defines the inference service that the orchestration calls.

Key integration points:

- Section 10.3 (GPU integration requirements) → detailed here.
- Section 10.4 (warm process / serving model) → detailed here.
- Section 6.2 (latency budgets) → refined here with Triton-specific targets.

---

## 17. Open Questions

### 17.1 Resolved

1. **Triton vs Ray Serve**: Triton selected for lower inference latency on
   transformer workloads. Ray Serve's advantages (Prefect integration, Python
   native) are outweighed by Triton's kernel optimization and serving maturity.

2. **Python backend vs TorchScript**: Start with Python backend to preserve
   existing code. TorchScript export is Phase 2 optimization.

3. **Game handling mode**: Per-game sequential submission from Prefect to keep
   GPU memory pressure predictable. True parallel execution would thrash VRAM.

### 17.2 Open

1. **CPU fallback policy**: Should Triton be configured with a CPU fallback
   instance, or should GPU unavailability be a hard failure?

2. **Canary traffic routing**: How to route a percentage of requests to a
   candidate model version for live validation?

3. **Model reload without restart**: Is zero-downtime model update worth the
   complexity, or is restart-based update acceptable?

4. **Priority queue**: Should late-breaking single-game requests preempt queued
   batch requests?

---

## 18. Agent Handoff

### 18.1 Current state (2026-03-10)

- GameTransformerV2 model is trained and promoted.
- Local (in-process) inference is working in `live_nba_pipeline_v3.py`.
- GPU hardware (RTX 3060 12GB) installed on 2026-03-10.
- Dockerized Triton runtime image is implemented:
  - `scripts/triton/Dockerfile`
  - `scripts/triton/requirements-gtv2-runtime.txt`
  - `scripts/triton/run_triton_gtv2.sh`
- Triton integration code is implemented:
  - `projections/pipeline/triton_inference_client.py`
  - `scripts/triton/model_repository/gtv2_scorer/`
  - `scripts/triton/setup_gtv2_model_repo.py`
  - `scripts/triton/smoke_test_gtv2.py`
  - `prefect_flows/live_nba_pipeline_v3.py` backend switch (`local|triton`)
- End-to-end Triton smoke (`score` + `worlds`) succeeded on 2026-03-10.
- Runtime bottleneck profile completed; major hotspot identified in
  `projections/rotation/joint_minutes.py`.
- `JointMinutes` projection path optimized (vectorized) and validated.
- `JointActiveSet` team selection path optimized with batched top-k.
- `_build_world_rows` fully vectorized to remove Python dict-per-row overhead.
- Single-game latency trajectory:
  - single-game worlds latency improved from ~132s to ~15.6-15.8s
  - latest local sampler core benchmark (`25k`, `chunk=5000`): ~2.20s
  - 5-game replay-mode full flow completed in 156.8s

### 18.2 Next steps

1. Complete CPU-vs-Triton world parity diff on the same frozen features.
2. Optimize remaining hotspots:
   - contract-check synchronization in `check_world_contracts`
   - further minutes projection speedups (`project_minutes_capped_simplex`)
3. Add automated latency regression guardrails in flow artifacts/alerts.
4. Re-run and record end-to-end 5-game replay flow with latest optimized sampler.
5. Decide CPU fallback policy (`strict` vs explicit degraded mode).

### 18.3 Dependencies

- NVIDIA driver and CUDA toolkit installed.
- Triton Inference Server installed (container or bare-metal).
- Model bundle accessible from Triton container/process.

---

## 19. Summary

This spec introduces a dedicated GPU inference server (Triton) for
GameTransformerV2 world generation with warm model serving and production-safe
failure semantics.

Key decisions:

- Triton with Python backend for MVP; TensorRT optimization later.
- Per-game sequential submission from Prefect (single in-flight request per run).
- Prefect remains the orchestrator; Triton is a service dependency.
- Fail-closed on inference failure; no alternate-model fallback.
- Explicit latency budgets: stretch target < 8s per game, < 60s for 5-game burst.

Current measured trajectory on March 10, 2026 is:

- historical per-game Triton worlds (`25k`, `chunk=5000`): ~132s → ~15.6-15.8s
- latest local sampler-core benchmark (`25k`, `chunk=5000`): ~2.20s
- latest documented 5-game replay-mode full flow: 156.8s

Remaining work is focused on contract-check synchronization/minutes-projection
hotspots plus refreshed end-to-end replay measurements with structured timing capture.
