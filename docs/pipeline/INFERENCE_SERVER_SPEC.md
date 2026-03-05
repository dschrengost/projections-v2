# Inference Server Spec

## Spec Status: DRAFT v0.1 (2026-03-03)

---

## 1. Motivation

### 1.1 Current gap

The current live pipeline runs model inference inline within Prefect tasks:

```
Prefect task
    → load GameTransformerV2 bundle (~5-10s cold)
    → build inference examples
    → forward pass + world sampling
    → write artifacts
```

This design has latency problems:

1. **Cold-start penalty**: Every task invocation pays model load cost.
2. **No GPU utilization**: CPU inference is 3-5x slower than GPU for transformers.
3. **Sequential game processing**: Late-news bursts (3-5 games) queue serially.
4. **No batching across games**: Each game is a separate subprocess invocation.

### 1.2 What this spec changes

1. Add a dedicated inference server (Triton) for GPU-accelerated scoring.
2. Keep Prefect as the orchestration layer for scraping, feature building,
   finalization, and publish gates.
3. Enable concurrent game scoring with warm model weights.
4. Define explicit latency budgets and failure modes.

### 1.3 Non-negotiable requirements

- Transformer inference latency < 8s per game for 25k worlds on GPU.
- 3-5 concurrent game requests complete in < 40s total.
- Late-breaking news triggers targeted game re-scoring, not full-slate rebuild.
- Inference failures are visible to operators and do not silently publish stale
  data.
- Inference server downtime does not corrupt published artifacts.

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
2. Prefect task sends inference request(s) to Triton via gRPC.
3. Triton queues and executes requests sequentially on GPU.
4. Triton returns world tensors to Prefect task.
5. Prefect task writes run-scoped artifacts and proceeds to finalize.

### 3.3 Concurrency model

The 3060 has 12GB VRAM. Running multiple games truly in parallel would thrash
memory. Instead:

- Triton accepts concurrent requests but executes them sequentially.
- Queue depth is bounded (5-8 games) to prevent unbounded backlog.
- Prefect submits all affected games concurrently; Triton serializes execution.
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

max_batch_size: 0  # Dynamic batching disabled; game-level requests

input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [-1, -1]  # (num_players, num_features)
  },
  {
    name: "game_metadata"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "worlds"
    data_type: TYPE_FP32
    dims: [-1, -1, -1]  # (num_worlds, num_players, num_stats)
  },
  {
    name: "summary"
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
    key: "bundle_dir"
    value: { string_value: "/path/to/promoted/gtv2/bundle" }
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

### 5.2 Concurrent game scoring

```python
@task
def score_affected_games(
    affected_games: list[str],
    features_by_game: dict[str, np.ndarray],
    triton_url: str,
) -> dict[str, dict]:
    """
    Submit all affected games concurrently.
    Triton serializes execution; Prefect gathers results.
    """
    futures = []
    for game_id in affected_games:
        fut = score_game_triton.submit(
            game_id=game_id,
            features=features_by_game[game_id],
            triton_url=triton_url,
        )
        futures.append((game_id, fut))

    results = {}
    for game_id, fut in futures:
        results[game_id] = fut.result()

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
| 5-game concurrent rerun | < 60s |
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

### 9.1 Systemd service

```ini
# /home/daniel/.config/systemd/user/triton-inference.service

[Unit]
Description=Triton Inference Server for GTV2
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/daniel/prod/projections-v2
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/usr/bin/tritonserver \
    --model-repository=/home/daniel/projections-data/triton_models \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002 \
    --log-verbose=1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

### 9.2 Management commands

```bash
# Start server
systemctl --user start triton-inference.service

# Check status
systemctl --user status triton-inference.service

# View logs
journalctl --user -u triton-inference.service -f

# Restart after model update
systemctl --user restart triton-inference.service
```

### 9.3 Model updates

When promoting a new GTV2 bundle:

1. Copy bundle to new version directory in model repository.
2. Update `config/gtv2_inference_current.json` with new version.
3. Restart Triton to load new version.
4. Verify health and run smoke test.

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
- Concurrent game submission.
- Timeout and retry behavior.

### 13.3 Load tests

- 5 concurrent games, measure total latency.
- 10 sequential games, measure throughput.
- Memory usage over extended session.

### 13.4 Smoke test script

```bash
# Verify Triton is serving and model responds
uv run python scripts/triton/smoke_test_gtv2.py \
    --triton-url localhost:8001 \
    --game-date 2026-03-03 \
    --game-id 0022500890
```

---

## 14. Implementation Roadmap

### Phase 1: Infrastructure setup

- [ ] Install Triton Inference Server.
- [ ] Configure model repository structure.
- [ ] Write Python backend wrapper for GTV2.
- [ ] Add systemd service for Triton.
- [ ] Verify health endpoints and basic metrics.

### Phase 2: Prefect integration

- [ ] Add `score_game_triton` task.
- [ ] Add Triton health check to scoring stage.
- [ ] Wire concurrent game submission.
- [ ] Add timeout and retry logic.
- [ ] Update `live_nba_pipeline_v3.py` to use Triton path.

### Phase 3: Validation

- [ ] Benchmark single-game latency (GPU vs CPU baseline).
- [ ] Benchmark 5-game concurrent latency.
- [ ] Run on live slate with `promote_pointers=false`.
- [ ] Compare world outputs to CPU baseline for parity.

### Phase 4: Production cutover

- [ ] Document rollback procedure.
- [ ] Enable Triton path for one slate with monitoring.
- [ ] Promote to default path after validation.
- [ ] Remove CPU fallback code (or keep as explicit degraded mode).

### Phase 5: Optimization (post-cutover)

- [ ] Profile inference bottlenecks.
- [ ] Export backbone to TensorRT.
- [ ] Re-benchmark and document gains.

---

## 15. Configuration Files

### 15.1 Inference server config

`config/gtv2_inference_server.json`:

```json
{
  "triton_url": "localhost:8001",
  "model_name": "gtv2_scorer",
  "model_version": "1",
  "timeout_seconds": 30,
  "num_worlds": 25000,
  "world_chunk_size": 5000,
  "device": "cuda:0",
  "enable_fp16": true
}
```

### 15.2 Current bundle pointer

`config/gtv2_inference_current.json`:

```json
{
  "bundle_dir": "/home/daniel/projections-data/artifacts/game_transformer_v2/bundles/phase3_seed42_20260301",
  "bundle_hash": "abc123...",
  "promoted_at": "2026-03-01T12:00:00Z",
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

3. **Concurrent game handling**: Sequential GPU execution with concurrent
   request queueing. True parallel execution would thrash VRAM.

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

### 18.1 Current state (2026-03-03)

- GameTransformerV2 model is trained and promoted.
- CPU inference is working in `live_nba_pipeline_v3.py`.
- GPU hardware (RTX 3060 12GB) is on order.
- No Triton infrastructure exists yet.

### 18.2 Next steps

1. Set up Triton development environment once GPU arrives.
2. Write Python backend wrapper that loads GTV2 bundle.
3. Benchmark GPU vs CPU inference latency.
4. Integrate with Prefect scoring tasks.
5. Validate on live slate before production cutover.

### 18.3 Dependencies

- NVIDIA driver and CUDA toolkit installed.
- Triton Inference Server installed (container or bare-metal).
- Model bundle accessible from Triton container/process.

---

## 19. Summary

This spec introduces a dedicated GPU inference server (Triton) to meet the
sub-10-second per-game latency target for GameTransformerV2 world generation.

Key decisions:

- Triton with Python backend for MVP; TensorRT optimization later.
- Sequential GPU execution with concurrent request queueing.
- Prefect remains the orchestrator; Triton is a service dependency.
- Fail-closed on inference failure; no alternate-model fallback.
- Explicit latency budgets: < 8s per game, < 60s for 5-game burst.

This architecture supports the late-news rerun pattern (3-5 games in < 60s)
required for competitive DFS operations.
