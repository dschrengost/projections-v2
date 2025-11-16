# L2 Team Minutes Reconciliation Layer (minutes_v1)

## 1. Objective

Introduce a **per-team L2 reconciliation layer** for NBA minutes projections in `projections-v2` that:

- Takes **conditional minutes quantiles** (after all model-level calibration, including promotion priors) as input.
- Adjusts them to satisfy **team-level constraints** (≈240 minutes per team, per game).
- Changes them as little as possible in a **weighted L2 sense**.
- Keeps `p_play` separate and untouched.
- Is implemented first for **p50** (median minutes), with a clear extension path for p10/p90.
- Is fully configurable via YAML and debuggable via CLI flags and tests.

The L2 layer is a **post-processing step** on top of the trained minutes model. It does *not* alter the training process.

---

## 2. Scope & Data Contract

### 2.1 Where this runs

The L2 reconciler runs inside the minutes scoring pipeline, after:

1. Raw model prediction of conditional quantiles (`minutes_p10/p50/p90`).
2. Any existing calibration (e.g., conformal, monotonic fixes).
3. The **promotion prior** calibrator (if enabled).

Then L2 reconciles **p50** (and later p10/p90, if desired), and only afterward are unconditional minutes (`minutes_pXX_uncond = p_play * minutes_pXX`) computed.

### 2.2 Required columns for each player-row

The scoring DataFrame passed into the L2 reconciler (per date/slate) must contain at least:

- Identification:
  - `team_id`
  - `player_id`

- Role / state:
  - `pos_bucket` (canonical positional bucket; values in `{G, W, BIG}`)
  - `is_projected_starter` (1/0, from roster snapshot used at inference)
  - `p_play` (play probability; untouched by L2)

- Minutes (conditional, after promotion prior):
  - `minutes_p10`
  - `minutes_p50`
  - `minutes_p90`

- Optional but recommended for diagnostics:
  - `minutes_p10_raw`, `minutes_p50_raw`, `minutes_p90_raw` (pre-L2 versions)
  - `ramp_flag` / `minutes_cap` / `minutes_floor` (if available)
  - `depth_rank` (1 = first unit, etc.)

The L2 layer **only operates on conditional minutes**. It must *not* touch `_uncond` columns or `p_play`.

---

## 3. Which Players Are Reconciled? (Rotation Mask)

We do **not** want to enforce 240 minutes across all players in a deep roster (including clear non-rotation guys). Define an "in-rotation" mask per team.

### 3.1 Rotation candidate mask

For each player-row, build:

```python
is_rotation_candidate = (
    (p_play >= config.p_play_min_rotation)
    & (minutes_p50 >= config.min_minutes_for_rotation)
) | (is_projected_starter == 1)
```

Config defaults (overridable via YAML):

```yaml
l2_reconcile:
  p_play_min_rotation: 0.05        # minimum play prob to be considered in rotation
  min_minutes_for_rotation: 4.0    # ignore pure garbage-time projections
```

Only players with `is_rotation_candidate == True` are included as decision variables in the L2 optimization. Others:

- Keep their minutes unchanged (often 0), and
- Are **not** part of the 240-minute constraint.

This means the L2 layer reconciles **team rotation minutes**, not the entire 15-man roster.

---

## 4. Team Minutes Target

We assume ~240 minutes per team in regulation, conditional on the set of rotation players actually playing.

Configurable target:

```yaml
l2_reconcile:
  team_minutes:
    target: 240.0
    tolerance: 0.0  # phase 1: strict equality; can be relaxed later
```

For **v1**, enforce strict equality:

\`\`\`
sum(minutes_p50_adj for rotation players) = 240.0
\`\`\`

Optional extension (v2): allow a band `[target - tolerance, target + tolerance]` and treat as inequality constraints.

---

## 5. Player-Level Floors & Caps

We need lower and upper bounds for each player’s reconciled minutes.

### 5.1 Lower bounds `L_i`

Per rotation player i:

- If `is_projected_starter == 1`:

  - `L_i = min(minutes_p50, config.bounds.starter_floor)`

- Else (bench / rotation-only):

  - `L_i = 0.0`

Config example:

```yaml
l2_reconcile:
  bounds:
    starter_floor: 16.0   # do not force starters below ~16 mins
```

We cap starter floors at their current `minutes_p50` to avoid *forcing* a huge upward move purely via floors.

### 5.2 Upper bounds `U_i`

Hierarchy:

1. If a ramp or explicit minutes cap is available (`minutes_cap` or similar), use:

   ```python
   U_i = minutes_cap
   ```

2. Else, use a combination of p90 and a max extra allowance above p50:

   ```python
   U_i = min(
       minutes_p90 * config.bounds.p90_cap_multiplier,
       minutes_p50 + config.bounds.max_extra_minutes_above_p50,
       config.bounds.hard_cap
   )
   ```

Config example:

```yaml
l2_reconcile:
  bounds:
    p90_cap_multiplier: 1.10
    max_extra_minutes_above_p50: 10.0
    hard_cap: 44.0
```

This ensures:

- Nobody gets egregious 46+ projections.
- There’s enough headroom for promotion starters like McBride to climb into the high 20s / low 30s when justified.

---

## 6. Objective: Weighted L2 Adjustment (p50)

For each **team T** and the p50 quantile, we solve a small convex QP.

### 6.1 Decision variables

For a given team T with rotation players indexed by i:

- Variables: \( m_i \) = reconciled `minutes_p50` for player i.

Known:

- \( \mu_i \) = current `minutes_p50` (post-promotion calibrator).
- Weights \( w_i > 0 \), penalty weights for moving away from \( \mu_i \).

### 6.2 Optimization problem

We solve:

\`\`\`math
\min_m \sum_i w_i (m_i - \mu_i)^2
\`\`\`

Subject to:

1. **Team minutes constraint** (strict equality, v1):

   \`\`\`math
   \sum_i m_i = M_{	ext{target}} \quad (pprox 240)
   \`\`\`

2. **Per-player bounds**:

   \`\`\`math
   L_i \le m_i \le U_i \quad orall i
   \`\`\`

### 6.3 Choosing penalty weights `w_i`

We want high-confidence, high-minute players to move less; lower-confidence/bench guys should soak more adjustment.

#### v1 scheme: role-based + spread scaling

For each player i:

- Compute spread (uncertainty proxy):

  ```python
  spread_i = max(minutes_p90 - minutes_p10, epsilon)
  ```

- Base role penalty:

  ```yaml
  l2_reconcile:
    weights:
      starter_penalty: 1.0
      rotation_penalty: 0.5
      deep_penalty: 0.1
```

Use:

```python
if is_projected_starter:
    base_penalty = starter_penalty
elif is_rotation_candidate:
    base_penalty = rotation_penalty
else:
    base_penalty = deep_penalty

# Use spread to adjust penalty:
w_i = base_penalty * (1.0 / (spread_i ** 2))
```

Interpretation:

- High-penalty \( w_i \)  ⇒ large cost for changing that player’s minutes (trusted player, narrow distribution).
- Low-penalty \( w_i \)   ⇒ cheap to adjust (noisy minutes, bench-level uncertainty).

Config can disable spread scaling if needed.

---

## 7. Solver Backend

We will use **cvxpy** with a QP-capable backend (e.g. OSQP).

### 7.1 Dependency

Add to `pyproject.toml` (or uv-equivalent):

- `cvxpy`
- `osqp` (or appropriate solver dependency)

### 7.2 Generic QP wrapper

Create a helper module, e.g.:

- `src/projections/math/qp_solvers.py`

With something like:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class QPProblem:
    Q: np.ndarray         # (n, n), PSD
    c: np.ndarray         # (n,)
    A_eq: np.ndarray | None
    b_eq: np.ndarray | None
    A_ineq: np.ndarray | None
    b_ineq: np.ndarray | None
    lb: np.ndarray | None  # lower bounds
    ub: np.ndarray | None  # upper bounds

def solve_qp(problem: QPProblem) -> np.ndarray:
    """Solve a small QP using cvxpy + OSQP, return optimal x (np.ndarray)."""
    ...
```

For the per-team p50 problem:

- Let `n` = number of rotation players.
- Construct Q and c from the L2 objective:

  ```python
  # Objective: (m - mu)^T W (m - mu)
  # Expand: m^T W m - 2 mu^T W m + const
  # So Q = 2W, c = -2W mu
  Q = 2 * np.diag(w_i)           # shape (n, n)
  c = -2 * w_i * mu              # elementwise
  ```

- Equality constraint for team minutes:

  ```python
  A_eq = np.ones((1, n))
  b_eq = np.array([team_minutes_target])
  ```

- Inequality constraints can be left None for v1 (we encode bounds via lb/ub).

- Bounds:

  ```python
  lb = L  # shape (n,)
  ub = U  # shape (n,)
  ```

Then call `solve_qp` to obtain optimal `m`.

Handle solver failures gracefully:

- If solver fails, log a warning, and fall back to the original `minutes_p50` for that team.

---

## 8. L2 Module & CLI Wiring

### 8.1 New reconciliation module

Create:
- `src/projections/minutes_v1/reconcile.py`

Key functions:

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class ReconcileConfig:
    team_minutes_target: float
    team_minutes_tolerance: float
    p_play_min_rotation: float
    min_minutes_for_rotation: float
    bounds_config: BoundsConfig
    weights_config: WeightsConfig
    # add debug flags as needed

def reconcile_team_minutes_p50(df_team: pd.DataFrame, config: ReconcileConfig) -> pd.Series:
    """
    Given a DataFrame for a single team with minutes_p50, etc., compute reconciled
    p50 minutes for rotation players and return a Series aligned with df_team.index.
    """
    ...

def reconcile_minutes_p50_all(df: pd.DataFrame, config: ReconcileConfig) -> pd.DataFrame:
    """
    Group df by team_id and apply reconcile_team_minutes_p50 for each group.
    Store original p50 in minutes_p50_raw (if not already present), and overwrite
    minutes_p50 with reconciled values. Return modified df.
    """
    ...
```

### 8.2 Wiring into `score_minutes_v1`

In the `projections.cli.score_minutes_v1` (or equivalent) CLI:

- Add CLI flags:

```bash
--reconcile-team-minutes {none,p50,p50_and_tails}
--reconcile-debug
```

- Load `ReconcileConfig` from YAML (e.g. `config/minutes-l2-reconcile.yaml`), with defaults pointing to the values described above.

- After promotion prior (if enabled) and before computing `_uncond`:

```python
if args.reconcile_team_minutes in ("p50", "p50_and_tails"):
    df = reconcile_minutes_p50_all(df, reconcile_config)
```

- Ensure that:

  - `minutes_p50_raw` is preserved (e.g. copy column before reconciliation if not already created).
  - `minutes_p50` holds the final reconciled p50 after the L2 step.

Unconditional minutes should then be derived from the reconciled `minutes_pXX`:
- `minutes_pXX_uncond = p_play * minutes_pXX`

### 8.3 Debug mode

When `--reconcile-debug` is set:

- For each team, log:

  - `team_id`
  - Pre- and post-reconciliation p50 team total:
    - `pre_total_p50 = df_team["minutes_p50_raw"].sum()`
    - `post_total_p50 = df_team["minutes_p50"].sum()`
  - Top 5 players by absolute change `|minutes_p50 - minutes_p50_raw|`.
- Optionally, dump a small CSV/Parquet `reconcile_debug_team=<team_id>.parquet` into an artifacts folder for inspection.

This is particularly useful for inspecting cases like the 2025-11-14 Knicks.

---

## 9. Extension Path for Tails (p10/p90) – Phase 2

For **v1**, only p50 must be reconciled. For future work, here is a spec to extend to p10 and p90:

1. After computing reconciled `m_i^(50)` (final p50), solve similar QPs for p10 and p90 with additional monotonicity constraints:

   - For each player i:
     - `m_i^(10) <= m_i^(50) <= m_i^(90)`

2. Optionally enforce team totals for p10/p90 as well:

   - Sum of p10 minutes close to 240.
   - Sum of p90 minutes close to 240.

3. Use a lighter weight scheme for tails (larger flexibility).

**Interim workaround (acceptable for v1):**

- After p50 reconciliation, simply enforce monotonicity by clamping:

  ```python
  minutes_p10 = np.minimum(minutes_p10, minutes_p50)
  minutes_p90 = np.maximum(minutes_p90, minutes_p50)
  ```

This keeps quantiles sane without fully reconciling tail team totals.

---

## 10. Tests

Create `tests/minutes_v1/test_reconcile.py` with at least the following tests.

### 10.1 Basic sum correction

- Build a synthetic team DataFrame with e.g. 5 players:
  - `minutes_p50 = [30, 28, 26, 20, 19]` (sum 123)
  - simple role flags (`is_projected_starter` for first 5)
- Run reconciliation with `team_minutes_target = 240`.
- Assert:

  - `df["minutes_p50"].sum() == approx(240)`
  - Starters (higher role penalty weight) move less in absolute terms than lower-penalty players.

### 10.2 Respect bounds

- Construct a team where one player has `U_i = 24`, another has `L_i = 18`.
- Force the team to need a large positive adjustment.
- Assert that:

  - Those players’ reconciled minutes do not go beyond [L_i, U_i].

### 10.3 Rotation mask behavior

- Add players with `p_play = 0.01` and `minutes_p50 = 0`.
- Ensure they are not included in the decision set and remain at 0 after reconciliation, even if the solver needs extra minutes.

### 10.4 Real-ish regression test (optional)

- Add a small fixture for the 2025-11-14 Knicks slice (subset of the real data) with raw minutes and expectations like:

  - Pre-L2: team p50 total ≈ 193
  - Post-L2: team p50 total ≈ 240
  - McBride’s `minutes_p50` increases from ~15.7 toward a higher value (e.g. > 20).

- Don’t hard-code an exact value for McBride’s minutes (to avoid brittle tests), but assert that:
  - His final `minutes_p50` is strictly greater than his raw one, and
  - The total team minutes are ~240.

---

## 11. Summary

The L2 reconciliation layer is a **small, per-team QP** that:

- Takes the minutes model’s conditional p50s (after promotion prior) as input.
- Imposes a 240-minute team constraint and per-player floors/caps.
- Minimizes a weighted L2 deviation from the model’s outputs, so we only change what we have to.
- Is explicitly configurable, debuggable, and tested.

This will:
- Fix pathologies like teams summing to ~193 minutes.
- Let promotion cases (e.g., McBride) naturally take more minutes when the team needs them and their caps permit it.
- Preserve the overall structure of your learned minutes model instead of hard-coding heuristic “starters get 32” rules.
