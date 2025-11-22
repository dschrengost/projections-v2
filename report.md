# NBA Projections Pipeline Hardening Report

## 1. Current State Analysis

### Live Pipeline Map
The live pipeline is currently orchestrated via systemd and a bash script, executing a sequence of Python CLIs.

**Flow:**
1.  **Trigger:** `systemd` timers (`live-pipeline-hourly.timer`, `live-pipeline-evening.timer`, etc.) trigger `live-pipeline.service`.
2.  **Orchestrator:** `scripts/run_live_pipeline.sh`
    - Checks `schedule.parquet` to see if the current time is within the scoring window.
    - Sets environment variables (`LIVE_SEASON`, `DATA_ROOT`, etc.).
3.  **Step 1: Scrape & Ingest** (`projections.cli.live_pipeline`)
    - Scrapes injuries, odds, roster, etc.
    - Writes "bronze" (raw) and "silver" (snapshot) data to `projections-data`.
4.  **Step 2: Feature Engineering** (`projections.cli.build_minutes_live`)
    - Loads snapshots (injuries, odds, roster) and schedule.
    - Uses `projections.minutes_v1.features.MinutesFeatureBuilder` to construct features.
    - Enforces `FEATURES_MINUTES_V1_SCHEMA`.
    - Writes `features.parquet` to `data/live/features_minutes_v1/YYYY-MM-DD/run=...`.
5.  **Step 3: Scoring** (`projections.cli.score_minutes_v1`)
    - Loads the LightGBM model bundle.
    - Loads the live features generated in Step 2.
    - Generates predictions (p10, p50, p90).
    - Writes `minutes.parquet` to `artifacts/minutes_v1/daily/YYYY-MM-DD/run=...`.

### Training Pipeline Map
The training pipeline is ad-hoc and manually triggered.

**Flow:**
1.  **Entry Point:** `projections.train` (via `train.py`).
2.  **Data Loading:** Directly reads `nba_minutes.csv` (raw labels) and other raw files.
3.  **Feature Engineering:**
    - Uses `projections.features.add_rolling_features` (generic rolling windows).
    - **CRITICAL:** Does NOT use `MinutesFeatureBuilder`.
    - Defines feature logic via `config/settings.yaml` (columns to keep, rolling windows).
4.  **Modeling:** Trains XGBoost/LightGBM models.

### Coupling & Risks
1.  **Feature Logic Divergence (Train-Serve Skew):**
    - **Live:** Uses `MinutesFeatureBuilder` (complex logic: archetype deltas, injury impact, roster checks).
    - **Train:** Uses simple rolling averages defined in `train.py` and `config`.
    - **Risk:** The model is trained on "apples" (simple rolling features) but asked to predict on "oranges" (complex point-in-time features). This is the biggest risk.
2.  **Implicit Schema Dependencies:**
    - The live pipeline strictly enforces `FEATURES_MINUTES_V1_SCHEMA`.
    - The training pipeline loosely defines columns in `settings.yaml`.
    - Changes to `MinutesFeatureBuilder` are not automatically reflected in training data.
3.  **Data Leakage:**
    - Live pipeline carefully handles "as-of" timestamps.
    - Training pipeline seems to load raw minutes and calculate rolling averages without strict point-in-time correctness (unless `nba_minutes.csv` is already point-in-time, which is unclear but unlikely to be as rigorous as the live snapshots).

## 2. Target Architecture

To decouple the pipelines and ensure consistency, we should adopt a **Feature Store** pattern (implemented simply via Parquet files).

### Core Principles
1.  **Single Source of Truth for Features:** Both training and live inference MUST use `MinutesFeatureBuilder`.
2.  **Point-in-Time Correctness:** Features for training must be generated "as of" the game time, simulating the live environment.
3.  **Prediction Logging:** Save live features and predictions to create a high-fidelity training set for the future.

### Proposed Data Flow

**1. Offline Feature Backfill (The Bridge):**
- **New CLI:** `projections.cli.backfill_features`
- **Logic:** Iterates over historical dates.
- **Action:** Calls `MinutesFeatureBuilder` using historical snapshots (simulating "as-of" times).
- **Output:** Writes `gold/features_minutes_v1/season=YYYY/features.parquet`.
- **Usage:** This dataset becomes the **only** input for model training.

**2. Training Pipeline (Refactored):**
- **Input:** Reads `gold/features_minutes_v1` (created by backfill).
- **Logic:** Joins features with Labels (actual minutes).
- **Action:** Trains model.
- **Benefit:** Guarantees that training features are identical to live features.

**3. Live Pipeline (Hardened):**
- **Step 1: Scrape** (Unchanged).
- **Step 2: Feature Build** (Unchanged, but now uses the *same* code that generated the training data).
- **Step 3: Score** (Unchanged).
- **Step 4: QC & Logging** (New).
    - Validates output (e.g., sum of minutes per team ~240).
    - Logs "prediction rows" to a long-term storage for future monitoring.

## 3. Concrete Recommendations

### A. Code Changes
1.  **Refactor `train.py`:**
    - Stop loading raw CSVs and calculating rolling features on the fly.
    - Change input to read from `gold/features_minutes_v1` (Parquet).
    - Remove `projections.features.add_rolling_features` usage from `train.py` (move any necessary logic into `MinutesFeatureBuilder` if it's missing).
2.  **Create `backfill_features.py`:**
    - A script that runs `MinutesFeatureBuilder` for every game in the past seasons.
    - Crucial: It must reconstruct the state of the world (injuries, roster) as it was *before* the game.
3.  **Unify Schema:**
    - Ensure `train.py` validates input against `FEATURES_MINUTES_V1_SCHEMA`.

### B. Systemd & Deployment
1.  **Split Services:**
    - `live-scrape.service`: Runs just the scrapers. Fails fast if network is down.
    - `live-score.service`: Runs feature build + scoring. Depends on `live-scrape.service` (or runs after it).
2.  **Health Checks:**
    - Add a simple `projections.cli.check_health` script.
    - Run it after scoring.
    - Checks:
        - Total projected minutes per team is within [238, 242] (excluding OT).
        - No `null` predictions for active players.
        - "As-of" timestamps are fresh (within last hour).
3.  **Observability:**
    - Use `systemd-cat` to pipe logs to journald with proper tags (already doing this, but can be improved).
    - Add a `on-failure` unit to systemd services to email/Slack you when the pipeline fails.

### C. Immediate Next Steps
1.  **Stop Training on Ad-Hoc Features:** Do not train any new models until you have a backfilled feature set derived from `MinutesFeatureBuilder`.
2.  **Implement Backfill CLI:** Write the script to generate historical features using the exact same class used in live inference.
3.  **Verify Parity:** Run the backfill for "today" and compare it with the live features generated by the live pipeline. They should be identical.
