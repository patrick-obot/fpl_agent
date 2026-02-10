# FPL Agent - Claude Code Memory

## Project Overview

Autonomous Fantasy Premier League agent. Collects player data, optimizes transfers/captaincy, and executes changes via the FPL API. Runs as a scheduler daemon with no human intervention.

## Architecture

```
main.py          CLI entry point + scheduler daemon + orchestration
src/
  config.py      Config from .env (credentials, thresholds, paths)
  fpl_client.py  Async FPL API client (rate-limited, cached, retry logic)
  data_collector.py  Fixture difficulty, player news, projected points
  optimizer.py   Transfer + captain recommendations (constraint solver)
  executor.py    Safe execution: dry-run, audit trail, state snapshots, rollback
```

## Key Commands

```bash
python main.py                        # Dry-run (default)
python main.py --mode live --confirm  # Live execution
python main.py schedule               # Scheduler daemon
python main.py price-check            # Price monitoring only
python main.py approve <plan_id>      # Approve pending plan (not used in autonomous mode)
```

## Scheduler

Runs via `python main.py schedule` (started by Docker). Three scheduled actions:
- **01:30 UTC daily**: Price change monitoring
- **48h before GW deadline**: Preparation run (analysis only, no execution)
- **2h before GW deadline**: Final execution (autonomous, `confirm=True`)

Checks every 5 minutes. Tracks `last_*` dates to avoid duplicate runs.

## Execution Flow

1. Check deadline proximity
2. Collect data (FPL API + projected_points.csv)
3. Optimize (transfers, captain, chips, lineup)
4. Build ExecutionPlan with alerts
5. Execute transfers + set lineup/captain/bench order via single API call
6. Log audit trail, save state snapshots, send Telegram/email notification

## Lineup Optimization

Starting XI and bench are always optimized by expected points (`src/optimizer.py:1370-1504`):

**Starting XI Selection:**
- All players sorted by xPts per position
- Tries all valid formations (3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1)
- Picks formation with highest total xPts
- Transferred-in players are always included in XI

**Bench Order:**
- Position 12: Bench GK (always)
- Position 13: Highest xPts outfield bench player
- Position 14: Second highest xPts
- Position 15: Third highest xPts

This ensures auto-subs bring on the best available player if a starter doesn't play.

## Safety Layers

- **Audit trail**: `data/audit_trail.json` - every action logged with gameweek
- **State snapshots**: `data/states/state_*.json` - team state before/after
- **Plan files**: `data/plans/plan_*.json` - full plan details with gameweek
- **Min gain threshold**: Only recommends transfers with >4pt net gain
- **Hit penalty logic**: Only takes -4 hits for injuries or >8pt exceptional gains
- **Email notifications**: Results emailed AFTER execution (not approval requests)

## Data Files

- `data/projected_points.csv` - External projections (GW25-34, manually uploaded)
- `data/data_cache.pkl` - 5-min API response cache
- `data/audit_trail.json` - Decision audit log
- `data/plans/*.json` - Execution plan history
- `data/states/*.json` - Team state snapshots
- `data/price_changes_*.csv` - Daily price reports

## Deployment (VPS with Docker)

**Platform**: Hostinger VPS with Docker

### Initial Setup
```bash
# Clone repo
git clone git@github.com:patrick-obot/fpl_agent.git
cd fpl_agent

# Create .env from template
cp .env.example .env
nano .env  # Fill in your credentials

# Build and start
docker compose up -d --build

# Check logs
docker compose logs -f
```

### Updating
```bash
git pull && docker compose up -d --build
```

### Manual Commands
```bash
docker compose exec fpl-agent python main.py                        # Dry-run
docker compose exec fpl-agent python main.py --mode live --confirm  # Live execution
docker compose exec fpl-agent python main.py price-check            # Price check
```

## Environment Variables

See `.env.example` for full list. Required:
- `FPL_EMAIL`, `FPL_PASSWORD`, `FPL_TEAM_ID` - FPL credentials
- `DRY_RUN=false` - Enable live execution
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` - Telegram notifications (recommended)
- `SMTP_*`, `NOTIFICATION_EMAIL` - Email notifications (optional fallback)

## Railway Backup

Railway config files are kept as backup in case of fallback:
- `railway.toml`, `nixpacks.toml`, `.railwayignore`, `Procfile`, `RAILWAY_DEPLOY.md`

## Current Status (Feb 10, 2026)

- **Mode**: Fully autonomous (`confirm=True`, no human approval needed)
- **Projections loaded**: GW25-34
- **Notifications**: Telegram (primary) + Email (fallback), sent AFTER execution
- **Approval workflow**: Disabled - scheduler uses `confirm=True` at `main.py:522`
- **All critical bugs fixed**: Stale price cache, captain not set, approval blocking

### How Autonomous Execution Works

1. Scheduler checks every 5 minutes
2. At 2h before GW deadline, calls `run_agent(mode='live', confirm=True)`
3. `confirm=True` bypasses all approval prompts
4. Agent executes transfers + lineup immediately
5. Email sent with results (success/failure) - no confirmation needed from you

## Testing

```bash
pytest tests/ -v                # Run all tests
pytest tests/ -v --cov=src      # With coverage
```

## Key Technical Details

- All API calls are async (aiohttp + aiolimiter at 100 req/min)
- `get_players()` has 5-min cache; transfers use `force_refresh=True` for live prices
- `ExecutionPlan.gameweek` tracks which GW each plan/execution belongs to
- Executor stores gameweek in audit trail details and plan JSON files
- FPL auth uses Playwright browser automation (headless Chromium)

---

## Change Log

### Feb 10, 2026 - Session 3: VPS deployment + Telegram notifications

- **Migration**: Moved from Railway to Hostinger VPS with Docker
- Added `docker-compose.yml` for easy deployment
- Added `.env.example` template for environment variables
- **Telegram notifications**: Added `_send_telegram()` to `NotificationService`
  - Sends formatted messages with emojis for transfers, captain, lineup
  - Config: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` in `.env`
  - Telegram is primary, email is fallback
- Updated CLAUDE.md with VPS deployment instructions
- Kept Railway config files (`railway.toml`, etc.) as backup

### Feb 3, 2026 - Session 2: Set lineup + bench order after transfers

- **Problem**: Optimizer calculated `starting_xi` and `bench_order` but they were never sent to the FPL API. Transferred-in players landed on the bench by default.
- **Fix**: Added `set_lineup()` to `FPLClient` (`src/fpl_client.py`) that POSTs starting XI (positions 1-11), bench order (positions 12-15), captain, and vice-captain in a single API call to `my-team/{team_id}/`.
- Added `starting_xi` and `bench_order` fields to `ExecutionPlan`, `lineup_set` to `ExecutionResult`.
- New `_execute_lineup()` method in executor replaces direct `_execute_captain()` call during plan execution. Falls back to `_execute_captain()` if no lineup data present.
- Plan JSON files and audit trail now include lineup data.
- Dry run display shows planned starting XI and bench order.
- `set_captain()` kept as-is for standalone/backward-compat use.

### Feb 2, 2026 - Session 1: Bug fixes + autonomous mode + logging fix

**Commit `2027dfa`**: Fix 3 execution bugs, add gameweek tracking, enable autonomous mode

Three bugs that prevented the agent from ever successfully executing a transfer:

1. **Stale price cache** (`src/fpl_client.py:1324`): `make_transfers()` used `get_players()` which returned 5-min cached prices. FPL API rejected transfers with 400 "Purchase price changed". Fixed by using `get_players(force_refresh=True)`.

2. **Captain never set** (`src/executor.py:1297-1300`): `_execute_captain()` called `client.set_captain()` without `confirm=True`. The method returned early with a fake success without making the API call. Fixed by adding `confirm=True`.

3. **Scheduler blocked on approval** (`main.py:522`): Final execution used `require_approval=True`, which waited for a CLI `approve` command impossible on Railway. Fixed by switching to `confirm=True`.

Also added:
- `gameweek` field to `ExecutionPlan` dataclass (set from team state before execution)
- Gameweek included in plan JSON files (`data/plans/*.json`) and audit trail entries
- `railway.toml` comment updated to "(autonomous execution)"

**Commit `2483104`**: Fix duplicate logging + add CLAUDE.md

- **Root cause**: `setup_logging()` in `src/config.py` added `StreamHandler`/`FileHandler` to the `fpl_agent` logger every time `Config` was instantiated. Since `logging.getLogger()` returns a singleton, handlers accumulated across scheduler cycles (~20 handlers = ~20x duplicate lines), hitting Railway's 500 logs/sec rate limit.
- **Fix**: Added `logger.handlers.clear()` at top of `setup_logging()` before adding new handlers. Now idempotent regardless of how many times it's called.

### Bugs from GW22-24 (Jan 17 - Jan 31)

All 3 execution attempts on Jan 17 (GW22) failed due to the stale price cache bug. Agent has been idle since, running only price checks and preparation runs. The approval timeout bug prevented GW24 execution on Jan 31 even though the pipeline ran successfully up to the execution step (logs showed "Approval requested. Deadline: 2026-01-31 13:00").
