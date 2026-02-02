# FPL Agent - Claude Code Memory

## Project Overview

Autonomous Fantasy Premier League agent that runs on Railway. Collects player data, optimizes transfers/captaincy, and executes changes via the FPL API. Runs as a scheduler daemon with no human intervention.

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
python main.py schedule               # Scheduler daemon (Railway)
python main.py price-check            # Price monitoring only
python main.py approve <plan_id>      # Approve pending plan (not used in autonomous mode)
```

## Scheduler (Railway daemon)

Runs via `railway.toml` -> `python main.py schedule`. Three scheduled actions:
- **01:30 UTC daily**: Price change monitoring
- **48h before GW deadline**: Preparation run (analysis only, no execution)
- **2h before GW deadline**: Final execution (autonomous, `confirm=True`)

Checks every 5 minutes. Tracks `last_*` dates to avoid duplicate runs.

## Execution Flow

1. Check deadline proximity
2. Collect data (FPL API + projected_points.csv)
3. Optimize (transfers, captain, chips)
4. Build ExecutionPlan with alerts
5. Execute transfers + set captain via API
6. Log audit trail, save state snapshots, send email notification

## Safety Layers

- **Audit trail**: `data/audit_trail.json` - every action logged with gameweek
- **State snapshots**: `data/states/state_*.json` - team state before/after
- **Plan files**: `data/plans/plan_*.json` - full plan details with gameweek
- **Min gain threshold**: Only recommends transfers with >4pt net gain
- **Hit penalty logic**: Only takes -4 hits for injuries or >8pt exceptional gains
- **Email notifications**: Results emailed after every execution

## Data Files

- `data/projected_points.csv` - External projections (GW25-34, manually uploaded)
- `data/data_cache.pkl` - 5-min API response cache
- `data/audit_trail.json` - Decision audit log
- `data/plans/*.json` - Execution plan history
- `data/states/*.json` - Team state snapshots
- `data/price_changes_*.csv` - Daily price reports

## Deployment

- **Platform**: Railway (auto-deploys from `master`)
- **Container**: Playwright Python base image (for auth)
- **Config**: `railway.toml`, `Dockerfile`, `.railwayignore`
- Push to `master` -> Railway auto-builds and deploys

## Environment Variables (in Railway)

FPL_EMAIL, FPL_PASSWORD, FPL_TEAM_ID, DRY_RUN (false), LOG_LEVEL,
SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFICATION_EMAIL

## Current Status (Feb 2, 2026)

- **GW24**: Completed (deadline was Jan 31 13:30 UTC)
- **Bugs fixed**: Stale price cache, captain never set, approval blocking on Railway
- **Mode**: Autonomous (`confirm=True`, no human approval needed)
- **Projections loaded**: GW25-34
- **Next action**: Agent will auto-execute at ~2h before GW25 deadline

## Known Issues

- **Duplicate logging**: Every log line appears ~20x on Railway, hitting 500 logs/sec rate limit. Likely duplicate handler attachment. Not blocking but wastes log quota.

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
