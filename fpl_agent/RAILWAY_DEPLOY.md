# Railway Deployment Guide

## Quick Start

1. **Create Railway Account**
   - Sign up at [railway.app](https://railway.app)

2. **Create New Project**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your `fpl_agent` repository

3. **Configure Environment Variables**
   - Go to your service → "Variables" tab
   - Add the required variables (see below)

4. **Deploy**
   - Railway will automatically build and deploy
   - Cron jobs will run on schedule

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `FPL_TEAM_ID` | Your FPL team ID (from URL) | `12450138` |
| `DRY_RUN` | Set to `false` for live mode | `true` |

### Optional (for live mode)

| Variable | Description |
|----------|-------------|
| `FPL_EMAIL` | Your FPL login email |
| `FPL_PASSWORD` | Your FPL login password |

### Optional (for notifications)

| Variable | Description |
|----------|-------------|
| `NOTIFICATION_EMAIL` | Email for notifications |
| `SMTP_HOST` | SMTP server host |
| `SMTP_PORT` | SMTP server port (default: 587) |
| `SMTP_USER` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `WEBHOOK_URL` | Slack/Discord webhook URL |

### Optional (tuning)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TRANSFERS_PER_WEEK` | `2` | Maximum transfers to recommend |
| `MIN_BANK_BALANCE` | `0.0` | Minimum bank to maintain |
| `MIN_GAIN_THRESHOLD` | `4.0` | Minimum points gain for transfer |
| `MIN_CONFIDENCE_THRESHOLD` | `0.8` | Minimum confidence (0-1) |
| `HIT_PENALTY` | `4` | Points penalty for extra transfers |

## Cron Schedule

The following jobs are configured in `railway.toml`:

| Schedule | Time (UTC) | Description |
|----------|------------|-------------|
| `30 1 * * *` | Daily 1:30 AM | Price change monitoring |
| `0 11 * * 4` | Thursday 11:00 AM | 48h preparation (Saturday GW) |
| `0 9 * * 6` | Saturday 9:00 AM | Final execution (Saturday GW) |
| `0 11 * * 1` | Monday 11:00 AM | 48h preparation (Midweek GW) |
| `0 16 * * 2` | Tuesday 4:00 PM | Final execution (Midweek GW) |

## Manual Commands

You can run commands manually via Railway CLI or dashboard:

```bash
# Run agent in dry-run mode
python main.py

# Run in live mode (requires credentials)
python main.py --mode live

# Check price changes
python main.py price-check

# Run with custom deadline threshold
python main.py --deadline-hours 4
```

## Monitoring

### Logs
- View logs in Railway dashboard → "Logs" tab
- Logs are also saved to `/app/logs/` in the container

### Audit Trail
- Execution history is saved to `/app/data/audit_trail.json`
- Check recent actions via logs output

## Switching to Live Mode

1. Set `DRY_RUN=false` in Railway variables
2. Add `FPL_EMAIL` and `FPL_PASSWORD`
3. Redeploy the service

**Warning**: Live mode will make actual transfers. Start with dry-run to verify recommendations.

## Troubleshooting

### Build Fails
- Check `requirements.txt` for version compatibility
- View build logs in Railway dashboard

### Cron Not Running
- Verify cron syntax in `railway.toml`
- Check service is deployed and running
- View scheduled jobs in Railway dashboard

### API Rate Limiting
- The client has built-in rate limiting (1 req/sec)
- If issues persist, check FPL API status

## Cost Estimate

Railway pricing (as of 2024):
- **Hobby Plan**: $5/month, includes 500 execution hours
- Cron jobs only run when triggered, minimal resource usage
- Estimated cost: ~$5/month for scheduled runs
