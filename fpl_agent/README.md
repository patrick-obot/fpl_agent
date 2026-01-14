# FPL Agent

An autonomous Fantasy Premier League (FPL) manager that analyzes fixtures, player data, and projected points to recommend optimal transfers and captain selections.

## Features

- **Async API Client**: Fast, rate-limited interactions with the FPL API using aiohttp
- **Data Collection**: Fixture difficulty analysis, player news/injury tracking, and projected points loading
- **Smart Optimization**: Transfer and captain recommendations based on form, fixtures, xG/xA, and availability
- **Safe Execution**: Dry-run mode for testing, confirmation steps, and detailed logging
- **Comprehensive Testing**: Full test suite with pytest

## Project Structure

```
fpl_agent/
├── src/
│   ├── __init__.py          # Package exports
│   ├── config.py             # Configuration management
│   ├── fpl_client.py         # FPL API client
│   ├── data_collector.py     # Data collection and analysis
│   ├── optimizer.py          # Transfer/captain optimization
│   └── executor.py           # Safe transfer execution
├── tests/
│   ├── test_config.py
│   ├── test_fpl_client.py
│   ├── test_data_collector.py
│   ├── test_optimizer.py
│   └── test_executor.py
├── data/
│   └── projected_points.csv  # External projections data
├── logs/
│   └── agent.log             # Application logs
├── .env                      # Environment configuration
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                   # Entry point
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fpl_agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your FPL credentials
```

## Configuration

Edit the `.env` file with your settings:

```env
# FPL Account Credentials
FPL_EMAIL=your_email@example.com
FPL_PASSWORD=your_password
FPL_TEAM_ID=your_team_id

# Agent Settings
DRY_RUN=true              # Set to false for live mode
LOG_LEVEL=INFO

# Optimization Settings
MAX_TRANSFERS_PER_WEEK=2
MIN_BANK_BALANCE=0.0
```

## Usage

### Dry Run Mode (Default - Safe)

Test the agent without making any changes:

```bash
python main.py
```

### Live Mode (Preview Only)

See what would be executed without confirmation:

```bash
python main.py --live
```

### Live Mode with Execution

Actually execute the recommended transfers:

```bash
python main.py --live --confirm
```

### Command Line Options

```
Options:
  --live            Run in live mode (default is dry-run)
  --confirm         Auto-confirm execution in live mode
  --gameweeks N     Number of gameweeks to analyze (default: 5)
  --log-level LEVEL Override log level (DEBUG, INFO, WARNING, ERROR)
  --env-file PATH   Path to custom .env file
```

## Workflow

1. **Data Collection**
   - Fetches current player statistics from FPL API
   - Analyzes fixture difficulty for upcoming gameweeks
   - Collects player availability/injury news
   - Loads external projected points (if available)

2. **Optimization**
   - Evaluates current squad for underperforming players
   - Identifies transfer targets based on form, fixtures, and xG/xA
   - Ranks captain options by expected points
   - Validates squad constraints (max 3 per team, position limits)

3. **Execution**
   - Previews recommended changes
   - In dry-run mode: logs what would happen
   - In live mode with confirmation: executes transfers via API

## Projected Points

Place your projected points data in `data/projected_points.csv`:

```csv
player_id,player_name,team,position,projected_points,confidence
1,Salah,14,3,8.5,0.85
2,Haaland,13,4,9.2,0.90
...
```

The agent will incorporate these projections into its optimization algorithm.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_optimizer.py

# Run with verbose output
pytest -v
```

## Logging

Logs are written to both console and `logs/agent.log`. The log includes:
- Timestamps
- Log level
- Module name
- Function and line number (in file logs)
- Message

## Safety Features

1. **Dry Run Mode**: Enabled by default - no changes made to your team
2. **Confirmation Required**: Live mode requires explicit `--confirm` flag
3. **Validation**: Squad constraints checked before execution
4. **Rate Limiting**: Built-in rate limiter (100 requests/minute)
5. **Error Handling**: Comprehensive error handling with retries
6. **Execution History**: All actions logged and tracked

## API Endpoints Used

- `GET /bootstrap-static/` - Players, teams, gameweeks
- `GET /fixtures/` - Match fixtures
- `GET /element-summary/{id}/` - Player history
- `GET /my-team/{id}/` - Current team
- `POST /transfers/` - Execute transfers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Disclaimer

This tool is for educational and personal use only. Use at your own risk. The author is not responsible for any points lost or transfers made by this agent. Always verify recommendations before executing in live mode.

## License

MIT License - see LICENSE file for details.
