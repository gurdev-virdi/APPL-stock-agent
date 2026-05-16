# AAPL Stock Agent

A lightweight daily briefing agent that tracks Apple news and AAPL stock movement, then posts a formatted summary to Slack. No LLM API calls — runs entirely on public data sources and rule-based logic.

## What it does

Every weekday at 1:30 PM PDT it:

1. **Scrapes Apple news** from MacRumors, 9to5Mac, and AppleInsider via RSS — categorises headlines into confirmed launches vs. credible rumors
2. **Fetches AAPL price data** from yfinance — live intraday price during market hours, daily close after
3. **Computes a full technical suite** — SMAs (20/50/200), RSI, MACD, Bollinger Bands, ATR, 52-week range, pivot support/resistance, volume vs. 30-day average
4. **Derives a signal verdict** — bullish/bearish/neutral based on the indicator stack
5. **Posts two Slack messages** — one for news, one for the stock dashboard

Example stock output in Slack:

```
📈 AAPL — 2025-04-08
$175.50  📈 +$3.50 (+2.03%)  |  O: $173.00  H: $176.00  L: $172.50
Vol: 75,000,000  (1.3× avg vol)
🔍 Bullish close above key moving averages on above-average volume

📰 Key Drivers
• Apple Q1 beats estimates — Bloomberg
• Golden cross active (50-day > 200-day) — long-term bullish structure intact

📊 Technical Dashboard
Moving Averages
  SMA(20)      $172.00     +2.0%   ✅ Above
  SMA(50)      $168.00     +4.5%   ✅ Above
  SMA(200)     $155.00    +13.2%   ✅ Above

Momentum
  RSI(14):  58.4 — Neutral
  MACD:     2.100  |  Signal: 1.800  |  Hist: +0.300  (bullish ▲)

Volatility
  Bollinger Bands:  $164.00 / $172.00 / $180.00  —  Mid-range
  ATR(14):          $2.80/day

Support & Resistance
  52-week High:  $200.00   (-12.2% from close)
  52-week Low:   $140.00   (+25.4% from close)
  Resistance:    $182.00
  Support:       $168.00

🧠 Analyst Synthesis
AAPL gained 2.03% to close at $175.50, trading above the 50-day SMA ($168.00).
MACD histogram (+0.300) signals building momentum; RSI at 58.4 is neutral.
```

## Project structure

```
├── agents/
│   ├── news_agent.py      # RSS scraper + launch/rumor categoriser
│   └── stock_agent.py     # yfinance fetcher + technical indicator engine
├── outputs/
│   └── slack_poster.py    # Slack Block Kit formatter + webhook poster
├── tests/                 # Full unit + integration test suite (unittest)
├── run_daily.py           # Orchestrator — runs both agents, saves reports, posts to Slack
├── requirements.txt
└── .github/workflows/
    └── apple-agent.yml    # GitHub Actions: runs tests, then the agent, weekdays at 1:30 PM PDT
```

## Running locally

**Prerequisites:** Python 3.11+

```bash
# Install dependencies
pip install -r requirements.txt pytest

# Run the test suite
python -m pytest tests/ -v

# Run the agent (set the webhook first, or it will skip the Slack post and just save reports)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... python run_daily.py
```

Without a `SLACK_WEBHOOK_URL` set, the agent still runs and saves daily report files to `reports/` — it just skips the Slack post.

## Setting up GitHub Actions

The workflow in `.github/workflows/apple-agent.yml` runs automatically on weekdays. To wire it up:

1. Fork or clone this repo into your GitHub account
2. Go to **Settings → Secrets and variables → Actions**
3. Add a secret named `SLACK_WEBHOOK_URL_APPL_NEWS` with your Slack incoming webhook URL
4. The workflow will run tests first — the agent only runs if tests pass

To trigger a run manually: **Actions → Apple Daily Briefing → Run workflow**

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | AAPL OHLCV data and news headlines |
| `requests` | Slack webhook POST |
| `python-dotenv` | Load `.env` for local development |

All data sources are public. No paid APIs or LLM calls — zero API cost per run.

## Local `.env` setup

For local development, create a `.env` file in the project root (it's gitignored):

```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```
