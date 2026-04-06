"""
Formats the news + stock results into a 4-block Slack message and posts via webhook.
Uses Slack Block Kit for rich formatting.
"""
import os
import requests


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _direction_emoji(pct: float) -> str:
    return "📈" if pct >= 0 else "📉"


def _ma_status(pct_from_close: float) -> str:
    """pct_from_close = (close - sma) / sma * 100  (positive = close above MA)"""
    return "✅ Above" if pct_from_close >= 0 else "❌ Below"


def _fmt_pct(v: float, sign=True) -> str:
    return f"{v:+.1f}%" if sign else f"{v:.1f}%"


def _vol_label(ratio: float) -> str:
    return f"{ratio:.1f}× avg vol"


def format_stock_blocks(stock: dict) -> list[dict]:
    """Return a list of Slack Block Kit blocks for the stock section."""
    if not stock["market_open"]:
        return [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*📈 AAPL Stock*\n_Market closed today — no data._"},
            }
        ]

    p  = stock["price"]
    ta = stock["ta"]
    d  = p["pct_change"]
    emoji = _direction_emoji(d)

    # ── Block 1 — Headline ──────────────────────────────────────────────────
    b1_lines = [
        f"*📈 AAPL — {stock['date']}*",
        f"*${p['close']:.2f}*  {emoji} ${p['change']:+.2f} ({p['pct_change']:+.2f}%)  "
        f"|  O: ${p['open']:.2f}  H: ${p['high']:.2f}  L: ${p['low']:.2f}",
        f"Vol: {ta['today_vol']:,.0f}  ({_vol_label(ta['vol_ratio'])})",
        f"🔍 *{stock['signal_verdict']}*",
    ]

    # ── Block 2 — Key Drivers ───────────────────────────────────────────────
    b2_lines = ["*📰 Key Drivers*", stock["key_drivers"]]

    # ── Block 3 — Technical Dashboard ──────────────────────────────────────
    def ma_row(label, val, pct):
        status = _ma_status(pct)
        return f"  {label:<12} ${val:>7.2f}   {pct:>+6.1f}%   {status}"

    rsi_bar = f"{ta['rsi']:.1f} — {ta['rsi_label']}"
    macd_dir = ta["macd_direction"]
    bb_pos = ta["bb_label"]

    b3_lines = [
        "*📊 Technical Dashboard*",
        "```",
        "Moving Averages",
        f"{'':2}{'':12} {'Value':>8}   {'vs Close':>8}   Status",
        ma_row("SMA(20)", ta["sma20"], ta["sma20_pct"]),
        ma_row("SMA(50)", ta["sma50"], ta["sma50_pct"]),
        ma_row("SMA(200)", ta["sma200"], ta["sma200_pct"]),
        f"  {ta['cross_status']}",
        "",
        "Momentum",
        f"  RSI(14):  {rsi_bar}",
        f"  MACD:     {ta['macd_line']:.3f}  |  Signal: {ta['signal_line']:.3f}  |  Hist: {ta['macd_hist']:+.3f}  ({macd_dir})",
        "",
        "Volatility",
        f"  Bollinger Bands:  ${ta['bb_lower']:.2f} / ${ta['bb_mid']:.2f} / ${ta['bb_upper']:.2f}  —  {bb_pos}",
        f"  ATR(14):          ${ta['atr']:.2f}/day",
        "",
        "Support & Resistance",
        f"  52-week High:  ${ta['wk52_high']:.2f}   ({ta['pct_from_52h']:.1f}% from close)",
        f"  52-week Low:   ${ta['wk52_low']:.2f}   ({ta['pct_from_52l']:+.1f}% from close)",
        f"  Resistance:    ${ta['resistance']:.2f}",
        f"  Support:       ${ta['support']:.2f}",
        "```",
    ]

    # ── Block 4 — Analyst Synthesis ─────────────────────────────────────────
    b4_lines = ["*🧠 Analyst Synthesis*", stock["synthesis"]]

    def section(lines):
        return {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}

    return [
        section(b1_lines),
        {"type": "divider"},
        section(b2_lines),
        {"type": "divider"},
        section(b3_lines),
        {"type": "divider"},
        section(b4_lines),
    ]


def format_daily_message(news: dict, stock: dict) -> dict:
    """
    Build the full Slack payload (Block Kit).
    Returns a dict ready to be JSON-serialised and POSTed to the webhook.
    """
    total_in  = news["input_tokens"] + stock.get("input_tokens", 0)
    total_out = news["output_tokens"] + stock.get("output_tokens", 0)
    cost = (total_in / 1_000_000 * 3.0) + (total_out / 1_000_000 * 15.0)

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🍎 Apple Daily Briefing — {news['date']}"},
        },
        # News section
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*📰 Apple News & Rumors*\n" + news["content"]},
        },
        {"type": "divider"},
        # Stock blocks (1-4)
        *format_stock_blocks(stock),
        {"type": "divider"},
        # Footer
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Tokens: {total_in:,} in / {total_out:,} out  |  "
                        f"Cost: ${cost:.4f}  |  "
                        f"Model: claude-sonnet-4-6"
                    ),
                }
            ],
        },
    ]

    return {"blocks": blocks}


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------

def post_to_slack(payload: dict) -> bool:
    """POST a Block Kit payload to Slack. Returns True on success."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("  SLACK_WEBHOOK_URL not set — skipping Slack post")
        return False

    resp = requests.post(webhook_url, json=payload, timeout=10)
    if resp.status_code != 200:
        print(f"  Slack post failed: {resp.status_code} {resp.text}")
        return False

    return True
