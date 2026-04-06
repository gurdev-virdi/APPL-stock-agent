"""
AAPL stock agent — fetches OHLCV data, computes full technical suite,
pulls top news headlines, then uses Claude to synthesise key drivers
and an analyst verdict into a 4-block Slack report.
"""
import os
from datetime import date, datetime, timezone
import pandas as pd
import numpy as np
import anthropic
import yfinance as yf

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_ohlcv() -> pd.DataFrame | None:
    """
    Fetch 1 year of daily OHLCV data for AAPL.
    Returns None if today's data is unavailable (weekend / holiday).
    """
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1y")

    if hist.empty:
        return None

    latest_date = hist.index[-1].date()
    if latest_date != date.today():
        return None   # Market closed today

    return hist


def _fetch_news(n: int = 5) -> list[str]:
    """
    Return the top-n AAPL headlines from yfinance.
    Falls back to an empty list if the API changes or is unavailable.
    """
    try:
        ticker = yf.Ticker("AAPL")
        raw = ticker.news or []
        headlines = []
        for item in raw[:n]:
            # yfinance ≥ 0.2.50 uses nested 'content' key
            content = item.get("content", item)
            title = (
                content.get("title")
                or content.get("headline")
                or item.get("title", "")
            )
            provider = (
                content.get("provider", {}).get("displayName")
                or item.get("publisher", "")
            )
            if title:
                headlines.append(f"{title} — {provider}" if provider else title)
        return headlines
    except Exception as exc:
        print(f"  [news] Could not fetch headlines: {exc}")
        return []


# ---------------------------------------------------------------------------
# Technical indicators (pure pandas, no extra deps)
# ---------------------------------------------------------------------------

def _sma(close: pd.Series, period: int) -> float:
    return float(close.rolling(period).mean().iloc[-1])


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> tuple[float, float, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])


def _bollinger(close: pd.Series, period=20, n_std=2) -> tuple[float, float, float]:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return float((sma + n_std * std).iloc[-1]), float(sma.iloc[-1]), float((sma - n_std * std).iloc[-1])


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> float:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _swing_levels(high: pd.Series, low: pd.Series, current: float, lookback=60) -> tuple[float, float]:
    """
    Find nearest pivot support (below current) and resistance (above current)
    from the last `lookback` bars using a simple 2-bar left/right comparison.
    Falls back to period high/low if no pivots are found.
    """
    h = high.iloc[-lookback:].reset_index(drop=True)
    l = low.iloc[-lookback:].reset_index(drop=True)

    pivot_highs, pivot_lows = [], []
    for i in range(2, len(h) - 2):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > h[i - 2] and h[i] > h[i + 2]:
            pivot_highs.append(float(h[i]))
        if l[i] < l[i - 1] and l[i] < l[i + 1] and l[i] < l[i - 2] and l[i] < l[i + 2]:
            pivot_lows.append(float(l[i]))

    resistances = [v for v in pivot_highs if v > current]
    supports    = [v for v in pivot_lows  if v < current]

    resistance = min(resistances) if resistances else float(high.iloc[-lookback:].max())
    support    = max(supports)    if supports    else float(low.iloc[-lookback:].min())
    return support, resistance


def _compute_technicals(hist: pd.DataFrame) -> dict:
    close  = hist["Close"]
    high   = hist["High"]
    low    = hist["Low"]
    volume = hist["Volume"]

    current = float(close.iloc[-1])

    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    sma200 = _sma(close, 200)

    # Golden / death cross
    if sma50 > sma200:
        cross_status = "Golden cross active (50-day > 200-day)"
    else:
        cross_status = "Death cross active (50-day < 200-day)"

    rsi = _rsi(close)
    if rsi > 70:
        rsi_label = "Overbought"
    elif rsi < 30:
        rsi_label = "Oversold"
    else:
        rsi_label = "Neutral"

    macd_line, signal_line, macd_hist = _macd(close)
    macd_direction = "bullish ▲" if macd_hist > 0 else "bearish ▼"

    bb_upper, bb_mid, bb_lower = _bollinger(close)
    bb_pct = (current - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) else 50
    if bb_pct > 80:
        bb_label = "Near upper band ⚠️"
    elif bb_pct < 20:
        bb_label = "Near lower band ⚠️"
    else:
        bb_label = "Mid-range"

    atr = _atr(high, low, close)

    wk52_high = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())
    wk52_low  = float(low.iloc[-252:].min())  if len(low)  >= 252 else float(low.min())

    support, resistance = _swing_levels(high, low, current)

    # Volume vs 30-day average
    avg_vol_30 = float(volume.iloc[-31:-1].mean())
    today_vol  = float(volume.iloc[-1])
    vol_ratio  = today_vol / avg_vol_30 if avg_vol_30 else 1.0

    return {
        "current": current,
        # MAs
        "sma20": sma20,  "sma50": sma50,  "sma200": sma200,
        "sma20_pct":  (current - sma20)  / sma20  * 100,
        "sma50_pct":  (current - sma50)  / sma50  * 100,
        "sma200_pct": (current - sma200) / sma200 * 100,
        "cross_status": cross_status,
        # Momentum
        "rsi": rsi,  "rsi_label": rsi_label,
        "macd_line": macd_line,  "signal_line": signal_line,
        "macd_hist": macd_hist,  "macd_direction": macd_direction,
        # Volatility
        "bb_upper": bb_upper,  "bb_mid": bb_mid,  "bb_lower": bb_lower,
        "bb_pct": bb_pct,  "bb_label": bb_label,
        "atr": atr,
        # Range
        "wk52_high": wk52_high,  "wk52_low": wk52_low,
        "pct_from_52h": (current - wk52_high) / wk52_high * 100,
        "pct_from_52l": (current - wk52_low)  / wk52_low  * 100,
        # S/R
        "support": support,  "resistance": resistance,
        # Volume
        "today_vol": today_vol,  "avg_vol_30": avg_vol_30,  "vol_ratio": vol_ratio,
    }


# ---------------------------------------------------------------------------
# Claude synthesis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a sell-side equity analyst covering Apple (AAPL).

Given today's closing data, technical snapshot, and top news headlines, produce:

1. SIGNAL_VERDICT: A single crisp line (≤12 words) summarising the technical posture.
   Examples: "Bullish close above key resistance on above-average volume"
             "Bearish reversal — RSI overbought, price rejecting upper Bollinger Band"

2. KEY_DRIVERS: Exactly 2–4 bullet points (use "•" prefix) covering:
   - Top news catalyst today
   - Sector/macro context relevant to Apple (consumer tech, iPhone demand, Services, tariffs, supply chain)
   - Any notable analyst upgrades/downgrades/price-target changes (if found)
   Keep each bullet to one sentence. Cite sources inline (e.g., "per Bloomberg").

3. SYNTHESIS: 1–2 sentences combining the news context with the technical picture.
   Be specific — mention at least one indicator by name.

Output format (use these exact section labels, nothing else):
SIGNAL_VERDICT: <text>

KEY_DRIVERS:
• <driver 1>
• <driver 2>
• <driver 3 if applicable>

SYNTHESIS:
<text>
"""


def _call_claude(ta: dict, headlines: list[str], price_info: dict) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    headlines_text = "\n".join(f"  - {h}" for h in headlines) if headlines else "  (none available)"

    prompt = f"""\
Today is {date.today().isoformat()}.

AAPL Closing Data:
  Close: ${price_info['close']:.2f}  Change: {price_info['pct_change']:+.2f}% (${price_info['change']:+.2f})
  Volume: {ta['today_vol']:,.0f} ({ta['vol_ratio']:.1f}× 30-day avg)

Technical Snapshot:
  SMA(20): ${ta['sma20']:.2f} ({ta['sma20_pct']:+.1f}% from close)
  SMA(50): ${ta['sma50']:.2f} ({ta['sma50_pct']:+.1f}% from close)
  SMA(200): ${ta['sma200']:.2f} ({ta['sma200_pct']:+.1f}% from close)
  Cross: {ta['cross_status']}
  RSI(14): {ta['rsi']:.1f} — {ta['rsi_label']}
  MACD: {ta['macd_line']:.2f} | Signal: {ta['signal_line']:.2f} | Hist: {ta['macd_hist']:+.2f} ({ta['macd_direction']})
  Bollinger Bands: ${ta['bb_lower']:.2f} / ${ta['bb_mid']:.2f} / ${ta['bb_upper']:.2f} — {ta['bb_label']}
  ATR(14): ${ta['atr']:.2f}/day
  52-week High: ${ta['wk52_high']:.2f} ({ta['pct_from_52h']:.1f}%)
  52-week Low: ${ta['wk52_low']:.2f} ({ta['pct_from_52l']:+.1f}%)
  Nearest Support: ${ta['support']:.2f} | Resistance: ${ta['resistance']:.2f}

Top AAPL Headlines (last 24h):
{headlines_text}

Search for any additional market-moving news, analyst actions, or macro events affecting AAPL today.
Then produce the SIGNAL_VERDICT, KEY_DRIVERS, and SYNTHESIS as instructed.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": prompt}],
    )

    raw = ""
    for block in response.content:
        if block.type == "text":
            raw = block.text
            break

    # Parse structured output
    signal_verdict = ""
    key_drivers = ""
    synthesis = ""

    if "SIGNAL_VERDICT:" in raw:
        parts = raw.split("SIGNAL_VERDICT:", 1)[1]
        signal_verdict = parts.split("\n")[0].strip()

    if "KEY_DRIVERS:" in raw:
        kd_raw = raw.split("KEY_DRIVERS:", 1)[1]
        if "SYNTHESIS:" in kd_raw:
            kd_raw = kd_raw.split("SYNTHESIS:")[0]
        key_drivers = kd_raw.strip()

    if "SYNTHESIS:" in raw:
        synthesis = raw.split("SYNTHESIS:", 1)[1].strip()

    return {
        "signal_verdict": signal_verdict or raw[:120],
        "key_drivers": key_drivers,
        "synthesis": synthesis,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_stock_agent() -> dict:
    hist = _fetch_ohlcv()

    if hist is None:
        return {
            "date": date.today().isoformat(),
            "market_open": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    close  = hist["Close"]
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else None
    current    = float(close.iloc[-1])
    change     = current - prev_close if prev_close else 0.0
    pct_change = change / prev_close * 100 if prev_close else 0.0

    price_info = {
        "close":      round(current, 2),
        "open":       round(float(hist["Open"].iloc[-1]), 2),
        "high":       round(float(hist["High"].iloc[-1]), 2),
        "low":        round(float(hist["Low"].iloc[-1]), 2),
        "prev_close": round(prev_close, 2) if prev_close else None,
        "change":     round(change, 2),
        "pct_change": round(pct_change, 2),
        "volume":     int(hist["Volume"].iloc[-1]),
    }

    ta       = _compute_technicals(hist)
    headlines = _fetch_news(5)
    claude   = _call_claude(ta, headlines, price_info)

    return {
        "date":         hist.index[-1].date().isoformat(),
        "market_open":  True,
        "price":        price_info,
        "ta":           ta,
        "headlines":    headlines,
        "signal_verdict": claude["signal_verdict"],
        "key_drivers":    claude["key_drivers"],
        "synthesis":      claude["synthesis"],
        "input_tokens":   claude["input_tokens"],
        "output_tokens":  claude["output_tokens"],
    }
