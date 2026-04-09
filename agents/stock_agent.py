"""
AAPL stock agent — fetches OHLCV data, computes full technical suite,
pulls top news headlines, then derives key drivers and an analyst verdict
using rule-based logic (no LLM / no API cost).
"""
from datetime import date, datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_ohlcv() -> pd.DataFrame | None:
    """
    Fetch 1 year of daily OHLCV data for AAPL.
    Returns None only when yfinance returns no data at all.
    The caller is responsible for deciding whether the most-recent bar
    is fresh enough to report.
    """
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1y")
    return hist if not hist.empty else None


def _fetch_live_bar() -> dict | None:
    """
    Fetch a 1-minute bar from the last 2 trading days.
    Returns a dict with keys: price, open, high, low, volume, is_today.
    Used to get the live price during market hours, or the most recent
    tick after hours, rather than relying on the daily close bar.
    Returns None on any failure.
    """
    try:
        ticker = yf.Ticker("AAPL")
        intraday = ticker.history(period="2d", interval="1m")
        if intraday.empty:
            return None
        last = intraday.iloc[-1]
        last_ts = intraday.index[-1]
        # Convert to UTC-aware datetime for comparison
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        else:
            last_ts = last_ts.tz_convert("UTC")
        is_today = last_ts.date() == datetime.now(timezone.utc).date()
        return {
            "price":    round(float(last["Close"]), 2),
            "open":     round(float(intraday.iloc[0]["Open"]), 2),
            "high":     round(float(intraday["High"].max()), 2),
            "low":      round(float(intraday["Low"].min()), 2),
            "volume":   int(intraday["Volume"].sum()),
            "is_today": is_today,
        }
    except Exception as exc:
        print(f"  [stock] Live bar fetch failed: {exc}")
        return None


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
# Rule-based synthesis (no LLM required)
# ---------------------------------------------------------------------------

def _rule_based_synthesis(ta: dict, headlines: list[str], price_info: dict) -> dict:
    """Derive signal verdict, key drivers, and synthesis from computed technicals."""

    # ── Signal verdict ────────────────────────────────────────────────────────
    bullish = sum([
        ta["sma20_pct"]  > 0,
        ta["sma50_pct"]  > 0,
        ta["sma200_pct"] > 0,
        ta["macd_hist"]  > 0,
    ])
    bearish = 4 - bullish

    rsi_note = ""
    if ta["rsi"] > 70:
        bearish += 1
        rsi_note = f"RSI overbought at {ta['rsi']:.0f}"
    elif ta["rsi"] < 30:
        bullish += 1
        rsi_note = f"RSI oversold at {ta['rsi']:.0f}"

    bias = "Bullish" if bullish > bearish else ("Bearish" if bearish > bullish else "Neutral")
    vol_part = "on above-average volume" if ta["vol_ratio"] > 1.2 else "on normal volume"

    above_both = ta["sma50_pct"] > 0 and ta["sma200_pct"] > 0
    below_both = ta["sma50_pct"] < 0 and ta["sma200_pct"] < 0
    sma_ctx = (
        "above key moving averages" if above_both else
        "below key moving averages" if below_both else
        "at mixed moving-average signals"
    )

    if bias == "Bullish":
        signal_verdict = (
            f"Bullish — {rsi_note}, {ta['macd_direction']} MACD, {vol_part}"
            if rsi_note else
            f"Bullish close {sma_ctx} {vol_part}"
        )
    elif bias == "Bearish":
        signal_verdict = (
            f"Bearish — {rsi_note}, {ta['macd_direction']} MACD, {vol_part}"
            if rsi_note else
            f"Bearish close {sma_ctx} {vol_part}"
        )
    else:
        signal_verdict = f"Neutral — mixed signals, {ta['macd_direction']} MACD, RSI {ta['rsi']:.0f}"

    # ── Key drivers ───────────────────────────────────────────────────────────
    drivers: list[str] = []

    for h in headlines[:2]:
        drivers.append(f"• {h}")

    if ta["rsi"] > 65 or ta["rsi"] < 35:
        ext = "overextended" if ta["rsi"] > 65 else "depressed"
        drivers.append(f"• RSI(14) at {ta['rsi']:.1f} ({ta['rsi_label']}) — momentum {ext}")

    if ta["vol_ratio"] > 1.5:
        drivers.append(f"• Volume {ta['vol_ratio']:.1f}× above 30-day average — elevated activity")
    elif ta["vol_ratio"] < 0.7:
        drivers.append(f"• Volume {ta['vol_ratio']:.1f}× below 30-day average — low-conviction move")

    cross = ta["cross_status"]
    if "Golden" in cross:
        drivers.append(f"• {cross} — long-term bullish structure intact")
    elif "Death" in cross:
        drivers.append(f"• {cross} — long-term bearish trend warning")

    key_drivers = "\n".join(drivers) if drivers else "• No significant catalysts identified"

    # ── Synthesis ─────────────────────────────────────────────────────────────
    verb = "gained" if price_info["pct_change"] > 0 else "lost"
    sma50_dir = "above" if ta["sma50_pct"] > 0 else "below"
    mom_dir = "building" if ta["macd_hist"] > 0 else "waning"

    synthesis = (
        f"AAPL {verb} {abs(price_info['pct_change']):.2f}% to close at ${price_info['close']:.2f}, "
        f"trading {sma50_dir} the 50-day SMA (${ta['sma50']:.2f}). "
        f"MACD histogram ({ta['macd_hist']:+.3f}) signals {mom_dir} momentum; "
        f"RSI at {ta['rsi']:.1f} is {ta['rsi_label'].lower()}."
    )

    return {
        "signal_verdict": signal_verdict,
        "key_drivers": key_drivers,
        "synthesis": synthesis,
        "input_tokens": 0,
        "output_tokens": 0,
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

    # Reject data older than 5 calendar days (catches genuine outages, not
    # weekends/holidays where the most recent bar is a day or two back).
    latest_hist_date = hist.index[-1].date()
    if (date.today() - latest_hist_date).days > 5:
        return {
            "date": date.today().isoformat(),
            "market_open": False,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    # Try to get a live/recent price tick (1-min bar).
    # Falls back gracefully to the daily close if unavailable.
    live = _fetch_live_bar()

    close      = hist["Close"]
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else None

    if live:
        current = live["price"]
        day_open  = live["open"]
        day_high  = live["high"]
        day_low   = live["low"]
        day_vol   = live["volume"]
    else:
        current  = float(close.iloc[-1])
        day_open = round(float(hist["Open"].iloc[-1]), 2)
        day_high = round(float(hist["High"].iloc[-1]), 2)
        day_low  = round(float(hist["Low"].iloc[-1]), 2)
        day_vol  = int(hist["Volume"].iloc[-1])

    change     = current - prev_close if prev_close else 0.0
    pct_change = change / prev_close * 100 if prev_close else 0.0

    price_info = {
        "close":      round(current, 2),
        "open":       round(day_open, 2),
        "high":       round(day_high, 2),
        "low":        round(day_low, 2),
        "prev_close": round(prev_close, 2) if prev_close else None,
        "change":     round(change, 2),
        "pct_change": round(pct_change, 2),
        "volume":     day_vol,
    }

    ta        = _compute_technicals(hist)
    headlines = _fetch_news(5)
    synthesis = _rule_based_synthesis(ta, headlines, price_info)

    # Use the live bar's date if it's from today, otherwise the last daily bar date.
    report_date = (
        date.today().isoformat()
        if (live and live["is_today"])
        else latest_hist_date.isoformat()
    )

    return {
        "date":         report_date,
        "market_open":  True,
        "price":        price_info,
        "ta":           ta,
        "headlines":    headlines,
        "signal_verdict": synthesis["signal_verdict"],
        "key_drivers":    synthesis["key_drivers"],
        "synthesis":      synthesis["synthesis"],
        "input_tokens":   0,
        "output_tokens":  0,
    }
