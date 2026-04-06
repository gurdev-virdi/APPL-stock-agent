import os
from datetime import date, timedelta
import anthropic
import yfinance as yf

SYSTEM_PROMPT = """You are a financial analyst summarizing why Apple (AAPL) stock moved today.

You will be given the exact closing price data. Your job is to search for today's news and explain the movement.

Format your response in Markdown with exactly these sections:
## AAPL Today
[Already filled in by the caller — do not repeat the price data]

## Why It Moved
- [3-5 bullet points: specific news events, analyst actions, macro factors, or sector moves that explain the price change]
- Cite sources inline (e.g., "per Bloomberg")

## Context
One sentence: is today's move notable vs. recent trend, or routine noise?

Be factual and concise. If the market was broadly up/down, say so briefly."""


def _fetch_aapl_data() -> dict | None:
    """Fetch AAPL price data via yfinance. Returns None if market data is unavailable for today."""
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")

    if hist.empty:
        return None

    latest_date = hist.index[-1].date()
    if latest_date != date.today():
        return None  # Market closed today (weekend/holiday)

    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else None

    prev_close = float(prev["Close"]) if prev is not None else None
    close = float(latest["Close"])
    change = close - prev_close if prev_close else None
    pct_change = (change / prev_close * 100) if prev_close else None

    return {
        "date": latest_date.isoformat(),
        "close": round(close, 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "prev_close": round(prev_close, 2) if prev_close else None,
        "change": round(change, 2) if change else None,
        "pct_change": round(pct_change, 2) if pct_change else None,
    }


def _format_price_block(data: dict) -> str:
    direction = "▲" if (data["pct_change"] or 0) >= 0 else "▼"
    return (
        f"*AAPL* closed at *${data['close']}* "
        f"{direction} {abs(data['pct_change'] or 0):.2f}% (${abs(data['change'] or 0):.2f}) "
        f"| Open: ${data['open']} | High: ${data['high']} | Low: ${data['low']} "
        f"| Volume: {data['volume']:,}"
    )


def run_stock_agent() -> dict:
    stock_data = _fetch_aapl_data()

    if stock_data is None:
        return {
            "date": date.today().isoformat(),
            "market_open": False,
            "stock_data": None,
            "content": None,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    price_summary = _format_price_block(stock_data)
    prompt = (
        f"Today is {stock_data['date']}. AAPL closing data:\n"
        f"Close: ${stock_data['close']}, Change: {stock_data['pct_change']:+.2f}%, "
        f"Volume: {stock_data['volume']:,} shares\n\n"
        "Search for today's news to explain why AAPL moved this way. "
        "Look for: earnings/guidance news, analyst upgrades/downgrades, product news, "
        "macro events (Fed, CPI), sector moves, or any Apple-specific headlines. "
        "Produce the 'Why It Moved' and 'Context' sections as specified."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=768,
        system=SYSTEM_PROMPT,
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 5,
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )

    narrative = ""
    for block in response.content:
        if block.type == "text":
            narrative = block.text
            break

    return {
        "date": stock_data["date"],
        "market_open": True,
        "stock_data": stock_data,
        "price_block": price_summary,
        "content": narrative,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
