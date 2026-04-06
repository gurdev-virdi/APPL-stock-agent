import os
import requests


def post_to_slack(message: str) -> bool:
    """Post a message to Slack via webhook. Returns True on success."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set — skipping Slack post")
        return False

    response = requests.post(webhook_url, json={"text": message}, timeout=10)
    if response.status_code != 200:
        print(f"Slack post failed: {response.status_code} {response.text}")
        return False

    return True


def format_daily_message(news_result: dict, stock_result: dict) -> str:
    """Combine news and stock results into a single Slack message."""
    today = news_result["date"]
    lines = [f":apple: *Apple Daily Briefing — {today}*", ""]

    # News section
    lines.append("*📰 Apple News & Rumors*")
    lines.append(news_result["content"])
    lines.append("")

    # Stock section
    lines.append("*📈 AAPL Stock*")
    if not stock_result["market_open"]:
        lines.append("_Market closed today — no stock data._")
    else:
        lines.append(stock_result["price_block"])
        lines.append("")
        lines.append(stock_result["content"])

    # Cost footer
    total_input = news_result["input_tokens"] + stock_result.get("input_tokens", 0)
    total_output = news_result["output_tokens"] + stock_result.get("output_tokens", 0)
    cost = (total_input / 1_000_000 * 3.0) + (total_output / 1_000_000 * 15.0)
    lines.append("")
    lines.append(f"_Tokens: {total_input:,} in / {total_output:,} out | Cost: ${cost:.4f}_")

    return "\n".join(lines)
