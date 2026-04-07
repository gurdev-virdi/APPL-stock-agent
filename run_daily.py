#!/usr/bin/env python3
"""
Daily Apple briefing agent.
Fetches Apple news + AAPL stock summary and posts to Slack.
Run manually or via GitHub Actions / launchd.
"""
import os
import sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# Load .env from the project directory (works for both local and CI runs)
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from agents.news_agent import run_news_agent
from agents.stock_agent import run_stock_agent
from outputs.slack_poster import format_news_message, format_stock_message, post_to_slack


def save_report(content: str, label: str, reports_dir: Path) -> None:
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / f"{date.today().isoformat()}-{label}.md"
    path.write_text(content)
    print(f"  Saved: {path}")


def main() -> int:
    print(f"[apple-agent] Running daily briefing for {date.today().isoformat()}")

    print("  Running news agent...")
    news = run_news_agent()
    print("  News done.")

    print("  Running stock agent...")
    stock = run_stock_agent()
    if stock["market_open"]:
        print(f"  Stock done. AAPL {stock['price']['pct_change']:+.2f}%")
    else:
        print("  Stock done. Market closed today.")

    # Save local archive (Markdown)
    reports_dir = Path(__file__).parent / "reports"
    save_report(news["content"], "news", reports_dir)
    if stock.get("market_open"):
        synthesis = stock.get("synthesis", "")
        drivers   = stock.get("key_drivers", "")
        save_report(f"{stock['signal_verdict']}\n\n{drivers}\n\n{synthesis}", "stock", reports_dir)

    # Post to Slack — two separate messages
    for label, payload in [("news", format_news_message(news)), ("stock", format_stock_message(stock))]:
        if post_to_slack(payload):
            print(f"  Posted {label} to Slack.")
        else:
            print(f"  Slack {label} post skipped (no webhook configured or post failed).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
