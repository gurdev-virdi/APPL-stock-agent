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


def _error_news(reason: str) -> dict:
    """Fallback news result used when RSS feeds are unavailable."""
    today = date.today().isoformat()
    return {
        "date": today,
        "content": (
            f"## Feed Unavailable\n"
            f"Could not fetch Apple news today: {reason}\n\n"
            "Sources will be retried at the next scheduled run."
        ),
        "input_tokens": 0,
        "output_tokens": 0,
    }


def main() -> int:
    print(f"[apple-agent] Running daily briefing for {date.today().isoformat()}")
    exit_code = 0

    # Warn loudly if the Slack webhook is missing — posts will be skipped silently
    # otherwise, which makes the run look successful when nothing was delivered.
    if not os.environ.get("SLACK_WEBHOOK_URL"):
        print("  WARNING: SLACK_WEBHOOK_URL is not set — nothing will be posted to Slack.")
        print("  Add it to your GitHub repo: Settings → Secrets → Actions → SLACK_WEBHOOK_URL")

    # --- News agent (failures are isolated so the stock post still runs) -----
    print("  Running news agent...")
    try:
        news = run_news_agent()
        print("  News done.")
    except Exception as exc:
        print(f"  News agent failed: {exc}")
        news = _error_news(str(exc))
        exit_code = 1  # Mark run as failed for CI, but continue

    # --- Stock agent ---------------------------------------------------------
    print("  Running stock agent...")
    try:
        stock = run_stock_agent()
        if stock["market_open"]:
            print(f"  Stock done. AAPL {stock['price']['pct_change']:+.2f}%")
        else:
            print("  Stock done. Market closed today.")
    except Exception as exc:
        print(f"  Stock agent failed: {exc}")
        stock = {"market_open": False, "date": date.today().isoformat(),
                 "input_tokens": 0, "output_tokens": 0}
        exit_code = 1

    # --- Save local archive --------------------------------------------------
    reports_dir = Path(__file__).parent / "reports"
    save_report(news["content"], "news", reports_dir)
    if stock.get("market_open"):
        synthesis = stock.get("synthesis", "")
        drivers   = stock.get("key_drivers", "")
        save_report(f"{stock['signal_verdict']}\n\n{drivers}\n\n{synthesis}", "stock", reports_dir)

    # --- Post to Slack -------------------------------------------------------
    for label, payload in [("news", format_news_message(news)), ("stock", format_stock_message(stock))]:
        if post_to_slack(payload):
            print(f"  Posted {label} to Slack.")
        else:
            print(f"  Slack {label} post skipped (no webhook configured or post failed).")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
