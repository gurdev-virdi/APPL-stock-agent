"""
Tests for run_daily.py — all agents and Slack I/O are mocked.
"""
import unittest
from datetime import date
from unittest.mock import patch, MagicMock

from run_daily import main, _error_news


# ---------------------------------------------------------------------------
# _error_news
# ---------------------------------------------------------------------------

class TestErrorNews(unittest.TestCase):
    def test_returns_required_keys(self):
        result = _error_news("network timeout")
        for k in ("date", "content", "input_tokens", "output_tokens"):
            self.assertIn(k, result)

    def test_content_contains_reason(self):
        result = _error_news("403 Forbidden")
        self.assertIn("403 Forbidden", result["content"])

    def test_zero_token_counts(self):
        result = _error_news("any reason")
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)

    def test_date_is_today(self):
        result = _error_news("test")
        self.assertEqual(result["date"], date.today().isoformat())


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

def _good_news():
    return {"date": "2025-04-08", "content": "## Confirmed Launches\n- Nothing today",
            "input_tokens": 0, "output_tokens": 0}

def _good_stock():
    return {
        "market_open": True,
        "date": "2025-04-08",
        "price": {"close": 175.0, "open": 173.0, "high": 176.0, "low": 172.0,
                  "prev_close": 172.0, "change": 3.0, "pct_change": 1.74, "volume": 70_000_000},
        "ta": {
            "current": 175.0, "sma20": 172.0, "sma50": 168.0, "sma200": 155.0,
            "sma20_pct": 1.7, "sma50_pct": 4.2, "sma200_pct": 12.9,
            "cross_status": "Golden cross active (50-day > 200-day)",
            "rsi": 57.0, "rsi_label": "Neutral",
            "macd_line": 1.8, "signal_line": 1.5, "macd_hist": 0.3, "macd_direction": "bullish ▲",
            "bb_upper": 180.0, "bb_mid": 172.0, "bb_lower": 164.0,
            "bb_pct": 68.0, "bb_label": "Mid-range", "atr": 2.5,
            "wk52_high": 200.0, "wk52_low": 140.0,
            "pct_from_52h": -12.5, "pct_from_52l": 25.0,
            "support": 168.0, "resistance": 181.0,
            "today_vol": 70_000_000, "avg_vol_30": 58_000_000, "vol_ratio": 1.21,
        },
        "headlines": [], "signal_verdict": "Bullish", "key_drivers": "• None",
        "synthesis": "AAPL gained.", "input_tokens": 0, "output_tokens": 0,
    }


def _patch_all(news_result=None, stock_result=None,
               news_exc=None, stock_exc=None, slack_ok=True):
    """Patch all external calls and return a context manager."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        news_side = news_exc if news_exc else (lambda: news_result or _good_news())
        stock_side = stock_exc if stock_exc else (lambda: stock_result or _good_stock())

        with patch("run_daily.run_news_agent",
                   side_effect=news_exc or None,
                   return_value=None if news_exc else (news_result or _good_news())), \
             patch("run_daily.run_stock_agent",
                   side_effect=stock_exc or None,
                   return_value=None if stock_exc else (stock_result or _good_stock())), \
             patch("run_daily.post_to_slack", return_value=slack_ok), \
             patch("run_daily.save_report"):
            yield

    return _ctx()


class TestMain(unittest.TestCase):
    def test_happy_path_returns_zero(self):
        with _patch_all():
            code = main()
        self.assertEqual(code, 0)

    def test_news_failure_still_posts_stock(self):
        """When RSS feeds fail, stock message must still be posted to Slack."""
        post_calls = []
        with patch("run_daily.run_news_agent", side_effect=RuntimeError("All RSS feeds failed")), \
             patch("run_daily.run_stock_agent", return_value=_good_stock()), \
             patch("run_daily.post_to_slack", side_effect=lambda p: post_calls.append(p) or True), \
             patch("run_daily.save_report"):
            code = main()

        # Exit code non-zero (CI will see failure)
        self.assertEqual(code, 1)
        # Both news and stock payloads must have been posted
        self.assertEqual(len(post_calls), 2)

    def test_news_failure_posts_error_content_to_slack(self):
        """Error notice must appear in the news Slack payload, not silent empty content."""
        posted_news_text = []

        def capture_post(payload):
            blocks = payload.get("blocks", [])
            for b in blocks:
                if b.get("type") == "section":
                    posted_news_text.append(b["text"]["text"])
            return True

        with patch("run_daily.run_news_agent", side_effect=RuntimeError("403 Forbidden")), \
             patch("run_daily.run_stock_agent", return_value=_good_stock()), \
             patch("run_daily.post_to_slack", side_effect=capture_post), \
             patch("run_daily.save_report"):
            main()

        combined = "\n".join(posted_news_text)
        self.assertIn("Feed Unavailable", combined)

    def test_stock_failure_still_posts_news(self):
        """When stock agent fails, news message must still be posted."""
        post_calls = []
        with patch("run_daily.run_news_agent", return_value=_good_news()), \
             patch("run_daily.run_stock_agent", side_effect=Exception("yfinance timeout")), \
             patch("run_daily.post_to_slack", side_effect=lambda p: post_calls.append(p) or True), \
             patch("run_daily.save_report"):
            code = main()

        self.assertEqual(code, 1)
        self.assertEqual(len(post_calls), 2)

    def test_both_agents_fail_returns_nonzero(self):
        with patch("run_daily.run_news_agent", side_effect=RuntimeError("feeds down")), \
             patch("run_daily.run_stock_agent", side_effect=Exception("yfinance down")), \
             patch("run_daily.post_to_slack", return_value=True), \
             patch("run_daily.save_report"):
            code = main()
        self.assertEqual(code, 1)

    def test_slack_not_configured_returns_zero(self):
        """Missing webhook is not a failure — just a skipped post."""
        with _patch_all(slack_ok=False):
            code = main()
        self.assertEqual(code, 0)

    def test_market_closed_still_posts(self):
        stock_closed = {"market_open": False, "date": "2025-04-08",
                        "input_tokens": 0, "output_tokens": 0}
        post_calls = []
        with patch("run_daily.run_news_agent", return_value=_good_news()), \
             patch("run_daily.run_stock_agent", return_value=stock_closed), \
             patch("run_daily.post_to_slack", side_effect=lambda p: post_calls.append(p) or True), \
             patch("run_daily.save_report"):
            code = main()

        self.assertEqual(code, 0)
        self.assertEqual(len(post_calls), 2)


if __name__ == "__main__":
    unittest.main()
