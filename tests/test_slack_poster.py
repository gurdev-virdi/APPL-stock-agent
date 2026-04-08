"""
Tests for outputs/slack_poster.py — requests.post is mocked.
"""
import unittest
from unittest.mock import patch, MagicMock

from outputs.slack_poster import (
    _truncate,
    _direction_emoji,
    _ma_status,
    _fmt_pct,
    _vol_label,
    _cost_footer,
    format_stock_blocks,
    format_news_message,
    format_stock_message,
    post_to_slack,
    _SLACK_SECTION_LIMIT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stock_closed():
    return {"market_open": False, "date": "2025-04-08", "input_tokens": 0, "output_tokens": 0}


def _stock_open():
    return {
        "market_open": True,
        "date": "2025-04-08",
        "price": {
            "close": 175.50, "open": 173.00, "high": 176.00, "low": 172.50,
            "prev_close": 172.00, "change": 3.50, "pct_change": 2.03, "volume": 75_000_000,
        },
        "ta": {
            "current": 175.50,
            "sma20": 172.00, "sma50": 168.00, "sma200": 155.00,
            "sma20_pct": 2.03, "sma50_pct": 4.46, "sma200_pct": 13.23,
            "cross_status": "Golden cross active (50-day > 200-day)",
            "rsi": 58.4, "rsi_label": "Neutral",
            "macd_line": 2.1, "signal_line": 1.8, "macd_hist": 0.3,
            "macd_direction": "bullish ▲",
            "bb_upper": 180.0, "bb_mid": 172.0, "bb_lower": 164.0,
            "bb_pct": 71.0, "bb_label": "Mid-range",
            "atr": 2.80,
            "wk52_high": 200.0, "wk52_low": 140.0,
            "pct_from_52h": -12.25, "pct_from_52l": 25.36,
            "support": 168.0, "resistance": 182.0,
            "today_vol": 75_000_000, "avg_vol_30": 60_000_000, "vol_ratio": 1.25,
        },
        "headlines": ["Apple Q1 beats — Bloomberg", "iPhone demand up — Reuters"],
        "signal_verdict": "Bullish close above key moving averages on above-average volume",
        "key_drivers": "• Apple Q1 beats — Bloomberg\n• iPhone demand up — Reuters",
        "synthesis": "AAPL gained 2.03% to $175.50, above 50-day SMA. MACD bullish.",
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _news():
    return {
        "date": "2025-04-08",
        "content": "## Confirmed Launches\n- Nothing confirmed today\n\n## Credible Rumors\n- Gurman: foldable iPhone in 2026",
        "input_tokens": 0,
        "output_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_truncate_short_string_unchanged(self):
        s = "hello"
        self.assertEqual(_truncate(s), s)

    def test_truncate_at_limit_unchanged(self):
        s = "x" * _SLACK_SECTION_LIMIT
        self.assertEqual(_truncate(s), s)

    def test_truncate_over_limit_adds_ellipsis(self):
        s = "x" * (_SLACK_SECTION_LIMIT + 100)
        result = _truncate(s)
        self.assertLessEqual(len(result), _SLACK_SECTION_LIMIT)
        self.assertTrue(result.endswith("…"))

    def test_direction_emoji_positive(self):
        self.assertEqual(_direction_emoji(1.0), "📈")

    def test_direction_emoji_zero(self):
        self.assertEqual(_direction_emoji(0.0), "📈")

    def test_direction_emoji_negative(self):
        self.assertEqual(_direction_emoji(-0.01), "📉")

    def test_ma_status_above(self):
        self.assertEqual(_ma_status(1.0), "✅ Above")

    def test_ma_status_zero(self):
        self.assertEqual(_ma_status(0.0), "✅ Above")

    def test_ma_status_below(self):
        self.assertEqual(_ma_status(-1.0), "❌ Below")

    def test_fmt_pct_with_sign(self):
        self.assertEqual(_fmt_pct(2.5), "+2.5%")
        self.assertEqual(_fmt_pct(-1.3), "-1.3%")

    def test_fmt_pct_without_sign(self):
        self.assertEqual(_fmt_pct(2.5, sign=False), "2.5%")

    def test_vol_label(self):
        self.assertEqual(_vol_label(1.5), "1.5× avg vol")


# ---------------------------------------------------------------------------
# _cost_footer
# ---------------------------------------------------------------------------

class TestCostFooter(unittest.TestCase):
    def test_is_context_block(self):
        footer = _cost_footer(0, 0)
        self.assertEqual(footer["type"], "context")

    def test_contains_elements(self):
        footer = _cost_footer(100, 50)
        self.assertIn("elements", footer)
        self.assertGreater(len(footer["elements"]), 0)

    def test_no_api_cost_text(self):
        footer = _cost_footer(0, 0)
        text = footer["elements"][0]["text"]
        self.assertIn("No API cost", text)


# ---------------------------------------------------------------------------
# format_stock_blocks
# ---------------------------------------------------------------------------

class TestFormatStockBlocks(unittest.TestCase):
    def test_market_closed_returns_single_section(self):
        blocks = format_stock_blocks(_stock_closed())
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "section")
        self.assertIn("closed", blocks[0]["text"]["text"].lower())

    def test_market_open_returns_multiple_blocks(self):
        blocks = format_stock_blocks(_stock_open())
        self.assertGreater(len(blocks), 3)

    def test_market_open_contains_dividers(self):
        blocks = format_stock_blocks(_stock_open())
        dividers = [b for b in blocks if b.get("type") == "divider"]
        self.assertGreaterEqual(len(dividers), 3)

    def test_signal_verdict_in_first_block(self):
        blocks = format_stock_blocks(_stock_open())
        first_text = blocks[0]["text"]["text"]
        self.assertIn("Bullish close above key moving averages", first_text)

    def test_key_drivers_in_second_block(self):
        # Second block is a section after the first divider
        blocks = format_stock_blocks(_stock_open())
        sections = [b for b in blocks if b.get("type") == "section"]
        second_section_text = sections[1]["text"]["text"]
        self.assertIn("Key Drivers", second_section_text)

    def test_technical_dashboard_has_code_block(self):
        blocks = format_stock_blocks(_stock_open())
        sections = [b for b in blocks if b.get("type") == "section"]
        dashboard_text = sections[2]["text"]["text"]
        self.assertIn("```", dashboard_text)
        self.assertIn("SMA", dashboard_text)

    def test_synthesis_in_last_section(self):
        blocks = format_stock_blocks(_stock_open())
        sections = [b for b in blocks if b.get("type") == "section"]
        last_text = sections[-1]["text"]["text"]
        self.assertIn("Analyst Synthesis", last_text)

    def test_all_section_text_within_slack_limit(self):
        blocks = format_stock_blocks(_stock_open())
        for block in blocks:
            if block.get("type") == "section":
                text = block["text"]["text"]
                self.assertLessEqual(len(text), 3000, f"Block exceeds Slack limit: {len(text)} chars")


# ---------------------------------------------------------------------------
# format_news_message
# ---------------------------------------------------------------------------

class TestFormatNewsMessage(unittest.TestCase):
    def test_returns_blocks_dict(self):
        msg = format_news_message(_news())
        self.assertIn("blocks", msg)
        self.assertIsInstance(msg["blocks"], list)

    def test_has_header_block(self):
        msg = format_news_message(_news())
        headers = [b for b in msg["blocks"] if b.get("type") == "header"]
        self.assertEqual(len(headers), 1)
        self.assertIn("Apple News", headers[0]["text"]["text"])

    def test_content_present_in_section(self):
        msg = format_news_message(_news())
        sections = [b for b in msg["blocks"] if b.get("type") == "section"]
        combined = "\n".join(s["text"]["text"] for s in sections)
        self.assertIn("Confirmed Launches", combined)

    def test_content_truncated_if_too_long(self):
        news = _news()
        news["content"] = "x" * 5000
        msg = format_news_message(news)
        sections = [b for b in msg["blocks"] if b.get("type") == "section"]
        for s in sections:
            self.assertLessEqual(len(s["text"]["text"]), 3000)


# ---------------------------------------------------------------------------
# format_stock_message
# ---------------------------------------------------------------------------

class TestFormatStockMessage(unittest.TestCase):
    def test_returns_blocks_dict(self):
        msg = format_stock_message(_stock_open())
        self.assertIn("blocks", msg)

    def test_has_header_block(self):
        msg = format_stock_message(_stock_open())
        headers = [b for b in msg["blocks"] if b.get("type") == "header"]
        self.assertEqual(len(headers), 1)
        self.assertIn("AAPL Stock", headers[0]["text"]["text"])

    def test_has_footer_context_block(self):
        msg = format_stock_message(_stock_open())
        footers = [b for b in msg["blocks"] if b.get("type") == "context"]
        self.assertEqual(len(footers), 1)


# ---------------------------------------------------------------------------
# post_to_slack
# ---------------------------------------------------------------------------

class TestPostToSlack(unittest.TestCase):
    def test_no_webhook_returns_false(self):
        with patch.dict("os.environ", {}, clear=True):
            # Ensure SLACK_WEBHOOK_URL is not set
            import os
            os.environ.pop("SLACK_WEBHOOK_URL", None)
            result = post_to_slack({"blocks": []})
        self.assertFalse(result)

    def test_successful_post_returns_true(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("outputs.slack_poster.requests.post", return_value=mock_resp), \
             patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/fake"}):
            result = post_to_slack({"blocks": []})
        self.assertTrue(result)

    def test_non_200_response_returns_false(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        with patch("outputs.slack_poster.requests.post", return_value=mock_resp), \
             patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/fake"}):
            result = post_to_slack({"blocks": []})
        self.assertFalse(result)

    def test_payload_sent_as_json(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        payload = {"blocks": [{"type": "section"}]}
        with patch("outputs.slack_poster.requests.post", return_value=mock_resp) as mock_post, \
             patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/fake"}):
            post_to_slack(payload)
        call_kwargs = mock_post.call_args
        self.assertEqual(call_kwargs.kwargs["json"], payload)


if __name__ == "__main__":
    unittest.main()
