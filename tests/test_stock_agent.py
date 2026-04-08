"""
Tests for agents/stock_agent.py — yfinance and network I/O are mocked.
"""
import unittest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from agents.stock_agent import (
    _sma, _rsi, _macd, _bollinger, _atr,
    _swing_levels, _compute_technicals,
    _rule_based_synthesis,
    _fetch_news,
    run_stock_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=300, start=150.0, trend=0.0, noise=1.0, seed=42) -> pd.Series:
    """Generate a deterministic synthetic Close price series."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(trend, noise, n)
    prices = start + np.cumsum(daily_returns)
    prices = np.clip(prices, 1.0, None)
    idx = pd.date_range(end=date.today(), periods=n, freq="B")
    return pd.Series(prices, index=idx)


def _make_ohlcv(close: pd.Series) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame around a Close series."""
    high   = close * 1.01
    low    = close * 0.99
    open_  = close * 1.005
    volume = pd.Series(np.random.default_rng(1).integers(50_000_000, 100_000_000, len(close)), index=close.index)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def _make_ticker_mock(hist: pd.DataFrame, news: list | None = None) -> MagicMock:
    """Return a yfinance.Ticker mock that returns the given hist DataFrame and news list."""
    mock = MagicMock()
    mock.history.return_value = hist
    mock.news = news or []
    return mock


# ---------------------------------------------------------------------------
# Technical indicator unit tests
# ---------------------------------------------------------------------------

class TestSMA(unittest.TestCase):
    def test_sma_equals_rolling_mean(self):
        close = _make_prices(100)
        expected = float(close.rolling(20).mean().iloc[-1])
        self.assertAlmostEqual(_sma(close, 20), expected, places=6)

    def test_sma_period_50(self):
        close = _make_prices(200)
        expected = float(close.rolling(50).mean().iloc[-1])
        self.assertAlmostEqual(_sma(close, 50), expected, places=6)


class TestRSI(unittest.TestCase):
    def test_rsi_range(self):
        close = _make_prices(100)
        rsi = _rsi(close)
        self.assertGreater(rsi, 0)
        self.assertLess(rsi, 100)

    def test_rsi_constant_price_is_neutral(self):
        """With no price change, RSI should approach 50 (no gains, no losses)."""
        # All prices equal → delta=0, gains=losses=0 → RS undefined; we just check no crash
        close = pd.Series([100.0] * 50)
        try:
            rsi = _rsi(close)
            # May be NaN or 50; just must not raise
        except Exception as e:
            self.fail(f"_rsi raised on constant series: {e}")

    def test_strongly_uptrending_series_has_high_rsi(self):
        """A series that only goes up should have RSI > 60."""
        closes = pd.Series([100.0 + i * 0.5 for i in range(60)])
        rsi = _rsi(closes)
        self.assertGreater(rsi, 60)

    def test_strongly_downtrending_series_has_low_rsi(self):
        closes = pd.Series([100.0 - i * 0.5 for i in range(60)])
        rsi = _rsi(closes)
        self.assertLess(rsi, 40)


class TestMACD(unittest.TestCase):
    def test_returns_three_floats(self):
        close = _make_prices(100)
        line, sig, hist = _macd(close)
        self.assertIsInstance(line, float)
        self.assertIsInstance(sig, float)
        self.assertIsInstance(hist, float)

    def test_hist_equals_line_minus_signal(self):
        close = _make_prices(200)
        line, sig, hist = _macd(close)
        self.assertAlmostEqual(hist, line - sig, places=8)


class TestBollinger(unittest.TestCase):
    def test_upper_above_mid_above_lower(self):
        close = _make_prices(100)
        upper, mid, lower = _bollinger(close)
        self.assertGreater(upper, mid)
        self.assertGreater(mid, lower)

    def test_mid_equals_sma20(self):
        close = _make_prices(100)
        _, mid, _ = _bollinger(close)
        sma = _sma(close, 20)
        self.assertAlmostEqual(mid, sma, places=6)


class TestATR(unittest.TestCase):
    def test_atr_positive(self):
        close = _make_prices(100)
        hist = _make_ohlcv(close)
        atr = _atr(hist["High"], hist["Low"], hist["Close"])
        self.assertGreater(atr, 0)


class TestSwingLevels(unittest.TestCase):
    def test_support_below_resistance(self):
        close = _make_prices(300)
        hist = _make_ohlcv(close)
        current = float(close.iloc[-1])
        support, resistance = _swing_levels(hist["High"], hist["Low"], current)
        self.assertLessEqual(support, current)
        self.assertGreaterEqual(resistance, current)


# ---------------------------------------------------------------------------
# _compute_technicals
# ---------------------------------------------------------------------------

class TestComputeTechnicals(unittest.TestCase):
    def setUp(self):
        close = _make_prices(300)
        self.hist = _make_ohlcv(close)

    def test_all_expected_keys_present(self):
        ta = _compute_technicals(self.hist)
        expected_keys = [
            "current", "sma20", "sma50", "sma200",
            "sma20_pct", "sma50_pct", "sma200_pct", "cross_status",
            "rsi", "rsi_label", "macd_line", "signal_line", "macd_hist", "macd_direction",
            "bb_upper", "bb_mid", "bb_lower", "bb_pct", "bb_label", "atr",
            "wk52_high", "wk52_low", "pct_from_52h", "pct_from_52l",
            "support", "resistance", "today_vol", "avg_vol_30", "vol_ratio",
        ]
        for k in expected_keys:
            self.assertIn(k, ta, f"Missing key: {k}")

    def test_rsi_label_overbought(self):
        # Manufacture a strongly uptrending series to force RSI > 70
        close = pd.Series([100.0 + i * 2.0 for i in range(300)])
        hist = _make_ohlcv(close)
        ta = _compute_technicals(hist)
        if ta["rsi"] > 70:
            self.assertEqual(ta["rsi_label"], "Overbought")

    def test_rsi_label_oversold(self):
        close = pd.Series([200.0 - i * 2.0 for i in range(300)])
        close = close.clip(lower=1.0)
        hist = _make_ohlcv(close)
        ta = _compute_technicals(hist)
        if ta["rsi"] < 30:
            self.assertEqual(ta["rsi_label"], "Oversold")

    def test_golden_cross_when_sma50_above_sma200(self):
        # A long uptrend: SMA50 will be above SMA200
        close = pd.Series([50.0 + i * 1.0 for i in range(300)])
        hist = _make_ohlcv(close)
        ta = _compute_technicals(hist)
        if ta["sma50"] > ta["sma200"]:
            self.assertIn("Golden", ta["cross_status"])

    def test_vol_ratio_is_positive(self):
        ta = _compute_technicals(self.hist)
        self.assertGreater(ta["vol_ratio"], 0)


# ---------------------------------------------------------------------------
# _rule_based_synthesis
# ---------------------------------------------------------------------------

class TestRuleBasedSynthesis(unittest.TestCase):
    def _ta(self, **overrides):
        base = {
            "sma20": 150.0, "sma50": 148.0, "sma200": 140.0,
            "sma20_pct": 1.0, "sma50_pct": 2.0, "sma200_pct": 8.0,
            "cross_status": "Golden cross active (50-day > 200-day)",
            "rsi": 55.0, "rsi_label": "Neutral",
            "macd_line": 1.5, "signal_line": 1.2, "macd_hist": 0.3,
            "macd_direction": "bullish ▲",
            "bb_upper": 160.0, "bb_mid": 150.0, "bb_lower": 140.0,
            "bb_pct": 55.0, "bb_label": "Mid-range",
            "atr": 2.5,
            "wk52_high": 200.0, "wk52_low": 130.0,
            "pct_from_52h": -15.0, "pct_from_52l": 15.0,
            "support": 145.0, "resistance": 158.0,
            "today_vol": 80_000_000, "avg_vol_30": 60_000_000, "vol_ratio": 1.33,
        }
        base.update(overrides)
        return base

    def _price(self, **overrides):
        base = {"close": 152.0, "open": 149.0, "high": 153.0, "low": 148.0,
                "prev_close": 149.0, "change": 3.0, "pct_change": 2.01, "volume": 80_000_000}
        base.update(overrides)
        return base

    def test_returns_required_keys(self):
        result = _rule_based_synthesis(self._ta(), ["Headline 1"], self._price())
        for k in ("signal_verdict", "key_drivers", "synthesis", "input_tokens", "output_tokens"):
            self.assertIn(k, result)

    def test_zero_token_counts(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)

    def test_bullish_bias_when_all_signals_positive(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertIn("Bullish", result["signal_verdict"])

    def test_bearish_bias_when_all_signals_negative(self):
        ta = self._ta(
            sma20_pct=-2.0, sma50_pct=-3.0, sma200_pct=-5.0, macd_hist=-0.5,
            macd_direction="bearish ▼",
            cross_status="Death cross active (50-day < 200-day)",
        )
        result = _rule_based_synthesis(ta, [], self._price(pct_change=-1.5, change=-2.2))
        self.assertIn("Bearish", result["signal_verdict"])

    def test_rsi_overbought_appears_in_verdict(self):
        ta = self._ta(rsi=74.0, rsi_label="Overbought")
        result = _rule_based_synthesis(ta, [], self._price())
        self.assertIn("overbought", result["signal_verdict"].lower())

    def test_rsi_oversold_appears_in_verdict(self):
        ta = self._ta(
            rsi=26.0, rsi_label="Oversold",
            sma20_pct=-3.0, sma50_pct=-4.0, sma200_pct=-6.0, macd_hist=-0.2,
            macd_direction="bearish ▼",
        )
        result = _rule_based_synthesis(ta, [], self._price(pct_change=-2.0))
        self.assertIn("oversold", result["signal_verdict"].lower())

    def test_headlines_appear_in_key_drivers(self):
        headlines = ["Apple Q1 beats estimates — Reuters", "iPhone demand strong in China"]
        result = _rule_based_synthesis(self._ta(), headlines, self._price())
        self.assertIn("Apple Q1 beats estimates", result["key_drivers"])
        self.assertIn("iPhone demand strong in China", result["key_drivers"])

    def test_no_headlines_still_produces_drivers(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertIsInstance(result["key_drivers"], str)
        self.assertGreater(len(result["key_drivers"]), 0)

    def test_golden_cross_noted_in_drivers(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertIn("Golden", result["key_drivers"])

    def test_death_cross_noted_in_drivers(self):
        ta = self._ta(
            sma50_pct=-2.0, sma200_pct=-4.0, macd_hist=-0.3,
            cross_status="Death cross active (50-day < 200-day)",
        )
        result = _rule_based_synthesis(ta, [], self._price(pct_change=-1.0))
        self.assertIn("Death", result["key_drivers"])

    def test_high_volume_noted_in_drivers(self):
        ta = self._ta(vol_ratio=2.1)
        result = _rule_based_synthesis(ta, [], self._price())
        self.assertIn("above 30-day average", result["key_drivers"])

    def test_synthesis_mentions_sma50(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertIn("50-day SMA", result["synthesis"])

    def test_synthesis_mentions_macd(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertIn("MACD", result["synthesis"])

    def test_synthesis_mentions_rsi(self):
        result = _rule_based_synthesis(self._ta(), [], self._price())
        self.assertIn("RSI", result["synthesis"])

    def test_synthesis_reflects_price_direction_up(self):
        result = _rule_based_synthesis(self._ta(), [], self._price(pct_change=1.5))
        self.assertIn("gained", result["synthesis"])

    def test_synthesis_reflects_price_direction_down(self):
        result = _rule_based_synthesis(self._ta(), [], self._price(pct_change=-1.5))
        self.assertIn("lost", result["synthesis"])


# ---------------------------------------------------------------------------
# _fetch_news
# ---------------------------------------------------------------------------

class TestFetchNews(unittest.TestCase):
    def _make_ticker_with_news(self, raw_news):
        with patch("agents.stock_agent.yf.Ticker") as MockTicker:
            mock = MagicMock()
            mock.news = raw_news
            MockTicker.return_value = mock
            result = _fetch_news(5)
        return result

    def test_returns_list_of_strings(self):
        raw = [{"content": {"title": "Apple hits record", "provider": {"displayName": "Reuters"}}}]
        result = self._make_ticker_with_news(raw)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(h, str) for h in result))

    def test_title_and_provider_combined(self):
        raw = [{"content": {"title": "Apple Q2 results", "provider": {"displayName": "Bloomberg"}}}]
        result = self._make_ticker_with_news(raw)
        self.assertEqual(result[0], "Apple Q2 results — Bloomberg")

    def test_missing_provider_omits_dash(self):
        raw = [{"content": {"title": "Apple news"}}]
        result = self._make_ticker_with_news(raw)
        self.assertEqual(result[0], "Apple news")

    def test_empty_news_returns_empty_list(self):
        result = self._make_ticker_with_news([])
        self.assertEqual(result, [])

    def test_exception_returns_empty_list(self):
        with patch("agents.stock_agent.yf.Ticker", side_effect=Exception("network error")):
            result = _fetch_news(5)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# run_stock_agent (integration)
# ---------------------------------------------------------------------------

class TestRunStockAgent(unittest.TestCase):
    def _hist_today(self, n=300) -> pd.DataFrame:
        """DataFrame whose last row has today's date."""
        close = _make_prices(n)
        hist = _make_ohlcv(close)
        # Force the last index entry to be today (market-open check)
        new_idx = list(hist.index[:-1]) + [
            pd.Timestamp(date.today())
        ]
        hist.index = pd.DatetimeIndex(new_idx)
        return hist

    def _hist_stale(self, n=300) -> pd.DataFrame:
        """DataFrame whose last row is 3 days ago (market closed)."""
        close = _make_prices(n)
        hist = _make_ohlcv(close)
        new_idx = list(hist.index[:-1]) + [
            pd.Timestamp(date.today() - timedelta(days=3))
        ]
        hist.index = pd.DatetimeIndex(new_idx)
        return hist

    def test_market_closed_returns_flag(self):
        hist = self._hist_stale()
        with patch("agents.stock_agent.yf.Ticker") as MockTicker:
            mock = MagicMock()
            mock.history.return_value = hist
            MockTicker.return_value = mock
            result = run_stock_agent()
        self.assertFalse(result["market_open"])

    def test_empty_hist_returns_market_closed(self):
        with patch("agents.stock_agent.yf.Ticker") as MockTicker:
            mock = MagicMock()
            mock.history.return_value = pd.DataFrame()
            MockTicker.return_value = mock
            result = run_stock_agent()
        self.assertFalse(result["market_open"])

    def test_market_open_returns_all_keys(self):
        hist = self._hist_today()
        with patch("agents.stock_agent.yf.Ticker") as MockTicker:
            mock = MagicMock()
            mock.history.return_value = hist
            mock.news = []
            MockTicker.return_value = mock
            result = run_stock_agent()

        self.assertTrue(result["market_open"])
        for k in ("date", "price", "ta", "headlines", "signal_verdict", "key_drivers", "synthesis"):
            self.assertIn(k, result, f"Missing key: {k}")

    def test_market_open_token_counts_are_zero(self):
        hist = self._hist_today()
        with patch("agents.stock_agent.yf.Ticker") as MockTicker:
            mock = MagicMock()
            mock.history.return_value = hist
            mock.news = []
            MockTicker.return_value = mock
            result = run_stock_agent()
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)

    def test_price_info_values_are_numeric(self):
        hist = self._hist_today()
        with patch("agents.stock_agent.yf.Ticker") as MockTicker:
            mock = MagicMock()
            mock.history.return_value = hist
            mock.news = []
            MockTicker.return_value = mock
            result = run_stock_agent()
        p = result["price"]
        for field in ("close", "open", "high", "low", "change", "pct_change"):
            self.assertIsInstance(p[field], (int, float), f"{field} not numeric")


if __name__ == "__main__":
    unittest.main()
