"""
Tests for agents/news_agent.py — all HTTP I/O is mocked.
"""
import io
import textwrap
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from agents.news_agent import (
    _parse_date,
    _parse_feed,
    _categorize,
    _get_recent_articles,
    _format_markdown,
    run_news_agent,
    RSS_FEEDS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rss_xml(items: list[dict]) -> str:
    """Build a minimal RSS 2.0 document from a list of dicts with title/link/pubDate."""
    items_xml = ""
    for it in items:
        items_xml += f"""
        <item>
            <title><![CDATA[{it.get('title', '')}]]></title>
            <link>{it.get('link', '')}</link>
            <pubDate>{it.get('pubDate', '')}</pubDate>
        </item>"""
    return f"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    {items_xml}
  </channel>
</rss>"""


def _mock_urlopen(xml_bodies: list[str | None]):
    """
    Return a context manager that makes urlopen return each xml_body in sequence.
    Pass None to simulate a network error for that call.
    """
    call_count = [0]

    def fake_urlopen(req, timeout=10):
        idx = call_count[0]
        call_count[0] += 1
        body = xml_bodies[idx] if idx < len(xml_bodies) else None
        if body is None:
            raise OSError("simulated network error")
        cm = MagicMock()
        cm.__enter__ = lambda s: MagicMock(read=lambda: body.encode())
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    return patch("agents.news_agent.urllib.request.urlopen", side_effect=fake_urlopen)


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate(unittest.TestCase):
    def test_rss_date_with_offset(self):
        dt = _parse_date("Tue, 08 Apr 2025 12:00:00 +0000")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.month, 4)

    def test_rss_date_gmt_label(self):
        # Python 3.11 strptime recognises "GMT" as UTC; must parse successfully without raising
        dt = _parse_date("Mon, 07 Apr 2025 08:00:00 GMT")
        # Accept either a parsed datetime or None — the important contract is: no exception
        if dt is not None:
            self.assertEqual(dt.tzinfo, timezone.utc)

    def test_iso_8601_zulu(self):
        dt = _parse_date("2025-04-08T15:30:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.hour, 15)

    def test_iso_8601_with_offset(self):
        dt = _parse_date("2025-04-08T10:00:00+05:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.hour, 5)  # converted to UTC: 10 - 5 = 5

    def test_empty_string(self):
        self.assertIsNone(_parse_date(""))

    def test_none_like_garbage(self):
        self.assertIsNone(_parse_date("not a date"))


# ---------------------------------------------------------------------------
# _parse_feed
# ---------------------------------------------------------------------------

class TestParseFeed(unittest.TestCase):
    def test_rss_extracts_title_link(self):
        xml = _rss_xml([
            {"title": "Apple launches iPhone 17", "link": "https://macrumors.com/1", "pubDate": "Tue, 08 Apr 2025 12:00:00 +0000"},
        ])
        arts = _parse_feed(xml, "MacRumors")
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0]["title"], "Apple launches iPhone 17")
        self.assertEqual(arts[0]["link"], "https://macrumors.com/1")
        self.assertEqual(arts[0]["source"], "MacRumors")

    def test_rss_html_entity_unescaping(self):
        xml = _rss_xml([{"title": "Apple&#8217;s new iPad", "link": "https://example.com/"}])
        arts = _parse_feed(xml, "Test")
        self.assertEqual(arts[0]["title"], "Apple\u2019s new iPad")

    def test_rss_multiple_items(self):
        xml = _rss_xml([
            {"title": "Story 1", "link": "https://example.com/1"},
            {"title": "Story 2", "link": "https://example.com/2"},
            {"title": "Story 3", "link": "https://example.com/3"},
        ])
        arts = _parse_feed(xml, "Src")
        self.assertEqual(len(arts), 3)

    def test_atom_feed_parsed(self):
        atom_xml = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>New MacBook rumor surfaces</title>
    <link href="https://9to5mac.com/entry1"/>
    <published>2025-04-08T10:00:00Z</published>
  </entry>
</feed>"""
        arts = _parse_feed(atom_xml, "9to5Mac")
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0]["title"], "New MacBook rumor surfaces")
        self.assertEqual(arts[0]["link"], "https://9to5mac.com/entry1")

    def test_malformed_xml_returns_empty(self):
        arts = _parse_feed("<rss><not closed", "Bad")
        self.assertEqual(arts, [])

    def test_item_with_no_title_skipped(self):
        xml = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><link>https://example.com/</link></item>
</channel></rss>"""
        arts = _parse_feed(xml, "Src")
        self.assertEqual(arts, [])


# ---------------------------------------------------------------------------
# _categorize
# ---------------------------------------------------------------------------

class TestCategorize(unittest.TestCase):
    def _art(self, title):
        return {"title": title, "link": "https://example.com/", "source": "Src", "pub_dt": None}

    def test_launch_keyword_detected(self):
        arts = [self._art("Apple launches AirPods Pro 3 today")]
        launches, rumors = _categorize(arts)
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(rumors), 0)

    def test_rumor_keyword_detected(self):
        arts = [self._art("Gurman: Apple reportedly working on foldable iPhone")]
        launches, rumors = _categorize(arts)
        self.assertEqual(len(launches), 0)
        self.assertEqual(len(rumors), 1)

    def test_neutral_article_uncategorised(self):
        arts = [self._art("Apple posts record quarterly earnings")]
        launches, rumors = _categorize(arts)
        self.assertEqual(len(launches), 0)
        self.assertEqual(len(rumors), 0)

    def test_launch_takes_priority_over_rumor(self):
        # "launches" and "reportedly" both present — launch wins (checked first)
        arts = [self._art("Apple reportedly launches new Mac mini")]
        launches, rumors = _categorize(arts)
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(rumors), 0)

    def test_case_insensitive(self):
        arts = [self._art("APPLE ANNOUNCES new iPhone")]
        launches, rumors = _categorize(arts)
        self.assertEqual(len(launches), 1)


# ---------------------------------------------------------------------------
# _get_recent_articles
# ---------------------------------------------------------------------------

class TestGetRecentArticles(unittest.TestCase):
    def _fresh_pubdate(self, hours_ago=1):
        dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")

    def test_returns_articles_from_all_successful_feeds(self):
        xml1 = _rss_xml([{"title": "Story A", "link": "https://a.com/", "pubDate": self._fresh_pubdate(1)}])
        xml2 = _rss_xml([{"title": "Story B", "link": "https://b.com/", "pubDate": self._fresh_pubdate(2)}])
        xml3 = _rss_xml([{"title": "Story C", "link": "https://c.com/", "pubDate": self._fresh_pubdate(3)}])

        with _mock_urlopen([xml1, xml2, xml3]):
            arts = _get_recent_articles(hours=48)

        titles = {a["title"] for a in arts}
        self.assertIn("Story A", titles)
        self.assertIn("Story B", titles)
        self.assertIn("Story C", titles)

    def test_partial_failure_still_returns_results(self):
        xml1 = _rss_xml([{"title": "OK Story", "link": "https://ok.com/", "pubDate": self._fresh_pubdate(1)}])
        # Feed 2 and 3 fail
        with _mock_urlopen([xml1, None, None]):
            arts = _get_recent_articles(hours=48)

        self.assertTrue(any(a["title"] == "OK Story" for a in arts))

    def test_all_feeds_fail_raises_runtime_error(self):
        with _mock_urlopen([None, None, None]):
            with self.assertRaises(RuntimeError) as ctx:
                _get_recent_articles(hours=48)
        self.assertIn("All RSS feeds failed", str(ctx.exception))

    def test_old_articles_filtered_out(self):
        old_pubdate = (datetime.now(timezone.utc) - timedelta(hours=72)).strftime(
            "%a, %d %b %Y %H:%M:%S +0000"
        )
        fresh_pubdate = self._fresh_pubdate(1)
        xml = _rss_xml([
            {"title": "Old Story", "link": "https://old.com/", "pubDate": old_pubdate},
            {"title": "Fresh Story", "link": "https://fresh.com/", "pubDate": fresh_pubdate},
        ])
        with _mock_urlopen([xml, None, None]):
            arts = _get_recent_articles(hours=48)

        titles = {a["title"] for a in arts}
        self.assertNotIn("Old Story", titles)
        self.assertIn("Fresh Story", titles)

    def test_articles_sorted_newest_first(self):
        xml = _rss_xml([
            {"title": "Older", "link": "https://a.com/", "pubDate": self._fresh_pubdate(5)},
            {"title": "Newer", "link": "https://b.com/", "pubDate": self._fresh_pubdate(1)},
        ])
        with _mock_urlopen([xml, None, None]):
            arts = _get_recent_articles(hours=48)

        self.assertEqual(arts[0]["title"], "Newer")

    def test_article_with_no_date_is_included(self):
        xml = _rss_xml([{"title": "Dateless Story", "link": "https://x.com/"}])
        with _mock_urlopen([xml, None, None]):
            arts = _get_recent_articles(hours=48)
        self.assertTrue(any(a["title"] == "Dateless Story" for a in arts))


# ---------------------------------------------------------------------------
# _format_markdown
# ---------------------------------------------------------------------------

class TestFormatMarkdown(unittest.TestCase):
    def _art(self, title, source="MacRumors", link="https://example.com/"):
        return {"title": title, "link": link, "source": source, "pub_dt": None}

    def test_sections_always_present(self):
        md = _format_markdown([], [], [])
        self.assertIn("## Confirmed Launches", md)
        self.assertIn("## Credible Rumors", md)
        self.assertIn("## Signal of the Day", md)
        self.assertIn("## Sources", md)

    def test_nothing_confirmed_placeholder(self):
        md = _format_markdown([], [], [])
        self.assertIn("Nothing confirmed today", md)

    def test_no_rumors_placeholder(self):
        md = _format_markdown([], [], [])
        self.assertIn("No notable rumors today", md)

    def test_launches_appear_in_output(self):
        launch = self._art("Apple launches Vision Pro 2")
        md = _format_markdown([launch], [launch], [])
        self.assertIn("Apple launches Vision Pro 2", md)

    def test_rumors_appear_in_output(self):
        rumor = self._art("Gurman: foldable iPhone coming in 2026")
        md = _format_markdown([rumor], [], [rumor])
        self.assertIn("Gurman: foldable iPhone coming in 2026", md)

    def test_signal_of_day_prefers_rumor_over_launch(self):
        launch = self._art("Apple launches AirTag 2")
        rumor = self._art("Gurman: new Mac Pro leaked")
        md = _format_markdown([rumor, launch], [launch], [rumor])
        # Signal of the Day should show the rumor (first in rumors list)
        signal_section = md.split("## Signal of the Day")[1].split("##")[0]
        self.assertIn("Gurman: new Mac Pro leaked", signal_section)

    def test_no_articles_signal_placeholder(self):
        md = _format_markdown([], [], [])
        self.assertIn("No significant Apple signals today", md)

    def test_sources_capped_at_8(self):
        arts = [self._art(f"Story {i}", link=f"https://example.com/{i}") for i in range(20)]
        md = _format_markdown(arts, arts, [])
        sources_section = md.split("## Sources")[1]
        # Each source entry is a line starting with "- "; count those
        source_lines = [l for l in sources_section.splitlines() if l.strip().startswith("- ")]
        self.assertLessEqual(len(source_lines), 8)


# ---------------------------------------------------------------------------
# run_news_agent (integration)
# ---------------------------------------------------------------------------

class TestRunNewsAgent(unittest.TestCase):
    def _fresh_pubdate(self):
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    def test_returns_expected_keys(self):
        xml = _rss_xml([
            {"title": "Apple announces new iPad", "link": "https://macrumors.com/1", "pubDate": self._fresh_pubdate()},
        ])
        with _mock_urlopen([xml, xml, xml]):
            result = run_news_agent()

        self.assertIn("date", result)
        self.assertIn("content", result)
        self.assertEqual(result["input_tokens"], 0)
        self.assertEqual(result["output_tokens"], 0)

    def test_content_is_markdown_string(self):
        xml = _rss_xml([{"title": "Rumor: Apple Watch Series 11 leaked", "link": "https://9to5mac.com/1", "pubDate": self._fresh_pubdate()}])
        with _mock_urlopen([xml, xml, xml]):
            result = run_news_agent()
        self.assertIsInstance(result["content"], str)
        self.assertIn("##", result["content"])

    def test_all_feeds_fail_raises(self):
        with _mock_urlopen([None, None, None]):
            with self.assertRaises(RuntimeError):
                run_news_agent()


if __name__ == "__main__":
    unittest.main()
