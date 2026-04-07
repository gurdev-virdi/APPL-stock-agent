"""
Apple news agent — fetches recent Apple news from public RSS feeds and formats
a daily briefing without any LLM API calls.
"""
import html
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Feed configuration
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    ("MacRumors",    "https://feeds.macrumors.com/MacRumors-All"),
    ("9to5Mac",      "https://9to5mac.com/feed/"),
    ("AppleInsider", "https://appleinsider.com/rss/news.rss"),
]

# Keywords indicating an official launch / announcement
LAUNCH_KEYWORDS = [
    "launches", "launched", "announces", "announced", "releases", "released",
    "now available", "ships", "shipping", "introduces", "unveiled", "official",
    "available now", "goes on sale", "is here",
]

# Keywords indicating a rumor / leak
RUMOR_KEYWORDS = [
    "rumor", "leak", "leaked", "exclusive", "report:", "sources:", "expected",
    "upcoming", "gurman", "ming-chi", "supply chain", "could", "may launch",
    "might", "allegedly", "reportedly", "insider", "tipster", "predicted",
]

_ATOM_NS = "http://www.w3.org/2005/Atom"


# ---------------------------------------------------------------------------
# HTTP + XML helpers
# ---------------------------------------------------------------------------

def _fetch_url(url: str, timeout: int = 10) -> str | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; apple-agent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  [rss] Failed to fetch {url}: {exc}")
        return None


def _parse_date(date_str: str) -> datetime | None:
    """Parse RSS pubDate or Atom published/updated into an aware UTC datetime."""
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _parse_feed(xml_text: str, source: str) -> list[dict]:
    """Parse RSS 2.0 or Atom into a list of article dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"  [rss] XML parse error ({source}): {exc}")
        return []

    articles = []

    # Atom feed
    if f"{{{_ATOM_NS}}}feed" == root.tag or root.tag.endswith("}feed"):
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            title_el = entry.find(f"{{{_ATOM_NS}}}title")
            link_el  = entry.find(f"{{{_ATOM_NS}}}link")
            date_el  = entry.find(f"{{{_ATOM_NS}}}published") or entry.find(f"{{{_ATOM_NS}}}updated")
            title  = html.unescape((title_el.text or "") if title_el is not None else "").strip()
            link   = (link_el.get("href") or "") if link_el is not None else ""
            pub_dt = _parse_date(date_el.text if date_el is not None else "")
            if title:
                articles.append({"title": title, "link": link, "source": source, "pub_dt": pub_dt})
        return articles

    # RSS 2.0
    channel = root.find("channel") or root
    for item in channel.findall("item"):
        title_el = item.find("title")
        link_el  = item.find("link")
        date_el  = item.find("pubDate")
        title  = html.unescape((title_el.text or "") if title_el is not None else "").strip()
        link   = (link_el.text or "").strip() if link_el is not None else ""
        pub_dt = _parse_date(date_el.text if date_el is not None else "")
        if title:
            articles.append({"title": title, "link": link, "source": source, "pub_dt": pub_dt})

    return articles


# ---------------------------------------------------------------------------
# Article retrieval and categorisation
# ---------------------------------------------------------------------------

def _get_recent_articles(hours: int = 48) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    all_articles: list[dict] = []

    for source, url in RSS_FEEDS:
        xml_text = _fetch_url(url)
        if not xml_text:
            continue
        for art in _parse_feed(xml_text, source):
            # Include articles with no parseable date (assume recent)
            if art["pub_dt"] is None or art["pub_dt"] >= cutoff:
                all_articles.append(art)

    # Newest first
    all_articles.sort(
        key=lambda a: a["pub_dt"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return all_articles


def _categorize(articles: list[dict]) -> tuple[list[dict], list[dict]]:
    launches, rumors = [], []
    for art in articles:
        lower = art["title"].lower()
        if any(kw in lower for kw in LAUNCH_KEYWORDS):
            launches.append(art)
        elif any(kw in lower for kw in RUMOR_KEYWORDS):
            rumors.append(art)
    return launches, rumors


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _format_markdown(articles: list[dict], launches: list[dict], rumors: list[dict]) -> str:
    lines: list[str] = []

    lines.append("## Confirmed Launches")
    if launches:
        for a in launches[:5]:
            lines.append(f"- {a['title']} — [{a['source']}]({a['link']})")
    else:
        lines.append("- Nothing confirmed today")
    lines.append("")

    lines.append("## Credible Rumors")
    if rumors:
        for a in rumors[:5]:
            lines.append(f"- {a['title']} — [{a['source']}]({a['link']})")
    else:
        lines.append("- No notable rumors today")
    lines.append("")

    lines.append("## Signal of the Day")
    signal = rumors[0] if rumors else (launches[0] if launches else (articles[0] if articles else None))
    if signal:
        lines.append(f"{signal['title']} — [{signal['source']}]({signal['link']})")
    else:
        lines.append("No significant Apple signals today.")
    lines.append("")

    lines.append("## Sources")
    seen: set[str] = set()
    for a in (launches + rumors)[:8]:
        if a["link"] and a["link"] not in seen:
            lines.append(f"- [{a['link']}]({a['link']}) — {a['source']}")
            seen.add(a["link"])
    if not seen:
        for src, url in RSS_FEEDS:
            lines.append(f"- {url} — {src}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_news_agent() -> dict:
    today = date.today().isoformat()
    articles = _get_recent_articles(hours=48)
    launches, rumors = _categorize(articles)
    content = _format_markdown(articles, launches, rumors)

    return {
        "date": today,
        "content": content,
        "input_tokens": 0,
        "output_tokens": 0,
    }
