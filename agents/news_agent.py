import os
from datetime import date
import anthropic

SYSTEM_PROMPT = """You are an Apple product analyst tracking upcoming Apple products, launches, and credible rumors.

Your job: search today's news and produce a concise daily briefing focused on:
1. New Apple products that were officially announced or launched today
2. Credible rumors about upcoming Apple products (prioritize Mark Gurman/Bloomberg, 9to5Mac, MacRumors)
3. The single most interesting signal about an unreleased Apple product

Format your response in Markdown with exactly these sections:
## Confirmed Launches
- [list items, or "Nothing confirmed today"]

## Credible Rumors
- [list items with source attribution, or "No notable rumors today"]

## Signal of the Day
One sentence on the most interesting upcoming product signal from today's news.

## Sources
- [URL] — [publication]

Be concise. Skip editorial fluff. If a rumor has no credible source, omit it."""


def run_news_agent() -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    today = date.today().isoformat()

    prompt = (
        f"Today is {today}. Search for Apple news from the last 24 hours. "
        "Focus on: (1) any new Apple product announcements or launches, "
        "(2) credible rumors about upcoming Apple hardware or software from known Apple journalists. "
        "Ignore unrelated Apple content (legal, earnings, app store policy unless tied to a product). "
        "Produce the briefing as specified."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 5,
                "allowed_domains": [
                    "macrumors.com",
                    "9to5mac.com",
                    "bloomberg.com",
                    "appleinsider.com",
                ],
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )

    content = ""
    for block in response.content:
        if block.type == "text":
            content = block.text
            break

    return {
        "date": today,
        "content": content,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
