"""
EdTech Brief Module
Searches for K-12 supplemental education news (math, science, literacy)
and Reddit sentiment. Produces a formatted brief for the daily note.

Covers:
- K-12 supplemental math, science, and literacy products
- Accessibility in EdTech
- Competitive landscape (IXL, Khan Academy, etc.)
- Reddit sentiment from r/Teachers, r/edtech, r/specialeducation
"""

import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime

logger = logging.getLogger(__name__)

# Search queries to rotate through for variety
SEARCH_QUERIES = [
    "K-12 supplemental math literacy science edtech news",
    "elementary reading intervention software 2026",
    "K-12 accessibility edtech products",
    "supplemental curriculum digital learning elementary",
    "EdWeek K-12 literacy math technology",
]

REDDIT_SUBREDDITS = ["Teachers", "edtech", "specialeducation"]
REDDIT_QUERIES = [
    "supplemental math program",
    "literacy intervention software",
    "edtech accessibility",
    "reading program elementary",
]


class EdTechBrief:
    """Fetches and summarizes K-12 EdTech news for the daily note."""

    def __init__(
        self,
        llm_provider: str = "ollama",
        gemini_api_key: str = "",
        ollama_model: str = "llama3.1:8b",
    ):
        self.llm_provider = llm_provider
        self.gemini_api_key = gemini_api_key
        self.ollama_model = ollama_model

    def generate_brief(self) -> str:
        """
        Fetch news and Reddit sentiment, then synthesize into a brief.
        Returns formatted markdown string for the daily note.
        """
        logger.info("Generating EdTech brief...")

        # Gather raw content
        news_results = self._search_news()
        reddit_results = self._search_reddit()

        if not news_results and not reddit_results:
            logger.warning("No EdTech content found — skipping brief")
            return ""

        # Synthesize with LLM
        brief = self._synthesize_brief(news_results, reddit_results)

        if brief:
            logger.info("EdTech brief generated successfully")
        return brief

    def _search_news(self) -> list[dict]:
        """Search for EdTech news via DuckDuckGo HTML (no API key needed)."""
        results = []
        today = datetime.now().strftime("%Y-%m-%d")

        # Rotate query based on day of week for variety
        day_idx = datetime.now().weekday() % len(SEARCH_QUERIES)
        query = SEARCH_QUERIES[day_idx]

        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ListeningAgent/1.0",
                },
            )

            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            # Simple HTML parsing — extract result snippets
            results = self._parse_ddg_results(html, max_results=8)
            logger.debug(f"Found {len(results)} news results for: {query}")

        except Exception as e:
            logger.warning(f"News search failed: {e}")

        return results

    def _search_reddit(self) -> list[dict]:
        """Search Reddit for EdTech discussions via JSON API."""
        results = []

        # Pick a query based on day
        day_idx = datetime.now().weekday() % len(REDDIT_QUERIES)
        query = REDDIT_QUERIES[day_idx]

        for subreddit in REDDIT_SUBREDDITS:
            try:
                encoded_query = urllib.parse.quote(query)
                url = (
                    f"https://www.reddit.com/r/{subreddit}/search.json"
                    f"?q={encoded_query}&sort=new&t=week&limit=3"
                )

                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "ListeningAgent/1.0 (EdTech research)",
                    },
                )

                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    post_data = post.get("data", {})
                    if post_data.get("title"):
                        results.append({
                            "subreddit": subreddit,
                            "title": post_data["title"],
                            "score": post_data.get("score", 0),
                            "num_comments": post_data.get("num_comments", 0),
                            "selftext": (post_data.get("selftext", "") or "")[:300],
                            "url": f"https://reddit.com{post_data.get('permalink', '')}",
                        })

            except Exception as e:
                logger.debug(f"Reddit search failed for r/{subreddit}: {e}")
                continue

        logger.debug(f"Found {len(results)} Reddit posts")
        return results

    def _synthesize_brief(self, news: list[dict], reddit: list[dict]) -> str:
        """Use LLM to synthesize news and Reddit into a concise brief."""
        # Build context for the LLM
        news_text = ""
        if news:
            news_text = "NEWS RESULTS:\n"
            for i, item in enumerate(news[:8], 1):
                news_text += f"{i}. {item.get('title', 'No title')} — {item.get('snippet', '')}\n"

        reddit_text = ""
        if reddit:
            reddit_text = "\nREDDIT DISCUSSIONS:\n"
            for item in reddit:
                reddit_text += (
                    f"- r/{item['subreddit']}: \"{item['title']}\" "
                    f"(score: {item['score']}, {item['num_comments']} comments)\n"
                )
                if item.get("selftext"):
                    reddit_text += f"  Preview: {item['selftext'][:150]}...\n"

        if not news_text and not reddit_text:
            return ""

        prompt = f"""You are a competitive intelligence analyst for a K-12 EdTech company that makes 
supplemental math, science, and literacy products for elementary/middle school.

Summarize the following search results into a brief daily intelligence report.

Focus on:
1. **Market Moves** — Any competitor product launches, acquisitions, or partnerships
2. **Trends** — What educators are asking for or complaining about
3. **Accessibility** — Any news about accessibility in EdTech or VPAT/compliance topics
4. **Sentiment** — What's the general vibe from teachers on Reddit about supplemental tools?

Keep it concise — 5-8 bullet points max. Skip anything irrelevant to K-12 supplemental education.
If nothing relevant was found, say "No significant EdTech news today."

{news_text}
{reddit_text}

FORMAT as markdown bullets:
- **Category:** Brief insight (source if notable)
"""

        try:
            if self.llm_provider == "gemini" and self.gemini_api_key:
                return self._call_gemini(prompt)
            else:
                return self._call_ollama(prompt)
        except Exception as e:
            logger.error(f"EdTech brief synthesis failed: {e}")
            # Return raw results as fallback
            return self._format_raw_results(news, reddit)

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama."""
        payload = json.dumps({
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=90) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()

    def _format_raw_results(self, news: list[dict], reddit: list[dict]) -> str:
        """Fallback: format raw results without LLM synthesis."""
        lines = []

        if news:
            lines.append("**Recent EdTech News:**")
            for item in news[:5]:
                lines.append(f"- {item.get('title', 'No title')}")

        if reddit:
            lines.append("")
            lines.append("**Reddit Discussions:**")
            for item in reddit[:5]:
                lines.append(f"- r/{item['subreddit']}: {item['title']} ({item['num_comments']} comments)")

        return "\n".join(lines) if lines else "No significant EdTech news today."

    def _parse_ddg_results(self, html: str, max_results: int = 8) -> list[dict]:
        """Parse DuckDuckGo HTML results page (simple regex-based)."""
        results = []

        # DuckDuckGo HTML results have class="result__a" for links
        # and class="result__snippet" for snippets
        import re

        # Find result titles
        title_pattern = re.compile(
            r'class="result__a"[^>]*>(.*?)</a>', re.DOTALL
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:td|span|div)', re.DOTALL
        )

        titles = title_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i in range(min(len(titles), max_results)):
            title = re.sub(r'<[^>]+>', '', titles[i]).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

            if title:
                results.append({
                    "title": title,
                    "snippet": snippet,
                })

        return results
