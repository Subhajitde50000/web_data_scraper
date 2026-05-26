"""
Web Data Scraper — Modular Architecture
Layers: URL Queue → HTTP Fetcher → HTML Parser → Data Extractor → Cleaner → Storage
"""

import time
import csv
import json
import sqlite3
import logging
import re
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class ScrapedItem:
    url: str
    title: str = ""
    text: str = ""
    links: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# 2. URL QUEUE  (dedup + priority)
# ─────────────────────────────────────────────

class URLQueue:
    def __init__(self, seed_urls: list[str]):
        self._queue: deque[str] = deque()
        self._seen: set[str] = set()
        for url in seed_urls:
            self.add(url)

    def add(self, url: str) -> bool:
        if url not in self._seen:
            self._seen.add(url)
            self._queue.append(url)
            return True
        return False

    def pop(self) -> Optional[str]:
        return self._queue.popleft() if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)


# ─────────────────────────────────────────────
# 3. RATE LIMITER
# ─────────────────────────────────────────────

class RateLimiter:
    def __init__(self, delay: float = 1.5, respect_robots: bool = True):
        self.delay = delay
        self.respect_robots = respect_robots
        self._last_request: float = 0.0
        self._robots_cache: dict[str, RobotFileParser] = {}

    def wait(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()

    def allowed(self, url: str, user_agent: str = "*") -> bool:
        if not self.respect_robots:
            return True
        origin = "{0.scheme}://{0.netloc}".format(urlparse(url))
        if origin not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None
            self._robots_cache[origin] = rp
        rp = self._robots_cache[origin]
        return rp.can_fetch(user_agent, url) if rp else True


# ─────────────────────────────────────────────
# 4. HTTP FETCHER  (retries + timeout)
# ─────────────────────────────────────────────

class HTTPFetcher:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; PyScraper/1.0; "
            "+https://github.com/example/scraper)"
        )
    }

    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def fetch(self, url: str) -> Optional[str]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                log.warning("Attempt %d/%d failed for %s: %s",
                            attempt, self.max_retries, url, exc)
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)   # exponential backoff
        log.error("Giving up on %s", url)
        return None


# ─────────────────────────────────────────────
# 5. HTML PARSER + DATA EXTRACTOR
# ─────────────────────────────────────────────

class HTMLParser:
    def parse(self, html: str, base_url: str) -> ScrapedItem:
        soup = BeautifulSoup(html, "lxml")

        title = (soup.title.get_text(strip=True) if soup.title else "")

        # Remove noise tags before extracting text
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        # Resolve and collect all internal links
        links = []
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            if urlparse(href).scheme in ("http", "https"):
                links.append(href)

        meta = {
            tag.get("name", tag.get("property", "")): tag.get("content", "")
            for tag in soup.find_all("meta")
            if tag.get("name") or tag.get("property")
        }

        return ScrapedItem(url=base_url, title=title,
                           text=text, links=links, meta=meta)


# ─────────────────────────────────────────────
# 6. DATA CLEANER
# ─────────────────────────────────────────────

class DataCleaner:
    def clean(self, item: ScrapedItem) -> ScrapedItem:
        item.title = item.title.strip()
        # Collapse whitespace
        item.text = re.sub(r"\s+", " ", item.text).strip()
        # Limit text to first 2000 chars (summary)
        item.text = item.text[:2000]
        # Remove duplicate links while preserving order
        seen: set[str] = set()
        unique_links = []
        for link in item.links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        item.links = unique_links
        return item


# ─────────────────────────────────────────────
# 7. STORAGE WRITERS
# ─────────────────────────────────────────────

class CSVWriter:
    def __init__(self, path: str = "output.csv"):
        self.path = path
        self._init_file()

    def _init_file(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["url", "title", "text"]).writeheader()

    def save(self, item: ScrapedItem):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "title", "text"])
            writer.writerow({"url": item.url,
                             "title": item.title,
                             "text": item.text})


class SQLiteWriter:
    def __init__(self, db_path: str = "output.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS pages "
            "(id INTEGER PRIMARY KEY, url TEXT UNIQUE, title TEXT, text TEXT, scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.commit()

    def save(self, item: ScrapedItem):
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO pages (url, title, text) VALUES (?,?,?)",
                (item.url, item.title, item.text)
            )
            self.conn.commit()
        except sqlite3.Error as exc:
            log.error("DB write failed: %s", exc)


class JSONWriter:
    def __init__(self, path: str = "output.jsonl"):
        self.path = path

    def save(self, item: ScrapedItem):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# 8. SCRAPER ORCHESTRATOR
# ─────────────────────────────────────────────

class Scraper:
    def __init__(
        self,
        seed_urls: list[str],
        max_pages: int = 10,
        delay: float = 1.5,
        follow_links: bool = False,
        same_domain_only: bool = True,
        storage: str = "csv",          # "csv" | "sqlite" | "json"
        output_path: str = "output",
    ):
        self.queue = URLQueue(seed_urls)
        self.fetcher = HTTPFetcher()
        self.limiter = RateLimiter(delay=delay)
        self.parser = HTMLParser()
        self.cleaner = DataCleaner()
        self.max_pages = max_pages
        self.follow_links = follow_links
        self.same_domain_only = same_domain_only
        self._seed_domains = {urlparse(u).netloc for u in seed_urls}

        if storage == "sqlite":
            self.writer = SQLiteWriter(f"{output_path}.db")
        elif storage == "json":
            self.writer = JSONWriter(f"{output_path}.jsonl")
        else:
            self.writer = CSVWriter(f"{output_path}.csv")

    def _in_scope(self, url: str) -> bool:
        if not self.same_domain_only:
            return True
        return urlparse(url).netloc in self._seed_domains

    def run(self):
        scraped = 0
        while self.queue and scraped < self.max_pages:
            url = self.queue.pop()
            if not url:
                break

            if not self.limiter.allowed(url):
                log.info("robots.txt blocks %s — skipping", url)
                continue

            log.info("[%d/%d] Fetching %s", scraped + 1, self.max_pages, url)
            self.limiter.wait()

            html = self.fetcher.fetch(url)
            if not html:
                continue

            item = self.parser.parse(html, url)
            item = self.cleaner.clean(item)
            self.writer.save(item)
            scraped += 1

            if self.follow_links:
                for link in item.links:
                    if self._in_scope(link):
                        self.queue.add(link)

        log.info("Done. Scraped %d pages.", scraped)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    scraper = Scraper(
        seed_urls=[
            "https://books.toscrape.com/",
        ],
        max_pages=5,
        delay=1.5,
        follow_links=True,
        same_domain_only=True,
        storage="csv",        # change to "sqlite" or "json" as needed
        output_path="output",
    )
    scraper.run()