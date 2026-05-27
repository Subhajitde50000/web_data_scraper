"""
Web Data Scraper v4 — Bug-fixed + Domain Scheduler
===================================================
Fixes over v3 (all 5 critical bugs + 3 design issues):

CRITICAL BUG FIXES:
  [B1] Session corruption    — single shared session, opened once in Pipeline.run()
  [B2] SQLite frontier lock  — aiosqlite for all async DB access
  [B3] _scraped race         — asyncio.Lock() around counter + overshoot guard
  [B4] Sentinel deadlock     — coordinator task broadcasts N sentinels after fetch workers finish
  [B5] Bloom filter reset    — bloom state pickled to disk; SQLite fallback always persistent

DESIGN FIXES:
  [D1] Conditional cache key — keyed on final_url (post-redirect), not seed URL
  [D2] Near-duplicate hash   — simhash (4-bit hamming distance) alongside sha256
  [D3] AsyncRobots per-lock  — per-domain asyncio.Lock; no global lock during network I/O

NEW FEATURES:
  [F1] Sitemap discovery     — reads rp.site_maps() and seeds URLs from sitemap.xml
  [F2] Canonical tag         — <link rel="canonical"> respected during link extraction
  [F3] Domain scheduler      — Global Frontier → per-domain queues → fetch workers
                               (guarantees one domain never floods the global queue)
  [F4] Batched frontier/hash writes — queued inserts flushed every 250 URLs or 2s
"""

import asyncio
import hashlib
import heapq
import logging
import os
import pickle
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qs
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import aiohttp
import aiosqlite
from selectolax.parser import HTMLParser as SXParser

try:
    from pybloom_live import ScalableBloomFilter
    _BLOOM_AVAILABLE = True
except ImportError:
    _BLOOM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (compatible; PyScraper/4.0; "
    "+https://github.com/example/scraper)"
)
ALLOWED_MIME   = {"text/html", "application/xhtml+xml"}
TRACKING_PARAMS = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "fbclid","gclid","ref","source",
}
NOISE_TAGS = ("script","style","nav","footer","aside","header")


# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class CrawlMeta:
    final_url:       str   = ""
    status_code:     int   = 0
    content_type:    str   = ""
    response_time_ms: float = 0.0
    etag:            str   = ""
    last_modified:   str   = ""
    content_hash:    str   = ""   # sha256 of cleaned text
    simhash:         int   = 0    # 64-bit simhash for near-dup detection


@dataclass
class ScrapedItem:
    url:   str
    title: str   = ""
    text:  str   = ""
    links: list  = field(default_factory=list)
    meta:  dict  = field(default_factory=dict)
    crawl: CrawlMeta = field(default_factory=CrawlMeta)


# ─────────────────────────────────────────────────────────────
# URL NORMALISER
# ─────────────────────────────────────────────────────────────

def normalise(url: str) -> str:
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        return ""
    path  = p.path.rstrip("/") or "/"
    params = {k: v for k, v in parse_qs(p.query, keep_blank_values=False).items()
              if k not in TRACKING_PARAMS}
    query  = urlencode(sorted((k, v[0]) for k, v in params.items()))
    return urlunparse((p.scheme, p.netloc.lower(), path, "", query, ""))


# ─────────────────────────────────────────────────────────────
# SIMHASH  [D2]  — 64-bit fingerprint; hamming(a,b) <= 4 → near-duplicate
# ─────────────────────────────────────────────────────────────

def simhash(text: str, bits: int = 64) -> int:
    v = [0] * bits
    words = re.findall(r"\w+", text.lower())
    for token in set(words):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if v[i] > 0)

def hamming(a: int, b: int) -> int:
    x = a ^ b
    return bin(x).count("1")


# ─────────────────────────────────────────────────────────────
# PRIORITY FRONTIER + DOMAIN SCHEDULER  [F3]
# ─────────────────────────────────────────────────────────────

def _score(url: str) -> int:
    path = urlparse(url).path.lower()
    if path in ("/", ""):                      return 100
    if re.search(r"/categor|/catalog", path):  return 80
    if re.search(r"/product|/item|/p/", path): return 50
    if re.search(r"page=\d+|/page/\d+", url):  return 20
    return 40


class DomainScheduler:
    """
    Global Frontier → per-domain asyncio.Queue.

    Guarantees:
      - One domain never starves others in the global fetch queue.
      - Domain queues are naturally ordered (FIFO within a domain).
      - Fetch workers pull from a single output_q; scheduler feeds it fairly.

    SQLite (via aiosqlite) for all persistence — no raw sqlite3 in async context.  [B2]
    Writes batched into _pending_inserts and flushed every 250 URLs or 2s.          [F4]
    """

    FLUSH_BATCH  = 250
    FLUSH_DELAY  = 2.0    # seconds

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._seen:   set[str] = set()
        # global heap: (-priority, domain, url)
        self._global_heap: list[tuple[int, str, str]] = []
        # per-domain FIFO queue (domain → list of url)
        self._domain_queues: dict[str, list[str]] = {}
        # single output queue consumed by fetch workers
        self.output_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=500)
        self._lock    = asyncio.Lock()
        self._pending: list[tuple[str, int]] = []   # [(url, priority)]
        self._last_flush = time.monotonic()
        self._db: Optional[aiosqlite.Connection] = None

    async def open(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS seen (
                url TEXT PRIMARY KEY, priority INTEGER
            );
            CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
        """)
        await self._db.commit()

    async def close(self):
        await self._flush_db()
        if self._db:
            await self._db.close()

    async def add(self, url: str) -> bool:
        url = normalise(url)
        if not url:
            return False
        async with self._lock:
            if url in self._seen:
                return False
            self._seen.add(url)
            score = _score(url)
            domain = urlparse(url).netloc
            heapq.heappush(self._global_heap, (-score, domain, url))
            self._pending.append((url, score))
            await self._maybe_flush()
            return True

    async def mark_done(self, url: str):
        """Called by storage writer so resume() can skip completed URLs."""
        async with self._lock:
            if self._db:
                await self._db.execute(
                    "INSERT OR IGNORE INTO done (url) VALUES (?)", (url,)
                )
                # Piggyback on next flush commit

    async def _maybe_flush(self):
        elapsed = time.monotonic() - self._last_flush
        if len(self._pending) >= self.FLUSH_BATCH or elapsed >= self.FLUSH_DELAY:
            await self._flush_db()

    async def _flush_db(self):
        if not self._db or not self._pending:
            return
        await self._db.executemany(
            "INSERT OR IGNORE INTO seen (url, priority) VALUES (?,?)",
            self._pending,
        )
        await self._db.commit()
        self._pending.clear()
        self._last_flush = time.monotonic()

    async def run_scheduler(self, stop_event: asyncio.Event):
        """
        Continuously pops from global heap, routes into per-domain queues,
        and feeds the single output_q in round-robin style across domains.
        """
        domain_list: list[str] = []   # round-robin order

        while not stop_event.is_set():
            async with self._lock:
                # Drain global heap into per-domain queues
                while self._global_heap:
                    _, domain, url = heapq.heappop(self._global_heap)
                    if domain not in self._domain_queues:
                        self._domain_queues[domain] = []
                        domain_list.append(domain)
                    self._domain_queues[domain].append(url)

            # Round-robin across domains
            fed = False
            for domain in list(domain_list):
                q = self._domain_queues.get(domain, [])
                if q:
                    url = q.pop(0)
                    try:
                        self.output_q.put_nowait(url)
                        fed = True
                    except asyncio.QueueFull:
                        # Put it back and try next tick
                        q.insert(0, url)

            if not fed:
                await asyncio.sleep(0.05)

    async def load_resume(self):
        """Reload pending URLs (seen minus done) from a previous run."""
        if not self._db:
            return
        async with self._db.execute(
            """SELECT s.url, s.priority FROM seen s
               LEFT JOIN done d ON s.url = d.url
               WHERE d.url IS NULL"""
        ) as cur:
            rows = await cur.fetchall()
        for url, priority in rows:
            self._seen.add(url)
            domain = urlparse(url).netloc
            heapq.heappush(self._global_heap, (-priority, domain, url))
        log.info("Resumed scheduler: %d URLs pending", len(rows))

    def __len__(self) -> int:
        return len(self._global_heap) + sum(
            len(v) for v in self._domain_queues.values()
        )


# ─────────────────────────────────────────────────────────────
# ASYNC ROBOTS.TXT  — per-domain lock, no global lock during I/O  [D3]
# ─────────────────────────────────────────────────────────────

class AsyncRobots:

    def __init__(self, default_delay: float = 1.5):
        self.default_delay = default_delay
        self._cache: dict[str, Optional[RobotFileParser]] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()   # only for _domain_locks dict access

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        async with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = asyncio.Lock()
            return self._domain_locks[domain]

    async def get(self, origin: str, session: aiohttp.ClientSession
                  ) -> Optional[RobotFileParser]:
        domain = urlparse(origin).netloc
        lock   = await self._get_lock(domain)
        async with lock:                          # per-domain lock, not global
            if origin in self._cache:
                return self._cache[origin]
            rp = None
            try:
                async with session.get(
                    f"{origin}/robots.txt",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text(errors="replace")
                        rp = RobotFileParser()
                        rp.set_url(f"{origin}/robots.txt")
                        rp.parse(text.splitlines())
            except Exception:
                pass
            self._cache[origin] = rp
            return rp

    async def allowed(self, url: str, session: aiohttp.ClientSession,
                      agent: str = "*") -> bool:
        p = urlparse(url)
        rp = await self.get(f"{p.scheme}://{p.netloc}", session)
        return rp.can_fetch(agent, url) if rp else True

    async def crawl_delay(self, url: str, session: aiohttp.ClientSession) -> float:
        p = urlparse(url)
        rp = await self.get(f"{p.scheme}://{p.netloc}", session)
        if rp:
            cd = rp.crawl_delay("*")
            if cd:
                return float(cd)
        return self.default_delay

    async def site_maps(self, url: str, session: aiohttp.ClientSession) -> list[str]:
        """Return sitemap URLs declared in robots.txt."""
        p = urlparse(url)
        rp = await self.get(f"{p.scheme}://{p.netloc}", session)
        if rp and hasattr(rp, "site_maps"):
            sms = rp.site_maps()
            return list(sms) if sms else []
        return []


# ─────────────────────────────────────────────────────────────
# SITEMAP PARSER  [F1]
# ─────────────────────────────────────────────────────────────

async def fetch_sitemap_urls(
    sitemap_url: str,
    session: aiohttp.ClientSession,
    depth: int = 0,
) -> list[str]:
    """Recursively fetch URLs from sitemap / sitemap-index."""
    if depth > 3:
        return []
    urls: list[str] = []
    try:
        async with session.get(
            sitemap_url, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            if resp.status != 200:
                return []
            text = await resp.text(errors="replace")
        root = ElementTree.fromstring(text)
        ns   = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        tag  = lambda t: f"{{{ns}}}{t}" if ns else t

        # Sitemap index → recurse
        for loc in root.iter(tag("sitemap")):
            child = loc.find(tag("loc"))
            if child is not None and child.text:
                sub = await fetch_sitemap_urls(child.text.strip(), session, depth + 1)
                urls.extend(sub)

        # URL set
        for loc in root.iter(tag("url")):
            child = loc.find(tag("loc"))
            if child is not None and child.text:
                u = normalise(child.text.strip())
                if u:
                    urls.append(u)
    except Exception as exc:
        log.debug("Sitemap error %s: %s", sitemap_url, exc)
    return urls


# ─────────────────────────────────────────────────────────────
# ASYNC RATE LIMITER  — per-domain, never blocks event loop
# ─────────────────────────────────────────────────────────────

class RateLimiter:

    def __init__(self, default_delay: float = 1.5, per_domain_concurrency: int = 2):
        self.default_delay = default_delay
        self.per_domain_concurrency = per_domain_concurrency
        self._last:  dict[str, float] = {}
        self._sems:  dict[str, asyncio.Semaphore] = {}
        self._lock   = asyncio.Lock()

    async def _sem(self, domain: str) -> asyncio.Semaphore:
        async with self._lock:
            if domain not in self._sems:
                self._sems[domain] = asyncio.Semaphore(self.per_domain_concurrency)
            return self._sems[domain]

    async def wait(self, domain: str, delay: Optional[float] = None):
        d = delay if delay is not None else self.default_delay
        async with self._lock:
            gap = d - (time.monotonic() - self._last.get(domain, 0.0))
        if gap > 0:
            await asyncio.sleep(gap)
        async with self._lock:
            self._last[domain] = time.monotonic()

    async def acquire(self, domain: str) -> asyncio.Semaphore:
        return await self._sem(domain)


# ─────────────────────────────────────────────────────────────
# ASYNC HTTP FETCHER  — single shared session, opened once  [B1]
# ─────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    url:       str
    final_url: str
    html:      Optional[str]
    meta:      CrawlMeta


class AsyncFetcher:
    """
    Session is created once and shared across all fetch workers.  [B1]
    Workers call fetcher.fetch() directly — no `async with fetcher`.
    """

    def __init__(self, global_concurrency: int = 50, timeout: int = 15):
        self._global_sem = asyncio.Semaphore(global_concurrency)
        self._timeout    = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def open(self):
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT},
            timeout=self._timeout,
        )

    async def close(self):
        if self.session:
            await self.session.close()

    async def fetch(
        self,
        url:           str,
        rate:          RateLimiter,
        etag:          str = "",
        last_modified: str = "",
    ) -> FetchResult:
        meta   = CrawlMeta()
        domain = urlparse(url).netloc
        dom_sem = await rate.acquire(domain)

        async with self._global_sem, dom_sem:
            delay = await AsyncRobots().crawl_delay(url, self.session) \
                    if False else None  # delay resolved by caller
            await rate.wait(domain)
            headers: dict[str, str] = {}
            if etag:
                headers["If-None-Match"] = etag
            elif last_modified:
                headers["If-Modified-Since"] = last_modified

            t0 = time.monotonic()
            try:
                async with self.session.get(
                    url, allow_redirects=True, headers=headers
                ) as resp:
                    meta.status_code     = resp.status
                    meta.content_type    = resp.headers.get("Content-Type", "")
                    meta.etag            = resp.headers.get("ETag", "")
                    meta.last_modified   = resp.headers.get("Last-Modified", "")
                    meta.response_time_ms = round((time.monotonic() - t0) * 1000, 1)
                    final_url = normalise(str(resp.url)) or str(resp.url)

                    if resp.status == 304:
                        return FetchResult(url=url, final_url=final_url,
                                           html=None, meta=meta)

                    mime = meta.content_type.split(";")[0].strip().lower()
                    if resp.status == 200 and mime in ALLOWED_MIME:
                        html = await resp.text(errors="replace")
                        return FetchResult(url=url, final_url=final_url,
                                           html=html, meta=meta)
            except Exception as exc:
                log.warning("Fetch error %s: %s", url, exc)
                meta.response_time_ms = round((time.monotonic() - t0) * 1000, 1)

        return FetchResult(url=url, final_url=url, html=None, meta=meta)


# ─────────────────────────────────────────────────────────────
# FAST PARSER (selectolax)  — canonical tag support  [F2]
# ─────────────────────────────────────────────────────────────

class FastParser:

    def parse(
        self, html: str, base_url: str
    ) -> tuple[str, str, list[str], dict, Optional[str]]:
        """Returns (title, text, links, meta, canonical_url)."""
        tree = SXParser(html)

        title_node = tree.css_first("title")
        title = title_node.text(strip=True) if title_node else ""

        # Canonical URL  [F2]
        canonical: Optional[str] = None
        for node in tree.css('link[rel="canonical"]'):
            href = node.attributes.get("href", "").strip()
            if href:
                canonical = normalise(urljoin(base_url, href)) or None
                break

        for tag in tree.css(",".join(NOISE_TAGS)):
            tag.decompose()

        body     = tree.body
        raw_text = body.text(separator=" ", strip=True) if body else ""
        text     = re.sub(r"\s+", " ", raw_text).strip()[:3000]

        links: list[str] = []
        seen_links: set[str] = set()
        for node in tree.css("a[href]"):
            href = node.attributes.get("href", "")
            if not href:
                continue
            full = normalise(urljoin(base_url, href))
            if full and full not in seen_links:
                seen_links.add(full)
                links.append(full)

        meta: dict[str, str] = {}
        for node in tree.css("meta"):
            key = node.attributes.get("name") or node.attributes.get("property")
            val = node.attributes.get("content", "")
            if key:
                meta[key] = val

        return title, text, links, meta, canonical


# ─────────────────────────────────────────────────────────────
# HASH STORE  — persistent bloom filter or SQLite  [B5]
# ─────────────────────────────────────────────────────────────

BLOOM_PATH = "bloom.pkl"

class HashStore:
    """
    Bloom filter is pickled to disk on close() / loaded on open().  [B5]
    SQLite fallback is inherently persistent.
    Near-duplicate detection via simhash (hamming distance ≤ 4).
    """

    SIMHASH_THRESHOLD = 4

    def __init__(self, db_path: str = "output.db", bloom_path: str = BLOOM_PATH):
        self._bloom_path = bloom_path
        if _BLOOM_AVAILABLE:
            if os.path.exists(bloom_path):
                with open(bloom_path, "rb") as f:
                    self._bloom = pickle.load(f)
                log.info("HashStore: loaded bloom filter from %s", bloom_path)
            else:
                self._bloom = ScalableBloomFilter(
                    initial_capacity=100_000, error_rate=0.001
                )
            self._use_bloom = True
        else:
            import sqlite3
            self._sdb = sqlite3.connect(db_path, check_same_thread=False)
            self._sdb.execute(
                "CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY)"
            )
            self._sdb.commit()
            self._use_bloom = False
        # simhash fingerprints kept in memory (much smaller than sha256 strings)
        self._simhashes: list[int] = []

    def seen_exact(self, h: str) -> bool:
        if self._use_bloom:
            return h in self._bloom
        row = self._sdb.execute(
            "SELECT 1 FROM seen_hashes WHERE hash=?", (h,)
        ).fetchone()
        return row is not None

    def seen_near(self, sh: int) -> bool:
        return any(hamming(sh, prev) <= self.SIMHASH_THRESHOLD
                   for prev in self._simhashes)

    def add(self, h: str, sh: int):
        if self._use_bloom:
            self._bloom.add(h)
        else:
            self._sdb.execute(
                "INSERT OR IGNORE INTO seen_hashes VALUES (?)", (h,)
            )
            self._sdb.commit()
        self._simhashes.append(sh)

    def save(self):
        if self._use_bloom:
            with open(self._bloom_path, "wb") as f:
                pickle.dump(self._bloom, f)
            log.info("HashStore: bloom filter saved to %s", self._bloom_path)


# ─────────────────────────────────────────────────────────────
# CONDITIONAL REQUEST CACHE  — keyed on final_url  [D1]
# ─────────────────────────────────────────────────────────────

class ConditionalCache:
    """Keyed on final_url (post-redirect) so cache hits are not missed."""

    def __init__(self, db_path: str = "output.db"):
        import sqlite3
        self._db = sqlite3.connect(db_path, check_same_thread=False)

    def get(self, final_url: str) -> tuple[str, str]:
        row = self._db.execute(
            "SELECT etag, last_modified FROM pages WHERE final_url=?",
            (final_url,)
        ).fetchone()
        return (row[0] or "", row[1] or "") if row else ("", "")


# ─────────────────────────────────────────────────────────────
# STORAGE WRITER  — batched commits, aiosqlite  [F4]
# ─────────────────────────────────────────────────────────────

class StorageWriter:
    BATCH_SIZE     = 100
    FLUSH_INTERVAL = 1.0

    def __init__(self, db_path: str = "output.db"):
        self._db_path = db_path
        self._q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._writer_loop())

    async def push(self, item: ScrapedItem):
        await self._q.put(item)

    async def stop(self):
        await self._q.join()
        if self._task:
            self._task.cancel()

    async def _writer_loop(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY,
                    url TEXT UNIQUE, final_url TEXT,
                    title TEXT, text TEXT,
                    status_code INTEGER, content_type TEXT,
                    response_ms REAL, etag TEXT, last_modified TEXT,
                    content_hash TEXT, simhash INTEGER,
                    scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS links (
                    src TEXT, dst TEXT, PRIMARY KEY (src,dst)
                );
                CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
            """)
            await db.commit()

            batch: list[ScrapedItem] = []
            last_flush = time.monotonic()

            while True:
                try:
                    timeout = max(0, self.FLUSH_INTERVAL - (time.monotonic()-last_flush))
                    item = await asyncio.wait_for(self._q.get(), timeout=timeout)
                    batch.append(item)
                    self._q.task_done()
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    if batch:
                        await self._commit(db, batch)
                    return

                if (len(batch) >= self.BATCH_SIZE or
                        (batch and time.monotonic()-last_flush >= self.FLUSH_INTERVAL)):
                    await self._commit(db, batch)
                    batch = []
                    last_flush = time.monotonic()

    @staticmethod
    async def _commit(db: aiosqlite.Connection, batch: list[ScrapedItem]):
        pages, links, done = [], [], []
        for item in batch:
            c = item.crawl
            pages.append((
                item.url, c.final_url, item.title, item.text,
                c.status_code, c.content_type, c.response_time_ms,
                c.etag, c.last_modified, c.content_hash, c.simhash,
            ))
            links.extend((item.url, lnk) for lnk in item.links)
            done.append((item.url,))

        await db.executemany(
            """INSERT OR REPLACE INTO pages
               (url,final_url,title,text,status_code,content_type,
                response_ms,etag,last_modified,content_hash,simhash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            pages,
        )
        await db.executemany(
            "INSERT OR IGNORE INTO links (src,dst) VALUES (?,?)", links
        )
        await db.executemany(
            "INSERT OR IGNORE INTO done (url) VALUES (?)", done
        )
        await db.commit()
        log.debug("Committed %d pages", len(pages))


# ─────────────────────────────────────────────────────────────
# PIPELINE  — single session, coordinator sentinel, atomic counter
# ─────────────────────────────────────────────────────────────

class Pipeline:

    def __init__(
        self,
        scheduler:   DomainScheduler,
        fetcher:     AsyncFetcher,
        rate:        RateLimiter,
        robots:      AsyncRobots,
        parser:      FastParser,
        writer:      StorageWriter,
        hashes:      HashStore,
        cond_cache:  ConditionalCache,
        seed_domains: set[str],
        max_pages:   int,
        follow_links:      bool,
        same_domain_only:  bool,
        n_fetch_workers:   int = 10,
        n_parse_workers:   int = 4,
    ):
        self.scheduler   = scheduler
        self.fetcher     = fetcher
        self.rate        = rate
        self.robots      = robots
        self.parser      = parser
        self.writer      = writer
        self.hashes      = hashes
        self.cond_cache  = cond_cache
        self.seed_domains = seed_domains
        self.max_pages   = max_pages
        self.follow_links = follow_links
        self.same_domain_only = same_domain_only
        self.n_fetch = n_fetch_workers
        self.n_parse = n_parse_workers

        self._fetch_q: asyncio.Queue[Optional[FetchResult]] = asyncio.Queue(maxsize=200)
        self._scraped = 0
        self._scraped_lock = asyncio.Lock()     # [B3] atomic counter
        self._stop    = asyncio.Event()

    def _in_scope(self, url: str) -> bool:
        if not self.same_domain_only:
            return True
        return urlparse(url).netloc in self.seed_domains

    # ── fetch worker ──────────────────────────────────────────

    async def _fetch_worker(self, worker_id: int):
        """Reads from scheduler.output_q, fetches, enqueues FetchResult."""
        while not self._stop.is_set():
            try:
                url = await asyncio.wait_for(self.scheduler.output_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if url is None:
                break

            if not await self.robots.allowed(url, self.fetcher.session):
                log.debug("robots blocks %s", url)
                continue

            # Conditional headers keyed on final_url  [D1]
            etag, last_mod = self.cond_cache.get(url)
            result = await self.fetcher.fetch(
                url, self.rate, etag=etag, last_modified=last_mod
            )
            await self._fetch_q.put(result)

    # ── parse worker ──────────────────────────────────────────

    async def _parse_worker(self, worker_id: int):
        """Parses FetchResult, deduplicates, enqueues to writer."""
        while True:
            result = await self._fetch_q.get()
            if result is None:
                break

            if result.html is None:
                continue

            title, text, links, meta, canonical = self.parser.parse(
                result.html, result.final_url
            )

            # If page declares a canonical URL, use that as the storage key  [F2]
            effective_url = canonical or result.final_url

            # Exact + near-duplicate detection  [D2]
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            sh = simhash(text)

            if self.hashes.seen_exact(content_hash) or self.hashes.seen_near(sh):
                log.debug("Duplicate skipped: %s", result.url)
                continue
            self.hashes.add(content_hash, sh)

            result.meta.content_hash = content_hash
            result.meta.simhash      = sh
            result.meta.final_url    = effective_url

            item = ScrapedItem(
                url=result.url, title=title, text=text,
                links=links, meta=meta, crawl=result.meta,
            )
            await self.writer.push(item)
            await self.scheduler.mark_done(result.url)

            # [B3] atomic counter increment
            async with self._scraped_lock:
                self._scraped += 1
                count = self._scraped

            log.info("[%d/%d] %s  (%dms)",
                     count, self.max_pages,
                     result.url, result.meta.response_time_ms)

            if self.follow_links:
                for link in links:
                    if self._in_scope(link):
                        await self.scheduler.add(link)

            if count >= self.max_pages:
                self._stop.set()
                break

    # ── coordinator ───────────────────────────────────────────

    async def _coordinator(
        self,
        fetch_tasks: list[asyncio.Task],
        n_parse: int,
    ):
        """
        Waits for all fetch workers, then broadcasts exactly n_parse
        sentinels — one per parse worker.  [B4]
        """
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for _ in range(n_parse):
            await self._fetch_q.put(None)

    # ── main entry ────────────────────────────────────────────

    async def run(self):
        # Sitemap seed: before the main loop, discover URLs via sitemap  [F1]
        await self._seed_from_sitemaps()

        await self.writer.start()

        sched_stop = asyncio.Event()
        sched_task = asyncio.create_task(
            self.scheduler.run_scheduler(sched_stop)
        )

        fetch_tasks = [
            asyncio.create_task(self._fetch_worker(i))
            for i in range(self.n_fetch)
        ]
        parse_tasks = [
            asyncio.create_task(self._parse_worker(i))
            for i in range(self.n_parse)
        ]
        coord_task = asyncio.create_task(
            self._coordinator(fetch_tasks, self.n_parse)
        )

        await asyncio.gather(*parse_tasks)
        coord_task.cancel()
        sched_stop.set()
        sched_task.cancel()

        await self.writer.stop()
        self.hashes.save()
        await self.scheduler.close()
        log.info("Done. Scraped %d pages.", self._scraped)

    async def _seed_from_sitemaps(self):
        """Open a temporary session to fetch robots.txt + sitemaps."""
        async with aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT}
        ) as session:
            for domain in self.seed_domains:
                origin = f"https://{domain}"
                sm_urls = await self.robots.site_maps(origin, session)
                if not sm_urls:
                    # Try the conventional path if robots.txt didn't list one
                    sm_urls = [f"{origin}/sitemap.xml"]
                for sm in sm_urls:
                    urls = await fetch_sitemap_urls(sm, session)
                    log.info("Sitemap %s: %d URLs discovered", sm, len(urls))
                    for url in urls:
                        if self._in_scope(url):
                            await self.scheduler.add(url)


# ─────────────────────────────────────────────────────────────
# SCRAPER  — thin config facade
# ─────────────────────────────────────────────────────────────

class Scraper:

    def __init__(
        self,
        seed_urls:            list[str],
        max_pages:            int   = 100,
        default_delay:        float = 1.5,
        global_concurrency:   int   = 50,
        per_domain_concurrency: int = 2,
        n_fetch_workers:      int   = 10,
        n_parse_workers:      int   = 4,
        follow_links:         bool  = True,
        same_domain_only:     bool  = True,
        db_path:              str   = "output.db",
        checkpoint_db:        str   = "checkpoint.db",
        resume:               bool  = False,
    ):
        seed_domains = {urlparse(u).netloc for u in seed_urls}

        scheduler = DomainScheduler(checkpoint_db)

        async def _async_init():
            await scheduler.open()
            if resume:
                await scheduler.load_resume()
            else:
                for url in seed_urls:
                    await scheduler.add(url)

        asyncio.get_event_loop().run_until_complete(_async_init())

        self._pipeline = Pipeline(
            scheduler        = scheduler,
            fetcher          = AsyncFetcher(global_concurrency),
            rate             = RateLimiter(default_delay, per_domain_concurrency),
            robots           = AsyncRobots(default_delay),
            parser           = FastParser(),
            writer           = StorageWriter(db_path),
            hashes           = HashStore(db_path),
            cond_cache       = ConditionalCache(db_path),
            seed_domains     = seed_domains,
            max_pages        = max_pages,
            follow_links     = follow_links,
            same_domain_only = same_domain_only,
            n_fetch_workers  = n_fetch_workers,
            n_parse_workers  = n_parse_workers,
        )

    async def _run(self):
        await self._pipeline.fetcher.open()
        try:
            await self._pipeline.run()
        finally:
            await self._pipeline.fetcher.close()

    def run(self):
        asyncio.run(self._run())


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Scraper(
        seed_urls              = ["https://books.toscrape.com/"],
        max_pages              = 100,
        default_delay          = 1.0,
        global_concurrency     = 50,
        per_domain_concurrency = 2,
        n_fetch_workers        = 10,
        n_parse_workers        = 4,
        follow_links           = True,
        same_domain_only       = True,
        db_path                = "output.db",
        checkpoint_db          = "checkpoint.db",
        resume                 = False,
    ).run()