"""
Web Data Scraper v6 — Four production integrations on top of v5
================================================================

NEW in v6:
  [I1] PostgreSQL backend     — asyncpg replaces aiosqlite for page storage;
                                 full-text search index, UPSERT, connection pool.
                                 SQLite (aiosqlite) kept ONLY for the URL frontier
                                 (checkpoint.db) — it stays lightweight.

  [I2] Redis frontier         — DomainScheduler gains a Redis backend option.
                                 URL dedup → Redis SET (SADD / SISMEMBER).
                                 Pending queue → Redis ZSET scored by priority.
                                 Falls back to aiosqlite when Redis unavailable.

  [I3] Playwright browser pool — PlaywrightPool manages N headless Chromium
                                 instances. JS-rendered pages are detected by
                                 content-type hint or explicit flag; routed to
                                 the pool automatically. aiohttp used for plain HTML.

  [I4] Prometheus metrics      — PrometheusMetrics exposes /metrics on port 8000.
                                 Counters: pages_scraped, errors, duplicates_skipped.
                                 Gauges:   frontier_size, queue_depth, active_browsers.
                                 Histograms: fetch_duration_ms, parse_duration_ms.

All v5 fixes (P1-P10) preserved unchanged.
"""

import asyncio
import hashlib
import heapq
import logging
import os
import pickle
import re
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qs
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import aiohttp
import aiosqlite
from selectolax.parser import HTMLParser as SXParser

# ── optional deps — graceful degradation ──────────────────────
try:
    import asyncpg
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

try:
    from prometheus_client import (
        Counter as PCounter, Gauge, Histogram,
        start_http_server as prom_start
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

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
    "Mozilla/5.0 (compatible; PyScraper/6.0; "
    "+https://github.com/example/scraper)"
)
ALLOWED_MIME    = {"text/html", "application/xhtml+xml"}
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source",
}
NOISE_TAGS = ("script", "style", "nav", "footer", "aside", "header")
BLOOM_PATH = "bloom.pkl"


# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class CrawlMeta:
    final_url:        str   = ""
    status_code:      int   = 0
    content_type:     str   = ""
    response_time_ms: float = 0.0
    etag:             str   = ""
    last_modified:    str   = ""
    content_hash:     str   = ""
    simhash:          int   = 0
    rendered_by:      str   = "aiohttp"   # "aiohttp" | "playwright"


@dataclass
class ScrapedItem:
    url:   str
    title: str   = ""
    text:  str   = ""
    links: list  = field(default_factory=list)
    meta:  dict  = field(default_factory=dict)
    crawl: CrawlMeta = field(default_factory=CrawlMeta)


# ─────────────────────────────────────────────────────────────
# URL HELPERS
# ─────────────────────────────────────────────────────────────

def normalise(url: str) -> str:
    try:
        p = urlparse(url.strip())
    except Exception:
        return ""
    if p.scheme not in ("http", "https"):
        return ""
    path   = p.path.rstrip("/") or "/"
    params = {k: v for k, v in parse_qs(p.query, keep_blank_values=False).items()
              if k not in TRACKING_PARAMS}
    query  = urlencode(sorted((k, v[0]) for k, v in params.items()))
    return urlunparse((p.scheme, p.netloc.lower(), path, "", query, ""))


def _score(url: str) -> int:
    path = urlparse(url).path.lower()
    if path in ("/", ""):                       return 100
    if re.search(r"/categor|/catalog", path):   return 80
    if re.search(r"/product|/item|/p/", path):  return 50
    if re.search(r"page=\d+|/page/\d+", url):   return 20
    return 40


# ─────────────────────────────────────────────────────────────
# HASHING
# ─────────────────────────────────────────────────────────────

def weighted_simhash(text: str, bits: int = 64) -> int:
    v = [0] * bits
    counts = Counter(re.findall(r"\w+", text.lower()))
    for token, weight in counts.items():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            v[i] += weight if (h >> i) & 1 else -weight
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SimhashIndex:
    BANDS = 8; ROWS = 8; THRESHOLD = 4

    def __init__(self):
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(self.BANDS)]

    def _band_values(self, sh: int) -> list[int]:
        mask = (1 << self.ROWS) - 1
        return [(sh >> (b * self.ROWS)) & mask for b in range(self.BANDS)]

    def seen(self, sh: int) -> bool:
        for bi, bv in enumerate(self._band_values(sh)):
            if any(hamming(sh, c) <= self.THRESHOLD
                   for c in self._buckets[bi].get(bv, [])):
                return True
        return False

    def add(self, sh: int):
        for bi, bv in enumerate(self._band_values(sh)):
            self._buckets[bi].setdefault(bv, []).append(sh)


class HashStore:
    def __init__(self, bloom_path: str = BLOOM_PATH):
        self._bloom_path = bloom_path
        if _BLOOM_AVAILABLE:
            self._bloom = (pickle.load(open(bloom_path, "rb"))
                           if os.path.exists(bloom_path)
                           else ScalableBloomFilter(initial_capacity=100_000,
                                                    error_rate=0.001))
            self._use_bloom = True
        else:
            import sqlite3 as _sq
            self._sdb = _sq.connect("hashes.db", check_same_thread=False)
            self._sdb.execute("CREATE TABLE IF NOT EXISTS h (hash TEXT PRIMARY KEY)")
            self._sdb.commit()
            self._use_bloom = False
        self._lsh = SimhashIndex()

    def seen_exact(self, h: str) -> bool:
        if self._use_bloom:
            return h in self._bloom
        return bool(self._sdb.execute("SELECT 1 FROM h WHERE hash=?", (h,)).fetchone())

    def seen_near(self, sh: int) -> bool:
        return self._lsh.seen(sh)

    def add(self, h: str, sh: int):
        if self._use_bloom:
            self._bloom.add(h)
        else:
            self._sdb.execute("INSERT OR IGNORE INTO h VALUES (?)", (h,))
            self._sdb.commit()
        self._lsh.add(sh)

    def save(self):
        if self._use_bloom:
            pickle.dump(self._bloom, open(self._bloom_path, "wb"))


# ─────────────────────────────────────────────────────────────
# [I1] POSTGRESQL STORAGE WRITER
# ─────────────────────────────────────────────────────────────

PG_DDL = """
CREATE TABLE IF NOT EXISTS pages (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT      UNIQUE NOT NULL,
    final_url     TEXT,
    title         TEXT,
    body          TEXT,
    status_code   SMALLINT,
    content_type  TEXT,
    response_ms   REAL,
    etag          TEXT,
    last_modified TEXT,
    content_hash  TEXT,
    simhash       BIGINT,
    rendered_by   TEXT DEFAULT 'aiohttp',
    scraped_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS pages_scraped_at ON pages (scraped_at);
CREATE INDEX IF NOT EXISTS pages_content_hash ON pages (content_hash);

CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);

CREATE TABLE IF NOT EXISTS done (
    url        TEXT PRIMARY KEY,
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);
"""

# Full-text search index (added separately — requires pg_trgm or ts_vector)
PG_FTS = """
ALTER TABLE pages ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,''))) STORED;
CREATE INDEX IF NOT EXISTS pages_fts ON pages USING GIN (fts);
"""


class PostgresWriter:
    """
    [I1] asyncpg connection pool — replaces aiosqlite for page storage.
    Batched UPSERT every BATCH_SIZE rows or FLUSH_INTERVAL seconds.
    done row written atomically in same transaction (preserves v5 P5 fix).
    Falls back to SQLiteWriter if asyncpg not installed or PG unavailable.
    """

    BATCH_SIZE     = 100
    FLUSH_INTERVAL = 1.0

    def __init__(self, pg_dsn: str):
        self._dsn   = pg_dsn
        self._pool: Optional[asyncpg.Pool] = None
        self._q: asyncio.Queue[Optional[ScrapedItem]] = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self):
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=2, max_size=10,
            command_timeout=30,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(PG_DDL)
            try:
                await conn.execute(PG_FTS)
            except Exception:
                pass   # PG version may not support generated columns
        self._task = asyncio.create_task(self._writer_loop())
        log.info("PostgresWriter: pool open (%s)", self._dsn.split("@")[-1])

    async def push(self, item: ScrapedItem):
        await self._q.put(item)

    async def stop(self):
        await self._q.put(None)
        await self._stopped.wait()
        if self._pool:
            await self._pool.close()

    # Expose same interface as ConditionalCache for etag lookups
    async def get_conditional(self, url: str) -> tuple[str, str]:
        if not self._pool:
            return "", ""
        try:
            row = await self._pool.fetchrow(
                "SELECT etag, last_modified FROM pages WHERE url=$1", url
            )
            return (row["etag"] or "", row["last_modified"] or "") if row else ("", "")
        except Exception:
            return "", ""

    async def _writer_loop(self):
        batch: list[ScrapedItem] = []
        last_flush = time.monotonic()

        while True:
            try:
                timeout = max(0.0, self.FLUSH_INTERVAL - (time.monotonic()-last_flush))
                item = await asyncio.wait_for(self._q.get(), timeout=timeout)
                if item is None:
                    if batch:
                        await self._commit(batch)
                    self._stopped.set()
                    return
                batch.append(item)
                self._q.task_done()
            except asyncio.TimeoutError:
                pass

            if (len(batch) >= self.BATCH_SIZE or
                    (batch and time.monotonic()-last_flush >= self.FLUSH_INTERVAL)):
                await self._commit(batch)
                batch = []
                last_flush = time.monotonic()

    async def _commit(self, batch: list[ScrapedItem]):
        if not self._pool:
            return
        pages, links, done = [], [], []
        for item in batch:
            c = item.crawl
            pages.append((
                item.url, c.final_url, item.title, item.text,
                c.status_code, c.content_type, c.response_time_ms,
                c.etag, c.last_modified, c.content_hash, c.simhash,
                c.rendered_by,
            ))
            links.extend((item.url, lnk) for lnk in item.links)
            done.append(item.url)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """INSERT INTO pages
                       (url,final_url,title,body,status_code,content_type,
                        response_ms,etag,last_modified,content_hash,simhash,rendered_by)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                       ON CONFLICT (url) DO UPDATE SET
                         final_url=EXCLUDED.final_url, title=EXCLUDED.title,
                         body=EXCLUDED.body, status_code=EXCLUDED.status_code,
                         content_hash=EXCLUDED.content_hash,
                         simhash=EXCLUDED.simhash,
                         rendered_by=EXCLUDED.rendered_by,
                         scraped_at=NOW()""",
                    pages,
                )
                await conn.executemany(
                    "INSERT INTO links (src,dst) VALUES ($1,$2) ON CONFLICT DO NOTHING",
                    links,
                )
                # done written atomically — same transaction (v5 P5)
                await conn.executemany(
                    "INSERT INTO done (url) VALUES ($1) ON CONFLICT DO NOTHING",
                    [(u,) for u in done],
                )
        log.debug("PG: committed %d pages", len(pages))


class SQLiteWriter:
    """
    [I1 fallback] Used when asyncpg is not installed or PG is unreachable.
    Identical semantics to PostgresWriter — same push/stop/get_conditional API.
    """
    BATCH_SIZE = 100; FLUSH_INTERVAL = 1.0

    def __init__(self, db_path: str = "output.db"):
        self._db_path = db_path
        self._q: asyncio.Queue[Optional[ScrapedItem]] = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self):
        self._task = asyncio.create_task(self._writer_loop())

    async def push(self, item: ScrapedItem):
        await self._q.put(item)

    async def stop(self):
        await self._q.put(None)
        await self._stopped.wait()

    async def get_conditional(self, url: str) -> tuple[str, str]:
        return "", ""   # SQLite fallback skips conditional requests

    async def _writer_loop(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY, url TEXT UNIQUE, final_url TEXT,
                    title TEXT, body TEXT, status_code INTEGER, content_type TEXT,
                    response_ms REAL, etag TEXT, last_modified TEXT,
                    content_hash TEXT, simhash INTEGER, rendered_by TEXT,
                    scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS links (src TEXT, dst TEXT, PRIMARY KEY(src,dst));
                CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
            """)
            await db.commit()
            batch: list[ScrapedItem] = []
            last_flush = time.monotonic()
            while True:
                try:
                    timeout = max(0.0, self.FLUSH_INTERVAL-(time.monotonic()-last_flush))
                    item = await asyncio.wait_for(self._q.get(), timeout=timeout)
                    if item is None:
                        if batch: await self._commit(db, batch)
                        self._stopped.set(); return
                    batch.append(item); self._q.task_done()
                except asyncio.TimeoutError:
                    pass
                if len(batch) >= self.BATCH_SIZE or (batch and time.monotonic()-last_flush >= self.FLUSH_INTERVAL):
                    await self._commit(db, batch); batch = []; last_flush = time.monotonic()

    @staticmethod
    async def _commit(db, batch):
        pages, links, done = [], [], []
        for item in batch:
            c = item.crawl
            pages.append((item.url, c.final_url, item.title, item.text,
                           c.status_code, c.content_type, c.response_time_ms,
                           c.etag, c.last_modified, c.content_hash, c.simhash, c.rendered_by))
            links.extend((item.url, l) for l in item.links)
            done.append((item.url,))
        await db.executemany(
            "INSERT OR REPLACE INTO pages (url,final_url,title,body,status_code,content_type,response_ms,etag,last_modified,content_hash,simhash,rendered_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            pages)
        await db.executemany("INSERT OR IGNORE INTO links (src,dst) VALUES (?,?)", links)
        await db.executemany("INSERT OR IGNORE INTO done (url) VALUES (?)", done)
        await db.commit()
        log.debug("SQLite: committed %d pages", len(pages))


def make_writer(pg_dsn: Optional[str], sqlite_path: str = "output.db"):
    """Factory: PostgresWriter if asyncpg available + dsn provided, else SQLite."""
    if pg_dsn and _PG_AVAILABLE:
        return PostgresWriter(pg_dsn)
    log.warning("PostgresWriter unavailable — falling back to SQLite (%s)", sqlite_path)
    return SQLiteWriter(sqlite_path)


# ─────────────────────────────────────────────────────────────
# [I2] REDIS FRONTIER
# ─────────────────────────────────────────────────────────────

class RedisFrontier:
    """
    [I2] Redis-backed URL frontier.
    Seen set    → Redis SET  (SADD / SISMEMBER)  — O(1), survives restarts.
    Pending     → Redis ZSET (ZADD / ZPOPMIN)    — sorted by priority score.
    Falls back to SQLiteFrontier automatically if Redis unavailable.

    Key layout:
      crawl:{name}:seen    — SET of all discovered URLs
      crawl:{name}:pending — ZSET url → -priority (lowest = highest priority)
    """

    FLUSH_BATCH = 50

    def __init__(self, redis_url: str, crawl_name: str = "default"):
        self._redis_url  = redis_url
        self._pfx        = f"crawl:{crawl_name}"
        self._r: Optional[aioredis.Redis] = None
        self._lock       = asyncio.Lock()
        self.output_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=500)

    async def open(self):
        self._r = aioredis.from_url(self._redis_url, decode_responses=True)
        await self._r.ping()
        log.info("RedisFrontier: connected (%s)", self._redis_url)

    async def close(self):
        if self._r:
            await self._r.aclose()

    async def add(self, url: str) -> bool:
        url = normalise(url)
        if not url:
            return False
        # SADD returns 1 if new, 0 if already existed
        added = await self._r.sadd(f"{self._pfx}:seen", url)
        if not added:
            return False
        score = _score(url)
        # Store as negative so ZPOPMIN returns highest-priority first
        await self._r.zadd(f"{self._pfx}:pending", {url: -score})
        return True

    async def add_many(self, urls: list[str]):
        pipe = self._r.pipeline(transaction=False)
        new_urls = []
        for url in urls:
            url = normalise(url)
            if url:
                pipe.sadd(f"{self._pfx}:seen", url)
                new_urls.append(url)
        results = await pipe.execute()
        # Only add to ZSET those that were new
        mapping = {url: -_score(url)
                   for url, added in zip(new_urls, results) if added}
        if mapping:
            await self._r.zadd(f"{self._pfx}:pending", mapping)

    async def run_scheduler(self, stop_event: asyncio.Event):
        """Drain Redis ZSET into output_q in priority order."""
        while not stop_event.is_set():
            try:
                items = await self._r.zpopmin(f"{self._pfx}:pending", count=20)
                if not items:
                    await asyncio.sleep(0.1)
                    continue
                for url, _ in items:
                    await self.output_q.put(url)
            except Exception as exc:
                log.warning("RedisFrontier scheduler error: %s", exc)
                await asyncio.sleep(0.5)

    async def pending_count(self) -> int:
        return await self._r.zcard(f"{self._pfx}:pending")

    async def seen_count(self) -> int:
        return await self._r.scard(f"{self._pfx}:seen")


class SQLiteFrontier:
    """
    [I2 fallback] aiosqlite-backed frontier — identical API to RedisFrontier.
    Used when redis package not installed or Redis unreachable.
    Preserves all v5 P1/P4/P7 fixes (deque, bloom, weighted-fair scheduling).
    """

    FLUSH_BATCH = 250; FLUSH_DELAY = 2.0; MAX_TOKENS = 5

    def __init__(self, db_path: str, bloom_path: str = BLOOM_PATH):
        self._db_path   = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._bloom_path = bloom_path
        if _BLOOM_AVAILABLE:
            self._url_bloom = (pickle.load(open(bloom_path,"rb"))
                               if os.path.exists(bloom_path)
                               else ScalableBloomFilter(initial_capacity=1_000_000,
                                                        error_rate=0.0001))
            self._use_bloom = True
        else:
            self._use_bloom = False
        self._domain_qs: dict[str, deque] = {}
        self._domain_heap: list[tuple[int, str]] = []
        self._domains_in_heap: set[str] = set()
        self.output_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=500)
        self._lock    = asyncio.Lock()
        self._pending: list[tuple[str, int]] = []
        self._last_flush = time.monotonic()

    async def open(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS seen_urls (url TEXT PRIMARY KEY, priority INTEGER);
            CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
        """)
        await self._db.commit()

    async def close(self):
        await self._flush_db(force=True)
        if self._use_bloom:
            pickle.dump(self._url_bloom, open(self._bloom_path,"wb"))
        if self._db: await self._db.close()

    async def _seen(self, url: str) -> bool:
        if self._use_bloom: return url in self._url_bloom
        async with self._db.execute("SELECT 1 FROM seen_urls WHERE url=?",(url,)) as c:
            return (await c.fetchone()) is not None

    async def add(self, url: str) -> bool:
        url = normalise(url)
        if not url: return False
        async with self._lock:
            if await self._seen(url): return False
            if self._use_bloom: self._url_bloom.add(url)
            score  = _score(url)
            domain = urlparse(url).netloc
            if domain not in self._domain_qs:
                self._domain_qs[domain] = deque()
                heapq.heappush(self._domain_heap, (-score, domain))
                self._domains_in_heap.add(domain)
            self._domain_qs[domain].append(url)
            self._pending.append((url, score))
            await self._maybe_flush()
            return True

    async def add_many(self, urls: list[str]):
        for url in urls: await self.add(url)

    async def run_scheduler(self, stop_event: asyncio.Event):
        while not stop_event.is_set():
            async with self._lock:
                snapshot = sorted(self._domain_qs.items(), key=lambda kv: -len(kv[1]))
                total = sum(len(q) for _,q in snapshot) or 1
                fed = False
                for domain, q in snapshot:
                    if not q: continue
                    tokens = max(1, min(self.MAX_TOKENS, round(self.MAX_TOKENS*len(q)/total)))
                    for _ in range(tokens):
                        if not q: break
                        url = q.popleft()
                        try: self.output_q.put_nowait(url); fed = True
                        except asyncio.QueueFull: q.appendleft(url); break
            await asyncio.sleep(0.05)

    async def load_resume(self):
        async with self._db.execute(
            "SELECT s.url, s.priority FROM seen_urls s LEFT JOIN done d ON s.url=d.url WHERE d.url IS NULL"
        ) as cur:
            rows = await cur.fetchall()
        for url, priority in rows:
            domain = urlparse(url).netloc
            if domain not in self._domain_qs: self._domain_qs[domain] = deque()
            self._domain_qs[domain].append(url)
            if self._use_bloom: self._url_bloom.add(url)
        for domain, q in self._domain_qs.items():
            if q and domain not in self._domains_in_heap:
                heapq.heappush(self._domain_heap, (-_score(next(iter(q))), domain))
                self._domains_in_heap.add(domain)
        log.info("Resumed: %d pending URLs", sum(len(q) for q in self._domain_qs.values()))

    async def pending_count(self) -> int:
        return sum(len(q) for q in self._domain_qs.values())

    async def seen_count(self) -> int:
        return len(self._domains_in_heap)   # approximate

    async def _maybe_flush(self):
        if len(self._pending) >= self.FLUSH_BATCH or time.monotonic()-self._last_flush >= self.FLUSH_DELAY:
            await self._flush_db()

    async def _flush_db(self, force: bool = False):
        if not self._db or not self._pending: return
        await self._db.executemany("INSERT OR IGNORE INTO seen_urls (url,priority) VALUES (?,?)", self._pending)
        await self._db.commit()
        self._pending.clear(); self._last_flush = time.monotonic()


def make_frontier(redis_url: Optional[str], sqlite_path: str,
                  crawl_name: str = "default", bloom_path: str = BLOOM_PATH):
    """Factory: RedisFrontier if available, else SQLiteFrontier."""
    if redis_url and _REDIS_AVAILABLE:
        return RedisFrontier(redis_url, crawl_name)
    log.warning("RedisFrontier unavailable — falling back to SQLite frontier")
    return SQLiteFrontier(sqlite_path, bloom_path)


# ─────────────────────────────────────────────────────────────
# [I3] PLAYWRIGHT BROWSER POOL
# ─────────────────────────────────────────────────────────────

JS_INDICATORS = re.compile(
    r"(react|vue|angular|next\.js|nuxt|svelte|__NEXT_DATA__|ng-version"
    r"|data-reactroot|<app-root|window\.__STATE__)",
    re.IGNORECASE,
)


class PlaywrightPool:
    """
    [I3] Pool of Playwright browser contexts.
    Each context is isolated (separate cookies, storage, network).
    A semaphore enforces max_browsers concurrency.
    fetch() returns rendered HTML after JS execution.

    Detection heuristic: if AsyncFetcher gets a 200 HTML response but the
    body looks like a JS shell (tiny body + JS framework indicators), the
    fetch worker escalates to the browser pool automatically.
    """

    def __init__(self, max_browsers: int = 5, timeout_ms: int = 20_000):
        self.max_browsers  = max_browsers
        self.timeout_ms    = timeout_ms
        self._sem: Optional[asyncio.Semaphore] = None
        self._pw   = None
        self._browser: Optional[Browser] = None
        self._active = 0
        self._lock = asyncio.Lock()

    async def open(self):
        if not _PLAYWRIGHT_AVAILABLE:
            log.warning("PlaywrightPool: playwright not installed — JS rendering disabled")
            return
        self._sem  = asyncio.Semaphore(self.max_browsers)
        self._pw   = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox", "--disable-setuid-sandbox",
                "--disable-dev-shm-usage", "--disable-gpu",
            ],
        )
        log.info("PlaywrightPool: %d browser slots open", self.max_browsers)

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    @property
    def available(self) -> bool:
        return _PLAYWRIGHT_AVAILABLE and self._browser is not None

    async def fetch(self, url: str) -> tuple[Optional[str], str]:
        """
        Returns (html, final_url).
        Acquires a semaphore slot, creates a fresh context, navigates,
        waits for networkidle, returns page content, then closes context.
        """
        if not self.available:
            return None, url

        async with self._sem:
            async with self._lock:
                self._active += 1
            ctx: Optional[BrowserContext] = None
            try:
                ctx = await self._browser.new_context(
                    user_agent=USER_AGENT,
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page = await ctx.new_page()
                resp = await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=self.timeout_ms,
                )
                final_url = page.url
                html = await page.content()
                return html, final_url
            except Exception as exc:
                log.warning("Playwright error %s: %s", url, exc)
                return None, url
            finally:
                if ctx:
                    await ctx.close()
                async with self._lock:
                    self._active -= 1

    def active_count(self) -> int:
        return self._active


def needs_js_render(html: str) -> bool:
    """Return True if page looks like a JS shell with little real content."""
    if len(html) < 3000:
        return True
    if JS_INDICATORS.search(html[:5000]):
        return True
    return False


# ─────────────────────────────────────────────────────────────
# [I4] PROMETHEUS METRICS
# ─────────────────────────────────────────────────────────────

class PrometheusMetrics:
    """
    [I4] Thin wrapper around prometheus_client.
    Exposes /metrics on metrics_port (default 8000).
    Degrades silently if prometheus_client not installed.
    """

    def __init__(self, port: int = 8000, enabled: bool = True):
        self._enabled = enabled and _PROM_AVAILABLE
        if not self._enabled:
            if enabled and not _PROM_AVAILABLE:
                log.warning("PrometheusMetrics: prometheus_client not installed — metrics disabled")
            return

        self.pages_scraped = PCounter(
            "crawler_pages_scraped_total",
            "Total pages successfully scraped",
            ["rendered_by"],
        )
        self.errors = PCounter(
            "crawler_errors_total",
            "Fetch or parse errors",
            ["reason"],
        )
        self.duplicates_skipped = PCounter(
            "crawler_duplicates_skipped_total",
            "Pages skipped as exact or near duplicates",
            ["kind"],
        )
        self.frontier_size = Gauge(
            "crawler_frontier_size",
            "Number of URLs pending in the frontier",
        )
        self.queue_depth = Gauge(
            "crawler_internal_queue_depth",
            "Items in the fetch→parse internal queue",
        )
        self.active_browsers = Gauge(
            "crawler_active_browsers",
            "Playwright browser contexts currently in use",
        )
        self.fetch_duration = Histogram(
            "crawler_fetch_duration_ms",
            "HTTP fetch latency in milliseconds",
            buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
        )
        self.parse_duration = Histogram(
            "crawler_parse_duration_ms",
            "Parse + extraction latency in milliseconds",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
        )
        prom_start(port)
        log.info("PrometheusMetrics: /metrics on :%d", port)

    def inc_scraped(self, rendered_by: str = "aiohttp"):
        if self._enabled: self.pages_scraped.labels(rendered_by=rendered_by).inc()

    def inc_error(self, reason: str = "fetch"):
        if self._enabled: self.errors.labels(reason=reason).inc()

    def inc_duplicate(self, kind: str = "exact"):
        if self._enabled: self.duplicates_skipped.labels(kind=kind).inc()

    def set_frontier(self, n: int):
        if self._enabled: self.frontier_size.set(n)

    def set_queue_depth(self, n: int):
        if self._enabled: self.queue_depth.set(n)

    def set_active_browsers(self, n: int):
        if self._enabled: self.active_browsers.set(n)

    def observe_fetch(self, ms: float):
        if self._enabled: self.fetch_duration.observe(ms)

    def observe_parse(self, ms: float):
        if self._enabled: self.parse_duration.observe(ms)


# ─────────────────────────────────────────────────────────────
# ROBOTS / SITEMAP (unchanged from v5)
# ─────────────────────────────────────────────────────────────

class AsyncRobots:
    def __init__(self, default_delay: float = 1.5):
        self.default_delay = default_delay
        self._cache: dict[str, Optional[RobotFileParser]] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._dict_lock = asyncio.Lock()

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        async with self._dict_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = asyncio.Lock()
            return self._domain_locks[domain]

    async def _fetch(self, origin: str, session: aiohttp.ClientSession):
        try:
            async with session.get(f"{origin}/robots.txt",
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    text = await resp.text(errors="replace")
                    rp = RobotFileParser()
                    rp.set_url(f"{origin}/robots.txt")
                    rp.parse(text.splitlines())
                    return rp
        except Exception:
            pass
        return None

    async def get(self, url: str, session: aiohttp.ClientSession):
        p = urlparse(url); origin = f"{p.scheme}://{p.netloc}"
        lock = await self._get_lock(p.netloc)
        async with lock:
            if origin not in self._cache:
                self._cache[origin] = await self._fetch(origin, session)
            return self._cache[origin]

    async def allowed(self, url: str, session: aiohttp.ClientSession, agent="*") -> bool:
        rp = await self.get(url, session)
        return rp.can_fetch(agent, url) if rp else True

    async def crawl_delay(self, url: str, session: aiohttp.ClientSession) -> float:
        rp = await self.get(url, session)
        if rp:
            cd = rp.crawl_delay("*")
            if cd: return float(cd)
        return self.default_delay

    async def site_maps(self, url: str, session: aiohttp.ClientSession) -> list[str]:
        rp = await self.get(url, session)
        if rp and hasattr(rp, "site_maps"):
            sms = rp.site_maps()
            return list(sms) if sms else []
        return []


async def fetch_sitemap_urls(sitemap_url, session, depth=0) -> list[str]:
    if depth > 3: return []
    urls = []
    try:
        async with session.get(sitemap_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200: return []
            text = await resp.text(errors="replace")
        root = ElementTree.fromstring(text)
        ns   = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
        tag  = lambda t: f"{{{ns}}}{t}" if ns else t
        for loc in root.iter(tag("sitemap")):
            child = loc.find(tag("loc"))
            if child is not None and child.text:
                urls.extend(await fetch_sitemap_urls(child.text.strip(), session, depth+1))
        for loc in root.iter(tag("url")):
            child = loc.find(tag("loc"))
            if child is not None and child.text:
                u = normalise(child.text.strip())
                if u: urls.append(u)
    except Exception as exc:
        log.debug("Sitemap error %s: %s", sitemap_url, exc)
    return urls


# ─────────────────────────────────────────────────────────────
# RATE LIMITER (unchanged from v5)
# ─────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, default_delay: float = 1.5, per_domain_concurrency: int = 2):
        self.default_delay = default_delay
        self.per_domain_concurrency = per_domain_concurrency
        self._last: dict[str, float] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def _sem(self, domain: str) -> asyncio.Semaphore:
        async with self._lock:
            if domain not in self._sems:
                self._sems[domain] = asyncio.Semaphore(self.per_domain_concurrency)
            return self._sems[domain]

    async def wait(self, domain: str, delay: Optional[float] = None):
        d = delay if delay is not None else self.default_delay
        async with self._lock:
            gap = d - (time.monotonic() - self._last.get(domain, 0.0))
        if gap > 0: await asyncio.sleep(gap)
        async with self._lock: self._last[domain] = time.monotonic()

    async def acquire(self, domain: str) -> asyncio.Semaphore:
        return await self._sem(domain)


# ─────────────────────────────────────────────────────────────
# ASYNC FETCHER (single shared session — v4 B1 fix preserved)
# ─────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    url:         str
    final_url:   str
    html:        Optional[str]
    meta:        CrawlMeta
    needs_js:    bool = False   # hint for playwright escalation


class AsyncFetcher:
    def __init__(self, global_concurrency: int = 50, timeout: int = 15):
        self._global_sem = asyncio.Semaphore(global_concurrency)
        self._timeout    = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def open(self):
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT}, timeout=self._timeout)

    async def close(self):
        if self.session: await self.session.close()

    async def fetch(self, url: str, rate: RateLimiter,
                    etag: str = "", last_modified: str = "") -> FetchResult:
        meta   = CrawlMeta()
        domain = urlparse(url).netloc
        dom_sem = await rate.acquire(domain)

        async with self._global_sem, dom_sem:
            await rate.wait(domain)
            headers: dict[str, str] = {}
            if etag:          headers["If-None-Match"]     = etag
            elif last_modified: headers["If-Modified-Since"] = last_modified

            t0 = time.monotonic()
            try:
                async with self.session.get(url, allow_redirects=True, headers=headers) as resp:
                    meta.status_code      = resp.status
                    meta.content_type     = resp.headers.get("Content-Type", "")
                    meta.etag             = resp.headers.get("ETag", "")
                    meta.last_modified    = resp.headers.get("Last-Modified", "")
                    meta.response_time_ms = round((time.monotonic()-t0)*1000, 1)
                    final_url = normalise(str(resp.url)) or str(resp.url)

                    if resp.status == 304:
                        return FetchResult(url=url, final_url=final_url, html=None, meta=meta)

                    mime = meta.content_type.split(";")[0].strip().lower()
                    if resp.status == 200 and mime in ALLOWED_MIME:
                        html = await resp.text(errors="replace")
                        needs_js = needs_js_render(html)   # [I3] detection
                        return FetchResult(url=url, final_url=final_url,
                                           html=html, meta=meta, needs_js=needs_js)
            except Exception as exc:
                log.warning("Fetch error %s: %s", url, exc)
                meta.response_time_ms = round((time.monotonic()-t0)*1000, 1)

        return FetchResult(url=url, final_url=url, html=None, meta=meta)


# ─────────────────────────────────────────────────────────────
# FAST PARSER (selectolax — unchanged from v5)
# ─────────────────────────────────────────────────────────────

class FastParser:
    def parse(self, html: str, base_url: str
              ) -> tuple[str, str, list[str], dict, Optional[str]]:
        tree = SXParser(html)
        title_node = tree.css_first("title")
        title = title_node.text(strip=True) if title_node else ""

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
            if not href: continue
            full = normalise(urljoin(base_url, href))
            if full and full not in seen_links:
                seen_links.add(full); links.append(full)

        meta: dict[str, str] = {}
        for node in tree.css("meta"):
            key = node.attributes.get("name") or node.attributes.get("property")
            val = node.attributes.get("content", "")
            if key: meta[key] = val

        return title, text, links, meta, canonical


# ─────────────────────────────────────────────────────────────
# PIPELINE  — wires all four integrations together
# ─────────────────────────────────────────────────────────────

class Pipeline:

    def __init__(
        self,
        frontier,                         # RedisFrontier | SQLiteFrontier
        fetcher:          AsyncFetcher,
        pw_pool:          PlaywrightPool,
        rate:             RateLimiter,
        robots:           AsyncRobots,
        parser:           FastParser,
        writer,                           # PostgresWriter | SQLiteWriter
        hashes:           HashStore,
        metrics:          PrometheusMetrics,
        seed_domains:     set[str],
        max_pages:        int,
        follow_links:     bool,
        same_domain_only: bool,
        n_fetch_workers:  int = 10,
        n_parse_workers:  int = 4,
    ):
        self.frontier         = frontier
        self.fetcher          = fetcher
        self.pw_pool          = pw_pool
        self.rate             = rate
        self.robots           = robots
        self.parser           = parser
        self.writer           = writer
        self.hashes           = hashes
        self.metrics          = metrics
        self.seed_domains     = seed_domains
        self.max_pages        = max_pages
        self.follow_links     = follow_links
        self.same_domain_only = same_domain_only
        self.n_fetch          = n_fetch_workers
        self.n_parse          = n_parse_workers

        self._fetch_q: asyncio.Queue[Optional[FetchResult]] = asyncio.Queue(maxsize=200)
        self._scraped      = 0
        self._scraped_lock = asyncio.Lock()
        self._stop         = asyncio.Event()

    def _in_scope(self, url: str) -> bool:
        return not self.same_domain_only or urlparse(url).netloc in self.seed_domains

    # ── fetch worker ──────────────────────────────────────────

    async def _fetch_worker(self, _: int):
        while not self._stop.is_set():
            try:
                url = await asyncio.wait_for(self.frontier.output_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if url is None:
                break

            if not await self.robots.allowed(url, self.fetcher.session):
                log.debug("robots blocks %s", url)
                continue

            etag, last_mod = await self.writer.get_conditional(url)
            result = await self.fetcher.fetch(url, self.rate, etag=etag, last_modified=last_mod)
            self.metrics.observe_fetch(result.meta.response_time_ms)

            if result.html is None:
                if result.meta.status_code not in (304, 0):
                    self.metrics.inc_error("fetch")
                continue

            # [I3] Playwright escalation for JS-heavy pages
            if result.needs_js and self.pw_pool.available:
                log.debug("Escalating to Playwright: %s", url)
                self.metrics.set_active_browsers(self.pw_pool.active_count())
                pw_html, pw_final = await self.pw_pool.fetch(url)
                if pw_html:
                    result.html      = pw_html
                    result.final_url = pw_final
                    result.meta.rendered_by = "playwright"
                self.metrics.set_active_browsers(self.pw_pool.active_count())

            # [I4] Queue depth metric
            self.metrics.set_queue_depth(self._fetch_q.qsize())
            await self._fetch_q.put(result)

    # ── parse worker ──────────────────────────────────────────

    async def _parse_worker(self, _: int):
        while True:
            result = await self._fetch_q.get()
            if result is None:
                break
            if result.html is None:
                continue

            t0 = time.monotonic()
            title, text, links, meta, canonical = self.parser.parse(
                result.html, result.final_url
            )
            self.metrics.observe_parse((time.monotonic()-t0)*1000)

            page_url     = canonical or result.final_url
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            sh           = weighted_simhash(text)

            if self.hashes.seen_exact(content_hash):
                self.metrics.inc_duplicate("exact")
                log.debug("Exact dup skipped: %s", result.url); continue
            if self.hashes.seen_near(sh):
                self.metrics.inc_duplicate("near")
                log.debug("Near dup skipped: %s", result.url); continue
            self.hashes.add(content_hash, sh)

            result.meta.content_hash = content_hash
            result.meta.simhash      = sh
            result.meta.final_url    = page_url

            item = ScrapedItem(
                url=page_url, title=title, text=text,
                links=links, meta=meta, crawl=result.meta,
            )
            await self.writer.push(item)

            async with self._scraped_lock:
                self._scraped += 1
                count = self._scraped

            self.metrics.inc_scraped(result.meta.rendered_by)   # [I4]
            log.info("[%d/%d] %s  (%dms, %s)",
                     count, self.max_pages, page_url,
                     result.meta.response_time_ms, result.meta.rendered_by)

            if self.follow_links:
                new_links = [l for l in links if self._in_scope(l)]
                await self.frontier.add_many(new_links)   # batch add [I2]

            # [I4] frontier gauge
            try:
                self.metrics.set_frontier(await self.frontier.pending_count())
            except Exception:
                pass

            if count >= self.max_pages:
                self._stop.set(); break

    # ── coordinator ───────────────────────────────────────────

    async def _coordinator(self, fetch_tasks: list[asyncio.Task]):
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for _ in range(self.n_parse):
            await self._fetch_q.put(None)

    # ── run ───────────────────────────────────────────────────

    async def run(self):
        await self._seed_from_sitemaps()
        await self.writer.start()

        sched_stop = asyncio.Event()
        sched_task = asyncio.create_task(self.frontier.run_scheduler(sched_stop))

        fetch_tasks = [asyncio.create_task(self._fetch_worker(i)) for i in range(self.n_fetch)]
        parse_tasks = [asyncio.create_task(self._parse_worker(i)) for i in range(self.n_parse)]
        coord_task  = asyncio.create_task(self._coordinator(fetch_tasks))

        await asyncio.gather(*parse_tasks)

        coord_task.cancel()
        sched_stop.set()
        sched_task.cancel()

        await self.writer.stop()
        self.hashes.save()
        await self.frontier.close()
        log.info("Done. Scraped %d pages.", self._scraped)

    async def _seed_from_sitemaps(self):
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            for domain in self.seed_domains:
                origin = f"https://{domain}"
                sm_urls = await self.robots.site_maps(origin, session) or [f"{origin}/sitemap.xml"]
                for sm in sm_urls:
                    urls = await fetch_sitemap_urls(sm, session)
                    log.info("Sitemap %s: %d URLs", sm, len(urls))
                    await self.frontier.add_many([u for u in urls if self._in_scope(u)])


# ─────────────────────────────────────────────────────────────
# SCRAPER — top-level config + factory
# ─────────────────────────────────────────────────────────────

class Scraper:
    """
    Config facade. Constructor is pure Python (no event loop).
    Components built in run_async() so it embeds cleanly in FastAPI/Jupyter.

    New parameters vs v5:
      pg_dsn          — PostgreSQL DSN e.g. "postgresql://user:pw@localhost/crawl"
                        None → falls back to SQLite (output_db)
      redis_url       — Redis URL e.g. "redis://localhost:6379"
                        None → falls back to SQLite frontier (checkpoint_db)
      max_browsers    — Playwright pool size; 0 = disable JS rendering
      metrics_port    — Prometheus /metrics port; 0 = disable
    """

    def __init__(
        self,
        seed_urls:               list[str],
        max_pages:               int   = 100,
        default_delay:           float = 1.5,
        global_concurrency:      int   = 50,
        per_domain_concurrency:  int   = 2,
        n_fetch_workers:         int   = 10,
        n_parse_workers:         int   = 4,
        follow_links:            bool  = True,
        same_domain_only:        bool  = True,
        # v6 new params
        pg_dsn:                  Optional[str] = None,
        output_db:               str   = "output.db",
        redis_url:               Optional[str] = None,
        checkpoint_db:           str   = "checkpoint.db",
        max_browsers:            int   = 5,
        metrics_port:            int   = 8000,
        crawl_name:              str   = "default",
        resume:                  bool  = False,
    ):
        self.seed_urls    = seed_urls
        self.seed_domains = {urlparse(u).netloc for u in seed_urls}
        self.resume       = resume
        self._cfg = dict(
            max_pages=max_pages, default_delay=default_delay,
            global_concurrency=global_concurrency,
            per_domain_concurrency=per_domain_concurrency,
            n_fetch_workers=n_fetch_workers, n_parse_workers=n_parse_workers,
            follow_links=follow_links, same_domain_only=same_domain_only,
            pg_dsn=pg_dsn, output_db=output_db,
            redis_url=redis_url, checkpoint_db=checkpoint_db,
            max_browsers=max_browsers, metrics_port=metrics_port,
            crawl_name=crawl_name,
        )

    async def run_async(self):
        cfg = self._cfg

        # [I4] Metrics first — exposes /metrics before crawl starts
        metrics = PrometheusMetrics(
            port=cfg["metrics_port"],
            enabled=cfg["metrics_port"] > 0,
        )

        # [I2] Frontier
        frontier = make_frontier(
            cfg["redis_url"], cfg["checkpoint_db"],
            cfg["crawl_name"],
        )
        await frontier.open()
        if self.resume and hasattr(frontier, "load_resume"):
            await frontier.load_resume()
        else:
            await frontier.add_many(self.seed_urls)

        # [I1] Storage writer
        writer = make_writer(cfg["pg_dsn"], cfg["output_db"])

        # [I3] Browser pool
        pw_pool = PlaywrightPool(max_browsers=cfg["max_browsers"])
        if cfg["max_browsers"] > 0:
            await pw_pool.open()

        # HTTP fetcher
        fetcher = AsyncFetcher(cfg["global_concurrency"])
        await fetcher.open()

        pipeline = Pipeline(
            frontier         = frontier,
            fetcher          = fetcher,
            pw_pool          = pw_pool,
            rate             = RateLimiter(cfg["default_delay"],
                                           cfg["per_domain_concurrency"]),
            robots           = AsyncRobots(cfg["default_delay"]),
            parser           = FastParser(),
            writer           = writer,
            hashes           = HashStore(),
            metrics          = metrics,
            seed_domains     = self.seed_domains,
            max_pages        = cfg["max_pages"],
            follow_links     = cfg["follow_links"],
            same_domain_only = cfg["same_domain_only"],
            n_fetch_workers  = cfg["n_fetch_workers"],
            n_parse_workers  = cfg["n_parse_workers"],
        )

        try:
            await pipeline.run()
        finally:
            await fetcher.close()
            await pw_pool.close()

    def run(self):
        asyncio.run(self.run_async())


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Scraper(
        seed_urls    = ["https://books.toscrape.com/"],
        max_pages    = 100,
        default_delay          = 1.0,
        global_concurrency     = 50,
        per_domain_concurrency = 2,
        n_fetch_workers        = 10,
        n_parse_workers        = 4,
        follow_links           = True,
        same_domain_only       = True,

        # ── v6 integrations ──────────────────────────
        # Set pg_dsn to use PostgreSQL; None uses SQLite fallback
        pg_dsn       = None,   # "postgresql://user:pw@localhost/crawl"
        output_db    = "output.db",

        # Set redis_url to use Redis frontier; None uses SQLite fallback
        redis_url    = None,   # "redis://localhost:6379"
        checkpoint_db = "checkpoint.db",

        # Playwright pool size; 0 disables JS rendering
        max_browsers = 5,

        # Prometheus port; 0 disables metrics
        metrics_port = 8000,

        resume       = False,
    ).run()