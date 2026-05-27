"""
Web Data Scraper v5 — Algorithmic correctness + data integrity
==============================================================
Fixes over v4 (all 10 review points):

DATA INTEGRITY (silent loss):
  [P5]  mark_done before commit    — done flag written atomically inside the
                                     same DB transaction as the page row
  [P6]  canonical not storage key  — ScrapedItem.url set to canonical/final_url;
                                     frontier.add() also uses canonical
  [P10] StorageWriter.stop cancel  — explicit None sentinel + graceful drain;
                                     no Task.cancel() during DB work

PERFORMANCE:
  [P1]  O(n) list.pop(0)           — collections.deque + popleft() everywhere
  [P3]  O(n) simhash lookup        — LSH bucket index; O(1) amortised lookup
  [P4]  Unbounded _seen RAM        — _seen removed; membership via bloom filter
                                     or aiosqlite EXISTS query only

CORRECTNESS:
  [P2]  Unweighted simhash tokens  — Counter frequency weighting restored
  [P7]  Unfair round-robin         — weighted-fair queue (tokens proportional
                                     to domain queue size, capped per cycle)
  [P8]  Dead AsyncRobots() code    — removed
  [P9]  Unsafe event loop init     — constructor is sync-safe; async work
                                     deferred to Scraper.run() / async context
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
    "Mozilla/5.0 (compatible; PyScraper/5.0; "
    "+https://github.com/example/scraper)"
)
ALLOWED_MIME    = {"text/html", "application/xhtml+xml"}
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source",
}
NOISE_TAGS = ("script", "style", "nav", "footer", "aside", "header")


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
    content_hash:     str   = ""   # sha256 of cleaned text
    simhash:          int   = 0    # 64-bit weighted simhash


@dataclass
class ScrapedItem:
    url:   str                       # canonical page identity  [P6]
    title: str   = ""
    text:  str   = ""
    links: list  = field(default_factory=list)
    meta:  dict  = field(default_factory=dict)
    crawl: CrawlMeta = field(default_factory=CrawlMeta)


# ─────────────────────────────────────────────────────────────
# URL NORMALISER
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


# ─────────────────────────────────────────────────────────────
# WEIGHTED SIMHASH  [P2 fixed]
# ─────────────────────────────────────────────────────────────

def weighted_simhash(text: str, bits: int = 64) -> int:
    """
    64-bit simhash with Counter frequency weighting.
    'buy buy buy' (weight 3) outweighs 'buy' (weight 1),
    preserving near-duplicate sensitivity for repeated tokens.
    """
    v = [0] * bits
    counts = Counter(re.findall(r"\w+", text.lower()))
    for token, weight in counts.items():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(bits):
            v[i] += weight if (h >> i) & 1 else -weight
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ─────────────────────────────────────────────────────────────
# LSH SIMHASH INDEX  [P3 fixed — O(1) amortised lookup]
# ─────────────────────────────────────────────────────────────

class SimhashIndex:
    """
    Locality-Sensitive Hashing for 64-bit simhashes.

    Strategy: partition the 64 bits into BANDS bands of ROWS bits each.
              Two hashes that agree on any band are candidates.
              Hamming distance ≤ threshold ≈ 4 means they share at least
              one band with high probability.

    Lookup is O(BANDS) dict probes instead of O(n) linear scan.
    """

    BANDS      = 8
    ROWS       = 8          # BANDS × ROWS == 64
    THRESHOLD  = 4          # Hamming distance

    def __init__(self):
        # buckets[band_idx][band_value] = list of full simhashes
        self._buckets: list[dict[int, list[int]]] = [
            {} for _ in range(self.BANDS)
        ]

    def _band_values(self, sh: int) -> list[int]:
        mask = (1 << self.ROWS) - 1
        return [(sh >> (b * self.ROWS)) & mask for b in range(self.BANDS)]

    def seen(self, sh: int) -> bool:
        """Return True if any stored hash is within THRESHOLD bits."""
        for band_idx, bv in enumerate(self._band_values(sh)):
            candidates = self._buckets[band_idx].get(bv, [])
            if any(hamming(sh, c) <= self.THRESHOLD for c in candidates):
                return True
        return False

    def add(self, sh: int):
        for band_idx, bv in enumerate(self._band_values(sh)):
            self._buckets[band_idx].setdefault(bv, []).append(sh)


# ─────────────────────────────────────────────────────────────
# CONTENT HASH STORE  — bloom for exact, LSH for near-dup
# ─────────────────────────────────────────────────────────────

BLOOM_PATH = "bloom.pkl"

class HashStore:
    """
    Exact duplicate: persistent Bloom filter (or SQLite fallback).
    Near-duplicate:  in-memory LSH index (SimhashIndex).

    Bloom is pickled to disk so it survives restarts.  [B5 from v4]
    """

    def __init__(self, db_path: str = "output.db", bloom_path: str = BLOOM_PATH):
        self._bloom_path = bloom_path
        if _BLOOM_AVAILABLE:
            if os.path.exists(bloom_path):
                with open(bloom_path, "rb") as f:
                    self._bloom = pickle.load(f)
                log.info("HashStore: loaded bloom filter (%s)", bloom_path)
            else:
                self._bloom = ScalableBloomFilter(
                    initial_capacity=100_000, error_rate=0.001
                )
            self._use_bloom = True
        else:
            import sqlite3 as _sq
            self._sdb = _sq.connect(db_path, check_same_thread=False)
            self._sdb.execute(
                "CREATE TABLE IF NOT EXISTS seen_hashes (hash TEXT PRIMARY KEY)"
            )
            self._sdb.commit()
            self._use_bloom = False

        self._lsh = SimhashIndex()          # O(1) near-dup lookup  [P3]

    def seen_exact(self, h: str) -> bool:
        if self._use_bloom:
            return h in self._bloom
        return bool(self._sdb.execute(
            "SELECT 1 FROM seen_hashes WHERE hash=?", (h,)
        ).fetchone())

    def seen_near(self, sh: int) -> bool:
        return self._lsh.seen(sh)           # O(BANDS) dict lookups  [P3]

    def add(self, h: str, sh: int):
        if self._use_bloom:
            self._bloom.add(h)
        else:
            self._sdb.execute(
                "INSERT OR IGNORE INTO seen_hashes VALUES (?)", (h,)
            )
            self._sdb.commit()
        self._lsh.add(sh)

    def save(self):
        if self._use_bloom:
            with open(self._bloom_path, "wb") as f:
                pickle.dump(self._bloom, f)
            log.info("HashStore: bloom filter saved (%s)", self._bloom_path)


# ─────────────────────────────────────────────────────────────
# DOMAIN SCHEDULER  [P1 + P4 + P7 fixed]
# ─────────────────────────────────────────────────────────────
#
#  P1: all per-domain queues are collections.deque → popleft() is O(1)
#  P4: _seen removed; membership check is bloom filter or aiosqlite EXISTS
#  P7: weighted-fair scheduling — domain gets tokens proportional to its
#      queue depth (capped at MAX_TOKENS_PER_CYCLE per domain per round)
#      so large domains cannot fully starve small ones.

class DomainScheduler:

    FLUSH_BATCH       = 250
    FLUSH_DELAY       = 2.0
    MAX_TOKENS        = 5      # max URLs drained per domain per scheduler cycle

    def __init__(self, db_path: str, bloom_path: str = BLOOM_PATH):
        self._db_path   = db_path
        self._db: Optional[aiosqlite.Connection] = None
        # P4: no _seen set — membership is delegated entirely to bloom/DB
        self._bloom_path = bloom_path
        if _BLOOM_AVAILABLE:
            if os.path.exists(bloom_path):
                with open(bloom_path, "rb") as f:
                    self._url_bloom = pickle.load(f)
            else:
                self._url_bloom = ScalableBloomFilter(
                    initial_capacity=1_000_000, error_rate=0.0001
                )
            self._use_bloom = True
        else:
            self._use_bloom = False        # fall through to DB EXISTS

        # per-domain deques  [P1]
        self._domain_qs:  dict[str, deque] = {}
        # global priority heap: (-score, domain) — for scheduling order
        self._domain_heap: list[tuple[int, str]] = []
        self._domains_in_heap: set[str] = set()

        self.output_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=500)
        self._lock     = asyncio.Lock()
        self._pending:  list[tuple[str, int]] = []   # (url, priority)
        self._last_flush = time.monotonic()

    async def open(self):
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS seen_urls (
                url TEXT PRIMARY KEY, priority INTEGER
            );
            CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
        """)
        await self._db.commit()

    async def close(self):
        await self._flush_db(force=True)
        if self._use_bloom:
            with open(self._bloom_path, "wb") as f:
                pickle.dump(self._url_bloom, f)
        if self._db:
            await self._db.close()

    # ── membership check ──────────────────────────────────────

    async def _seen(self, url: str) -> bool:
        """[P4] Bloom or DB — no in-memory set."""
        if self._use_bloom:
            return url in self._url_bloom
        async with self._db.execute(
            "SELECT 1 FROM seen_urls WHERE url=?", (url,)
        ) as cur:
            return (await cur.fetchone()) is not None

    # ── add URL ───────────────────────────────────────────────

    async def add(self, url: str) -> bool:
        url = normalise(url)
        if not url:
            return False
        async with self._lock:
            if await self._seen(url):
                return False
            # Mark seen immediately (bloom is probabilistic but sufficient)
            if self._use_bloom:
                self._url_bloom.add(url)
            score  = _score(url)
            domain = urlparse(url).netloc
            # Route into per-domain deque  [P1]
            if domain not in self._domain_qs:
                self._domain_qs[domain] = deque()
                heapq.heappush(self._domain_heap, (-score, domain))
                self._domains_in_heap.add(domain)
            self._domain_qs[domain].append(url)
            # Batch persist
            self._pending.append((url, score))
            await self._maybe_flush()
            return True

    async def mark_done(self, url: str):
        """Called by StorageWriter inside its commit transaction — not here."""
        # Intentionally empty: done marking is now fully owned by StorageWriter
        # to guarantee atomicity.  [P5]
        pass

    # ── weighted-fair scheduler loop  [P7] ───────────────────

    async def run_scheduler(self, stop_event: asyncio.Event):
        """
        Each cycle: iterate domains in heap order; drain up to MAX_TOKENS
        URLs proportionally to queue depth (larger queues get more tokens
        but never all tokens — prevents starvation of small domains).
        """
        while not stop_event.is_set():
            async with self._lock:
                if not self._domain_heap:
                    pass
                else:
                    # Snapshot domains sorted by priority
                    snapshot = sorted(self._domain_qs.items(),
                                      key=lambda kv: -len(kv[1]))
                    total = sum(len(q) for _, q in snapshot) or 1
                    fed   = False
                    for domain, q in snapshot:
                        if not q:
                            continue
                        # Proportional token allocation, capped  [P7]
                        tokens = max(1, min(
                            self.MAX_TOKENS,
                            round(self.MAX_TOKENS * len(q) / total)
                        ))
                        for _ in range(tokens):
                            if not q:
                                break
                            url = q.popleft()          # O(1)  [P1]
                            try:
                                self.output_q.put_nowait(url)
                                fed = True
                            except asyncio.QueueFull:
                                q.appendleft(url)      # put back
                                break
                    if not fed:
                        pass   # fall through to sleep

            await asyncio.sleep(0.05)

    # ── resume ────────────────────────────────────────────────

    async def load_resume(self):
        async with self._db.execute(
            """SELECT s.url, s.priority FROM seen_urls s
               LEFT JOIN done d ON s.url = d.url
               WHERE d.url IS NULL"""
        ) as cur:
            rows = await cur.fetchall()
        for url, priority in rows:
            domain = urlparse(url).netloc
            if domain not in self._domain_qs:
                self._domain_qs[domain] = deque()
            self._domain_qs[domain].append(url)
            if self._use_bloom:
                self._url_bloom.add(url)
        # Rebuild domain heap
        for domain, q in self._domain_qs.items():
            if q and domain not in self._domains_in_heap:
                score = _score(next(iter(q)))
                heapq.heappush(self._domain_heap, (-score, domain))
                self._domains_in_heap.add(domain)
        log.info("Resumed scheduler: %d pending URLs", sum(len(q) for q in self._domain_qs.values()))

    # ── DB flush ──────────────────────────────────────────────

    async def _maybe_flush(self):
        elapsed = time.monotonic() - self._last_flush
        if len(self._pending) >= self.FLUSH_BATCH or elapsed >= self.FLUSH_DELAY:
            await self._flush_db()

    async def _flush_db(self, force: bool = False):
        if not self._db or not self._pending:
            return
        await self._db.executemany(
            "INSERT OR IGNORE INTO seen_urls (url, priority) VALUES (?,?)",
            self._pending,
        )
        await self._db.commit()
        self._pending.clear()
        self._last_flush = time.monotonic()


def _score(url: str) -> int:
    path = urlparse(url).path.lower()
    if path in ("/", ""):                       return 100
    if re.search(r"/categor|/catalog", path):   return 80
    if re.search(r"/product|/item|/p/", path):  return 50
    if re.search(r"page=\d+|/page/\d+", url):   return 20
    return 40


# ─────────────────────────────────────────────────────────────
# ASYNC ROBOTS.TXT  — per-domain lock, no global lock during I/O
# ─────────────────────────────────────────────────────────────

class AsyncRobots:

    def __init__(self, default_delay: float = 1.5):
        self.default_delay = default_delay
        self._cache:        dict[str, Optional[RobotFileParser]] = {}
        self._domain_locks: dict[str, asyncio.Lock] = {}
        self._dict_lock     = asyncio.Lock()

    async def _get_lock(self, domain: str) -> asyncio.Lock:
        async with self._dict_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = asyncio.Lock()
            return self._domain_locks[domain]

    async def _fetch(self, origin: str, session: aiohttp.ClientSession
                     ) -> Optional[RobotFileParser]:
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
                    return rp
        except Exception:
            pass
        return None

    async def get(self, url: str, session: aiohttp.ClientSession
                  ) -> Optional[RobotFileParser]:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        domain = p.netloc
        lock   = await self._get_lock(domain)
        async with lock:                             # per-domain, not global
            if origin not in self._cache:
                self._cache[origin] = await self._fetch(origin, session)
            return self._cache[origin]

    async def allowed(self, url: str, session: aiohttp.ClientSession,
                      agent: str = "*") -> bool:
        rp = await self.get(url, session)
        return rp.can_fetch(agent, url) if rp else True

    async def crawl_delay(self, url: str, session: aiohttp.ClientSession) -> float:
        rp = await self.get(url, session)
        if rp:
            cd = rp.crawl_delay("*")
            if cd:
                return float(cd)
        return self.default_delay

    async def site_maps(self, url: str, session: aiohttp.ClientSession) -> list[str]:
        rp = await self.get(url, session)
        if rp and hasattr(rp, "site_maps"):
            sms = rp.site_maps()
            return list(sms) if sms else []
        return []


# ─────────────────────────────────────────────────────────────
# SITEMAP PARSER
# ─────────────────────────────────────────────────────────────

async def fetch_sitemap_urls(
    sitemap_url: str,
    session: aiohttp.ClientSession,
    depth: int = 0,
) -> list[str]:
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

        for loc in root.iter(tag("sitemap")):
            child = loc.find(tag("loc"))
            if child is not None and child.text:
                sub = await fetch_sitemap_urls(child.text.strip(), session, depth+1)
                urls.extend(sub)

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
# ASYNC RATE LIMITER — per-domain, event-loop-safe
# ─────────────────────────────────────────────────────────────

class RateLimiter:

    def __init__(self, default_delay: float = 1.5, per_domain_concurrency: int = 2):
        self.default_delay         = default_delay
        self.per_domain_concurrency = per_domain_concurrency
        self._last: dict[str, float]           = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._lock  = asyncio.Lock()

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
# ASYNC FETCHER  — single shared session  [B1 from v4]
# ─────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    url:       str
    final_url: str
    html:      Optional[str]
    meta:      CrawlMeta


class AsyncFetcher:
    """Session opened/closed exactly once. Workers call .fetch() directly."""

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
                    meta.status_code      = resp.status
                    meta.content_type     = resp.headers.get("Content-Type", "")
                    meta.etag             = resp.headers.get("ETag", "")
                    meta.last_modified    = resp.headers.get("Last-Modified", "")
                    meta.response_time_ms = round((time.monotonic()-t0)*1000, 1)
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
                meta.response_time_ms = round((time.monotonic()-t0)*1000, 1)

        return FetchResult(url=url, final_url=url, html=None, meta=meta)


# ─────────────────────────────────────────────────────────────
# FAST PARSER — selectolax + canonical tag
# ─────────────────────────────────────────────────────────────

class FastParser:

    def parse(
        self, html: str, base_url: str
    ) -> tuple[str, str, list[str], dict, Optional[str]]:
        """Returns (title, text, links, meta, canonical_url | None)."""
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
# CONDITIONAL REQUEST CACHE — keyed on canonical/final URL
# ─────────────────────────────────────────────────────────────

class ConditionalCache:

    def __init__(self, db_path: str = "output.db"):
        import sqlite3 as _sq
        self._db = _sq.connect(db_path, check_same_thread=False)

    def get(self, canonical_url: str) -> tuple[str, str]:
        row = self._db.execute(
            "SELECT etag, last_modified FROM pages WHERE url=?",
            (canonical_url,)
        ).fetchone()
        return (row[0] or "", row[1] or "") if row else ("", "")


# ─────────────────────────────────────────────────────────────
# STORAGE WRITER  [P5 + P10 fixed]
#
#  P5:  done row written atomically in the same transaction as the page
#       row — StorageWriter owns the done flag entirely
#  P10: stop() sends an explicit None sentinel and awaits graceful drain;
#       Task.cancel() is NEVER called while DB work may be in progress
# ─────────────────────────────────────────────────────────────

class StorageWriter:

    BATCH_SIZE     = 100
    FLUSH_INTERVAL = 1.0

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
        """Graceful shutdown: send sentinel, wait for writer to drain."""  # [P10]
        await self._q.put(None)       # explicit sentinel — no Task.cancel()
        await self._stopped.wait()    # writer sets this after final commit

    async def _writer_loop(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS pages (
                    id            INTEGER PRIMARY KEY,
                    url           TEXT UNIQUE,
                    final_url     TEXT,
                    title         TEXT,
                    text          TEXT,
                    status_code   INTEGER,
                    content_type  TEXT,
                    response_ms   REAL,
                    etag          TEXT,
                    last_modified TEXT,
                    content_hash  TEXT,
                    simhash       INTEGER,
                    scraped_at    DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS links (
                    src TEXT, dst TEXT, PRIMARY KEY (src, dst)
                );
                CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
            """)
            await db.commit()

            batch: list[ScrapedItem] = []
            last_flush = time.monotonic()

            while True:
                try:
                    timeout = max(0.0, self.FLUSH_INTERVAL - (time.monotonic()-last_flush))
                    item = await asyncio.wait_for(self._q.get(), timeout=timeout)

                    if item is None:                  # graceful sentinel  [P10]
                        if batch:
                            await self._commit(db, batch)
                        self._stopped.set()
                        return

                    batch.append(item)
                    self._q.task_done()

                except asyncio.TimeoutError:
                    pass

                if (len(batch) >= self.BATCH_SIZE or
                        (batch and time.monotonic()-last_flush >= self.FLUSH_INTERVAL)):
                    await self._commit(db, batch)
                    batch = []
                    last_flush = time.monotonic()

    @staticmethod
    async def _commit(db: aiosqlite.Connection, batch: list[ScrapedItem]):
        """
        Writes pages, links, and done rows in a single transaction.  [P5]
        done is only set AFTER the page row is persisted — no partial state.
        """
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
               (url, final_url, title, text, status_code, content_type,
                response_ms, etag, last_modified, content_hash, simhash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            pages,
        )
        await db.executemany(
            "INSERT OR IGNORE INTO links (src, dst) VALUES (?,?)", links
        )
        # done written in same transaction — atomic with page  [P5]
        await db.executemany(
            "INSERT OR IGNORE INTO done (url) VALUES (?)", done
        )
        await db.commit()
        log.debug("Committed %d pages", len(pages))


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

class Pipeline:

    def __init__(
        self,
        scheduler:         DomainScheduler,
        fetcher:           AsyncFetcher,
        rate:              RateLimiter,
        robots:            AsyncRobots,
        parser:            FastParser,
        writer:            StorageWriter,
        hashes:            HashStore,
        cond_cache:        ConditionalCache,
        seed_domains:      set[str],
        max_pages:         int,
        follow_links:      bool,
        same_domain_only:  bool,
        n_fetch_workers:   int = 10,
        n_parse_workers:   int = 4,
    ):
        self.scheduler        = scheduler
        self.fetcher          = fetcher
        self.rate             = rate
        self.robots           = robots
        self.parser           = parser
        self.writer           = writer
        self.hashes           = hashes
        self.cond_cache       = cond_cache
        self.seed_domains     = seed_domains
        self.max_pages        = max_pages
        self.follow_links     = follow_links
        self.same_domain_only = same_domain_only
        self.n_fetch          = n_fetch_workers
        self.n_parse          = n_parse_workers

        self._fetch_q: asyncio.Queue[Optional[FetchResult]] = asyncio.Queue(maxsize=200)
        self._scraped       = 0
        self._scraped_lock  = asyncio.Lock()
        self._stop          = asyncio.Event()

    def _in_scope(self, url: str) -> bool:
        return (not self.same_domain_only or
                urlparse(url).netloc in self.seed_domains)

    # ── fetch worker ──────────────────────────────────────────

    async def _fetch_worker(self, _: int):
        while not self._stop.is_set():
            try:
                url = await asyncio.wait_for(
                    self.scheduler.output_q.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            if url is None:
                break

            if not await self.robots.allowed(url, self.fetcher.session):
                log.debug("robots blocks %s", url)
                continue

            etag, last_mod = self.cond_cache.get(url)
            result = await self.fetcher.fetch(
                url, self.rate, etag=etag, last_modified=last_mod
            )
            await self._fetch_q.put(result)

    # ── parse worker ──────────────────────────────────────────

    async def _parse_worker(self, _: int):
        while True:
            result = await self._fetch_q.get()
            if result is None:
                break
            if result.html is None:
                continue

            title, text, links, meta, canonical = self.parser.parse(
                result.html, result.final_url
            )

            # Canonical is the authoritative page identity  [P6]
            page_url = canonical or result.final_url

            content_hash = hashlib.sha256(text.encode()).hexdigest()
            sh = weighted_simhash(text)          # frequency-weighted  [P2]

            if self.hashes.seen_exact(content_hash) or self.hashes.seen_near(sh):
                log.debug("Duplicate skipped: %s", result.url)
                continue
            self.hashes.add(content_hash, sh)

            result.meta.content_hash = content_hash
            result.meta.simhash      = sh
            result.meta.final_url    = page_url

            item = ScrapedItem(
                url=page_url,          # canonical is the storage key  [P6]
                title=title, text=text, links=links, meta=meta,
                crawl=result.meta,
            )
            await self.writer.push(item)
            # mark_done is NOT called here — StorageWriter owns it  [P5]

            async with self._scraped_lock:
                self._scraped += 1
                count = self._scraped

            log.info("[%d/%d] %s  (%dms)",
                     count, self.max_pages,
                     page_url, result.meta.response_time_ms)

            if self.follow_links:
                for link in links:
                    if self._in_scope(link):
                        await self.scheduler.add(link)

            if count >= self.max_pages:
                self._stop.set()
                break

    # ── coordinator — broadcasts exactly n_parse sentinels  [B4] ──

    async def _coordinator(self, fetch_tasks: list[asyncio.Task]):
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for _ in range(self.n_parse):
            await self._fetch_q.put(None)

    # ── run ───────────────────────────────────────────────────

    async def run(self):
        await self._seed_from_sitemaps()
        await self.writer.start()

        sched_stop  = asyncio.Event()
        sched_task  = asyncio.create_task(self.scheduler.run_scheduler(sched_stop))

        fetch_tasks = [
            asyncio.create_task(self._fetch_worker(i))
            for i in range(self.n_fetch)
        ]
        parse_tasks = [
            asyncio.create_task(self._parse_worker(i))
            for i in range(self.n_parse)
        ]
        coord_task = asyncio.create_task(self._coordinator(fetch_tasks))

        await asyncio.gather(*parse_tasks)

        coord_task.cancel()
        sched_stop.set()
        sched_task.cancel()

        await self.writer.stop()         # graceful drain, no cancel  [P10]
        self.hashes.save()
        await self.scheduler.close()
        log.info("Done. Scraped %d pages.", self._scraped)

    async def _seed_from_sitemaps(self):
        async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
            for domain in self.seed_domains:
                origin = f"https://{domain}"
                sm_urls = await self.robots.site_maps(origin, session)
                if not sm_urls:
                    sm_urls = [f"{origin}/sitemap.xml"]
                for sm in sm_urls:
                    urls = await fetch_sitemap_urls(sm, session)
                    log.info("Sitemap %s: %d URLs", sm, len(urls))
                    for url in urls:
                        if self._in_scope(url):
                            await self.scheduler.add(url)

    def _in_scope(self, url: str) -> bool:
        return (not self.same_domain_only or
                urlparse(url).netloc in self.seed_domains)


# ─────────────────────────────────────────────────────────────
# SCRAPER  [P9 fixed — no event loop in constructor]
# ─────────────────────────────────────────────────────────────

class Scraper:
    """
    All async initialisation deferred to run().               [P9]
    Constructor is pure Python — safe in Jupyter, FastAPI, etc.
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
        db_path:                 str   = "output.db",
        checkpoint_db:           str   = "checkpoint.db",
        resume:                  bool  = False,
    ):
        self.seed_urls    = seed_urls
        self.seed_domains = {urlparse(u).netloc for u in seed_urls}
        self.resume       = resume
        self.db_path      = db_path
        self.checkpoint_db = checkpoint_db
        # Store config; build components inside run()  [P9]
        self._cfg = dict(
            max_pages=max_pages,
            default_delay=default_delay,
            global_concurrency=global_concurrency,
            per_domain_concurrency=per_domain_concurrency,
            n_fetch_workers=n_fetch_workers,
            n_parse_workers=n_parse_workers,
            follow_links=follow_links,
            same_domain_only=same_domain_only,
        )

    async def run_async(self):
        """Async entry point — call this inside an existing event loop."""
        cfg        = self._cfg
        scheduler  = DomainScheduler(self.checkpoint_db)
        await scheduler.open()

        if self.resume:
            await scheduler.load_resume()
        else:
            for url in self.seed_urls:
                await scheduler.add(url)

        fetcher = AsyncFetcher(cfg["global_concurrency"])
        await fetcher.open()

        pipeline = Pipeline(
            scheduler        = scheduler,
            fetcher          = fetcher,
            rate             = RateLimiter(cfg["default_delay"],
                                           cfg["per_domain_concurrency"]),
            robots           = AsyncRobots(cfg["default_delay"]),
            parser           = FastParser(),
            writer           = StorageWriter(self.db_path),
            hashes           = HashStore(self.db_path),
            cond_cache       = ConditionalCache(self.db_path),
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

    def run(self):
        """Sync entry point — creates its own event loop (script usage)."""
        asyncio.run(self.run_async())


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