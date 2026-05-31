"""
Frontier module
===============
[F9]   Per-domain crawl budget allocation
[F10]  PageRank-boosted URL priority scoring

Backends:
  RedisFrontier   — ZSET + SET (survives restarts, multi-process safe)
  SQLiteFrontier  — aiosqlite (single-process fallback)
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import math
import os
import pickle
from collections import deque
from typing import Optional
from urllib.parse import urlparse

import aiosqlite

from .utils import normalise, score_url, domain

log = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    _REDIS_OK = True
except ImportError:
    _REDIS_OK = False

try:
    from pybloom_live import ScalableBloomFilter
    _BLOOM_OK = True
except ImportError:
    _BLOOM_OK = False

BLOOM_PATH = "url_bloom.pkl"


# ─────────────────────────────────────────────────────────────
# [F9] DOMAIN BUDGET MANAGER
# ─────────────────────────────────────────────────────────────

class DomainBudget:
    """
    [F9] Per-domain crawl budget.

    Each domain gets a maximum number of pages. When exhausted, URLs
    from that domain are silently dropped. Supports priority tiers:
      tier=1 → 1× budget  (default)
      tier=2 → 5× budget
      tier=3 → 20× budget (primary targets)
    """

    def __init__(
        self,
        default_budget: int = 1000,
        domain_overrides: Optional[dict[str, int]] = None,
        domain_tiers: Optional[dict[str, int]] = None,
        tier_multipliers: Optional[dict[int, int]] = None,
    ):
        self._default   = default_budget
        self._overrides = domain_overrides or {}
        self._tiers     = domain_tiers or {}           # domain → tier int
        self._mults     = tier_multipliers or {1: 1, 2: 5, 3: 20}
        self._counts:    dict[str, int] = {}           # domain → pages scraped

    def budget(self, dom: str) -> int:
        if dom in self._overrides:
            return self._overrides[dom]
        tier = self._tiers.get(dom, 1)
        return self._default * self._mults.get(tier, 1)

    def has_budget(self, dom: str) -> bool:
        return self._counts.get(dom, 0) < self.budget(dom)

    def consume(self, dom: str):
        self._counts[dom] = self._counts.get(dom, 0) + 1

    def remaining(self, dom: str) -> int:
        return max(0, self.budget(dom) - self._counts.get(dom, 0))

    def stats(self) -> dict:
        return {d: {"budget": self.budget(d), "used": c, "remaining": self.remaining(d)}
                for d, c in self._counts.items()}


# ─────────────────────────────────────────────────────────────
# [F10] PAGERANK PRIORITY BOOSTER
# ─────────────────────────────────────────────────────────────

class PageRankBooster:
    """
    [F10] Boosts frontier priority scores with PageRank-derived authority.

    Reads the links table from PostgreSQL / SQLite and runs a lightweight
    in-process PageRank (power iteration). Scores are cached in memory.

    Use:
        booster = PageRankBooster()
        await booster.load_from_db(pool)   # or load_from_sqlite(path)
        priority = booster.boost(url, base_priority)
    """

    def __init__(self, damping: float = 0.85, iterations: int = 20):
        self._d     = damping
        self._iters = iterations
        self._pr:   dict[str, float] = {}   # url → PageRank score

    async def load_from_pg(self, pool) -> int:
        """Load link graph from asyncpg pool and run PageRank."""
        try:
            rows = await pool.fetch("SELECT src, dst FROM links LIMIT 2000000")
            return self._run([(r["src"], r["dst"]) for r in rows])
        except Exception as exc:
            log.warning("PageRankBooster PG load failed: %s", exc)
            return 0

    async def load_from_sqlite(self, db_path: str) -> int:
        """Load link graph from SQLite and run PageRank."""
        try:
            async with aiosqlite.connect(db_path) as db:
                async with db.execute(
                    "SELECT src, dst FROM links LIMIT 2000000"
                ) as cur:
                    rows = await cur.fetchall()
            return self._run([(r[0], r[1]) for r in rows])
        except Exception as exc:
            log.warning("PageRankBooster SQLite load failed: %s", exc)
            return 0

    def _run(self, edges: list[tuple[str, str]]) -> int:
        if not edges:
            return 0
        nodes: set[str] = set()
        out_links: dict[str, list[str]] = {}
        for src, dst in edges:
            nodes.add(src); nodes.add(dst)
            out_links.setdefault(src, []).append(dst)

        n = len(nodes)
        pr = {u: 1.0 / n for u in nodes}

        for _ in range(self._iters):
            new_pr: dict[str, float] = {}
            for u in nodes:
                rank = (1 - self._d) / n
                for v in nodes:
                    out = out_links.get(v, [])
                    if u in out and out:
                        rank += self._d * pr[v] / len(out)
                new_pr[u] = rank
            pr = new_pr

        # Normalise to [0, 100]
        max_pr = max(pr.values()) or 1
        self._pr = {u: (v / max_pr) * 100 for u, v in pr.items()}
        log.info("PageRankBooster: computed for %d nodes", n)
        return n

    def boost(self, url: str, base_priority: int) -> int:
        """Return priority boosted by PageRank (capped at 200)."""
        pr_score = self._pr.get(url, 0)
        return min(200, base_priority + int(pr_score * 0.5))


# ─────────────────────────────────────────────────────────────
# REDIS FRONTIER
# ─────────────────────────────────────────────────────────────

class RedisFrontier:
    """
    Redis ZSET-backed frontier with:
      - [F9] Per-domain budget enforcement
      - [F10] PageRank-boosted priority
      - Weighted-fair domain scheduling
    """

    def __init__(
        self,
        redis_url:   str,
        crawl_name:  str          = "default",
        budget:      Optional[DomainBudget] = None,
        booster:     Optional[PageRankBooster] = None,
    ):
        self._url     = redis_url
        self._pfx     = f"crawl:{crawl_name}"
        self._r: Optional[aioredis.Redis] = None
        self._budget  = budget  or DomainBudget()
        self._booster = booster or PageRankBooster()
        self.output_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=500)

    async def open(self):
        self._r = aioredis.from_url(self._url, decode_responses=True)
        await self._r.ping()
        log.info("RedisFrontier: connected")

    async def close(self):
        if self._r:
            await self._r.aclose()

    async def add(self, url: str) -> bool:
        url = normalise(url)
        if not url:
            return False
        dom = domain(url)
        if not self._budget.has_budget(dom):
            return False
        added = await self._r.sadd(f"{self._pfx}:seen", url)
        if not added:
            return False
        base  = score_url(url)
        score = self._booster.boost(url, base)
        await self._r.zadd(f"{self._pfx}:pending", {url: -score})
        return True

    async def add_many(self, urls: list[str]):
        pipe = self._r.pipeline(transaction=False)
        valid = []
        for url in urls:
            url = normalise(url)
            if url and self._budget.has_budget(domain(url)):
                pipe.sadd(f"{self._pfx}:seen", url)
                valid.append(url)
        if not valid:
            return
        results = await pipe.execute()
        mapping = {}
        for url, added in zip(valid, results):
            if added:
                base  = score_url(url)
                score = self._booster.boost(url, base)
                mapping[url] = -score
        if mapping:
            await self._r.zadd(f"{self._pfx}:pending", mapping)

    async def run_scheduler(self, stop_event: asyncio.Event):
        while not stop_event.is_set():
            try:
                items = await self._r.zpopmin(f"{self._pfx}:pending", count=20)
                if not items:
                    await asyncio.sleep(0.1)
                    continue
                for url, _ in items:
                    dom = domain(url)
                    if self._budget.has_budget(dom):
                        await self.output_q.put(url)
            except Exception as exc:
                log.warning("RedisFrontier scheduler: %s", exc)
                await asyncio.sleep(0.5)

    async def pending_count(self) -> int:
        return await self._r.zcard(f"{self._pfx}:pending")

    def consume_budget(self, url: str):
        self._budget.consume(domain(url))


# ─────────────────────────────────────────────────────────────
# SQLITE FRONTIER  (fallback)
# ─────────────────────────────────────────────────────────────

class SQLiteFrontier:
    """
    aiosqlite-backed frontier with [F9] budget + [F10] PageRank boost.
    All v5 fixes preserved: deque popleft O(1), bloom O(1), weighted-fair.
    """

    FLUSH_BATCH = 250; FLUSH_DELAY = 2.0; MAX_TOKENS = 5

    def __init__(
        self,
        db_path:    str,
        bloom_path: str               = BLOOM_PATH,
        budget:     Optional[DomainBudget]    = None,
        booster:    Optional[PageRankBooster] = None,
    ):
        self._db_path   = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._budget    = budget  or DomainBudget()
        self._booster   = booster or PageRankBooster()
        self._bloom_path = bloom_path
        if _BLOOM_OK:
            self._url_bloom = (
                pickle.load(open(bloom_path, "rb")) if os.path.exists(bloom_path)
                else ScalableBloomFilter(initial_capacity=1_000_000, error_rate=0.0001)
            )
            self._use_bloom = True
        else:
            self._use_bloom = False
        self._domain_qs: dict[str, deque] = {}
        self._domain_heap: list[tuple[int, str]] = []
        self._domains_in_heap: set[str] = set()
        self.output_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=500)
        self._lock    = asyncio.Lock()
        self._pending: list[tuple[str, int]] = []
        self._last_flush = 0.0

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
            pickle.dump(self._url_bloom, open(self._bloom_path, "wb"))
        if self._db:
            await self._db.close()

    async def _is_seen(self, url: str) -> bool:
        if self._use_bloom:
            return url in self._url_bloom
        async with self._db.execute("SELECT 1 FROM seen_urls WHERE url=?", (url,)) as c:
            return (await c.fetchone()) is not None

    async def add(self, url: str) -> bool:
        url = normalise(url)
        if not url:
            return False
        dom = domain(url)
        if not self._budget.has_budget(dom):
            return False
        async with self._lock:
            if await self._is_seen(url):
                return False
            if self._use_bloom:
                self._url_bloom.add(url)
            base   = score_url(url)
            prio   = self._booster.boost(url, base)
            if dom not in self._domain_qs:
                self._domain_qs[dom] = deque()
                heapq.heappush(self._domain_heap, (-prio, dom))
                self._domains_in_heap.add(dom)
            self._domain_qs[dom].append(url)
            self._pending.append((url, prio))
            await self._maybe_flush()
            return True

    async def add_many(self, urls: list[str]):
        for url in urls:
            await self.add(url)

    async def run_scheduler(self, stop_event: asyncio.Event):
        import time
        while not stop_event.is_set():
            async with self._lock:
                snapshot = sorted(self._domain_qs.items(), key=lambda kv: -len(kv[1]))
                total = sum(len(q) for _, q in snapshot) or 1
                fed = False
                for dom, q in snapshot:
                    if not q:
                        continue
                    tokens = max(1, min(self.MAX_TOKENS, round(self.MAX_TOKENS * len(q) / total)))
                    for _ in range(tokens):
                        if not q:
                            break
                        url = q.popleft()
                        if not self._budget.has_budget(dom):
                            continue
                        try:
                            self.output_q.put_nowait(url)
                            fed = True
                        except asyncio.QueueFull:
                            q.appendleft(url)
                            break
            await asyncio.sleep(0.05)

    async def load_resume(self):
        async with self._db.execute(
            "SELECT s.url, s.priority FROM seen_urls s LEFT JOIN done d ON s.url=d.url WHERE d.url IS NULL"
        ) as cur:
            rows = await cur.fetchall()
        for url, prio in rows:
            dom = domain(url)
            if dom not in self._domain_qs:
                self._domain_qs[dom] = deque()
            self._domain_qs[dom].append(url)
            if self._use_bloom:
                self._url_bloom.add(url)
        for dom, q in self._domain_qs.items():
            if q and dom not in self._domains_in_heap:
                heapq.heappush(self._domain_heap, (-score_url(next(iter(q))), dom))
                self._domains_in_heap.add(dom)
        log.info("SQLiteFrontier resumed: %d URLs", sum(len(q) for q in self._domain_qs.values()))

    def consume_budget(self, url: str):
        self._budget.consume(domain(url))

    async def pending_count(self) -> int:
        return sum(len(q) for q in self._domain_qs.values())

    async def _maybe_flush(self):
        import time
        if len(self._pending) >= self.FLUSH_BATCH or time.monotonic() - self._last_flush >= self.FLUSH_DELAY:
            await self._flush_db()

    async def _flush_db(self, force: bool = False):
        import time
        if not self._db or not self._pending:
            return
        await self._db.executemany(
            "INSERT OR IGNORE INTO seen_urls (url, priority) VALUES (?,?)", self._pending
        )
        await self._db.commit()
        self._pending.clear()
        self._last_flush = time.monotonic()


def make_frontier(
    redis_url:    Optional[str],
    sqlite_path:  str,
    crawl_name:   str                    = "default",
    budget:       Optional[DomainBudget] = None,
    booster:      Optional[PageRankBooster] = None,
):
    if redis_url and _REDIS_OK:
        return RedisFrontier(redis_url, crawl_name, budget, booster)
    log.warning("Frontier: using SQLite fallback (set redis_url for production)")
    return SQLiteFrontier(sqlite_path, budget=budget, booster=booster)
