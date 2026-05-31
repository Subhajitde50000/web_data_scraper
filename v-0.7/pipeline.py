"""
Pipeline module
===============
Five-stage async worker pipeline:

  Frontier → [Fetch workers] → fetch_q → [Parse/Enrich workers]
           → writer_q → [Storage writer]
           → discovered links → Frontier (feedback loop)

Integrates all v7 modules:
  anti_bot   — proxy rotation, TLS fingerprint, CAPTCHA
  extraction — trafilatura, JSON-LD, NER, language detection
  dedup      — exact + near + semantic
  frontier   — Redis / SQLite with per-domain budget + PageRank
  fetch      — httpx HTTP/2, circuit breaker, Playwright infinite scroll
  storage    — PostgreSQL / SQLite, WAL, OpenSearch
  metrics    — Prometheus, OpenTelemetry tracing
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional
from urllib.parse import urlparse
from xml.etree import ElementTree

import aiohttp   # used only for sitemap fetching

from .models import ScrapedItem, FetchResult
from .utils import normalise, domain
from .anti_bot import ProxyPool, TLSFetcher, CaptchaSolver
from .extraction import ExtractionPipeline
from .dedup import DedupPipeline
from .fetch import HTTPFetcher, PlaywrightPool, AsyncRobots, RateLimiter
from .storage import WALLog, OpenSearchWriter
from .metrics import PrometheusMetrics, TracingMiddleware

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; PyScraper/7.0; "
    "+https://github.com/example/scraper)"
)


# ─────────────────────────────────────────────────────────────
# SITEMAP SEEDER  (unchanged from v6)
# ─────────────────────────────────────────────────────────────

async def fetch_sitemap_urls(url: str, session, depth: int = 0) -> list[str]:
    if depth > 3:
        return []
    urls: list[str] = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
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
                if u:
                    urls.append(u)
    except Exception as exc:
        log.debug("Sitemap error %s: %s", url[:60], exc)
    return urls


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

class Pipeline:
    """
    Orchestrates all v7 components into a single coherent crawl run.

    Worker counts:
      n_fetch   — concurrent URL fetchers (httpx + optional Playwright)
      n_enrich  — concurrent parse+enrich workers (CPU-bound NLP)
    """

    def __init__(
        self,
        frontier,
        fetcher:          HTTPFetcher,
        tls_fetcher:      TLSFetcher,
        pw_pool:          PlaywrightPool,
        rate:             RateLimiter,
        robots:           AsyncRobots,
        extractor:        ExtractionPipeline,
        dedup:            DedupPipeline,
        writer,
        os_writer:        Optional[OpenSearchWriter],
        wal:              WALLog,
        metrics:          PrometheusMetrics,
        tracer:           TracingMiddleware,
        seed_domains:     set[str],
        max_pages:        int,
        follow_links:     bool,
        same_domain_only: bool,
        use_tls_spoof:    bool,
        n_fetch:          int = 10,
        n_enrich:         int = 4,
    ):
        self.frontier         = frontier
        self.fetcher          = fetcher
        self.tls_fetcher      = tls_fetcher
        self.pw_pool          = pw_pool
        self.rate             = rate
        self.robots           = robots
        self.extractor        = extractor
        self.dedup            = dedup
        self.writer           = writer
        self.os_writer        = os_writer
        self.wal              = wal
        self.metrics          = metrics
        self.tracer           = tracer
        self.seed_domains     = seed_domains
        self.max_pages        = max_pages
        self.follow_links     = follow_links
        self.same_domain_only = same_domain_only
        self.use_tls_spoof    = use_tls_spoof
        self.n_fetch          = n_fetch
        self.n_enrich         = n_enrich

        self._fetch_q: asyncio.Queue[Optional[FetchResult]] = asyncio.Queue(maxsize=300)
        self._scraped      = 0
        self._scraped_lock = asyncio.Lock()
        self._stop         = asyncio.Event()

    def _in_scope(self, url: str) -> bool:
        if not self.same_domain_only:
            return True
        return domain(url) in self.seed_domains

    # ── fetch worker ──────────────────────────────────────────

    async def _fetch_worker(self, worker_id: int):
        while not self._stop.is_set():
            try:
                url = await asyncio.wait_for(
                    self.frontier.output_q.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            if url is None:
                break

            # robots.txt check
            if not await self.robots.allowed(url, self.fetcher._client):
                log.debug("robots blocks %s", url)
                await self.wal.log_skipped(url, "robots")
                continue

            # Circuit breaker check (via fetcher internals)
            dom = domain(url)

            # Conditional headers
            etag, last_mod = await self.writer.get_conditional(url)

            # Span: fetch
            root_span  = self.tracer.start_url_span(url)
            fetch_span = self.tracer.child_span(root_span, "crawler.fetch", url=url)

            t0 = time.monotonic()

            # Choose fetch path: TLS spoof → httpx → (Playwright later)
            if self.use_tls_spoof:
                html, status, resp_headers = await self.tls_fetcher.fetch(url)
                from .models import CrawlMeta
                meta = CrawlMeta(
                    status_code=status,
                    content_type=resp_headers.get("content-type", ""),
                    response_time_ms=round((time.monotonic()-t0)*1000, 1),
                    rendered_by="curl-cffi",
                )
                result = FetchResult(
                    url=url, final_url=url,
                    html=html if status == 200 else None,
                    meta=meta,
                    needs_js=(len(html or "") < 3000),
                )
            else:
                result = await self.fetcher.fetch(url, self.rate, etag, last_mod)

            self.tracer.end_span(
                fetch_span,
                error=None if result.html else f"status={result.meta.status_code}"
            )
            self.metrics.observe_fetch(result.meta.response_time_ms)
            await self.wal.log_fetched(url, result.meta.status_code, result.meta.response_time_ms)

            if result.meta.circuit_open:
                self.metrics.inc_circuit_open(dom)
                self.tracer.end_span(root_span, "circuit_open")
                continue

            if result.html is None:
                if result.meta.status_code not in (304, 0):
                    self.metrics.inc_error("fetch")
                    await self.wal.log_error(url, f"status={result.meta.status_code}")
                self.tracer.end_span(root_span)
                continue

            # Playwright escalation for JS-heavy pages
            if result.needs_js and self.pw_pool.available:
                self.metrics.inc_js_escalation()
                self.metrics.set_active_browsers(self.pw_pool.active_count())
                pw_span = self.tracer.child_span(root_span, "crawler.playwright", url=url)
                pw_html, pw_final = await self.pw_pool.fetch(url)
                self.tracer.end_span(pw_span)
                self.metrics.set_active_browsers(self.pw_pool.active_count())
                if pw_html:
                    result.html      = pw_html
                    result.final_url = pw_final
                    result.meta.rendered_by = "playwright"

            # Attach span so enrich worker can close it
            result._root_span = root_span
            self.metrics.set_queue_depth(self._fetch_q.qsize())
            await self._fetch_q.put(result)

        # Sentinel — worker done
        await self._fetch_q.put(None)

    # ── enrich/parse worker ───────────────────────────────────

    async def _enrich_worker(self, worker_id: int, n_fetch: int):
        sentinels = 0
        while True:
            result = await self._fetch_q.get()
            if result is None:
                sentinels += 1
                if sentinels >= n_fetch:
                    break
                continue

            if result.html is None:
                continue

            root_span = getattr(result, "_root_span", None)

            # ── Parse span
            parse_span = self.tracer.child_span(
                root_span or self.tracer.start_url_span(result.url),
                "crawler.parse", url=result.url
            )
            t0 = time.monotonic()
            content = self.extractor.run(result.html, result.final_url)
            self.metrics.observe_parse((time.monotonic() - t0) * 1000)
            self.tracer.end_span(parse_span)

            # Language scope check
            if not self.extractor.langdet.in_scope(content.language):
                log.debug("Language %s out of scope: %s", content.language, result.url)
                self.metrics.inc_duplicate("lang_filtered")
                if root_span:
                    self.tracer.end_span(root_span, "language_filtered")
                continue

            # ── Enrich span (NER already done inside extractor.run)
            enrich_span = self.tracer.child_span(
                root_span, "crawler.enrich", url=result.url
            ) if root_span else None

            t1 = time.monotonic()
            # Dedup check
            text_for_dedup = content.article or content.text
            is_dup, dup_reason, embedding = self.dedup.check_and_add(
                text_for_dedup, result.final_url
            )
            if embedding:
                content.embedding = embedding
            self.metrics.observe_enrich((time.monotonic() - t1) * 1000)
            if enrich_span:
                self.tracer.end_span(enrich_span)

            if is_dup:
                self.metrics.inc_duplicate(dup_reason)
                log.debug("Dup [%s] skipped: %s", dup_reason, result.url)
                await self.wal.log_skipped(result.url, f"dup:{dup_reason}")
                if root_span:
                    self.tracer.end_span(root_span, f"duplicate:{dup_reason}")
                continue

            # Determine canonical page URL
            page_url = content.canonical or result.final_url

            # Crawl meta enrichment
            result.meta.final_url     = page_url
            result.meta.content_hash  = self.dedup.exact.seen(text_for_dedup) and "" or ""
            result.meta.language      = content.language

            item = ScrapedItem(url=page_url, content=content, crawl=result.meta)

            # ── Store span
            store_span = self.tracer.child_span(root_span, "crawler.store", url=page_url) if root_span else None
            await self.writer.push(item)
            if self.os_writer:
                await self.os_writer.push(item)
            if store_span:
                self.tracer.end_span(store_span)

            # Budget consumption
            if hasattr(self.frontier, "consume_budget"):
                self.frontier.consume_budget(result.url)

            # Counter + logging
            async with self._scraped_lock:
                self._scraped += 1
                count = self._scraped

            self.metrics.inc_scraped(result.meta.rendered_by, content.language)
            log.info(
                "[%d/%d] %s  (%dms, %s, lang=%s)",
                count, self.max_pages, page_url,
                result.meta.response_time_ms,
                result.meta.rendered_by,
                content.language or "?",
            )

            # Link discovery
            if self.follow_links and content.links:
                in_scope = [l for l in content.links if self._in_scope(l)]
                await self.frontier.add_many(in_scope)

            # Frontier gauge
            try:
                self.metrics.set_frontier(await self.frontier.pending_count())
            except Exception:
                pass

            if root_span:
                self.tracer.end_span(root_span)

            if count >= self.max_pages:
                self._stop.set()
                break

    # ── coordinator: guarantees exactly n_enrich sentinels ───

    async def _coordinator(self, fetch_tasks: list[asyncio.Task]):
        await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for _ in range(self.n_enrich):
            await self._fetch_q.put(None)

    # ── sitemap seed ─────────────────────────────────────────

    async def _seed_from_sitemaps(self):
        async with aiohttp.ClientSession(
            headers={"User-Agent": USER_AGENT}
        ) as session:
            for dom in self.seed_domains:
                origin  = f"https://{dom}"
                sm_urls = await self.robots.site_maps(origin, self.fetcher._client) \
                          if self.fetcher._client else []
                if not sm_urls:
                    sm_urls = [f"{origin}/sitemap.xml"]
                for sm in sm_urls:
                    urls = await fetch_sitemap_urls(sm, session)
                    log.info("Sitemap %s: %d URLs", sm, len(urls))
                    await self.frontier.add_many(
                        [u for u in urls if self._in_scope(u)]
                    )

    # ── run ──────────────────────────────────────────────────

    async def run(self):
        await self.wal.open()
        await self._seed_from_sitemaps()
        await self.writer.start()

        if self.os_writer:
            await self.os_writer.start()

        sched_stop = asyncio.Event()
        sched_task = asyncio.create_task(
            self.frontier.run_scheduler(sched_stop)
        )

        fetch_tasks = [
            asyncio.create_task(self._fetch_worker(i))
            for i in range(self.n_fetch)
        ]
        enrich_tasks = [
            asyncio.create_task(self._enrich_worker(i, self.n_fetch))
            for i in range(self.n_enrich)
        ]
        coord_task = asyncio.create_task(self._coordinator(fetch_tasks))

        await asyncio.gather(*enrich_tasks)

        coord_task.cancel()
        sched_stop.set()
        sched_task.cancel()

        await self.writer.stop()
        if self.os_writer:
            await self.os_writer.stop()
        self.dedup.save()
        await self.wal.close()
        await self.frontier.close()

        log.info("Pipeline complete. Scraped %d pages.", self._scraped)
