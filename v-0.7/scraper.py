"""
Scraper v7 — top-level facade
==============================
All async initialisation deferred to run_async() — safe in Jupyter / FastAPI.
Constructor is pure Python, zero side effects.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

from .anti_bot import ProxyPool, TLSFetcher, CaptchaSolver
from .dedup import DedupPipeline, ExactDedup, NearDedup, SemanticDedup
from .extraction import ExtractionPipeline, ArticleExtractor, JSONLDExtractor, EntityExtractor, LanguageDetector
from .fetch import HTTPFetcher, PlaywrightPool, AsyncRobots, RateLimiter
from .frontier import make_frontier, DomainBudget, PageRankBooster
from .metrics import PrometheusMetrics, TracingMiddleware, GrafanaDashboard
from .pipeline import Pipeline
from .storage import WALLog, OpenSearchWriter, make_writer

log = logging.getLogger(__name__)


class Scraper:
    """
    PyScraper v7 — production web crawler.

    All parameters have sensible defaults.
    Every integration degrades gracefully when its dep is missing.

    Quick start (zero services):
        Scraper(seed_urls=["https://example.com/"]).run()

    Full stack:
        Scraper(
            seed_urls    = ["https://example.com/"],
            pg_dsn       = "postgresql://user:pw@localhost/crawl",
            redis_url    = "redis://localhost:6379",
            qdrant_url   = "http://localhost:6333",
            os_hosts     = ["http://localhost:9200"],
            otlp_endpoint = "http://localhost:4317",
            max_browsers = 5,
            metrics_port = 8000,
            proxy_urls   = ["http://user:pass@proxy:port"],
            use_tls_spoof = True,
        ).run()
    """

    def __init__(
        self,
        # ── core ──────────────────────────────────────────────
        seed_urls:               list[str],
        max_pages:               int   = 100,
        default_delay:           float = 1.5,
        global_concurrency:      int   = 50,
        per_domain_concurrency:  int   = 2,
        n_fetch_workers:         int   = 10,
        n_enrich_workers:        int   = 4,
        follow_links:            bool  = True,
        same_domain_only:        bool  = True,
        crawl_name:              str   = "default",
        resume:                  bool  = False,
        target_languages:        Optional[list[str]] = None,  # e.g. ["en","de"]

        # ── storage [I1] ──────────────────────────────────────
        pg_dsn:                  Optional[str] = None,
        output_db:               str   = "output.db",

        # ── frontier [I2] ────────────────────────────────────
        redis_url:               Optional[str] = None,
        checkpoint_db:           str   = "checkpoint.db",

        # ── per-domain budget [F9] ────────────────────────────
        default_budget:          int   = 1000,
        domain_overrides:        Optional[dict[str, int]]  = None,
        domain_tiers:            Optional[dict[str, int]]  = None,

        # ── playwright [I3] + CAPTCHA [F3] ───────────────────
        max_browsers:            int   = 5,
        captcha_api_key:         str   = "",
        captcha_provider:        str   = "2captcha",

        # ── proxy rotation [F1] ───────────────────────────────
        proxy_urls:              Optional[list[str]] = None,
        proxy_sticky:            bool  = True,

        # ── TLS fingerprint [F2] ─────────────────────────────
        use_tls_spoof:           bool  = False,
        rotate_fingerprint:      bool  = True,

        # ── semantic dedup [F8] ───────────────────────────────
        qdrant_url:              Optional[str] = None,  # "http://localhost:6333"
        semantic_threshold:      float = 0.92,

        # ── OpenSearch [F16] ─────────────────────────────────
        os_hosts:                Optional[list[str]] = None,  # ["http://localhost:9200"]

        # ── observability [F17/F18] ───────────────────────────
        metrics_port:            int   = 8000,
        otlp_endpoint:           Optional[str] = None,  # "http://localhost:4317"

        # ── WAL log [F15] ────────────────────────────────────
        wal_path:                str   = "crawler.wal",

        # ── Grafana [F19] ────────────────────────────────────
        grafana_dashboard_path:  Optional[str] = None,  # write dashboard JSON on init
    ):
        self.seed_urls    = seed_urls
        self.seed_domains = {urlparse(u).netloc for u in seed_urls}
        self.resume       = resume
        self._cfg = dict(
            max_pages=max_pages,
            default_delay=default_delay,
            global_concurrency=global_concurrency,
            per_domain_concurrency=per_domain_concurrency,
            n_fetch_workers=n_fetch_workers,
            n_enrich_workers=n_enrich_workers,
            follow_links=follow_links,
            same_domain_only=same_domain_only,
            crawl_name=crawl_name,
            target_languages=target_languages,
            pg_dsn=pg_dsn,
            output_db=output_db,
            redis_url=redis_url,
            checkpoint_db=checkpoint_db,
            default_budget=default_budget,
            domain_overrides=domain_overrides,
            domain_tiers=domain_tiers,
            max_browsers=max_browsers,
            captcha_api_key=captcha_api_key,
            captcha_provider=captcha_provider,
            proxy_urls=proxy_urls,
            proxy_sticky=proxy_sticky,
            use_tls_spoof=use_tls_spoof,
            rotate_fingerprint=rotate_fingerprint,
            qdrant_url=qdrant_url,
            semantic_threshold=semantic_threshold,
            os_hosts=os_hosts,
            metrics_port=metrics_port,
            otlp_endpoint=otlp_endpoint,
            wal_path=wal_path,
        )

        # Write Grafana dashboard to disk at init time if requested
        if grafana_dashboard_path:
            GrafanaDashboard().save(grafana_dashboard_path)

    async def run_async(self):
        """Async entry point — embed in existing event loop (FastAPI / Jupyter)."""
        cfg = self._cfg

        # [F17] Prometheus metrics
        metrics = PrometheusMetrics(
            port=cfg["metrics_port"],
            enabled=cfg["metrics_port"] > 0,
        )

        # [F18] Distributed tracing
        tracer = TracingMiddleware(
            otlp_endpoint=cfg["otlp_endpoint"] or "http://localhost:4317",
            enabled=bool(cfg["otlp_endpoint"]),
        )

        # [F15] WAL log
        wal = WALLog(cfg["wal_path"])

        # [F9] Per-domain budget
        budget = DomainBudget(
            default_budget=cfg["default_budget"],
            domain_overrides=cfg["domain_overrides"],
            domain_tiers=cfg["domain_tiers"],
        )

        # [F10] PageRank booster (loads from DB on resume)
        booster = PageRankBooster()

        # [I2] Frontier
        frontier = make_frontier(
            cfg["redis_url"], cfg["checkpoint_db"],
            cfg["crawl_name"], budget, booster,
        )
        await frontier.open()
        if self.resume and hasattr(frontier, "load_resume"):
            await frontier.load_resume()
        else:
            await frontier.add_many(self.seed_urls)

        # [I1] Storage writer
        writer = make_writer(cfg["pg_dsn"], cfg["output_db"], wal)

        # [F16] OpenSearch writer
        os_writer = None
        if cfg["os_hosts"]:
            os_writer = OpenSearchWriter(hosts=cfg["os_hosts"])

        # [F1] Proxy pool
        proxy_pool = None
        if cfg["proxy_urls"]:
            proxy_pool = ProxyPool(cfg["proxy_urls"], sticky=cfg["proxy_sticky"])
            metrics.set_proxy_alive(len(proxy_pool._alive))

        # [F3] CAPTCHA solver
        captcha = CaptchaSolver(
            api_key=cfg["captcha_api_key"],
            provider=cfg["captcha_provider"],
        ) if cfg["captcha_api_key"] else None

        # [F2] TLS fingerprint fetcher
        tls_fetcher = TLSFetcher(
            rotate_fingerprint=cfg["rotate_fingerprint"],
            enabled=cfg["use_tls_spoof"],
        ) if cfg["use_tls_spoof"] else TLSFetcher(enabled=False)

        # [I3] Playwright pool
        pw_pool = PlaywrightPool(
            max_browsers=cfg["max_browsers"],
            captcha_solver=captcha,
        )
        if cfg["max_browsers"] > 0:
            await pw_pool.open()

        # HTTP/2 fetcher [F11]
        fetcher = HTTPFetcher(
            global_concurrency=cfg["global_concurrency"],
            proxy_pool=proxy_pool,
        )
        await fetcher.open()

        # Robots + rate limiter
        robots = AsyncRobots(cfg["default_delay"])
        rate   = RateLimiter(cfg["default_delay"], cfg["per_domain_concurrency"])

        # [F4-F7] Extraction pipeline
        extractor = ExtractionPipeline(
            article_extractor = ArticleExtractor(),
            jsonld_extractor  = JSONLDExtractor(),
            entity_extractor  = EntityExtractor(),
            language_detector = LanguageDetector(cfg["target_languages"]),
        )

        # [F8] Dedup pipeline
        dedup = DedupPipeline(
            semantic=SemanticDedup(
                qdrant_url=cfg["qdrant_url"] or "http://localhost:6333",
                threshold=cfg["semantic_threshold"],
                enabled=bool(cfg["qdrant_url"]),
            )
        )

        # PageRank pre-load on resume
        if self.resume:
            if cfg["pg_dsn"] and writer._pool if hasattr(writer, "_pool") else False:
                await booster.load_from_pg(writer._pool)
            else:
                await booster.load_from_sqlite(cfg["output_db"])

        pipeline = Pipeline(
            frontier          = frontier,
            fetcher           = fetcher,
            tls_fetcher       = tls_fetcher,
            pw_pool           = pw_pool,
            rate              = rate,
            robots            = robots,
            extractor         = extractor,
            dedup             = dedup,
            writer            = writer,
            os_writer         = os_writer,
            wal               = wal,
            metrics           = metrics,
            tracer            = tracer,
            seed_domains      = self.seed_domains,
            max_pages         = cfg["max_pages"],
            follow_links      = cfg["follow_links"],
            same_domain_only  = cfg["same_domain_only"],
            use_tls_spoof     = cfg["use_tls_spoof"],
            n_fetch           = cfg["n_fetch_workers"],
            n_enrich          = cfg["n_enrich_workers"],
        )

        try:
            await pipeline.run()
        finally:
            await fetcher.close()
            await pw_pool.close()

    def run(self):
        """Sync entry point for standalone script usage."""
        asyncio.run(self.run_async())
