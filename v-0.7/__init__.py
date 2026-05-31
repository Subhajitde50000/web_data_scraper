"""
PyScraper v7
============
A production-grade web crawler with 20 advanced features.

Quick start:
    from scraper_v7 import Scraper
    Scraper(seed_urls=["https://example.com/"]).run()

Full API:
    from scraper_v7 import (
        Scraper,
        ProxyPool, TLSFetcher, CaptchaSolver,
        ArticleExtractor, JSONLDExtractor, EntityExtractor, LanguageDetector,
        SemanticDedup, DedupPipeline,
        DomainBudget, PageRankBooster,
        PrometheusMetrics, TracingMiddleware, GrafanaDashboard,
        WALLog, OpenSearchWriter,
    )
"""

from .scraper import Scraper
from .anti_bot import ProxyPool, TLSFetcher, CaptchaSolver
from .extraction import (
    ArticleExtractor, JSONLDExtractor,
    EntityExtractor, LanguageDetector, ExtractionPipeline,
)
from .dedup import ExactDedup, NearDedup, SemanticDedup, DedupPipeline
from .frontier import DomainBudget, PageRankBooster, make_frontier
from .fetch import HTTPFetcher, PlaywrightPool, CircuitBreaker, RateLimiter
from .storage import WALLog, OpenSearchWriter, make_writer
from .metrics import PrometheusMetrics, TracingMiddleware, GrafanaDashboard
from .models import ScrapedItem, CrawlMeta, ExtractedContent, FetchResult

__version__ = "7.0.0"
__all__ = [
    "Scraper",
    "ProxyPool", "TLSFetcher", "CaptchaSolver",
    "ArticleExtractor", "JSONLDExtractor", "EntityExtractor",
    "LanguageDetector", "ExtractionPipeline",
    "ExactDedup", "NearDedup", "SemanticDedup", "DedupPipeline",
    "DomainBudget", "PageRankBooster", "make_frontier",
    "HTTPFetcher", "PlaywrightPool", "CircuitBreaker", "RateLimiter",
    "WALLog", "OpenSearchWriter", "make_writer",
    "PrometheusMetrics", "TracingMiddleware", "GrafanaDashboard",
    "ScrapedItem", "CrawlMeta", "ExtractedContent", "FetchResult",
]
