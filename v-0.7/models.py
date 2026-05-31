"""
Shared data models for PyScraper v7.
All other modules import from here — no circular deps.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrawlMeta:
    final_url:        str   = ""
    status_code:      int   = 0
    content_type:     str   = ""
    response_time_ms: float = 0.0
    etag:             str   = ""
    last_modified:    str   = ""
    content_hash:     str   = ""    # SHA-256 of cleaned text
    simhash:          int   = 0     # 64-bit weighted simhash
    rendered_by:      str   = "httpx"   # "httpx" | "playwright"
    proxy_used:       str   = ""
    http_version:     str   = ""    # "HTTP/1.1" | "HTTP/2" | "HTTP/3"
    language:         str   = ""
    circuit_open:     bool  = False


@dataclass
class ExtractedContent:
    title:      str  = ""
    text:       str  = ""           # raw selectolax text
    article:    str  = ""           # trafilatura clean article
    author:     str  = ""
    date:       str  = ""
    language:   str  = ""
    json_ld:    list = field(default_factory=list)   # list of dicts
    schema_org: dict = field(default_factory=dict)
    entities:   dict = field(default_factory=dict)   # spaCy NER results
    summary:    str  = ""
    links:      list = field(default_factory=list)
    meta:       dict = field(default_factory=dict)
    canonical:  Optional[str] = None
    embedding:  Optional[list] = None  # sentence-transformer vector


@dataclass
class ScrapedItem:
    url:       str
    content:   ExtractedContent = field(default_factory=ExtractedContent)
    crawl:     CrawlMeta        = field(default_factory=CrawlMeta)


@dataclass
class FetchResult:
    url:       str
    final_url: str
    html:      Optional[str]
    meta:      CrawlMeta
    needs_js:  bool = False


@dataclass
class CrawlEvent:
    """Append-only WAL event."""
    event:     str   = ""    # "fetched" | "parsed" | "stored" | "error" | "skipped"
    url:       str   = ""
    detail:    str   = ""
    ts:        float = 0.0
