"""
Content extraction module
=========================
[F4]  ArticleExtractor  — trafilatura clean article body, author, date
[F5]  JSONLDExtractor   — JSON-LD and schema.org structured data
[F6]  EntityExtractor   — spaCy NER (people, orgs, locations, dates, money)
[F7]  LanguageDetector  — lingua / langdetect language identification
      FastParser        — selectolax base parser (title, links, meta, canonical)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional
from urllib.parse import urljoin

from selectolax.parser import HTMLParser as SXParser

from .models import ExtractedContent
from .utils import normalise, resolve

log = logging.getLogger(__name__)

NOISE_TAGS = ("script", "style", "nav", "footer", "aside", "header", "noscript")

# ── optional imports ──────────────────────────────────────────
try:
    import trafilatura
    from trafilatura.settings import use_config
    _TRAFILATURA_OK = True
except ImportError:
    _TRAFILATURA_OK = False
    log.warning("extraction: trafilatura not installed — using raw text fallback")

try:
    import spacy
    _SPACY_OK = True
except ImportError:
    _SPACY_OK = False

try:
    from lingua import Language, LanguageDetectorBuilder
    _LINGUA_OK = True
except ImportError:
    try:
        from langdetect import detect as _langdetect
        _LINGUA_OK = False
        _LANGDETECT_OK = True
    except ImportError:
        _LINGUA_OK = False
        _LANGDETECT_OK = False


# ─────────────────────────────────────────────────────────────
# [F4] ARTICLE EXTRACTOR  (trafilatura)
# ─────────────────────────────────────────────────────────────

class ArticleExtractor:
    """
    [F4] Uses trafilatura's boilerplate-removal pipeline to extract:
      - Clean article body (ads, navbars, footers stripped)
      - Author, publish date, title
      - Language (if trafilatura detects it)

    Falls back to raw selectolax text if trafilatura is not installed.
    """

    def __init__(self, include_comments: bool = False, include_tables: bool = True):
        self._comments = include_comments
        self._tables   = include_tables

        if _TRAFILATURA_OK:
            self._cfg = use_config()
            self._cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")
            log.info("ArticleExtractor: trafilatura ready")
        else:
            self._cfg = None

    def extract(self, html: str, url: str = "") -> tuple[str, str, str, str]:
        """
        Returns (article_text, author, date, language).
        Falls back to empty strings on failure.
        """
        if not _TRAFILATURA_OK:
            return "", "", "", ""

        try:
            result = trafilatura.extract(
                html,
                url=url,
                config=self._cfg,
                include_comments=self._comments,
                include_tables=self._tables,
                output_format="python",
                with_metadata=True,
            )
            if not result:
                return "", "", "", ""

            # result is a dict when output_format="python"
            if isinstance(result, dict):
                return (
                    result.get("text", "")    or "",
                    result.get("author", "")  or "",
                    result.get("date", "")    or "",
                    result.get("language", "") or "",
                )
            return str(result), "", "", ""
        except Exception as exc:
            log.debug("ArticleExtractor error %s: %s", url[:60], exc)
            return "", "", "", ""


# ─────────────────────────────────────────────────────────────
# [F5] JSON-LD + SCHEMA.ORG EXTRACTOR
# ─────────────────────────────────────────────────────────────

class JSONLDExtractor:
    """
    [F5] Parses <script type="application/ld+json"> blocks.
    Returns a list of dicts (one per block) plus a flattened schema_org dict
    containing the most useful fields regardless of nesting.

    No extra library needed — pure JSON parsing.
    """

    _USEFUL_FIELDS = {
        "name", "headline", "description", "author", "datePublished",
        "dateModified", "publisher", "url", "image", "price", "priceCurrency",
        "ratingValue", "reviewCount", "articleBody", "breadcrumb",
        "@type", "itemListElement",
    }

    def extract(self, html: str) -> tuple[list[dict], dict]:
        """Returns (json_ld_blocks, flattened_schema_org)."""
        tree = SXParser(html)
        blocks: list[dict] = []

        for node in tree.css('script[type="application/ld+json"]'):
            raw = node.text(strip=True)
            if not raw:
                continue
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    blocks.extend(data)
                elif isinstance(data, dict):
                    blocks.append(data)
            except json.JSONDecodeError:
                pass

        # Flatten useful fields from all blocks
        flat: dict = {}
        for block in blocks:
            self._flatten(block, flat)

        return blocks, flat

    def _flatten(self, obj, acc: dict, depth: int = 0):
        if depth > 4 or not isinstance(obj, dict):
            return
        for k, v in obj.items():
            if k in self._USEFUL_FIELDS:
                if k not in acc:
                    acc[k] = v
            if isinstance(v, dict):
                self._flatten(v, acc, depth + 1)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        self._flatten(item, acc, depth + 1)


# ─────────────────────────────────────────────────────────────
# [F6] ENTITY EXTRACTOR  (spaCy NER)
# ─────────────────────────────────────────────────────────────

class EntityExtractor:
    """
    [F6] spaCy Named Entity Recognition.
    Extracts: PERSON, ORG, GPE (geopolitical), LOC, DATE, MONEY, PRODUCT.
    Results stored as JSONB in PostgreSQL for structured entity search.

    Requires: pip install spacy && python -m spacy download en_core_web_sm
    For higher accuracy: python -m spacy download en_core_web_trf
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self._nlp = None
        if _SPACY_OK:
            try:
                import spacy as _spacy
                self._nlp = _spacy.load(model, disable=["parser", "lemmatizer"])
                log.info("EntityExtractor: spaCy %s loaded", model)
            except OSError:
                log.warning(
                    "EntityExtractor: model '%s' not found. "
                    "Run: python -m spacy download %s", model, model
                )
        else:
            log.warning("EntityExtractor: spaCy not installed — NER disabled")

    def extract(self, text: str, max_chars: int = 5000) -> dict:
        """
        Returns dict: { "PERSON": [...], "ORG": [...], "GPE": [...], ... }
        Text is truncated to max_chars to keep inference fast.
        """
        if not self._nlp or not text:
            return {}
        try:
            doc = self._nlp(text[:max_chars])
            result: dict[str, list] = {}
            for ent in doc.ents:
                if ent.label_ in ("PERSON","ORG","GPE","LOC","DATE","MONEY","PRODUCT"):
                    result.setdefault(ent.label_, [])
                    val = ent.text.strip()
                    if val and val not in result[ent.label_]:
                        result[ent.label_].append(val)
            return result
        except Exception as exc:
            log.debug("EntityExtractor error: %s", exc)
            return {}


# ─────────────────────────────────────────────────────────────
# [F7] LANGUAGE DETECTOR
# ─────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    [F7] Detects the natural language of page text.
    Priority: lingua (more accurate) → langdetect → trafilatura's built-in.

    Install: pip install lingua-language-detector
         or: pip install langdetect
    """

    def __init__(self, target_languages: Optional[list[str]] = None):
        """
        target_languages: ISO 639-1 codes to filter results to.
          e.g. ["en", "de", "fr"] — pages in other languages are flagged.
          None = detect all, no filtering.
        """
        self._targets = set(target_languages) if target_languages else None
        self._detector = None

        if _LINGUA_OK:
            langs = [Language.ENGLISH, Language.GERMAN, Language.FRENCH,
                     Language.SPANISH, Language.ITALIAN, Language.PORTUGUESE,
                     Language.DUTCH, Language.RUSSIAN, Language.CHINESE,
                     Language.JAPANESE, Language.ARABIC, Language.HINDI]
            self._detector = LanguageDetectorBuilder.from_languages(*langs).build()
            log.info("LanguageDetector: lingua ready")
        elif _LANGDETECT_OK:
            log.info("LanguageDetector: langdetect ready (install lingua for better accuracy)")
        else:
            log.warning("LanguageDetector: no library installed — lang detection disabled")

    def detect(self, text: str) -> str:
        """Returns ISO 639-1 code e.g. 'en', 'de', or '' on failure."""
        if not text or len(text) < 20:
            return ""
        sample = text[:500]
        try:
            if _LINGUA_OK and self._detector:
                lang = self._detector.detect_language_of(sample)
                return lang.iso_code_639_1.name.lower() if lang else ""
            elif _LANGDETECT_OK:
                return _langdetect(sample)
        except Exception:
            pass
        return ""

    def in_scope(self, lang_code: str) -> bool:
        """True if no target filter set, or lang is in the target set."""
        if not self._targets:
            return True
        return lang_code in self._targets


# ─────────────────────────────────────────────────────────────
# BASE FAST PARSER  (selectolax)
# ─────────────────────────────────────────────────────────────

class FastParser:
    """
    selectolax-based base parser.
    Extracts: title, raw text, links, meta tags, canonical URL.
    Used as first stage; ArticleExtractor refines the text.
    """

    def parse(
        self, html: str, base_url: str
    ) -> tuple[str, str, list[str], dict, Optional[str]]:
        """Returns (title, raw_text, links, meta, canonical_url)."""
        tree = SXParser(html)

        title_node = tree.css_first("title")
        title = title_node.text(strip=True) if title_node else ""

        canonical: Optional[str] = None
        for node in tree.css('link[rel="canonical"]'):
            href = (node.attributes.get("href") or "").strip()
            if href:
                canonical = normalise(urljoin(base_url, href)) or None
                break

        for tag in tree.css(",".join(NOISE_TAGS)):
            tag.decompose()

        body     = tree.body
        raw_text = body.text(separator=" ", strip=True) if body else ""
        raw_text = re.sub(r"\s+", " ", raw_text).strip()[:4000]

        links: list[str] = []
        seen: set[str] = set()
        for node in tree.css("a[href]"):
            href = (node.attributes.get("href") or "")
            if not href:
                continue
            full = resolve(href, base_url)
            if full and full not in seen:
                seen.add(full)
                links.append(full)

        meta: dict[str, str] = {}
        for node in tree.css("meta"):
            key = node.attributes.get("name") or node.attributes.get("property")
            val = node.attributes.get("content", "")
            if key:
                meta[key] = val

        return title, raw_text, links, meta, canonical


# ─────────────────────────────────────────────────────────────
# COMBINED EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """
    Runs all extractors in sequence and produces a fully populated
    ExtractedContent object.
    """

    def __init__(
        self,
        article_extractor:  Optional[ArticleExtractor]  = None,
        jsonld_extractor:   Optional[JSONLDExtractor]   = None,
        entity_extractor:   Optional[EntityExtractor]   = None,
        language_detector:  Optional[LanguageDetector]  = None,
        target_languages:   Optional[list[str]]         = None,
    ):
        self.articles  = article_extractor  or ArticleExtractor()
        self.jsonld    = jsonld_extractor   or JSONLDExtractor()
        self.entities  = entity_extractor   or EntityExtractor()
        self.langdet   = language_detector  or LanguageDetector(target_languages)
        self._base     = FastParser()

    def run(self, html: str, base_url: str) -> ExtractedContent:
        # 1. Base parse
        title, raw_text, links, meta, canonical = self._base.parse(html, base_url)

        # 2. Article extraction
        article, author, date, lang_trafil = self.articles.extract(html, base_url)

        # 3. JSON-LD
        json_ld, schema_org = self.jsonld.extract(html)

        # 4. Language detection
        text_for_lang = article or raw_text
        language = lang_trafil or self.langdet.detect(text_for_lang)

        # 5. NER (run on article if available, else raw)
        ner_text = article or raw_text
        ents = self.entities.extract(ner_text)

        # 6. Override title from JSON-LD if missing
        if not title and "headline" in schema_org:
            title = schema_org["headline"]

        return ExtractedContent(
            title=title,
            text=raw_text,
            article=article,
            author=author or schema_org.get("author", {}).get("name", ""),
            date=date   or schema_org.get("datePublished", ""),
            language=language,
            json_ld=json_ld,
            schema_org=schema_org,
            entities=ents,
            links=links,
            meta=meta,
            canonical=canonical,
        )
