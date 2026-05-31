"""
Storage module
==============
[F15]  WAL append-only event log (aiofiles)
[F16]  OpenSearch / Elasticsearch full-text index streaming
       PostgresWriter   — asyncpg batched UPSERT with JSONB columns
       SQLiteWriter     — aiosqlite fallback
       make_writer()    — factory with auto-fallback
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import aiosqlite

from .models import ScrapedItem, CrawlEvent

log = logging.getLogger(__name__)

try:
    import asyncpg
    _PG_OK = True
except ImportError:
    _PG_OK = False

try:
    import aiofiles
    _AIOFILES_OK = True
except ImportError:
    _AIOFILES_OK = False

try:
    from opensearchpy import AsyncOpenSearch
    _OS_OK = True
except ImportError:
    _OS_OK = False


# ─────────────────────────────────────────────────────────────
# [F15] WAL APPEND-ONLY EVENT LOG
# ─────────────────────────────────────────────────────────────

class WALLog:
    """
    [F15] Append-only event log for crash recovery and audit.

    Every significant event (fetched, parsed, stored, error, skipped)
    is written to a JSONL file before the DB commit. On restart, the
    log can be replayed to recover partially-committed state.

    Requires: pip install aiofiles
    """

    def __init__(self, path: str = "crawler.wal"):
        self._path    = path
        self._enabled = _AIOFILES_OK
        self._fh      = None
        self._lock    = asyncio.Lock()

        if not _AIOFILES_OK:
            log.warning("WALLog: aiofiles not installed — WAL disabled")
            log.warning("  pip install aiofiles")

    async def open(self):
        if not self._enabled:
            return
        self._fh = await aiofiles.open(self._path, "a", encoding="utf-8")
        log.info("WALLog: opened %s", self._path)

    async def close(self):
        if self._fh:
            await self._fh.flush()
            await self._fh.close()

    async def write(self, event: CrawlEvent):
        if not self._enabled or not self._fh:
            return
        line = json.dumps({
            "event":  event.event,
            "url":    event.url,
            "detail": event.detail,
            "ts":     event.ts or time.time(),
        }, ensure_ascii=False) + "\n"
        async with self._lock:
            await self._fh.write(line)

    async def log_fetched(self, url: str, status: int, ms: float):
        await self.write(CrawlEvent(
            event="fetched", url=url,
            detail=f"status={status} ms={ms:.0f}", ts=time.time()
        ))

    async def log_stored(self, url: str):
        await self.write(CrawlEvent(event="stored", url=url, ts=time.time()))

    async def log_error(self, url: str, reason: str):
        await self.write(CrawlEvent(event="error", url=url, detail=reason, ts=time.time()))

    async def log_skipped(self, url: str, reason: str):
        await self.write(CrawlEvent(event="skipped", url=url, detail=reason, ts=time.time()))

    async def replay(self) -> list[dict]:
        """Read all WAL events — useful for crash recovery analysis."""
        events = []
        try:
            async with aiofiles.open(self._path, "r") as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            pass
        return events


# ─────────────────────────────────────────────────────────────
# [F16] OPENSEARCH WRITER
# ─────────────────────────────────────────────────────────────

OS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "url":        {"type": "keyword"},
            "title":      {"type": "text", "analyzer": "english"},
            "article":    {"type": "text", "analyzer": "english"},
            "author":     {"type": "keyword"},
            "date":       {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time||epoch_millis"},
            "language":   {"type": "keyword"},
            "entities":   {"type": "object"},
            "schema_org": {"type": "object"},
            "scraped_at": {"type": "date"},
            "domain":     {"type": "keyword"},
        }
    }
}


class OpenSearchWriter:
    """
    [F16] Streams scraped content to OpenSearch / Elasticsearch.

    Complements PostgreSQL: Postgres for structured relational data,
    OpenSearch for full-text search, faceting, and aggregations.

    Requires: pip install opensearch-py
    """

    BATCH_SIZE     = 50
    FLUSH_INTERVAL = 2.0
    INDEX          = "pages"

    def __init__(self, hosts: list[str], index: str = "pages", **kwargs):
        self._hosts   = hosts
        self._index   = index
        self._kwargs  = kwargs
        self._client: Optional[AsyncOpenSearch] = None
        self._q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

        if not _OS_OK:
            log.warning("OpenSearchWriter: opensearch-py not installed — OpenSearch disabled")
            log.warning("  pip install opensearch-py")

    async def start(self):
        if not _OS_OK:
            return
        self._client = AsyncOpenSearch(hosts=self._hosts, **self._kwargs)
        await self._ensure_index()
        self._task = asyncio.create_task(self._writer_loop())
        log.info("OpenSearchWriter: streaming to %s/%s", self._hosts, self._index)

    async def _ensure_index(self):
        try:
            if not await self._client.indices.exists(index=self._index):
                await self._client.indices.create(
                    index=self._index, body=OS_INDEX_MAPPING
                )
                log.info("OpenSearchWriter: created index '%s'", self._index)
        except Exception as exc:
            log.warning("OpenSearchWriter: index setup failed — %s", exc)

    async def push(self, item: ScrapedItem):
        if not _OS_OK or not self._client:
            return
        await self._q.put(item)

    async def stop(self):
        await self._q.put(None)
        await self._stopped.wait()
        if self._client:
            await self._client.close()

    async def _writer_loop(self):
        batch: list[ScrapedItem] = []
        last_flush = time.monotonic()

        while True:
            try:
                timeout = max(0.0, self.FLUSH_INTERVAL - (time.monotonic() - last_flush))
                item = await asyncio.wait_for(self._q.get(), timeout=timeout)
                if item is None:
                    if batch:
                        await self._bulk_index(batch)
                    self._stopped.set()
                    return
                batch.append(item)
                self._q.task_done()
            except asyncio.TimeoutError:
                pass

            if len(batch) >= self.BATCH_SIZE or (
                batch and time.monotonic() - last_flush >= self.FLUSH_INTERVAL
            ):
                await self._bulk_index(batch)
                batch = []
                last_flush = time.monotonic()

    async def _bulk_index(self, batch: list[ScrapedItem]):
        from urllib.parse import urlparse
        actions = []
        for item in batch:
            c = item.content
            actions.append({"index": {"_index": self._index, "_id": item.url}})
            actions.append({
                "url":        item.url,
                "title":      c.title,
                "article":    c.article or c.text,
                "author":     c.author,
                "date":       c.date or None,
                "language":   c.language,
                "entities":   c.entities,
                "schema_org": c.schema_org,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "domain":     urlparse(item.url).netloc,
            })
        try:
            await self._client.bulk(body=actions)
            log.debug("OpenSearch: indexed %d docs", len(batch))
        except Exception as exc:
            log.warning("OpenSearch bulk error: %s", exc)


# ─────────────────────────────────────────────────────────────
# POSTGRESQL WRITER
# ─────────────────────────────────────────────────────────────

PG_DDL = """
CREATE TABLE IF NOT EXISTS pages (
    id            BIGSERIAL PRIMARY KEY,
    url           TEXT      UNIQUE NOT NULL,
    final_url     TEXT,
    title         TEXT,
    body          TEXT,
    article       TEXT,
    author        TEXT,
    pub_date      TEXT,
    language      TEXT,
    status_code   SMALLINT,
    content_type  TEXT,
    response_ms   REAL,
    etag          TEXT,
    last_modified TEXT,
    content_hash  TEXT,
    simhash       BIGINT,
    rendered_by   TEXT      DEFAULT 'httpx',
    http_version  TEXT,
    proxy_used    TEXT,
    entities      JSONB,
    schema_org    JSONB,
    json_ld       JSONB,
    embedding     VECTOR(384),
    scraped_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS pages_scraped_at    ON pages (scraped_at);
CREATE INDEX IF NOT EXISTS pages_content_hash  ON pages (content_hash);
CREATE INDEX IF NOT EXISTS pages_language      ON pages (language);
CREATE INDEX IF NOT EXISTS pages_entities_gin  ON pages USING GIN (entities);

CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);
CREATE INDEX IF NOT EXISTS links_dst ON links (dst);

CREATE TABLE IF NOT EXISTS done (
    url        TEXT PRIMARY KEY,
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);
"""

PG_FTS = """
ALTER TABLE pages ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(title,'') || ' ' ||
            coalesce(article,'') || ' ' ||
            coalesce(body,''))
    ) STORED;
CREATE INDEX IF NOT EXISTS pages_fts ON pages USING GIN (fts);
"""


class PostgresWriter:
    BATCH_SIZE = 100; FLUSH_INTERVAL = 1.0

    def __init__(self, pg_dsn: str, wal: Optional[WALLog] = None):
        self._dsn  = pg_dsn
        self._wal  = wal
        self._pool = None
        self._q: asyncio.Queue[Optional[ScrapedItem]] = asyncio.Queue(maxsize=1000)
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self):
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=2, max_size=10, command_timeout=30
        )
        async with self._pool.acquire() as conn:
            await conn.execute(PG_DDL)
            try:
                await conn.execute(PG_FTS)
            except Exception:
                pass
        self._task = asyncio.create_task(self._writer_loop())
        log.info("PostgresWriter: pool open")

    async def push(self, item: ScrapedItem):
        await self._q.put(item)

    async def stop(self):
        await self._q.put(None)
        await self._stopped.wait()
        if self._pool:
            await self._pool.close()

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
                timeout = max(0.0, self.FLUSH_INTERVAL - (time.monotonic() - last_flush))
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
            if len(batch) >= self.BATCH_SIZE or (
                batch and time.monotonic() - last_flush >= self.FLUSH_INTERVAL
            ):
                await self._commit(batch)
                batch = []
                last_flush = time.monotonic()

    async def _commit(self, batch: list[ScrapedItem]):
        pages, links, done = [], [], []
        for item in batch:
            c = item.content
            cr = item.crawl
            embedding = c.embedding  # list[float] | None
            pages.append((
                item.url, cr.final_url, c.title, c.text, c.article,
                c.author, c.date, c.language,
                cr.status_code, cr.content_type, cr.response_time_ms,
                cr.etag, cr.last_modified, cr.content_hash, cr.simhash,
                cr.rendered_by, cr.http_version, cr.proxy_used,
                json.dumps(c.entities) if c.entities else None,
                json.dumps(c.schema_org) if c.schema_org else None,
                json.dumps(c.json_ld) if c.json_ld else None,
                embedding,
            ))
            links.extend((item.url, lnk) for lnk in (c.links or []))
            done.append((item.url,))

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """INSERT INTO pages
                       (url,final_url,title,body,article,author,pub_date,language,
                        status_code,content_type,response_ms,etag,last_modified,
                        content_hash,simhash,rendered_by,http_version,proxy_used,
                        entities,schema_org,json_ld,embedding)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,
                               $14,$15,$16,$17,$18,$19,$20,$21,$22)
                       ON CONFLICT (url) DO UPDATE SET
                         final_url=EXCLUDED.final_url, title=EXCLUDED.title,
                         body=EXCLUDED.body, article=EXCLUDED.article,
                         language=EXCLUDED.language,
                         content_hash=EXCLUDED.content_hash,
                         entities=EXCLUDED.entities,
                         schema_org=EXCLUDED.schema_org,
                         rendered_by=EXCLUDED.rendered_by,
                         scraped_at=NOW()""",
                    pages,
                )
                await conn.executemany(
                    "INSERT INTO links (src,dst) VALUES ($1,$2) ON CONFLICT DO NOTHING",
                    links,
                )
                await conn.executemany(
                    "INSERT INTO done (url) VALUES ($1) ON CONFLICT DO NOTHING",
                    done,
                )
        if self._wal:
            for item in batch:
                await self._wal.log_stored(item.url)
        log.debug("PG: committed %d pages", len(pages))


class SQLiteWriter:
    BATCH_SIZE = 100; FLUSH_INTERVAL = 1.0

    def __init__(self, db_path: str = "output.db", wal: Optional[WALLog] = None):
        self._db_path = db_path
        self._wal     = wal
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
        return "", ""

    async def _writer_loop(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY, url TEXT UNIQUE, final_url TEXT,
                    title TEXT, body TEXT, article TEXT, author TEXT,
                    pub_date TEXT, language TEXT, status_code INTEGER,
                    content_type TEXT, response_ms REAL, etag TEXT,
                    last_modified TEXT, content_hash TEXT, simhash INTEGER,
                    rendered_by TEXT, entities TEXT, schema_org TEXT,
                    scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS links (src TEXT, dst TEXT, PRIMARY KEY(src,dst));
                CREATE TABLE IF NOT EXISTS done (url TEXT PRIMARY KEY);
            """)
            await db.commit()
            batch: list[ScrapedItem] = []
            last_flush = time.monotonic()
            while True:
                try:
                    timeout = max(0.0, self.FLUSH_INTERVAL - (time.monotonic() - last_flush))
                    item = await asyncio.wait_for(self._q.get(), timeout=timeout)
                    if item is None:
                        if batch:
                            await self._commit(db, batch)
                        self._stopped.set()
                        return
                    batch.append(item)
                    self._q.task_done()
                except asyncio.TimeoutError:
                    pass
                if len(batch) >= self.BATCH_SIZE or (
                    batch and time.monotonic() - last_flush >= self.FLUSH_INTERVAL
                ):
                    await self._commit(db, batch)
                    batch = []
                    last_flush = time.monotonic()

    async def _commit(self, db, batch: list[ScrapedItem]):
        pages, links, done = [], [], []
        for item in batch:
            c = item.content; cr = item.crawl
            pages.append((
                item.url, cr.final_url, c.title, c.text, c.article,
                c.author, c.date, c.language, cr.status_code,
                cr.content_type, cr.response_time_ms, cr.etag,
                cr.last_modified, cr.content_hash, cr.simhash,
                cr.rendered_by,
                json.dumps(c.entities) if c.entities else None,
                json.dumps(c.schema_org) if c.schema_org else None,
            ))
            links.extend((item.url, lnk) for lnk in (c.links or []))
            done.append((item.url,))
        await db.executemany(
            "INSERT OR REPLACE INTO pages (url,final_url,title,body,article,author,pub_date,"
            "language,status_code,content_type,response_ms,etag,last_modified,content_hash,"
            "simhash,rendered_by,entities,schema_org) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            pages)
        await db.executemany("INSERT OR IGNORE INTO links (src,dst) VALUES (?,?)", links)
        await db.executemany("INSERT OR IGNORE INTO done (url) VALUES (?)", done)
        await db.commit()
        if self._wal:
            for item in batch:
                await self._wal.log_stored(item.url)
        log.debug("SQLite: committed %d pages", len(pages))


def make_writer(
    pg_dsn:      Optional[str],
    sqlite_path: str        = "output.db",
    wal:         Optional[WALLog] = None,
):
    if pg_dsn and _PG_OK:
        return PostgresWriter(pg_dsn, wal)
    log.warning("Storage: using SQLite fallback (set pg_dsn for production)")
    return SQLiteWriter(sqlite_path, wal)
