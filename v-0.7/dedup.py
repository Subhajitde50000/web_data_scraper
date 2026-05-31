"""
Deduplication module
====================
[F8]  SemanticDedup    — sentence-transformer embeddings + Qdrant ANN search
      ExactDedup       — SHA-256 bloom filter (persistent)
      NearDedup        — LSH simhash index
      DedupPipeline    — runs all three layers in order
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

from .utils import sha256, weighted_simhash, SimhashIndex

log = logging.getLogger(__name__)

BLOOM_PATH = "bloom.pkl"

# ── optional imports ──────────────────────────────────────────
try:
    from pybloom_live import ScalableBloomFilter
    _BLOOM_OK = True
except ImportError:
    _BLOOM_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, SearchRequest,
    )
    _QDRANT_OK = True
except ImportError:
    _QDRANT_OK = False


# ─────────────────────────────────────────────────────────────
# EXACT DEDUP  (persistent bloom filter)
# ─────────────────────────────────────────────────────────────

class ExactDedup:
    """SHA-256 exact duplicate detection via persistent Bloom filter."""

    def __init__(self, bloom_path: str = BLOOM_PATH):
        self._path = bloom_path
        if _BLOOM_OK:
            self._bloom = (pickle.load(open(bloom_path, "rb"))
                           if os.path.exists(bloom_path)
                           else ScalableBloomFilter(
                               initial_capacity=100_000, error_rate=0.001))
            self._use_bloom = True
        else:
            self._seen: set[str] = set()
            self._use_bloom = False
            log.warning("ExactDedup: pybloom-live not installed — using in-memory set (not persistent)")

    def seen(self, text: str) -> bool:
        h = sha256(text)
        if self._use_bloom:
            return h in self._bloom
        return h in self._seen

    def add(self, text: str):
        h = sha256(text)
        if self._use_bloom:
            self._bloom.add(h)
        else:
            self._seen.add(h)

    def save(self):
        if self._use_bloom:
            pickle.dump(self._bloom, open(self._path, "wb"))
            log.debug("ExactDedup: bloom filter saved")


# ─────────────────────────────────────────────────────────────
# NEAR DEDUP  (LSH simhash)
# ─────────────────────────────────────────────────────────────

class NearDedup:
    """
    Near-duplicate detection using weighted simhash + LSH index.
    O(1) amortised lookup — faster than linear scan (v5 fix preserved).
    """

    def __init__(self):
        self._lsh = SimhashIndex()

    def seen(self, text: str) -> bool:
        sh = weighted_simhash(text)
        return self._lsh.seen(sh)

    def add(self, text: str) -> int:
        sh = weighted_simhash(text)
        self._lsh.add(sh)
        return sh


# ─────────────────────────────────────────────────────────────
# [F8] SEMANTIC DEDUP  (sentence-transformers + Qdrant)
# ─────────────────────────────────────────────────────────────

class SemanticDedup:
    """
    [F8] Near-duplicate detection using dense embeddings.

    Embeds page text with a small sentence-transformer model,
    queries Qdrant for ANN neighbours, and flags pages whose
    cosine similarity exceeds a threshold as semantic duplicates.

    Catches paraphrased content that simhash misses (same article
    rewritten across syndication networks).

    Requires:
      pip install sentence-transformers qdrant-client

    Qdrant:
      docker run -p 6333:6333 qdrant/qdrant  (local)
      or use Qdrant Cloud (free tier available)
    """

    DEFAULT_MODEL     = "all-MiniLM-L6-v2"    # 384-dim, 80 MB, fast
    COLLECTION        = "page_embeddings"
    SIM_THRESHOLD     = 0.92                   # cosine similarity cutoff

    def __init__(
        self,
        qdrant_url:  str   = "http://localhost:6333",
        model_name:  str   = DEFAULT_MODEL,
        threshold:   float = SIM_THRESHOLD,
        enabled:     bool  = True,
    ):
        self._enabled    = enabled and _ST_OK and _QDRANT_OK
        self._threshold  = threshold
        self._model      = None
        self._client     = None
        self._dim        = 0
        self._counter    = 0

        if not enabled:
            return
        if not _ST_OK:
            log.warning("SemanticDedup: sentence-transformers not installed")
            log.warning("  pip install sentence-transformers")
            return
        if not _QDRANT_OK:
            log.warning("SemanticDedup: qdrant-client not installed")
            log.warning("  pip install qdrant-client")
            return

        try:
            self._model  = SentenceTransformer(model_name)
            self._dim    = self._model.get_sentence_embedding_dimension()
            self._client = QdrantClient(url=qdrant_url, timeout=10)
            self._ensure_collection()
            log.info("SemanticDedup: %s (%d-dim) → Qdrant %s",
                     model_name, self._dim, qdrant_url)
        except Exception as exc:
            log.warning("SemanticDedup: init failed — %s", exc)
            self._enabled = False

    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if self.COLLECTION not in existing:
            self._client.create_collection(
                self.COLLECTION,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
            )

    def embed(self, text: str) -> Optional[list[float]]:
        if not self._enabled or not self._model:
            return None
        try:
            vec = self._model.encode(text[:1000], normalize_embeddings=True)
            return vec.tolist()
        except Exception:
            return None

    def seen(self, text: str) -> bool:
        """Return True if a semantically similar page is already stored."""
        if not self._enabled:
            return False
        vec = self.embed(text)
        if not vec:
            return False
        try:
            results = self._client.search(
                collection_name=self.COLLECTION,
                query_vector=vec,
                limit=1,
                score_threshold=self._threshold,
            )
            return len(results) > 0
        except Exception:
            return False

    def add(self, text: str, url: str) -> Optional[list[float]]:
        """Embed text and store in Qdrant. Returns the vector."""
        if not self._enabled:
            return None
        vec = self.embed(text)
        if not vec:
            return None
        try:
            self._counter += 1
            self._client.upsert(
                collection_name=self.COLLECTION,
                points=[PointStruct(
                    id=self._counter,
                    vector=vec,
                    payload={"url": url},
                )],
            )
        except Exception as exc:
            log.debug("SemanticDedup.add error: %s", exc)
        return vec


# ─────────────────────────────────────────────────────────────
# DEDUP PIPELINE
# ─────────────────────────────────────────────────────────────

class DedupPipeline:
    """
    Three-stage deduplication:
      1. SHA-256 exact match   (O(1) bloom filter)
      2. Simhash near-dup      (O(1) LSH)
      3. Semantic similarity   (O(log n) Qdrant ANN)

    Returns (is_duplicate, reason_str).
    """

    def __init__(
        self,
        exact:    Optional[ExactDedup]    = None,
        near:     Optional[NearDedup]     = None,
        semantic: Optional[SemanticDedup] = None,
    ):
        self.exact    = exact    or ExactDedup()
        self.near     = near     or NearDedup()
        self.semantic = semantic or SemanticDedup(enabled=False)

    def check_and_add(
        self, text: str, url: str
    ) -> tuple[bool, str, Optional[list[float]]]:
        """
        Returns (is_dup, reason, embedding_vector).
        Adds to all stores if not duplicate.
        """
        if self.exact.seen(text):
            return True, "exact", None

        if self.near.seen(text):
            return True, "near", None

        if self.semantic.seen(text):
            return True, "semantic", None

        # Not a duplicate — add to all stores
        self.exact.add(text)
        self.near.add(text)
        vec = self.semantic.add(text, url)
        return False, "", vec

    def save(self):
        self.exact.save()
