"""
Utility functions: URL normalisation, priority scoring,
weighted simhash, Hamming distance, LSH index.
"""
from __future__ import annotations
import hashlib
import re
from collections import Counter
from urllib.parse import urljoin, urlparse, urlunparse, urlencode, parse_qs

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid",
}


# ─────────────────────────────────────────────────────────────
# URL helpers
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


def resolve(href: str, base: str) -> str:
    return normalise(urljoin(base, href))


def domain(url: str) -> str:
    return urlparse(url).netloc.lower()


# ─────────────────────────────────────────────────────────────
# Priority scoring
# ─────────────────────────────────────────────────────────────

def score_url(url: str) -> int:
    path = urlparse(url).path.lower()
    if path in ("/", ""):                        return 100
    if re.search(r"/categor|/catalog|/topic",    path): return 80
    if re.search(r"/product|/item|/p/|/article", path): return 60
    if re.search(r"/blog|/news|/post",           path): return 55
    if re.search(r"page=\d+|/page/\d+",          url):  return 20
    return 40


# ─────────────────────────────────────────────────────────────
# Hashing
# ─────────────────────────────────────────────────────────────

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def weighted_simhash(text: str, bits: int = 64) -> int:
    """64-bit simhash with Counter frequency weighting (v5 fix preserved)."""
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
    """LSH index for O(1) amortised near-duplicate lookup (v5)."""
    BANDS = 8; ROWS = 8; THRESHOLD = 4

    def __init__(self):
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(self.BANDS)]

    def _band_vals(self, sh: int) -> list[int]:
        mask = (1 << self.ROWS) - 1
        return [(sh >> (b * self.ROWS)) & mask for b in range(self.BANDS)]

    def seen(self, sh: int) -> bool:
        for bi, bv in enumerate(self._band_vals(sh)):
            if any(hamming(sh, c) <= self.THRESHOLD
                   for c in self._buckets[bi].get(bv, [])):
                return True
        return False

    def add(self, sh: int):
        for bi, bv in enumerate(self._band_vals(sh)):
            self._buckets[bi].setdefault(bv, []).append(sh)
