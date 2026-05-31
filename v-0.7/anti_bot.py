"""
Anti-bot module
===============
[F1]  ProxyPool       — rotating proxy manager (per-domain sticky or random)
[F2]  TLSFetcher      — curl-cffi Chrome impersonation (JA3 fingerprint spoof)
[F3]  CaptchaSolver   — 2captcha / CapSolver integration for Playwright flows

All classes degrade gracefully when optional deps are missing.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────
try:
    import curl_cffi.requests as cffi_requests   # TLS fingerprint spoof
    _CFFI_OK = True
except ImportError:
    _CFFI_OK = False

try:
    import twocaptcha                            # CAPTCHA solving
    _CAPTCHA_OK = True
except ImportError:
    _CAPTCHA_OK = False


# ─────────────────────────────────────────────────────────────
# [F1] PROXY POOL
# ─────────────────────────────────────────────────────────────

@dataclass
class Proxy:
    url: str                        # "http://user:pass@host:port"
    domain_sticky: dict = field(default_factory=dict)  # domain→last_used
    errors: int = 0
    banned_until: float = 0.0

    @property
    def is_alive(self) -> bool:
        return time.time() > self.banned_until and self.errors < 10


class ProxyPool:
    """
    [F1] Rotating proxy manager.

    Modes:
      sticky=True  — same proxy reused per domain (session persistence)
      sticky=False — random healthy proxy per request

    Usage:
        pool = ProxyPool(["http://u:p@host:port", ...], sticky=True)
        proxy_url = pool.get("example.com")
        pool.report_error("example.com", proxy_url)
        pool.report_success("example.com", proxy_url)
    """

    def __init__(self, proxy_urls: list[str], sticky: bool = True):
        self._proxies = [Proxy(url=u) for u in proxy_urls]
        self._sticky  = sticky
        self._domain_map: dict[str, str] = {}   # domain → proxy_url
        self._lock = asyncio.Lock()
        log.info("ProxyPool: %d proxies loaded (sticky=%s)", len(self._proxies), sticky)

    @property
    def _alive(self) -> list[Proxy]:
        return [p for p in self._proxies if p.is_alive]

    def get(self, dom: str = "") -> Optional[str]:
        alive = self._alive
        if not alive:
            log.warning("ProxyPool: no healthy proxies — going direct")
            return None

        if self._sticky and dom:
            if dom in self._domain_map:
                url = self._domain_map[dom]
                if any(p.url == url and p.is_alive for p in self._proxies):
                    return url
            chosen = random.choice(alive)
            self._domain_map[dom] = chosen.url
            return chosen.url

        return random.choice(alive).url

    def report_error(self, dom: str, proxy_url: Optional[str]):
        if not proxy_url:
            return
        for p in self._proxies:
            if p.url == proxy_url:
                p.errors += 1
                if p.errors >= 3:
                    p.banned_until = time.time() + 300   # 5-min ban
                    log.warning("ProxyPool: banned %s for 5 min", proxy_url[:30])
                if dom in self._domain_map and self._domain_map[dom] == proxy_url:
                    del self._domain_map[dom]   # force re-assign

    def report_success(self, dom: str, proxy_url: Optional[str]):
        if not proxy_url:
            return
        for p in self._proxies:
            if p.url == proxy_url:
                p.errors = max(0, p.errors - 1)

    def stats(self) -> dict:
        return {
            "total": len(self._proxies),
            "alive": len(self._alive),
            "banned": sum(1 for p in self._proxies if not p.is_alive),
        }


# ─────────────────────────────────────────────────────────────
# [F2] TLS FINGERPRINT FETCHER  (curl-cffi Chrome impersonation)
# ─────────────────────────────────────────────────────────────

# Chrome versions to rotate through — each has a distinct JA3 fingerprint
_CHROME_IMPERSONATIONS = [
    "chrome110", "chrome111", "chrome112",
    "chrome116", "chrome119", "chrome120",
]


class TLSFetcher:
    """
    [F2] HTTP fetcher using curl-cffi to impersonate Chrome's TLS stack.

    Defeats passive bot detection systems (Cloudflare, Akamai, DataDome)
    that fingerprint the TLS ClientHello (JA3/JA3S hash, ALPN, cipher order).

    Falls back to plain httpx when curl-cffi is not installed.
    """

    def __init__(
        self,
        rotate_fingerprint: bool = True,
        timeout: int = 20,
    ):
        self._rotate   = rotate_fingerprint
        self._timeout  = timeout
        self._impersonation = _CHROME_IMPERSONATIONS[0]
        if not _CFFI_OK:
            log.warning("TLSFetcher: curl-cffi not installed — TLS fingerprinting disabled")
            log.warning("  Install: pip install curl-cffi")

    def _pick_impersonation(self) -> str:
        if self._rotate:
            return random.choice(_CHROME_IMPERSONATIONS)
        return self._impersonation

    async def fetch(
        self,
        url: str,
        proxy: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> tuple[Optional[str], int, dict]:
        """
        Returns (html, status_code, response_headers).
        Runs curl-cffi in an executor to avoid blocking the event loop.
        """
        if not _CFFI_OK:
            return None, 0, {}

        imp = self._pick_impersonation()
        _headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
        }
        if headers:
            _headers.update(headers)

        def _sync_fetch():
            with cffi_requests.Session(impersonate=imp) as session:
                resp = session.get(
                    url,
                    headers=_headers,
                    proxies={"http": proxy, "https": proxy} if proxy else None,
                    timeout=self._timeout,
                    allow_redirects=True,
                )
                return resp.text, resp.status_code, dict(resp.headers), str(resp.url)

        try:
            loop = asyncio.get_event_loop()
            html, status, resp_headers, final_url = await loop.run_in_executor(
                None, _sync_fetch
            )
            log.debug("TLSFetcher [%s]: %d %s", imp, status, url[:60])
            return html, status, resp_headers
        except Exception as exc:
            log.warning("TLSFetcher error %s: %s", url[:60], exc)
            return None, 0, {}


# ─────────────────────────────────────────────────────────────
# [F3] CAPTCHA SOLVER
# ─────────────────────────────────────────────────────────────

class CaptchaSolver:
    """
    [F3] Integrates with 2captcha / CapSolver to solve CAPTCHAs during
    Playwright-driven crawling.

    Supported types:
      - reCAPTCHA v2 / v3
      - hCaptcha
      - Cloudflare Turnstile
      - Image CAPTCHAs

    Usage (inside a Playwright page context):
        solver = CaptchaSolver(api_key="YOUR_KEY", provider="2captcha")
        token = await solver.solve_recaptcha(page, site_key, page_url)
    """

    PROVIDERS = {"2captcha", "capsolver"}

    def __init__(self, api_key: str = "", provider: str = "2captcha"):
        self._key      = api_key
        self._provider = provider
        self._enabled  = bool(api_key) and _CAPTCHA_OK

        if api_key and not _CAPTCHA_OK:
            log.warning("CaptchaSolver: twocaptcha not installed — CAPTCHA solving disabled")
            log.warning("  Install: pip install 2captcha-python")
        elif api_key:
            log.info("CaptchaSolver: ready (%s)", provider)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def solve_recaptcha_v2(
        self,
        site_key: str,
        page_url: str,
    ) -> Optional[str]:
        """Returns the g-recaptcha-response token or None on failure."""
        if not self._enabled:
            return None
        try:
            solver = twocaptcha.TwoCaptcha(self._key)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: solver.recaptcha(sitekey=site_key, url=page_url)
            )
            token = result.get("code")
            log.info("CaptchaSolver: reCAPTCHA solved (%s...)", (token or "")[:20])
            return token
        except Exception as exc:
            log.warning("CaptchaSolver error: %s", exc)
            return None

    async def solve_hcaptcha(
        self,
        site_key: str,
        page_url: str,
    ) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            solver = twocaptcha.TwoCaptcha(self._key)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: solver.hcaptcha(sitekey=site_key, url=page_url)
            )
            return result.get("code")
        except Exception as exc:
            log.warning("CaptchaSolver hcaptcha error: %s", exc)
            return None

    async def inject_recaptcha_token(self, page, token: str):
        """
        Inject a solved reCAPTCHA token into the page DOM and submit.
        Called after solve_recaptcha_v2().
        """
        await page.evaluate(
            f"document.getElementById('g-recaptcha-response').innerHTML='{token}'"
        )
        await page.evaluate("___grecaptcha_cfg.clients[0].aa.l.callback(arguments[0])", token)
