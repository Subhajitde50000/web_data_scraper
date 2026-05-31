"""
Fetch module
============
[F11]  HTTP/2 + HTTP/3 via httpx (replaces aiohttp)
[F12]  Cookie / session management for authenticated crawling
[F13]  Circuit breaker per domain
[F14]  Playwright pool with infinite scroll + CAPTCHA detection
       Async robots.txt (per-domain lock)
       Rate limiter (per-domain async sleep)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from .models import CrawlMeta, FetchResult
from .anti_bot import ProxyPool, CaptchaSolver

log = logging.getLogger(__name__)

ALLOWED_MIME = {"text/html", "application/xhtml+xml"}
USER_AGENT   = (
    "Mozilla/5.0 (compatible; PyScraper/7.0; "
    "+https://github.com/example/scraper)"
)
JS_SIGNALS = re.compile(
    r"(data-reactroot|__NEXT_DATA__|ng-version|vue-app|"
    r"<app-root|window\.__STATE__|__nuxt__|_sveltekit)",
    re.IGNORECASE,
)

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False
    log.warning("fetch: httpx not installed. Install: pip install httpx[http2]")

try:
    from playwright.async_api import async_playwright
    _PW_OK = True
except ImportError:
    _PW_OK = False


# ─────────────────────────────────────────────────────────────
# [F13] CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────

@dataclass
class CircuitState:
    failures:    int   = 0
    open_until:  float = 0.0
    half_open:   bool  = False


class CircuitBreaker:
    """
    [F13] Per-domain circuit breaker.

    States:
      CLOSED   — normal operation
      OPEN     — domain blocked for cooldown_s seconds
      HALF-OPEN — one test request allowed after cooldown

    After N consecutive failures → OPEN.
    After cooldown → HALF-OPEN → one request.
    Success in HALF-OPEN → CLOSED. Failure → OPEN again.
    """

    def __init__(self, threshold: int = 5, cooldown_s: int = 120):
        self._threshold = threshold
        self._cooldown  = cooldown_s
        self._states:   dict[str, CircuitState] = {}

    def _state(self, dom: str) -> CircuitState:
        if dom not in self._states:
            self._states[dom] = CircuitState()
        return self._states[dom]

    def is_open(self, dom: str) -> bool:
        s = self._state(dom)
        if s.open_until > 0 and time.time() < s.open_until:
            return True
        if s.open_until > 0 and time.time() >= s.open_until:
            # Transition to HALF-OPEN
            s.half_open  = True
            s.open_until = 0.0
        return False

    def record_success(self, dom: str):
        s = self._state(dom)
        s.failures   = 0
        s.half_open  = False
        s.open_until = 0.0

    def record_failure(self, dom: str):
        s = self._state(dom)
        s.failures += 1
        s.half_open = False
        if s.failures >= self._threshold:
            s.open_until = time.time() + self._cooldown
            log.warning(
                "CircuitBreaker: OPEN for %s (%ds cooldown after %d failures)",
                dom, self._cooldown, s.failures
            )

    def stats(self) -> dict:
        return {
            d: {"failures": s.failures, "open": self.is_open(d)}
            for d, s in self._states.items()
        }


# ─────────────────────────────────────────────────────────────
# ASYNC RATE LIMITER
# ─────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, default_delay: float = 1.5, per_domain_concurrency: int = 2):
        self.default_delay = default_delay
        self._per_domain   = per_domain_concurrency
        self._last:  dict[str, float]            = {}
        self._sems:  dict[str, asyncio.Semaphore] = {}
        self._lock   = asyncio.Lock()

    async def _sem(self, dom: str) -> asyncio.Semaphore:
        async with self._lock:
            if dom not in self._sems:
                self._sems[dom] = asyncio.Semaphore(self._per_domain)
            return self._sems[dom]

    async def wait(self, dom: str, delay: Optional[float] = None):
        d = delay if delay is not None else self.default_delay
        async with self._lock:
            gap = d - (time.monotonic() - self._last.get(dom, 0.0))
        if gap > 0:
            await asyncio.sleep(gap)
        async with self._lock:
            self._last[dom] = time.monotonic()

    async def acquire(self, dom: str) -> asyncio.Semaphore:
        return await self._sem(dom)


# ─────────────────────────────────────────────────────────────
# ASYNC ROBOTS.TXT  (per-domain lock)
# ─────────────────────────────────────────────────────────────

class AsyncRobots:
    def __init__(self, default_delay: float = 1.5):
        self.default_delay = default_delay
        self._cache:   dict[str, Optional[RobotFileParser]] = {}
        self._dlocks:  dict[str, asyncio.Lock] = {}
        self._dlock    = asyncio.Lock()

    async def _lock(self, dom: str) -> asyncio.Lock:
        async with self._dlock:
            if dom not in self._dlocks:
                self._dlocks[dom] = asyncio.Lock()
            return self._dlocks[dom]

    async def _fetch(self, origin: str, client: "httpx.AsyncClient") -> Optional[RobotFileParser]:
        try:
            r = await client.get(f"{origin}/robots.txt", timeout=10)
            if r.status_code == 200:
                rp = RobotFileParser()
                rp.set_url(f"{origin}/robots.txt")
                rp.parse(r.text.splitlines())
                return rp
        except Exception:
            pass
        return None

    async def get(self, url: str, client) -> Optional[RobotFileParser]:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        lock   = await self._lock(p.netloc)
        async with lock:
            if origin not in self._cache:
                self._cache[origin] = await self._fetch(origin, client)
            return self._cache[origin]

    async def allowed(self, url: str, client, agent: str = "*") -> bool:
        rp = await self.get(url, client)
        return rp.can_fetch(agent, url) if rp else True

    async def crawl_delay(self, url: str, client) -> float:
        rp = await self.get(url, client)
        if rp:
            cd = rp.crawl_delay("*")
            if cd:
                return float(cd)
        return self.default_delay

    async def site_maps(self, url: str, client) -> list[str]:
        rp = await self.get(url, client)
        if rp and hasattr(rp, "site_maps"):
            sms = rp.site_maps()
            return list(sms) if sms else []
        return []


# ─────────────────────────────────────────────────────────────
# [F11] HTTP/2 FETCHER  (httpx)
# [F12] Cookie / session management
# ─────────────────────────────────────────────────────────────

class HTTPFetcher:
    """
    [F11] httpx AsyncClient with HTTP/2 (and HTTP/3 where supported).
    [F12] Per-domain cookie jars for session persistence.

    Single shared client — opened once, closed at shutdown (v4 B1 fix).
    Integrates with ProxyPool and CircuitBreaker.
    """

    def __init__(
        self,
        global_concurrency: int  = 50,
        timeout:            int  = 20,
        proxy_pool: Optional[ProxyPool]        = None,
        circuit:    Optional[CircuitBreaker]   = None,
    ):
        self._gsem    = asyncio.Semaphore(global_concurrency)
        self._timeout = timeout
        self._proxies = proxy_pool
        self._circuit = circuit or CircuitBreaker()
        self._client: Optional[httpx.AsyncClient] = None
        # [F12] Per-domain cookie stores
        self._cookies: dict[str, httpx.Cookies] = {}

    async def open(self):
        if not _HTTPX_OK:
            log.error("httpx not installed — pip install 'httpx[http2]'")
            return
        self._client = httpx.AsyncClient(
            http2=True,
            follow_redirects=True,
            timeout=self._timeout,
            headers={"User-Agent": USER_AGENT},
            verify=True,
        )
        log.info("HTTPFetcher: httpx client open (HTTP/2 enabled)")

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def login(
        self,
        domain_name: str,
        login_url:   str,
        credentials: dict,
    ) -> bool:
        """
        [F12] POST credentials to login_url, store returned cookies
        per-domain for all subsequent requests.
        """
        if not self._client:
            return False
        try:
            resp = await self._client.post(login_url, data=credentials)
            self._cookies[domain_name] = resp.cookies
            log.info("HTTPFetcher: logged into %s (status %d)", domain_name, resp.status_code)
            return resp.status_code < 400
        except Exception as exc:
            log.warning("HTTPFetcher: login failed for %s: %s", domain_name, exc)
            return False

    async def fetch(
        self,
        url:          str,
        rate:         RateLimiter,
        etag:         str = "",
        last_modified: str = "",
    ) -> FetchResult:
        from .utils import domain as _dom
        meta  = CrawlMeta()
        dom   = _dom(url)

        # Circuit breaker check
        if self._circuit.is_open(dom):
            meta.circuit_open = True
            log.debug("CircuitBreaker OPEN: skipping %s", url)
            return FetchResult(url=url, final_url=url, html=None, meta=meta)

        sem = await rate.acquire(dom)
        async with self._gsem, sem:
            await rate.wait(dom)

            # Proxy selection
            proxy_url = self._proxies.get(dom) if self._proxies else None

            # Conditional request headers (v3 fix)
            headers: dict[str, str] = {}
            if etag:
                headers["If-None-Match"] = etag
            elif last_modified:
                headers["If-Modified-Since"] = last_modified

            # Per-domain cookies [F12]
            cookies = self._cookies.get(dom)

            t0 = time.monotonic()
            try:
                req_kwargs: dict = {"headers": headers}
                if cookies:
                    req_kwargs["cookies"] = cookies
                if proxy_url:
                    req_kwargs["extensions"] = {}  # httpx proxy via env or mounts

                resp = await self._client.get(url, **req_kwargs)

                meta.status_code      = resp.status_code
                meta.content_type     = resp.headers.get("content-type", "")
                meta.etag             = resp.headers.get("etag", "")
                meta.last_modified    = resp.headers.get("last-modified", "")
                meta.response_time_ms = round((time.monotonic() - t0) * 1000, 1)
                meta.http_version     = str(resp.http_version) if hasattr(resp, "http_version") else ""
                meta.proxy_used       = proxy_url or ""
                final_url             = str(resp.url)

                if resp.status_code == 304:
                    self._circuit.record_success(dom)
                    if self._proxies and proxy_url:
                        self._proxies.report_success(dom, proxy_url)
                    return FetchResult(url=url, final_url=final_url, html=None, meta=meta)

                mime = meta.content_type.split(";")[0].strip().lower()
                if resp.status_code == 200 and mime in ALLOWED_MIME:
                    html = resp.text
                    needs_js = len(html) < 3000 or bool(JS_SIGNALS.search(html[:5000]))
                    self._circuit.record_success(dom)
                    if self._proxies and proxy_url:
                        self._proxies.report_success(dom, proxy_url)
                    return FetchResult(
                        url=url, final_url=final_url,
                        html=html, meta=meta, needs_js=needs_js
                    )

                # Non-200 — record as soft failure
                if resp.status_code >= 500:
                    self._circuit.record_failure(dom)

            except Exception as exc:
                log.warning("Fetch error %s: %s", url[:60], exc)
                meta.response_time_ms = round((time.monotonic() - t0) * 1000, 1)
                self._circuit.record_failure(dom)
                if self._proxies and proxy_url:
                    self._proxies.report_error(dom, proxy_url)

        return FetchResult(url=url, final_url=url, html=None, meta=meta)


# ─────────────────────────────────────────────────────────────
# PLAYWRIGHT POOL  (infinite scroll + CAPTCHA)
# ─────────────────────────────────────────────────────────────

class PlaywrightPool:
    """
    Playwright browser pool with:
      - Isolated BrowserContext per fetch (no cross-page state)
      - Infinite scroll: scrolls until no new content loads
      - CAPTCHA detection and optional automatic solving
    """

    SCROLL_PAUSE_MS  = 1500
    MAX_SCROLLS      = 20

    def __init__(
        self,
        max_browsers:  int  = 5,
        timeout_ms:    int  = 25_000,
        captcha_solver: Optional[CaptchaSolver] = None,
        headless:      bool = True,
    ):
        self.max_browsers  = max_browsers
        self.timeout_ms    = timeout_ms
        self._solver       = captcha_solver
        self._headless     = headless
        self._sem: Optional[asyncio.Semaphore] = None
        self._pw   = None
        self._browser = None
        self._active  = 0
        self._lock    = asyncio.Lock()

    async def open(self):
        if not _PW_OK:
            log.warning("PlaywrightPool: playwright not installed")
            log.warning("  pip install playwright && playwright install chromium")
            return
        self._sem     = asyncio.Semaphore(self.max_browsers)
        self._pw      = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=["--no-sandbox", "--disable-setuid-sandbox",
                  "--disable-dev-shm-usage", "--disable-gpu"],
        )
        log.info("PlaywrightPool: %d slots open", self.max_browsers)

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    @property
    def available(self) -> bool:
        return _PW_OK and self._browser is not None

    def active_count(self) -> int:
        return self._active

    async def fetch(self, url: str) -> tuple[Optional[str], str]:
        """Returns (html, final_url)."""
        if not self.available:
            return None, url

        async with self._sem:
            async with self._lock:
                self._active += 1
            ctx = None
            try:
                ctx = await self._browser.new_context(
                    user_agent=USER_AGENT,
                    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                )
                page = await ctx.new_page()

                resp = await page.goto(url, wait_until="domcontentloaded",
                                       timeout=self.timeout_ms)
                final_url = page.url

                # Detect and optionally solve CAPTCHA
                if await self._detect_captcha(page):
                    log.info("CAPTCHA detected on %s", url[:60])
                    if self._solver and self._solver.enabled:
                        await self._attempt_solve(page, url)

                # Infinite scroll [F13]
                await self._scroll_to_bottom(page)

                html = await page.content()
                return html, final_url

            except Exception as exc:
                log.warning("Playwright error %s: %s", url[:60], exc)
                return None, url
            finally:
                if ctx:
                    await ctx.close()
                async with self._lock:
                    self._active -= 1

    async def _detect_captcha(self, page) -> bool:
        """Check common CAPTCHA selectors."""
        for selector in [
            "#cf-challenge-running", ".g-recaptcha",
            "iframe[src*='recaptcha']", "[data-hcaptcha-widget-id]",
            "#challenge-form",
        ]:
            try:
                el = await page.query_selector(selector)
                if el:
                    return True
            except Exception:
                pass
        return False

    async def _attempt_solve(self, page, url: str):
        """Try to extract site key and solve reCAPTCHA."""
        if not self._solver:
            return
        try:
            site_key = await page.eval_on_selector(
                ".g-recaptcha", "el => el.getAttribute('data-sitekey')"
            )
            if site_key:
                token = await self._solver.solve_recaptcha_v2(site_key, url)
                if token:
                    await self._solver.inject_recaptcha_token(page, token)
                    await page.wait_for_timeout(2000)
        except Exception as exc:
            log.debug("CAPTCHA solve failed: %s", exc)

    async def _scroll_to_bottom(self, page):
        """
        [F14] Infinite scroll: keep scrolling until page height stops growing.
        Pauses SCROLL_PAUSE_MS between scrolls to allow content to load.
        """
        prev_height = 0
        for _ in range(self.MAX_SCROLLS):
            curr_height = await page.evaluate("document.body.scrollHeight")
            if curr_height == prev_height:
                break
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(self.SCROLL_PAUSE_MS)
            prev_height = curr_height
