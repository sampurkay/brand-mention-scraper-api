from __future__ import annotations

import asyncio
import time
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import httpx
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright, Page, BrowserContext, Response  # type: ignore


DEFAULT_TIMEOUT = 20.0
DEFAULT_DELAY = 1.0
MAX_TEXT_CHARS = 200_000


# -------------------------
# Helpers
# -------------------------

def _normalize_url(url: str) -> str:
    p = urllib.parse.urlparse(url)
    scheme = p.scheme or "https"
    netloc = (p.netloc or "").lower()
    path = p.path or "/"
    return urllib.parse.urlunparse((scheme, netloc, path, "", p.query or "", ""))


def _same_domain(a: str, b: str) -> bool:
    return urllib.parse.urlparse(a).netloc.lower() == urllib.parse.urlparse(b).netloc.lower()


def _compile_patterns(patterns: Optional[List[str]]) -> List[Pattern[str]]:
    import re
    out: List[Pattern[str]] = []
    for p in patterns or []:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return out


def _matches_any(url: str, pats: List[Pattern[str]]) -> bool:
    return any(p.search(url) is not None for p in pats)


def _is_likely_html(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return ("text/html" in ct) or ("application/xhtml+xml" in ct) or ("text/plain" in ct)


def _extract_title(soup: BeautifulSoup) -> str:
    try:
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    return ""


def _html_to_text(soup: BeautifulSoup) -> str:
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    txt = soup.get_text(" ", strip=True)
    if len(txt) > MAX_TEXT_CHARS:
        txt = txt[:MAX_TEXT_CHARS]
    return txt


def _extract_links(base_url: str, soup: BeautifulSoup, *, same_domain_only: bool) -> List[str]:
    links: List[str] = []
    base_dom = urllib.parse.urlparse(base_url).netloc.lower()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urllib.parse.urljoin(base_url, href)
        abs_url = _normalize_url(abs_url)
        if not abs_url.startswith("http"):
            continue
        if same_domain_only and urllib.parse.urlparse(abs_url).netloc.lower() != base_dom:
            continue
        links.append(abs_url)
    return links


def _is_skippable_url(url: str) -> bool:
    path = (urllib.parse.urlparse(url).path or "").lower()
    skip_ext = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        ".zip", ".rar", ".7z", ".tar", ".gz",
        ".mp4", ".mov", ".avi", ".mp3", ".wav",
    )
    return any(path.endswith(ext) for ext in skip_ext)


def _looks_blocked(status_code: int, body_or_html: str) -> bool:
    t = (body_or_html or "").lower()
    if status_code in (401, 403, 429):
        return True
    markers = (
        "captcha",
        "cloudflare",
        "access denied",
        "attention required",
        "verify you are human",
        "unusual traffic",
    )
    return any(m in t for m in markers)


def _looks_like_js_shell(html: str) -> bool:
    h = (html or "")
    markers = ("__NEXT_DATA__", 'id="__next"', "data-reactroot", 'id="app"', "ng-version")
    return any(m in h for m in markers)


def _exp_backoff(base_delay: float, attempt: int) -> float:
    return base_delay * (2 ** attempt)


def _relevance_score(url: str, keywords: List[str]) -> int:
    u = url.lower()
    return sum(1 for k in keywords if k and k.lower() in u)


# -------------------------
# Crawl params
# -------------------------

@dataclass
class PlaywrightParams:
    wait_for_selector: Optional[str] = None
    selector_timeout_ms: int = 15000

    block_images: bool = True
    block_fonts: bool = True
    block_media: bool = True
    block_stylesheets: bool = False

    prefer_inner_text: bool = True

    enable_infinite_scroll: bool = False
    max_scrolls: int = 12
    scroll_pause_ms: int = 600
    stable_rounds: int = 2

    enable_pagination: bool = False
    pagination_max_pages: int = 4
    next_selectors: List[str] = field(default_factory=list)


@dataclass
class RetryParams:
    base_delay_s: float = 0.75
    max_retries: int = 2


@dataclass
class CrawlParams:
    depth: int = 0
    max_pages: int = 30
    include_link_keywords: List[str] = field(default_factory=list)
    include_url_patterns: List[str] = field(default_factory=list)
    exclude_url_patterns: List[str] = field(default_factory=list)

    render_js: bool = False
    js_only: bool = False
    wait_until: str = "networkidle"
    wait_ms: int = 0

    same_domain_only: bool = True
    max_concurrency: int = 2
    respect_robots: bool = True

    # NEW: relevance controls
    link_relevance_mode: str = "prioritize"  # "prioritize" | "filter"
    link_priority_keywords: List[str] = field(default_factory=list)

    # NEW: advanced controls
    playwright: PlaywrightParams = field(default_factory=PlaywrightParams)
    retry: RetryParams = field(default_factory=RetryParams)


@dataclass
class PageResult:
    url: str
    status: str
    title: str
    text: str


# -------------------------
# PoliteScraper
# -------------------------

class PoliteScraper:
    def __init__(
        self,
        *,
        user_agent: str = "brand-mention-scraper-api/1.0 (internal)",
        default_timeout: float = DEFAULT_TIMEOUT,
        crawl_delay: float = DEFAULT_DELAY,
    ) -> None:
        self.user_agent = user_agent
        self.default_timeout = float(default_timeout)

        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
            timeout=httpx.Timeout(self.default_timeout),
        )

        self.crawl_delay: Dict[str, float] = {}
        self.last_access: Dict[str, float] = {}
        self._pw_lock = asyncio.Lock()
        self._pw: Optional["Playwright"] = None
        self._browser: Optional["Browser"] = None

        self._robots: Dict[str, RobotFileParser] = {}
        self._robots_lock = asyncio.Lock()

        # Jobs
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._job_lock = asyncio.Lock()
        self._job_tasks: Dict[str, asyncio.Task] = {}

        # Default politeness if domain not in crawl_delay map
        self._default_delay = float(crawl_delay)

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._close_browser()

    # -------------------------
    # Robots + politeness
    # -------------------------

    def get_domain(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc.lower()

    async def _ensure_robots(self, url: str) -> RobotFileParser:
        dom = self.get_domain(url)
        async with self._robots_lock:
            if dom in self._robots:
                return self._robots[dom]

            rp = RobotFileParser()
            robots_url = urllib.parse.urlunparse(("https", dom, "/robots.txt", "", "", ""))
            try:
                resp = await self._client.get(robots_url)
                if resp.status_code >= 400:
                    rp.parse([])
                else:
                    rp.parse((resp.text or "").splitlines())
            except Exception:
                rp.parse([])

            self._robots[dom] = rp
            return rp

    async def can_fetch(self, url: str, *, respect_robots: bool) -> bool:
        if not respect_robots:
            return True
        rp = await self._ensure_robots(url)
        try:
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True

    async def obey_rate_limit(self, url: str) -> None:
        domain = self.get_domain(url)
        delay: float = self.crawl_delay.get(domain, self._default_delay)
        last: float = self.last_access.get(domain, 0.0)
        now: float = time.time()
        wait: float = delay - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_access[domain] = time.time()

    # -------------------------
    # Playwright
    # -------------------------

    async def _ensure_browser(self) -> "Browser":
        try:
            from playwright.async_api import async_playwright  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Playwright is not installed. Add 'playwright' to requirements.txt and ensure browsers are installed."
            ) from e

        async with self._pw_lock:
            if self._browser is not None:
                return self._browser
            if self._pw is None:
                self._pw = await async_playwright().start()

            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            )
            return self._browser

    async def _close_browser(self) -> None:
        async with self._pw_lock:
            try:
                if self._browser is not None:
                    await self._browser.close()
            except Exception:
                pass
            self._browser = None
            try:
                if self._pw is not None:
                    await self._pw.stop()
            except Exception:
                pass
            self._pw = None

    async def _configure_routing(self, page: "Page", pw: PlaywrightParams) -> None:
        async def route_handler(route, request):
            rtype = request.resource_type
            if pw.block_images and rtype == "image":
                return await route.abort()
            if pw.block_fonts and rtype == "font":
                return await route.abort()
            if pw.block_media and rtype == "media":
                return await route.abort()
            if pw.block_stylesheets and rtype == "stylesheet":
                return await route.abort()
            return await route.continue_()

        await page.route("**/*", route_handler)

    async def _scroll_infinite(self, page: "Page", pw: PlaywrightParams) -> None:
        last_h = 0
        stable = 0
        for _ in range(max(0, pw.max_scrolls)):
            h = await page.evaluate("() => document.body ? document.body.scrollHeight : 0")
            if h == last_h:
                stable += 1
                if stable >= max(1, pw.stable_rounds):
                    break
            else:
                stable = 0
                last_h = h

            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(max(0, pw.scroll_pause_ms))

    async def _paginate_click_next(self, page: "Page", pw: PlaywrightParams, wait_until: str) -> List[str]:
        snaps: List[str] = [await page.content()]
        selectors = pw.next_selectors or [
            "a[rel='next']",
            "button:has-text('Next')",
            "a:has-text('Next')",
            "button:has-text('›')",
            "a:has-text('›')",
        ]
        for _ in range(max(1, pw.pagination_max_pages) - 1):
            clicked = False
            for sel in selectors:
                try:
                    loc = page.locator(sel)
                    if await loc.count() > 0 and await loc.first.is_visible():
                        await loc.first.click(timeout=1500)
                        await page.wait_for_load_state(wait_until, timeout=15000)
                        snaps.append(await page.content())
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                break
        return snaps

    async def _get_html_playwright(
        self,
        url: str,
        *,
        wait_until: str,
        wait_ms: int,
        timeout: Optional[float],
        pw: PlaywrightParams,
        retry: RetryParams,
    ) -> Tuple[str, str, str]:
        """
        Returns (status, html, extracted_text).
        Adds: wait_for_selector, resource blocking, response intercept, evaluate(), infinite scroll, pagination.
        """
        browser = await self._ensure_browser()
        to_ms = int(((timeout or (self.default_timeout * 2)) * 1000))

        for attempt in range(retry.max_retries + 1):
            context: Optional["BrowserContext"] = None
            page: Optional["Page"] = None
            try:
                context = await browser.new_context(user_agent=self.user_agent, java_script_enabled=True)
                page = await context.new_page()

                await self._configure_routing(page, pw)

                # response intercept (kept for future use; useful for debugging)
                last_main: Dict[str, Any] = {"status": None, "url": None, "ctype": None}

                async def on_response(resp: "Response"):
                    try:
                        if resp.request.is_navigation_request():
                            last_main["status"] = resp.status
                            last_main["url"] = resp.url
                            last_main["ctype"] = (resp.headers.get("content-type") or "").lower()
                    except Exception:
                        pass

                page.on("response", on_response)

                await page.goto(url, wait_until=wait_until, timeout=to_ms)  # type: ignore[arg-type]
                if wait_ms > 0:
                    await page.wait_for_timeout(wait_ms)

                # explicit wait_for_selector if provided (best for JS heavy sites)
                if pw.wait_for_selector:
                    try:
                        await page.wait_for_selector(pw.wait_for_selector, timeout=pw.selector_timeout_ms)
                    except Exception:
                        pass
                else:
                    # heuristic: if it looks like a JS shell, wait for common containers
                    try:
                        html0 = await page.content()
                        if _looks_like_js_shell(html0):
                            await page.wait_for_selector(
                                "main, article, [role='main'], #app, #__next, body",
                                timeout=min(15000, to_ms),
                            )
                    except Exception:
                        pass

                # infinite scroll (optional)
                if pw.enable_infinite_scroll:
                    try:
                        await self._scroll_infinite(page, pw)
                    except Exception:
                        pass

                # pagination clicking (optional)
                if pw.enable_pagination:
                    try:
                        htmls = await self._paginate_click_next(page, pw, wait_until=wait_until)
                    except Exception:
                        htmls = [await page.content()]
                else:
                    htmls = [await page.content()]

                # Extract text efficiently
                extracted_text = ""
                if pw.prefer_inner_text:
                    try:
                        extracted_text = await page.evaluate(
                            "() => document.body ? document.body.innerText : ''"
                        )
                    except Exception:
                        extracted_text = ""

                # fallback parse
                if not extracted_text.strip():
                    soup = BeautifulSoup(" ".join(htmls), "html.parser")
                    extracted_text = _html_to_text(soup)

                html = htmls[-1] if htmls else ""
                if _looks_blocked(200, html) or _looks_blocked(200, extracted_text):
                    return ("blocked", "", "")

                if len(extracted_text) > MAX_TEXT_CHARS:
                    extracted_text = extracted_text[:MAX_TEXT_CHARS]

                return ("ok", html, extracted_text)

            except (asyncio.TimeoutError, Exception):
                if attempt < retry.max_retries:
                    await asyncio.sleep(_exp_backoff(retry.base_delay_s, attempt))
                    continue
                return ("error_playwright", "", "")
            finally:
                try:
                    if page:
                        await page.close()
                except Exception:
                    pass
                try:
                    if context:
                        await context.close()
                except Exception:
                    pass

        return ("error_playwright", "", "")

    # -------------------------
    # HTTP
    # -------------------------

    async def _get_html_http(self, url: str, retry: RetryParams) -> Tuple[str, str, str]:
        for attempt in range(retry.max_retries + 1):
            try:
                resp = await self._client.get(url, timeout=self.default_timeout)
                ct = resp.headers.get("content-type", "") or ""
                body = resp.text if isinstance(resp.text, str) else ""
                if _looks_blocked(resp.status_code, body):
                    return ("blocked", "", ct)
                if resp.status_code != 200:
                    return (f"http_{resp.status_code}", "", ct)
                return ("ok", body, ct)
            except httpx.TimeoutException:
                if attempt < retry.max_retries:
                    await asyncio.sleep(_exp_backoff(retry.base_delay_s, attempt))
                    continue
                return ("error_timeout", "", "")
            except httpx.HTTPError:
                if attempt < retry.max_retries:
                    await asyncio.sleep(_exp_backoff(retry.base_delay_s, attempt))
                    continue
                return ("error_http", "", "")
            except Exception as e:
                if attempt < retry.max_retries:
                    await asyncio.sleep(_exp_backoff(retry.base_delay_s, attempt))
                    continue
                return (f"error_{type(e).__name__}", "", "")
        return ("error_http", "", "")

    # -------------------------
    # Public fetch used by crawler
    # -------------------------

    async def fetch(
        self,
        url: str,
        *,
        render_js: bool,
        js_only: bool,
        wait_until: str,
        wait_ms: int,
        respect_robots: bool,
        pw: PlaywrightParams,
        retry: RetryParams,
    ) -> Tuple[str, str, str]:
        """
        Returns (status, title, text)
        """
        url = _normalize_url(url)

        if not await self.can_fetch(url, respect_robots=respect_robots):
            return ("blocked_by_robots", "", "")

        await self.obey_rate_limit(url)

        if render_js and js_only:
            st_html, html, text = await self._get_html_playwright(
                url,
                wait_until=wait_until,
                wait_ms=wait_ms,
                timeout=None,
                pw=pw,
                retry=retry,
            )
            if not st_html.startswith("ok"):
                return (st_html, "", "")
            soup = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")
            title = _extract_title(soup)
            return ("ok", title, text)

        st, body, ct = await self._get_html_http(url, retry=retry)
        if st.startswith("ok") and (_is_likely_html(ct) or "<html" in (body or "").lower()):
            soup = BeautifulSoup(body, "html.parser")
            title = _extract_title(soup)
            text = _html_to_text(soup)
            # If JS shell / empty and render_js enabled, escalate to Playwright
            if render_js and (not text.strip() or _looks_like_js_shell(body)):
                st_html, html, text2 = await self._get_html_playwright(
                    url,
                    wait_until=wait_until,
                    wait_ms=wait_ms,
                    timeout=None,
                    pw=pw,
                    retry=retry,
                )
                if st_html.startswith("ok"):
                    soup2 = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")
                    title2 = _extract_title(soup2) or title
                    return ("ok", title2, text2)
            return ("ok", title, text)

        # fallback to Playwright if enabled
        if render_js and (st in ("blocked", "error_timeout", "error_http") or st.startswith("http_")):
            st_html, html, text = await self._get_html_playwright(
                url,
                wait_until=wait_until,
                wait_ms=wait_ms,
                timeout=None,
                pw=pw,
                retry=retry,
            )
            if st_html.startswith("ok"):
                soup = BeautifulSoup(html, "html.parser") if html else BeautifulSoup("", "html.parser")
                title = _extract_title(soup)
                return ("ok", title, text)
            return (st_html, "", "")

        return (st, "", "")

    async def fetch_with_links(
        self,
        url: str,
        *,
        render_js: bool,
        js_only: bool,
        wait_until: str,
        wait_ms: int,
        same_domain_only: bool,
        respect_robots: bool,
        pw: PlaywrightParams,
        retry: RetryParams,
    ) -> Tuple[str, str, List[str], str]:
        url = _normalize_url(url)

        status, title, text = await self.fetch(
            url,
            render_js=render_js,
            js_only=js_only,
            wait_until=wait_until,
            wait_ms=wait_ms,
            respect_robots=respect_robots,
            pw=pw,
            retry=retry,
        )

        # link extraction: prefer HTML when available
        html = ""
        if status.startswith("ok"):
            st_http, body, ct = await self._get_html_http(url, retry=retry)
            if st_http.startswith("ok") and (_is_likely_html(ct) or "<html" in (body or "").lower()):
                html = body
            elif render_js:
                st_html, html2, _txt = await self._get_html_playwright(
                    url,
                    wait_until=wait_until,
                    wait_ms=wait_ms,
                    timeout=None,
                    pw=pw,
                    retry=retry,
                )
                if st_html.startswith("ok"):
                    html = html2

        links: List[str] = []
        if html:
            soup = BeautifulSoup(html, "html.parser")
            links = _extract_links(url, soup, same_domain_only=same_domain_only)

        return status, text, links, title

    # -------------------------
    # URL allow rules
    # -------------------------

    def _url_allowed(
        self,
        url: str,
        *,
        same_domain_only: bool,
        start_url: str,
        include_patterns: List[Pattern[str]],
        exclude_patterns: List[Pattern[str]],
    ) -> bool:
        url = _normalize_url(url)

        if _is_skippable_url(url):
            return False
        if same_domain_only and not _same_domain(url, start_url):
            return False
        if exclude_patterns and _matches_any(url, exclude_patterns):
            return False
        if include_patterns:
            return _matches_any(url, include_patterns)
        return True

    # -------------------------
    # Crawl (relevance aware)
    # -------------------------

    async def crawl_relevant(
        self,
        start_url: str,
        *,
        depth: int = 0,
        max_pages: int = 30,
        include_link_keywords: Optional[List[str]] = None,
        include_url_patterns: Optional[List[str]] = None,
        exclude_url_patterns: Optional[List[str]] = None,
        render_js: bool = False,
        js_only: bool = False,
        wait_until: str = "networkidle",
        wait_ms: int = 0,
        same_domain_only: bool = True,
        max_concurrency: int = 2,
        respect_robots: bool = True,
        playwright: Optional[PlaywrightParams] = None,
        retry: Optional[RetryParams] = None,
        link_relevance_mode: str = "prioritize",
        link_priority_keywords: Optional[List[str]] = None,
        job_id: Optional[str] = None,  # progress updates if provided
    ) -> List[Tuple[str, str, str, str]]:
        start_url = _normalize_url(start_url)

        include_keywords = [kw for kw in (include_link_keywords or []) if (kw or "").strip()]
        include_patterns = _compile_patterns(include_url_patterns)
        exclude_patterns = _compile_patterns(exclude_url_patterns)

        pw = playwright or PlaywrightParams()
        rt = retry or RetryParams()
        pri_kw = [k for k in (link_priority_keywords or []) if (k or "").strip()]

        visited: Set[str] = set()
        queue: Deque[Tuple[str, int]] = deque([(start_url, 0)])
        results: List[Tuple[str, str, str, str]] = []

        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def process_one(url_in: str, d: int) -> Tuple[str, int, str, str, str, List[str]]:
            async with sem:
                url_norm = _normalize_url(url_in)

                if job_id is not None and await self._job_cancelled(job_id):
                    return (url_norm, d, "cancelled", "", "", [])

                st, txt, links, title = await self.fetch_with_links(
                    url_norm,
                    render_js=render_js,
                    js_only=js_only,
                    wait_until=wait_until,
                    wait_ms=wait_ms,
                    same_domain_only=same_domain_only,
                    respect_robots=respect_robots,
                    pw=pw,
                    retry=rt,
                )

                if job_id is not None:
                    async with self._job_lock:
                        rec = self._jobs.get(job_id)
                        if rec is not None:
                            rec["pages_attempted"] += 1
                            rec["updated_at"] = time.time()

                    await self._emit_event(
                        job_id,
                        {"type": "page_done", "job_id": job_id, "url": url_norm, "status": st, "depth": d},
                    )

                return (url_norm, d, st, title, txt, links)

        while queue and len(results) < max_pages:
            if job_id is not None and await self._job_cancelled(job_id):
                break

            batch: List[Tuple[str, int]] = []
            while queue and len(batch) < max_concurrency and (len(results) + len(batch) < max_pages):
                u, d = queue.popleft()
                u = _normalize_url(u)
                if u in visited:
                    continue
                if not self._url_allowed(
                    u,
                    same_domain_only=same_domain_only,
                    start_url=start_url,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                ):
                    continue
                visited.add(u)
                batch.append((u, d))

            if not batch:
                continue

            batch_results = await asyncio.gather(*(process_one(u, d) for u, d in batch))

            for url_norm, d, status, title, text, links in batch_results:
                if status == "cancelled":
                    continue

                results.append((url_norm, status, title, text))
                if len(results) >= max_pages:
                    break

                if d >= depth:
                    continue
                if status.startswith(("blocked", "error")) or status.startswith("http_"):
                    continue

                # include_link_keywords (old behavior) still supported
                if include_keywords:
                    low_keywords = [k.lower() for k in include_keywords]
                    links = [lnk for lnk in links if any(k in lnk.lower() for k in low_keywords)]

                # NEW: relevance prioritize/filter
                scored: List[Tuple[int, str]] = []
                for lnk in links:
                    lnk_n = _normalize_url(lnk)
                    if lnk_n in visited:
                        continue
                    if not self._url_allowed(
                        lnk_n,
                        same_domain_only=same_domain_only,
                        start_url=start_url,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                    ):
                        continue

                    score = _relevance_score(lnk_n, pri_kw) if pri_kw else 0
                    if link_relevance_mode == "filter" and pri_kw and score == 0:
                        continue
                    scored.append((score, lnk_n))

                scored.sort(key=lambda x: x[0], reverse=True)

                for score, lnk_n in scored:
                    if len(results) + len(queue) >= max_pages:
                        break
                    if lnk_n in visited:
                        continue
                    # Higher relevance goes earlier
                    if score > 0:
                        queue.appendleft((lnk_n, d + 1))
                    else:
                        queue.append((lnk_n, d + 1))

        return results

    # -------------------------
    # Job system
    # -------------------------

    async def submit_crawl_job(self, seeds: List[str], params: CrawlParams) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        rec = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "error": None,
            "seeds_total": len(seeds),
            "seeds_done": 0,
            "pages_attempted": 0,
            "pages_stored": 0,
            "max_pages_stored": 10_000,
            "results": [],
            "cancelled": False,
            "params": params,
        }
        async with self._job_lock:
            self._jobs[job_id] = rec

        task = asyncio.create_task(self._run_job(job_id, seeds))
        async with self._job_lock:
            self._job_tasks[job_id] = task
        return job_id

    async def _job_cancelled(self, job_id: str) -> bool:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            return bool(rec and rec.get("cancelled"))

    async def cancel_job(self, job_id: str) -> bool:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return False
            rec["cancelled"] = True
            rec["status"] = "cancelled"
            rec["updated_at"] = time.time()
        return True

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return {
                "job_id": rec["job_id"],
                "status": rec["status"],
                "created_at": rec["created_at"],
                "updated_at": rec["updated_at"],
                "error": rec["error"],
                "seeds_total": rec["seeds_total"],
                "seeds_done": rec["seeds_done"],
                "pages_attempted": rec["pages_attempted"],
                "pages_stored": rec["pages_stored"],
                "max_pages_stored": rec["max_pages_stored"],
            }

    async def get_job_results(self, job_id: str, *, offset: int, limit: int) -> Optional[List[PageResult]]:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            res = rec.get("results", [])
            # stored as PageResult already
            return res[offset : offset + limit]

    async def _emit_event(self, job_id: str, evt: Dict[str, Any]) -> None:
        # placeholder hook; keep compatibility with your main.py job status
        return

    async def _run_job(self, job_id: str, seeds: List[str]) -> None:
        try:
            async with self._job_lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec["status"] = "running"
                rec["updated_at"] = time.time()
                params: CrawlParams = rec["params"]

            for s in seeds:
                if await self._job_cancelled(job_id):
                    return

                pages = await self.crawl_relevant(
                    s,
                    depth=params.depth,
                    max_pages=params.max_pages,
                    include_link_keywords=params.include_link_keywords,
                    include_url_patterns=params.include_url_patterns,
                    exclude_url_patterns=params.exclude_url_patterns,
                    render_js=params.render_js,
                    js_only=params.js_only,
                    wait_until=params.wait_until,
                    wait_ms=params.wait_ms,
                    same_domain_only=params.same_domain_only,
                    max_concurrency=params.max_concurrency,
                    respect_robots=params.respect_robots,
                    playwright=params.playwright,
                    retry=params.retry,
                    link_relevance_mode=params.link_relevance_mode,
                    link_priority_keywords=params.link_priority_keywords,
                    job_id=job_id,
                )

                # store
                async with self._job_lock:
                    rec = self._jobs.get(job_id)
                    if rec is None:
                        return
                    for (u, st, title, text) in pages:
                        rec["results"].append(PageResult(url=u, status=st, title=title, text=text))
                        rec["pages_stored"] += 1
                    rec["seeds_done"] += 1
                    rec["updated_at"] = time.time()

            async with self._job_lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                if rec["status"] != "cancelled":
                    rec["status"] = "done"
                    rec["updated_at"] = time.time()

        except Exception as e:
            async with self._job_lock:
                rec = self._jobs.get(job_id)
                if rec is None:
                    return
                rec["status"] = "error"
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["updated_at"] = time.time()
