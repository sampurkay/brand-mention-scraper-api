from __future__ import annotations

import asyncio
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Deque, Dict, Iterable, List, Optional, Pattern, Set, Tuple
from collections import deque

import httpx
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

# Playwright is optional; imported lazily when render_js=True
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright, Response  # type: ignore


# -----------------------------
# Tunables / defaults
# -----------------------------

DEFAULT_TIMEOUT_S: float = 20.0
DEFAULT_CONNECT_TIMEOUT_S: float = 10.0
DEFAULT_READ_TIMEOUT_S: float = 20.0

DEFAULT_DELAY_PER_DOMAIN_S: float = 1.0   # politeness
MAX_TEXT_CHARS: int = 200_000

DEFAULT_WORKERS: int = 2
DEFAULT_JOB_TTL_SECONDS: int = 60 * 60
DEFAULT_MAX_PAGES_STORED_PER_JOB: int = 2000
DEFAULT_EVENT_QUEUE_SIZE: int = 1000

# Detection heuristics
_BLOCK_MARKERS = (
    "captcha",
    "cloudflare",
    "attention required",
    "access denied",
    "temporarily unavailable",
    "unusual traffic",
    "verify you are human",
)

_JS_SHELL_MARKERS = (
    "__NEXT_DATA__",
    'id="__next"',
    "data-reactroot",
    "ng-version",
    'id="app"',
)

_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".css", ".js", ".xml",
)


# -----------------------------
# Data models
# -----------------------------

@dataclass
class CrawlParams:
    depth: int = 0
    max_pages: int = 30
    include_link_keywords: List[str] = field(default_factory=list)
    include_url_patterns: List[str] = field(default_factory=list)
    exclude_url_patterns: List[str] = field(default_factory=list)

    render_js: bool = False
    js_only: bool = False
    wait_until: str = "networkidle"  # "load" | "domcontentloaded" | "networkidle"
    wait_ms: int = 0

    same_domain_only: bool = True
    max_concurrency: int = 2
    respect_robots: bool = True


@dataclass
class PageResult:
    url: str
    status: str
    title: str
    text: str


@dataclass
class JobRecord:
    job_id: str
    created_at: float
    updated_at: float
    status: str  # "queued" | "running" | "done" | "error" | "cancelled"
    params: CrawlParams
    seed_urls: List[str]

    pages_stored: int = 0
    pages_attempted: int = 0
    seeds_total: int = 0
    seeds_done: int = 0

    results: List[PageResult] = field(default_factory=list)
    error: Optional[str] = None

    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    events: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_EVENT_QUEUE_SIZE))


# -----------------------------
# Utilities
# -----------------------------

def _normalize_url(url: str) -> str:
    """Normalize URL for dedupe: strip fragment; normalize scheme/host."""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query
    return urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))


def _domain(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def _same_domain(a: str, b: str) -> bool:
    return _domain(a) == _domain(b)


def _looks_like_block(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _BLOCK_MARKERS)


def _looks_like_js_shell(html: str) -> bool:
    h = html or ""
    return any(m in h for m in _JS_SHELL_MARKERS)


def _should_skip_url(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    path = (p.path or "").lower()
    return any(path.endswith(ext) for ext in _SKIP_EXTENSIONS)


def _compile_patterns(patterns: List[str]) -> List[Pattern[str]]:
    out: List[Pattern[str]] = []
    for p in patterns:
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            # ignore invalid patterns
            continue
    return out


def _match_any(patterns: List[Pattern[str]], text: str) -> bool:
    return any(p.search(text) is not None for p in patterns)


def _extract_title_and_text_from_html(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html or "", "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    return title, text


def _pick_wait_selector(html: str) -> Optional[str]:
    """
    Heuristic selector to wait for on JS-heavy sites so we don't snapshot too early.
    We avoid being overly specific; prefer semantic containers.
    """
    # If it's clearly a JS shell, wait for common content roots.
    if _looks_like_js_shell(html):
        return "main, article, [role='main'], #app, #__next, body"
    return None


def _exp_backoff(base_delay: float, attempt: int) -> float:
    # wait_time = base_delay * (2 ** attempt)
    return base_delay * (2 ** attempt)


# -----------------------------
# PoliteScraper
# -----------------------------

class PoliteScraper:
    """
    Polite crawler + optional Playwright rendering + async job management.
    Public API used by main.py:
      - crawl_relevant(...)
      - submit_crawl_job(...)
      - get_job_status(...)
      - get_job_results(...)
      - cancel_job(...)
      - aclose()
    """

    def __init__(
        self,
        user_agent: str = "brand-mention-scraper-api/1.0 (internal)",
        *,
        default_timeout: float = DEFAULT_TIMEOUT_S,
        job_workers: int = DEFAULT_WORKERS,
        job_ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS,
        max_pages_stored_per_job: int = DEFAULT_MAX_PAGES_STORED_PER_JOB,
    ) -> None:
        self.user_agent = user_agent
        self.default_timeout = float(default_timeout)

        self._client = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=httpx.Timeout(
                timeout=self.default_timeout,
                connect=DEFAULT_CONNECT_TIMEOUT_S,
                read=DEFAULT_READ_TIMEOUT_S,
            ),
            follow_redirects=True,
        )

        # robots cache per domain
        self._robots: Dict[str, RobotFileParser] = {}
        self._robots_lock = asyncio.Lock()

        # per-domain rate limiting
        self._domain_next_ok: Dict[str, float] = {}
        self._domain_lock = asyncio.Lock()

        # Playwright resources (lazy)
        self._pw: Optional["Playwright"] = None
        self._browser: Optional["Browser"] = None
        self._pw_lock = asyncio.Lock()

        # Jobs
        self._jobs: Dict[str, JobRecord] = {}
        self._jobs_lock = asyncio.Lock()
        self._job_queue: asyncio.Queue[str] = asyncio.Queue()
        self._job_workers: List[asyncio.Task] = []
        self._job_ttl_seconds = int(job_ttl_seconds)
        self._max_pages_stored_per_job = int(max_pages_stored_per_job)

        for _ in range(int(job_workers)):
            self._job_workers.append(asyncio.create_task(self._job_worker_loop()))

        # Cleanup loop
        self._cleanup_task = asyncio.create_task(self._cleanup_jobs_loop())

    async def aclose(self) -> None:
        # stop workers
        for t in self._job_workers:
            t.cancel()
        self._cleanup_task.cancel()

        await self._client.aclose()
        await self._close_playwright()

    # -----------------------------
    # Robots / politeness
    # -----------------------------

    async def _ensure_robots(self, url: str) -> Optional[RobotFileParser]:
        dom = _domain(url)
        async with self._robots_lock:
            if dom in self._robots:
                return self._robots[dom]

            rp = RobotFileParser()
            robots_url = urllib.parse.urlunparse(("https", dom, "/robots.txt", "", "", ""))
            try:
                r = await self._client.get(robots_url)
                if r.status_code >= 400:
                    rp.parse([])
                else:
                    rp.parse(r.text.splitlines())
            except Exception:
                rp.parse([])

            self._robots[dom] = rp
            return rp

    async def _polite_wait(self, url: str) -> None:
        dom = _domain(url)
        async with self._domain_lock:
            now = time.time()
            next_ok = self._domain_next_ok.get(dom, 0.0)
            sleep_for = max(0.0, next_ok - now)
            self._domain_next_ok[dom] = max(next_ok, now) + DEFAULT_DELAY_PER_DOMAIN_S

        if sleep_for > 0:
            await asyncio.sleep(sleep_for)

    # -----------------------------
    # HTTP fetch (fast path)
    # -----------------------------

    async def _fetch_html_httpx(self, url: str, *, max_retries: int = 2) -> Tuple[str, str]:
        """
        Returns (status, html). status starts with:
          - ok_http
          - error_http_<code>
          - error_http_timeout
          - error_http
        """
        for attempt in range(max_retries + 1):
            try:
                await self._polite_wait(url)
                resp = await self._client.get(url)
                if resp.status_code >= 400:
                    return (f"error_http_{resp.status_code}", "")
                ctype = (resp.headers.get("content-type") or "").lower()
                if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                    # still try; some sites lie
                    pass
                html = resp.text or ""
                return ("ok_http", html)
            except httpx.TimeoutException:
                if attempt < max_retries:
                    await asyncio.sleep(_exp_backoff(0.5, attempt))
                    continue
                return ("error_http_timeout", "")
            except Exception:
                if attempt < max_retries:
                    await asyncio.sleep(_exp_backoff(0.5, attempt))
                    continue
                return ("error_http", "")
        return ("error_http", "")

    # -----------------------------
    # Playwright (JS render path)
    # -----------------------------

    async def _ensure_playwright(self) -> None:
        async with self._pw_lock:
            if self._pw and self._browser:
                return

            try:
                from playwright.async_api import async_playwright  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Playwright is not installed or not available. Add 'playwright' to requirements and ensure browsers are installed."
                ) from e

            self._pw = await async_playwright().start()
            # Chromium is the most common for scraping
            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )

    async def _close_playwright(self) -> None:
        async with self._pw_lock:
            if self._browser:
                try:
                    await self._browser.close()
                except Exception:
                    pass
            self._browser = None
            if self._pw:
                try:
                    await self._pw.stop()
                except Exception:
                    pass
            self._pw = None

    async def _new_context(self) -> "BrowserContext":
        assert self._browser is not None
        ctx = await self._browser.new_context(
            user_agent=self.user_agent,
            java_script_enabled=True,
            viewport={"width": 1365, "height": 768},
        )
        return ctx

    async def _configure_routing(
        self,
        page: "Page",
        *,
        block_images: bool = True,
        block_fonts: bool = True,
        block_media: bool = True,
        block_stylesheets: bool = False,  # optional: can break rendering on some sites
    ) -> None:
        async def route_handler(route, request):
            rtype = request.resource_type
            if block_images and rtype == "image":
                return await route.abort()
            if block_fonts and rtype == "font":
                return await route.abort()
            if block_media and rtype in ("media",):
                return await route.abort()
            if block_stylesheets and rtype == "stylesheet":
                return await route.abort()
            return await route.continue_()

        await page.route("**/*", route_handler)

    async def _scroll_infinite(
        self,
        page: "Page",
        *,
        max_scrolls: int = 12,
        scroll_pause_ms: int = 600,
        stable_rounds: int = 2,
    ) -> None:
        """
        Scroll down until height stops growing for `stable_rounds` iterations or max_scrolls is hit.
        """
        last_h = 0
        stable = 0

        for _ in range(max_scrolls):
            h = await page.evaluate("() => document.body.scrollHeight")
            if h == last_h:
                stable += 1
                if stable >= stable_rounds:
                    break
            else:
                stable = 0
                last_h = h

            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(scroll_pause_ms)

    async def _paginate_click_next(
        self,
        page: "Page",
        *,
        max_pages: int = 4,
        wait_until: str = "networkidle",
        next_selectors: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Heuristic pagination: click a "next" control a few times and collect HTML snapshots.
        Returns list of HTML strings (including initial page already loaded).
        """
        snapshots: List[str] = [await page.content()]
        selectors = next_selectors or [
            "a[rel='next']",
            "button:has-text('Next')",
            "a:has-text('Next')",
            "button:has-text('›')",
            "a:has-text('›')",
        ]

        for _ in range(max_pages - 1):
            clicked = False
            for sel in selectors:
                loc = page.locator(sel)
                try:
                    if await loc.count() > 0 and await loc.first.is_visible():
                        await loc.first.click(timeout=1500)
                        await page.wait_for_load_state(wait_until, timeout=15000)
                        snapshots.append(await page.content())
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                break
        return snapshots

    async def _render_html_playwright(
        self,
        url: str,
        *,
        wait_until: str,
        wait_ms: int,
        max_retries: int = 2,
        enable_scroll: bool = True,
        enable_pagination: bool = False,
    ) -> Tuple[str, str, str]:
        """
        Returns (status, title, text) from Playwright rendering.
        - smart waits (selector heuristic)
        - resource blocking
        - optional infinite scroll
        - optional click pagination
        - request/response interception for debugging signals (not exposed)
        """
        await self._ensure_playwright()

        to_ms = int(self.default_timeout * 1000)

        for attempt in range(max_retries + 1):
            ctx: Optional["BrowserContext"] = None
            page: Optional["Page"] = None
            try:
                await self._polite_wait(url)

                ctx = await self._new_context()
                page = await ctx.new_page()

                # Block unnecessary resources to speed up + reduce bot friction
                await self._configure_routing(
                    page,
                    block_images=True,
                    block_fonts=True,
                    block_media=True,
                    block_stylesheets=False,
                )

                last_main_response: Dict[str, Any] = {"status": None, "url": None, "ctype": None}

                async def on_response(resp: "Response"):
                    try:
                        if resp.url == page.url or resp.request.is_navigation_request():
                            last_main_response["status"] = resp.status
                            last_main_response["url"] = resp.url
                            last_main_response["ctype"] = (resp.headers.get("content-type") or "").lower()
                    except Exception:
                        pass

                page.on("response", on_response)

                # Navigate
                await page.goto(url, wait_until=wait_until, timeout=to_ms)

                # Extra settle time if requested
                if wait_ms and wait_ms > 0:
                    await page.wait_for_timeout(wait_ms)

                # Smart wait: if initial HTML looks like a JS shell, wait for main containers
                initial_html = await page.content()
                sel = _pick_wait_selector(initial_html)
                if sel:
                    try:
                        await page.wait_for_selector(sel, timeout=min(15000, to_ms))
                    except Exception:
                        # selector not found; proceed anyway
                        pass

                # Optional scroll for infinite pages
                if enable_scroll:
                    try:
                        await self._scroll_infinite(page)
                    except Exception:
                        pass

                # Optional pagination click to gather additional content
                htmls: List[str]
                if enable_pagination:
                    try:
                        htmls = await self._paginate_click_next(page, wait_until=wait_until)
                    except Exception:
                        htmls = [await page.content()]
                else:
                    htmls = [await page.content()]

                # Faster text extraction via JS (often better than soup on dynamic sites)
                # Fallback to HTML parsing if innerText is empty.
                all_text_chunks: List[str] = []
                title = ""
                for h in htmls:
                    # Use page-evaluated innerText only for the final state of the page
                    # (For earlier snapshots, parse HTML.)
                    t, txt = _extract_title_and_text_from_html(h)
                    if not title and t:
                        title = t
                    if txt:
                        all_text_chunks.append(txt)

                # Try to get live innerText from current page
                try:
                    live_title = await page.title()
                    live_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
                    if live_title:
                        title = live_title
                    if live_text and len(live_text.strip()) > 0:
                        all_text_chunks.insert(0, live_text)
                except Exception:
                    pass

                combined = " ".join(x.strip() for x in all_text_chunks if x).strip()
                if len(combined) > MAX_TEXT_CHARS:
                    combined = combined[:MAX_TEXT_CHARS]

                # Block detection
                if _looks_like_block(combined) or _looks_like_block(" ".join(htmls)[:5000]):
                    return ("error_blocked", title, combined)

                return ("ok_playwright", title, combined)

            except Exception as e:
                if attempt < max_retries:
                    await asyncio.sleep(_exp_backoff(0.75, attempt))
                    continue
                return ("error_playwright", "", "")
            finally:
                try:
                    if page:
                        await page.close()
                except Exception:
                    pass
                try:
                    if ctx:
                        await ctx.close()
                except Exception:
                    pass

        return ("error_playwright", "", "")

    # -----------------------------
    # Crawl core
    # -----------------------------

    async def _fetch_page(
        self,
        url: str,
        *,
        render_js: bool,
        js_only: bool,
        wait_until: str,
        wait_ms: int,
    ) -> Tuple[str, str, str]:
        """
        Returns (status, title, text).
        - If js_only => Playwright only
        - Else try httpx first; if JS shell or empty or blocked and render_js enabled => Playwright
        """
        if js_only:
            return await self._render_html_playwright(
                url,
                wait_until=wait_until,
                wait_ms=wait_ms,
                enable_scroll=True,
                enable_pagination=False,
            )

        st_http, html = await self._fetch_html_httpx(url)
        if st_http.startswith("ok"):
            title, text = _extract_title_and_text_from_html(html)

            # If render_js, only escalate to Playwright when it looks like a shell/empty/blocked
            if render_js:
                if not text or _looks_like_js_shell(html) or _looks_like_block(text):
                    st_pw, title_pw, text_pw = await self._render_html_playwright(
                        url,
                        wait_until=wait_until,
                        wait_ms=wait_ms,
                        enable_scroll=True,
                        enable_pagination=False,
                    )
                    if st_pw.startswith("ok") and text_pw:
                        return (st_pw, title_pw, text_pw)

            if _looks_like_block(text):
                return ("error_blocked", title, text)
            return ("ok_http", title, text)

        # HTTP failed; optionally fallback to Playwright
        if render_js:
            return await self._render_html_playwright(
                url,
                wait_until=wait_until,
                wait_ms=wait_ms,
                enable_scroll=True,
                enable_pagination=False,
            )

        return (st_http, "", "")

    def _extract_links(self, base_url: str, html: str) -> List[str]:
        soup = BeautifulSoup(html or "", "lxml")
        links: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            href = href.strip()
            if not href:
                continue
            abs_url = urllib.parse.urljoin(base_url, href)
            # strip fragments
            abs_url = _normalize_url(abs_url)
            if abs_url.startswith("http"):
                links.append(abs_url)
        return links

    def _passes_filters(
        self,
        url: str,
        *,
        include_keywords: List[str],
        include_patterns: List[Pattern[str]],
        exclude_patterns: List[Pattern[str]],
        same_domain_only: bool,
        seed_domain: str,
    ) -> bool:
        if _should_skip_url(url):
            return False

        if same_domain_only and _domain(url) != seed_domain:
            return False

        if exclude_patterns and _match_any(exclude_patterns, url):
            return False

        if include_patterns and not _match_any(include_patterns, url):
            # If include patterns are given, require at least one match
            return False

        if include_keywords:
            # keep if any keyword in URL path/query
            u = url.lower()
            if not any(k.lower() in u for k in include_keywords):
                # still allow the seed/root pages
                pass

        return True

    async def crawl_relevant(
        self,
        seed_url: str,
        *,
        depth: int,
        max_pages: int,
        include_link_keywords: List[str],
        include_url_patterns: List[str],
        exclude_url_patterns: List[str],
        render_js: bool,
        js_only: bool,
        wait_until: str,
        wait_ms: int,
        same_domain_only: bool,
        max_concurrency: int,
        respect_robots: bool,
    ) -> List[Tuple[str, str, str, str]]:
        """
        Returns list of (url, status, title, text)
        """
        seed_url = _normalize_url(seed_url)
        seed_dom = _domain(seed_url)

        include_pats = _compile_patterns(include_url_patterns)
        exclude_pats = _compile_patterns(exclude_url_patterns)

        seen: Set[str] = set()
        q: Deque[Tuple[str, int]] = deque([(seed_url, 0)])

        results: List[Tuple[str, str, str, str]] = []
        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def worker(url: str, d: int) -> None:
            if url in seen:
                return
            seen.add(url)

            if not self._passes_filters(
                url,
                include_keywords=include_link_keywords,
                include_patterns=include_pats,
                exclude_patterns=exclude_pats,
                same_domain_only=same_domain_only,
                seed_domain=seed_dom,
            ):
                return

            if respect_robots:
                rp = await self._ensure_robots(url)
                if rp and not rp.can_fetch(self.user_agent, url):
                    results.append((url, "error_robots", "", ""))
                    return

            async with sem:
                st, title, text = await self._fetch_page(
                    url,
                    render_js=render_js,
                    js_only=js_only,
                    wait_until=wait_until,
                    wait_ms=wait_ms,
                )

            results.append((url, st, title, text))

        # BFS by depth
        while q and len(results) < max_pages:
            batch: List[Tuple[str, int]] = []
            # pull a small batch
            while q and len(batch) < max(2, max_concurrency * 2) and len(results) + len(batch) < max_pages:
                batch.append(q.popleft())

            await asyncio.gather(*(worker(u, d) for u, d in batch))

            # Expand links for successful pages, only if we can still go deeper
            if depth <= 0:
                continue

            # We only have text in results; for link expansion we need HTML.
            # Strategy: for expansion, refetch HTML cheaply via httpx where possible.
            for (u, st, _title, _text) in results[-len(batch):]:
                if not st.startswith("ok"):
                    continue
                # Only expand if within depth limit
                # We stored depth in batch; rebuild from batch map
            # Use batch map:
            depth_map = {u: d for (u, d) in batch}
            for (u, st, _title, _text) in results[-len(batch):]:
                d = depth_map.get(u, 0)
                if d >= depth:
                    continue
                if not st.startswith("ok"):
                    continue

                # Fetch HTML for links expansion (httpx only; fast)
                st2, html2 = await self._fetch_html_httpx(u, max_retries=1)
                if not st2.startswith("ok") or not html2:
                    continue

                for link in self._extract_links(u, html2):
                    if link not in seen:
                        q.append((link, d + 1))

        return results

    # -----------------------------
    # Jobs API
    # -----------------------------

    async def submit_crawl_job(self, seed_urls: List[str], params: CrawlParams) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        rec = JobRecord(
            job_id=job_id,
            created_at=now,
            updated_at=now,
            status="queued",
            params=params,
            seed_urls=[_normalize_url(u) for u in seed_urls],
            seeds_total=len(seed_urls),
            seeds_done=0,
        )
        async with self._jobs_lock:
            self._jobs[job_id] = rec
        await self._job_queue.put(job_id)
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._jobs_lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return {
                "job_id": rec.job_id,
                "status": rec.status,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "error": rec.error,
                "seeds_total": rec.seeds_total,
                "seeds_done": rec.seeds_done,
                "pages_attempted": rec.pages_attempted,
                "pages_stored": rec.pages_stored,
                "max_pages_stored": self._max_pages_stored_per_job,
            }

    async def get_job_results(self, job_id: str, *, offset: int, limit: int) -> Optional[List[PageResult]]:
        async with self._jobs_lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return rec.results[offset: offset + limit]

    async def cancel_job(self, job_id: str) -> bool:
        async with self._jobs_lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return False
            rec.cancel_event.set()
            rec.status = "cancelled"
            rec.updated_at = time.time()
            return True

    async def stream_job_events(self, job_id: str) -> AsyncGenerator[dict, None]:
        """
        Optional: event streaming for UIs. Not required by main.py.
        """
        async with self._jobs_lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            q = rec.events

        while True:
            evt = await q.get()
            yield evt
            if evt.get("type") in ("done", "error", "cancelled"):
                return

    async def _emit(self, rec: JobRecord, evt: dict) -> None:
        try:
            rec.events.put_nowait(evt)
        except asyncio.QueueFull:
            # drop events under pressure
            pass

    async def _job_worker_loop(self) -> None:
        while True:
            job_id = await self._job_queue.get()
            async with self._jobs_lock:
                rec = self._jobs.get(job_id)
            if not rec:
                continue
            if rec.cancel_event.is_set():
                continue
            try:
                rec.status = "running"
                rec.updated_at = time.time()
                await self._emit(rec, {"type": "running", "job_id": rec.job_id})

                # Execute seeds sequentially (each seed can be concurrent internally)
                for seed in rec.seed_urls:
                    if rec.cancel_event.is_set():
                        rec.status = "cancelled"
                        rec.updated_at = time.time()
                        await self._emit(rec, {"type": "cancelled", "job_id": rec.job_id})
                        break

                    pages = await self.crawl_relevant(
                        seed,
                        depth=rec.params.depth,
                        max_pages=rec.params.max_pages,
                        include_link_keywords=rec.params.include_link_keywords,
                        include_url_patterns=rec.params.include_url_patterns,
                        exclude_url_patterns=rec.params.exclude_url_patterns,
                        render_js=rec.params.render_js,
                        js_only=rec.params.js_only,
                        wait_until=rec.params.wait_until,
                        wait_ms=rec.params.wait_ms,
                        same_domain_only=rec.params.same_domain_only,
                        max_concurrency=rec.params.max_concurrency,
                        respect_robots=rec.params.respect_robots,
                    )

                    rec.seeds_done += 1
                    rec.updated_at = time.time()

                    # Store bounded results
                    for (u, st, title, text) in pages:
                        rec.pages_attempted += 1
                        if rec.pages_stored >= self._max_pages_stored_per_job:
                            continue
                        rec.results.append(PageResult(url=u, status=st, title=title, text=text))
                        rec.pages_stored += 1

                    await self._emit(
                        rec,
                        {
                            "type": "progress",
                            "job_id": rec.job_id,
                            "seeds_done": rec.seeds_done,
                            "seeds_total": rec.seeds_total,
                            "pages_attempted": rec.pages_attempted,
                            "pages_stored": rec.pages_stored,
                        },
                    )

                if rec.status not in ("cancelled", "error"):
                    rec.status = "done"
                    rec.updated_at = time.time()
                    await self._emit(rec, {"type": "done", "job_id": rec.job_id})

            except Exception as e:
                rec.status = "error"
                rec.error = f"{type(e).__name__}: {e}"
                rec.updated_at = time.time()
                await self._emit(rec, {"type": "error", "job_id": rec.job_id, "error": rec.error})
            finally:
                self._job_queue.task_done()

    async def _cleanup_jobs_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            cutoff = time.time() - self._job_ttl_seconds
            async with self._jobs_lock:
                to_delete = [jid for jid, rec in self._jobs.items() if rec.updated_at < cutoff]
                for jid in to_delete:
                    del self._jobs[jid]
