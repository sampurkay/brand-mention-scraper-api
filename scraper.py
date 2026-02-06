from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import httpx
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

# Playwright is optional. We import lazily at runtime when render_js is enabled.
if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright  # type: ignore


DEFAULT_DELAY: float = 1.0  # seconds per domain
DEFAULT_TIMEOUT: float = 12.0
MAX_TEXT_CHARS: int = 200_000

# ---- Async-first job defaults ----
DEFAULT_WORKERS: int = 2
DEFAULT_JOB_TTL_SECONDS: int = 60 * 60  # 1 hour
DEFAULT_MAX_PAGES_STORED_PER_JOB: int = 2000
DEFAULT_EVENT_QUEUE_SIZE: int = 1000

# Heuristics to detect blocks / bot walls (not exhaustive)
_BLOCK_MARKERS = (
    "captcha",
    "cloudflare",
    "attention required",
    "access denied",
    "temporarily unavailable",
    "unusual traffic",
    "verify you are human",
)

# Heuristics to detect JS-shell pages that often need rendering
_JS_SHELL_MARKERS = (
    "__NEXT_DATA__",
    "id=\"__next\"",
    "data-reactroot",
    "ng-version",
    "id=\"app\"",
)

# Skip common non-content file types during crawling
_SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".mp4",
    ".mov",
    ".avi",
    ".mp3",
    ".wav",
    ".css",
    ".js",
    ".xml",
)


def _normalize_url(url: str) -> str:
    """Normalize URL for dedupe (strip fragment, normalize scheme/host)."""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query
    return urllib.parse.urlunparse((scheme, netloc, path, "", query, ""))


def _same_domain(a: str, b: str) -> bool:
    return urllib.parse.urlparse(a).netloc.lower() == urllib.parse.urlparse(b).netloc.lower()


def _looks_blocked(status_code: int, body_text: str) -> bool:
    if status_code in (401, 403, 429):
        return True
    lower = (body_text or "").lower()
    return any(m in lower for m in _BLOCK_MARKERS)


def _looks_js_shell(html: str) -> bool:
    if not html:
        return True
    if len(html) < 1200:
        return True
    lower = html.lower()
    return any(m.lower() in lower for m in _JS_SHELL_MARKERS)


def _extract_title(soup: BeautifulSoup) -> str:
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


def _html_to_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def _is_likely_html(content_type: str) -> bool:
    return "text/html" in (content_type or "").lower()


def _is_likely_json(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return "application/json" in ct or ct.endswith("+json")


def _json_to_text(payload: object, max_chars: int = MAX_TEXT_CHARS) -> str:
    strings: List[str] = []

    def walk(x: object) -> None:
        if x is None:
            return
        if isinstance(x, str):
            if x.strip():
                strings.append(x.strip())
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, list):
            for i in x:
                walk(i)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return

    walk(payload)
    joined = " ".join(strings)
    if joined:
        return joined[:max_chars]
    try:
        s = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        s = str(payload)
    return s[:max_chars]


def _attr_to_str(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        parts = [p for p in val if isinstance(p, str)]
        return " ".join(parts)
    return str(val)


def _extract_links(base_url: str, soup: BeautifulSoup, same_domain_only: bool) -> List[str]:
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = _attr_to_str(a.get("href")).strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        abs_url = _normalize_url(urllib.parse.urljoin(base_url, href))
        if same_domain_only and not _same_domain(abs_url, base_url):
            continue
        links.append(abs_url)
    return links


def _compile_patterns(patterns: Optional[List[str]]) -> List[Pattern[str]]:
    import re

    compiled: List[Pattern[str]] = []
    for p in patterns or []:
        p = (p or "").strip()
        if not p:
            continue
        try:
            compiled.append(re.compile(p, flags=re.IGNORECASE))
        except Exception:
            continue
    return compiled


def _matches_any(url: str, patterns: List[Pattern[str]]) -> bool:
    return any(p.search(url) is not None for p in patterns)


def _is_skippable_url(url: str) -> bool:
    lower = url.lower()
    if lower.endswith(_SKIP_EXTENSIONS):
        return True
    if lower.startswith(("mailto:", "tel:", "javascript:")):
        return True
    return False


# -----------------------------
# Async-first job infrastructure
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
    wait_until: str = "networkidle"
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

    # progress
    pages_stored: int = 0
    pages_attempted: int = 0
    seeds_total: int = 0
    seeds_done: int = 0

    # storage (bounded)
    results: List[PageResult] = field(default_factory=list)

    # error
    error: Optional[str] = None

    # cancellation
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # progress events
    events: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_EVENT_QUEUE_SIZE))


class PoliteScraper:
    """
    Keeps your original crawl/fetch behavior, and adds:
      - submit_crawl_job(...) -> job_id (returns immediately)
      - get_job_status(job_id) -> dict
      - get_job_results(job_id, offset, limit) -> List[PageResult]
      - cancel_job(job_id)
      - stream_job_events(job_id) -> async generator of progress events

    This enables an async-first architecture where GPT Actions can:
      1) submit job
      2) poll status
      3) fetch results in pages
    """

    def __init__(
        self,
        user_agent: str = "ReplitScraperBot/1.0",
        *,
        default_timeout: float = DEFAULT_TIMEOUT,
        job_workers: int = DEFAULT_WORKERS,
        job_ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS,
        max_pages_stored_per_job: int = DEFAULT_MAX_PAGES_STORED_PER_JOB,
    ) -> None:
        self.user_agent = user_agent
        self.default_timeout = default_timeout

        self.domain_rules: Dict[str, RobotFileParser] = {}
        self.last_access: Dict[str, float] = {}
        self.crawl_delay: Dict[str, float] = {}

        self._client: httpx.AsyncClient = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

        # Playwright (lazy)
        self._pw: Optional["Playwright"] = None
        self._browser: Optional["Browser"] = None
        self._pw_lock = asyncio.Lock()

        # Job infra
        self._jobs: Dict[str, JobRecord] = {}
        self._job_lock = asyncio.Lock()
        self._job_queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers_started = False
        self._worker_tasks: List[asyncio.Task[None]] = []
        self._job_workers = max(1, int(job_workers))
        self._job_ttl_seconds = int(job_ttl_seconds)
        self._max_pages_stored_per_job = int(max_pages_stored_per_job)

    # -----------------
    # Job public methods
    # -----------------

    async def submit_crawl_job(self, seed_urls: List[str], params: CrawlParams) -> str:
        """
        Enqueue a crawl job and return a job_id immediately.
        """
        # start workers lazily under a running loop
        await self._ensure_workers()

        norm_seeds = [_normalize_url(str(u)) for u in seed_urls if str(u).strip()]
        job_id = uuid.uuid4().hex
        now = time.time()

        rec = JobRecord(
            job_id=job_id,
            created_at=now,
            updated_at=now,
            status="queued",
            params=params,
            seed_urls=norm_seeds,
            seeds_total=len(norm_seeds),
            seeds_done=0,
        )

        async with self._job_lock:
            self._jobs[job_id] = rec
            await self._job_queue.put(job_id)

        await self._emit_event(job_id, {"type": "job_queued", "job_id": job_id, "seeds_total": rec.seeds_total})
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[dict]:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
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

    async def get_job_results(self, job_id: str, *, offset: int = 0, limit: int = 100) -> Optional[List[PageResult]]:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            o = max(0, int(offset))
            l = max(1, min(1000, int(limit)))
            return rec.results[o : o + l]

    async def cancel_job(self, job_id: str) -> bool:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return False
            rec.cancel_event.set()
            rec.updated_at = time.time()
            # Worker will mark status as cancelled soon; we mark intent here.
            if rec.status in ("queued", "running"):
                rec.status = "cancelled"
        await self._emit_event(job_id, {"type": "job_cancel_requested", "job_id": job_id})
        return True

    async def stream_job_events(self, job_id: str) -> Optional[asyncio.Queue[dict]]:
        """
        Returns the per-job event queue. Your FastAPI layer can convert this into SSE/websocket.
        """
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return rec.events

    # -----------------------
    # Internal worker routines
    # -----------------------

    async def _ensure_workers(self) -> None:
        if self._workers_started:
            return
        # This must run inside an event loop (FastAPI runtime is fine).
        self._workers_started = True
        for i in range(self._job_workers):
            self._worker_tasks.append(asyncio.create_task(self._job_worker(i)))

        # cleanup task
        self._worker_tasks.append(asyncio.create_task(self._job_cleanup_loop()))

    async def _job_worker(self, worker_id: int) -> None:
        while True:
            job_id = await self._job_queue.get()
            try:
                await self._run_job(job_id, worker_id=worker_id)
            except Exception as e:
                # mark job error if still exists
                async with self._job_lock:
                    rec = self._jobs.get(job_id)
                    if rec is not None:
                        rec.status = "error"
                        rec.error = f"{type(e).__name__}: {e}"
                        rec.updated_at = time.time()
                await self._emit_event(job_id, {"type": "job_error", "job_id": job_id, "error": str(e)})
            finally:
                self._job_queue.task_done()

    async def _job_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            cutoff = time.time() - self._job_ttl_seconds
            async with self._job_lock:
                to_delete = [jid for jid, rec in self._jobs.items() if rec.updated_at < cutoff]
                for jid in to_delete:
                    self._jobs.pop(jid, None)

    async def _emit_event(self, job_id: str, event: dict) -> None:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            # non-blocking put (drop if full)
            try:
                rec.events.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def _run_job(self, job_id: str, *, worker_id: int) -> None:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            # if cancelled before start
            if rec.cancel_event.is_set() or rec.status == "cancelled":
                rec.status = "cancelled"
                rec.updated_at = time.time()
                return
            rec.status = "running"
            rec.updated_at = time.time()

        await self._emit_event(job_id, {"type": "job_started", "job_id": job_id, "worker_id": worker_id})

        # Process each seed URL sequentially (each seed itself uses concurrency internally)
        for seed in await self._get_job_seeds(job_id):
            if await self._job_cancelled(job_id):
                await self._emit_event(job_id, {"type": "job_cancelled", "job_id": job_id})
                return

            await self._emit_event(job_id, {"type": "seed_started", "job_id": job_id, "seed_url": seed})

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
                job_id=job_id,  # enables progress increments
            )

            # Store results (bounded)
            async with self._job_lock:
                rec2 = self._jobs.get(job_id)
                if rec2 is None:
                    return
                for (url, status, title, text) in pages:
                    if rec2.pages_stored >= self._max_pages_stored_per_job:
                        break
                    rec2.results.append(PageResult(url=url, status=status, title=title, text=text))
                    rec2.pages_stored += 1
                rec2.seeds_done += 1
                rec2.updated_at = time.time()

            await self._emit_event(job_id, {"type": "seed_done", "job_id": job_id, "seed_url": seed})

        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            if rec.status != "cancelled":
                rec.status = "done"
            rec.updated_at = time.time()

        await self._emit_event(job_id, {"type": "job_done", "job_id": job_id})

    async def _get_job_seeds(self, job_id: str) -> List[str]:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            return rec.seed_urls[:] if rec else []

    async def _job_cancelled(self, job_id: str) -> bool:
        async with self._job_lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return True
            return rec.cancel_event.is_set() or rec.status == "cancelled"

    # -----------------------------
    # Original scraper functionality
    # -----------------------------

    def get_domain(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc.lower()

    def _robots_url(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        return f"{scheme}://{netloc}/robots.txt"

    async def load_robots(self, url: str) -> None:
        domain = self.get_domain(url)
        robots_url = self._robots_url(url)

        rp = RobotFileParser()
        try:
            resp = await self._client.get(robots_url, timeout=5.0)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.parse([])
        except Exception:
            rp.parse([])

        raw_delay: Union[str, float, None] = rp.crawl_delay(self.user_agent)
        if raw_delay is None:
            delay: float = DEFAULT_DELAY
        else:
            try:
                delay = float(raw_delay)
            except (TypeError, ValueError):
                delay = DEFAULT_DELAY

        self.domain_rules[domain] = rp
        self.crawl_delay[domain] = delay

    async def can_fetch(self, url: str, *, respect_robots: bool = True) -> bool:
        if not respect_robots:
            return True
        domain = self.get_domain(url)
        if domain not in self.domain_rules:
            await self.load_robots(url)
        rp = self.domain_rules[domain]
        return rp.can_fetch(self.user_agent, url)

    async def obey_rate_limit(self, url: str) -> None:
        domain = self.get_domain(url)
        delay: float = self.crawl_delay.get(domain, DEFAULT_DELAY)
        last: float = self.last_access.get(domain, 0.0)
        now: float = time.time()
        wait: float = delay - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self.last_access[domain] = time.time()

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

    async def _get_html_http(self, url: str) -> Tuple[str, str, str]:
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
            return ("error_timeout", "", "")
        except httpx.HTTPError:
            return ("error_http", "", "")
        except Exception as e:
            return (f"error_{type(e).__name__}", "", "")

    async def _get_html_playwright(
        self,
        url: str,
        *,
        wait_until: str = "networkidle",
        wait_ms: int = 0,
        timeout: Optional[float] = None,
    ) -> Tuple[str, str]:
        try:
            browser = await self._ensure_browser()
            context = await browser.new_context(user_agent=self.user_agent, java_script_enabled=True)
            page = await context.new_page()
            to_ms = int((timeout or (self.default_timeout * 2)) * 1000)

            try:
                await page.goto(url, wait_until=wait_until, timeout=to_ms)  # type: ignore[arg-type]
                if wait_ms > 0:
                    await page.wait_for_timeout(wait_ms)
                html = await page.content()
            finally:
                await page.close()
                await context.close()

            if _looks_blocked(200, html):
                return ("blocked", "")
            return ("ok", html)
        except asyncio.TimeoutError:
            return ("error_playwright_timeout", "")
        except Exception:
            return ("error_playwright", "")

    async def fetch(
        self,
        url: str,
        *,
        render_js: bool = False,
        js_only: bool = False,
        wait_until: str = "networkidle",
        wait_ms: int = 0,
        respect_robots: bool = True,
    ) -> Tuple[str, str]:
        url = _normalize_url(url)

        if not await self.can_fetch(url, respect_robots=respect_robots):
            return ("blocked_by_robots", "")

        await self.obey_rate_limit(url)

        if render_js and js_only:
            st_html, html = await self._get_html_playwright(url, wait_until=wait_until, wait_ms=wait_ms)
            if not st_html.startswith("ok"):
                return (st_html, "")
            soup = BeautifulSoup(html, "html.parser")
            text = _html_to_text(soup)
            return ("ok", text if text else html[:MAX_TEXT_CHARS])

        st, body, ct = await self._get_html_http(url)
        if not st.startswith("ok"):
            # fallback to Playwright if enabled
            if render_js and (st in ("blocked", "error_timeout", "error_http") or st.startswith("http_")):
                st_html, html = await self._get_html_playwright(url, wait_until=wait_until, wait_ms=wait_ms)
                if not st_html.startswith("ok"):
                    return (st_html, "")
                soup = BeautifulSoup(html, "html.parser")
                text = _html_to_text(soup)
                return ("ok", text if text else html[:MAX_TEXT_CHARS])
            return (st, "")

        if _is_likely_json(ct):
            try:
                data = json.loads(body)
            except Exception:
                return ("ok", body[:MAX_TEXT_CHARS])
            return ("ok_json", _json_to_text(data))

        if _is_likely_html(ct) or "<html" in body.lower():
            if render_js and _looks_js_shell(body):
                st_html, html = await self._get_html_playwright(url, wait_until=wait_until, wait_ms=wait_ms)
                if st_html.startswith("ok"):
                    soup = BeautifulSoup(html, "html.parser")
                    text = _html_to_text(soup)
                    return ("ok", text if text else html[:MAX_TEXT_CHARS])
            soup = BeautifulSoup(body, "html.parser")
            return ("ok", _html_to_text(soup))

        return ("ok", body[:MAX_TEXT_CHARS])

    async def fetch_with_links(
        self,
        url: str,
        *,
        render_js: bool = False,
        js_only: bool = False,
        wait_until: str = "networkidle",
        wait_ms: int = 0,
        same_domain_only: bool = True,
        respect_robots: bool = True,
    ) -> Tuple[str, str, List[str], str]:
        url = _normalize_url(url)

        status, text = await self.fetch(
            url,
            render_js=render_js,
            js_only=js_only,
            wait_until=wait_until,
            wait_ms=wait_ms,
            respect_robots=respect_robots,
        )

        html: str = ""
        title: str = ""
        links: List[str] = []

        if status.startswith("ok"):
            if render_js and js_only:
                st_html, html = await self._get_html_playwright(url, wait_until=wait_until, wait_ms=wait_ms)
                if not st_html.startswith("ok"):
                    return status, text, [], ""
            else:
                if not await self.can_fetch(url, respect_robots=respect_robots):
                    return status, text, [], ""
                await self.obey_rate_limit(url)
                st_http, body, ct = await self._get_html_http(url)
                if st_http.startswith("ok") and (_is_likely_html(ct) or "<html" in body.lower()):
                    html = body
                elif render_js:
                    st_html, html = await self._get_html_playwright(url, wait_until=wait_until, wait_ms=wait_ms)
                    if not st_html.startswith("ok"):
                        html = ""

        if html:
            soup = BeautifulSoup(html, "html.parser")
            title = _extract_title(soup)
            links = _extract_links(url, soup, same_domain_only=same_domain_only)

        return status, text, links, title

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
        job_id: Optional[str] = None,  # NEW: progress updates if provided
    ) -> List[Tuple[str, str, str, str]]:
        start_url = _normalize_url(start_url)

        include_keywords = [kw for kw in (include_link_keywords or []) if (kw or "").strip()]
        include_patterns = _compile_patterns(include_url_patterns)
        exclude_patterns = _compile_patterns(exclude_url_patterns)

        visited: Set[str] = set()
        queue: Deque[Tuple[str, int]] = deque([(start_url, 0)])
        results: List[Tuple[str, str, str, str]] = []

        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def process_one(url_in: str, d: int) -> Tuple[str, int, str, str, str, List[str]]:
            async with sem:
                url_norm = _normalize_url(url_in)

                # If job cancelled, short-circuit quickly
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
                )

                if job_id is not None:
                    async with self._job_lock:
                        rec = self._jobs.get(job_id)
                        if rec is not None:
                            rec.pages_attempted += 1
                            rec.updated_at = time.time()
                    await self._emit_event(
                        job_id,
                        {
                            "type": "page_done",
                            "job_id": job_id,
                            "url": url_norm,
                            "status": st,
                            "depth": d,
                        },
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
                # cancellation marker
                if status == "cancelled":
                    continue

                results.append((url_norm, status, title, text))
                if len(results) >= max_pages:
                    break

                if d >= depth:
                    continue
                if status.startswith(("blocked", "error")) or status.startswith("http_"):
                    continue

                if include_keywords:
                    low_keywords = [k.lower() for k in include_keywords]
                    next_links = []
                    for lnk in links:
                        lnk_n = _normalize_url(lnk)
                        if any(k in lnk_n.lower() for k in low_keywords):
                            next_links.append(lnk_n)
                else:
                    next_links = [_normalize_url(lnk) for lnk in links]

                for lnk in next_links:
                    if len(results) + len(queue) >= max_pages:
                        break
                    if lnk in visited:
                        continue
                    if not self._url_allowed(
                        lnk,
                        same_domain_only=same_domain_only,
                        start_url=start_url,
                        include_patterns=include_patterns,
                        exclude_patterns=exclude_patterns,
                    ):
                        continue
                    queue.append((lnk, d + 1))

        return results

    async def aclose(self) -> None:
        await self._client.aclose()

        async with self._pw_lock:
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
                self._browser = None
            if self._pw is not None:
                try:
                    await self._pw.stop()
                except Exception:
                    pass
                self._pw = None
