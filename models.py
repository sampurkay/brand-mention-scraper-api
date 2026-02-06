from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, HttpUrl


class Product(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)
    skus: List[str] = Field(default_factory=list)


class Brand(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)
    # Allow empty list so you can do brand-only matching if desired
    products: List[Product] = Field(default_factory=list)


class ScrapeOptions(BaseModel):
    context_chars: int = Field(default=200, ge=0, le=5000)


class PlaywrightOptions(BaseModel):
    """
    Controls specific to JS-rendered pages.
    These are safe defaults; enable features only when needed.
    """

    # --- Navigation / waiting ---
    wait_for_selector: Optional[str] = Field(
        default=None,
        description="If set, Playwright will wait for this selector before extracting content (useful for SPAs).",
    )
    selector_timeout_ms: int = Field(
        default=15000,
        ge=0,
        le=60000,
        description="Timeout for wait_for_selector.",
    )

    # --- Resource blocking (speed + lower bandwidth) ---
    block_images: bool = Field(default=True)
    block_fonts: bool = Field(default=True)
    block_media: bool = Field(default=True)
    block_stylesheets: bool = Field(
        default=False,
        description="Can speed up but may break layout-dependent rendering.",
    )

    # --- Extraction ---
    prefer_inner_text: bool = Field(
        default=True,
        description="If true, tries document.body.innerText (often better for dynamic pages) before HTML parsing fallback.",
    )

    # --- Infinite scroll ---
    enable_infinite_scroll: bool = Field(
        default=False,
        description="Enable scrolling to load additional content (feeds, long pages).",
    )
    max_scrolls: int = Field(default=12, ge=0, le=200)
    scroll_pause_ms: int = Field(default=600, ge=0, le=10000)
    stable_rounds: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Stop scrolling after page height doesn't change for this many rounds.",
    )

    # --- Pagination clicking (heuristic) ---
    enable_pagination: bool = Field(
        default=False,
        description="Enable heuristic clicking of Next controls to traverse pagination.",
    )
    pagination_max_pages: int = Field(default=4, ge=1, le=50)
    next_selectors: List[str] = Field(
        default_factory=list,
        description="Optional CSS/text selectors tried in order to click 'next'. If empty, built-in defaults are used.",
    )


class RetryOptions(BaseModel):
    base_delay_s: float = Field(
        default=0.75,
        ge=0.0,
        le=30.0,
        description="Exponential backoff base delay in seconds.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Max retries per page fetch (HTTP or Playwright).",
    )


class ScrapeRequest(BaseModel):
    # Seed URLs: keep strict/validated
    urls: List[HttpUrl]
    brands: List[Brand]
    options: ScrapeOptions = Field(default_factory=ScrapeOptions)

    # ---- Crawl controls ----
    crawl_depth: int = Field(
        default=0,
        ge=0,
        le=10,
        description="BFS crawl depth from each seed URL (0 = only seed page).",
    )
    max_pages: int = Field(
        default=30,
        ge=1,
        le=2000,
        description="Max pages to fetch per seed URL (cap to avoid runaway crawls).",
    )
    same_domain_only: bool = Field(
        default=True,
        description="Restrict crawling to the same domain as the seed URL.",
    )

    # If empty => follow all internal links (subject to depth/max_pages)
    include_link_keywords: List[str] = Field(
        default_factory=list,
        description="If provided, only follow links whose URL contains any of these keywords.",
    )

    # Optional regex patterns for more control (applied to absolute URL strings)
    include_url_patterns: List[str] = Field(
        default_factory=list,
        description="If provided, only URLs matching at least one regex will be crawled.",
    )
    exclude_url_patterns: List[str] = Field(
        default_factory=list,
        description="URLs matching any regex will be skipped (e.g. login/cart/pdf).",
    )

    # ---- JS rendering controls ----
    render_js: bool = Field(
        default=False,
        description="Enable Playwright rendering for JS-heavy pages.",
    )
    js_only: bool = Field(
        default=False,
        description="If true and render_js=true, always use Playwright (not just fallback).",
    )
    wait_until: Literal["load", "domcontentloaded", "networkidle"] = Field(
        default="networkidle",
        description="Playwright navigation wait condition.",
    )
    wait_ms: int = Field(
        default=0,
        ge=0,
        le=30000,
        description="Extra milliseconds to wait after navigation (helps SPAs hydrate).",
    )

    playwright: PlaywrightOptions = Field(
        default_factory=PlaywrightOptions,
        description="Playwright-specific controls (waiting, scrolling, pagination, resource blocking).",
    )

    retry: RetryOptions = Field(
        default_factory=RetryOptions,
        description="Retry behavior with exponential backoff.",
    )

    # ---- Performance / politeness ----
    max_concurrency: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Max concurrent fetches during crawling. Keep low for JS rendering.",
    )
    respect_robots: bool = Field(
        default=True,
        description="If false, skip robots.txt checks (not recommended).",
    )


class MatchSnippet(BaseModel):
    brand: str
    product: Optional[str] = None
    matched_term: str
    text_snippet: str


class ProductMatch(BaseModel):
    brand: str
    product: Optional[str] = None
    count: int
    matches: List[MatchSnippet]


class UrlResult(BaseModel):
    # Crawled/discovered URLs may fail strict HttpUrl validation; keep as str for robustness
    url: str
    status: str
    product_mentions: List[ProductMatch]


class SummaryItem(BaseModel):
    brand: str
    product: Optional[str] = None
    total_mentions: int
    urls_with_mentions: int


class ScrapeSummary(BaseModel):
    total_urls: int
    items: List[SummaryItem]


class ScrapeResponse(BaseModel):
    summary: ScrapeSummary
    results_by_url: List[UrlResult]
