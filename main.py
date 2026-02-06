from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app.models import (
    ScrapeRequest,
    ScrapeResponse,
    UrlResult,
    ScrapeSummary,
    SummaryItem,
)
from app.scraper import PoliteScraper, CrawlParams
from app.matching import find_matches_in_text


# Global scraper instance (closed via lifespan)
scraper = PoliteScraper()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await scraper.aclose()


app = FastAPI(lifespan=lifespan)


# -----------------------
# Small helper models
# -----------------------

class JobSubmitResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    updated_at: float
    error: Optional[str] = None
    seeds_total: int
    seeds_done: int
    pages_attempted: int
    pages_stored: int
    max_pages_stored: int


class JobResultsResponse(BaseModel):
    job_id: str
    offset: int
    limit: int
    total_returned: int
    results: List[UrlResult]


class JobSummaryResponse(BaseModel):
    job_id: str
    summary: ScrapeSummary


class JobCancelResponse(BaseModel):
    job_id: str
    cancelled: bool


# -----------------------
# Basic endpoints
# -----------------------

@app.get("/")
def home() -> Dict[str, str]:
    return {"status": "ok", "message": "Scraper API running"}


# --------------------------------------------------------------------
# Legacy single-shot endpoint (kept for manual testing; may timeout)
# --------------------------------------------------------------------

@app.post("/api/scrape-product-mentions", response_model=ScrapeResponse)
async def scrape(payload: ScrapeRequest) -> ScrapeResponse:
    """
    Single-shot scrape (can time out for deep crawls / JS sites).
    Prefer the async-first job endpoints below for GPT Actions.
    """
    results: List[UrlResult] = []
    summary: dict[str, dict[str, dict[str, int]]] = {b.name: {} for b in payload.brands}
    url_sets: dict[tuple[str, str], set[str]] = {}

    for seed_url in payload.urls:
        seed_url_str = str(seed_url)

        pages = await scraper.crawl_relevant(
            seed_url_str,
            depth=payload.crawl_depth,
            max_pages=payload.max_pages,
            include_link_keywords=payload.include_link_keywords,
            include_url_patterns=payload.include_url_patterns,
            exclude_url_patterns=payload.exclude_url_patterns,
            render_js=payload.render_js,
            js_only=payload.js_only,
            wait_until=payload.wait_until,
            wait_ms=payload.wait_ms,
            same_domain_only=payload.same_domain_only,
            max_concurrency=payload.max_concurrency,
            respect_robots=payload.respect_robots,
        )

        for page_url_str, status, _title, text in pages:
            if not status.startswith("ok"):
                results.append(
                    UrlResult(url=page_url_str, status=status, product_mentions=[])
                )
                continue

            matches = find_matches_in_text(
                text=text, brands=payload.brands, options=payload.options
            )

            for m in matches:
                brand_key: str = m.brand
                product_key: str = m.product or "_brand_only"

                summary.setdefault(brand_key, {})
                summary[brand_key].setdefault(product_key, {"mentions": 0, "urls": 0})
                summary[brand_key][product_key]["mentions"] += int(m.count)

                pair = (brand_key, product_key)
                url_sets.setdefault(pair, set())
                if page_url_str not in url_sets[pair]:
                    summary[brand_key][product_key]["urls"] += 1
                    url_sets[pair].add(page_url_str)

            results.append(
                UrlResult(url=page_url_str, status=status, product_mentions=matches)
            )

    summary_items: List[SummaryItem] = []
    for brand_key, pdata in summary.items():
        for product_key, stats in pdata.items():
            summary_items.append(
                SummaryItem(
                    brand=brand_key,
                    product=None if product_key == "_brand_only" else product_key,
                    total_mentions=stats["mentions"],
                    urls_with_mentions=stats["urls"],
                )
            )

    return ScrapeResponse(
        summary=ScrapeSummary(total_urls=len(results), items=summary_items),
        results_by_url=results,
    )


# ----------------------------------------------------------
# Async-first, Action-friendly endpoints (recommended)
# ----------------------------------------------------------

@app.post("/api/jobs", response_model=JobSubmitResponse)
async def create_job(payload: ScrapeRequest) -> JobSubmitResponse:
    """
    Submit a crawl job and return a job_id immediately.
    This avoids HTTP timeouts and GPT action execution limits.
    """
    params = CrawlParams(
        depth=payload.crawl_depth,
        max_pages=payload.max_pages,
        include_link_keywords=payload.include_link_keywords,
        include_url_patterns=payload.include_url_patterns,
        exclude_url_patterns=payload.exclude_url_patterns,
        render_js=payload.render_js,
        js_only=payload.js_only,
        wait_until=payload.wait_until,
        wait_ms=payload.wait_ms,
        same_domain_only=payload.same_domain_only,
        max_concurrency=payload.max_concurrency,
        respect_robots=payload.respect_robots,
    )
    job_id = await scraper.submit_crawl_job([str(u) for u in payload.urls], params)
    return JobSubmitResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    st = await scraper.get_job_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JobStatusResponse(**st)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobCancelResponse)
async def job_cancel(job_id: str) -> JobCancelResponse:
    ok = await scraper.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job_id not found")
    return JobCancelResponse(job_id=job_id, cancelled=True)


@app.get("/api/jobs/{job_id}/results", response_model=JobResultsResponse)
async def job_results(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> JobResultsResponse:
    page_results = await scraper.get_job_results(job_id, offset=offset, limit=limit)
    if page_results is None:
        raise HTTPException(status_code=404, detail="job_id not found")

    # Return page URLs + status only (mentions are computed via /summary)
    url_results: List[UrlResult] = [
        UrlResult(url=pr.url, status=pr.status, product_mentions=[])
        for pr in page_results
    ]

    return JobResultsResponse(
        job_id=job_id,
        offset=offset,
        limit=limit,
        total_returned=len(url_results),
        results=url_results,
    )


@app.post("/api/jobs/{job_id}/summary", response_model=JobSummaryResponse)
async def job_summary(job_id: str, payload: ScrapeRequest) -> JobSummaryResponse:
    """
    Compute mention summary for a completed or running job using the same matching rules
    as the single-shot endpoint.
    """
    stored = await scraper.get_job_results(job_id, offset=0, limit=10_000)
    if stored is None:
        raise HTTPException(status_code=404, detail="job_id not found")

    url_results: List[UrlResult] = []
    for pr in stored:
        if not pr.status.startswith("ok"):
            url_results.append(
                UrlResult(url=pr.url, status=pr.status, product_mentions=[])
            )
            continue

        matches = find_matches_in_text(
            text=pr.text, brands=payload.brands, options=payload.options
        )
        url_results.append(
            UrlResult(url=pr.url, status=pr.status, product_mentions=matches)
        )

    summary = _summarize_url_results(url_results, payload)
    return JobSummaryResponse(job_id=job_id, summary=summary)


def _summarize_url_results(
    url_results: List[UrlResult], payload: ScrapeRequest
) -> ScrapeSummary:
    summary: dict[str, dict[str, dict[str, int]]] = {b.name: {} for b in payload.brands}
    url_sets: dict[tuple[str, str], set[str]] = {}

    for r in url_results:
        if not r.status.startswith("ok"):
            continue

        for pm in r.product_mentions:
            brand_key = pm.brand
            product_key = pm.product or "_brand_only"

            summary.setdefault(brand_key, {})
            summary[brand_key].setdefault(product_key, {"mentions": 0, "urls": 0})
            summary[brand_key][product_key]["mentions"] += int(pm.count)

            pair = (brand_key, product_key)
            url_sets.setdefault(pair, set())
            if str(r.url) not in url_sets[pair]:
                summary[brand_key][product_key]["urls"] += 1
                url_sets[pair].add(str(r.url))

    items: List[SummaryItem] = []
    for brand_key, pdata in summary.items():
        for product_key, stats in pdata.items():
            items.append(
                SummaryItem(
                    brand=brand_key,
                    product=None if product_key == "_brand_only" else product_key,
                    total_mentions=stats["mentions"],
                    urls_with_mentions=stats["urls"],
                )
            )

    return ScrapeSummary(total_urls=len(url_results), items=items)
