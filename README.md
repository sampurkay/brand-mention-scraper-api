# brand-mention-scraper-api

Internal API service for crawling web pages and extracting brand and product mentions at scale.

The service supports polite crawling, optional JavaScript rendering via Playwright, and asynchronous job execution to avoid request timeouts. It is designed primarily for GPT-based automation workflows, but can be used as a general-purpose internal scraping and analysis service.

---

## Core Capabilities

- Seed-based web crawling with depth and page limits
- Same-domain enforcement and URL include/exclude rules
- Optional JavaScript rendering (Playwright)
- Brand- and product-level mention extraction
- Asynchronous job execution with polling
- Structured summaries optimized for LLM consumption
- Robots.txt and crawl-delay aware behavior

---

## Architecture Overview

- **FastAPI** for the API layer
- **Playwright (Chromium)** for JS-rendered pages
- **Async job model** to support long-running crawls
- **In-memory or pluggable job storage** (can be extended to Redis/S3)

The API exposes:
- A legacy synchronous endpoint for testing
- Job-based endpoints for production and GPT Actions

---

## Key Endpoints

### Submit a crawl job
