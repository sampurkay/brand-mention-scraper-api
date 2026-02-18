# brand-mention-scraper-api

Internal API service for crawling web pages and extracting brand and product mentions.

Supports polite crawling, optional JavaScript rendering via Playwright, relevance-aware link following (prioritize/filter), and asynchronous job execution to avoid request timeouts. Designed for GPT Actions / internal automation.

---

## Capabilities

- Seed-based crawl with depth + page caps
- Same-domain enforcement + include/exclude URL patterns
- Relevance-aware link following (e.g., prioritize `/treatment`, `/safety`, `/hcp`, `/patient`, `/resources`)
- Playwright JS rendering with:
  - `page.goto()` navigation
  - `wait_for_selector()` (optional)
  - resource blocking (images/fonts/media; optional stylesheets)
  - JS evaluation-based extraction (`document.body.innerText`)
  - optional infinite scroll + heuristic pagination
- Exponential backoff retries
- Async job API for long-running crawls
- Structured summaries optimized for LLM consumption

---

## API Endpoints

### Health
- `GET /` → `{ "status": "ok", ... }`

### Job-based (recommended)
- `POST /api/jobs` → returns `job_id`
- `GET /api/jobs/{job_id}` → status/progress
- `GET /api/jobs/{job_id}/results?offset=0&limit=100` → URLs + status
- `POST /api/jobs/{job_id}/summary` → mention summary (small, LLM-friendly)
- `POST /api/jobs/{job_id}/cancel` → cancel

### Single-shot (manual testing; may time out)
- `POST /api/scrape-product-mentions`

---

## Example: Submit a job

```bash
curl -X POST "$BASE_URL/api/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://www.hemequity.com/"],
    "brands": [{"name": "Ferinject", "aliases": [], "products": []}],
    "crawl_depth": 2,
    "max_pages": 80,
    "same_domain_only": true,
    "exclude_url_patterns": [
      "(/login|/signin|/account|/register)",
      "(/cart|/checkout)",
      "(/search|\\?s=|\\?q=|\\?query=)",
      "(\\.(pdf|jpg|jpeg|png|gif|svg|zip))$"
    ],
    "render_js": true,
    "wait_until": "networkidle",
    "wait_ms": 500,
    "link_relevance_mode": "prioritize",
    "link_priority_keywords": ["treatment","safety","hcp","patient","resources","faq","dosing","indication","prescribing"],
    "max_concurrency": 2,
    "respect_robots": true
  }'


---

## Key Endpoints

### Submit a crawl job
