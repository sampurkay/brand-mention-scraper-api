from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.models import Brand, ScrapeOptions, MatchSnippet, ProductMatch


@dataclass(frozen=True)
class _Term:
    brand: str
    product: Optional[str]
    term: str


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _build_terms(brands: Sequence[Brand]) -> List[_Term]:
    """
    Build a flat list of match terms:
      - brand names + brand aliases => product=None
      - product names + product aliases + skus => product=<name>
    """
    terms: List[_Term] = []

    for b in brands:
        bname = _normalize(b.name)
        if bname:
            terms.append(_Term(brand=b.name, product=None, term=bname))

        for a in (b.aliases or []):
            a = _normalize(a)
            if a:
                terms.append(_Term(brand=b.name, product=None, term=a))

        for p in (b.products or []):
            pname = _normalize(p.name)
            if pname:
                terms.append(_Term(brand=b.name, product=p.name, term=pname))

            for pa in (p.aliases or []):
                pa = _normalize(pa)
                if pa:
                    terms.append(_Term(brand=b.name, product=p.name, term=pa))

            for sku in (p.skus or []):
                sku = _normalize(sku)
                if sku:
                    terms.append(_Term(brand=b.name, product=p.name, term=sku))

    # Deduplicate identical (brand, product, term)
    seen: set[Tuple[str, Optional[str], str]] = set()
    uniq: List[_Term] = []
    for t in terms:
        key = (t.brand, t.product, t.term.lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)

    # Longer terms first reduces overlaps (e.g., "Ferinject" before "Iron")
    uniq.sort(key=lambda x: len(x.term), reverse=True)
    return uniq


def _find_all_matches(text: str, term: str) -> List[Tuple[int, int]]:
    """
    Case-insensitive, word-boundary-ish matching.
    Uses \b when term is alphanumeric-ish; otherwise falls back to substring search.
    Returns list of (start, end) spans.
    """
    if not term:
        return []

    # If term is purely word characters/spaces, use boundaries around words
    if re.fullmatch(r"[\w\s\-]+", term):
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        return [(m.start(), m.end()) for m in pattern.finditer(text)]
    else:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _make_snippet(text: str, start: int, end: int, context: int) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    snippet = text[left:right].strip()
    return snippet


def find_matches_in_text(text: str, brands: List[Brand], options: ScrapeOptions) -> List[ProductMatch]:
    """
    Returns ProductMatch entries with:
      - brand
      - optional product
      - count (number of occurrences across all matched terms for that brand/product)
      - matches: list of MatchSnippet with the matched term and a surrounding snippet
    """
    src = text or ""
    if not src.strip():
        return []

    context = int(getattr(options, "context_chars", 200) or 200)

    terms = _build_terms(brands)

    # Aggregate by (brand, product)
    agg: Dict[Tuple[str, Optional[str]], Dict[str, any]] = {}

    for t in terms:
        spans = _find_all_matches(src, t.term)
        if not spans:
            continue

        key = (t.brand, t.product)
        if key not in agg:
            agg[key] = {"count": 0, "snips": []}

        for (s, e) in spans:
            agg[key]["count"] += 1
            agg[key]["snips"].append(
                MatchSnippet(
                    brand=t.brand,
                    product=t.product,
                    matched_term=t.term,
                    text_snippet=_make_snippet(src, s, e, context),
                )
            )

    # Convert to ProductMatch list
    out: List[ProductMatch] = []
    for (brand, product), data in agg.items():
        out.append(
            ProductMatch(
                brand=brand,
                product=product,
                count=int(data["count"]),
                matches=data["snips"],
            )
        )

    # Stable ordering: highest counts first
    out.sort(key=lambda pm: pm.count, reverse=True)
    return out
