import re
from typing import List, Dict, Any, Tuple, Set
from models import Brand, ScrapeOptions, MatchSnippet, ProductMatch

def _build_term_list(brands: List[Brand]) -> List[Dict[str, Any]]:
    terms: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()  # (brand, product_or_empty, normalized_term)

    for brand in brands:
        brand_terms = [brand.name, *brand.aliases]
        for t in brand_terms:
            norm = t.strip().lower()
            if not norm:
                continue
            key = (brand.name, "", norm)
            if key in seen:
                continue
            seen.add(key)
            terms.append({"brand": brand.name, "product": None, "term": t.strip()})

        for product in brand.products:
            product_terms = [product.name, *product.aliases, *product.skus]
            for t in product_terms:
                norm = t.strip().lower()
                if not norm:
                    continue
                key = (brand.name, product.name, norm)
                if key in seen:
                    continue
                seen.add(key)
                terms.append({"brand": brand.name, "product": product.name, "term": t.strip()})

    terms.sort(key=lambda x: len(x["term"]), reverse=True)
    return terms

def _compile_term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    # If term is purely letters/spaces, use word boundaries to avoid substring matches
    if re.fullmatch(r"[A-Za-z ]+", term):
        return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
    return re.compile(escaped, flags=re.IGNORECASE)

def find_matches_in_text(
    text: str,
    brands: List[Brand],
    options: ScrapeOptions
) -> List[ProductMatch]:

    terms = _build_term_list(brands)
    all_matches: Dict[str, ProductMatch] = {}

    half_ctx = max(0, options.context_chars // 2)

    for term_info in terms:
        term = term_info["term"]
        if not term:
            continue

        pattern = _compile_term_pattern(term)
        for m in pattern.finditer(text):
            start = max(0, m.start() - half_ctx)
            end = min(len(text), m.end() + half_ctx)
            snippet_text = text[start:end]

            key = f"{term_info['brand']}||{term_info['product'] or ''}"
            pm = all_matches.get(key)
            if pm is None:
                pm = ProductMatch(
                    brand=term_info["brand"],
                    product=term_info["product"],
                    count=0,
                    matches=[]
                )
                all_matches[key] = pm

            pm.count += 1
            pm.matches.append(
                MatchSnippet(
                    brand=term_info["brand"],
                    product=term_info["product"],
                    matched_term=term,
                    text_snippet=snippet_text
                )
            )

    return list(all_matches.values())
    
