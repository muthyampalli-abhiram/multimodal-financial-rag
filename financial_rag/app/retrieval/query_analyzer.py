"""Query intent analysis and heuristic classification for financial RAG."""

import re
from typing import Any, Dict, List

# Core financial keywords and metric terminology
FINANCIAL_KEYWORDS = {
    "revenue",
    "margin",
    "profit",
    "earnings",
    "growth",
    "segment",
    "geographic",
    "shareholder",
    "dividend",
    "capital",
    "tsr",
    "income",
    "ebitda",
    "cash flow",
    "operating",
    "expense",
    "cost",
    "sales",
    "guidance",
    "debt",
    "equity",
    "asset",
    "liability",
    "valuation",
    "capex",
    "balance sheet",
    "tax",
    "return on",
    "roi",
    "roe",
    "eps",
    "diluted",
    "financial",
    "liquidity",
    "free cash flow",
}

# Comparison and multi-period trend indicators
COMPARISON_KEYWORDS = {
    "compare",
    "comparison",
    "vs",
    "versus",
    "trend",
    "trends",
    "over time",
    "year over year",
    "year-over-year",
    "yoy",
    "quarter over quarter",
    "quarter-over-quarter",
    "qoq",
    "change",
    "changes",
    "difference",
    "differ",
    "historical",
    "historically",
    "evolution",
    "growth rate",
}

# Regex pattern for calendar years (2020-2039, 1990-1999) and fiscal year designations (FY23, FY 2024)
YEAR_REGEX = re.compile(r"\b(?:FY\s*\d{2,4}|20\d{2}|19\d{2})\b", re.IGNORECASE)


def detect_query_intent(query: str) -> Dict[str, Any]:
    """Detect domain intent, comparison requirements, and mentioned years from user query.

    Uses fast heuristic matching without incurring LLM latency or cost.

    Args:
        query: User or analyst query string.

    Returns:
        Dict[str, Any]:
            - wants_financial_data (bool): True if financial concepts or metrics are detected.
            - wants_comparison (bool): True if comparative phrasing or multiple time periods detected.
            - mentioned_years (List[str]): Extracted calendar or fiscal years.
    """
    if not query or not query.strip():
        return {
            "wants_financial_data": False,
            "wants_comparison": False,
            "mentioned_years": [],
        }

    query_clean = query.strip()
    query_lower = query_clean.lower()

    # 1. Extract mentioned years / fiscal periods
    raw_years = YEAR_REGEX.findall(query_clean)
    mentioned_years: List[str] = []
    seen_normalized = set()

    for yr in raw_years:
        normalized = re.sub(r"\s+", "", yr.upper())
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            mentioned_years.append(yr.strip())

    # 2. Check for financial keywords (regex boundary for acronyms like 'tsr', substring for words)
    wants_financial_data = False
    for kw in FINANCIAL_KEYWORDS:
        if kw in ("tsr", "eps", "roi", "roe", "ebitda"):
            if re.search(rf"\b{kw}\b", query_lower):
                wants_financial_data = True
                break
        else:
            if kw in query_lower:
                wants_financial_data = True
                break

    # 3. Check for comparison intent
    wants_comparison = False
    if len(mentioned_years) >= 2:
        wants_comparison = True
    else:
        for ck in COMPARISON_KEYWORDS:
            if ck in ("vs", "yoy", "qoq"):
                if re.search(rf"\b{ck}\b", query_lower):
                    wants_comparison = True
                    break
            else:
                if ck in query_lower:
                    wants_comparison = True
                    break

    return {
        "wants_financial_data": wants_financial_data,
        "wants_comparison": wants_comparison,
        "mentioned_years": mentioned_years,
    }
