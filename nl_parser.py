"""
nl_parser.py — Minimal NL-to-slots parser for the librarian.
Extracts: intent, title (quoted or 'called/titled' or after 'classify'),
authors ('by ...'), category aliases, audience, year ranges, negative subject,
sort, limit, and k=v fallbacks.
"""
import re
from typing import Dict, Optional

CATEGORY_ALIASES = {
    "sci fi": "ScienceFiction", "sci-fi": "ScienceFiction", "scifi": "ScienceFiction",
    "science fiction": "ScienceFiction",
    "fantasy": "Fantasy", "children's fantasy": "ChildrensFantasy", "childrens fantasy": "ChildrensFantasy",
    "mystery": "Mystery", "crime fiction": "CrimeFiction",
    "history": "History", "science": "Science",
}
AUD_ALIASES = {
    "kids": "Children", "kid": "Children", "children": "Children", "children's": "Children",
    "ya": "YoungAdult", "young adult": "YoungAdult", "adult": "Adult"
}
SUBJECT_SYNONYMS = {
    "adventure": "Adventure", "space": "Space", "crime": "Crime", "urban": "Urban",
    "myth": "Myth", "education": "Education", "romance": "Romance",
    "friendship": "Friendship", "dragon": "Dragon", "quest": "Quest", "magic": "Magic",
    "physics": "Physics", "history": "History", "science": "Science"
}
STOPWORDS = {
    "a","an","the","book","books","list","show","find","every","all","any","of","with","about",
    "me","please","you","for","after","before","since","between","and","in","on","from","to",
    "called","titled","title","showing","give","tell","tell me","explain","why","is","it","movie","novel"
}

def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())

def parse_kv(text: str) -> Dict:
    kv = {}
    for m in re.finditer(r'(\b[\w\-]+\b)\s*[:=]\s*([^;\n,]+)', text, re.IGNORECASE):
        kv[m.group(1).strip().lower()] = m.group(2).strip()
    return kv

def extract_title(text: str) -> Optional[str]:
    m = re.search(r"'([^']+)'", text) or re.search(r"\"([^\"]+)\"", text)
    if m: return m.group(1).strip()
    m = re.search(r"\b(?:titled|called)\s+([A-Za-z0-9][^,.;]+)", text, flags=re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def extract_author(text: str) -> Optional[str]:
    m = re.search(r"\bby\s+([A-Za-z][A-Za-z .'-]+)", text, flags=re.IGNORECASE)
    if m: return m.group(1).strip()
    return None

def find_category(text: str) -> Optional[str]:
    t = _clean(text)
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in t:
            return cat
    for c in ["ScienceFiction","Fantasy","ChildrensFantasy","Mystery","CrimeFiction","History","Science"]:
        if _clean(c).replace(" ", "") in t.replace(" ", ""):
            return c
    return None

def parse_years(text: str):
    t = _clean(text)
    y_min = y_max = None
    m = re.search(r"after\s+(\d{4})", t)
    if m: y_min = int(m.group(1)) + 1
    m = re.search(r"since\s+(\d{4})", t)
    if m: y_min = int(m.group(1))
    m = re.search(r"before\s+(\d{4})", t)
    if m: y_max = int(m.group(1)) - 1
    m = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", t)
    if m: y_min, y_max = int(m.group(1)), int(m.group(2))
    m = re.search(r"\bin\s+(\d{4})", t)
    if m and not (y_min or y_max):
        y_min = y_max = int(m.group(1))
    return y_min, y_max

def extract_terms(text: str):
    toks = [w for w in _clean(text).split() if len(w) > 2 and w not in STOPWORDS]
    return toks

def parse_message(msg: str) -> Dict:
    """Return {'intent': ..., 'slots': {...}}"""
    t = msg.strip()
    low = t.lower()
    kv = parse_kv(t)

    # SPECIAL: "classify <free title or 'quoted'>"
    m = re.match(r'^\s*classify\s+(.+)$', t, flags=re.IGNORECASE)
    if m:
        remainder = m.group(1).strip()
        # keep existing k=v if present, but try to infer title if not given
        if "title" not in kv or not kv["title"]:
            t_guess = extract_title(remainder)
            if t_guess:
                kv["title"] = t_guess
            else:
                # fallback: treat remainder as a raw title phrase
                kv["title"] = remainder
        return {"intent": "classify", "slots": kv}

    # explicit intents (k=v style)
    if re.match(r'^\s*(add|why|search)\b', low):
        i = low.split()[0]
        title = extract_title(t)
        if title: kv["title"] = title
        return {"intent": i, "slots": kv}

    # about intent
    if re.search(r'\btell me about\b', low) or extract_title(t):
        kv["title"] = kv.get("title") or extract_title(t)
        auth = extract_author(t)
        if auth: kv["authors"] = auth
        return {"intent": "about", "slots": kv}

    # why intent
    if re.search(r'\b(explain|why)\b', low):
        kv["title"] = kv.get("title") or extract_title(t)
        return {"intent": "why", "slots": kv}

    # NL search
    if re.search(r'\b(list|show|find|give me|lookup|look up|search)\b', low):
        cat = find_category(low)
        y_min, y_max = parse_years(low)
        audience = None
        for k, v in AUD_ALIASES.items():
            if re.search(rf"\b{k}\b", low):
                audience = v; break
        terms = extract_terms(low)
        if cat:
            for w in _clean(cat).split():
                terms = [tt for tt in terms if tt != w]
        order = "none"
        if "newest" in low or "latest" in low: order = "newest"
        if "oldest" in low: order = "oldest"
        m = re.search(r'\btop\s+(\d+)\b', low)
        limit = int(m.group(1)) if m else None
        neg_subj = None
        for s in SUBJECT_SYNONYMS:
            if re.search(rf'\b(not|no)\s+{re.escape(s)}\b', low):
                neg_subj = SUBJECT_SYNONYMS[s]; break
        return {"intent":"search","slots":{
            "terms": terms,
            "text": " ".join(terms),
            "category": cat or "Any",
            "audience": audience or "",
            "year_min": y_min,
            "year_max": y_max,
            "order": order,
            "limit": limit,
            "neg_subject": neg_subj,
        }}

    # fallback: free text search
    return {"intent":"search","slots":{"terms": extract_terms(low), "text": " ".join(extract_terms(low))}}
