# --- Natural-language helpers ---
CATEGORY_ALIASES = {
    "sci fi": "ScienceFiction", "sci-fi": "ScienceFiction", "scifi": "ScienceFiction",
    "science fiction": "ScienceFiction",
    "fantasy": "Fantasy", "children's fantasy": "ChildrensFantasy", "childrens fantasy": "ChildrensFantasy",
    "mystery": "Mystery", "crime fiction": "CrimeFiction",
    "history": "History", "science": "Science",
}
AUD_ALIASES = {
    "kids": "Children", "kid": "Children", "children": "Children", "children's": "Children",
    "ya": "YoungAdult", "young adult": "YoungAdult",
    "adult": "Adult"
}
STOPWORDS = {
    "a","an","the","book","books","list","show","find","every","all","any","of","with","about",
    "me","please","you","for","after","before","since","between","and","in","on","from","to"
}

import re

def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())

def _find_category(text: str):
    t = _clean(text)
    # alias hits
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in t:
            return cat
    # direct match to taxonomy names
    for c in sorted(CATEGORIES, key=len, reverse=True):
        if _clean(c).replace(" ", "") in t.replace(" ", ""):
            return c
    return None

def _parse_years(text: str):
    t = _clean(text)
    y_min = y_max = None
    m = re.search(r"after\s+(\d{4})", t)
    if m: y_min = int(m.group(1)) + 0
    m = re.search(r"since\s+(\d{4})", t)
    if m: y_min = int(m.group(1))
    m = re.search(r"before\s+(\d{4})", t)
    if m: y_max = int(m.group(1)) - 0
    m = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", t)
    if m: y_min, y_max = int(m.group(1)), int(m.group(2))
    m = re.search(r"in\s+(\d{4})", t)
    if m and not (y_min or y_max):
        y_min = y_max = int(m.group(1))
    return y_min, y_max

def _extract_terms(text: str):
    toks = [w for w in _clean(text).split() if len(w) > 3 and w not in STOPWORDS]
    return toks
