import streamlit as st
import json, csv, math, re
from collections import defaultdict
from pathlib import Path

st.set_page_config(page_title="Librarian KR — Chat", layout="wide")

# ========= Paths & Bootstrap (creates data/ if missing) =========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TAXONOMY_JSON = DATA_DIR / "taxonomy.json"
BN_JSON = DATA_DIR / "bn.json"
BOOKS_CSV = DATA_DIR / "books.csv"

def bootstrap_data():
    DATA_DIR.mkdir(exist_ok=True)

    if not TAXONOMY_JSON.exists():
        TAXONOMY_JSON.write_text(json.dumps({
            "edges": [
                ["Mystery", "CrimeFiction"],
                ["CrimeFiction", "Fiction"],
                ["ScienceFiction", "Fiction"],
                ["Fantasy", "Fiction"],
                ["ChildrensFantasy", "Fantasy"],
                ["History", "NonFiction"],
                ["Science", "NonFiction"]
            ],
            "disjoint": [["Fiction", "NonFiction"]]
        }, indent=2), encoding="utf-8")

    if not BN_JSON.exists():
        BN_JSON.write_text(json.dumps({
            "categories": ["Mystery","ScienceFiction","Fantasy","ChildrensFantasy","History","Science"],
            "priors": {"Mystery":0.17,"ScienceFiction":0.17,"Fantasy":0.17,"ChildrensFantasy":0.16,"History":0.16,"Science":0.17},
            "features": ["HasMurder","HasSpaceship","HasMagic","SubjectCrime","SubjectSpace","SubjectHistory","AudienceChildren"],
            "cpt": {
                "HasMurder":{"Mystery":0.85,"ScienceFiction":0.05,"Fantasy":0.05,"ChildrensFantasy":0.01,"History":0.01,"Science":0.01},
                "HasSpaceship":{"Mystery":0.02,"ScienceFiction":0.85,"Fantasy":0.05,"ChildrensFantasy":0.05,"History":0.01,"Science":0.02},
                "HasMagic":{"Mystery":0.02,"ScienceFiction":0.02,"Fantasy":0.80,"ChildrensFantasy":0.70,"History":0.01,"Science":0.01},
                "SubjectCrime":{"Mystery":0.80,"ScienceFiction":0.05,"Fantasy":0.02,"ChildrensFantasy":0.01,"History":0.01,"Science":0.01},
                "SubjectSpace":{"Mystery":0.01,"ScienceFiction":0.80,"Fantasy":0.02,"ChildrensFantasy":0.05,"History":0.01,"Science":0.01},
                "SubjectHistory":{"Mystery":0.01,"ScienceFiction":0.01,"Fantasy":0.01,"ChildrensFantasy":0.01,"History":0.85,"Science":0.10},
                "AudienceChildren":{"Mystery":0.05,"ScienceFiction":0.10,"Fantasy":0.20,"ChildrensFantasy":0.85,"History":0.10,"Science":0.10}
            }
        }, indent=2), encoding="utf-8")

    if not BOOKS_CSV.exists():
        with BOOKS_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title","authors","year","audience","keywords","subjects"])
            w.writerow(["Spaceship Hearts","A. Cole",2018,"YoungAdult","spaceship, planet, romance","Space, Adventure"])
            w.writerow(["Moon Dagger","L. Noor",2012,"Adult","magic, quest, prophecy","Myth, Fantasy"])
            w.writerow(["Detective Nile","M. Azmi",2021,"Adult","detective, murder, clue","Crime, Urban"])
            w.writerow(["Starlight Kids","R. Baker",2019,"Children","spaceship, friendship","Space, Fantasy"])
            w.writerow(["Ancient Empires","S. Rao",2015,"Adult","kingdoms, archival","History"])
            w.writerow(["Everyday Physics","T. Ahmed",2020,"Adult","energy, forces","Science, Education"])

bootstrap_data()

# ========= Loaders =========
@st.cache_data
def load_taxonomy():
    data = json.loads(TAXONOMY_JSON.read_text(encoding="utf-8"))
    edges = data["edges"]
    disjoint_pairs = [tuple(x) for x in data.get("disjoint", [])]
    parent = defaultdict(list); child = defaultdict(list)
    for a,b in edges:
        parent[a].append(b)
        child[b].append(a)
    return parent, child, disjoint_pairs

@st.cache_data
def load_bn():
    return json.loads(BN_JSON.read_text(encoding="utf-8"))

def read_books():
    rows = []
    if BOOKS_CSV.exists():
        with BOOKS_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows

def append_book(book):
    fieldnames = ["title","authors","year","audience","keywords","subjects"]
    exists = BOOKS_CSV.exists() and BOOKS_CSV.stat().st_size > 0
    with BOOKS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists: writer.writeheader()
        writer.writerow(book)

PARENT, CHILD, DISJOINT = load_taxonomy()
BN = load_bn()
CATEGORIES = set(PARENT.keys()) | {p for vals in PARENT.values() for p in vals}

# ========= KR helpers =========
def normalize_list(x):
    if not x: return []
    if isinstance(x, str):
        return [t.strip().lower() for t in x.split(",") if t.strip()]
    return []

def rule_mystery(book):
    kws = set(normalize_list(book.get("keywords"))); subs = set(normalize_list(book.get("subjects")))
    if {"murder","detective"} & kws or "crime" in subs:
        return True, "R1: murder/detective/Crime → Mystery", ["Mystery"]
    return False, "", []

def rule_scifi(book):
    kws = set(normalize_list(book.get("keywords"))); subs = set(normalize_list(book.get("subjects")))
    if {"spaceship","planet","space"} & kws or "space" in subs:
        return True, "R2: spaceship/planet/space → ScienceFiction", ["ScienceFiction"]
    return False, "", []

def rule_fantasy(book):
    if {"magic","dragon","quest"} & set(normalize_list(book.get("keywords"))):
        return True, "R3: magic/dragon/quest → Fantasy", ["Fantasy"]
    return False, "", []

def rule_childrens_fantasy(book):
    aud = book.get("audience","").lower()
    kws = set(normalize_list(book.get("keywords"))); subs = set(normalize_list(book.get("subjects")))
    if aud == "children" and ({"magic","dragon","quest"} & kws or "fantasy" in subs):
        return True, "R4: Children + fantasy signals → ChildrensFantasy", ["ChildrensFantasy"]
    return False, "", []

def rule_history(book):
    if "history" in set(normalize_list(book.get("subjects"))):
        return True, "R5: subject History → History", ["History"]
    return False, "", []

def rule_science(book):
    subs = set(normalize_list(book.get("subjects"))); kws=set(normalize_list(book.get("keywords")))
    if "science" in subs or {"physics","biology","chemistry"} & kws:
        return True, "R6: subject Science / physics/biology/chemistry → Science", ["Science"]
    return False, "", []

RULES = [rule_mystery, rule_scifi, rule_fantasy, rule_childrens_fantasy, rule_history, rule_science]

def ancestors(cat):
    seen=set(); stack=[cat]
    while stack:
        c=stack.pop()
        for p in PARENT.get(c, []):
            if p not in seen: seen.add(p); stack.append(p)
    return seen

def fol_closure(categories):
    full=set(categories)
    for c in list(categories): full |= ancestors(c)
    conflicts=[]
    for a,b in DISJOINT:
        if a in full and b in full: conflicts.append((a,b))
    return sorted(full), conflicts

def extract_features(book):
    kws=set(normalize_list(book.get("keywords"))); subs=set(normalize_list(book.get("subjects"))); aud=book.get("audience","").lower()
    return {
        "HasMurder": ("murder" in kws or "detective" in kws),
        "HasSpaceship": ({"spaceship","planet","space"} & kws != set() or "space" in subs),
        "HasMagic": ({"magic","dragon","quest"} & kws != set()),
        "SubjectCrime": ("crime" in subs),
        "SubjectSpace": ("space" in subs),
        "SubjectHistory": ("history" in subs),
        "AudienceChildren": (aud=="children"),
    }

def naive_bayes_category(book):
    feats = extract_features(book)
    cats = BN["categories"]; priors = BN["priors"]; cpt=BN["cpt"]
    scores={}
    for cat in cats:
        prior = max(1e-6, float(priors.get(cat, 1/len(cats))))
        logp = math.log(prior)
        for feat in BN["features"]:
            p_true = max(1e-6, min(1-1e-6, float(cpt[feat][cat])))
            p = p_true if feats[feat] else (1 - p_true)
            logp += math.log(p)
        scores[cat]=logp
    maxlog=max(scores.values()); exps={k:math.exp(v-maxlog) for k,v in scores.items()}; Z=sum(exps.values())
    probs={k:v/Z for k,v in exps.items()}
    ranked=sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return ranked, feats

def classify_book(book):
    explanations=[]; cats=[]
    for rule in RULES:
        ok, exp, newcats = rule(book)
        if ok:
            explanations.append(exp); cats.extend(newcats)
    cats=list(dict.fromkeys(cats))
    bn_used=False; bn_top=[]
    if not cats:
        bn_used=True
        ranked, feats = naive_bayes_category(book)
        top1, p1 = ranked[0]
        cats.append(top1); bn_top=ranked[:3]
        explanations.append("BN: chose %s (p=%.2f) using features: %s" %
                            (top1, p1, ", ".join([k for k,v in feats.items() if v]) or "∅"))
        if len(ranked)>1 and ranked[1][1] > 0.35:
            cats.append(ranked[1][0])
    closure, conflicts = fol_closure(cats)
    return {"categories":cats, "closure":closure, "conflicts":conflicts, "explanations":explanations, "bn_used":bn_used, "bn_top":bn_top}

# ========= Natural-language helpers =========
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

def _clean(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())

def _find_category(text: str):
    t = _clean(text)
    for alias, cat in CATEGORY_ALIASES.items():
        if alias in t:
            return cat
    for c in sorted(CATEGORIES, key=len, reverse=True):
        if _clean(c).replace(" ", "") in t.replace(" ", ""):
            return c
    return None

def _parse_years(text: str):
    t = _clean(text)
    y_min = y_max = None
    m = re.search(r"after\s+(\d{4})", t)
    if m: y_min = int(m.group(1)) + 1        # after 2015 -> >=2016
    m = re.search(r"since\s+(\d{4})", t)
    if m: y_min = int(m.group(1))            # since 2015 -> >=2015
    m = re.search(r"before\s+(\d{4})", t)
    if m: y_max = int(m.group(1)) - 1        # before 2020 -> <=2019
    m = re.search(r"between\s+(\d{4})\s+and\s+(\d{4})", t)
    if m: y_min, y_max = int(m.group(1)), int(m.group(2))
    m = re.search(r"\bin\s+(\d{4})", t)
    if m and not (y_min or y_max):
        y_min = y_max = int(m.group(1))
    return y_min, y_max

def _extract_terms(text: str):
    toks = [w for w in _clean(text).split() if len(w) > 3 and w not in STOPWORDS]
    return toks

# ========= Chat helpers =========
def parse_kv(text):
    kv = {}
    for m in re.finditer(r'(\b[\w\-]+\b)\s*[:=]\s*([^;\n,]+)', text, re.IGNORECASE):
        kv[m.group(1).strip().lower()] = m.group(2).strip()
    return kv

def intent_and_slots(msg):
    t = msg.strip()
    low = t.lower()
    kv = parse_kv(t)

    # explicit intents still supported
    if re.match(r'^\s*(classify|add|why|search)\b', low):
        i = low.split()[0]
        return i, kv

    # NL intents (list/show/find…)
    if re.search(r'\b(list|show|find|give me|lookup|look up|search)\b', low):
        intent = "search"
        cat = _find_category(low)
        y_min, y_max = _parse_years(low)
        # audience
        audience = None
        for k, v in AUD_ALIASES.items():
            if k in low:
                audience = v
                break
        terms = _extract_terms(low)
        if cat:
            for w in _clean(cat).split():
                terms = [t for t in terms if t != w]
        return intent, {
            "terms": terms,
            "text": " ".join(terms),
            "category": cat or "Any",
            "audience": audience or "",
            "year_min": y_min,
            "year_max": y_max,
        }

    # fallback: free text search
    return "search", {"terms": _extract_terms(low), "text": " ".join(_extract_terms(low))}

def chat_handle(message):
    intent, kv = intent_and_slots(message)

    if intent in ("classify","why"):
        book = {
            "title": kv.get("title",""),
            "authors": kv.get("authors",""),
            "year": kv.get("year",""),
            "audience": kv.get("audience","Adult"),
            "keywords": kv.get("keywords",""),
            "subjects": kv.get("subjects",""),
        }
        if not (book["title"] or book["keywords"] or book["subjects"]):
            return ("Please provide at least `title`, `keywords`, or `subjects`.\n"
                    "Example: `classify title=Moon Dagger; audience=Adult; keywords=magic, quest`")
        res = classify_book(book)
        lines = [f"**Intent:** {intent}",
                 f"**Categories (leaves):** {', '.join(res['categories']) or '—'}",
                 f"**Closure:** {', '.join(res['closure']) or '—'}"]
        if res["conflicts"]:
            lines.append("**Conflicts:** " + ", ".join([f"{a} vs {b}" for a,b in res["conflicts"]]))
        if res["bn_used"]:
            lines.append("**BN posterior (top):** " + ", ".join([f"{c}:{p:.2f}" for c,p in res["bn_top"]]))
        if res["explanations"]:
            lines.append("**Why:**"); lines += [f"- {e}" for e in res["explanations"]]
        return "\n".join(lines)

    if intent == "add":
        book = {
            "title": kv.get("title",""),
            "authors": kv.get("authors",""),
            "year": kv.get("year",""),
            "audience": kv.get("audience","Adult"),
            "keywords": kv.get("keywords",""),
            "subjects": kv.get("subjects",""),
        }
        if not book["title"]:
            return ("To add, include at least `title=`.\n"
                    "Example: `add title=Detective Nile; authors=M. Azmi; year=2021; audience=Adult; "
                    "keywords=detective, murder; subjects=Crime`")
        append_book(book)
        return f"✅ Added **{book['title']}** to catalog."

    # ------- SEARCH (supports NL slots) -------
    terms   = kv.get("terms") or []
    text    = kv.get("text") or ""
    cat     = kv.get("category") or kv.get("cat") or "Any"
    audience= kv.get("audience","")
    y_min   = kv.get("year_min")
    y_max   = kv.get("year_max")

    rows = read_books()
    out = []
    for r in rows:
        hay = " ".join([r.get("title",""), r.get("authors",""), r.get("audience",""),
                        r.get("keywords",""), r.get("subjects","")]).lower()

        # require ANY term to appear when terms provided
        if terms and not any(t.lower() in hay for t in terms):
            continue

        # category via KR classification
        if cat and cat != "Any":
            cres = classify_book(r)
            if cat not in cres["closure"]:
                continue

        # audience filter
        if audience and r.get("audience","") != audience:
            continue

        # year filters
        try:
            yr = int(r.get("year", 0))
        except Exception:
            yr = 0
        if y_min is not None and yr < int(y_min): continue
        if y_max is not None and yr > int(y_max): continue

        out.append(r)

    if not out:
        return "No matches. Try: `list children fantasy after 2015` or `find YA sci-fi space`"

    md = ["| Title | Authors | Year | Audience | Keywords | Subjects |", "|-|-|-|-|-|-|"]
    for r in out[:20]:
        md.append(f"| {r.get('title','')} | {r.get('authors','')} | {r.get('year','')} | "
                  f"{r.get('audience','')} | {r.get('keywords','')} | {r.get('subjects','')} |")
    if len(out) > 20:
        md.append(f"\n_Showing 20 of {len(out)} results…_")
    return "\n".join(md)

# ========= UI (Two Tabs: Chat + Catalog) =========
tab_chat, tab_catalog = st.tabs(["💬 Chat", "📚 Catalog UI"])

with tab_chat:
    st.markdown("**Talk to your librarian.** Examples:")
    st.code("""list every Adventure book
find children fantasy after 2015
show YA sci-fi space books
classify title=Moon Dagger; audience=Adult; keywords=magic, quest
why title=Detective Nile; keywords=detective, murder""")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant","content":"Hi! I can classify, search, explain (why), and add books. Try: `list children fantasy after 2015`"}
        ]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type a message…")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        reply = chat_handle(prompt)
        st.session_state.messages.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"): st.markdown(reply)

    if st.button("Reset chat"):
        st.session_state.messages = []
        st.rerun()

with tab_catalog:
    st.header("Add / Classify (Form)")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", "")
        authors = st.text_input("Authors", "")
        year = st.number_input("Year", min_value=1451, max_value=2100, value=2020, step=1)
        audience = st.selectbox("Audience", ["Adult","YoungAdult","Children"], index=0)
    with col2:
        keywords = st.text_input("Keywords (comma)", "")
        subjects = st.text_input("Subjects (comma)", "")
        do_classify = st.button("Classify")
        do_save = st.button("Save to catalog")

    book = {"title":title, "authors":authors, "year":str(year), "audience":audience, "keywords":keywords, "subjects":subjects}

    if do_classify:
        res = classify_book(book)
        st.subheader("Result")
        st.write("**Categories (leaves):** ", ", ".join(res["categories"]) or "—")
        st.write("**Closure (+ancestors):** ", ", ".join(res["closure"]) or "—")
        if res["conflicts"]:
            st.error("Conflicts: " + ", ".join([f"{a} vs {b}" for a,b in res["conflicts"]]))
        if res["bn_used"]:
            st.caption("BN posterior (top):")
            st.table({c: f"{p:.2f}" for c,p in res["bn_top"]})
        if res["explanations"]:
            with st.expander("Why? (rules/BN features)"):
                for e in res["explanations"]:
                    st.write("• " + e)

    if do_save:
        append_book(book)
        st.success("Saved to data/books.csv")

    st.markdown("---")
    st.header("Search Catalog")
    q = st.text_input("Search text", "")
    cat = st.selectbox("Filter by Category", ["Any"] + sorted(list(CATEGORIES)), index=0)

    rows = read_books()
    filtered=[]
    for r in rows:
        hay = " ".join([r.get("title",""), r.get("authors",""), r.get("audience",""),
                        r.get("keywords",""), r.get("subjects","")]).lower()
        if q and q.lower() not in hay: continue
        if cat != "Any":
            cres = classify_book(r)
            if cat not in cres["closure"]: continue
        filtered.append(r)

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption("Tip: the category filter classifies each row with the same KR pipeline.")
