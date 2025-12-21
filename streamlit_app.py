import streamlit as st
import json, csv, os, math
from collections import defaultdict

st.set_page_config(page_title="Librarian KR — Streamlit", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
BOOKS_CSV = os.path.join(DATA_DIR, "books.csv")
TAXONOMY_JSON = os.path.join(DATA_DIR, "taxonomy.json")
BN_JSON = os.path.join(DATA_DIR, "bn.json")

# ---------- Loaders ----------
@st.cache_data
def load_taxonomy():
    with open(TAXONOMY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = data["edges"]
    disjoint_pairs = [tuple(x) for x in data.get("disjoint", [])]
    parent = defaultdict(list); child = defaultdict(list)
    for a,b in edges:
        parent[a].append(b)
        child[b].append(a)
    return parent, child, disjoint_pairs

@st.cache_data
def load_bn():
    with open(BN_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def read_books():
    rows = []
    with open(BOOKS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def append_book(book):
    fieldnames = ["title","authors","year","audience","keywords","subjects"]
    exists = os.path.exists(BOOKS_CSV) and os.path.getsize(BOOKS_CSV) > 0
    with open(BOOKS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(book)

PARENT, CHILD, DISJOINT = load_taxonomy()
BN = load_bn()

CATEGORIES = set(PARENT.keys())
for v in PARENT.values():
    for p in v: CATEGORIES.add(p)

# ---------- KR helpers ----------
def normalize_list(x):
    if not x: return []
    if isinstance(x, str):
        return [t.strip().lower() for t in x.split(",") if t.strip()]
    return []

def rule_mystery(book):
    kws = set(normalize_list(book.get("keywords")))
    subs = set(normalize_list(book.get("subjects")))
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
    kws = set(normalize_list(book.get("keywords")))
    if aud=="children" and ({"magic","dragon"} & kws or True):
        return True, "R4: Children + Fantasy signals → ChildrensFantasy", ["ChildrensFantasy"]
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
        import math
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
        explanations.append("BN: chose %s (p=%.2f) using features: %s" % (top1, p1, ", ".join([k for k,v in feats.items() if v]) or "∅"))
        if len(ranked)>1 and ranked[1][1] > 0.35:
            cats.append(ranked[1][0])
    closure, conflicts = fol_closure(cats)
    return {"categories":cats, "closure":closure, "conflicts":conflicts, "explanations":explanations, "bn_used":bn_used, "bn_top":bn_top}

# ---------- UI ----------
st.title("📚 Librarian KR Chatbot — Streamlit")

with st.sidebar:
    st.header("➕ Add / Classify Book")
    title = st.text_input("Title", "")
    authors = st.text_input("Authors", "")
    year = st.number_input("Year", min_value=1451, max_value=2100, value=2020, step=1)
    audience = st.selectbox("Audience", ["Adult","YoungAdult","Children"], index=0)
    keywords = st.text_input("Keywords (comma)", "")
    subjects = st.text_input("Subjects (comma)", "")
    colb1, colb2 = st.columns(2)
    do_classify = colb1.button("Classify")
    do_save = colb2.button("Save to catalog")

book_dict = {"title":title, "authors":authors, "year":str(year), "audience":audience, "keywords":keywords, "subjects":subjects}

if do_classify:
    res = classify_book(book_dict)
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
    append_book(book_dict)
    st.success("Saved to data/books.csv")

st.markdown("---")
st.header("🔎 Search Catalog")
q = st.text_input("Search text", "")
cat = st.selectbox("Filter by Category (classification on the fly)", ["Any"] + sorted(list(CATEGORIES)), index=0)

rows = read_books()
filtered=[]
for r in rows:
    hay = " ".join([r.get("title",""), r.get("authors",""), r.get("audience",""), r.get("keywords",""), r.get("subjects","")]).lower()
    if q and q.lower() not in hay:
        continue
    if cat != "Any":
        cres = classify_book(r)
        if cat not in cres["closure"]:
            continue
    filtered.append(r)

st.dataframe(filtered, use_container_width=True, hide_index=True)
st.caption("Tip: the category filter classifies each row with the same KR pipeline.")

st.markdown("---")
with st.expander("ℹ️ Project Notes"):
    st.write("- **Taxonomy** governs inheritance; **FOL** adds ancestors and checks disjointness.")
    st.write("- **Production rules**: fast/explainable classification.")
    st.write("- **Naïve Bayes** backs off when rules don't fire (uncertainty).")
    st.write("- Edit JSON/CSV in `data/` to change taxonomy/BN/seed books.")
