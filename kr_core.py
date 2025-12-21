"""
kr_core.py — Knowledge Representation core for a librarian chatbot.
Includes: Taxonomy/Frames, FOL closure, Production Rules, Naive Bayes (BN), and search utilities.
No Streamlit imports here; pure Python so you can present it as your KR code.
"""
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json, csv, math, re

# ---------- Paths & bootstrap ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TAXONOMY_JSON = DATA_DIR / "taxonomy.json"
BN_JSON = DATA_DIR / "bn.json"
BOOKS_CSV = DATA_DIR / "books.csv"

def ensure_data() -> None:
    """Create minimal demo data if missing (safe to call from any runtime)."""
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

# ---------- Data access ----------
def load_taxonomy():
    """Frames/Taxonomy graph + disjointness (FOL constraints)."""
    data = json.loads(TAXONOMY_JSON.read_text(encoding="utf-8"))
    edges = data["edges"]
    disjoint_pairs = [tuple(x) for x in data.get("disjoint", [])]
    parent = defaultdict(list); child = defaultdict(list)
    for a,b in edges:
        parent[a].append(b)
        child[b].append(a)
    return parent, child, disjoint_pairs

def load_bn():
    """Bayesian/Naive Bayes parameters."""
    return json.loads(BN_JSON.read_text(encoding="utf-8"))

def read_books() -> List[Dict]:
    rows: List[Dict] = []
    if BOOKS_CSV.exists():
        with BOOKS_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows

def append_book(book: Dict) -> None:
    fieldnames = ["title","authors","year","audience","keywords","subjects"]
    exists = BOOKS_CSV.exists() and BOOKS_CSV.stat().st_size > 0
    with BOOKS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(book)

# ---------- Production rules ----------
def _normalize_list(x):
    if not x: return []
    if isinstance(x, str):
        return [t.strip().lower() for t in x.split(",") if t.strip()]
    return []

def rule_mystery(book):
    kws = set(_normalize_list(book.get("keywords"))); subs = set(_normalize_list(book.get("subjects")))
    if {"murder","detective"} & kws or "crime" in subs:
        return True, "R1: murder/detective/Crime → Mystery", ["Mystery"]
    return False, "", []

def rule_scifi(book):
    kws = set(_normalize_list(book.get("keywords"))); subs = set(_normalize_list(book.get("subjects")))
    if {"spaceship","planet","space"} & kws or "space" in subs:
        return True, "R2: spaceship/planet/space → ScienceFiction", ["ScienceFiction"]
    return False, "", []

def rule_fantasy(book):
    if {"magic","dragon","quest"} & set(_normalize_list(book.get("keywords"))):
        return True, "R3: magic/dragon/quest → Fantasy", ["Fantasy"]
    return False, "", []

def rule_childrens_fantasy(book):
    aud = (book.get("audience") or "").lower()
    kws = set(_normalize_list(book.get("keywords"))); subs = set(_normalize_list(book.get("subjects")))
    if aud == "children" and ({"magic","dragon","quest"} & kws or "fantasy" in subs):
        return True, "R4: Children + fantasy signals → ChildrensFantasy", ["ChildrensFantasy"]
    return False, "", []

def rule_history(book):
    if "history" in set(_normalize_list(book.get("subjects"))):
        return True, "R5: subject History → History", ["History"]
    return False, "", []

def rule_science(book):
    subs = set(_normalize_list(book.get("subjects"))); kws=set(_normalize_list(book.get("keywords")))
    if "science" in subs or {"physics","biology","chemistry"} & kws:
        return True, "R6: subject Science / physics/biology/chemistry → Science", ["Science"]
    return False, "", []

RULES = [rule_mystery, rule_scifi, rule_fantasy, rule_childrens_fantasy, rule_history, rule_science]

# ---------- FOL closure on taxonomy ----------
def ancestors(parent_graph: Dict[str,List[str]], cat: str):
    seen=set(); stack=[cat]
    while stack:
        c=stack.pop()
        for p in parent_graph.get(c, []):
            if p not in seen: seen.add(p); stack.append(p)
    return seen

def fol_closure(categories: List[str], parent_graph, disjoint_pairs):
    full=set(categories)
    for c in list(categories): full |= ancestors(parent_graph, c)
    conflicts=[]
    for a,b in disjoint_pairs:
        if a in full and b in full: conflicts.append((a,b))
    return sorted(full), conflicts

# ---------- Naive Bayes (as a lightweight Bayesian Network) ----------
def extract_features(book: Dict):
    kws=set(_normalize_list(book.get("keywords"))); subs=set(_normalize_list(book.get("subjects"))); aud=(book.get("audience") or "").lower()
    return {
        "HasMurder": ("murder" in kws or "detective" in kws),
        "HasSpaceship": ({"spaceship","planet","space"} & kws != set() or "space" in subs),
        "HasMagic": ({"magic","dragon","quest"} & kws != set()),
        "SubjectCrime": ("crime" in subs),
        "SubjectSpace": ("space" in subs),
        "SubjectHistory": ("history" in subs),
        "AudienceChildren": (aud=="children"),
    }

def naive_bayes_category(book: Dict, BN: Dict):
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

# ---------- Master classifier ----------
def classify_book(book: Dict, parent_graph=None, disjoint_pairs=None, BN=None):
    """Combine production rules + FOL closure + Naive Bayes backoff."""
    if parent_graph is None or disjoint_pairs is None:
        parent_graph, _, disjoint_pairs = load_taxonomy()
    if BN is None:
        BN = load_bn()

    explanations=[]; cats=[]
    for rule in RULES:
        ok, exp, newcats = rule(book)
        if ok:
            explanations.append(exp); cats.extend(newcats)
    cats=list(dict.fromkeys(cats))  # de-dup, keep order

    bn_used=False; bn_top=[]
    if not cats:
        bn_used=True
        ranked, feats = naive_bayes_category(book, BN)
        top1, p1 = ranked[0]
        cats.append(top1); bn_top=ranked[:3]
        feat_names = ", ".join([k for k,v in feats.items() if v]) or "∅"
        explanations.append(f"BN: chose {top1} (p={p1:.2f}) using features: {feat_names}")
        if len(ranked)>1 and ranked[1][1] > 0.35:
            cats.append(ranked[1][0])

    closure, conflicts = fol_closure(cats, parent_graph, disjoint_pairs)
    return {"categories":cats, "closure":closure, "conflicts":conflicts, "explanations":explanations, "bn_used":bn_used, "bn_top":bn_top}

# ---------- Utilities for search & lookup ----------
def find_by_title(title: Optional[str], rows: List[Dict]) -> Optional[Dict]:
    if not title: return None
    def _clean(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()
    tnorm = _clean(title)
    best = None; best_score = 0.0
    for r in rows:
        rtitle = r.get("title","")
        rnorm = _clean(rtitle)
        if rnorm == tnorm:
            return r
        a = set(tnorm.split()); b = set(rnorm.split())
        if not a or not b: continue
        score = len(a & b) / len(a | b)
        if score > best_score:
            best_score = score; best = r
    return best

def search_catalog(
    rows: List[Dict],
    parent_graph,
    disjoint_pairs,
    BN,
    terms: List[str] = None,
    category: str = "Any",
    audience: str = "",
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    neg_subject: Optional[str] = None,
    order: str = "none",      # "newest" | "oldest" | "none"
    limit: Optional[int] = None,
) -> List[Dict]:
    """Filter rows by free-text terms + KR category via classifier + meta filters."""
    terms = terms or []
    out = []
    for r in rows:
        hay = " ".join([
            r.get("title",""), r.get("authors",""), r.get("audience",""),
            r.get("keywords",""), r.get("subjects","")
        ]).lower()

        if terms and not any(t.lower() in hay for t in terms):
            continue

        if category and category != "Any":
            cres = classify_book(r, parent_graph, disjoint_pairs, BN)
            if category not in cres["closure"]:
                continue

        if audience and r.get("audience","") != audience:
            continue

        try:
            yr = int(r.get("year", 0))
        except Exception:
            yr = 0
        if year_min is not None and yr < int(year_min): continue
        if year_max is not None and yr > int(year_max): continue

        if neg_subject and neg_subject.lower() in (r.get("subjects","").lower()):
            continue

        out.append(r)

    if order == "newest":
        out = sorted(out, key=lambda r: int(r.get("year",0) or 0), reverse=True)
    elif order == "oldest":
        out = sorted(out, key=lambda r: int(r.get("year",0) or 0))

    if limit:
        out = out[:int(limit)]
    return out
