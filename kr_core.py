"""
kr_core.py — Knowledge Representation core for a librarian chatbot.
Methods: Taxonomy/Frames, FOL closure, Production Rules, Naive Bayes (BN).
Also includes search utilities. No Streamlit imports.
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
    data = json.loads(TAXONOMY_JSON.read_text(encoding="utf-8"))
    edges = data["edges"]
    disjoint_pairs = [tuple(x) for x in data.get("disjoint", [])]
    parent = defaultdict(list); child = defaultdict(list)
    for a,b in edges:
        parent[a].append(b)   # parent -> children
        child[b].append(a)    # child  -> parents
    return parent, child, disjoint_pairs

def load_bn():
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
        if not exists: writer.writeheader()
        writer.writerow(book)

# ---------- Helpers ----------
def _normalize_list(x):
    if not x: return []
    if isinstance(x, str):
        return [t.strip().lower() for t in x.split(",") if t.strip()]
    return []

# ---------- Production rules (return matched evidence) ----------
def rule_mystery(book):
    kws = set(_normalize_list(book.get("keywords"))); subs = set(_normalize_list(book.get("subjects")))
    hits = ({"murder","detective"} & kws) | ({"crime"} & subs)
    if hits:
        return True, f"R1: {', '.join(sorted(hits))} → Mystery", ["Mystery"], sorted(hits)
    return False, "", [], []

def rule_scifi(book):
    kws = set(_normalize_list(book.get("keywords"))); subs = set(_normalize_list(book.get("subjects")))
    hits = ({"spaceship","planet","space"} & kws) | ({"space"} & subs)
    if hits:
        return True, f"R2: {', '.join(sorted(hits))} → ScienceFiction", ["ScienceFiction"], sorted(hits)
    return False, "", [], []

def rule_fantasy(book):
    hits = {"magic","dragon","quest"} & set(_normalize_list(book.get("keywords")))
    if hits:
        return True, f"R3: {', '.join(sorted(hits))} → Fantasy", ["Fantasy"], sorted(hits)
    return False, "", [], []

def rule_childrens_fantasy(book):
    aud = (book.get("audience") or "").lower()
    kws = set(_normalize_list(book.get("keywords"))); subs = set(_normalize_list(book.get("subjects")))
    hits = ({"magic","dragon","quest"} & kws) | ({"fantasy"} & subs)
    if aud == "children" and hits:
        return True, f"R4: audience=children + {', '.join(sorted(hits))} → ChildrensFantasy", ["ChildrensFantasy"], ["children"] + sorted(hits)
    return False, "", [], []

def rule_history(book):
    subs = set(_normalize_list(book.get("subjects")))
    hits = {"history"} & subs
    if hits:
        return True, "R5: subject=History → History", ["History"], sorted(hits)
    return False, "", [], []

def rule_science(book):
    subs = set(_normalize_list(book.get("subjects"))); kws=set(_normalize_list(book.get("keywords")))
    hits = ({"science"} & subs) | ({"physics","biology","chemistry"} & kws)
    if hits:
        return True, f"R6: {', '.join(sorted(hits))} → Science", ["Science"], sorted(hits)
    return False, "", [], []

RULES = [rule_mystery, rule_scifi, rule_fantasy, rule_childrens_fantasy, rule_history, rule_science]

# ---------- FOL closure ----------
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

# ---------- Naive Bayes (lightweight BN) ----------
FEATURE_LABELS = {
    "HasMurder": "keywords include murder/detective",
    "HasSpaceship": "keywords include spaceship/planet/space OR subject Space",
    "HasMagic": "keywords include magic/dragon/quest",
    "SubjectCrime": "subject Crime",
    "SubjectSpace": "subject Space",
    "SubjectHistory": "subject History",
    "AudienceChildren": "audience = Children"
}

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

# ---------- Confidence policy ----------
def _confidence_label(rules_fired: bool, top_prob: float, n_true_feats: int) -> Tuple[str, float]:
    if rules_fired:
        return "high", 1.0
    if n_true_feats == 0:
        return "uncertain", 0.0
    if top_prob >= 0.70:
        return "medium", top_prob
    if top_prob >= 0.50:
        return "medium", top_prob
    return "low", top_prob

# ---------- Master classifier ----------
def classify_book(book: Dict, parent_graph=None, disjoint_pairs=None, BN=None):
    if parent_graph is None or disjoint_pairs is None:
        parent_graph, _, disjoint_pairs = load_taxonomy()
    if BN is None:
        BN = load_bn()

    explanations: List[str] = []
    rules_evidence: List[str] = []
    cats: List[str] = []

    for rule in RULES:
        ok, exp, newcats, ev = rule(book)
        if ok:
            explanations.append(exp); cats.extend(newcats); rules_evidence.extend(ev)

    cats = list(dict.fromkeys(cats))
    rules_fired = len(cats) > 0

    bn_used = False
    bn_top: List[Tuple[str, float]] = []
    bn_features_true: List[str] = []

    if not rules_fired:
        bn_used = True
        ranked, feats = naive_bayes_category(book, BN)
        bn_top = ranked[:3]
        bn_features_true = [FEATURE_LABELS[k] for k, v in feats.items() if v]
        n_true = len(bn_features_true)

        top_prob = ranked[0][1] if ranked else 0.0
        conf_label, conf_score = _confidence_label(False, top_prob, n_true)

        if conf_label == "uncertain":
            closure, conflicts = [], []
            explanations.append("Uncertain: no rule or BN feature matched. Add keywords/subjects (e.g., 'magic', 'space', 'history').")
            return {
                "categories": [],
                "closure": closure,
                "conflicts": conflicts,
                "explanations": explanations,
                "bn_used": True,
                "bn_top": bn_top,
                "evidence": {"rules": rules_evidence, "bn_features": bn_features_true},
                "confidence": conf_label,
                "confidence_score": conf_score
            }

        top1 = ranked[0][0]
        cats.append(top1)
        explanations.append(f"BN: chose {top1} (p={ranked[0][1]:.2f}) using features: "
                            f"{', '.join(bn_features_true) if bn_features_true else '∅'}")
        closure, conflicts = fol_closure(cats, parent_graph, disjoint_pairs)
        return {
            "categories": cats,
            "closure": closure,
            "conflicts": conflicts,
            "explanations": explanations,
            "bn_used": True,
            "bn_top": bn_top,
            "evidence": {"rules": rules_evidence, "bn_features": bn_features_true},
            "confidence": conf_label,
            "confidence_score": conf_score
        }

    closure, conflicts = fol_closure(cats, parent_graph, disjoint_pairs)
    conf_label, conf_score = _confidence_label(True, 1.0, 999)
    return {
        "categories": cats,
        "closure": closure,
        "conflicts": conflicts,
        "explanations": explanations,
        "bn_used": False,
        "bn_top": [],
        "evidence": {"rules": rules_evidence, "bn_features": []},
        "confidence": conf_label,
        "confidence_score": conf_score
    }

# ---------- Simple hierarchy for cards ----------
def hierarchy_from_classification(classification: Dict, child_graph: Dict[str,List[str]]):
    """
    Build (domain, category, subcategory, path) from a classification dict.
    Uses CHILD graph (child -> parents). Chooses first parent at each step for a linear path.
    """
    if not classification or not classification.get("categories"):
        return None
    leaf = classification["categories"][0]

    # Build chain: leaf -> parent1 -> parent2 -> ... (stop at Fiction/NonFiction or top)
    chain = [leaf]
    curr = leaf
    seen = set()
    while True:
        parents = child_graph.get(curr, [])
        if not parents: break
        p = parents[0]
        if p in seen: break
        chain.append(p)
        if p in ("Fiction","NonFiction"): break
        seen.add(p)
        curr = p

    # Domain = topmost (prefer Fiction/NonFiction if present)
    domain = chain[-1]
    # Category = the node just below domain (if any)
    below = chain[-2] if len(chain) >= 2 else leaf
    category = below if below != domain else leaf
    # Sub-Category = leaf when it differs from category, else "—"
    subcat = "—" if category == leaf else leaf
    path = " > ".join(reversed(chain))
    return {"domain": domain, "category": category, "subcategory": subcat, "path": path}

# ---------- Search ----------
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
    order: str = "none",
    limit: Optional[int] = None,
) -> List[Dict]:
    terms = terms or []
    out = []
    for r in rows:
        hay = " ".join([r.get("title",""), r.get("authors",""), r.get("audience",""),
                        r.get("keywords",""), r.get("subjects","")]).lower()

        if terms and not any(t.lower() in hay for t in terms): continue
        if category and category != "Any":
            cres = classify_book(r, parent_graph, disjoint_pairs, BN)
            if category not in cres["closure"]: continue
        if audience and r.get("audience","") != audience: continue

        try: yr = int(r.get("year", 0))
        except Exception: yr = 0
        if year_min is not None and yr < int(year_min): continue
        if year_max is not None and yr > int(year_max): continue

        if neg_subject and neg_subject.lower() in (r.get("subjects","").lower()): continue

        out.append(r)

    if order == "newest":
        out = sorted(out, key=lambda r: int(r.get("year",0) or 0), reverse=True)
    elif order == "oldest":
        out = sorted(out, key=lambda r: int(r.get("year",0) or 0))

    if limit: out = out[:int(limit)]
    return out