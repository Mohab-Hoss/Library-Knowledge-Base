import streamlit as st
from pathlib import Path
import csv

from kr_core import (
    ensure_data, load_taxonomy, load_bn, read_books, append_book,
    classify_book, find_by_title, search_catalog
)
from nl_parser import parse_message

st.set_page_config(page_title="Librarian KR — Chat", layout="wide")

# Bootstrap & load KR models
ensure_data()
PARENT, CHILD, DISJOINT = load_taxonomy()
BN = load_bn()

# ---------- Small helpers (UI-only rendering) ----------
def render_book_md(row, classification=None):
    lines = []
    lines.append(f"**{row.get('title','')}**")
    lines.append(f"- Authors: {row.get('authors','')}")
    lines.append(f"- Year: {row.get('year','')}")
    lines.append(f"- Audience: {row.get('audience','')}")
    lines.append(f"- Keywords: {row.get('keywords','')}")
    lines.append(f"- Subjects: {row.get('subjects','')}")
    if classification:
        lines.append(f"- Categories: {', '.join(classification['categories']) or '—'}")
        lines.append(f"- Closure: {', '.join(classification['closure']) or '—'}")
        if classification["conflicts"]:
            lines.append("- Conflicts: " + ", ".join([f"{a} vs {b}" for a,b in classification["conflicts"]]))
        if classification["explanations"]:
            lines.append("**Why:**")
            for e in classification["explanations"]:
                lines.append(f"  - {e}")
        elif classification["bn_used"]:
            top = ", ".join([f"{c}:{p:.2f}" for c,p in classification["bn_top"]])
            lines.append(f"- BN posterior (top): {top}")
    return "\n".join(lines)

# ---------- UI (Two Tabs) ----------
tab_chat, tab_catalog = st.tabs(["💬 Chat", "📚 Catalog UI"])

with tab_chat:
    st.markdown("**Talk to your librarian.** Examples:")
    st.code("""tell me about 'Moon Dagger'
why is it fantasy?
find YA sci-fi space top 3 newest
list not history after 2015
classify title=Detective Nile; audience=Adult; keywords=detective, murder""")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant","content":"Hi! Ask: `tell me about 'Moon Dagger'` or `why is it fantasy?`"}
        ]
    if "last_book" not in st.session_state:
        st.session_state.last_book = None

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type a message…")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)

        parsed = parse_message(prompt)
        intent = parsed["intent"]; slots = parsed["slots"]

        if intent == "about":
            row = find_by_title(slots.get("title"), read_books())
            if not row:
                reply = "I couldn't find that title. Try: `tell me about 'Moon Dagger'`"
            else:
                st.session_state.last_book = row
                cls = classify_book(row, PARENT, DISJOINT, BN)
                reply = render_book_md(row, cls)

        elif intent == "why":
            row = None
            if slots.get("title"):
                row = find_by_title(slots["title"], read_books())
            if not row:
                row = st.session_state.last_book
            if not row:
                reply = "Say the title first, e.g., `tell me about 'Moon Dagger'`"
            else:
                st.session_state.last_book = row
                cls = classify_book(row, PARENT, DISJOINT, BN)
                reply = render_book_md(row, cls)

        elif intent == "classify":
            book = {
                "title": slots.get("title",""),
                "authors": slots.get("authors",""),
                "year": slots.get("year",""),
                "audience": slots.get("audience","Adult"),
                "keywords": slots.get("keywords",""),
                "subjects": slots.get("subjects",""),
            }
            if not (book["title"] or book["keywords"] or book["subjects"]):
                reply = ("Provide at least `title`, `keywords`, or `subjects`.\n"
                         "Example: `classify title=Moon Dagger; audience=Adult; keywords=magic, quest`")
            else:
                st.session_state.last_book = book
                res = classify_book(book, PARENT, DISJOINT, BN)
                md = [f"**Categories (leaves):** {', '.join(res['categories']) or '—'}",
                      f"**Closure:** {', '.join(res['closure']) or '—'}"]
                if res["conflicts"]:
                    md.append("**Conflicts:** " + ", ".join([f"{a} vs {b}" for a,b in res["conflicts"]]))
                if res["bn_used"]:
                    md.append("**BN posterior (top):** " + ", ".join([f"{c}:{p:.2f}" for c,p in res["bn_top"]]))
                if res["explanations"]:
                    md.append("**Why:**"); md += [f"- {e}" for e in res["explanations"]]
                reply = "\n".join(md)

        elif intent == "add":
            book = {
                "title": slots.get("title",""),
                "authors": slots.get("authors",""),
                "year": slots.get("year",""),
                "audience": slots.get("audience","Adult"),
                "keywords": slots.get("keywords",""),
                "subjects": slots.get("subjects",""),
            }
            if not book["title"]:
                reply = ("To add, include at least `title=`.\n"
                         "Example: `add title=Detective Nile; authors=M. Azmi; year=2021; audience=Adult; "
                         "keywords=detective, murder; subjects=Crime`")
            else:
                append_book(book)
                st.session_state.last_book = book
                reply = f"✅ Added **{book['title']}** to catalog."

        else:  # search (NL or k=v)
            rows = read_books()
            out = search_catalog(
                rows, PARENT, DISJOINT, BN,
                terms=slots.get("terms") or [],
                category=slots.get("category","Any"),
                audience=slots.get("audience",""),
                year_min=slots.get("year_min"),
                year_max=slots.get("year_max"),
                neg_subject=slots.get("neg_subject"),
                order=slots.get("order","none"),
                limit=slots.get("limit"),
            )
            if not out:
                reply = "No matches. Try: `list children fantasy after 2015` or `find YA sci-fi space top 3 newest`"
            else:
                # "query understood" summary
                understood = []
                if slots.get("terms"): understood.append(f"terms={slots['terms']}")
                if slots.get("category") and slots["category"] != "Any": understood.append(f"category={slots['category']}")
                if slots.get("audience"): understood.append(f"audience={slots['audience']}")
                if slots.get("year_min") is not None or slots.get("year_max") is not None:
                    understood.append(f"year_range={[slots.get('year_min'), slots.get('year_max')]}")
                if slots.get("order","none") != "none": understood.append(f"sort={slots['order']}")
                if slots.get("limit"): understood.append(f"limit={slots['limit']}")
                if slots.get("neg_subject"): understood.append(f"not_subject={slots['neg_subject']}")

                md = []
                if understood:
                    md.append("_Query understood:_ " + ", ".join(understood) + "\n")
                md += ["| Title | Authors | Year | Audience | Keywords | Subjects |", "|-|-|-|-|-|-|"]
                for r in out[:20]:
                    md.append(f"| {r.get('title','')} | {r.get('authors','')} | {r.get('year','')} | "
                              f"{r.get('audience','')} | {r.get('keywords','')} | {r.get('subjects','')} |")
                if len(out) > 20:
                    md.append(f"\n_Showing 20 of {len(out)} results…_")
                reply = "\n".join(md)

        st.session_state.messages.append({"role":"assistant","content":reply})
        with st.chat_message("assistant"): st.markdown(reply)

    if st.button("Reset chat"):
        st.session_state.messages = []
        st.session_state.last_book = None
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

    book = {"title":title, "authors":authors, "year":str(year),
            "audience":audience, "keywords":keywords, "subjects":subjects}

    if do_classify:
        res = classify_book(book, PARENT, DISJOINT, BN)
        st.session_state.last_book = book
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
    cat = st.selectbox("Filter by Category", ["Any","ScienceFiction","Fantasy","ChildrensFantasy","Mystery","CrimeFiction","History","Science"], index=0)

    rows = read_books()
    filtered = []
    for r in rows:
        hay = " ".join([r.get("title",""), r.get("authors",""), r.get("audience",""),
                        r.get("keywords",""), r.get("subjects","")]).lower()
        if q and q.lower() not in hay: continue
        if cat != "Any":
            cres = classify_book(r, PARENT, DISJOINT, BN)
            if cat not in cres["closure"]: continue
        filtered.append(r)

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption("This table uses the same KR pipeline for category filtering.")
