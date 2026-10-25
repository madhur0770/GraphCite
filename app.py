"""GraphCite - Streamlit front end.

Run with:  streamlit run app.py
Demo mode needs no services. Live mode needs Neo4j and Ollama (see README).
"""
from __future__ import annotations

import os

import streamlit as st

from graphcite import pipeline
from graphcite.demo_data import DOCUMENTS, SAMPLE_QUESTIONS, ScriptedLLM
from graphcite.llm import LLMError, OllamaLLM
from graphcite.store import InMemoryGraphStore, Neo4jGraphStore

STATUS_LABEL = {
    "supported": "Supported",
    "weak": "Weak support",
    "uncited": "No citation",
    "invalid-citation": "Invalid citation",
}

st.set_page_config(page_title="GraphCite", page_icon=":mag:", layout="wide")
st.title("GraphCite")
st.caption("Multi-hop question answering over a knowledge graph, with citations checked against the source sentences.")

mode = st.sidebar.radio("Mode", ["Demo (no services)", "Live (Neo4j + Ollama)"], key="mode")
depth = st.sidebar.slider("Graph hops", 1, 4, 2, help="How far to expand from the entities found in the question.")

if mode.startswith("Demo"):
    st.sidebar.info("Demo mode uses a scripted stand-in for the LLM and an in-memory graph. It only understands the three bundled fictional documents.")
else:
    neo4j_uri = st.sidebar.text_input("Neo4j URI", os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = st.sidebar.text_input("Neo4j user", os.environ.get("NEO4J_USER", "neo4j"))
    neo4j_password = st.sidebar.text_input("Neo4j password", os.environ.get("NEO4J_PASSWORD", ""), type="password")
    model_name = st.sidebar.text_input("Ollama model", os.environ.get("GRAPHCITE_MODEL", "llama3.2"))


def connect():
    """Return (store, llm) for the current mode, cached in the session."""
    key = (mode, None if mode.startswith("Demo") else (neo4j_uri, neo4j_user, model_name))
    if st.session_state.get("conn_key") != key:
        old = st.session_state.get("store")
        if old is not None:
            old.close()
        if mode.startswith("Demo"):
            store, llm = InMemoryGraphStore(), ScriptedLLM()
            for name, text in DOCUMENTS.items():
                pipeline.ingest(store, llm, name, text)
        else:
            store, llm = Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_password), OllamaLLM(model_name)
        st.session_state.update(conn_key=key, store=store, llm=llm)
    return st.session_state["store"], st.session_state["llm"]


try:
    store, llm = connect()
except Exception as exc:  # noqa: BLE001 - show connection problems to the user
    st.error(f"Could not connect: {exc}")
    st.stop()

tab_ask, tab_ingest, tab_graph = st.tabs(["Ask", "Ingest", "Graph"])

with tab_ask:
    if mode.startswith("Demo"):
        choice = st.selectbox("Sample question", [q for q, _ in SAMPLE_QUESTIONS])
        if st.button("Use sample question"):
            st.session_state.question = choice
            st.session_state.depth_hint = dict(SAMPLE_QUESTIONS)[choice]
        if st.session_state.get("depth_hint"):
            st.caption(f"This sample needs about {st.session_state.depth_hint} hop(s); set the slider in the sidebar.")
    question = st.text_input("Question", key="question", placeholder="Who founded the company that develops Helios Drive?")
    if st.button("Answer with citations", type="primary", icon=":material/fact_check:"):
        if not question.strip():
            st.error("Enter a question.")
        else:
            with st.spinner("Searching the graph..."):
                try:
                    st.session_state.result = pipeline.ask(store, llm, question, depth)
                except LLMError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Query failed: {exc}")
    result = st.session_state.get("result")
    if result:
        st.subheader("Answer")
        with st.container(border=True):
            st.write(result.text)
        if result.claims:
            m1, m2, m3 = st.columns(3)
            m1.metric("Claims supported by their citations", f"{result.supported_ratio:.0%}")
            m2.metric("Claims checked", len(result.claims))
            m3.metric("Evidence sentences", len(result.evidence))
            st.dataframe(
                [{"Claim": c.sentence, "Cites": ", ".join(map(str, c.refs)) or "-", "Word overlap": c.support, "Check": STATUS_LABEL[c.status]} for c in result.claims],
                hide_index=True,
            )
        st.subheader("Evidence")
        for item in result.evidence:
            cited = " (cited)" if item.ref in result.cited_refs else ""
            with st.expander(f"[{item.ref}] {item.doc}{cited}"):
                st.write(item.sentence)
                st.caption("Graph path: " + "  >  ".join(item.path))
        with st.expander("Retrieval trace"):
            for step in result.steps:
                st.text(step)

with tab_ingest:
    st.write("Extract entities and relations from a document and add them to the graph. Every fact must be backed by a sentence quoted from the text; anything else is dropped and listed below.")
    if mode.startswith("Demo"):
        st.info("Demo mode can only ingest the bundled documents (already loaded). Switch to Live mode to ingest your own text.")
    name = st.text_input("Document name", "my-document")
    text = st.text_area("Document text", height=180)
    if st.button("Extract and add to graph"):
        if not text.strip():
            st.error("Paste some text first.")
        else:
            with st.spinner("Extracting..."):
                try:
                    extraction = pipeline.ingest(store, llm, name.strip() or "document", text)
                    st.success(f"Added {len(extraction.entities)} entities and {len(extraction.relations)} relations.")
                    if extraction.dropped:
                        with st.expander(f"{len(extraction.dropped)} item(s) dropped by grounding checks"):
                            for reason in extraction.dropped:
                                st.write("- " + reason)
                except LLMError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ingestion failed: {exc}")

with tab_graph:
    stats = store.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Entities", stats["entities"])
    c2.metric("Relations", stats["relations"])
    c3.metric("Documents", stats["documents"])
    entities = {e.id: e for e in store.all_entities()}
    relations = store.all_relations()
    if relations:
        dot = ["digraph G { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];"]
        for e in entities.values():
            dot.append(f'"{e.id}" [label="{e.name.replace(chr(34), "")}\\n({e.type})"];')
        for r in relations:
            dot.append(f'"{r.source_id}" -> "{r.target_id}" [label="{r.type}", fontsize=10];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot))
    if not mode.startswith("Demo"):
        confirm = st.checkbox("I understand this deletes every GraphCite entity and relation from the connected database.")
        if st.button("Clear graph", disabled=not confirm):
            store.clear()
            st.session_state.pop("result", None)
            st.rerun()
