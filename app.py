"""
Ainete soovitaja – Tartu Ülikool
=================================
Streamlit rakendus, mis otsib semantiliselt sarnaseid aineid kasutaja
kirjelduse põhjal. Mudelina kasutatakse mitmekeelset sentence-transformers
mudelit; vektorite hoidlana ChromaDB-d.

Käivitamine:
    conda run -n oisi_projekt streamlit run app.py
"""

import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────
CHROMA_PATH = "data/chroma_db"
COLLECTION  = "courses"
MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 10   # number of results to retrieve

# ── Page setup ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TÜ ainete soovitaja",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Tartu Ülikooli ainete soovitaja")
st.markdown(
    "Kirjelda, mida soovid õppida, ja me leiame sulle sobivad ained. "
    "Saad kirjutada nii eesti kui ka inglise keeles."
)

# ── Cached resource loading ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Laadin mudelit …")
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Ühendun andmebaasiga …")
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION)


model      = load_model()
collection = load_collection()

# ── Sidebar filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtrid")

    semester_options = ["Kõik", "spring", "autumn"]
    semester = st.selectbox("Semester", semester_options)

    language_options = ["Kõik", "Estonian", "English", "Russian"]
    language = st.selectbox("Õppetöö keel", language_options)

    level_options = ["Kõik", "bachelor's studies", "master's studies", "doctoral studies"]
    level = st.selectbox("Õppeaste", level_options)

    n_results = st.slider("Tulemuste arv", min_value=3, max_value=20, value=5)

    st.markdown("---")
    st.caption(
        f"Mudel: `{MODEL_NAME}`  \n"
        f"Andmebaas: `{CHROMA_PATH}`  \n"
        f"Aineid indeksis: **{collection.count():,}**"
    )

# ── Main search area ────────────────────────────────────────────────────────
query = st.text_area(
    "Mida soovid õppida?",
    placeholder="Näiteks: tahan õppida masinõpet ja andmeanalüüsi / "
                "I want to learn about machine learning and data science",
    height=120,
)

search_clicked = st.button("Otsi aineid", type="primary", use_container_width=True)

# ── Search logic ─────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("Otsin sobivaid aineid …"):
        # Build ChromaDB where-filter
        where_clauses = []
        if semester != "Kõik":
            where_clauses.append({"semester": {"$eq": semester}})
        if language != "Kõik":
            where_clauses.append({"study_languages_en": {"$contains": language}})
        if level != "Kõik":
            where_clauses.append({"study_levels_en": {"$contains": level}})

        if len(where_clauses) > 1:
            where = {"$and": where_clauses}
        elif len(where_clauses) == 1:
            where = where_clauses[0]
        else:
            where = None

        # Embed the query
        q_embedding = model.encode(
            query.strip(),
            normalize_embeddings=True,
        ).tolist()

        # Query ChromaDB
        kwargs = {
            "query_embeddings": [q_embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            st.error(f"Viga päringu tegemisel: {e}")
            st.stop()

    # ── Display results ────────────────────────────────────────────────────
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    documents = results["documents"][0]

    if not metadatas:
        st.warning("Antud filtritega aineid ei leitud. Proovi filtrid eemaldada.")
    else:
        st.markdown(f"### Leitud {len(metadatas)} ainet")

        for rank, (meta, dist, doc) in enumerate(zip(metadatas, distances, documents), 1):
            # Cosine similarity from distance (ChromaDB returns cosine distance = 1 - similarity)
            similarity = 1.0 - dist
            title_en = meta.get("title_en") or ""
            title_et = meta.get("title_et") or ""
            code     = meta.get("code", "")
            eap      = meta.get("eap", "")
            sem      = meta.get("semester", "")
            city     = meta.get("city", "")
            langs    = meta.get("study_languages_en", "")
            levels   = meta.get("study_levels_en", "")
            scale    = meta.get("assessment_scale", "")

            display_title = title_en or title_et or code

            with st.expander(
                f"**{rank}. {display_title}** &nbsp;&nbsp; `{code}` &nbsp; "
                f"· {eap} EAP &nbsp; · sarnasus {similarity:.0%}",
                expanded=(rank <= 3),
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Kood:** {code}")
                    st.markdown(f"**Pealkiri (EN):** {title_en}")
                    st.markdown(f"**Pealkiri (ET):** {title_et}")
                    st.markdown(f"**EAP:** {eap}")
                    st.markdown(f"**Semester:** {sem}")
                with col2:
                    st.markdown(f"**Asukoht:** {city}")
                    st.markdown(f"**Õppetöö keel:** {langs}")
                    st.markdown(f"**Õppeaste:** {levels}")
                    st.markdown(f"**Hindamine:** {scale}")

                # ÕIS link
                ois_url = f"https://ois2.ut.ee/ainekava/{code}"
                st.markdown(f"[Ava ÕISis]({ois_url})")

                # Show a snippet of the retrieved rag_text
                st.markdown("---")
                st.markdown("**Tekst (katkend):**")
                snippet = doc[:800] + ("…" if len(doc) > 800 else "")
                st.code(snippet, language=None)

elif search_clicked and not query.strip():
    st.warning("Palun sisesta otsitav kirjeldus.")

# ── Empty state ───────────────────────────────────────────────────────────
else:
    st.info(
        "Sisesta kirjeldus ülalpool ja vajuta **Otsi aineid**. "
        "Soovi korral täpsusta otsingut vasakul olevate filtritega."
    )
