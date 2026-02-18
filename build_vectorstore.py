"""
Build and persist the ChromaDB vector store from the filtered course dataset.

Run once:
    conda run -n oisi_projekt python build_vectorstore.py

Output:
    data/chroma_db/   – persisted ChromaDB collection called 'courses'
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/rag_courses_filtered.parquet"
CHROMA_PATH = "data/chroma_db"
COLLECTION  = "courses"

# ── Model ──────────────────────────────────────────────────────────────────
# paraphrase-multilingual-MiniLM-L12-v2:
#   - 118 MB, supports 50+ languages incl. Estonian and English
#   - good balance of speed vs. quality for multilingual RAG
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Load data ──────────────────────────────────────────────────────────────
print(f"Loading {DATA_PATH} …")
df = pd.read_parquet(DATA_PATH)
print(f"  {len(df):,} courses loaded.")

# Fill NaN rag_text just in case
df["rag_text"] = df["rag_text"].fillna("")

# ── Embed ──────────────────────────────────────────────────────────────────
print(f"\nLoading embedding model: {MODEL_NAME} …")
model = SentenceTransformer(MODEL_NAME)

print("Embedding rag_text … (this may take a few minutes)")
embeddings = model.encode(
    df["rag_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,   # cosine similarity via dot product
)
print(f"  Embeddings shape: {embeddings.shape}")

# ── Build ChromaDB collection ──────────────────────────────────────────────
os.makedirs(CHROMA_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Drop + recreate so re-runs are idempotent
existing = [c.name for c in client.list_collections()]
if COLLECTION in existing:
    client.delete_collection(COLLECTION)
    print(f"\nDropped existing collection '{COLLECTION}'.")

collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)
print(f"Created collection '{COLLECTION}'.")

# Build metadata dicts for each course (used for display + filtering)
def safe_str(v) -> str:
    """Convert any value to a clean string; return '' for NA."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s in ("nan", "None", "NaN") else s


metadatas = []
for _, row in df.iterrows():
    metadatas.append({
        "code":                 safe_str(row.get("code")),
        "title_en":             safe_str(row.get("title_en")),
        "title_et":             safe_str(row.get("title_et")),
        "eap":                  safe_str(row.get("eap")),
        "semester":             safe_str(row.get("semester")),
        "city":                 safe_str(row.get("city")),
        "target_language":      safe_str(row.get("target_language")),
        "study_languages_en":   safe_str(row.get("study_languages_en")),
        "study_levels_en":      safe_str(row.get("study_levels_en")),
        "assessment_scale":     safe_str(row.get("assessment_scale")),
        "course_type":          safe_str(row.get("course_type")),
        "is_continuous_learning": safe_str(row.get("is_continuous_learning")),
    })

# ChromaDB requires string IDs
ids = [str(i) for i in range(len(df))]

# Add in batches of 500 to avoid memory spikes
BATCH = 500
for start in range(0, len(df), BATCH):
    end = min(start + BATCH, len(df))
    collection.add(
        ids=ids[start:end],
        embeddings=embeddings[start:end].tolist(),
        documents=df["rag_text"].iloc[start:end].tolist(),
        metadatas=metadatas[start:end],
    )
    print(f"  Added {end:,} / {len(df):,} documents …")

print(f"\nVector store persisted to: {CHROMA_PATH}")
print(f"Collection '{COLLECTION}' contains {collection.count()} documents.")
print("\nDone.")
