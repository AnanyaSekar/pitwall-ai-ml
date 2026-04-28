import chromadb
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# ── SETUP ──
MODEL_NAME = "all-MiniLM-L6-v2"  # fast, free, runs locally
DB_PATH = "data/chromadb"

def build_vector_db():
    """Convert all 22,126 laps into vectors and store in ChromaDB"""
    print("Loading laps CSV...")
    df = pd.read_csv("data/laps.csv")
    df = df[(df["lap_time_s"] > 60) & (df["lap_time_s"] < 300)].reset_index(drop=True)
    print(f"Building vector DB from {len(df)} laps...")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("f1_laps")
    except:
        pass

    collection = client.create_collection("f1_laps")

    # Build documents — each lap becomes a text description
    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        text = (
            f"Driver {row['driver']} at {row['gp']} {int(row['year'])}: "
            f"Lap {int(row['lap'])}, tyre {row['compound']} age {int(row['tyre_life'])} laps, "
            f"lap time {row['lap_time_s']:.3f}s, stint {int(row['stint'])}"
        )
        docs.append(text)
        ids.append(str(i))
        metas.append({
            "driver": str(row["driver"]),
            "gp": str(row["gp"]),
            "year": int(row["year"]),
            "lap": int(row["lap"]),
            "compound": str(row["compound"]),
            "lap_time_s": float(row["lap_time_s"]),
        })

        # Batch insert every 500
        if len(docs) == 500:
            embeddings = model.encode(docs).tolist()
            collection.add(documents=docs, embeddings=embeddings,
                          ids=ids, metadatas=metas)
            docs, ids, metas = [], [], []
            print(f"  Indexed {i+1}/{len(df)} laps...")

    # Insert remaining
    if docs:
        embeddings = model.encode(docs).tolist()
        collection.add(documents=docs, embeddings=embeddings,
                      ids=ids, metadatas=metas)

    print(f"Vector DB built! {collection.count()} laps indexed.")
    return collection

def get_collection():
    """Load existing vector DB"""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection("f1_laps")

def retrieve(question: str, n=15) -> str:
    """Find the most relevant laps for a question"""
    model = SentenceTransformer(MODEL_NAME)
    collection = get_collection()
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n
    )
    docs = results["documents"][0]
    return "\n".join(docs)

def rag_answer(question: str, groq_client) -> str:
    """Full RAG pipeline — retrieve then generate"""
    # Step 1: Retrieve relevant laps
    context = retrieve(question)

    # Step 2: Generate grounded answer
    prompt = f"""You are an expert F1 analyst with access to real race data.

REAL F1 LAP DATA (retrieved from 22,126 actual laps):
{context}

Based on this real data, answer the following question accurately and specifically.
Reference actual lap times, drivers, and races from the data above.

Question: {question}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an F1 data analyst. Always ground your answers in the provided real race data."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    build_vector_db()
    print("\nRAG pipeline ready!")
