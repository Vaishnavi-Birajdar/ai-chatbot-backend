import os

import httpx
import chromadb


# -----------------------------
# Configuration
# -----------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50
DEFAULT_TOP_K = 3
MAX_DISTANCE = 0.65


# -----------------------------
# ChromaDB Configuration
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={"hnsw:space": "cosine"},
)


# -----------------------------
# Step 1: Generate Embeddings
# -----------------------------

def generate_embedding(text: str) -> list[float]:

    if not text or not text.strip():
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    response = httpx.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120.0,
    )

    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(
            "Ollama returned no embeddings. "
            "Check that nomic-embed-text is installed and running."
        )

    return embeddings[0]


# -----------------------------
# Step 2: Split PDF into Chunks
# -----------------------------

def chunk_pdf_pages(
    pages: list,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:

    if (
        chunk_size <= 0
        or overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError(
            "chunk_size must be positive and overlap "
            "must be between zero and chunk_size - 1."
        )

    chunks = []
    step = chunk_size - overlap

    for page_number, page in enumerate(pages, start=1):

        if isinstance(page, dict):
            text = page.get("text", "")
            page_number = page.get("page", page_number)
        else:
            text = page

        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()

        for start in range(0, len(text), step):

            chunk_text = text[
                start:start + chunk_size
            ].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": int(page_number),
                })

            if start + chunk_size >= len(text):
                break

    return chunks


# -----------------------------
# Step 3: Store Chunks in ChromaDB
# -----------------------------

def store_chunks(
    document_id: str,
    chunks: list[dict],
) -> int:

    if not chunks:
        return 0

    stored_count = 0

    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for index, chunk in enumerate(batch, start=start):

            chunk_text = chunk["text"].strip()

            if not chunk_text:
                continue

            ids.append(
                f"{document_id}_{index}"
            )

            documents.append(chunk_text)

            embeddings.append(
                generate_embedding(chunk_text)
            )

            metadatas.append({
                "document_id": document_id,
                "page": int(chunk["page"]),
                "chunk_index": index,
            })

        if not ids:
            continue

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        stored_count += len(ids)

    return stored_count


# -----------------------------
# Step 4: Search Relevant PDF Chunks
# -----------------------------

def search_chunks(
    question: str,
    document_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:

    if not question or not question.strip():
        return []

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    query_embedding = generate_embedding(question)

    search_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if document_id:
        search_args["where"] = {
            "document_id": document_id
        }

    results = collection.query(**search_args)

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    matches = []

    for index, text in enumerate(documents):

        metadata = metadatas[index]
        distance = distances[index]

        # Filter weak similarity matches.
        if distance > MAX_DISTANCE:
            continue

        matches.append({
            "text": text,
            "page": metadata["page"],
            "document_id": metadata["document_id"],
            "distance": distance,
        })

    return matches