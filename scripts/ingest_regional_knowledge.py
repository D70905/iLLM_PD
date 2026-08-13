# -*- coding: utf-8 -*-
"""
scripts/ingest_regional_knowledge.py
====================================
Ingest regional_knowledge.md into the RAG vector store.

Splits by ### headers into independent chunks (~200-500 chars each),
embeds with BGE-large-zh, and adds to the existing Chroma collection.

Usage:
    conda activate illm_pd
    cd /d <PROJECT_ROOT>
    set PYTHONPATH=.
    python scripts/ingest_regional_knowledge.py

Skip sections marked as placeholder (§4.2 carbon, §4.3 cost).
"""

from __future__ import annotations
import logging, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl.rag import RAGStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_regional")

MD_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "regional_knowledge.md")

# Sections to SKIP (indicative placeholder data — unsafe for LLM to use as facts)
SKIP_PREFIXES = [
    "4.2",   # carbon factors — indicative placeholders
    "4.3",   # material costs — indicative placeholders
]

def split_md_by_headings(path: str) -> list[dict]:
    """Split markdown into chunks at ### boundaries."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove the verification banner (lines starting with "> " before first "##")
    # Keep the banner text as metadata but don't split it as chunks
    chunks = []
    current_title = "preamble"
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("### "):
            # Save previous chunk
            body = "\n".join(current_lines).strip()
            if body and len(body) >= 80:
                chunks.append({"title": current_title, "text": body})
            current_title = line.replace("### ", "").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Last chunk
    body = "\n".join(current_lines).strip()
    if body and len(body) >= 80:
        chunks.append({"title": current_title, "text": body})

    return chunks


def main():
    logger.info("Reading %s ...", MD_PATH)
    all_chunks = split_md_by_headings(MD_PATH)

    # Filter out placeholder sections
    chunks_to_ingest = []
    skipped = []
    for c in all_chunks:
        should_skip = any(c["title"].startswith(p) for p in SKIP_PREFIXES)
        if should_skip:
            skipped.append(c["title"])
        else:
            chunks_to_ingest.append(c)

    logger.info("Total chunks: %d  |  Skipped (placeholder): %d  |  To ingest: %d",
                len(all_chunks), len(skipped), len(chunks_to_ingest))
    if skipped:
        logger.info("Skipped sections: %s", ", ".join(skipped))

    # Initialize RAG
    rag = RAGStore()
    if not rag.enabled:
        logger.error("RAG not initialized. Check chromadb + sentence_transformers installation.")
        return 1

    existing = rag.count()
    logger.info("Existing chunks in DB: %d", existing)

    # Embed and add
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence_transformers not installed. Run: pip install sentence-transformers")
        return 1

    embedder = SentenceTransformer(rag.embedding_model_name)

    # Prepare data
    ids = []
    documents = []
    metadatas = []
    for i, c in enumerate(chunks_to_ingest):
        # Create a safe ID
        safe_title = re.sub(r'[^a-zA-Z0-9_一-鿿-]', '_', c["title"])[:60]
        ids.append(f"regional_{i:03d}_{safe_title}")
        documents.append(c["text"])
        metadatas.append({
            "source": "regional_knowledge.md",
            "section": c["title"],
            "chunk_idx": i,
        })

    logger.info("Embedding %d chunks with %s ...", len(documents), rag.embedding_model_name)
    embeddings = embedder.encode(documents, batch_size=16, show_progress_bar=True,
                                 normalize_embeddings=True)

    logger.info("Adding to Chroma collection '%s' ...", rag.collection_name)
    try:
        rag._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
    except Exception as e:
        logger.warning("add() failed (%s), trying upsert ...", e)
        rag._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    new_count = rag.count()
    logger.info("Done. DB chunks: %d -> %d (added %d)",
                existing, new_count, new_count - existing)
    return 0


if __name__ == "__main__":
    sys.exit(main())