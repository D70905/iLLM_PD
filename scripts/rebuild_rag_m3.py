# -*- coding: utf-8 -*-
"""
scripts/rebuild_rag_m3.py
=========================
Rebuild RAG vector store with BGE-M3 multilingual embedding model.

Steps:
  1. Delete old Chroma DB (BGE-large-zh pavement_specs is kept as-is)
  2. Create new RAG store with BGE-M3 + new collection
  3. Ingest original PDFs (JTG D50-2017, NCHRP 1-37A) — if found
  4. Ingest regional knowledge MD
  5. Report chunk counts

Usage:
    conda activate illm_pd
    cd /d <PROJECT_ROOT>
    set PYTHONPATH=.
    set HF_HUB_OFFLINE=1
    python scripts/rebuild_rag_m3.py
"""
from __future__ import annotations
import logging, os, re, shutil, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rebuild_m3")

# ── Config ──
# Use SAME persist_dir as rag.py (default: output/rag_db).
# Different collection name (pavement_specs vs pavement_specs_m3)
# keeps old and new collections separate in the same Chroma DB.
PERSIST_DIR = PROJECT / "output" / "rag_db"
KNOWLEDGE_MD = PROJECT / "knowledge_base" / "regional_knowledge.md"

# PDF sources — JTG + NCHRP specification PDFs
PDF_CANDIDATES = [
    PROJECT / "docs" / "regulations" / "JTG_D50_2017.pdf",
    PROJECT / "docs" / "regulations" / "NCHRP_1-37A_Implementation_Guide.pdf",
]

# Sections to skip during MD ingest
SKIP_PREFIXES = ["4.2", "4.3"]  # carbon/cost placeholders — unsafe for RAG


def split_md_by_headings(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = []
    current_title = "preamble"
    current_lines = []
    for line in text.split("\n"):
        if line.startswith("### "):
            body = "\n".join(current_lines).strip()
            if body and len(body) >= 80:
                chunks.append({"title": current_title, "text": body})
            current_title = line.replace("### ", "").strip()
            current_lines = []
        else:
            current_lines.append(line)
    body = "\n".join(current_lines).strip()
    if body and len(body) >= 80:
        chunks.append({"title": current_title, "text": body})
    return chunks


def main():
    # ── Step 1: Init RAG with BGE-M3 (same persist_dir, new collection) ──
    from rl.rag import RAGStore
    rag = RAGStore(persist_dir=str(PERSIST_DIR))
    if not rag.enabled:
        logger.error("RAG not initialized. Check dependencies.")
        return 1

    # ── Step 3: Ingest PDFs ──
    pdfs_found = [p for p in PDF_CANDIDATES if p.exists()]
    if pdfs_found:
        total = rag.ingest([str(p) for p in pdfs_found])
        logger.info("PDF ingest: %d chunks from %d files", total, len(pdfs_found))
    else:
        logger.warning("No PDFs found in knowledge_base. Run PDF ingest separately with "
                       "rag.ingest(['path/to/JTG.pdf', 'path/to/NCHRP.pdf']).")

    # ── Step 4: Ingest regional knowledge ──
    all_chunks = split_md_by_headings(str(KNOWLEDGE_MD))
    to_ingest = [c for c in all_chunks
                 if not any(c["title"].startswith(p) for p in SKIP_PREFIXES)]
    skipped = [c["title"] for c in all_chunks
               if any(c["title"].startswith(p) for p in SKIP_PREFIXES)]
    logger.info("MD chunks: %d total, %d skipped, %d to ingest",
                len(all_chunks), len(skipped), len(to_ingest))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence_transformers not installed.")
        return 1

    embedder = SentenceTransformer(rag.embedding_model_name)
    ids, documents, metadatas = [], [], []
    for i, c in enumerate(to_ingest):
        safe_title = re.sub(r'[^a-zA-Z0-9_一-鿿-]', '_', c["title"])[:60]
        ids.append("m3_regional_{:03d}_{}".format(i, safe_title))
        documents.append(c["text"])
        metadatas.append({"source": "regional_knowledge.md", "section": c["title"], "chunk_idx": i})

    logger.info("Embedding %d chunks with %s ...", len(documents), rag.embedding_model_name)
    embeddings = embedder.encode(documents, batch_size=8, show_progress_bar=True,
                                 normalize_embeddings=True)
    try:
        rag._collection.add(ids=ids, documents=documents,
                            embeddings=embeddings.tolist(), metadatas=metadatas)
    except Exception as e:
        logger.warning("add() failed: %s, trying upsert", e)
        rag._collection.upsert(ids=ids, documents=documents,
                               embeddings=embeddings.tolist(), metadatas=metadatas)

    # ── Step 5: Report ──
    final_count = rag.count()
    logger.info("=" * 50)
    logger.info("Rebuild complete.")
    logger.info("  Model:    %s", rag.embedding_model_name)
    logger.info("  DB path:  %s (collection: %s)", PERSIST_DIR, rag.collection_name)
    logger.info("  Chunks:   %d", final_count)
    logger.info("  Old collection 'pavement_specs' preserved (unchanged)")
    logger.info("=" * 50)
    logger.info("To test: python scripts/dump_rag_queries.py")
    logger.info("To revert: change collection_name back to 'pavement_specs' in rag.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
