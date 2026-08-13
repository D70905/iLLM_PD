# -*- coding: utf-8 -*-
"""
rl.rag — RAG over JTG D50-2017 + NCHRP 1-37A (Phase 2C)
==========================================================

BGE-large-zh embeddings + Chroma vector store.

Workflow:
  1. ingest(pdf_paths): split PDFs into chunks, embed, store in Chroma
  2. retrieve(query, top_k=3): return top relevant passages

Storage:
  ./output/rag_db/    (Chroma persistent dir)

This is for the GENERATOR to retrieve regulation context.

NOTE: Ingestion takes ~5-10 min one time (downloads BGE model ~1.3 GB).
After that, retrieve() is < 1 sec.

If you don't have the PDFs yet, you can run with `ingest=False` and RAG
will return empty list — Generator will work without regulation context
(falls back to default knowledge).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    text: str
    source: str        # e.g., 'JTG_D50_2017_p47'
    score: float       # cosine similarity 0-1


class RAGStore:
    """
    Lazy-init RAG with graceful fallback.

    If dependencies missing (chromadb, sentence-transformers, pypdf), or
    no PDFs ingested yet, retrieve() returns [].
    """

    def __init__(self,
                 persist_dir: str = './output/rag_db',
                 embedding_model: str = 'BAAI/bge-large-zh-v1.5',
                 collection_name: str = 'pavement_specs',
                 ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        self._collection = None
        self._embedder = None
        self._enabled = False
        self._init_attempted = False

    def _try_init(self) -> bool:
        """Lazy init. Returns True if successful."""
        if self._init_attempted:
            return self._enabled
        self._init_attempted = True

        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.warning('RAG dependencies missing ({}). '
                'Install: pip install chromadb sentence-transformers pypdf'.format(e))
            return False

        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_dir,
                                               settings=Settings(anonymized_telemetry=False))
            # Get or create collection
            try:
                self._collection = client.get_collection(self.collection_name)
                count = self._collection.count()
                logger.info('Loaded RAG collection: {} chunks'.format(count))
            except Exception:
                self._collection = client.create_collection(
                    self.collection_name, metadata={'hnsw:space': 'cosine'})
                logger.info('Created empty RAG collection')
        except Exception as e:
            logger.warning('Chroma init failed: {}'.format(e))
            return False

        try:
            logger.info('Loading embedding model {} ...'.format(self.embedding_model_name))
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info('Embedding model loaded')
        except Exception as e:
            logger.warning('Embedder load failed: {}'.format(e))
            return False

        self._enabled = True
        return True

    @property
    def enabled(self) -> bool:
        return self._try_init()

    def ingest(self, pdf_paths: List[str], chunk_size: int = 400,
               overlap: int = 80) -> int:
        """
        Ingest PDFs into the vector store.

        Returns number of chunks added. Call once after collecting PDFs.

        chunk_size / overlap are in characters (not tokens).
        """
        if not self._try_init():
            logger.error('RAG not initialized, cannot ingest')
            return 0

        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError:
                logger.error('Install pypdf: pip install pypdf')
                return 0

        total_chunks = 0
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.warning('Skipping missing PDF: {}'.format(pdf_path))
                continue

            doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
            logger.info('Ingesting {}'.format(pdf_path))

            try:
                reader = PdfReader(pdf_path)
            except Exception as e:
                logger.warning('Could not read {}: {}'.format(pdf_path, e))
                continue

            chunks = []
            metadatas = []
            ids = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ''
                except Exception:
                    continue
                text = text.strip()
                if len(text) < 50:
                    continue
                # Simple character-based chunking with overlap
                start = 0
                chunk_idx = 0
                while start < len(text):
                    chunk = text[start:start + chunk_size]
                    if len(chunk) < 50:
                        break
                    chunks.append(chunk)
                    metadatas.append({'source': '{}_p{}'.format(doc_id, page_num),
                                      'doc': doc_id, 'page': page_num,
                                      'chunk_idx': chunk_idx})
                    ids.append('{}_p{}_c{}'.format(doc_id, page_num, chunk_idx))
                    start += chunk_size - overlap
                    chunk_idx += 1

            if not chunks:
                continue

            # Batch embed
            logger.info('  Embedding {} chunks...'.format(len(chunks)))
            embeddings = self._embedder.encode(chunks, batch_size=32,
                                               show_progress_bar=False,
                                               normalize_embeddings=True)
            try:
                self._collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                )
                total_chunks += len(chunks)
                logger.info('  Added {} chunks from {}'.format(len(chunks), doc_id))
            except Exception as e:
                # Likely "ids already exist" — try update
                logger.warning('  add() failed ({}), trying upsert'.format(e))
                try:
                    self._collection.upsert(
                        ids=ids,
                        documents=chunks,
                        embeddings=embeddings.tolist(),
                        metadatas=metadatas,
                    )
                    total_chunks += len(chunks)
                except Exception as e2:
                    logger.error('  upsert also failed: {}'.format(e2))

        logger.info('Ingest complete. Total chunks: {}'.format(total_chunks))
        return total_chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedPassage]:
        """Return top-k most relevant passages. Empty list if RAG disabled."""
        if not self._try_init():
            return []
        if self._collection.count() == 0:
            return []

        try:
            query_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
            results = self._collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
            )
        except Exception as e:
            logger.warning('RAG retrieve failed: {}'.format(e))
            return []

        passages: List[RetrievedPassage] = []
        docs   = results.get('documents', [[]])[0]
        metas  = results.get('metadatas', [[]])[0]
        dists  = results.get('distances', [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            score = max(0.0, 1.0 - float(dist))
            passages.append(RetrievedPassage(
                text=doc, source=meta.get('source', '?'), score=score,
            ))
        return passages

    def count(self) -> int:
        if not self._try_init():
            return 0
        return self._collection.count()
