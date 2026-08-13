# -*- coding: utf-8 -*-
"""
One-time RAG corpus ingestion (run after collecting PDFs).

Prerequisites:
    pip install chromadb sentence-transformers pypdf

    Place PDFs at:
        ./docs/regulations/JTG_D50_2017.pdf
        ./docs/regulations/NCHRP_1-37A_Implementation_Guide.pdf

Expected time:
    - First run: ~5-10 minutes (downloads BGE-large-zh model ~1.3 GB, embeds chunks)
    - Subsequent runs (re-ingestion): ~1-2 minutes

The vector DB is persisted under ./output/rag_db/ — only needs to be done once.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    from rl.rag import RAGStore

    pdf_paths = [
        os.path.join(PROJECT_ROOT, 'docs', 'regulations', 'JTG_D50_2017.pdf'),
        os.path.join(PROJECT_ROOT, 'docs', 'regulations', 'NCHRP_1-37A_Implementation_Guide.pdf'),
    ]

    print('=' * 70)
    print('RAG Corpus Ingestion')
    print('=' * 70)
    print()
    print('Expected PDFs:')
    for p in pdf_paths:
        exists = '✓' if os.path.exists(p) else '✗ MISSING'
        print('  [{}] {}'.format(exists, p))
    print()

    missing = [p for p in pdf_paths if not os.path.exists(p)]
    if missing:
        print('WARNING: {} PDF(s) missing. Will ingest only available ones.'.format(len(missing)))
        print('         (You can re-run this script after adding the missing PDFs.)')
        print()
        pdf_paths = [p for p in pdf_paths if os.path.exists(p)]

    if not pdf_paths:
        print('ERROR: No PDFs found. Place them at the expected paths above.')
        sys.exit(1)

    rag = RAGStore()
    if not rag.enabled:
        print('ERROR: RAG dependencies not installed.')
        print('Install: pip install chromadb sentence-transformers pypdf')
        sys.exit(1)

    print('Initial chunk count: {}'.format(rag.count()))
    print()
    print('Ingesting (this can take several minutes on first run)...')
    print()

    n_chunks = rag.ingest(pdf_paths, chunk_size=400, overlap=80)

    print()
    print('=' * 70)
    print('Ingestion complete: {} chunks added'.format(n_chunks))
    print('Total chunks in DB: {}'.format(rag.count()))
    print('Persisted at: {}'.format(rag.persist_dir))
    print('=' * 70)
    print()

    # Quick retrieval test
    if rag.count() > 0:
        print('Quick retrieval test:')
        for query in ['沥青疲劳寿命计算', 'permanent deformation rutting',
                      '半刚性基层 σ_t']:
            print('  Query: "{}"'.format(query))
            passages = rag.retrieve(query, top_k=2)
            for i, p in enumerate(passages):
                print('    [{}] {} (score={:.3f})'.format(i+1, p.source, p.score))
                print('        "{}"'.format(p.text[:120].replace('\n', ' ')))


if __name__ == '__main__':
    main()
