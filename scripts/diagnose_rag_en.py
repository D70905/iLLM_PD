"""
diagnose_rag_en.py — English-query RAG health check
====================================================
Tests whether the RAG store (built from JTG D50-2017 + NCHRP 1-37A PDFs)
can retrieve B.1-B.4 clauses and key parameters using ENGLISH queries,
matching the language of the manuscript (Nature Communications).
"""

import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl.rag import RAGStore

# ── Anchor keywords and verified values per clause (from jtg_d50.json) ──
CLAUSES = [
    {
        "name": "B.1 Asphalt fatigue cracking",
        "query": "asphalt fatigue cracking life Nf tensile strain epsilon 6.32 15.96 3.97 VFA thickness",
        "anchors": ["fatigue", "asphalt", "Nf", "tensile strain"],
        "values": [6.32, 15.96, 3.97],
    },
    {
        "name": "B.2 Semi-rigid base fatigue",
        "query": "semi-rigid base fatigue chemically stabilized tensile stress sigma_t R_s flexural strength",
        "anchors": ["fatigue", "chemically stabilized", "tensile stress", "R_s"],
        "values": [],
    },
    {
        "name": "B.3 AC permanent deformation (rutting)",
        "query": "asphalt permanent deformation rutting R_a allowable 15 mm compressive stress sublayer",
        "anchors": ["permanent deformation", "rutting", "R_a", "asphalt"],
        "values": [15.0, 10.0],
    },
    {
        "name": "B.4 Subgrade vertical strain",
        "query": "subgrade vertical compressive strain epsilon_z allowable 1.25 Ne k_T3",
        "anchors": ["subgrade", "compressive strain", "epsilon", "vertical"],
        "values": [0.9, 1.25, 0.21],
    },
    {
        "name": "3.0.1 Target reliability",
        "query": "target reliability index beta 1.65 95 percent expressway highway",
        "anchors": ["reliability", "beta"],
        "values": [95, 1.65, 90],
    },
    {
        "name": "3.0.3 Standard axle BZZ-100",
        "query": "standard axle BZZ-100 tire contact pressure 0.7 MPa 100 kN",
        "anchors": ["axle", "BZZ", "tire", "contact pressure"],
        "values": [100.0, 0.7],
    },
    {
        "name": "3.0.2 Design life",
        "query": "design life years expressway highway 15 years pavement structure",
        "anchors": ["design life", "years", "expressway"],
        "values": [15, 12, 8],
    },
]


def main():
    store = RAGStore()
    if not store._try_init():
        print("ERROR: RAG not initialized")
        return

    total = store._collection.count()
    print(f"RAG chunks: {total}")
    print()

    anchor_hits = 0
    value_hits = 0
    total_values = 0

    for clause in CLAUSES:
        print(f"{'='*70}")
        print(f"  {clause['name']}")
        print(f"{'='*70}")
        print(f"  Query: {clause['query'][:100]}")

        results = store.retrieve(clause["query"], top_k=2)
        if not results:
            print(f"  !! NO RESULTS")
            print()
            continue

        for i, r in enumerate(results[:2]):
            text = r.text.strip()
            snippet = text[:200].replace("\n", " ")
            print(f"  Top{i+1}: source={r.source} score={r.score:.2f}")
            print(f"    -> {snippet}")

            if i == 0:  # only check anchors on top result
                # Check anchor keywords
                hits = [a for a in clause["anchors"]
                        if a.lower() in text.lower()]
                miss = [a for a in clause["anchors"]
                        if a.lower() not in text.lower()]
                if hits:
                    print(f"    Anchors HIT: {hits}")
                if miss:
                    print(f"    Anchors MISS: {miss}")

                anchor_hit_count = len(hits)
                # Relaxed: hit if >=2 anchors found (not all)
                if anchor_hit_count >= 2:
                    anchor_hits += 1
                    print(f"    -> ANCHOR OK ({anchor_hit_count}/{len(clause['anchors'])})")
                else:
                    print(f"    -> ANCHOR WEAK ({anchor_hit_count}/{len(clause['anchors'])})")

                # Check verified values
                if clause["values"]:
                    vals_found = []
                    vals_missed = []
                    text_nums = re.findall(r'\b\d+\.?\d*\b', text)
                    text_floats = set()
                    for n in text_nums:
                        try:
                            text_floats.add(float(n))
                        except ValueError:
                            pass
                    for v in clause["values"]:
                        # Check if value or close value appears
                        found = False
                        for tf in text_floats:
                            if abs(tf - v) / max(abs(v), 1) < 0.05:
                                found = True
                                break
                        if found:
                            vals_found.append(v)
                        else:
                            vals_missed.append(v)

                    if vals_found:
                        print(f"    Values FOUND: {vals_found}")
                    if vals_missed:
                        print(f"    Values MISSED: {vals_missed}")

                    value_hits += len(vals_found)
                    total_values += len(clause["values"])

        print()

    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Anchor hits (>=2 anchors):  {anchor_hits}/{len(CLAUSES)}")
    if total_values > 0:
        print(f"  Verified value recall:      {value_hits}/{total_values}")
    print()

    if anchor_hits >= 5:
        print("  VERDICT: English RAG is USABLE for specification clause verification.")
        print("  Output explainer can do BOTH:")
        print("    1. Numeric assertion verification (against margins + jtg_d50.json)")
        print("    2. Specification clause verification (RAG retrieval + anchor match)")
    elif anchor_hits >= 3:
        print("  VERDICT: English RAG is PARTIALLY usable.")
        print("  Spec clause verification possible with caveats for weak clauses.")
    else:
        print("  VERDICT: English RAG still has gaps.")
        print("  Stick to Route 1: numeric assertion verification only.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()