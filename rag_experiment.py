
from pathlib import Path
import json
import random
import re
import argparse

import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# KONFIGURASI
# ============================================================

BASE = Path(__file__).resolve().parent
DOC_DIR = BASE / "dokumen"
OUT = BASE / "hasil"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZES = (500, 1000, 1500)
OVERLAP_RATIO = 0.15
TEST_SIZES = (500, 1000)
TOP_K_VALUES = (1, 3, 5)


# ============================================================
# 1. MEMBACA PDF
# ============================================================

def baca_pdf(path):
    """
    Mengekstrak teks PDF dan menyertakan nomor halaman.
    Menggunakan PyMuPDF.
    """

    pages = []

    doc = pymupdf.open(path)

    try:
        for i, page in enumerate(doc, start=1):

            text = page.get_text(
                "text",
                sort=True
            ).strip()

            if text:
                pages.append(
                    f"[HALAMAN {i}]\n{text}"
                )

    finally:
        doc.close()

    return "\n\n".join(pages)


# ============================================================
# 2. MEMOTONG TEKS (CHUNKING)
# ============================================================

def potong(teks, ukuran=1000, overlap=150):
    """
    Memotong teks berdasarkan jumlah karakter.
    Berusaha mempertahankan batas kalimat dan paragraf.
    """

    if ukuran <= 0 or overlap < 0 or overlap >= ukuran:
        raise ValueError(
            "Pastikan ukuran > overlap >= 0"
        )

    teks = re.sub(r"[ \t]+", " ", teks).strip()

    hasil = []
    start = 0
    n = len(teks)

    while start < n:

        end = min(start + ukuran, n)

        if end < n:

            floor = start + int(ukuran * 0.70)

            separators = (
                "\n\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", "
            )

            candidates = [
                teks.rfind(separator, floor, end)
                for separator in separators
            ]

            boundary = max(candidates)

            if boundary > floor:

                separator = next(
                    s for s in separators
                    if teks.startswith(s, boundary)
                )

                end = boundary + len(separator)

        piece = teks[start:end].strip()

        if piece:
            hasil.append(piece)

        if end >= n:
            break

        # Mempertahankan overlap antarchunk
        start = max(start + 1, end - overlap)

    return hasil


# ============================================================
# 3. MEMUAT SELURUH DOKUMEN
# ============================================================

def load_docs():

    paths = sorted(DOC_DIR.glob("*.pdf"))

    if len(paths) < 2:
        raise FileNotFoundError(
            f"Ditemukan {len(paths)} PDF di {DOC_DIR}. "
            "Masukkan minimal 2-3 PDF ke folder dokumen."
        )

    docs = []

    for path in paths:

        print(f"\nMembaca: {path.name}")

        text = baca_pdf(path)

        if not text.strip():

            print(
                f"PERINGATAN: {path.name} "
                "tidak menghasilkan teks. "
                "Kemungkinan PDF hasil scan membutuhkan OCR."
            )

            continue

        print(
            f"Berhasil membaca {len(text):,} karakter."
        )

        docs.append((path.name, text))

    if not docs:
        raise RuntimeError(
            "Semua PDF kosong atau tidak terbaca."
        )

    return docs


# ============================================================
# 4. MEMBANGUN KOLEKSI CHUNK
# ============================================================

def build_collection(
    docs,
    size,
    overlap_ratio=OVERLAP_RATIO
):

    rows = []

    overlap = round(size * overlap_ratio)

    print(
        f"\nMembangun koleksi ukuran {size} "
        f"(overlap {overlap})"
    )

    for filename, text in docs:

        chunks = potong(
            text,
            ukuran=size,
            overlap=overlap
        )

        print(
            f"{filename}: {len(chunks)} chunk"
        )

        for i, chunk in enumerate(chunks):

            rows.append({
                "source": filename,
                "chunk_id": i,
                "text": chunk
            })

    return rows


# ============================================================
# 5. MEMBUAT EMBEDDING / INDEKS
# ============================================================

def make_index(rows, model):

    if not rows:
        raise ValueError(
            "Tidak ada chunk untuk dibuat embedding."
        )

    embeddings = model.encode(
        [row["text"] for row in rows],
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# 6. QUERY / PENCARIAN SEMANTIK
# ============================================================

def query(
    question,
    rows,
    embeddings,
    model,
    top_k=5
):

    if not rows:
        return []

    if len(rows) != len(embeddings):
        raise ValueError(
            "Jumlah chunk dan embedding tidak sama."
        )

    q = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Cosine similarity karena embedding dinormalisasi
    scores = embeddings @ q

    top_k = min(top_k, len(rows))

    ids = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in ids:

        result = dict(rows[int(i)])

        result["score"] = float(
            scores[int(i)]
        )

        results.append(result)

    return results


# ============================================================
# 7. AUDIT TIGA CHUNK ACAK
# ============================================================

def random_chunk_audit(
    rows,
    size,
    count=3,
    seed=42
):

    rng = random.Random(seed + size)

    chosen = rng.sample(
        rows,
        min(count, len(rows))
    )

    audit = []

    for row in chosen:

        text = row["text"].strip()

        last = text[-1:] if text else ""

        audit.append({

            "source": row["source"],

            "chunk_id": row["chunk_id"],

            "length": len(text),

            "preview_start": text[:220],

            "preview_end": text[-220:],

            "ends_mid_sentence": bool(
                last and last not in ".!?;:)]}\""
            )

        })

    return audit


# ============================================================
# 8. HIT@K
# ============================================================

def hit_at_k(
    test_questions,
    rows,
    embeddings,
    model,
    k
):

    hits = 0
    details = []

    for item in test_questions:

        question = item["question"]

        expected_sources = item.get(
            "sources", []
        )

        keywords = [
            str(keyword).lower()
            for keyword in item.get("keywords", [])
            if str(keyword).strip()
        ]

        results = query(
            question,
            rows,
            embeddings,
            model,
            top_k=k
        )

        hit = False
        matched_chunk = None

        for result in results:

            text = result["text"].lower()

            source_match = (
                result["source"] in expected_sources
            )

            keyword_count = sum(
                1 for keyword in keywords
                if keyword in text
            )

            # Minimal 2 keyword jika tersedia 2 atau lebih.
            # Jika hanya ada 1 keyword, gunakan 1.
            required_keywords = min(
                2,
                len(keywords)
            )

            keyword_match = (
                required_keywords > 0
                and keyword_count >= required_keywords
            )

            if source_match and keyword_match:

                hit = True

                matched_chunk = {
                    "source": result["source"],
                    "chunk_id": result["chunk_id"],
                    "matched_keywords": keyword_count
                }

                break

        hits += int(hit)

        details.append({

            "question": question,

            "hit": hit,

            "matched_chunk": matched_chunk,

            "top_sources": [
                result["source"]
                for result in results
            ],

            "top_scores": [
                round(result["score"], 4)
                for result in results
            ]

        })

    total = len(test_questions)

    score = (
        hits / total
        if total > 0
        else 0
    )

    return {

        "k": k,

        "hits": hits,

        "total": total,

        "hit_at_k": score,

        "details": details

    }


# ============================================================
# 9. MENYIMPAN HASIL JSON
# ============================================================

def save_json(filename, data):

    path = OUT / filename

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


# ============================================================
# 10. PROGRAM UTAMA
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LANGKAH 1: MEMBACA PDF
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LANGKAH 1: MEMBACA PDF")
    print("=" * 60)

    docs = load_docs()

    first_name, first_text = docs[0]

    print("\nPDF pertama:", first_name)

    print(
        "Jumlah karakter:",
        len(first_text)
    )

    print("\n500 KARAKTER PERTAMA:")

    print(first_text[:500])

    # --------------------------------------------------------
    # LANGKAH 2-3: CHUNKING DAN AUDIT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LANGKAH 2-3: CHUNKING DAN AUDIT ACAK")
    print("=" * 60)

    collections = {}

    for size in CHUNK_SIZES:

        rows = build_collection(
            docs,
            size
        )

        collections[size] = rows

        print(
            f"\nTOTAL CHUNK UKURAN {size}: "
            f"{len(rows)}"
        )

        audit = random_chunk_audit(
            rows,
            size,
            count=3,
            seed=args.seed
        )

        save_json(
            f"audit_chunk_{size}.json",
            audit
        )

        print("\nTIGA SAMPEL CHUNK:")

        for a in audit:

            print(
                f"\nFile: {a['source']}"
            )

            print(
                f"Chunk ID: {a['chunk_id']}"
            )

            print(
                f"Panjang: {a['length']} karakter"
            )

            print(
                "Akhir tampak terpotong:",
                a["ends_mid_sentence"]
            )

            print(
                "Akhir teks:",
                a["preview_end"][-120:]
            )

    # --------------------------------------------------------
    # LANGKAH 4: MEMBUAT INDEKS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LANGKAH 4: MEMBUAT INDEKS")
    print("=" * 60)

    print(
        "\nMemuat model:",
        MODEL_NAME
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    indexes = {}

    for size, rows in collections.items():

        print(
            f"\nMembuat embedding ukuran {size}"
        )

        embeddings = make_index(
            rows,
            model
        )

        indexes[size] = embeddings

        np.save(
            OUT / f"embedding_{size}.npy",
            embeddings
        )

        save_json(
            f"chunks_{size}.json",
            rows
        )

        print(
            f"Indeks ukuran {size}: "
            f"{len(embeddings)} embedding"
        )

    # --------------------------------------------------------
    # MEMUAT 15 PERTANYAAN UJI
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MEMUAT BANK PERTANYAAN")
    print("=" * 60)

    question_path = BASE / "pertanyaan_uji.json"

    if not question_path.exists():

        raise FileNotFoundError(
            "File pertanyaan_uji.json tidak ditemukan."
        )

    tests = json.loads(
        question_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(tests, list) or len(tests) != 15:

        raise ValueError(
            "pertanyaan_uji.json harus berisi "
            "tepat 15 pertanyaan."
        )

    for i, item in enumerate(tests, start=1):

        if not all(
            key in item
            for key in ("question", "keywords", "sources")
        ):

            raise ValueError(
                f"Pertanyaan nomor {i} "
                "harus memiliki question, keywords, dan sources."
            )

    print(
        f"Berhasil memuat {len(tests)} pertanyaan."
    )

    # --------------------------------------------------------
    # LANGKAH 5-6: QUERY DAN PARAFRASE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LANGKAH 5-6: QUERY DAN PARAFRASE")
    print("=" * 60)

    paraphrases = [
        item for item in tests
        if item.get("paraphrase")
    ]

    if len(paraphrases) < 2:

        raise ValueError(
            "Bank pertanyaan harus memiliki minimal "
            "2 pertanyaan dengan paraphrase."
        )

    demo_questions = (
        tests[:5] + paraphrases[:2]
    )

    query_reports = {}

    for size in TEST_SIZES:

        print(
            f"\n--- PENGUJIAN QUERY CHUNK {size} ---"
        )

        size_reports = []

        for item in demo_questions:

            question = item["question"]

            is_paraphrase = bool(
                item.get("paraphrase")
            )

            print(
                "\nPertanyaan:",
                question
            )

            if is_paraphrase:

                print("[PARAFRASE]")

            results = query(
                question,
                collections[size],
                indexes[size],
                model,
                top_k=3
            )

            result_items = []

            for rank, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"\n{rank}. "
                    f"{result['source']} "
                    f"#{result['chunk_id']}"
                )

                print(
                    f"Similarity: {result['score']:.4f}"
                )

                print(
                    result["text"][:240].replace(
                        "\n", " "
                    )
                )

                result_items.append(result)

            size_reports.append({

                "question": question,

                "paraphrase": is_paraphrase,

                "results": result_items

            })

        query_reports[str(size)] = size_reports

    save_json(
        "hasil_query.json",
        query_reports
    )

    # --------------------------------------------------------
    # LANGKAH 8: EVALUASI HIT@K
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("LANGKAH 8: EVALUASI HIT@K")
    print("=" * 60)

    reports = {}

    for size in TEST_SIZES:

        print(
            f"\nEVALUASI CHUNK SIZE {size}"
        )

        reports[str(size)] = {}

        for k in TOP_K_VALUES:

            result = hit_at_k(
                tests,
                collections[size],
                indexes[size],
                model,
                k
            )

            reports[str(size)][str(k)] = result

            print(
                f"Hit@{k}: "
                f"{result['hits']}/"
                f"{result['total']} "
                f"= {result['hit_at_k']:.3f}"
            )

    save_json(
        "hasil_evaluasi.json",
        reports
    )

    # --------------------------------------------------------
    # RINGKASAN JUMLAH CHUNK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RINGKASAN JUMLAH CHUNK")
    print("=" * 60)

    summary = {}

    for size, rows in collections.items():

        per_file = {}

        for row in rows:

            source = row["source"]

            per_file[source] = (
                per_file.get(source, 0) + 1
            )

        summary[str(size)] = {

            "total_chunk": len(rows),

            "per_file": per_file

        }

        print(
            f"\nChunk size {size}: "
            f"{len(rows)} chunk"
        )

        for filename, count in per_file.items():

            print(
                f"  {filename}: {count} chunk"
            )

    save_json(
        "ringkasan_chunk.json",
        summary
    )

    # --------------------------------------------------------
    # SELESAI
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EKSPERIMEN RAG SELESAI")
    print("=" * 60)

    print("\nSeluruh hasil tersimpan di folder hasil/:")

    for file in sorted(OUT.iterdir()):

        print(" -", file.name)

    print("\nPeriksa hasil sebelum membuat kesimpulan.")


if __name__ == "__main__":
    main()