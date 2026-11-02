# Eksperimen RAG — Dokumen Kemendikdasmen

## Status data sumber
Folder `dokumen/` disiapkan untuk tiga PDF yang diberikan. File PDF tidak berhasil diunduh dari lingkungan penyusunan ini (koneksi ke host repositori mengalami kegagalan DNS/timeout), sehingga angka chunk, sampel acak, hasil retrieval, dan Hit@k **belum dapat diklaim sebagai hasil eksperimen nyata**. Jalankan di komputer yang dapat mengakses repositori setelah menaruh ketiga PDF di folder tersebut. Nama file perlu sama dengan nama di `pertanyaan_uji.json`, atau sesuaikan field `sources`.

Sumber: 
1. https://repositori.kemendikdasmen.go.id/33290/ — *Panduan Pembelajaran dan Asesmen PAUD, Pendidikan Dasar, dan Pendidikan Menengah edisi revisi 2025*.
2. https://repositori.kemendikdasmen.go.id/25455/ — *Panduan penggunaan rapor pendidikan untuk Dinas Pendidikan* (2022).
3. https://repositori.kemendikdasmen.go.id/25488/ — *Buku panduan capaian hasil asesmen nasional untuk satuan pendidikan* (2022).

## Menjalankan

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python rag_experiment.py
```

Model embedding multilingual akan diunduh pertama kali saat dijalankan. Pastikan koneksi internet tersedia.

## Yang dilakukan skrip
- `baca_pdf(path)`: ekstraksi teks PDF via PyMuPDF.
- `potong(teks, ukuran, overlap)`: chunk karakter; overlap 15%.
- Mencetak 500 karakter pertama PDF pertama.
- Membuat koleksi ukuran 500, 1000, 1500 dan mencatat jumlah per PDF.
- Menyimpan tiga sampel acak per ukuran di `hasil/audit_chunk_*.json` untuk inspeksi manual.
- Mengindeks dengan `paraphrase-multilingual-MiniLM-L12-v2` dan cosine similarity.
- Menampilkan 5 pertanyaan faktual dan 2 parafrase pertama dari bank uji.
- Menghitung Hit@1/3/5 pada koleksi ukuran 500 dan 1000.

## Definisi metrik dan catatan interpretasi
Hit@k dihitung sebagai proporsi pertanyaan yang memiliki setidaknya satu chunk dalam top-k yang berasal dari dokumen sumber yang diharapkan dan memuat minimal satu kata kunci jawaban. Ini adalah evaluasi retrieval berbasis keyword, bukan penilaian pemahaman generatif. Periksa `details` di JSON dan validasi manual karena sinonim, tabel, OCR, atau kata kunci yang tidak muncul persis dapat menyebabkan false negative.

`ends_mid_sentence` di audit adalah indikator heuristik berdasarkan karakter terakhir, bukan penilaian bahasa yang sempurna. Baca preview awal/akhir tiap chunk dan periksa tabel secara manual.

## Penting: validasi bank soal
Pertanyaan 1–10 disusun dari abstrak/metadata repositori yang berhasil dibaca. Pertanyaan 11–12 untuk dokumen 33290 hanya menguji cakupan jenjang dan identitas panduan berdasarkan metadata judul. Setelah PDF tersedia, cocokkan kembali kata kunci dan tambahkan/perbaiki pertanyaan jika redaksi atau isi PDF berbeda. Jangan menganggap skor Hit@k sebagai final sebelum validasi manual.
