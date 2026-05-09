# Bab I - Pendahuluan (Breakdown)

> **Catatan perubahan (Update 26 April 2026)**: Latar belakang ditulis ulang total dengan alur deduktif ketat. Struktur baru mengikuti pola Fenomena → Gap Fenomena → Masalah → State of the Art → Gap Riset → Novelty → Urgensi. Justifikasi tegas ditambahkan untuk setiap pilihan penelitian (Hindia, emosi vs sentimen, Transformer vs tradisional, XLM-R vs BERT). Rumusan masalah, tujuan, manfaat, dan batasan dilengkapi paragraf pembuka.

## 1.1 Latar Belakang

### Struktur paragraf (12 paragraf)

1. **Fenomena umum** — Musik dan lirik sebagai media ekspresi emosional universal; era digital memperluas konsumsi lirik.
2. **Fenomena spesifik** — Lirik ≠ teks biasa; metafora, ironi, multi-emosi implisit.
3. **Konteks sosial + justifikasi Hindia** — Lirik Hindia reflektif, naratif, emosi berlapis, code-mixing; berbeda dari musisi pop eksplisit; 54 lagu konsisten di Spotify.
4. **Gap fenomena** — Interpretasi publik masih subjektif dan intuitif; tidak konsisten dalam skala besar.
5. **Masalah: emosi vs sentimen** — Sentimen terlalu sederhana (positif/negatif); analisis emosi lebih kaya dan sesuai untuk lirik.
6. **Masalah: NLP vs manual** — NLP memungkinkan analisis konsisten dan terukur; penerapan pada lirik Indonesia masih terbatas.
7. **State of the art: Transformer vs tradisional** — Bag-of-words gagal tangkap konteks; Transformer bidirectional dan kontekstual.
8. **State of the art: XLM-R vs BERT** — BERT monolingual; lirik Hindia code-mixing; XLM-RoBERTa multibahasa 100+ bahasa.
9. **Bukti empiris** — Agatha (BERT sentimen lirik 87%), Chan (Word2Vec+BERT rekomendasi 84%), Sjoraida (BERT ulasan film >0.8).
10. **Gap riset (aspek sosial)** — Belum ada penelitian Transformer multibahasa untuk emosi lirik Indonesia; lirik sebagai refleksi sosial generasi.
11. **Novelty + urgensi** — Fine-tuning XLM-R, klasifikasi probabilistik multi-emosi, visualisasi Streamlit.
12. **Penutup** — Judul penelitian.

### Sitasi yang digunakan

- Avandra et al. (2023), Purnama & Fadillah (2025), Syaira & Hermandra (2024), Manopo et al. (2022), Aulia & Rahmadhani (2025), Haedariah et al. (2023), Manueke et al. (2023), Azizah et al. (2025), Siwi et al. (2026), Cahyadi et al. (2024), Fardhina et al. (2025), Wiciaputra et al. (2021), Khairani et al. (2024), Agatha et al. (2023), Chan et al. (2024), Sjoraida et al. (2024)

## 1.2 Rumusan Masalah

- Paragraf pembuka menghubungkan latar belakang ke pertanyaan penelitian.
- Bagaimana kinerja model XLM-RoBERTa dalam mengklasifikasikan emosi pada teks bahasa Indonesia setelah fine-tuning menggunakan dataset emosi berlabel dari Nusacrowd, dievaluasi menggunakan accuracy, precision, recall, F1-score, dan confusion matrix?
- Bagaimana distribusi dan pola emosi yang teridentifikasi pada 54 lirik lagu Hindia setelah inferensi menggunakan model dengan kinerja terbaik?

## 1.3 Tujuan Penelitian

- Paragraf pembuka yang sejalan dengan rumusan masalah.
- Melakukan fine-tuning model XLM-RoBERTa menggunakan dataset emosi berlabel bahasa Indonesia dari Nusacrowd serta evaluasi kuantitatif.
- Menggunakan model terbaik untuk mengklasifikasikan emosi pada 54 lirik lagu Hindia dan menganalisis distribusi serta pola emosi.

## 1.4 Manfaat Penelitian

- Paragraf pembuka tentang kontribusi akademis dan praktis.
- Gambaran empiris kinerja XLM-RoBERTa sebagai referensi penelitian NLP dan SI.
- Pendekatan analisis emosi komputasional yang terstruktur, objektif, dan reproducible.
- Distribusi dan pola emosi lirik Hindia melalui visualisasi interaktif.

## 1.5 Batasan Masalah

- Paragraf pembuka tentang fokus dan keterukuran.
- Data: 54 lagu Hindia di Spotify (main artist).
- Analisis: hanya teks lirik, tanpa aspek audio.
- Model: XLM-RoBERTa fine-tuned pada Nusacrowd (supervised learning).
- Evaluasi: kuantitatif tanpa validasi anotator ahli.
- Hasil: prediksi model, bukan makna emosional absolut.
