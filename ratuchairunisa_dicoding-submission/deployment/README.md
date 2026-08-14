# Deployment — Prototype Sistem Machine Learning

Prototype ini memprediksi status mahasiswa (**Dropout / Enrolled / Graduate**) menggunakan
model Random Forest yang telah dilatih pada data performa mahasiswa Jaya Jaya Institut.

## Isi Folder
- `predict.py` — aplikasi Streamlit (form prediksi individu + prediksi massal via upload CSV)

Model dan artefak pendukung (`dropout_model.pkl`, `scaler.pkl`, `label_encoder.pkl`,
`feature_columns.pkl`) berada di folder `../model/` dan dimuat otomatis oleh `predict.py`.

## Menjalankan secara Lokal

```bash
# Dari root folder project
pip install -r requirements.txt
streamlit run deployment/predict.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## Deploy ke Streamlit Community Cloud

1. Push seluruh folder project (termasuk folder `model/`) ke repository GitHub publik.
2. Buka [share.streamlit.io](https://share.streamlit.io) → login dengan akun GitHub.
3. Klik **New app** → pilih repo ini → branch `main` → file path `deployment/predict.py`.
4. Klik **Deploy**.
5. Setelah live, salin link aplikasinya dan tempelkan pada `README.md` di root project
   (bagian *Machine Learning Prototype*).

## Cara Penggunaan Aplikasi

**Tab "Prediksi Individu":** isi form data mahasiswa (data admisi, latar belakang, dan
performa akademik semester 1 & 2), lalu klik **Prediksi** untuk melihat status yang
diprediksi beserta probabilitas tiap kelas.

**Tab "Prediksi Massal (Upload CSV)":** unggah berkas CSV dengan struktur kolom yang sama
seperti `data/data_raw.csv` (delimiter `;`) untuk memprediksi banyak mahasiswa sekaligus,
lalu unduh hasilnya dalam bentuk CSV.
