# Menyelesaikan Permasalahan Institusi Pendidikan
## Prediksi Dropout Mahasiswa — Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000
dan telah mencetak banyak lulusan dengan reputasi baik. Namun, institusi ini menghadapi
tantangan berupa **tingginya angka dropout (mahasiswa yang tidak menyelesaikan pendidikannya)**.

Jumlah dropout yang tinggi merupakan masalah besar bagi sebuah institusi pendidikan, baik dari
sisi reputasi, efisiensi sumber daya, maupun kesempatan mahasiswa itu sendiri. Oleh karena itu,
Jaya Jaya Institut ingin **mendeteksi sedini mungkin** mahasiswa yang berpotensi dropout agar
dapat diberikan bimbingan khusus.

### Permasalahan Bisnis
1. Belum ada cara sistematis untuk mengidentifikasi mahasiswa berisiko dropout sejak dini.
2. Bimbingan khusus sulit diberikan tepat waktu tanpa adanya sistem deteksi risiko.
3. Belum ada dashboard terpusat bagi manajemen untuk memonitor performa dan faktor risiko
   dropout mahasiswa.

### Cakupan Proyek
1. Melakukan analisis data performa mahasiswa (business understanding → deployment).
2. Membangun model machine learning untuk memprediksi status mahasiswa
   (Dropout / Enrolled / Graduate).
3. Membuat dashboard bisnis untuk memonitor faktor-faktor penting terkait dropout.
4. Membangun prototype sistem prediksi berbasis Streamlit yang siap digunakan.
5. Merumuskan rekomendasi action items berbasis data.

### Persiapan

**Sumber data:** [Students' Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)
(4.424 baris data mahasiswa, 36 fitur + 1 target `Status`, tanpa missing value)

**Setup environment:**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Menjalankan notebook analisis:
```bash
jupyter notebook notebook/students_dropout_prediction.ipynb
```
---

## Business Dashboard

Dashboard dibangun menggunakan **dataset final yang sudah dibersihkan dan diberi label yang
mudah dibaca**, tersedia di `data/students_dashboard_final.csv`. Dataset ini mengganti kode
numerik (mis. kode kualifikasi, kode program studi) menjadi label yang jelas, serta menambahkan
metrik turunan (`Approval_Rate_1st_Sem`, `Approval_Rate_2nd_Sem`, `Avg_Grade_Both_Sem`) agar
mudah divisualisasikan.

Dashboard menampilkan (minimal):
- **Distribusi status mahasiswa** (Dropout / Enrolled / Graduate)
- **Tingkat kelulusan SKS per semester** dikaitkan dengan status
- **Status pembayaran UKT** vs status mahasiswa
- **Status penerima beasiswa** vs status mahasiswa
- **Usia saat masuk kuliah** vs status mahasiswa

Berkas project dashboard (Tableau Workbook) berada di `dashboard/`. Screenshot pratinjau
dashboard: `dashboard/username_dicoding-dashboard/` (lihat bagian *Catatan Submission*
di bawah mengenai cara melengkapinya).

> **Link dashboard (Tableau Public):** [Dashboard Monitoring Dropout Mahasiswa Jaya-Jaya Institute](https://public.tableau.com/views/DashboardMonitoringDropoutMahasiswaJayaJayaInstitute/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## Machine Learning Prototype

Prototype sistem machine learning dibangun menggunakan **Streamlit**, memungkinkan pengguna
(staf akademik Jaya Jaya Institut) memasukkan data seorang mahasiswa dan langsung mendapatkan
prediksi status (Dropout / Enrolled / Graduate) beserta tingkat keyakinan model, atau mengunggah
CSV untuk prediksi massal.

### Menjalankan prototype secara lokal
```bash
pip install -r requirements.txt
streamlit run deployment/predict.py
```
Aplikasi akan terbuka otomatis di `http://localhost:8501`.

### Streamlit Community Cloud
> **Link prototype (Streamlit Community Cloud):** [Prediksi Risiko Dropout Mahasiswa](https://menyelesaikan-permasalahan-institusi-pendidikan-e78uhauxqxorb7.streamlit.app/)
---

## Conclusion

Berdasarkan analisis data dan pemodelan yang dilakukan:

1. **Angka dropout di Jaya Jaya Institut cukup tinggi**, yaitu sekitar **32%** dari total
   mahasiswa dalam dataset (1.421 dari 4.424 mahasiswa), sehingga menjadi masalah yang layak
   mendapat prioritas penanganan.
2. **Performa akademik di semester-semester awal** (jumlah SKS yang lulus dan rata-rata nilai
   pada semester 1 & 2) merupakan **prediktor terkuat** terhadap risiko dropout — mahasiswa
   dengan SKS lulus rendah di semester awal (Semester 1 dan 2) jauh lebih berisiko dropout.
3. **Faktor finansial berperan signifikan.** Mahasiswa yang belum melunasi uang kuliah (UKT)
   tepat waktu memiliki proporsi dropout jauh lebih tinggi (±87%) dibanding yang lunas tepat
   waktu (±25%). Penerima beasiswa juga menunjukkan tingkat kelulusan yang lebih baik.
4. **Usia saat masuk kuliah** turut berkontribusi — mahasiswa yang masuk pada usia lebih tua
   cenderung memiliki risiko dropout/status tidak selesai yang lebih tinggi dibanding yang
   masuk pada usia standar (18-19 tahun).
5. Model **Random Forest Classifier** yang dibangun mencapai **akurasi 75,71%** dan
   **macro F1-score 70,18** dalam memprediksi status mahasiswa, sehingga layak digunakan
   sebagai alat bantu deteksi dini — dengan catatan hasil prediksi tetap perlu dikombinasikan
   dengan penilaian pihak akademik, bukan sebagai keputusan final otomatis.

Model dan dashboard pada proyek ini diharapkan membantu Jaya Jaya Institut mendeteksi
mahasiswa berisiko dropout lebih dini, sehingga bimbingan khusus bisa diberikan tepat waktu
dan angka dropout dapat ditekan.

## Rekomendasi Action Items

1. **Terapkan sistem peringatan dini (early warning system) berbasis SKS lulus semester 1.**
   Mahasiswa dengan tingkat kelulusan SKS di bawah rata-rata pada semester pertama harus
   otomatis ditandai sebagai "berisiko" dan dijadwalkan sesi bimbingan akademik dalam
   2 minggu pertama semester berikutnya.
2. **Perkuat program bantuan finansial dan kebijakan keterlambatan UKT.** Mengingat status
   pelunasan UKT sangat berkorelasi dengan dropout, institusi dapat menawarkan skema
   cicilan UKT atau dana talangan darurat bagi mahasiswa yang menunjukkan tanda-tanda
   kesulitan finansial, sebelum keterlambatan berujung pada dropout.
3. **Perluas cakupan beasiswa berbasis kebutuhan (need-based), bukan hanya prestasi**, karena
   data menunjukkan penerima beasiswa memiliki tingkat kelulusan yang lebih tinggi.
4. **Berikan pendampingan khusus (mentoring/onboarding) bagi mahasiswa yang masuk pada usia
   lebih tua** (non-fresh graduate SMA), karena kelompok ini menunjukkan variasi status yang
   lebih besar dan berpotensi membutuhkan dukungan adaptasi yang berbeda.
5. **Gunakan dashboard secara rutin (misal mingguan/bulanan)** oleh tim akademik dan
   kemahasiswaan untuk memonitor tren dropout per program studi/angkatan, sehingga
   intervensi dapat dilakukan secara proaktif, bukan reaktif.
6. **Integrasikan model prediksi ke dalam proses akademik reguler**, misalnya menjalankan
   prediksi massal (melalui fitur upload CSV pada prototype) setiap akhir semester untuk
   seluruh mahasiswa aktif, lalu menindaklanjuti mahasiswa dengan probabilitas dropout
   tertinggi terlebih dahulu.
