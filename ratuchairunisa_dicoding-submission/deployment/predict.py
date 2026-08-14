"""
Prototype Sistem Machine Learning — Prediksi Dropout Mahasiswa
Jaya Jaya Institut

Cara menjalankan lokal:
    pip install -r requirements.txt
    streamlit run deployment/predict.py

Cara deploy ke Streamlit Community Cloud:
    1. Push seluruh folder project ini ke repository GitHub publik.
    2. Buka https://share.streamlit.io, login dengan akun GitHub.
    3. Klik "New app", pilih repo ini, branch "main", dan file path
       "deployment/predict.py".
    4. Klik "Deploy". Streamlit Cloud akan otomatis membaca requirements.txt
       yang berada di root folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa — Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "dropout_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, scaler, label_encoder, feature_cols


model, scaler, label_encoder, feature_cols = load_artifacts()

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.caption("Jaya Jaya Institut — Prototype Machine Learning untuk deteksi dini mahasiswa berisiko dropout")

st.markdown(
    "Isi data mahasiswa pada form di bawah ini, lalu klik **Prediksi** untuk "
    "melihat status yang diprediksi (Dropout / Enrolled / Graduate) beserta "
    "tingkat keyakinan model."
)

tab1, tab2 = st.tabs(["🔮 Prediksi Individu", "📁 Prediksi Massal (Upload CSV)"])

# ---------------------------------------------------------------------------
# TAB 1 — Prediksi individu lewat form
# ---------------------------------------------------------------------------
with tab1:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Data Diri & Admisi")
            marital_status = st.selectbox("Status Pernikahan (kode)", list(range(1, 7)), index=0)
            application_mode = st.number_input("Kode Mode Aplikasi", min_value=1, max_value=60, value=1)
            application_order = st.number_input("Urutan Pilihan Aplikasi", min_value=0, max_value=9, value=1)
            course = st.number_input("Kode Program Studi", min_value=1, value=9254)
            attendance = st.selectbox("Waktu Kuliah", [1, 0], format_func=lambda x: "Siang" if x == 1 else "Malam")
            prev_qualification = st.number_input("Kode Kualifikasi Sebelumnya", min_value=1, value=1)
            prev_qualification_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 130.0)
            nationality = st.number_input("Kode Kewarganegaraan", min_value=1, value=1)
            mother_qualification = st.number_input("Kode Kualifikasi Ibu", min_value=1, value=1)
            father_qualification = st.number_input("Kode Kualifikasi Ayah", min_value=1, value=1)

        with col2:
            st.subheader("Latar Belakang Keluarga & Status")
            mother_occupation = st.number_input("Kode Pekerjaan Ibu", min_value=0, value=5)
            father_occupation = st.number_input("Kode Pekerjaan Ayah", min_value=0, value=5)
            admission_grade = st.slider("Nilai Admisi", 0.0, 200.0, 130.0)
            displaced = st.selectbox("Mahasiswa Pindahan (Displaced)", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            special_needs = st.selectbox("Kebutuhan Khusus", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            debtor = st.selectbox("Memiliki Tunggakan (Debtor)", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            tuition_up_to_date = st.selectbox("Uang Kuliah Lunas Tepat Waktu", [1, 0], format_func=lambda x: "Ya" if x else "Tidak")
            gender = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
            scholarship = st.selectbox("Penerima Beasiswa", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
            age = st.slider("Usia saat Masuk Kuliah", 17, 70, 20)
            international = st.selectbox("Mahasiswa Internasional", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")

        with col3:
            st.subheader("Performa Akademik")
            units_1_credited = st.number_input("SKS Diakui Sem-1", min_value=0, value=0)
            units_1_enrolled = st.number_input("SKS Diambil Sem-1", min_value=0, value=6)
            units_1_eval = st.number_input("Jumlah Evaluasi Sem-1", min_value=0, value=6)
            units_1_approved = st.number_input("SKS Lulus Sem-1", min_value=0, value=5)
            units_1_grade = st.slider("Rata-rata Nilai Sem-1", 0.0, 20.0, 12.0)
            units_1_no_eval = st.number_input("Tanpa Evaluasi Sem-1", min_value=0, value=0)
            units_2_credited = st.number_input("SKS Diakui Sem-2", min_value=0, value=0)
            units_2_enrolled = st.number_input("SKS Diambil Sem-2", min_value=0, value=6)
            units_2_eval = st.number_input("Jumlah Evaluasi Sem-2", min_value=0, value=6)
            units_2_approved = st.number_input("SKS Lulus Sem-2", min_value=0, value=5)
            units_2_grade = st.slider("Rata-rata Nilai Sem-2", 0.0, 20.0, 12.0)
            units_2_no_eval = st.number_input("Tanpa Evaluasi Sem-2", min_value=0, value=0)

        st.subheader("Indikator Makroekonomi (saat pendaftaran)")
        col4, col5, col6 = st.columns(3)
        with col4:
            unemployment = st.slider("Tingkat Pengangguran (%)", 0.0, 20.0, 10.8)
        with col5:
            inflation = st.slider("Tingkat Inflasi (%)", -5.0, 10.0, 1.4)
        with col6:
            gdp = st.slider("GDP", -5.0, 5.0, 1.74)

        submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

    if submitted:
        input_dict = {
            'Marital_status': marital_status, 'Application_mode': application_mode,
            'Application_order': application_order, 'Course': course,
            'Daytime_evening_attendance': attendance, 'Previous_qualification': prev_qualification,
            'Previous_qualification_grade': prev_qualification_grade, 'Nacionality': nationality,
            'Mothers_qualification': mother_qualification, 'Fathers_qualification': father_qualification,
            'Mothers_occupation': mother_occupation, 'Fathers_occupation': father_occupation,
            'Admission_grade': admission_grade, 'Displaced': displaced,
            'Educational_special_needs': special_needs, 'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_up_to_date, 'Gender': gender,
            'Scholarship_holder': scholarship, 'Age_at_enrollment': age,
            'International': international,
            'Curricular_units_1st_sem_credited': units_1_credited,
            'Curricular_units_1st_sem_enrolled': units_1_enrolled,
            'Curricular_units_1st_sem_evaluations': units_1_eval,
            'Curricular_units_1st_sem_approved': units_1_approved,
            'Curricular_units_1st_sem_grade': units_1_grade,
            'Curricular_units_1st_sem_without_evaluations': units_1_no_eval,
            'Curricular_units_2nd_sem_credited': units_2_credited,
            'Curricular_units_2nd_sem_enrolled': units_2_enrolled,
            'Curricular_units_2nd_sem_evaluations': units_2_eval,
            'Curricular_units_2nd_sem_approved': units_2_approved,
            'Curricular_units_2nd_sem_grade': units_2_grade,
            'Curricular_units_2nd_sem_without_evaluations': units_2_no_eval,
            'Unemployment_rate': unemployment, 'Inflation_rate': inflation, 'GDP': gdp,
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        if pred_label == "Dropout":
            st.error(f"⚠️ Status Diprediksi: **{pred_label}**")
        elif pred_label == "Enrolled":
            st.warning(f"🟡 Status Diprediksi: **{pred_label}**")
        else:
            st.success(f"✅ Status Diprediksi: **{pred_label}**")

        proba_df = pd.DataFrame({
            "Status": label_encoder.classes_,
            "Probabilitas": proba
        }).sort_values("Probabilitas", ascending=False)
        st.bar_chart(proba_df.set_index("Status"))

        if pred_label == "Dropout":
            st.info(
                "💡 **Rekomendasi:** Mahasiswa ini terindikasi berisiko tinggi dropout. "
                "Disarankan untuk segera dijadwalkan sesi bimbingan akademik/konseling, "
                "dan dipantau progres SKS serta status pembayaran uang kuliahnya."
            )

# ---------------------------------------------------------------------------
# TAB 2 — Prediksi massal
# ---------------------------------------------------------------------------
with tab2:
    st.markdown(
        "Unggah berkas CSV dengan kolom yang sama seperti data mentah "
        "(gunakan delimiter `;`, sesuai format dataset asli) untuk memprediksi "
        "banyak mahasiswa sekaligus."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, sep=';')
            missing_cols = [c for c in feature_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Kolom berikut hilang dari berkas yang diunggah: {missing_cols}")
            else:
                X_batch = batch_df[feature_cols]
                X_batch_scaled = scaler.transform(X_batch)
                preds = model.predict(X_batch_scaled)
                pred_labels = label_encoder.inverse_transform(preds)
                batch_df['Predicted_Status'] = pred_labels

                st.success(f"Berhasil memprediksi {len(batch_df)} baris data.")
                st.dataframe(batch_df[['Predicted_Status']].value_counts().rename("Jumlah"))
                st.dataframe(batch_df)

                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Unduh Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_dropout.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca berkas: {e}")

st.divider()
st.caption(
    "Model: Random Forest Classifier | Akurasi ≈ 76% | Macro F1 ≈ 0.70 | "
    "Dibangun untuk Jaya Jaya Institut sebagai bagian dari sistem deteksi dini dropout mahasiswa."
)
