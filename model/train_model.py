"""
Training script - Model prediksi status akhir mahasiswa (Dropout vs Graduate).

PENTING: Model ini HANYA dilatih menggunakan data mahasiswa berstatus
'Dropout' dan 'Graduate'. Data berstatus 'Enrolled' TIDAK dilibatkan dalam
proses training, karena mahasiswa Enrolled belum memiliki label akhir
(outcome-nya belum diketahui) -- melibatkan mereka sebagai data latih akan
membuat target menjadi tidak jelas dan menurunkan validitas model.

Data Enrolled tetap bisa dimanfaatkan, tapi hanya pada tahap INFERENSI/PREDIKSI
(lihat bagian akhir script ini) -- untuk memprediksi kemungkinan status akhir
mereka nanti, bukan sebagai data latih.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

df_full = pd.read_csv('data/data_raw.csv', sep=';')

# --- Filter: hanya Dropout & Graduate yang dipakai untuk training ---------
df_model = df_full[df_full['Status'].isin(['Dropout', 'Graduate'])].copy()
print("Baris untuk training (Dropout + Graduate saja):", len(df_model))
print(df_model['Status'].value_counts())

# Data Enrolled disisihkan (TIDAK ikut training), hanya untuk contoh inferensi
df_enrolled = df_full[df_full['Status'] == 'Enrolled'].copy()

le_status = LabelEncoder()
df_model['Status_encoded'] = le_status.fit_transform(df_model['Status'])  # Dropout=0, Graduate=1
print("\nMapping label:", dict(zip(le_status.classes_, le_status.transform(le_status.classes_))))

feature_cols = [c for c in df_model.columns if c not in ['Status', 'Status_encoded']]
X = df_model[feature_cols]
y = df_model['Status_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300, max_depth=14, min_samples_split=5,
    min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print("\nAccuracy:", round(acc, 4))
print("Macro F1:", round(f1, 4))
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 fitur penting:")
print(importances.head(10))

# Simpan artefak
joblib.dump(model, 'model/dropout_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le_status, 'model/label_encoder.pkl')
joblib.dump(feature_cols, 'model/feature_columns.pkl')
importances.to_csv('model/feature_importance.csv')
print("\nModel & artefak tersimpan di folder model/")

# ---------------------------------------------------------------------------
# CONTOH INFERENSI PADA DATA ENROLLED (opsional)
# Data Enrolled TIDAK dipakai untuk training/evaluasi di atas. Di sini kita
# hanya mendemonstrasikan bagaimana model dipakai untuk memprediksi
# kemungkinan status akhir mahasiswa yang saat ini masih Enrolled.
# ---------------------------------------------------------------------------
if len(df_enrolled) > 0:
    X_enrolled = df_enrolled[feature_cols]
    X_enrolled_scaled = scaler.transform(X_enrolled)
    pred_enrolled = model.predict(X_enrolled_scaled)
    pred_label_enrolled = le_status.inverse_transform(pred_enrolled)
    proba_enrolled = model.predict_proba(X_enrolled_scaled)

    result_enrolled = df_enrolled[['Status']].copy()
    result_enrolled['Predicted_Final_Status'] = pred_label_enrolled
    result_enrolled['Probability_Dropout'] = proba_enrolled[:, list(le_status.classes_).index('Dropout')]
    result_enrolled['Probability_Graduate'] = proba_enrolled[:, list(le_status.classes_).index('Graduate')]

    print(f"\nContoh inferensi pada {len(df_enrolled)} mahasiswa Enrolled:")
    print(result_enrolled['Predicted_Final_Status'].value_counts())
    result_enrolled.to_csv('model/enrolled_inference_example.csv', index=False)
    print("Hasil contoh inferensi disimpan di model/enrolled_inference_example.csv")
