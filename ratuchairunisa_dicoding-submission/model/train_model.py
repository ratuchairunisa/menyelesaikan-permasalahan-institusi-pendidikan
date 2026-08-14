"""
Training script - Model prediksi dropout mahasiswa.
Menghasilkan model/dropout_model.pkl siap pakai untuk prototype Streamlit.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

df = pd.read_csv('data/data_raw.csv', sep=';')

# Target: gabungkan ke masalah biner "berisiko dropout" vs "tidak" agar aktionable,
# namun tetap simpan target 3-kelas asli untuk pelaporan.
le_status = LabelEncoder()
df['Status_encoded'] = le_status.fit_transform(df['Status'])  # Dropout, Enrolled, Graduate

feature_cols = [c for c in df.columns if c not in ['Status', 'Status_encoded']]
X = df[feature_cols]
y = df['Status_encoded']

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
print("Accuracy:", round(acc, 4))
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
