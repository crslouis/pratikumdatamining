"""
Streamlit App - Prediksi Harga Mobil (Car Price Regression)
Project Akhir Praktikum Data Mining 2026

Dataset: Automotive Price Prediction Dataset (Kaggle - metawave/vehicle-price-prediction)
Target: price (harga mobil)
Algoritma: Regresi Linier (+ Decision Tree Regressor sebagai pembanding)

Catatan desain: kolom fitur dipilih secara interaktif oleh pengguna karena
dataset asli punya 20 kolom dan strukturnya bisa sedikit berbeda tergantung
versi file yang diunduh dari Kaggle.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset harga mobil (.csv)", type=["csv"]
)

sample_n = st.sidebar.number_input(
    "Batasi jumlah baris diproses (0 = semua baris, gunakan untuk dataset besar)",
    min_value=0, value=50000, step=5000,
)

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.title("🚗 Prediksi Harga Mobil dengan Regresi Linier")
st.caption(
    "Project Akhir Praktikum Data Mining 2026 — "
    "Dataset: Automotive Price Prediction Dataset (Kaggle)"
)

if uploaded_file is None:
    st.info(
        "⬅️ Silakan upload file CSV dataset harga mobil melalui sidebar. "
        "Dataset harus memiliki satu kolom harga (misal `price`) dan "
        "beberapa kolom fitur numerik/kategorikal (misal `year`, `mileage`, "
        "`horsepower`, `brand`, `fuel_type`, dll)."
    )
    st.stop()

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data(file, n_rows):
    df = pd.read_csv(file)
    if n_rows and n_rows > 0 and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    return df

df = load_data(uploaded_file, sample_n)
st.success(f"Dataset berhasil dimuat: **{len(df):,} baris**, {df.shape[1]} kolom.")

# ---------------------------------------------------------------
# COLUMN SELECTION
# ---------------------------------------------------------------
all_cols = list(df.columns)
guessed_target = next(
    (c for c in all_cols if c.strip().lower() in ("price", "sale price", "selling_price")),
    all_cols[-1],
)

st.sidebar.subheader("Pilih Kolom")
target_col = st.sidebar.selectbox(
    "Kolom target (harga)", all_cols,
    index=all_cols.index(guessed_target),
)

default_features = [c for c in all_cols if c != target_col][:8]
feature_candidates = [c for c in all_cols if c != target_col]
feature_cols_selected = st.sidebar.multiselect(
    "Kolom fitur (prediktor)", feature_candidates, default=default_features,
)

algo = st.sidebar.selectbox(
    "Algoritma", ["Regresi Linier", "Decision Tree Regressor"]
)

test_size = st.sidebar.slider("Proporsi data test", 0.1, 0.4, 0.2, 0.05)

if not feature_cols_selected:
    st.warning("Pilih minimal satu kolom fitur di sidebar untuk melanjutkan.")
    st.stop()

# ---------------------------------------------------------------
# TABS
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data (EDA)", "🤖 Model & Evaluasi", "🔍 Coba Prediksi"])

# ---------------------------------------------------------------
# TAB 1: EDA
# ---------------------------------------------------------------
with tab1:
    st.subheader("Cuplikan Data")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Distribusi {target_col}")
        fig, ax = plt.subplots()
        sns.histplot(df[target_col].dropna(), kde=True, ax=ax, color="#065A82")
        st.pyplot(fig)

    with col2:
        numeric_cols = df.select_dtypes(include=np.number).columns
        if target_col in numeric_cols and len(numeric_cols) > 1:
            st.subheader("Korelasi Fitur Numerik terhadap Target")
            corr = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
            fig, ax = plt.subplots()
            corr.drop(target_col, errors="ignore").plot(kind="barh", ax=ax, color="#1C7293")
            ax.set_xlabel(f"Korelasi dengan {target_col}")
            st.pyplot(fig)

    st.subheader("Scatter Plot: Fitur Numerik vs Target")
    numeric_features = [c for c in feature_cols_selected if c in df.select_dtypes(include=np.number).columns]
    if numeric_features:
        chosen_feat = st.selectbox("Pilih fitur numerik", numeric_features)
        fig, ax = plt.subplots()
        ax.scatter(df[chosen_feat], df[target_col], alpha=0.3, s=10, color="#065A82")
        ax.set_xlabel(chosen_feat)
        ax.set_ylabel(target_col)
        st.pyplot(fig)
    else:
        st.info("Tidak ada fitur numerik yang dipilih untuk scatter plot.")

# ---------------------------------------------------------------
# PREPROCESSING (shared by tab2 & tab3)
# ---------------------------------------------------------------
work = df[feature_cols_selected + [target_col]].dropna().copy()

encoders = {}
for c in feature_cols_selected:
    if not pd.api.types.is_numeric_dtype(work[c]):
        le = LabelEncoder()
        work[c] = le.fit_transform(work[c].astype(str))
        encoders[c] = le

X = work[feature_cols_selected]
y = work[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

if algo == "Regresi Linier":
    model = LinearRegression()
else:
    model = DecisionTreeRegressor(max_depth=8, random_state=42)

model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# ---------------------------------------------------------------
# TAB 2: MODEL & EVALUASI
# ---------------------------------------------------------------
with tab2:
    st.subheader(f"Hasil Model: {algo}")
    st.write(f"**Fitur yang digunakan:** {', '.join(feature_cols_selected)}")
    st.write(f"**Target:** {target_col}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Train", f"{r2_train:.3f}")
    c2.metric("R² Test", f"{r2_test:.3f}")
    c3.metric("MAE (Test)", f"{mae_test:,.0f}")
    c4.metric("RMSE (Test)", f"{rmse_test:,.0f}")

    if r2_test >= 0.80:
        st.success("✅ Model memiliki performa sangat baik (R² Test ≥ 0.80).")
    elif r2_test >= 0.5:
        st.info("ℹ️ Model memiliki performa cukup baik (R² Test ≥ 0.50).")
    else:
        st.warning(
            "⚠️ R² Test masih rendah. Coba tambah/ganti fitur prediktor di sidebar."
        )

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_test, alpha=0.3, s=10, color="#065A82")
    lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
    ax.plot(lims, lims, color="#DD8452", linewidth=2, label="Ideal (y=x)")
    ax.set_xlabel(f"Aktual ({target_col})")
    ax.set_ylabel(f"Prediksi ({target_col})")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Residual Plot")
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots()
    ax.scatter(y_pred_test, residuals, alpha=0.3, s=10, color="#1C7293")
    ax.axhline(0, color="#DD8452", linewidth=2)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Residual (Aktual - Prediksi)")
    st.pyplot(fig)

    if algo == "Regresi Linier":
        st.subheader("Koefisien Regresi")
        coef_df = pd.DataFrame({
            "Fitur": feature_cols_selected, "Koefisien": model.coef_
        }).sort_values("Koefisien", key=abs, ascending=True)
        fig, ax = plt.subplots()
        ax.barh(coef_df["Fitur"], coef_df["Koefisien"], color="#55A868")
        st.pyplot(fig)
        st.caption(f"Intercept: {model.intercept_:,.2f}")
    else:
        st.subheader("Feature Importance")
        importance = pd.Series(model.feature_importances_, index=feature_cols_selected).sort_values()
        fig, ax = plt.subplots()
        importance.plot(kind="barh", ax=ax, color="#55A868")
        st.pyplot(fig)

# ---------------------------------------------------------------
# TAB 3: COBA PREDIKSI MANUAL
# ---------------------------------------------------------------
with tab3:
    st.subheader("Coba Prediksi dengan Input Manual")
    st.write("Masukkan nilai fitur untuk memprediksi harga mobil.")

    input_values = {}
    cols = st.columns(3)
    for i, c in enumerate(feature_cols_selected):
        with cols[i % 3]:
            if c in encoders:
                options = list(encoders[c].classes_)
                val = st.selectbox(c, options, key=f"in_{c}")
                input_values[c] = encoders[c].transform([val])[0]
            else:
                default_val = float(pd.to_numeric(df[c], errors="coerce").median())
                val = st.number_input(c, value=default_val, key=f"in_{c}")
                input_values[c] = val

    if st.button("Prediksi Harga"):
        input_df = pd.DataFrame([input_values])[feature_cols_selected]
        pred_price = model.predict(input_df)[0]
        st.success(f"Estimasi {target_col}: **{pred_price:,.2f}**")
