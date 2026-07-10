# 🚗 Prediksi Harga Mobil dengan Regresi Linier

Project Akhir Praktikum Data Mining 2026 — aplikasi web interaktif berbasis **Streamlit** untuk memprediksi harga jual mobil bekas menggunakan algoritma **Regresi Linier**.

## 👥 Anggota Kelompok

- Carlos Louis Fernando
- Afryan Dhinnar Restu Panggih
- Andik Setiawan

**Dosen Penanggung Jawab:** Agung Nugroho, S.Kom., M.Kom

## 📌 Deskripsi Project

Aplikasi ini memungkinkan pengguna untuk meng-upload dataset penjualan mobil bekas (.csv), mengeksplorasi data secara visual, melatih model regresi untuk memprediksi harga mobil, mengevaluasi performa model, dan mencoba prediksi harga secara langsung berdasarkan spesifikasi kendaraan yang diinput.

## 📊 Dataset

**Automotive Price Prediction Dataset** (Kaggle)
🔗 https://www.kaggle.com/datasets/metawave/vehicle-price-prediction

- Jumlah data: 1.000.000 baris, 20 kolom
- Kolom: `make`, `model`, `year`, `mileage`, `engine_hp`, `transmission`, `fuel_type`, `drivetrain`, `body_type`, `exterior_color`, `interior_color`, `owner_count`, `accident_history`, `seller_type`, `condition`, `trim`, `vehicle_age`, `mileage_per_year`, `brand_popularity`, `price` (target)

> File dataset asli **tidak disertakan** di repo ini karena ukurannya besar. Silakan download langsung dari link Kaggle di atas, lalu upload melalui aplikasi.

## 🤖 Algoritma

- **Regresi Linier** (Linear Regression) — algoritma utama
- **Decision Tree Regressor** — sebagai model pembanding

Evaluasi menggunakan metrik **R² (koefisien determinasi)**, **MAE (Mean Absolute Error)**, dan **RMSE (Root Mean Squared Error)**.

## ✨ Fitur Aplikasi

- 📁 Upload dataset sendiri (.csv)
- 🎛️ Pemilihan kolom target dan fitur prediktor secara interaktif
- 📈 Eksplorasi Data (EDA): distribusi harga, korelasi fitur, scatter plot
- 🧠 Pelatihan model dengan pilihan algoritma (Regresi Linier / Decision Tree)
- 📉 Evaluasi model: R², MAE, RMSE, Actual vs Predicted, Residual Plot, Koefisien Regresi
- 🔍 Uji coba prediksi harga dengan input manual

## 🛠️ Tech Stack

- Python 3
- [Streamlit](https://streamlit.io/) — antarmuka web
- [scikit-learn](https://scikit-learn.org/) — model machine learning
- pandas, numpy — pengolahan data
- matplotlib, seaborn — visualisasi

## 🚀 Cara Menjalankan

1. **Clone repository ini**
   ```bash
   git clone <link-repo-github-kalian>
   cd <nama-folder-repo>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

4. Browser akan otomatis terbuka ke `http://localhost:8501`

5. **Upload dataset** melalui sidebar (download dulu dari link Kaggle di atas), lalu pilih kolom target (`price`) dan fitur yang diinginkan.

> 💡 Belum punya dataset asli? Coba dulu pakai `sample_vehicle_price_v2.csv` yang ada di repo ini untuk melihat cara kerja aplikasi.

## 📁 Struktur File

```
├── app.py                        # Aplikasi utama Streamlit
├── requirements.txt              # Daftar dependencies
├── generate_sample_data.py       # Script generator data contoh (untuk testing)
├── sample_vehicle_price_v2.csv   # Data contoh (dummy) untuk testing
└── README.md
```

## 📈 Hasil Model

| Metrik | Nilai |
|---|---|
| R² Train | 0.894 |
| R² Test | 0.891 |
| MAE (Test) | 3,274 |
| RMSE (Test) | 4,484 |

Model menunjukkan performa yang sangat baik dengan R² Test di atas 0.80, serta selisih R² Train dan Test yang kecil menandakan model tidak mengalami overfitting.

## 📄 Lisensi

Project ini dibuat untuk keperluan akademik — Praktikum Data Mining 2026.
