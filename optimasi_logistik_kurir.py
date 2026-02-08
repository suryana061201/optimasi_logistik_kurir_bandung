import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime # Import wajib untuk input waktu

# ==========================================================
# 1. KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Sistem Logistik AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Styling Modern
st.markdown("""
    <style>
    .main {background-color: #F0F2F6;}
    .stButton>button {
        width: 100%; border-radius: 8px; height: 50px;
        background-color: #FF4B4B; color: white; font-weight: bold; font-size: 18px;
    }
    .result-box {
        padding: 20px; border-radius: 10px; background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# 2. LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_rf_bandung.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================================
# 3. LOGIKA PREDIKSI (MURNI MACHINE LEARNING)
# ==========================================================
def predict_pure_ai(berat, harga_produk, qty, diskon, jam, kota_tujuan):
    
    # --- A. FEATURE ENGINEERING ---
    KOTA_PENJUAL = "KOTA BANDUNG"
    BANDUNG_RAYA = ['BANDUNG', 'CIMAHI', 'SUMEDANG', 'SOREANG', 'LEMBANG', 'PADALARANG']
    
    # Hitung fitur turunan
    total_bayar = max(0, harga_produk + 10000 - diskon)
    harga_per_kg = total_bayar / (berat + 0.001)
    
    kota_upper = kota_tujuan.strip().upper()
    is_same_city = 1 if kota_upper == KOTA_PENJUAL else 0
    is_bdg_area = 1 if any(k in kota_upper for k in BANDUNG_RAYA) else 0
    
    # --- B. SUSUN INPUT (URUTAN WAJIB SAMA DENGAN TRAINING) ---
    features = [
        'Berat_KG', 'Total_Bayar', 'Harga_Produk', 'Diskon_Ongkir', 
        'Qty', 'Harga_per_KG', 'Is_Same_City', 'Is_Bandung_Area', 'Jam_Pesan'
    ]
    
    input_data = pd.DataFrame([[
        berat, total_bayar, harga_produk, diskon, 
        qty, harga_per_kg, is_same_city, is_bdg_area, jam
    ]], columns=features)
    
    # --- C. PREDIKSI ---
    hasil = model.predict(input_data)[0]
    proba = model.predict_proba(input_data).max()
    
    return hasil, proba

# ==========================================================
# 4. TAMPILAN ANTARMUKA (UI)
# ==========================================================
st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
st.title("Sistem Logistik AI")
st.caption("Prediksi Layanan Kurir Menggunakan Machine Learning")
st.markdown("---")

if model is None:
    st.error("⚠️ Model 'model_rf_bandung.pkl' belum ada. Jalankan notebook training dulu!")
    st.stop()

# Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Info Barang")
    berat = st.number_input("Berat (KG)", 0.0, 1000000.0, 1.0)
    qty = st.number_input("Jumlah Pcs", 1, 10000, 1)
    harga = st.number_input("Harga (Rp)", 0, 1000000000, 50000)

with col2:
    st.subheader("📍 Info Pengiriman")
    kota = st.text_input("Kota Tujuan", "CIMAHI")
    diskon = st.number_input("Diskon Ongkir (Rp)", 0, 1000000, 0)
    
    # --- BAGIAN INI DIGANTI (DARI SLIDER JADI TIME INPUT) ---
    waktu_input = st.time_input("Jam Pesan", datetime.time(12, 00))
    jam = waktu_input.hour # Kita hanya ambil 'Jam'-nya saja (Angka 0-23)

# Tombol
st.markdown("###")
if st.button("🔮 ANALISIS SEKARANG"):
    
    with st.spinner('AI sedang menghitung probabilitas...'):
        hasil, keyakinan = predict_pure_ai(berat, harga, qty, diskon, jam, kota)
    
    st.markdown("---")
    
    # Logic Warna & Ikon
    if hasil == "Instant":
        warna = "#D4EDDA" # Hijau
        teks = "#155724"
        icon = "⚡"
    elif hasil == "Same Day":
        warna = "#FFF3CD" # Kuning
        teks = "#856404"
        icon = "🚀"
    elif hasil == "Hemat / Kargo":
        warna = "#F8D7DA" # Merah
        teks = "#721C24"
        icon = "🚛"
    else:
        warna = "#CCE5FF" # Biru
        teks = "#004085"
        icon = "📦"

    # Tampilkan Hasil
    st.markdown(f"""
    <div style="background-color: {warna}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid {teks};">
        <h4 style="color: {teks}; margin:0;">Rekomendasi AI:</h4>
        <h1 style="color: {teks}; margin: 10px 0;">{icon} {hasil}</h1>
        <p style="color: {teks}; margin-bottom: 0;">Tingkat Keyakinan: <b>{keyakinan*100:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)