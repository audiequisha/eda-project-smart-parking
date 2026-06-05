import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Dashboard Analitik SmartPark AI",
    page_icon="🚗",
    layout="wide" 
)
sns.set_theme(style="whitegrid")

st.title("🚗 Dashboard Analitik SmartPark AI")
st.markdown("""
Laporan interaktif ini merangkum *Exploratory Data Analysis* (EDA) untuk metrik tingkat keterisian (***occupancy rate***) dan tren historis penggunaan lahan parkir. 
Analisis ini menjadi landasan untuk pembentukan fitur time-series pada pemodelan prediktif SmartPark AI. 
Gunakan filter di menu samping untuk mengeksplorasi variabel secara spesifik.
""")

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_dataset.csv")
    
    # Deteksi timestamp dan ekstrak datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    if 'datetime' in df.columns:
        df['day_type'] = df['datetime'].dt.dayofweek.apply(
            lambda x: 'Weekend' if pd.notnull(x) and x >= 5 else 'Weekday'
        )
    elif 'day_of_week' in df.columns:
        df['day_of_week_num'] = pd.to_numeric(df['day_of_week'], errors='coerce')
        df['day_type'] = df['day_of_week_num'].apply(
            lambda x: 'Weekend' if pd.notnull(x) and x >= 5 else 'Weekday'
        )
        
    if 'weather_label' not in df.columns and 'weather' in df.columns:
        df['weather_label'] = df['weather'].map({
            'O': 'Mendung (Overcast)', 'R': 'Hujan (Rainy)', 'S': 'Cerah (Sunny)'
        })
        
    # Asumsikan data okupansi ada dalam bentuk 0-1, kita ubah jadi persentase
    if 'occupied' in df.columns:
        df['occupancy_rate'] = df['occupied'] * 100
    elif 'occupancy' in df.columns:
        df['occupancy_rate'] = df['occupancy'] * 100
    elif 'occupancy_rate' in df.columns and df['occupancy_rate'].max() <= 1.0:
        df['occupancy_rate'] = df['occupancy_rate'] * 100
        
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'dashboard_dataset.csv' tidak ditemukan.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Filter Parameter Data")

if 'day_type' in df.columns:
    list_hari = ["Semua Hari"] + list(df['day_type'].dropna().unique())
    selected_day_type = st.sidebar.selectbox("Pilih Tipe Hari:", list_hari)
else:
    selected_day_type = "Semua Hari"

if 'weather_label' in df.columns:
    list_cuaca = ["Semua Cuaca"] + list(df['weather_label'].dropna().unique())
    selected_weather = st.sidebar.selectbox("Pilih Kondisi Cuaca:", list_cuaca)
else:
    selected_weather = "Semua Cuaca"

if 'hour' in df.columns:
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
    min_hour = int(df['hour'].min()) if pd.notnull(df['hour'].min()) else 0
    max_hour = int(df['hour'].max()) if pd.notnull(df['hour'].max()) else 23
    selected_hours = st.sidebar.slider("Rentang Jam Operasional:", min_hour, max_hour, (min_hour, max_hour))
else:
    selected_hours = (0, 23)

cam_col = 'camera_id' if 'camera_id' in df.columns else 'camera'
if cam_col in df.columns:
    list_kamera = list(df[cam_col].dropna().unique())
    selected_cameras = st.sidebar.multiselect("Pilih ID Kamera Spesifik:", list_kamera, default=list_kamera)
else:
    selected_cameras = []

df_filtered = df.copy()
if selected_day_type != "Semua Hari" and 'day_type' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['day_type'] == selected_day_type]
if selected_weather != "Semua Cuaca" and 'weather_label' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['weather_label'] == selected_weather]
if 'hour' in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered['hour'] >= selected_hours[0]) & (df_filtered['hour'] <= selected_hours[1])]
if len(selected_cameras) > 0 and cam_col in df_filtered.columns:
    df_filtered = df_filtered[df_filtered[cam_col].isin(selected_cameras)]

# ==========================================
# 4. KARTU RINGKASAN UTAMA (KPI)
# ==========================================
occ_col = 'occupancy_rate' if 'occupancy_rate' in df_filtered.columns else df_filtered.columns[-1]
slot_cols = [col for col in df_filtered.columns if 'slot' in col.lower()]
slot_col = slot_cols[0] if len(slot_cols) > 0 else df_filtered.columns[0]

total_kamera = df_filtered[cam_col].nunique() if cam_col in df_filtered.columns else 0
total_slot = df_filtered[slot_col].nunique() if slot_col in df_filtered.columns else 0
avg_occupancy = df_filtered[occ_col].mean() if occ_col in df_filtered.columns else 0

st.markdown("### Ringkasan Eksekutif")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Kamera Aktif", value=f"{total_kamera} Unit")
with col2:
    st.metric(label="Kapasitas Slot Dimonitor", value=f"{total_slot} Lapak")
with col3:
    st.metric(label="Rata-rata Tingkat Okupansi", value=f"{avg_occupancy:.2f}%")
st.markdown("---")

# ==========================================
# 5. VISUALISASI TIME-SERIES UTAMA
# ==========================================
st.subheader("1. Pola Fluktuasi Time-Series Aktual")
if 'datetime' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
    df_ts = df_filtered.sort_values('datetime').copy()
    if len(df_ts) > 2000:
        df_ts = df_ts.sample(2000).sort_values('datetime') # Sampling if too large
    
    fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=df_ts, x='datetime', y=occ_col, hue=cam_col if total_kamera > 1 else None, alpha=0.7, ax=ax_ts)
    ax_ts.set_xlabel("Waktu (Timestamp)")
    ax_ts.set_ylabel("Okupansi Rate")
    ax_ts.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xticks(rotation=45)
    st.pyplot(fig_ts)
    
    st.info("💡 **Business Insight:** Grafik *time-series* di atas memperlihatkan fluktuasi histori yang sebenarnya. Anda bisa melihat adanya volatilitas mendadak di jam-jam tertentu. Meng-capture volatilitas (*sequence*) beruntun inilah alasan utama mengapa pendekatan algoritma LSTM atau Bidirectional LSTM digunakan untuk prediksi masa depan, karena model regresi statis tidak akan mampu membacanya.")
    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. ANALISIS DISTRIBUSI & KOMPARASI
# ==========================================
# Membagi layout menjadi 2 kolom
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("2. Kepadatan Berdasarkan Area Kamera")
    if cam_col in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
        camera_occ = df_filtered.groupby(cam_col)[occ_col].mean().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=camera_occ.index, y=camera_occ.values, palette="Blues_r", ax=ax1)
        ax1.set_ylabel("Rata-rata Okupansi Rate")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        st.pyplot(fig1)
        
    st.subheader("4. Perbandingan Hari Kerja vs Akhir Pekan")
    if 'day_type' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
        day_occ = df_filtered.groupby('day_type')[occ_col].mean().reset_index()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=day_occ, x='day_type', y=occ_col, palette="Pastel1", ax=ax3)
        ax3.set_xlabel("Tipe Hari")
        ax3.set_ylabel("Rata-Rata Okupansi Rate")
        ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
        st.pyplot(fig3)

with col_right:
    st.subheader("3. Kepadatan Berdasarkan Jam Operasional")
    if 'hour' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
        hourly_occ = df_filtered.groupby('hour')[occ_col].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(x=hourly_occ.index, y=hourly_occ.values, marker="o", color="dodgerblue", linewidth=2, ax=ax2)
        ax2.set_xlabel("Jam Operasional")
        ax2.set_ylabel("Rata-rata Okupansi Rate")
        ax2.set_xticks(range(int(df_filtered['hour'].min()), int(df_filtered['hour'].max()) + 1, 2))
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        st.pyplot(fig2)

    st.subheader("5. Pengaruh Cuaca Terhadap Kepadatan")
    if 'weather_label' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
        weather_occ = df_filtered.groupby("weather_label")[occ_col].mean().reset_index()
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=weather_occ, x="weather_label", y=occ_col, palette="Set2", ax=ax5)
        ax5.set_xlabel("Kondisi Cuaca")
        ax5.set_ylabel("Rata-rata Tingkat Okupansi Rate")
        ax5.yaxis.set_major_formatter(mtick.PercentFormatter())
        st.pyplot(fig5)

st.markdown("---")

# ==========================================
# 7. GRAFIK KORELASI (HEATMAP) & RAW DATA
# ==========================================
st.subheader("6. Matriks Korelasi Ekstensif (Fitur Rekayasa)")
st.markdown("Agar *heatmap* memiliki makna secara analitik, kami mengonversi (*encode*) tipe data kategorikal (Tipe Hari, Cuaca) menjadi bentuk *numeric* terlebih dahulu.")

# Rekayasa Fitur untuk Korelasi
numeric_df = df_filtered.copy()
if 'day_type' in numeric_df.columns:
    numeric_df['is_weekend'] = numeric_df['day_type'].apply(lambda x: 1 if x == 'Weekend' else 0)
if 'weather' in numeric_df.columns:
    # 1: Cerah, 2: Mendung, 3: Hujan
    numeric_df['weather_severity'] = numeric_df['weather'].map({'S': 1, 'O': 2, 'R': 3})

numeric_df = numeric_df.select_dtypes(include=['number'])

if not numeric_df.empty and len(numeric_df.columns) > 1:
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)
else:
    st.warning("Data numerik tidak cukup untuk menampilkan matriks korelasi.")

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("🔎 Lihat Sampel Data Mentah (Raw Data)"):
    st.dataframe(df_filtered.head(100))

# ==========================================
# 8. CLOSING STATEMENT
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 14px; color: #555;'>
    <em>Laporan Analisis Historis & Eksploratif <strong>SmartPark AI</strong>.<br>
    Model Prediksi 30 Menit telah di-deploy secara terpisah melalui Cloud Serverless.</em>
</div>
""", unsafe_allow_html=True)
