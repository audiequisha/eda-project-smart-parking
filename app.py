import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE
# ==========================================
st.set_page_config(
    page_title="Smart Parking Occupancy Dashboard",
    page_icon="🚗",
    layout="wide"
)

sns.set_style("whitegrid")

# Title Dashboard
st.title("🚗 Smart Parking Real-Time Occupancy Dashboard")
st.markdown("Dashboard interaktif untuk memonitor tingkat kepadatan dan pola penggunaan lahan parkir berdasarkan data observasi kamera.")

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    # Menggunakan file dataset yang sudah bersih dari tahap EDA
    df = pd.read_csv("dashboard_dataset.csv")
    
    # 1. Memastikan kolom waktu bertipe datetime jika belum
    if 'datetime' in df.columns:
        # errors='coerce' akan mengubah format yang tidak valid menjadi NaT (Not a Time) agar tidak error
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
    # 2. Mapping cuaca jika kolom weather_label belum terbentuk
    if 'weather_label' not in df.columns and 'weather' in df.columns:
        df['weather_label'] = df['weather'].map({
            'O': 'Overcast (Mendung)', 
            'R': 'Rainy (Hujan)', 
            'S': 'Sunny (Cerah)'
        })
        
    # 3. FIX ERROR: Membuat kolom day_type dengan aman
    if 'day_type' not in df.columns:
        if 'datetime' in df.columns:
            # Mengambil angka hari secara otomatis dari datetime (0=Senin, ..., 5=Sabtu, 6=Minggu)
            # Kita gunakan pd.notnull(x) untuk menghindari error jika ada data waktu yang kosong (NaT)
            df['day_type'] = df['datetime'].dt.dayofweek.apply(
                lambda x: 'Weekend' if pd.notnull(x) and x >= 5 else 'Weekday'
            )
        elif 'day_of_week' in df.columns:
            # Opsi cadangan jika datetime tidak ada: 
            # Paksa ubah teks di day_of_week menjadi angka (numeric), yang gagal akan jadi NaN
            df['day_of_week_num'] = pd.to_numeric(df['day_of_week'], errors='coerce')
            df['day_type'] = df['day_of_week_num'].apply(
                lambda x: 'Weekend' if pd.notnull(x) and x >= 5 else 'Weekday'
            )
            
    return df
try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'dashboard_dataset.csv' tidak ditemukan. Pastikan file csv berada dalam satu folder dengan script app.py ini.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS (FILTER INTERAKTIF)
# ==========================================
st.sidebar.header("⚙️ Filter Eksplorasi")

# Filter Cuaca
list_cuaca = ["Semua"] + list(df['weather_label'].dropna().unique())
selected_weather = st.sidebar.selectbox("Pilih Kondisi Cuaca:", list_cuaca)

# Filter Tipe Hari
if 'day_type' in df.columns:
    list_hari = ["Semua"] + list(df['day_type'].unique())
    selected_day_type = st.sidebar.selectbox("Pilih Tipe Hari:", list_hari)
else:
    selected_day_type = "Semua"

# Mengaplikasikan Filter ke Dataset
df_filtered = df.copy()

if selected_weather != "Semua":
    df_filtered = df_filtered[df_filtered['weather_label'] == selected_weather]

if selected_day_type != "Semua":
    df_filtered = df_filtered[df_filtered['day_type'] == selected_day_type]


# ==========================================
# 4. KARTU RINGKASAN UTAMA (KPI METRICS)
# ==========================================
total_kamera = df_filtered['camera_id'].nunique() if 'camera_id' in df_filtered.columns else df_filtered['camera'].nunique()
total_slot = df_filtered['slot_id'].nunique()
avg_occupancy = df_filtered['occupied'].mean() * 100 if 'occupied' in df_filtered.columns else df_filtered['occupancy'].mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Kamera Aktif", value=f"{total_kamera} Kamera")
with col2:
    st.metric(label="Total Slot Parkir Dimonitor", value=f"{total_slot} Slot")
with col3:
    st.metric(label="Rata-rata Tingkat Okupansi", value=f"{avg_occupancy:.2f}%")

st.markdown("---")

# ==========================================
# 5. BARIS GRAFIK 1: KEPADATAN & TREN WAKTU
# ==========================================
graph_col1, graph_col2 = st.columns(2)

with graph_col1:
    st.subheader("📊 Kepadatan Parkir per Area Kamera")
    cam_col = 'camera_id' if 'camera_id' in df_filtered.columns else 'camera'
    occ_col = 'occupied' if 'occupied' in df_filtered.columns else 'occupancy'
    
    camera_occ = df_filtered.groupby(cam_col)[occ_col].mean().sort_values(ascending=False)
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    camera_occ.plot(kind="bar", color="coral", ax=ax1)
    ax1.set_xlabel("ID Kamera")
    ax1.set_ylabel("Tingkat Okupansi")
    ax1.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax1.get_yticks()])
    st.pyplot(fig1)

with graph_col2:
    st.subheader("📈 Pola Okupansi Berdasarkan Jam (Peak Hours)")
    if 'hour' in df_filtered.columns:
        hourly_occ = df_filtered.groupby('hour')[occ_col].mean()
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        hourly_occ.plot(kind="line", marker="o", color="dodgerblue", linewidth=2, ax=ax2)
        ax2.set_xlabel("Jam Operasional")
        ax2.set_ylabel("Rata-rata Okupansi")
        ax2.set_xticks(range(0, 24))
        ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])
        st.pyplot(fig2)
    else:
        st.info("Kolom 'hour' tidak ditemukan untuk membuat tren jam.")

# ==========================================
# 6. BARIS GRAFIK 2: TOP SLOT & FAKTOR CUACA
# ==========================================
graph_col3, graph_col4 = st.columns(2)

with graph_col3:
    st.subheader("🏆 Top 10 Slot Parkir Paling Padat")
    slot_occ = df_filtered.groupby("slot_id")[occ_col].mean().sort_values(ascending=False).head(10)
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    slot_occ.sort_values(ascending=True).plot(kind="barh", color="teal", ax=ax3)
    ax3.set_xlabel("Rata-rata Tingkat Okupansi")
    ax3.set_ylabel("ID Slot")
    ax3.set_xticklabels(['{:.0f}%'.format(x*100) for x in ax3.get_xticks()])
    st.pyplot(fig3)

with graph_col4:
    st.subheader("🌤️ Tingkat Okupansi Berdasarkan Cuaca")
    weather_occ = df_filtered.groupby("weather_label")[occ_col].mean().reset_index()
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=weather_occ, x="weather_label", y=occ_col, palette="Set2", ax=ax4)
    ax4.set_xlabel("Kondisi Cuaca")
    ax4.set_ylabel("Rata-rata Tingkat Okupansi")
    ax4.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax4.get_yticks()])
    st.pyplot(fig4)

# ==========================================
# 7. PREVIEW DATA BERSIH
# ==========================================
st.markdown("---")
st.subheader("📋 Sampel Data Terfilter")
st.dataframe(df_filtered.head(100))
