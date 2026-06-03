import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Smart Parking Analytics Dashboard",
    page_icon="🚗",
    layout="wide"
)
sns.set_style("whitegrid")

st.title("🚗 Smart Parking Executive Dashboard")
st.markdown("Analisis optimasi utilitas lahan parkir berbasis observasi sensor kamera.")

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_dataset.csv")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if 'weather_label' not in df.columns and 'weather' in df.columns:
        df['weather_label'] = df['weather'].map({
            'O': 'Mendung (Overcast)', 'R': 'Hujan (Rainy)', 'S': 'Cerah (Sunny)'
        })
    if 'day_type' not in df.columns and 'datetime' in df.columns:
        df['day_type'] = df['datetime'].dt.dayofweek.apply(
            lambda x: 'Weekend' if pd.notnull(x) and x >= 5 else 'Weekday'
        )
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'dashboard_dataset.csv' tidak ditemukan.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONTROLS (FILTER YANG DITINGKATKAN)
# ==========================================
st.sidebar.header("⚙️ Filter Eksplorasi Data")

# Filter 1: Tipe Hari
list_hari = ["Semua Hari"] + list(df['day_type'].dropna().unique())
selected_day_type = st.sidebar.selectbox("Pilih Tipe Hari:", list_hari)

# Filter 2: Kondisi Cuaca
list_cuaca = ["Semua Cuaca"] + list(df['weather_label'].dropna().unique())
selected_weather = st.sidebar.selectbox("Pilih Kondisi Cuaca:", list_cuaca)

# Filter 3: Rentang Jam (Slider Interaktif)
if 'hour' in df.columns:
    min_hour, max_hour = int(df['hour'].min()), int(df['hour'].max())
    selected_hours = st.sidebar.slider("Rentang Jam Operasional:", min_hour, max_hour, (min_hour, max_hour))
else:
    selected_hours = (0, 23)

# Filter 4: Multi-select ID Kamera
cam_col = 'camera_id' if 'camera_id' in df.columns else 'camera'
list_kamera = list(df[cam_col].dropna().unique())
selected_cameras = st.sidebar.multiselect("Pilih ID Kamera Spesifik:", list_kamera, default=list_kamera)

# Aplikasi Filter ke Dataset
df_filtered = df.copy()
if selected_day_type != "Semua Hari":
    df_filtered = df_filtered[df_filtered['day_type'] == selected_day_type]
if selected_weather != "Semua Cuaca":
    df_filtered = df_filtered[df_filtered['weather_label'] == selected_weather]
if 'hour' in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered['hour'] >= selected_hours[0]) & (df_filtered['hour'] <= selected_hours[1])]
df_filtered = df_filtered[df_filtered[cam_col].isin(selected_cameras)]

# ==========================================
# 4. KARTU RINGKASAN UTAMA (KPI METRICS)
# ==========================================
occ_col = 'occupied' if 'occupied' in df_filtered.columns else 'occupancy'
slot_cols = [col for col in df_filtered.columns if 'slot' in col.lower()]
slot_col = slot_cols[0] if len(slot_cols) > 0 else df_filtered.columns[0]

total_kamera = df_filtered[cam_col].nunique()
total_slot = df_filtered[slot_col].nunique()
avg_occupancy = df_filtered[occ_col].mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Kamera Aktif", value=f"{total_kamera} Unit")
    st.caption("Cakupan infrastruktur sensor pemantau di area parkir.")
with col2:
    st.metric(label="Total Slot Parkir Dimonitor", value=f"{total_slot} Lapak")
    st.caption("Kapasitas ruang parkir maksimal yang terdata.")
with col3:
    st.metric(label="Rata-rata Tingkat Okupansi", value=f"{avg_occupancy:.2f}%")
    st.caption("Persentase produktivitas utilitas pengisian lahan parkir.")

st.markdown("---")

# ==========================================
# 5. BARIS GRAFIK 1: ANALISIS KAMERA & WAKTU
# ==========================================
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader("📊 Kepadatan Parkir per Area Kamera")
    camera_occ = df_filtered.groupby(cam_col)[occ_col].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    camera_occ.plot(kind="bar", color="coral", ax=ax1)
    ax1.set_ylabel("Okupansi (%)")
    ax1.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax1.get_yticks()])
    st.pyplot(fig1)
    
    with st.expander("💡 Narasi Data & Insight"):
        st.write(f"Grafik ini mengidentifikasi area kamera mana yang paling sering penuh. Saat ini, **Kamera {camera_occ.idxmax()}** memiliki tingkat okupansi tertinggi, menjadikannya titik paling padat. Manajemen dapat menggunakan data ini untuk merelokasi petugas pengawas ke area yang paling kritis terisi.")

with g_col2:
    st.subheader("📈 Pola Tren Okupansi Berdasarkan Jam")
    if 'hour' in df_filtered.columns:
        hourly_occ = df_filtered.groupby('hour')[occ_col].mean()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        hourly_occ.plot(kind="line", marker="o", color="dodgerblue", linewidth=2, ax=ax2)
        ax2.set_xlabel("Jam Operasional")
        ax2.set_xticks(range(int(df['hour'].min()), int(df['hour'].max())+1))
        ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])
        st.pyplot(fig2)
        
        with st.expander("💡 Narasi Data & Insight"):
            st.write(f"Grafik garis memetakan jam sibuk (*peak hours*). Lonjakan grafik menandakan waktu kedatangan kendaraan tertinggi. Ini mempermudah penentuan kebijakan penarifan dinamis (*dynamic pricing*) atau penjadwalan pembersihan area saat okupansi sedang menyentuh titik terendah.")

# ==========================================
# 6. BARIS GRAFIK 2: GRAFIK BARU (WEEKDAY VS WEEKEND) & TOP SLOT
# ==========================================
g_col3, g_col4 = st.columns(2)

with g_col3:
    st.subheader("🗓️ Kepadatan: Hari Kerja (Weekday) vs Akhir Pekan (Weekend)")
    if 'day_type' in df_filtered.columns:
        day_occ = df_filtered.groupby('day_type')[occ_col].mean().reset_index()
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=day_occ, x='day_type', y=occ_col, palette="Pastel1", ax=ax3)
        ax3.set_ylabel("Rata-Rata Okupansi")
        ax3.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax3.get_yticks()])
        st.pyplot(fig3)
        
        with st.expander("💡 Narasi Data & Insight"):
            st.write("Grafik ini membandingkan perilaku parkir antara hari kerja dan hari libur. Perbedaan tinggi batang menunjukkan apakah lahan parkir ini lebih didominasi oleh mobilitas harian pekerja kantor (*weekday*) atau pengunjung rekreasi (*weekend*).")

with g_col4:
    st.subheader("🏆 Top 10 Slot Parkir Paling Padat")
    slot_occ = df_filtered.groupby(slot_col)[occ_col].mean().sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    slot_occ.sort_values(ascending=True).plot(kind="barh", color="teal", ax=ax4)
    ax4.set_xticklabels(['{:.0f}%'.format(x*100) for x in ax4.get_xticks()])
    st.pyplot(fig4)
    
    with st.expander("💡 Narasi Data & Insight"):
        st.write(f"Menampilkan peringkat mikro 10 lapak parkir spesifik yang paling disukai pengemudi. Slot **{slot_occ.idxmax()}** menempati urutan pertama. Informasi ini sangat berguna bagi tim teknis pemeliharaan jalan untuk memantau tingkat keausan permukaan aspal di slot-slot favorit tersebut.")

# ==========================================
# 7. BARIS GRAFIK 3: FAKTOR CUACA
# ==========================================
st.markdown("---")
st.subheader("🌤️ Pengaruh Faktor Cuaca Terhadap Okupansi Lahan Parkir")
if 'weather_label' in df_filtered.columns:
    weather_occ = df_filtered.groupby("weather_label")[occ_col].mean().reset_index()
    fig5, ax5 = plt.subplots(figsize=(15, 4))
    sns.barplot(data=weather_occ, x="weather_label", y=occ_col, palette="Set2", ax=ax5)
    ax5.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax5.get_yticks()])
    st.pyplot(fig5)
    
    st.info("💡 **Narasi Analisis Cuaca:** Grafik di atas menganalisis dampak kondisi eksternal (cuaca) terhadap keputusan pengendara untuk memarkirkan kendaraannya di area terbuka. Berguna untuk memprediksi penurunan volume kendaraan saat kondisi hujan melanda.")
