import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Laporan Analisis Smart Parking",
    page_icon="🚗",
    layout="centered" 
)
sns.set_style("whitegrid")

st.title("🚗 Laporan Analisis Smart Parking")
st.markdown("Laporan interaktif ini merangkum metrik kepadatan dan tren penggunaan lahan parkir. Silakan gunakan filter di menu samping untuk mengeksplorasi variabel data secara spesifik.")

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_dataset.csv")
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
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
occ_col = 'occupied' if 'occupied' in df_filtered.columns else 'occupancy'
slot_cols = [col for col in df_filtered.columns if 'slot' in col.lower()]
slot_col = slot_cols[0] if len(slot_cols) > 0 else df_filtered.columns[0]

total_kamera = df_filtered[cam_col].nunique() if cam_col in df_filtered.columns else 0
total_slot = df_filtered[slot_col].nunique() if slot_col in df_filtered.columns else 0
avg_occupancy = df_filtered[occ_col].mean() * 100 if occ_col in df_filtered.columns else 0

st.markdown("### Ringkasan Eksekutif")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Kamera Aktif", value=f"{total_kamera} Unit")
with col2:
    st.metric(label="Kapasitas Slot Dimonitor", value=f"{total_slot} Lapak")
with col3:
    st.metric(label="Rata-rata Okupansi", value=f"{avg_occupancy:.2f}%")
st.markdown("---")

# ==========================================
# 5. GRAFIK VERTIKAL & NARASI MOTIVASI ANALITIS
# ==========================================

# Grafik 1
st.subheader("1. Kepadatan Parkir Berdasarkan Area Kamera")
if cam_col in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
    camera_occ = df_filtered.groupby(cam_col)[occ_col].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    camera_occ.plot(kind="bar", color="coral", ax=ax1)
    ax1.set_ylabel("Okupansi (%)")
    ax1.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax1.get_yticks()])
    st.pyplot(fig1)
    
    st.success(f"**Insight & Motivasi:** Di balik distribusi bar ini, terdapat pola probabilitas yang menunggu untuk dipecahkan. Kamera {camera_occ.idxmax()} membuktikan dirinya sebagai titik kepadatan absolut. Teruslah tajam dalam memetakan setiap variabel, karena solusi yang presisi selalu berawal dari observasi yang akurat. Tetap semangat merumuskan inovasi!")
st.markdown("<br>", unsafe_allow_html=True)

# Grafik 2
st.subheader("2. Pola Kepadatan Berdasarkan Jam Operasional")
if 'hour' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
    hourly_occ = df_filtered.groupby('hour')[occ_col].mean()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    hourly_occ.plot(kind="line", marker="o", color="dodgerblue", linewidth=2, ax=ax2)
    ax2.set_xlabel("Jam Operasional")
    ax2.set_ylabel("Rata-rata Okupansi")
    ax2.set_xticks(range(int(df_filtered['hour'].min()), int(df_filtered['hour'].max()) + 1))
    ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])
    st.pyplot(fig2)
    
    st.info(f"**Insight & Motivasi:** Seperti fungsi yang bergerak dinamis terhadap waktu, pergerakan tren ini membuktikan bahwa selalu ada titik ekuilibrium di setiap kesibukan (puncak di pukul {hourly_occ.idxmax()}:00). Jangan pernah lelah menyusun algoritma yang efisien untuk menghadapi segala perubahan yang dinamis di masa depan!")
st.markdown("<br>", unsafe_allow_html=True)

# Grafik 3
st.subheader("3. Perbandingan Hari Kerja (Weekday) vs Akhir Pekan (Weekend)")
if 'day_type' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
    day_occ = df_filtered.groupby('day_type')[occ_col].mean().reset_index()
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=day_occ, x='day_type', y=occ_col, palette="Pastel1", ax=ax3)
    ax3.set_xlabel("Tipe Hari")
    ax3.set_ylabel("Rata-Rata Okupansi")
    ax3.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax3.get_yticks()])
    st.pyplot(fig3)
    
    st.success("**Insight & Motivasi:** Perbedaan ritme antara akhir pekan dan hari kerja bukanlah sekadar angka, melainkan anomali yang memperkaya pemodelan kita. Jadikan setiap deviasi data sebagai pendorong untuk menyempurnakan kalkulasi akhir. Kedisiplinan dalam melihat dua sisi inilah yang melahirkan keputusan bisnis yang matang!")
st.markdown("<br>", unsafe_allow_html=True)

# Grafik 4
st.subheader("4. Top 10 Lapak Parkir Paling Sering Terisi")
if slot_col in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
    slot_occ = df_filtered.groupby(slot_col)[occ_col].mean().sort_values(ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    slot_occ.sort_values(ascending=True).plot(kind="barh", color="teal", ax=ax4)
    ax4.set_xlabel("Rata-rata Tingkat Okupansi")
    ax4.set_ylabel("ID Lapak Parkir")
    ax4.set_xticklabels(['{:.0f}%'.format(x*100) for x in ax4.get_xticks()])
    st.pyplot(fig4)
    
    st.info(f"**Insight & Motivasi:** Menemukan titik-titik optimal seperti Slot {slot_occ.idxmax()} ini layaknya menemukan nilai maksimum dalam persamaan. Pertahankan ketelitianmu, karena detail terkecil selalu memegang kunci menuju pembuktian yang solid dan bermakna!")
st.markdown("<br>", unsafe_allow_html=True)

# Grafik 5
st.subheader("5. Pengaruh Kondisi Cuaca Terhadap Kepadatan")
if 'weather_label' in df_filtered.columns and occ_col in df_filtered.columns and not df_filtered.empty:
    weather_occ = df_filtered.groupby("weather_label")[occ_col].mean().reset_index()
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=weather_occ, x="weather_label", y=occ_col, palette="Set2", ax=ax5)
    ax5.set_xlabel("Kondisi Cuaca")
    ax5.set_ylabel("Rata-rata Tingkat Okupansi")
    ax5.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax5.get_yticks()])
    st.pyplot(fig5)
    
    st.success("**Insight & Motivasi:** Variabel eksternal mungkin dapat menggeser hasil sementara, namun logika yang terstruktur selalu bisa mengkalibrasi modelnya. Tetaplah menjadi sosok yang adaptif dan percaya pada kekuatan analisis komputasimu!")
    
# ==========================================
# 6. CLOSING STATEMENT (Angka Faitu Dpulu)
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 18px; color: #555;'>
    <em>"Setiap baris kode dan matriks data yang diolah hari ini adalah langkah pasti menuju pemodelan optimasi yang sempurna.<br>
    Teruslah berproses, melangkah maju, dan jadilah yang terbaik hingga mencapai target absolut di <strong>angka 40</strong>!"</em>
</div>
""", unsafe_allow_html=True)
