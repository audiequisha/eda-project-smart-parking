import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="AI Live Prediction (BiDir LSTM)", page_icon="🧠", layout="wide")
sns.set_theme(style="whitegrid")

# ==========================================
# 1. Custom Layer & Model Loader
# ==========================================
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.score = tf.keras.layers.Dense(1, name='score')

    def call(self, x):
        e = self.score(x)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

@st.cache_resource
def load_ml_assets():
    base_dir = Path("models")
    
    with open(base_dir / "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(base_dir / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    with open(base_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
        
    model = tf.keras.models.load_model(str(base_dir / "best_model.keras"), custom_objects={
        'TemporalAttention': TemporalAttention,
        'Orthogonal': tf.keras.initializers.Orthogonal,
        'GlorotUniform': tf.keras.initializers.GlorotUniform,
        'Zeros': tf.keras.initializers.Zeros,
        'Ones': tf.keras.initializers.Ones
    }, compile=False)
    
    return model, scaler_X, scaler_y, feature_cols

try:
    model, scaler_X, scaler_y, feature_cols = load_ml_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model/scaler AI. Pastikan folder 'models' berisi model Keras dan Scaler. Error: {e}")
    assets_loaded = False

# ==========================================
# 2. FEATURE ENGINEERING (Reimplementation)
# ==========================================
def build_features_for_inference(df_window, feature_cols):
    """Reimplementasi logika preprocessing persis seperti di API produksi"""
    df = df_window.copy()
    
    # Mapping weather
    weather_map = {'S': 0, 'C': 1, 'R': 2, 'SUNNY': 0, 'OVERCAST': 1, 'RAINY': 2, 'O': 1}
    if 'weather' in df.columns:
        df['weather_encoded'] = df['weather'].map(weather_map).fillna(0)
    else:
        df['weather_encoded'] = 0
        
    # Asumsikan day_of_week dari 0-6 (Senin-Minggu)
    if 'datetime' in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek
    elif 'day_of_week' in df.columns:
        df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0)
        
    if 'hour' not in df.columns and 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour
        
    # Cyclical Time Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Peak/Rush Hour Flags
    df['is_morning_peak'] = df['hour'].between(8, 11).astype(int)
    df['is_evening_peak'] = df['hour'].between(16, 19).astype(int)
    df['is_rush_hour']    = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['is_weekend']      = (df['day_of_week'] >= 5).astype(int)

    # Lag & Rolling (karena df sudah merupakan sekuens waktu 18 step ke belakang)
    if 'occupancy_rate' in df.columns and df['occupancy_rate'].max() > 1:
        df['occupancy_rate'] = df['occupancy_rate'] / 100.0 # Convert to 0-1
    elif 'occupancy' in df.columns:
        df['occupancy_rate'] = df['occupancy']
        
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df['occupancy_rate'].shift(lag).fillna(0)
        
    for w in [3, 6, 12, 24, 48]:
        df[f'roll_mean_{w}'] = df['occupancy_rate'].rolling(w, min_periods=1).mean().fillna(0)
        df[f'roll_std_{w}']  = df['occupancy_rate'].rolling(w, min_periods=1).std().fillna(0)
        
    df['momentum']     = df['occupancy_rate'].diff().fillna(0)
    df['acceleration'] = df['momentum'].diff().fillna(0)
    df['ema_01']       = df['occupancy_rate'].ewm(alpha=0.1).mean().fillna(0)
    df['ema_03']       = df['occupancy_rate'].ewm(alpha=0.3).mean().fillna(0)
    
    for c in feature_cols:
        if c not in df.columns: 
            df[c] = 0.0
            
    df[feature_cols] = df[feature_cols].bfill().ffill().fillna(0)
    return df

# ==========================================
# 3. UI DASHBOARD
# ==========================================
st.title("🧠 Live Inference: BiDir LSTM Model")
st.markdown("""
Halaman ini mendemonstrasikan **Live Inference** langsung menggunakan model *Deep Learning* yang sudah di-training (`best_model.keras`).
Model secara otomatis akan mengambil histori 18 observasi beruntun (sekuens waktu) dari dataset, merekayasa fitur temporal, melakukan normalisasi, dan memprediksi okupansi **30 menit ke depan** secara interaktif!
""")

@st.cache_data
def load_raw_data():
    df = pd.read_csv("dashboard_dataset.csv")
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Pastikan okupansi standar (0-1)
    if 'occupied' in df.columns and 'total_slot' in df.columns:
        df['occupancy_rate'] = df['occupied'] / df['total_slot']
    elif 'occupancy_rate' in df.columns and df['occupancy_rate'].max() > 1:
        df['occupancy_rate'] = df['occupancy_rate'] / 100.0
    return df

if assets_loaded:
    df_raw = load_raw_data()
    WINDOW_SIZE = 18
    
    st.markdown("---")
    st.subheader("1. Pilih Titik Waktu (Simulasi Live Data)")
    
    # Pilih index untuk prediksi (mulai dari WINDOW_SIZE sampai N)
    max_idx = len(df_raw) - 2
    selected_idx = st.slider("Geser *slider* untuk memilih waktu saat ini (Time Machine):", 
                             min_value=WINDOW_SIZE, max_value=max_idx, value=max_idx - 100)
    
    # Data historis (18 observasi)
    df_window = df_raw.iloc[selected_idx - WINDOW_SIZE + 1 : selected_idx + 1].copy()
    current_time = df_window['datetime'].iloc[-1]
    current_occ = df_window['occupancy_rate'].iloc[-1]
    
    # Data masa depan (Ground Truth 30 menit ke depan)
    future_row = df_raw.iloc[selected_idx + 1]
    future_time = future_row['datetime']
    actual_future_occ = future_row['occupancy_rate']
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🕒 **Waktu Saat Ini:** {current_time.strftime('%Y-%m-%d %H:%M')}\n🚗 **Okupansi Sekarang:** {current_occ*100:.1f}%")
    with col2:
        st.success(f"⏭️ **Waktu Target Prediksi (T+30M):** {future_time.strftime('%Y-%m-%d %H:%M')}")

    # ==========================================
    # 4. PROSES INFERENCE AI
    # ==========================================
    st.markdown("---")
    st.subheader("2. Eksekusi Prediksi AI")
    
    if st.button("🚀 Jalankan Inferensi Deep Learning", use_container_width=True):
        with st.spinner("Mengekstrak fitur dan menjalankan Feedforward Neural Network..."):
            # 1. Feature Engineering
            df_features = build_features_for_inference(df_window, feature_cols)
            
            # 2. Scaling
            df_scaled = df_features.copy()
            df_scaled[feature_cols] = scaler_X.transform(df_features[feature_cols])
            
            # 3. Reshaping untuk LSTM [1, window_size, features]
            seq = df_scaled[feature_cols].values.reshape(1, WINDOW_SIZE, len(feature_cols))
            
            # 4. Model Predict
            pred_scaled = model.predict(seq, verbose=0)
            
            # 5. Inverse Scaling
            pred_occ = float(np.clip(scaler_y.inverse_transform(pred_scaled).flatten()[0], 0, 1))
            
        # Tampilkan Hasil
        st.markdown("### 📊 Hasil Prediksi vs Aktual")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Keterisian Saat Ini", f"{current_occ*100:.1f}%")
        
        # Hitung selisih
        pred_delta = pred_occ - current_occ
        actual_error = pred_occ - actual_future_occ
        
        c2.metric("Prediksi AI (30 Menit)", f"{pred_occ*100:.1f}%", 
                  delta=f"{pred_delta*100:+.1f}% Fluktuasi")
        c3.metric("Kenyataan (Ground Truth)", f"{actual_future_occ*100:.1f}%",
                  delta=f"{actual_error*100:+.2f}% Margin Error", delta_color="inverse")
        
        # Plot Histori + Prediksi
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot 18 histori ke belakang
        ax.plot(df_window['datetime'], df_window['occupancy_rate'] * 100, marker='o', label="Histori Observasi (18 Interval)", color='dodgerblue')
        
        # Plot titik aktual ke depan
        ax.plot([current_time, future_time], [current_occ*100, actual_future_occ*100], marker='s', linestyle='--', color='gray', alpha=0.6, label="Aktual 30 Menit (Real)")
        
        # Plot Prediksi
        ax.plot([current_time, future_time], [current_occ*100, pred_occ*100], marker='X', markersize=10, linestyle='-', color='coral', label="Prediksi AI (BiDir LSTM)")
        
        ax.set_title("Lintasan Waktu (Time-Series Trajectory) - T-18 hingga T+1")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Tingkat Keterisian (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(rotation=45)
        ax.legend()
        st.pyplot(fig)
        
        st.info("💡 **Insight Arsitektur:** Model mengekstrak *Temporal Attention* dari 18 langkah histori untuk membaca pola kompleks. Hal ini memampukan AI memprediksi ke mana arah lonjakan okupansi secara adaptif.")
