import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Parking Occupancy Dashboard",
    layout="wide"
)

# ==================
# LOAD DATA
# ==================
df = pd.read_csv("data/cnrpark_clean.csv")

# kalau timestamp belum datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==================
# SIDEBAR
# ==================
st.sidebar.title("Filter")

camera = st.sidebar.multiselect(
    "Camera",
    df["camera_id"].unique(),
    default=df["camera_id"].unique()
)

weather = st.sidebar.multiselect(
    "Weather",
    df["weather"].unique(),
    default=df["weather"].unique()
)

filtered = df[
    (df["camera_id"].isin(camera)) &
    (df["weather"].isin(weather))
]

# ==================
# HEADER
# ==================
st.title("🚗 Parking Occupancy Dashboard")

col1,col2,col3,col4 = st.columns(4)

col1.metric(
    "Total Records",
    len(filtered)
)

col2.metric(
    "Camera",
    filtered["camera_id"].nunique()
)

col3.metric(
    "Slot",
    filtered["slot_id"].nunique()
)

col4.metric(
    "Avg Occupancy",
    f"{filtered['occupied'].mean()*100:.1f}%"
)

# ==================
# OCCUPANCY PER HOUR
# ==================
hourly = (
    filtered
    .groupby("hour")["occupied"]
    .mean()
    .reset_index()
)

fig = px.line(
    hourly,
    x="hour",
    y="occupied",
    markers=True,
    title="Occupancy Rate by Hour"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================
# CAMERA ANALYSIS
# ==================
cam = (
    filtered
    .groupby("camera_id")
    ["occupied"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    cam,
    x="camera_id",
    y="occupied",
    title="Average Occupancy per Camera"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==================
# WEATHER
# ==================
weather_occ = (
    filtered
    .groupby("weather")
    ["occupied"]
    .mean()
    .reset_index()
)

fig3 = px.pie(
    weather_occ,
    names="weather",
    values="occupied",
    title="Occupancy by Weather"
)

st.plotly_chart(
    fig3,
    use_container_width=True)
