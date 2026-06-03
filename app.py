import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Smart Parking Dashboard",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("data/dashboard_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# ==========================
# SIDEBAR
# ==========================

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

# ==========================
# HEADER
# ==========================

st.title("🚗 Smart Parking Dashboard")
st.caption("CNRPark Occupancy Analytics")

# ==========================
# KPI
# ==========================

col1,col2,col3=st.columns(3)

with col1:
    st.metric(
        "Average Occupancy",
        f"{filtered['occupancy_rate'].mean()*100:.1f}%"
    )

with col2:
    st.metric(
        "Peak Occupied",
        int(filtered["occupied"].max())
    )

with col3:
    st.metric(
        "Total Observations",
        len(filtered)
    )

# ==========================
# TIME SERIES
# ==========================

st.subheader("Occupancy Trend")

trend = (
    filtered
    .groupby("timestamp")["occupancy_rate"]
    .mean()
    .reset_index()
)

fig = px.line(
    trend,
    x="timestamp",
    y="occupancy_rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# CAMERA ANALYSIS
# ==========================

left,right=st.columns(2)

with left:

    cam = (
        filtered
        .groupby("camera_id")
        ["occupancy_rate"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        cam,
        x="camera_id",
        y="occupancy_rate"
    )

    st.subheader("Occupancy by Camera")

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

with right:

    hourly = (
        filtered
        .groupby("hour")
        ["occupancy_rate"]
        .mean()
        .reset_index()
    )

    fig3 = px.line(
        hourly,
        x="hour",
        y="occupancy_rate"
    )

    st.subheader("Hourly Pattern")

    st.plotly_chart(
        fig3,
        use_container_width=True)

# ==========================
# HEATMAP
# ==========================

st.subheader("Heatmap")

pivot = (
    filtered
    .pivot_table(
        values="occupancy_rate",
        index="day_of_week",
        columns="hour",
        aggfunc="mean"
    )
)

fig4 = px.imshow(
    pivot,
    aspect="auto"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ==========================
# DATA
# ==========================

st.subheader("Raw Data")

st.dataframe(filtered)
