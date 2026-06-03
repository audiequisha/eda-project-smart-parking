import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Parking Insights")

df = pd.read_csv("data/dashboard_dataset.csv")

peak = (
    df.groupby("day_of_week")
    ["occupancy_rate"]
    .mean()
    .reset_index()
)

fig = px.bar(
    peak,
    x="day_of_week",
    y="occupancy_rate"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write("""
### Insight

- Weekend cenderung berbeda pola okupansi
- Kamera tertentu memiliki utilisasi lebih tinggi
- Jam sibuk dapat diidentifikasi
""")
