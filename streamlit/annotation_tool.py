import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import os

# Config
DATA_PATH = "validated_data/validated_cleaned.parquet"
ANNOTATION_FILE = "annotations.csv"
SAMPLE_SIZE = 200
SAVE_INTERVAL = 10

st.set_page_config(page_title="Flight Annotation Tool", layout="wide")
st.title("✈️ Flight Annotation Tool (Ultra-Fast + 3D)")

@st.cache_data
def load_data(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

# Load data
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

if not {"flight_id", "lat", "lon", "alt"}.issubset(df.columns):
    st.error("Dataset must contain: flight_id, lat, lon, alt")
    st.stop()

# Init state
if "remaining_ids" not in st.session_state:
    if os.path.exists(ANNOTATION_FILE):
        annotations = pd.read_csv(ANNOTATION_FILE)
    else:
        annotations = pd.DataFrame(columns=["flight_id", "label"])

    annotated_ids = set(annotations["flight_id"])
    flight_ids = df["flight_id"].unique().tolist()
    remaining_ids = [fid for fid in flight_ids if fid not in annotated_ids]

    if len(remaining_ids) > SAMPLE_SIZE:
        remaining_ids = random.sample(remaining_ids, SAMPLE_SIZE)

    st.session_state.remaining_ids = remaining_ids
    st.session_state.annotations = annotations
    st.session_state.count = 0

remaining_ids = st.session_state.remaining_ids
annotations = st.session_state.annotations

if not remaining_ids:
    st.success("All selected flights annotated!")
    st.stop()

# Sidebar progress
st.sidebar.write(f"Flights left: **{len(remaining_ids)}**")
st.sidebar.write(f"Labeled: **{len(annotations)}**")

# Current flight
current_id = remaining_ids[0]
flight_data = df[df["flight_id"] == current_id]

# 3D Plot
fig = go.Figure(data=[go.Scatter3d(
    x=flight_data["lon"], 
    y=flight_data["lat"], 
    z=flight_data["alt"], 
    mode='lines+markers',
    marker=dict(size=4, color='blue'),
    line=dict(color='blue')
)])
fig.update_layout(
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title="Altitude"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Save function
def save_annotation(label):
    new_row = pd.DataFrame({"flight_id": [current_id], "label": [label]})
    st.session_state.annotations = pd.concat([annotations, new_row], ignore_index=True)
    remaining_ids.pop(0)
    st.session_state.count += 1

    if st.session_state.count >= SAVE_INTERVAL or not remaining_ids:
        st.session_state.annotations.to_csv(ANNOTATION_FILE, index=False)
        st.session_state.count = 0

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Valid (V)"):
        save_annotation("valid")
with col2:
    if st.button("Invalid (I)"):
        save_annotation("invalid")
with col3:
    if st.button("Skip (S)"):
        remaining_ids.append(remaining_ids.pop(0))

st.write("Use keyboard shortcuts: V = Valid, I = Invalid, S = Skip")