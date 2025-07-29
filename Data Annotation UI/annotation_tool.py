import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import os

DATA_PATH = "validated_data/validated_cleaned.parquet"
ANNOTATION_FILE = "annotations.csv"
SAMPLE_SIZE = 200
SAVE_INTERVAL = 10  # Save after every 10 labels

st.title("Flight Annotation Tool (Fast Mode)")

# Load data only once
@st.cache_data
def load_data(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)

if not {"flight_id", "lat", "lon"}.issubset(df.columns):
    st.error("Dataset must contain: flight_id, lat, lon")
    st.stop()

# Initialize session state
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
    st.session_state.count = 0  # To track saving intervals

remaining_ids = st.session_state.remaining_ids
annotations = st.session_state.annotations

if not remaining_ids:
    st.success("All selected flights have been annotated!")
    st.stop()

# Progress info
st.write(f"Flights left: {len(remaining_ids)} | Labeled: {len(annotations)}")

# Current flight
current_id = remaining_ids[0]
flight_data = df[df["flight_id"] == current_id]

fig, ax = plt.subplots()
ax.plot(flight_data["lon"], flight_data["lat"], marker='o', color='blue', linewidth=2)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Flight ID: {current_id}")
st.pyplot(fig)

def save_annotation(label):
    new_row = pd.DataFrame({"flight_id": [current_id], "label": [label]})
    st.session_state.annotations = pd.concat([st.session_state.annotations, new_row], ignore_index=True)
    remaining_ids.pop(0)
    st.session_state.count += 1

    # Save progress every N annotations
    if st.session_state.count >= SAVE_INTERVAL or not remaining_ids:
        st.session_state.annotations.to_csv(ANNOTATION_FILE, index=False)
        st.session_state.count = 0

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Valid"):
        save_annotation("valid")
with col2:
    if st.button("Invalid"):
        save_annotation("invalid")
with col3:
    if st.button("Skip"):
        remaining_ids.append(remaining_ids.pop(0))