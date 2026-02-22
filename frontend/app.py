import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
import fastf1

# Set up page configurations
st.set_page_config(page_title="ApexHunter Dashboard", layout="wide")

# Directory where Parquet files are stored (Now points to the cleaned data lake)
# Note: Since this file is in 'frontend/', we step out one level to reach 'data_lake'
DATA_LAKE_DIR = Path("../data_lake/clean_data")

# Basic cache directory for fastf1
fastf1.Cache.enable_cache('../cache') 

# --- HEADER ---
st.title("🏎️ ApexHunter F1 Data Explorer")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Data")

# 1. Year Filter
available_years = [2023, 2024]
selected_year = st.sidebar.selectbox("Year", available_years)

# Determine the available session files for the chosen year
available_files = list(DATA_LAKE_DIR.glob(f"{selected_year}_*.parquet"))
if not available_files:
    st.error(f"No data files found for {selected_year} in {DATA_LAKE_DIR}.")
    st.stop()

# Fetch the event schedule for the selected year
@st.cache_data(show_spinner=False)
def get_event_schedule(year):
    try:
        schedule = fastf1.get_event_schedule(year)
        # Create a dictionary mapping round number to event name
        return dict(zip(schedule['RoundNumber'], schedule['EventName']))
    except Exception:
        return {}

event_names = get_event_schedule(selected_year)

# Build mapping for sessions (e.g., '1_Q', '2_R') to human-readable labels
session_options = []
file_mapping = {}

for f in available_files:
    parts = f.stem.split('_')
    if len(parts) == 3:
        year, round_num, session_type = parts
        
        # Get Event Name (e.g., 'Bahrain Grand Prix')
        event_name = event_names.get(int(round_num), "Unknown Race")
        
        label_map = {"Q": "Qualifying", "R": "Race", "Sprint": "Sprint", "SQ": "Sprint Shootout"}
        full_session = label_map.get(session_type, session_type)
        label = f"Round {round_num}: {event_name} - {full_session}"
        session_options.append(label)
        file_mapping[label] = f
    else:
        label = f.stem
        session_options.append(label)
        file_mapping[label] = f

# Sort sessions numerically by round 
def get_round_num(x):
    try:
        if "Round " in x:
            # Handle format "Round X: Event Name - Session"
            parts = x.split("Round ")[1]
            round_str = parts.split(":")[0]
            return int(round_str)
    except:
        pass
    return 999

session_options = sorted(session_options, key=get_round_num)

# 2. Session Filter
selected_session = st.sidebar.selectbox("Session", session_options)

# Load selected session data
selected_file = file_mapping[selected_session]
try:
    df = pd.read_parquet(selected_file)
except Exception as e:
    st.error(f"Error loading {selected_file.name}: {e}")
    st.stop()

# Ensure expected columns are present (graceful degradation)
expected_cols = ['Driver', 'Speed', 'RPM', 'Throttle', 'Brake', 'X', 'Y', 'Time', 'SessionTime', 'nGear']
for c in expected_cols:
    if c not in df.columns:
        if c == 'Driver':
            df['Driver'] = 'UNKNOWN'
        elif c in ['Time', 'SessionTime']:
            df[c] = pd.to_timedelta(np.arange(len(df)), unit='s')
        elif c == 'nGear':
            df['nGear'] = 8
# 3. Driver Filter
available_driver_numbers = sorted(df['Driver'].dropna().unique())

DRIVER_MAPPING = {
    '1': 'Max Verstappen', '2': 'Logan Sargeant', '3': 'Daniel Ricciardo', '4': 'Lando Norris',
    '10': 'Pierre Gasly', '11': 'Sergio Perez', '14': 'Fernando Alonso', '16': 'Charles Leclerc',
    '18': 'Lance Stroll', '20': 'Kevin Magnussen', '21': 'Nyck de Vries', '22': 'Yuki Tsunoda',
    '23': 'Alexander Albon', '24': 'Zhou Guanyu', '27': 'Nico Hulkenberg', '31': 'Esteban Ocon',
    '38': 'Oliver Bearman', '40': 'Liam Lawson', '43': 'Franco Colapinto', '44': 'Lewis Hamilton',
    '55': 'Carlos Sainz', '63': 'George Russell', '77': 'Valtteri Bottas', '81': 'Oscar Piastri'
}

# Sort driver list alphabetically by Driver Name
driver_list = []
for d in available_driver_numbers:
    name = DRIVER_MAPPING.get(d, "Unknown Driver")
    driver_list.append((name, f"{name} (#{d})"))

# Sort driver list alphabetically by Driver Name
driver_list.sort(key=lambda x: x[0])
driver_labels = [row[1] for row in driver_list]

selected_driver_label = st.sidebar.selectbox("Driver", driver_labels)
# Extract the driver number from the label
selected_driver = selected_driver_label.split(" (#")[1].replace(")", "")
selected_driver_name = selected_driver_label.split(" (#")[0]

# Beginner Mode Toggle
beginner_mode = st.sidebar.toggle("Beginner Mode", value=False)

# Labels based on mode
speed_label = "Speed (km/h)" if not beginner_mode else "Vehicle Speed"
throttle_label = "Throttle (%)" if not beginner_mode else "Gas Pedal"
brake_label = "Brake" if not beginner_mode else "Brakes Applied"
rpm_label = "RPM" if not beginner_mode else "Engine Revs"
gear_label = "nGear" if not beginner_mode else "Current Gear"

# Filter Data by Driver
df_driver = df[df['Driver'] == selected_driver].copy()

# --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("ApexHunter v1.0 | Dev: Parin Shah | ID: 23001091")

# --- MAIN DASHBOARD ---
if df_driver.empty:
    st.warning(f"No telemetry data found for driver {selected_driver_name} (#{selected_driver}).")
    st.stop()

# Info box for telemetry
with st.expander("ℹ️ Why is there a gap in the telemetry data early in the session?"):
    st.write(
        """
        **Telemetry timestamps (`SessionTime`) measure from when the official session window formally starts.**
        
        - **In a Race**, the `SessionTime` begins heavily in advance of the actual race start (often tracking buildup, pit lane opening, and the formation lap). The actual "lights out" frequently occurs ~1 hour into the official track `SessionTime`.
        - **In Qualifying**, the data begins slightly earlier before cars are released.
        
        Because cars generate no active track telemetry while sitting still in the garage during these pre-session periods, you'll see a gap or offset in the timeline before the first recorded data points appear!
        """
    )

# We will use 'Time' as the primary time column for scrubbing
time_col = 'SessionTime' if 'SessionTime' in df_driver.columns and not df_driver['SessionTime'].isnull().all() else 'Time'

if pd.api.types.is_timedelta64_dtype(df_driver[time_col]):
    # Time scrubber logic
    min_time = df_driver[time_col].dt.total_seconds().min()
    max_time = df_driver[time_col].dt.total_seconds().max()

    # User scrubs through the data
    st.markdown("### Telemetry Playback")
    scrub_time = st.slider("Scrub Session Time", min_value=float(min_time), max_value=float(max_time), value=float(max_time), format="%.1f s")
    
    df_filtered = df_driver[df_driver[time_col].dt.total_seconds() <= scrub_time].copy()
    total_seconds = scrub_time
else:
    # Fallback if Time is just integers/floats and not timedelta
    min_time = df_driver[time_col].min()
    max_time = df_driver[time_col].max()
    st.markdown("### Telemetry Playback")
    scrub_time = st.slider("Scrub Session", min_value=float(min_time), max_value=float(max_time), value=float(max_time), format="%.1f")
    df_filtered = df_driver[df_driver[time_col] <= scrub_time].copy()
    total_seconds = scrub_time

if df_filtered.empty:
    st.warning("No data in selected time range.")
    st.stop()

# Convert total_seconds into string HH:MM:SS.ms
hrs, remainder = divmod(total_seconds, 3600)
mins, secs = divmod(remainder, 60)
ms = (secs - int(secs)) * 1000
time_str = f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}.{int(ms):03d}"

# Metrics row
st.markdown("---")
col1, col2, col3 = st.columns(3)

# 1. Live Telemetry Timer
col1.metric("Session Time", time_str)

# 2. Top Speed Metric
top_speed = float(df_filtered['Speed'].max())
col3.metric("Top Speed (km/h)" if not beginner_mode else "Max Speed", f"{top_speed:.1f}")

# 3. Gamified Perfect Corner Score
# Simple mock algorithm: percentage of time the driver is NOT overlapping brake and throttle
overlap_penalty = (df_filtered['Brake'] > 0) & (df_filtered['Throttle'] > 0)
score = 100 - (overlap_penalty.sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 100
corner_score = np.clip(score, 0, 100)

with col2:
    st.metric("Perfect Corner Score", f"{corner_score:.1f}")
    st.progress(int(corner_score), text="Apex Accuracy")

st.markdown("---")

# Chart X-axis data
if pd.api.types.is_timedelta64_dtype(df_filtered[time_col]):
    x_data = df_filtered[time_col].dt.total_seconds()
else:
    x_data = df_filtered[time_col]

# --- INTERACTIVE TELEMETRY CHART ---
st.subheader(f"Telemetry Data - {selected_driver_name} (#{selected_driver})")

fig_telemetry = make_subplots(
    rows=3, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08,
    subplot_titles=(speed_label, throttle_label, brake_label)
)

hover_template = "At this moment, the car is traveling at %{customdata[0]:.1f} km/h in %{customdata[1]} gear.<extra></extra>"
custom_data = np.stack((df_filtered['Speed'], df_filtered['nGear']), axis=-1)

# Speed
trace_speed = go.Scatter(
    x=x_data, y=df_filtered['Speed'], 
    name=speed_label, line=dict(color='cyan'),
    customdata=custom_data,
    hovertemplate=hover_template
)
fig_telemetry.add_trace(trace_speed, row=1, col=1)

# Throttle
trace_throttle = go.Scatter(
    x=x_data, y=df_filtered['Throttle'], 
    name=throttle_label, line=dict(color='green'),
    customdata=custom_data,
    hovertemplate=hover_template
)
fig_telemetry.add_trace(trace_throttle, row=2, col=1)

# Brake
trace_brake = go.Scatter(
    x=x_data, y=df_filtered['Brake'], 
    name=brake_label, line=dict(color='red'),
    customdata=custom_data,
    hovertemplate=hover_template
)
fig_telemetry.add_trace(trace_brake, row=3, col=1)

fig_telemetry.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40, l=40, r=40))
fig_telemetry.update_xaxes(title_text="Time (s)", row=3, col=1)

# Render Chart cleanly
st.plotly_chart(fig_telemetry, use_container_width=True)

st.markdown("---")

# --- INTERACTIVE TRACK MAP ---
st.subheader("Traffic Light Track Map")

fig_map = px.scatter(
    df_filtered, x='X', y='Y', color='Speed',
    color_continuous_scale=['red', 'yellow', 'green'],
    labels={'X': 'X Coordinate', 'Y': 'Y Coordinate', 'Speed': speed_label},
    custom_data=['Speed', 'nGear']
)

fig_map.update_traces(hovertemplate=hover_template)

# Scale axes to maintain track shape proportions
fig_map.update_layout(
    yaxis=dict(scaleanchor="x", scaleratio=1), 
    margin=dict(t=40, b=40, l=40, r=40),
    coloraxis_colorbar=dict(title=speed_label)
)

# Render Track Map cleanly
st.plotly_chart(fig_map, use_container_width=True)
