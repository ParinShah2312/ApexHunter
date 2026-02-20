import streamlit as st
import pandas as pd
from pathlib import Path

# Constants
DATA_LAKE_DIR = Path("data_lake/season_data")

st.set_page_config(page_title="ApexHunter Dashboard", layout="wide")

st.title("🏎️ ApexHunter F1 Data Explorer")

# Sidebar
st.sidebar.header("Filter Data")

# 1. Load available data to populate filters
if not DATA_LAKE_DIR.exists():
    st.error(f"Data directory {DATA_LAKE_DIR} not found. Please run the downloader script.")
    st.stop()

# Scan files to get available Years, Rounds, Sessions
# Filename format: {year}_{round}_{session}.parquet
files = list(DATA_LAKE_DIR.glob("*.parquet"))

if not files:
    st.warning("No data files found in the data lake.")
    st.stop()

# Parse filenames
data_index = []
for f in files:
    try:
        parts = f.stem.split('_')
        # Expecting year, round, session
        if len(parts) == 3:
            year, round_num, session = parts
            data_index.append({
                'Year': int(year),
                'Round': int(round_num),
                'Session': session,
                'File': f
            })
    except Exception as e:
        continue

if not data_index:
    st.error("Could not parse any filenames. Ensure they follow 'year_round_session.parquet' format.")
    st.stop()

df_index = pd.DataFrame(data_index)

# Filters
available_years = sorted(df_index['Year'].unique())
selected_year = st.sidebar.selectbox("Select Year", available_years)

# Filter rounds based on year
rounds_in_year = sorted(df_index[df_index['Year'] == selected_year]['Round'].unique())
selected_round = st.sidebar.selectbox("Select Round", rounds_in_year)

# Filter sessions based on round
sessions_in_round = sorted(df_index[(df_index['Year'] == selected_year) & (df_index['Round'] == selected_round)]['Session'].unique())
selected_session = st.sidebar.selectbox("Select Session", sessions_in_round)

# Identify the file
selected_file_row = df_index[
    (df_index['Year'] == selected_year) & 
    (df_index['Round'] == selected_round) & 
    (df_index['Session'] == selected_session)
]

if selected_file_row.empty:
    st.error("File not found for selected filters.")
else:
    file_path = selected_file_row.iloc[0]['File']
    
    # Load Data
    st.subheader(f"Data for {selected_year} Round {selected_round} - Session {selected_session}")
    st.info(f"Loading {file_path.name}...")
    
    try:
        df = pd.read_parquet(file_path)
        
        # Driver Filter
        available_drivers = sorted(df['Driver'].unique())
        selected_driver = st.sidebar.selectbox("Select Driver", ["All"] + list(available_drivers))
        
        if selected_driver != "All":
            df = df[df['Driver'] == selected_driver]
        
        # Display Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df))
        col2.metric("Drivers", len(df['Driver'].unique()))
        col3.metric("Data Size (MB)", f"{file_path.stat().st_size / (1024*1024):.2f}")

        # Display Data
        st.dataframe(df.head(1000) if len(df) > 1000 else df)
        
        if len(df) > 1000:
            st.caption("Showing first 1000 rows.")

    except Exception as e:
        st.error(f"Error loading file: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**ApexHunter v1.0**")
st.sidebar.markdown("Dev: Parin Shah | ID: 23001091")
