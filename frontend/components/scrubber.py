"""
ApexHunter Frontend - Session-time scrubber state
One shared position (st.session_state["scrub_seconds"]) drives every view. The master scrubber
(Race Intelligence) and the Telemetry-tab slider are two widgets on that same value. Moving one
runs its on_change callback, which writes the new position to the shared value AND to the other
slider (Streamlit's pattern for linked widgets). Widget values are only written from the script
body when missing or out of range; doing it on every run makes Streamlit drop the next user move.
"""

import datetime

import streamlit as st

SCRUB_KEY = "scrub_seconds"
MASTER_KEY = "scrub_dt"
TELEMETRY_KEY = "telemetry_scrub_seconds"

# The master slider shows a clock (HH:mm:ss), so it works on datetimes offset from this date
_BASE_DATE = datetime.datetime(2000, 1, 1)


def current_scrub(min_t: float, max_t: float) -> float:
    """Return the shared scrub position clamped to [min_t, max_t] (defaults to the session end)."""
    value = max(min_t, min(st.session_state.get(SCRUB_KEY, max_t), max_t))
    st.session_state[SCRUB_KEY] = value
    return value


def _to_datetime(seconds: float) -> datetime.datetime:
    return _BASE_DATE + datetime.timedelta(seconds=seconds)


def _set_all(seconds: float) -> None:
    st.session_state[SCRUB_KEY] = seconds
    st.session_state[MASTER_KEY] = _to_datetime(seconds)
    st.session_state[TELEMETRY_KEY] = seconds


def _on_master_change() -> None:
    _set_all((st.session_state[MASTER_KEY] - _BASE_DATE).total_seconds())


def _on_telemetry_change() -> None:
    _set_all(float(st.session_state[TELEMETRY_KEY]))


def render_master_scrubber(min_t: float, max_t: float) -> float:
    """Render the master session-time slider and return the shared scrub position (s)."""
    scrub = current_scrub(min_t, max_t)
    if st.session_state.get(MASTER_KEY) != _to_datetime(scrub):   # first draw, or clamped
        st.session_state[MASTER_KEY] = _to_datetime(scrub)
    st.slider(
        "Master session time",
        min_value=_to_datetime(min_t),
        max_value=_to_datetime(max_t),
        format="HH:mm:ss.S",
        key=MASTER_KEY,
        label_visibility="collapsed",
        step=datetime.timedelta(milliseconds=100),
        on_change=_on_master_change,
    )
    return scrub


def render_telemetry_scrubber(min_t: float, max_t: float) -> float:
    """Render the Telemetry-tab slider on the same shared position and return it (s)."""
    scrub = current_scrub(min_t, max_t)
    if st.session_state.get(TELEMETRY_KEY) != scrub:              # first draw, or clamped
        st.session_state[TELEMETRY_KEY] = scrub
    st.slider(
        "Session Time",
        min_value=min_t,
        max_value=max_t,
        format="%.1f s",
        key=TELEMETRY_KEY,
        on_change=_on_telemetry_change,
    )
    return scrub
