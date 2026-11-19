"""Attendance dashboard.   Run:  streamlit run dashboard.py"""
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title="Face Attendance", layout="wide")
st.title("Face Attendance")

if not config.DB_FILE.exists():
    st.info("No attendance yet. Run `python attend.py` first.")
    st.stop()

con = sqlite3.connect(config.DB_FILE)
df = pd.read_sql("SELECT * FROM attendance ORDER BY time", con)
enrolled = []
if config.EMBEDDINGS_FILE.exists():
    enrolled = list(np.load(config.EMBEDDINGS_FILE)["names"])

sessions = sorted(df["session"].unique(), reverse=True)
if not sessions:
    st.info("No attendance yet.")
    st.stop()
session = st.selectbox("Session", sessions)
sdf = df[df["session"] == session]

present = set(sdf["name"])
absent = sorted(set(enrolled) - present)
c1, c2, c3 = st.columns(3)
c1.metric("Enrolled", len(enrolled))
c2.metric("Present", len(present))
c3.metric("Absent", len(absent))

left, right = st.columns([2, 1])
with left:
    st.subheader("Present")
    st.dataframe(sdf[["name", "time", "similarity", "face_px"]], use_container_width=True, hide_index=True)
    st.download_button("Download CSV", sdf.to_csv(index=False), f"attendance_{session}.csv", "text/csv")
with right:
    st.subheader("Absent")
    st.write("\n".join(f"- {n}" for n in absent) or "Everyone is present")

st.subheader("Proof snapshots")
cols = st.columns(6)
for i, row in enumerate(sdf.itertuples()):
    p = config.SNAPSHOT_DIR / str(row.snapshot)
    if p.exists():
        cols[i % 6].image(str(p), caption=f"{row.name} ({row.similarity:.2f})")

st.subheader("All sessions (student x session)")
pivot = pd.crosstab(df["name"], df["session"]).reindex(enrolled or None, fill_value=0)
st.dataframe(pivot.replace({1: "P", 0: "A"}), use_container_width=True)
