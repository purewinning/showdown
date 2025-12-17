
import streamlit as st
import pandas as pd
import numpy as np
from builder import build_lineups, rank_captains

st.set_page_config(page_title="DK Showdown Builder", layout="wide")

st.title("DraftKings Showdown Builder – First Place Focus")

proj_file = st.file_uploader("Upload Projections CSV", type=["csv"])
standings_file = st.file_uploader("Upload Contest Standings (optional)", type=["csv"])

if proj_file:
    proj = pd.read_csv(proj_file)
    st.subheader("Captain Rankings")
    cap_rank = rank_captains(proj)
    st.dataframe(cap_rank)

    st.subheader("Generate Lineups")
    n_lineups = st.slider("Number of Lineups", 1, 50, 10)
    if st.button("Build Lineups"):
        lineups = build_lineups(proj, n_lineups=n_lineups)
        st.dataframe(lineups)
        st.download_button("Download Lineups CSV", lineups.to_csv(index=False), "lineups.csv")
