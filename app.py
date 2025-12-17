import streamlit as st
import pandas as pd
from builder import build_lineups, rank_captains, Constraints

st.set_page_config(page_title="DK Showdown Builder", layout="wide")
st.title("DK Showdown Builder – First Place Focus")

proj_file = st.file_uploader("Upload Projections CSV", type=["csv"])
if not proj_file:
    st.info("Upload a projections CSV to begin.")
    st.stop()

proj = pd.read_csv(proj_file)

tab_build, tab_cpt = st.tabs(["Build Lineups", "Captain Rankings"])

with tab_cpt:
    st.subheader("Captain Rankings")
    top_n = st.slider("Show top N captains", 5, 30, 20)
    cap = rank_captains(proj, top_n=top_n)
    st.dataframe(cap, use_container_width=True)

with tab_build:
    st.subheader("Constraints")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_lineups = st.number_input("Final lineups", min_value=1, max_value=150, value=10, step=1)
        n_candidates = st.number_input("Candidates to generate", min_value=300, max_value=20000, value=3000, step=100)
    with c2:
        n_sims = st.number_input("Monte Carlo sims", min_value=500, max_value=20000, value=2500, step=250)
        min_salary = st.number_input("Min salary", min_value=0, max_value=50000, value=49000, step=100)
    with c3:
        max_team = st.slider("Max from one team", 3, 5, 5)
        diversity = st.slider("Min unique players between lineups", 0, 5, 2)

    st.subheader("Exposure caps (final set)")
    e1, e2 = st.columns(2)
    with e1:
        max_player_exp = st.slider("Max player exposure", 0.10, 1.00, 0.70, 0.05)
    with e2:
        max_cpt_exp = st.slider("Max captain exposure", 0.10, 1.00, 0.40, 0.05)

    cons = Constraints(
        min_salary=int(min_salary),
        max_from_one_team=int(max_team),
        max_same_player_across_lineups=float(max_player_exp),
        max_same_captain_across_lineups=float(max_cpt_exp),
        diversity_min_unique_players_between_lineups=int(diversity),
    )

    if st.button("Build lineups"):
        with st.spinner("Generating candidates + scoring..."):
            final = build_lineups(
                proj,
                n_lineups=int(n_lineups),
                n_candidates=int(n_candidates),
                n_sims=int(n_sims),
                cons=cons,
                rng_seed=42,
            )

        st.success(f"Built {len(final)} lineups.")
        show_cols = [
            "banger_score", "fp_win_prob_proxy", "top1pct_prob_proxy",
            "salary", "salary_left", "proj_mean", "proj_ceiling", "uniq_score", "team_split", "Lineup"
        ]
        st.dataframe(final[show_cols], use_container_width=True)

        st.download_button(
            "Download lineups (CSV)",
            data=final.to_csv(index=False),
            file_name="showdown_lineups_scored.csv",
            mime="text/csv"
        )

