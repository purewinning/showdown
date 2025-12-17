import streamlit as st
import pandas as pd
from builder import build_lineups, rank_captains, Constraints

st.set_page_config(page_title="DK Showdown Builder (Banger)", layout="wide")
st.title("DK Showdown Builder – Stochastic Banger Framework")

c_up, s_up = st.columns(2)
with c_up:
    proj_file = st.file_uploader("Upload Projections CSV", type=["csv"])
with s_up:
    standings_file = st.file_uploader("Upload Contest Standings CSV (optional, for outcome-based CPT)", type=["csv"])

if not proj_file:
    st.info("Upload a projections CSV to begin.")
    st.stop()

proj = pd.read_csv(proj_file)
standings = pd.read_csv(standings_file) if standings_file else None

tab_build, tab_cpt = st.tabs(["Build Lineups", "Captain Rankings"])

with tab_cpt:
    st.subheader("Captain Rankings")
    use_outcomes = st.checkbox("Use outcome-based CPT metric (requires standings upload)", value=bool(standings_file))
    top_frac = st.select_slider("Define 'banger' as top X% of contest", options=[0.0025, 0.005, 0.01], value=0.005)
    min_samples = st.number_input("Min samples per CPT (to rank)", min_value=10, max_value=500, value=50, step=10)
    top_n = st.slider("Show top N captains", 5, 30, 20)

    cap = rank_captains(
        proj,
        standings=standings,
        use_outcomes=use_outcomes and standings is not None,
        top_n=top_n,
        top_frac=float(top_frac),
        min_samples=int(min_samples),
    )
    st.dataframe(cap, use_container_width=True)

with tab_build:
    st.subheader("Builder Controls")
    left, mid, right = st.columns(3)

    with left:
        n_lineups = st.number_input("Final lineups", min_value=1, max_value=150, value=10, step=1)
        n_candidates = st.number_input("Candidates to generate", min_value=500, max_value=30000, value=4000, step=250)
        n_sims = st.number_input("Monte Carlo sims", min_value=500, max_value=20000, value=3000, step=250)

    with mid:
        min_salary = st.number_input("Min salary", min_value=0, max_value=50000, value=49000, step=100)
        max_team = st.slider("Max from one team", 3, 5, 5)
        min_cpt_teammates = st.slider("Min CPT teammates", 0, 4, 2)
        bringback = st.checkbox("Require bring-back (at least 1 from other team)", value=True)

    with right:
        punt_thr = st.number_input("Punt salary threshold", min_value=2000, max_value=8000, value=4200, step=100)
        max_punts = st.slider("Max punts in lineup", 0, 3, 1)
        min_util_proj = st.number_input("Min UTIL projection (optional)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)

    st.subheader("Exposure + Diversity (final set)")
    e1, e2, e3 = st.columns(3)
    with e1:
        max_player_exp = st.slider("Max player exposure", 0.10, 1.00, 0.70, 0.05)
    with e2:
        max_cpt_exp = st.slider("Max captain exposure", 0.10, 1.00, 0.40, 0.05)
    with e3:
        diversity = st.slider("Min unique players between lineups", 0, 5, 2)

    use_outcome_cpt = st.checkbox("Use outcome-based CPT pool (if standings uploaded)", value=bool(standings_file))

    cons = Constraints(
        min_salary=int(min_salary),
        max_from_one_team=int(max_team),
        min_cpt_teammates=int(min_cpt_teammates),
        require_bringback=bool(bringback),
        punt_salary_threshold=int(punt_thr),
        max_punts=int(max_punts),
        min_util_projection=float(min_util_proj),
        max_same_player_across_lineups=float(max_player_exp),
        max_same_captain_across_lineups=float(max_cpt_exp),
        diversity_min_unique_players_between_lineups=int(diversity),
    )

    if st.button("Build banger lineups"):
        with st.spinner("Generating candidates + Monte Carlo scoring..."):
            final = build_lineups(
                proj=proj,
                standings=standings,
                use_outcome_cpt=use_outcome_cpt and standings is not None,
                n_lineups=int(n_lineups),
                n_candidates=int(n_candidates),
                n_sims=int(n_sims),
                cons=cons,
                rng_seed=42,
            )

        st.success(f"Built {len(final)} lineups.")
        show_cols = [
            "banger_score", "fp_win_prob_proxy", "top1pct_prob_proxy",
            "salary", "salary_left", "proj_mean", "proj_ceiling",
            "uniq_score", "team_split", "punts", "Lineup"
        ]
        st.dataframe(final[show_cols], use_container_width=True)

        st.download_button(
            "Download lineups (CSV)",
            data=final.to_csv(index=False),
            file_name="showdown_bangers.csv",
            mime="text/csv"
        )
