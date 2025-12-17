
import numpy as np
import pandas as pd
import math

def captain_score(row, k_ceiling=1.75, alpha_leverage=0.35, beta_own=0.15):
    m = row["Projection"]
    sd = row["Std Dev"]
    cpt_ceiling = 1.5*(m + k_ceiling*sd)
    lev = (row.get("CPT Optimal %", 0) - row.get("CPT Ownership %", 0)) / 100.0
    lev_pos = max(lev, 0)
    cpt_own = max(row.get("CPT Ownership %", 30)/100.0, 1e-6)
    return cpt_ceiling * (1 + alpha_leverage*lev_pos) * (1 + beta_own*(1 - cpt_own))

def rank_captains(proj):
    proj = proj.copy()
    for c in ["Projection","Std Dev","Ownership %","CPT Ownership %","CPT Optimal %","Salary"]:
        if c in proj.columns:
            proj[c] = pd.to_numeric(proj[c], errors="coerce")
    proj["cpt_score"] = proj.apply(captain_score, axis=1)
    return proj.sort_values("cpt_score", ascending=False)[[
        "Player","Team","Salary","Projection","Std Dev","Ownership %","cpt_score"
    ]]

def build_lineups(proj, n_lineups=10, salary_cap=50000):
    proj = proj.copy()
    proj["value"] = proj["Projection"] / proj["Salary"]
    captains = rank_captains(proj).head(5)["Player"].tolist()
    lineups = []
    for cpt in captains:
        pool = proj[proj["Player"] != cpt].sort_values("value", ascending=False)
        utils = pool.head(5)["Player"].tolist()
        lineups.append({
            "CPT": cpt,
            "UTILS": ",".join(utils)
        })
        if len(lineups) >= n_lineups:
            break
    return pd.DataFrame(lineups)
