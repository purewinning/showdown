import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


REQUIRED_PROJ_COLS = ["Player", "Salary", "Team", "Projection", "Std Dev"]


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def validate_projections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_PROJ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Projections CSV missing required columns: {missing}")

    for c in ["Salary", "Projection", "Std Dev", "Ownership %", "CPT Ownership %", "CPT Optimal %"]:
        if c in df.columns:
            df[c] = _num(df[c])

    if "Ownership %" not in df.columns:
        df["Ownership %"] = np.nan

    df["Player"] = df["Player"].astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()

    df = df.dropna(subset=["Player", "Team", "Salary", "Projection", "Std Dev"]).reset_index(drop=True)
    return df


def parse_captain_from_lineup(lineup_str: str) -> Optional[str]:
    if not isinstance(lineup_str, str) or not lineup_str.strip():
        return None
    toks = lineup_str.strip().split()
    try:
        i = toks.index("CPT")
    except ValueError:
        return None
    name = []
    for t in toks[i+1:]:
        if t in ("UTIL", "CPT"):
            break
        name.append(t)
    return " ".join(name).strip() if name else None


# -----------------------------
# Captain metrics
# -----------------------------

def captain_score_projection(row: pd.Series, k_ceiling: float = 1.75, alpha_lev: float = 0.35, beta_own: float = 0.15) -> float:
    '''
    Projection-based CPT score (fallback when you do not have outcomes):
    - Base: 1.5*(mean + k*sd)
    - Optional boost for positive leverage (CPT Optimal% > CPT Ownership%)
    - Optional boost for lower CPT ownership
    '''
    m = float(row["Projection"])
    sd = float(row["Std Dev"])
    base = 1.5 * (m + k_ceiling * sd)

    cpt_opt = row.get("CPT Optimal %", np.nan)
    cpt_own = row.get("CPT Ownership %", np.nan)
    cpt_opt = float(cpt_opt) if pd.notna(cpt_opt) else 0.0
    cpt_own = float(cpt_own) if pd.notna(cpt_own) else 0.0

    lev = max((cpt_opt - cpt_own) / 100.0, 0.0)
    cpt_own_frac = max(cpt_own / 100.0, 1e-6)

    return base * (1.0 + alpha_lev * lev) * (1.0 + beta_own * (1.0 - cpt_own_frac))


def build_outcome_based_captain_table(
    proj: pd.DataFrame,
    standings: pd.DataFrame,
    top_frac: float = 0.005,
    min_samples: int = 50,
) -> pd.DataFrame:
    '''
    Outcome-based CPT metric from a prior slate standings file.
    '''
    proj_df = validate_projections(proj)
    st = standings.copy()
    st.columns = [c.strip() for c in st.columns]
    if "Lineup" not in st.columns:
        raise ValueError("Standings CSV missing 'Lineup' column.")
    if "Points" not in st.columns:
        if "FPTS" in st.columns:
            st["Points"] = _num(st["FPTS"])
        else:
            raise ValueError("Standings CSV missing 'Points' (or 'FPTS') column.")

    st = st[st["Lineup"].notna()].copy()
    st["CPT"] = st["Lineup"].apply(parse_captain_from_lineup)
    st = st[st["CPT"].notna()].copy()

    n = len(st)
    top_n = max(1, int(top_frac * n))
    thr = st["Points"].nlargest(top_n).min()
    st["is_top"] = st["Points"] >= thr

    cap_stats = (
        st.groupby("CPT")
        .agg(
            lineups=("CPT", "size"),
            avg_pts=("Points", "mean"),
            p90_pts=("Points", lambda s: float(pd.Series(s).quantile(0.90))),
            top_rate=("is_top", "mean"),
        )
        .reset_index()
    )

    look = proj_df.set_index("Player")[["Salary", "Ownership %", "Projection", "Std Dev"]].copy()
    look.columns = ["sal", "own", "proj", "sd"]
    cap_stats = cap_stats.merge(look.reset_index().rename(columns={"Player": "CPT"}), on="CPT", how="left")

    cap_stats["sal"] = cap_stats["sal"].fillna(cap_stats["sal"].median())
    cap_stats["own"] = cap_stats["own"].fillna(cap_stats["own"].median())

    p90_med = float(cap_stats["p90_pts"].median()) if len(cap_stats) else 1.0
    sal_med = float(cap_stats["sal"].median()) if len(cap_stats) else 1.0
    own_med = float(cap_stats["own"].median()) if len(cap_stats) else 1.0

    cap_stats["outcome_value_score"] = (
        (cap_stats["top_rate"] + 1e-6) * (cap_stats["p90_pts"] / (p90_med if p90_med > 0 else 1.0))
    ) / (
        ((cap_stats["own"] / (own_med if own_med > 0 else 1.0)) + 1e-3) *
        ((cap_stats["sal"] / (sal_med if sal_med > 0 else 1.0)) + 1e-3)
    )

    out = cap_stats[cap_stats["lineups"] >= min_samples].sort_values("outcome_value_score", ascending=False).reset_index(drop=True)
    return out


def rank_captains(
    proj: pd.DataFrame,
    standings: Optional[pd.DataFrame] = None,
    use_outcomes: bool = False,
    top_n: int = 20,
    top_frac: float = 0.005,
    min_samples: int = 50,
) -> pd.DataFrame:
    if use_outcomes and standings is not None:
        out = build_outcome_based_captain_table(proj, standings, top_frac=top_frac, min_samples=min_samples)
        cols = ["CPT", "lineups", "top_rate", "p90_pts", "sal", "own", "outcome_value_score"]
        return out[cols].head(top_n).reset_index(drop=True)

    df = validate_projections(proj).copy()
    df["cpt_score"] = df.apply(captain_score_projection, axis=1)
    cols = ["Player", "Team", "Salary", "Projection", "Std Dev", "Ownership %", "cpt_score"]
    return df.sort_values("cpt_score", ascending=False)[cols].head(top_n).reset_index(drop=True)


# -----------------------------
# Lineup representation
# -----------------------------

@dataclass(frozen=True)
class Lineup:
    captain: str
    utils: Tuple[str, str, str, str, str]

    def as_dk_string(self) -> str:
        parts = ["CPT", self.captain]
        for u in self.utils:
            parts += ["UTIL", u]
        return " ".join(parts)

    def players(self) -> Tuple[str, ...]:
        return (self.captain,) + self.utils


@dataclass
class Constraints:
    salary_cap: int = 50000
    min_salary: int = 49000

    max_from_one_team: int = 5
    disallow_6_0: bool = True
    min_cpt_teammates: int = 2
    require_bringback: bool = True

    punt_salary_threshold: int = 4200
    max_punts: int = 1
    min_util_projection: float = 0.0

    max_same_player_across_lineups: float = 0.70
    max_same_captain_across_lineups: float = 0.40
    diversity_min_unique_players_between_lineups: int = 2


def _lookup(df: pd.DataFrame) -> Dict[str, Dict]:
    return df.set_index("Player").to_dict(orient="index")


def lineup_salary(L: Lineup, look: Dict[str, Dict]) -> float:
    sal = 1.5 * float(look[L.captain]["Salary"])
    for u in L.utils:
        sal += float(look[u]["Salary"])
    return sal


def lineup_mean_sd(L: Lineup, look: Dict[str, Dict]) -> Tuple[float, float]:
    mean = 1.5 * float(look[L.captain]["Projection"])
    var = (1.5 * float(look[L.captain]["Std Dev"])) ** 2
    for u in L.utils:
        mean += float(look[u]["Projection"])
        var += float(look[u]["Std Dev"]) ** 2
    return mean, math.sqrt(var)


def uniqueness_score(L: Lineup, look: Dict[str, Dict]) -> float:
    prod = 1.0
    for p in L.players():
        own = look[p].get("Ownership %", np.nan)
        if own is None or (isinstance(own, float) and np.isnan(own)) or float(own) <= 0:
            own = 20.0
        prod *= float(own) / 100.0
    return -math.log(max(prod, 1e-12))


def team_counts(L: Lineup, look: Dict[str, Dict]) -> Dict[str, int]:
    teams = [look[p]["Team"] for p in L.players()]
    return dict(pd.Series(teams).value_counts())


def punt_count(L: Lineup, look: Dict[str, Dict], threshold: int) -> int:
    return sum(1 for p in L.players() if float(look[p]["Salary"]) <= threshold)


def is_valid(L: Lineup, look: Dict[str, Dict], cons: Constraints) -> bool:
    ps = L.players()
    if len(set(ps)) != 6:
        return False

    sal = lineup_salary(L, look)
    if sal > cons.salary_cap or sal < cons.min_salary:
        return False

    tc = team_counts(L, look)
    mx = max(tc.values())
    if cons.disallow_6_0 and mx == 6:
        return False
    if mx > cons.max_from_one_team:
        return False

    cpt_team = look[L.captain]["Team"]
    teammates = sum(1 for u in L.utils if look[u]["Team"] == cpt_team)
    if teammates < cons.min_cpt_teammates:
        return False

    if cons.require_bringback and len(tc.keys()) < 2:
        return False

    if punt_count(L, look, cons.punt_salary_threshold) > cons.max_punts:
        return False

    if cons.min_util_projection > 0:
        for u in L.utils:
            if float(look[u]["Projection"]) < cons.min_util_projection:
                return False

    return True


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)


def _weighted_choice(rng: np.random.Generator, items: List[str], weights: np.ndarray) -> str:
    return items[int(rng.choice(len(items), p=weights))]


def _util_weight(row: pd.Series, k_ceiling: float = 1.5, own_fade: float = 0.20) -> float:
    m = float(row["Projection"])
    sd = float(row["Std Dev"])
    base = m + k_ceiling * sd
    own = row.get("Ownership %", np.nan)
    own = float(own) if pd.notna(own) else 25.0
    return base * (1.0 - own_fade * (own / 100.0))


def generate_candidates(
    proj: pd.DataFrame,
    standings: Optional[pd.DataFrame],
    use_outcome_cpt: bool,
    n_candidates: int,
    cons: Constraints,
    captain_pool: int = 10,
    util_pool: int = 45,
    rng_seed: int = 42,
    outcome_top_frac: float = 0.005,
    outcome_min_samples: int = 50,
) -> pd.DataFrame:
    df = validate_projections(proj)
    look = _lookup(df)
    rng = np.random.default_rng(rng_seed)

    if use_outcome_cpt and standings is not None:
        ctab = build_outcome_based_captain_table(df, standings, top_frac=outcome_top_frac, min_samples=outcome_min_samples)
        cap_items = ctab["CPT"].head(captain_pool).tolist()
        cap_scores = ctab.set_index("CPT")["outcome_value_score"].reindex(cap_items).fillna(0.0).values
    else:
        df["cpt_score"] = df.apply(captain_score_projection, axis=1)
        cap_df = df.sort_values("cpt_score", ascending=False).head(captain_pool).reset_index(drop=True)
        cap_items = cap_df["Player"].tolist()
        cap_scores = cap_df["cpt_score"].values

    cap_w = _softmax(cap_scores, temp=1.0)

    df["util_weight"] = df.apply(_util_weight, axis=1)
    util_df = df.sort_values("util_weight", ascending=False).head(util_pool).reset_index(drop=True)
    util_items = util_df["Player"].tolist()
    util_w = _softmax(util_df["util_weight"].values, temp=1.0)

    rows = []
    seen = set()
    attempts = 0
    max_attempts = max(30000, n_candidates * 25)

    while len(rows) < n_candidates and attempts < max_attempts:
        attempts += 1
        cpt = _weighted_choice(rng, cap_items, cap_w)

        utils = []
        for _ in range(800):
            p = _weighted_choice(rng, util_items, util_w)
            if p != cpt and p not in utils:
                utils.append(p)
            if len(utils) == 5:
                break
        if len(utils) < 5:
            continue

        L = Lineup(captain=cpt, utils=tuple(utils))
        if not is_valid(L, look, cons):
            continue

        key = L.as_dk_string()
        if key in seen:
            continue
        seen.add(key)

        mean, sd = lineup_mean_sd(L, look)
        ceiling = mean + 1.75 * sd
        sal = lineup_salary(L, look)
        uniq = uniqueness_score(L, look)
        tc = team_counts(L, look)
        mx = max(tc.values())
        mn = min(tc.values()) if len(tc) > 1 else 0

        rows.append({
            "Lineup": key,
            "CPT": L.captain,
            "UTIL1": L.utils[0],
            "UTIL2": L.utils[1],
            "UTIL3": L.utils[2],
            "UTIL4": L.utils[3],
            "UTIL5": L.utils[4],
            "salary": sal,
            "salary_left": cons.salary_cap - sal,
            "proj_mean": mean,
            "proj_sd": sd,
            "proj_ceiling": ceiling,
            "uniq_score": uniq,
            "team_split": f"{mx}-{mn}",
            "punts": int(punt_count(L, look, cons.punt_salary_threshold)),
        })

    return pd.DataFrame(rows).sort_values("proj_ceiling", ascending=False).reset_index(drop=True)


def score_candidates_monte_carlo(
    proj: pd.DataFrame,
    candidates: pd.DataFrame,
    n_sims: int = 3000,
    rng_seed: int = 7,
) -> pd.DataFrame:
    df = validate_projections(proj)
    look = _lookup(df)
    cand = candidates.copy()

    players = set()
    cand_players = []
    for _, r in cand.iterrows():
        ps = [r["CPT"], r["UTIL1"], r["UTIL2"], r["UTIL3"], r["UTIL4"], r["UTIL5"]]
        cand_players.append(ps)
        players.update(ps)

    players = [p for p in players if p in look]
    p_to_i = {p: i for i, p in enumerate(players)}

    means = np.array([float(look[p]["Projection"]) for p in players], dtype=float)
    sds = np.array([float(look[p]["Std Dev"]) for p in players], dtype=float)

    rng = np.random.default_rng(rng_seed)
    samples = rng.normal(loc=means[:, None], scale=sds[:, None], size=(len(players), n_sims))
    samples = np.clip(samples, 0, None)

    n_lineups = len(cand)
    idx = np.zeros((n_lineups, 6), dtype=int)
    mult = np.ones((n_lineups, 6), dtype=float)
    mult[:, 0] = 1.5

    for i, ps in enumerate(cand_players):
        for j, p in enumerate(ps):
            idx[i, j] = p_to_i[p]

    points = np.zeros((n_lineups, n_sims), dtype=float)
    for j in range(6):
        points += mult[:, j:j+1] * samples[idx[:, j], :]

    winners = np.argmax(points, axis=0)
    win_counts = np.bincount(winners, minlength=n_lineups)
    win_prob = win_counts / float(n_sims)

    topk = max(1, int(0.01 * n_lineups))
    thr = np.partition(points, -topk, axis=0)[-topk, :]
    top1pct = (points >= thr[None, :]).mean(axis=1)

    cand["fp_win_prob_proxy"] = win_prob
    cand["top1pct_prob_proxy"] = top1pct

    med_uniq = float(cand["uniq_score"].median()) if len(cand) else 1.0
    cand["banger_score"] = (
        0.60 * cand["fp_win_prob_proxy"] +
        0.30 * cand["top1pct_prob_proxy"] +
        0.10 * (cand["uniq_score"] / (med_uniq if med_uniq > 0 else 1.0)) -
        0.06 * cand["punts"]
    )

    return cand.sort_values("banger_score", ascending=False).reset_index(drop=True)


def _lineup_players(row) -> List[str]:
    return [row["CPT"], row["UTIL1"], row["UTIL2"], row["UTIL3"], row["UTIL4"], row["UTIL5"]]


def select_final_lineups(scored: pd.DataFrame, n_lineups: int, cons: Constraints) -> pd.DataFrame:
    out_rows = []
    player_counts: Dict[str, int] = {}
    cpt_counts: Dict[str, int] = {}

    def can_add(row) -> bool:
        ps = _lineup_players(row)

        for p in ps:
            if player_counts.get(p, 0) / max(1, n_lineups) >= cons.max_same_player_across_lineups:
                return False

        cpt = row["CPT"]
        if cpt_counts.get(cpt, 0) / max(1, n_lineups) >= cons.max_same_captain_across_lineups:
            return False

        for r2 in out_rows:
            ps2 = set(_lineup_players(r2))
            overlap = len(set(ps) & ps2)
            if (6 - overlap) < cons.diversity_min_unique_players_between_lineups:
                return False

        return True

    for _, row in scored.iterrows():
        if len(out_rows) >= n_lineups:
            break
        if can_add(row):
            out_rows.append(row)
            for p in _lineup_players(row):
                player_counts[p] = player_counts.get(p, 0) + 1
            c = row["CPT"]
            cpt_counts[c] = cpt_counts.get(c, 0) + 1

    if not out_rows:
        raise RuntimeError("No lineups selected. Loosen min_salary / punts / correlation / exposure / diversity.")

    return pd.DataFrame(out_rows).reset_index(drop=True)


def build_lineups(
    proj: pd.DataFrame,
    standings: Optional[pd.DataFrame],
    use_outcome_cpt: bool,
    n_lineups: int = 10,
    n_candidates: int = 4000,
    n_sims: int = 3000,
    cons: Constraints = Constraints(),
    rng_seed: int = 42,
) -> pd.DataFrame:
    candidates = generate_candidates(
        proj=proj,
        standings=standings,
        use_outcome_cpt=use_outcome_cpt,
        n_candidates=n_candidates,
        cons=cons,
        rng_seed=rng_seed,
    )
    if len(candidates) == 0:
        raise RuntimeError("Generated 0 candidates. Loosen constraints (punts/correlation/min_salary/util_pool).")

    scored = score_candidates_monte_carlo(
        proj=proj,
        candidates=candidates,
        n_sims=n_sims,
        rng_seed=rng_seed + 1,
    )

    final = select_final_lineups(scored, n_lineups=n_lineups, cons=cons)
    return final
