import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


REQUIRED_COLS = ["Player", "Salary", "Team", "Projection", "Std Dev"]


def validate_projections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Projections CSV missing required columns: {missing}")

    # Numeric coercion
    for c in ["Salary", "Projection", "Std Dev", "Ownership %", "CPT Ownership %", "CPT Optimal %"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ownership defaults if absent
    if "Ownership %" not in df.columns:
        df["Ownership %"] = np.nan
    if "CPT Ownership %" not in df.columns:
        df["CPT Ownership %"] = np.nan
    if "CPT Optimal %" not in df.columns:
        df["CPT Optimal %"] = np.nan

    # Clean strings
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()

    df = df.dropna(subset=["Player", "Team", "Salary", "Projection", "Std Dev"]).reset_index(drop=True)
    return df


def captain_score(row: pd.Series, k_ceiling: float = 1.75, alpha_leverage: float = 0.35, beta_own: float = 0.15) -> float:
    # Base: captain ceiling proxy
    m = float(row["Projection"])
    sd = float(row["Std Dev"])
    base = 1.5 * (m + k_ceiling * sd)

    # Optional leverage boost if CPT Optimal% > CPT Ownership%
    cpt_opt = row.get("CPT Optimal %", np.nan)
    cpt_own = row.get("CPT Ownership %", np.nan)
    cpt_opt = float(cpt_opt) if pd.notna(cpt_opt) else 0.0
    cpt_own = float(cpt_own) if pd.notna(cpt_own) else 0.0

    lev = max((cpt_opt - cpt_own) / 100.0, 0.0)
    cpt_own_frac = max(cpt_own / 100.0, 1e-6)

    return base * (1.0 + alpha_leverage * lev) * (1.0 + beta_own * (1.0 - cpt_own_frac))


def rank_captains(proj: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    df = validate_projections(proj)
    df = df.copy()
    df["cpt_score"] = df.apply(captain_score, axis=1)
    out = df.sort_values("cpt_score", ascending=False).head(top_n)[
        ["Player", "Team", "Salary", "Projection", "Std Dev", "Ownership %", "CPT Ownership %", "CPT Optimal %", "cpt_score"]
    ]
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class Lineup:
    captain: str
    utils: Tuple[str, str, str, str, str]

    def as_dk_string(self) -> str:
        parts = ["CPT", self.captain]
        for u in self.utils:
            parts += ["UTIL", u]
        return " ".join(parts)


@dataclass
class Constraints:
    salary_cap: int = 50000
    min_salary: int = 49000
    max_from_one_team: int = 5
    disallow_6_0: bool = True
    max_same_player_across_lineups: float = 0.70
    max_same_captain_across_lineups: float = 0.40
    diversity_min_unique_players_between_lineups: int = 2


def lineup_salary(proj_lookup: Dict[str, Dict], lineup: Lineup) -> float:
    sal = 1.5 * float(proj_lookup[lineup.captain]["Salary"])
    for u in lineup.utils:
        sal += float(proj_lookup[u]["Salary"])
    return sal


def lineup_mean_sd(proj_lookup: Dict[str, Dict], lineup: Lineup) -> Tuple[float, float]:
    mean = 1.5 * float(proj_lookup[lineup.captain]["Projection"])
    var = (1.5 * float(proj_lookup[lineup.captain]["Std Dev"])) ** 2
    for u in lineup.utils:
        mean += float(proj_lookup[u]["Projection"])
        var += float(proj_lookup[u]["Std Dev"]) ** 2
    return mean, math.sqrt(var)


def uniqueness_score(proj_lookup: Dict[str, Dict], lineup: Lineup) -> float:
    prod = 1.0
    for p in (lineup.captain,) + lineup.utils:
        own = proj_lookup[p].get("Ownership %", np.nan)
        if own is None or (isinstance(own, float) and np.isnan(own)) or float(own) <= 0:
            own = 20.0
        prod *= float(own) / 100.0
    return -math.log(max(prod, 1e-12))


def team_split(proj_lookup: Dict[str, Dict], lineup: Lineup) -> Tuple[int, int]:
    teams = [proj_lookup[p]["Team"] for p in (lineup.captain,) + lineup.utils]
    # Normalize to max-min count
    counts = pd.Series(teams).value_counts().tolist()
    mx = max(counts)
    mn = min(counts) if len(counts) > 1 else 0
    return mx, mn


def is_valid_lineup(proj_lookup: Dict[str, Dict], lineup: Lineup, cons: Constraints) -> bool:
    players = (lineup.captain,) + lineup.utils
    if len(set(players)) != 6:
        return False

    sal = lineup_salary(proj_lookup, lineup)
    if sal > cons.salary_cap or sal < cons.min_salary:
        return False

    teams = [proj_lookup[p]["Team"] for p in players]
    counts = pd.Series(teams).value_counts()
    mx = int(counts.max())
    if cons.disallow_6_0 and mx == 6:
        return False
    if mx > cons.max_from_one_team:
        return False
    return True


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)


def _weighted_choice(rng: np.random.Generator, items: List[str], weights: np.ndarray) -> str:
    idx = rng.choice(len(items), p=weights)
    return items[int(idx)]


def _util_score(row: pd.Series, k_ceiling: float = 1.5, own_fade: float = 0.25) -> float:
    m = float(row["Projection"])
    sd = float(row["Std Dev"])
    base = m + k_ceiling * sd
    own = row.get("Ownership %", np.nan)
    own = float(own) if pd.notna(own) else 25.0
    return base * (1.0 - own_fade * (own / 100.0))


def generate_candidates(
    proj: pd.DataFrame,
    n_candidates: int = 3000,
    cons: Constraints = Constraints(),
    captain_pool: int = 10,
    util_pool: int = 35,
    rng_seed: int = 42,
) -> pd.DataFrame:
    df = validate_projections(proj)
    rng = np.random.default_rng(rng_seed)

    proj_lookup = df.set_index("Player").to_dict(orient="index")

    df["cpt_score"] = df.apply(captain_score, axis=1)
    cap_df = df.sort_values("cpt_score", ascending=False).head(captain_pool).reset_index(drop=True)
    cap_items = cap_df["Player"].tolist()
    cap_w = _softmax(cap_df["cpt_score"].values, temp=1.0)

    df["util_score"] = df.apply(_util_score, axis=1)
    util_df = df.sort_values("util_score", ascending=False).head(util_pool).reset_index(drop=True)
    util_items = util_df["Player"].tolist()
    util_w = _softmax(util_df["util_score"].values, temp=1.0)

    seen = set()
    rows = []
    attempts = 0
    max_attempts = max(25000, n_candidates * 20)

    while len(rows) < n_candidates and attempts < max_attempts:
        attempts += 1
        cpt = _weighted_choice(rng, cap_items, cap_w)

        utils = []
        for _ in range(200):
            p = _weighted_choice(rng, util_items, util_w)
            if p != cpt and p not in utils:
                utils.append(p)
            if len(utils) == 5:
                break
        if len(utils) < 5:
            continue

        lineup = Lineup(captain=cpt, utils=tuple(utils))
        if not is_valid_lineup(proj_lookup, lineup, cons):
            continue

        key = lineup.as_dk_string()
        if key in seen:
            continue
        seen.add(key)

        mean, sd = lineup_mean_sd(proj_lookup, lineup)
        ceiling = mean + 1.75 * sd
        sal = lineup_salary(proj_lookup, lineup)
        uniq = uniqueness_score(proj_lookup, lineup)
        mx, mn = team_split(proj_lookup, lineup)

        rows.append({
            "Lineup": key,
            "CPT": lineup.captain,
            "UTIL1": lineup.utils[0],
            "UTIL2": lineup.utils[1],
            "UTIL3": lineup.utils[2],
            "UTIL4": lineup.utils[3],
            "UTIL5": lineup.utils[4],
            "salary": sal,
            "salary_left": cons.salary_cap - sal,
            "proj_mean": mean,
            "proj_sd": sd,
            "proj_ceiling": ceiling,
            "uniq_score": uniq,
            "team_split": f"{mx}-{mn}",
        })

    return pd.DataFrame(rows).sort_values("proj_ceiling", ascending=False).reset_index(drop=True)


def score_candidates_monte_carlo(
    proj: pd.DataFrame,
    candidates: pd.DataFrame,
    n_sims: int = 2500,
    rng_seed: int = 7,
) -> pd.DataFrame:
    df = validate_projections(proj)
    cand = candidates.copy()

    proj_lookup = df.set_index("Player").to_dict(orient="index")

    # unique players in candidate pool
    players = set()
    cand_players = []
    for _, r in cand.iterrows():
        ps = [r["CPT"], r["UTIL1"], r["UTIL2"], r["UTIL3"], r["UTIL4"], r["UTIL5"]]
        cand_players.append(ps)
        players.update(ps)

    players = [p for p in players if p in proj_lookup]
    p_to_i = {p: i for i, p in enumerate(players)}

    means = np.array([float(proj_lookup[p]["Projection"]) for p in players], dtype=float)
    sds = np.array([float(proj_lookup[p]["Std Dev"]) for p in players], dtype=float)

    rng = np.random.default_rng(rng_seed)
    samples = rng.normal(loc=means[:, None], scale=sds[:, None], size=(len(players), n_sims))
    samples = np.clip(samples, 0, None)

    n_lineups = len(cand)
    idx = np.zeros((n_lineups, 6), dtype=int)
    mult = np.ones((n_lineups, 6), dtype=float)
    mult[:, 0] = 1.5  # captain slot

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

    # Combined "banger score" (tunable)
    med_uniq = float(cand["uniq_score"].median()) if len(cand) else 1.0
    cand["banger_score"] = (
        0.60 * cand["fp_win_prob_proxy"] +
        0.30 * cand["top1pct_prob_proxy"] +
        0.10 * (cand["uniq_score"] / (med_uniq if med_uniq > 0 else 1.0))
    )

    return cand.sort_values("banger_score", ascending=False).reset_index(drop=True)


def _lineup_players(row) -> List[str]:
    return [row["CPT"], row["UTIL1"], row["UTIL2"], row["UTIL3"], row["UTIL4"], row["UTIL5"]]


def select_final_lineups(
    scored: pd.DataFrame,
    n_lineups: int = 10,
    cons: Constraints = Constraints(),
) -> pd.DataFrame:
    out_rows = []
    player_counts = {}
    cpt_counts = {}

    def can_add(row) -> bool:
        ps = _lineup_players(row)

        for p in ps:
            if player_counts.get(p, 0) / max(1, n_lineups) >= cons.max_same_player_across_lineups:
                return False

        cpt = row["CPT"]
        if cpt_counts.get(cpt, 0) / max(1, n_lineups) >= cons.max_same_captain_across_lineups:
            return False

        # Diversity: must differ by at least K players from each selected lineup
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
        raise RuntimeError("No lineups selected. Loosen min_salary / exposure / diversity.")

    return pd.DataFrame(out_rows).reset_index(drop=True)


def build_lineups(
    proj: pd.DataFrame,
    n_lineups: int = 10,
    n_candidates: int = 3000,
    n_sims: int = 2500,
    cons: Constraints = Constraints(),
    rng_seed: int = 42,
) -> pd.DataFrame:
    candidates = generate_candidates(
        proj=proj,
        n_candidates=n_candidates,
        cons=cons,
        rng_seed=rng_seed,
    )
    scored = score_candidates_monte_carlo(
        proj=proj,
        candidates=candidates,
        n_sims=n_sims,
        rng_seed=rng_seed + 1,
    )
    final = select_final_lineups(
        scored=scored,
        n_lineups=n_lineups,
        cons=cons,
    )
    return final
