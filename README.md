"""
build_credit_cycle_markov.py
----------------------------
Pull public FRED data, label each quarter with a credit-cycle stage, count the observed
transitions, and export the transition matrices your Monte Carlo needs.

Run:   python build_credit_cycle_markov.py
Needs: pip install pandas numpy openpyxl     (no API key — FRED's CSV endpoint is public)

Pipeline
  1. FETCH    quarterly series from FRED
  2. LABEL    each quarter -> Expansion / LateCycle / Contraction / Recovery  (rules below)
  3. COUNT    N[i, j] = number of quarters in state i followed by state j
  4. MATRIX   P = row-normalised counts (quarterly); P^4 (annual); stationary and terminal probabilities
  5. EXPORT   everything to credit_cycle_markov.xlsx
"""

import numpy as np
import pandas as pd

# =============================================================== 1. SETTINGS
START = "1990-01-01"            # SLOOS begins 1990; Baa spread and recessions go back further

SERIES = {                      # FRED id -> short name
    "BAA10Y":   "baa_spread",   # Moody's Baa yield minus 10-yr Treasury (daily) — credit risk premium
    "DRTSCILM": "sloos_tight",  # SLOOS: net % of banks tightening C&I standards (quarterly, 1990-)
    "USREC":    "nber_rec",     # NBER recession indicator (monthly, 0/1) — used as a sanity check only
}
# Optional CRE-specific lending standards, spliced (all-CRE to 2013Q3, then nonfarm-nonresidential).
USE_CRE_SLOOS = False
CRE_SERIES = {"DRTSCREL": "cre_old", "SUBLPDRCSN": "cre_new"}

STATES = ["Expansion", "LateCycle", "Contraction", "Recovery"]   # ordinal cycle order

# Labelling thresholds (the judgment inputs — move these and watch the matrix change)
HIGH_SPREAD_PCT = 0.70   # spread above this full-sample percentile = stressed
MID_SPREAD_PCT  = 0.50   # spread above this = elevated
TIGHT_STRONG    = 20.0   # SLOOS net % tightening that signals a credit crunch
TIGHT_MILD      = 10.0   # SLOOS net % tightening that signals late cycle
SPREAD_RISING   = 0.25   # 4-quarter change in spread (pct points) that signals late cycle
MIN_RUN         = 2      # smooth out single-quarter blips: runs shorter than this get absorbed

LAPLACE = 0.5            # prior pseudo-count added to every cell before normalising (0 = raw counts)
HOLD_YEARS = 5           # horizon for the terminal-probability table
OUTPUT = "credit_cycle_markov.xlsx"


# =============================================================== 2. FETCH
def fred(series_id: str) -> pd.Series:
    """Download one FRED series as a date-indexed float Series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, na_values=".")
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def fetch_quarterly() -> pd.DataFrame:
    """Pull every series and put it on a common quarterly grid."""
    q = pd.DataFrame()
    q["baa_spread"]  = fred("BAA10Y").resample("QE").mean()          # average level over the quarter
    q["sloos_tight"] = fred("DRTSCILM").resample("QE").last()        # survey value for the quarter
    q["nber_rec"]    = fred("USREC").resample("QE").max()            # 1 if any month in recession

    if USE_CRE_SLOOS:
        old = fred("DRTSCREL").resample("QE").last()
        new = fred("SUBLPDRCSN").resample("QE").last()
        q["sloos_tight"] = old.combine_first(new)                    # old series where it exists, new after

    q = q.loc[START:].dropna(subset=["baa_spread", "sloos_tight"])
    return q


# =============================================================== 3. LABEL
def label_states(q: pd.DataFrame) -> pd.DataFrame:
    """Turn the indicators into one credit-cycle stage per quarter."""
    q = q.copy()
    q["spread_pct"] = q["baa_spread"].rank(pct=True)          # where this quarter's spread sits, 0-1
    q["d_spread_4q"] = q["baa_spread"].diff(4)                # change over the last year

    stressed  = q["spread_pct"] >= HIGH_SPREAD_PCT
    elevated  = q["spread_pct"] >= MID_SPREAD_PCT
    rising    = q["d_spread_4q"] > 0
    tight     = q["sloos_tight"] > TIGHT_STRONG
    mild_tight = (q["sloos_tight"] > TIGHT_MILD) | (q["d_spread_4q"] > SPREAD_RISING)

    state = pd.Series("Expansion", index=q.index)
    state[~elevated & mild_tight]              = "LateCycle"    # spreads still low, but credit is tightening
    state[elevated & ~(stressed & (rising | tight))] = "Recovery"  # spreads elevated but easing
    state[stressed & (rising | tight)]         = "Contraction"  # spreads high and rising, or banks pulling back

    # Smooth: absorb runs shorter than MIN_RUN into the preceding state
    if MIN_RUN > 1:
        runs = (state != state.shift()).cumsum()
        run_len = state.groupby(runs).transform("size")
        prev = state.shift()
        short = (run_len < MIN_RUN) & prev.notna()
        state[short] = prev[short]

    q["state"] = pd.Categorical(state, categories=STATES, ordered=True)
    return q


# =============================================================== 4. COUNT & MATRICES
def transition_counts(state: pd.Series) -> pd.DataFrame:
    """N[i, j]: number of times state i (this quarter) was followed by state j (next quarter)."""
    pairs = pd.DataFrame({"from": state.shift(1), "to": state}).dropna()
    counts = pd.crosstab(pairs["from"], pairs["to"]).reindex(index=STATES, columns=STATES, fill_value=0)
    return counts


def transition_matrix(counts: pd.DataFrame, prior: float = 0.0) -> pd.DataFrame:
    """Row-normalise (optionally after adding a small prior to every cell)."""
    c = counts + prior
    return c.div(c.sum(axis=1), axis=0)


def stationary(P: pd.DataFrame) -> pd.Series:
    """Long-run share of time in each state: left eigenvector of P with eigenvalue 1."""
    vals, vecs = np.linalg.eig(P.values.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    v = v / v.sum()
    return pd.Series(v, index=P.index)


def terminal(P_annual: pd.DataFrame, start_state: str, years: int) -> pd.DataFrame:
    """Probability of each state at the end of each year, starting from start_state today."""
    pi = pd.Series(0.0, index=P_annual.index); pi[start_state] = 1.0
    rows = {}
    for t in range(1, years + 1):
        pi = pd.Series(pi.values @ P_annual.values, index=P_annual.index)
        rows[f"Year {t}"] = pi
    return pd.DataFrame(rows).T


# =============================================================== 5. MAIN
def main():
    q = fetch_quarterly()
    q = label_states(q)

    counts = transition_counts(q["state"])
    P_q    = transition_matrix(counts, LAPLACE)                      # quarterly
    P_a    = pd.DataFrame(np.linalg.matrix_power(P_q.values, 4),     # annual = quarterly^4
                          index=STATES, columns=STATES)
    pi_inf = stationary(P_q)
    current = str(q["state"].iloc[-1])
    term   = terminal(P_a, current, HOLD_YEARS)

    # ---- report
    pd.set_option("display.width", 120)
    print(f"Sample: {q.index[0]:%Y-%m} to {q.index[-1]:%Y-%m}  ({len(q)} quarters)   current state: {current}\n")
    print("Quarters in each state:\n", q["state"].value_counts().reindex(STATES).to_string(), "\n")
    print("Observed transition counts (rows = from, cols = to):\n", counts, "\n")
    print(f"Quarterly transition matrix (Laplace prior {LAPLACE}):\n", P_q.round(3), "\n")
    print("Annual transition matrix (P^4):\n", P_a.round(3), "\n")
    print("Stationary distribution:\n", pi_inf.round(3).to_string(), "\n")
    print(f"Terminal probabilities from '{current}':\n", term.round(3), "\n")
    print("Sanity check — share of NBER recession quarters labelled Contraction:",
          round(float(q.loc[q["nber_rec"] == 1, "state"].eq("Contraction").mean()), 2))

    # ---- export
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as xw:
        q.to_excel(xw, sheet_name="Data_Labels")
        counts.to_excel(xw, sheet_name="Counts")
        P_q.to_excel(xw, sheet_name="P_quarterly")
        P_a.to_excel(xw, sheet_name="P_annual")
        pi_inf.to_frame("stationary").to_excel(xw, sheet_name="Stationary")
        term.to_excel(xw, sheet_name="Terminal")
        # cumulative rows of the annual matrix, ready for the Monte Carlo's MATCH() step
        P_a.cumsum(axis=1).shift(1, axis=1).fillna(0).to_excel(xw, sheet_name="CumP_annual")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
