"""Leakage-free feature engineering.

One row = one player at one valuation date t. Every feature is computed only from
records dated on or before t ("end of day t"). Windowed sums (last 365 days, last 3
years) are built from per-group cumulative sums looked up with backward
`merge_asof`, so a lookup at date q can only ever see records dated <= q.

Deliberately NOT used (they describe the player *today*, not at t):
  players.csv: contract_expiration_date, international_caps, current_club_*,
               market_value_in_eur, highest_market_value_in_eur
  player_valuations.csv: current_club_id, player_club_domestic_competition_id
               (for rows with no earlier transfer these hold the 2026 club)
"""
import numpy as np
import pandas as pd

from .config import PROCESSED, HORIZON_DAYS, HORIZON_TOL_DAYS

DAY = pd.Timedelta(days=1)
LEAGUE_GAP_DAYS = 150    # a club counts as "in league L" if it played an L game in the last 150 days
CLUB_LOOKBACK_DAYS = 365  # appearances older than this are not used to decide the club at t


# ----------------------------------------------------------------------------- loading
def load_tables(cutoff=None) -> dict:
    """Load the Parquet tables we need. With `cutoff`, every dated record after it is dropped
    (used by the leakage test: features for rows with t <= cutoff must not change)."""
    rd = lambda name, cols: pd.read_parquet(PROCESSED / f"{name}.parquet", columns=cols)
    d = {
        "players": rd("players", ["player_id", "date_of_birth", "position", "sub_position",
                                  "foot", "height_in_cm", "country_of_citizenship"]),
        "vals": rd("player_valuations", ["player_id", "date", "market_value_in_eur"]),
        "apps": rd("appearances", ["player_id", "date", "player_club_id", "competition_id",
                                   "minutes_played", "goals", "assists", "yellow_cards", "red_cards"]),
        "transfers": rd("transfers", ["player_id", "transfer_date", "from_club_id", "to_club_id",
                                      "transfer_fee"]),
        "games": rd("games", ["game_id", "competition_id", "date", "home_club_id", "away_club_id"]),
        "club_games": rd("club_games", ["game_id", "club_id", "own_goals", "opponent_goals", "is_win"]),
        "comps": rd("competitions", ["competition_id", "type", "country_name"]),
    }
    for df in d.values():                          # plain strings + one datetime unit everywhere
        for c in df.columns:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df[c] = df[c].astype("string")
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].astype("datetime64[ns]")
    d["transfers"] = d["transfers"].dropna(subset=["transfer_date"])
    d["club_games"] = d["club_games"].merge(d["games"][["game_id", "date"]], on="game_id")
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        for name, col in [("vals", "date"), ("apps", "date"), ("transfers", "transfer_date"),
                          ("games", "date"), ("club_games", "date")]:
            d[name] = d[name][d[name][col] <= cutoff]
    return d


# ----------------------------------------------------------------------------- helpers
def _asof(left, right, left_on, right_on, by, tolerance=None, direction="backward"):
    """merge_asof that keeps the left row order and tolerates missing `by` keys.
    Backward direction: each left row gets the last right row with right_on <= left_on."""
    left = left.reset_index(drop=True)
    ok = left[by].notna() & left[left_on].notna()
    l = left[ok].assign(_i=np.flatnonzero(ok))
    l[by] = l[by].astype("int64")
    r = right.dropna(subset=[by, right_on]).copy()
    r[by] = r[by].astype("int64")
    out = pd.merge_asof(l.sort_values(left_on, kind="stable"), r.sort_values(right_on, kind="stable"),
                        left_on=left_on, right_on=right_on, by=by,
                        tolerance=tolerance, direction=direction)
    new_cols = [c for c in out.columns if c not in left.columns and c != "_i"]
    return out.set_index("_i")[new_cols].reindex(range(len(left))).reset_index(drop=True)


def _cum_table(events, by, date, cols):
    """Per (by, date): running totals of `cols` up to and including that date."""
    e = events[[by, date] + cols].copy()
    e[cols] = e[cols].astype("float64")
    e = e.groupby([by, date], sort=True)[cols].sum()
    return e.groupby(level=0).cumsum().reset_index().rename(columns={date: "_d"})


def _sum_between(keys, cum, by, cols, start_days, end_days):
    """Sum of `cols` over events dated in (t - start_days, t - end_days]. Only dates <= t are read."""
    hi = _asof(keys[[by]].assign(q=keys["t"] - end_days * DAY), cum, "q", "_d", by)
    lo = _asof(keys[[by]].assign(q=keys["t"] - start_days * DAY), cum, "q", "_d", by)
    return pd.DataFrame(hi[cols].fillna(0).to_numpy() - lo[cols].fillna(0).to_numpy(), columns=cols)


# ----------------------------------------------------------------------------- building blocks
def base_rows(vals):
    """All valuations with a positive value, sorted by player and date."""
    b = (vals[vals["market_value_in_eur"] > 0]
         .rename(columns={"date": "t", "market_value_in_eur": "value_t"})
         .sort_values(["player_id", "t"], kind="stable").reset_index(drop=True))
    b["log_value_t"] = np.log(b["value_t"].astype("float64"))
    return b


def add_target(b, vals):
    """Target = log of the valuation closest to t + 365 days (must lie within +/- 60 days)."""
    fut = (vals[vals["market_value_in_eur"] > 0]
           .rename(columns={"date": "target_date", "market_value_in_eur": "value_target"}))
    m = _asof(b[["player_id"]].assign(q=b["t"] + HORIZON_DAYS * DAY), fut, "q", "target_date",
              "player_id", tolerance=HORIZON_TOL_DAYS * DAY, direction="nearest")
    b["target_date"] = m["target_date"].to_numpy()
    b["value_target"] = m["value_target"].astype("float64").to_numpy()
    b["y"] = np.log(b["value_target"])
    return b


def value_history(b):
    g = b.groupby("player_id", sort=False)
    f = pd.DataFrame(index=b.index)
    f["log_value"] = b["log_value_t"]
    f["log_max_value_so_far"] = g["log_value_t"].cummax()
    f["value_vs_max"] = f["log_value"] - f["log_max_value_so_far"]
    f["n_valuations_so_far"] = g.cumcount() + 1
    f["days_since_prev_valuation"] = (b["t"] - g["t"].shift(1)).dt.days
    f["change_since_prev_valuation"] = b["log_value_t"] - g["log_value_t"].shift(1)
    f["years_since_first_valuation"] = (b["t"] - g["t"].cummin()).dt.days / 365.25
    hist = b[["player_id", "t", "log_value_t"]].rename(columns={"t": "_d", "log_value_t": "_v"})
    # Valuations come roughly every 5-6 months, so "6 months ago" = the last valuation dated
    # 120-270 days before t, and "12 months ago" = the last one dated 300-450 days before t.
    for months, newest, oldest in [(6, 120, 270), (12, 300, 450)]:
        m = _asof(b[["player_id"]].assign(q=b["t"] - newest * DAY), hist, "q", "_d", "player_id",
                  tolerance=(oldest - newest) * DAY)
        f[f"log_value_{months}m_ago"] = m["_v"].to_numpy()
        f[f"growth_{months}m"] = f["log_value"] - f[f"log_value_{months}m_ago"]
    return f


def club_at_t(keys, transfers, apps):
    """Club at t = club of the most recent of (a) the last transfer on/before t (its to_club),
    (b) the last appearance on/before t (within 365 days). NaN if neither exists."""
    tr = transfers[["player_id", "transfer_date", "to_club_id"]].rename(
        columns={"transfer_date": "tr_date", "to_club_id": "tr_club"})
    ap = apps[["player_id", "date", "player_club_id"]].rename(
        columns={"date": "ap_date", "player_club_id": "ap_club"})
    a = _asof(keys, tr, "t", "tr_date", "player_id")
    b = _asof(keys, ap, "t", "ap_date", "player_id", tolerance=CLUB_LOOKBACK_DAYS * DAY)
    use_ap = b["ap_date"].notna() & (a["tr_date"].isna() | (b["ap_date"] > a["tr_date"]))
    club = np.where(use_ap, b["ap_club"].astype("float64"), a["tr_club"].astype("float64"))
    src = np.where(use_ap, b["ap_date"], a["tr_date"])
    return pd.DataFrame({"club_t": club, "club_source_date": pd.to_datetime(src)})


def league_at_t(keys, games, comps):
    """League at t = domestic league of the club's most recent league game on/before t
    (within 150 days). 'OTHER' = club known but not playing in a covered league; 'UNKNOWN' = no club."""
    lg_ids = comps.loc[comps["type"] == "domestic_league", "competition_id"]
    lg = games[games["competition_id"].isin(lg_ids)]
    long = pd.concat([lg[["home_club_id", "date", "competition_id"]].rename(columns={"home_club_id": "club_t"}),
                      lg[["away_club_id", "date", "competition_id"]].rename(columns={"away_club_id": "club_t"})])
    long = long.rename(columns={"date": "lg_date", "competition_id": "league_t"})
    m = _asof(keys[["club_t", "t"]], long, "t", "lg_date", "club_t", tolerance=LEAGUE_GAP_DAYS * DAY)
    league = m["league_t"].astype("object")
    league = league.where(league.notna(), np.where(keys["club_t"].isna(), "UNKNOWN", "OTHER"))
    return pd.DataFrame({"league_t": league.astype("string").to_numpy(),
                         "league_source_date": m["lg_date"].to_numpy()})


def performance(keys, apps, comps):
    """Appearance stats in the last 365 days, split league / other, plus the 365-730 day window."""
    a = apps.merge(comps[["competition_id", "type"]], on="competition_id", how="left")
    lg = (a["type"] == "domestic_league").fillna(False).to_numpy(dtype=bool)
    a = a.assign(matches=1,
                 lg_matches=lg.astype(int), lg_minutes=a["minutes_played"].where(lg, 0),
                 lg_goals=a["goals"].where(lg, 0), lg_assists=a["assists"].where(lg, 0))
    cols = ["matches", "minutes_played", "goals", "assists", "yellow_cards", "red_cards",
            "lg_matches", "lg_minutes", "lg_goals", "lg_assists"]
    cum = _cum_table(a, "player_id", "date", cols)
    w = _sum_between(keys, cum, "player_id", cols, 365, 0)
    prev = _sum_between(keys, cum, "player_id", ["minutes_played", "goals", "assists"], 730, 365)
    last = _asof(keys[["player_id", "t"]], cum[["player_id", "_d"]].assign(last_app=cum["_d"]),
                 "t", "_d", "player_id")

    f = pd.DataFrame(index=keys.index)
    f["matches_365"] = w["matches"]
    f["minutes_365"] = w["minutes_played"]
    f["goals_365"] = w["goals"]
    f["assists_365"] = w["assists"]
    f["ga_per90_365"] = 90 * (w["goals"] + w["assists"]) / np.maximum(w["minutes_played"], 90)
    f["yellow_365"] = w["yellow_cards"]
    f["red_365"] = w["red_cards"]
    f["league_matches_365"] = w["lg_matches"]
    f["league_minutes_365"] = w["lg_minutes"]
    f["league_ga_per90_365"] = 90 * (w["lg_goals"] + w["lg_assists"]) / np.maximum(w["lg_minutes"], 90)
    f["other_comp_minutes_365"] = w["minutes_played"] - w["lg_minutes"]
    f["minutes_prev_365"] = prev["minutes_played"]          # days 365-730 before t
    f["minutes_change"] = f["minutes_365"] - f["minutes_prev_365"]
    f["ga_per90_prev_365"] = 90 * (prev["goals"] + prev["assists"]) / np.maximum(prev["minutes_played"], 90)
    f["days_since_last_appearance"] = (keys["t"] - last["last_app"]).dt.days.to_numpy()
    return f


def player_transfers(keys, transfers):
    """The player's own transfer history up to t."""
    tr = transfers.sort_values(["player_id", "transfer_date"], kind="stable").copy()
    fee = tr["transfer_fee"].fillna(0).astype("float64")
    tr["n_tr"] = tr.groupby("player_id").cumcount() + 1
    tr["last_fee"] = fee
    tr["max_fee"] = fee.groupby(tr["player_id"]).cummax()
    tr = tr[["player_id", "transfer_date", "n_tr", "last_fee", "max_fee"]]
    m = _asof(keys[["player_id", "t"]], tr, "t", "transfer_date", "player_id")
    f = pd.DataFrame(index=keys.index)
    f["n_transfers_so_far"] = m["n_tr"].fillna(0).to_numpy()
    f["days_since_last_transfer"] = (keys["t"] - m["transfer_date"]).dt.days.to_numpy()
    f["log_last_transfer_fee"] = np.log1p(m["last_fee"].astype("float64")).to_numpy()
    f["log_max_transfer_fee"] = np.log1p(m["max_fee"].astype("float64")).to_numpy()
    return f


def club_finance_and_form(keys, transfers, club_games):
    """Club spending / income on fees in the last 3 years, and club results in the last 365 days."""
    paid = transfers[transfers["transfer_fee"] > 0]
    buy = _cum_table(paid.rename(columns={"to_club_id": "club_t"}), "club_t", "transfer_date", ["transfer_fee"])
    sell = _cum_table(paid.rename(columns={"from_club_id": "club_t"}), "club_t", "transfer_date", ["transfer_fee"])
    spend = _sum_between(keys, buy, "club_t", ["transfer_fee"], 3 * 365, 0)["transfer_fee"]
    income = _sum_between(keys, sell, "club_t", ["transfer_fee"], 3 * 365, 0)["transfer_fee"]

    cg = club_games.assign(n=1, gd=club_games["own_goals"].astype(float) - club_games["opponent_goals"])
    cg = cg.rename(columns={"club_id": "club_t"})
    form = _sum_between(keys, _cum_table(cg, "club_t", "date", ["n", "is_win", "gd"]),
                        "club_t", ["n", "is_win", "gd"], 365, 0)
    known = keys["club_t"].notna().to_numpy()
    f = pd.DataFrame(index=keys.index)
    f["club_log_spend_3y"] = np.where(known, np.log1p(spend), np.nan)
    f["club_log_income_3y"] = np.where(known, np.log1p(income), np.nan)
    f["club_net_spend_3y_m"] = np.where(known, (spend - income) / 1e6, np.nan)   # in EUR millions
    played = form["n"].to_numpy()
    f["club_games_365"] = np.where(known, played, np.nan)
    f["club_win_rate_365"] = np.where(played > 0, form["is_win"] / np.maximum(played, 1), np.nan)
    f["club_gd_per_game_365"] = np.where(played > 0, form["gd"] / np.maximum(played, 1), np.nan)
    return f


def squad_snapshots(b):
    """Month-end snapshots of club and league strength. Snapshot at month-end S uses each player's
    latest valuation in (S-365d, S] together with the club/league that player had at that valuation.
    A row at date t then uses the latest snapshot S <= t."""
    cols = ["player_id", "t", "value_t", "log_value_t", "club_t", "league_t"]
    bt = b[cols].sort_values("t", kind="stable")
    tarr = bt["t"].to_numpy()
    clubs, leagues = [], []
    for S in pd.date_range(bt["t"].min(), bt["t"].max(), freq="ME"):
        lo = np.searchsorted(tarr, np.datetime64(S - 365 * DAY), side="right")
        hi = np.searchsorted(tarr, np.datetime64(S), side="right")
        w = bt.iloc[lo:hi].drop_duplicates("player_id", keep="last")
        c = (w.dropna(subset=["club_t"]).groupby("club_t")
              .agg(club_mean_log_value=("log_value_t", "mean"),
                   club_total_value=("value_t", "sum"),
                   club_n_valued_players=("player_id", "size")).reset_index())
        clubs.append(c.assign(snap=S))
        leagues.append(w.groupby("league_t")["log_value_t"].median()
                        .rename("league_median_log_value").reset_index().assign(snap=S))
    clubs = pd.concat(clubs, ignore_index=True)
    leagues = pd.concat(leagues, ignore_index=True)
    leagues["league_code"] = pd.factorize(leagues["league_t"])[0]
    return clubs, leagues


def squad_features(keys, clubs, leagues):
    c = _asof(keys[["club_t", "t"]], clubs, "t", "snap", "club_t", tolerance=40 * DAY)
    code = dict(zip(leagues["league_t"], leagues["league_code"]))
    k = keys[["t"]].assign(league_code=keys["league_t"].map(code))
    l = _asof(k, leagues[["league_code", "snap", "league_median_log_value"]], "t", "snap", "league_code",
              tolerance=40 * DAY)
    f = pd.DataFrame(index=keys.index)
    f["club_mean_log_value"] = c["club_mean_log_value"].astype("float64").to_numpy()
    f["club_log_total_value"] = np.log1p(c["club_total_value"].astype("float64")).to_numpy()
    f["club_n_valued_players"] = c["club_n_valued_players"].astype("float64").to_numpy()
    f["value_vs_club_mean"] = keys["log_value_t"].to_numpy() - f["club_mean_log_value"]
    f["league_median_log_value"] = l["league_median_log_value"].astype("float64").to_numpy()
    f["value_vs_league_median"] = keys["log_value_t"].to_numpy() - f["league_median_log_value"]
    f["_club_snap"] = c["snap"].to_numpy()
    f["_league_snap"] = l["snap"].to_numpy()
    return f


def player_static(keys, players, comps):
    """Attributes that do not change over time (date of birth, foot, height; position is the
    current one - a small caveat noted in the paper)."""
    p = keys[["player_id", "t", "league_t"]].merge(players, on="player_id", how="left")
    country = dict(zip(comps["competition_id"], comps["country_name"]))
    league_country = p["league_t"].map(country).astype("object")
    citizen = p["country_of_citizenship"].replace({"Turkey": "Türkiye"}).astype("object")
    f = pd.DataFrame(index=keys.index)
    f["age"] = ((p["t"] - p["date_of_birth"]).dt.days / 365.25).to_numpy()
    f["height_cm"] = p["height_in_cm"].to_numpy()
    f["position"] = p["position"].to_numpy()
    f["sub_position"] = p["sub_position"].to_numpy()
    f["foot"] = p["foot"].to_numpy()
    f["is_foreign"] = np.where(league_country.isna() | citizen.isna(), np.nan,
                               (league_country.fillna("?").to_numpy()
                                != citizen.fillna("?").to_numpy()).astype(float))
    return f


# ----------------------------------------------------------------------------- full pipeline
def build_features(d: dict, verbose=True) -> pd.DataFrame:
    log = print if verbose else (lambda *a, **k: None)
    b = base_rows(d["vals"])
    log(f"base rows: {len(b):,}")
    keys = b[["player_id", "t", "log_value_t"]].copy()
    keys = pd.concat([keys, club_at_t(keys, d["transfers"], d["apps"])], axis=1)
    keys = pd.concat([keys, league_at_t(keys, d["games"], d["comps"])], axis=1)
    b = pd.concat([b, keys[["club_t", "club_source_date", "league_t", "league_source_date"]]], axis=1)
    log("club & league at t done")

    clubs, leagues = squad_snapshots(b)
    log(f"squad snapshots: {clubs['snap'].nunique()} month-ends")
    known = pd.DataFrame({"club_known": keys["club_t"].notna().astype("float64")})
    parts = [b, known,
             player_static(keys, d["players"], d["comps"]),
             value_history(b),
             performance(keys, d["apps"], d["comps"]),
             player_transfers(keys, d["transfers"]),
             club_finance_and_form(keys, d["transfers"], d["club_games"]),
             squad_features(keys, clubs, leagues)]
    log("all feature groups done")
    return pd.concat(parts, axis=1)


# ----------------------------------------------------------------------------- feature catalogue
# name: (group, plain-English description). Every one uses only records dated <= t.
FEATURE_INFO = {
    # player
    "age": ("player", "Age in years at t (from date of birth)"),
    "height_cm": ("player", "Height in cm (time-invariant)"),
    "position": ("player", "Main position: Goalkeeper / Defender / Midfield / Attack"),
    "sub_position": ("player", "Detailed position, e.g. Centre-Back, Left Winger"),
    "foot": ("player", "Preferred foot"),
    "is_foreign": ("player", "1 if citizenship differs from the country of the league at t"),
    # value history
    "log_value": ("value history", "log market value at t"),
    "log_value_6m_ago": ("value history", "log value of the last valuation dated 120-270 days before t"),
    "log_value_12m_ago": ("value history", "log value of the last valuation dated 300-450 days before t"),
    "growth_6m": ("value history", "log_value - log_value_6m_ago"),
    "growth_12m": ("value history", "log_value - log_value_12m_ago"),
    "log_max_value_so_far": ("value history", "Highest log value up to t"),
    "value_vs_max": ("value history", "log_value - log_max_value_so_far (0 = at career peak)"),
    "n_valuations_so_far": ("value history", "Number of valuations up to t"),
    "days_since_prev_valuation": ("value history", "Days since the previous valuation"),
    "change_since_prev_valuation": ("value history", "log change since the previous valuation"),
    "years_since_first_valuation": ("value history", "Years since the first valuation"),
    # performance
    "matches_365": ("performance", "Matches played in the last 365 days (all competitions)"),
    "minutes_365": ("performance", "Minutes played in the last 365 days"),
    "goals_365": ("performance", "Goals in the last 365 days"),
    "assists_365": ("performance", "Assists in the last 365 days"),
    "ga_per90_365": ("performance", "(Goals + assists) per 90 minutes, last 365 days"),
    "yellow_365": ("performance", "Yellow cards in the last 365 days"),
    "red_365": ("performance", "Red cards in the last 365 days"),
    "league_matches_365": ("performance", "Domestic-league matches in the last 365 days"),
    "league_minutes_365": ("performance", "Domestic-league minutes in the last 365 days"),
    "league_ga_per90_365": ("performance", "Domestic-league (goals + assists) per 90, last 365 days"),
    "other_comp_minutes_365": ("performance", "Minutes in cups / European competitions, last 365 days"),
    "minutes_prev_365": ("performance", "Minutes played 365-730 days before t"),
    "minutes_change": ("performance", "minutes_365 - minutes_prev_365"),
    "ga_per90_prev_365": ("performance", "(Goals + assists) per 90, 365-730 days before t"),
    "days_since_last_appearance": ("performance", "Days since the last recorded match"),
    # player transfers
    "n_transfers_so_far": ("transfers", "Number of the player's transfers up to t"),
    "days_since_last_transfer": ("transfers", "Days since the player's last transfer"),
    "log_last_transfer_fee": ("transfers", "log(1 + fee) of the player's last transfer"),
    "log_max_transfer_fee": ("transfers", "log(1 + highest fee) ever paid for the player up to t"),
    # club & league
    "club_known": ("club/league", "1 if the club at t is known (from a transfer or match on/before t), else 0"),
    "league_t": ("club/league", "League at t (from the club's league games); OTHER / UNKNOWN otherwise"),
    "club_log_spend_3y": ("club/league", "log(1 + fees paid by the club) in the last 3 years"),
    "club_log_income_3y": ("club/league", "log(1 + fees received by the club) in the last 3 years"),
    "club_net_spend_3y_m": ("club/league", "Club net transfer spend in the last 3 years (EUR million)"),
    "club_games_365": ("club/league", "Club matches in the data in the last 365 days"),
    "club_win_rate_365": ("club/league", "Club win rate in the last 365 days"),
    "club_gd_per_game_365": ("club/league", "Club goal difference per game, last 365 days"),
    "club_mean_log_value": ("club/league", "Mean log value of the club's players (latest month-end snapshot)"),
    "club_log_total_value": ("club/league", "log total squad value (latest month-end snapshot)"),
    "club_n_valued_players": ("club/league", "Number of valued players at the club (snapshot)"),
    "value_vs_club_mean": ("club/league", "log_value - club_mean_log_value"),
    "league_median_log_value": ("club/league", "Median log value in the player's league (snapshot)"),
    "value_vs_league_median": ("club/league", "log_value - league_median_log_value"),
}
FEATURES = list(FEATURE_INFO)
CATEGORICAL = ["position", "sub_position", "foot", "league_t"]
