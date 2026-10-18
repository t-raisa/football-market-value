"""Write the paper's tables and in-text numbers from results/ (and data/processed/).

Outputs (re-create after any re-run of the notebooks):
  paper/generated/numbers.tex   \\newcommand macros for every number quoted in the text
  paper/generated/tab_*.tex     LaTeX tables (booktabs)
Nothing in the paper is typed by hand, so text, tables and notebooks cannot disagree.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, PROC, OUT = ROOT / "results", ROOT / "data" / "processed", ROOT / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)


def r(name, **kw):
    return pd.read_csv(RES / f"{name}.csv", **kw)


def num(x, d=1):
    """Number with d decimals; integers get LaTeX thousands separators; no '-0.00'."""
    if d == 0:
        return f"{int(round(float(x))):,}".replace(",", "{,}")
    out = f"{float(x):.{d}f}"
    return out[1:] if re.fullmatch(r"-0\.0*", out) else out


def tex(s: str) -> str:
    """Escape plain text for LaTeX and replace the Unicode symbols used in our tables."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    for a, b in [("≥", r"$\geq$"), ("≤", r"$\leq$"), ("−", "$-$"), ("±", r"$\pm$"), ("→", r"$\to$"),
                 ("–", "--"), ("€", "EUR "), ("|", r"$|$"), ("ρ", r"$\rho$"), ("…", r"\dots{}")]:
        s = s.replace(a, b)
    return s


def ci_str(s, d=None):
    """'[0.3840, 0.4021]' -> '[0.384, 0.402]' (rounded to d decimals if given), with proper minus signs."""
    lo, hi = (float(v) for v in re.findall(r"-?\d+\.?\d*", str(s)))
    if d is None:
        d = max(len(v.split(".")[1]) if "." in v else 0 for v in re.findall(r"-?\d+\.?\d*", str(s)))
    return f"[{num(lo, d)}, {num(hi, d)}]".replace("-", "$-$")


N = {}          # macro name -> value text


def put(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), f"macro names must be letters only: {name}"
    assert name not in N, f"duplicate macro {name}"
    N[name] = value


# ----------------------------------------------------------------------------- data
raw = r("table01_raw_tables_overview").set_index("table")
put("NRawPlayers", num(raw.loc["players", "rows"], 0))
put("NRawValuations", num(raw.loc["player_valuations", "rows"], 0))
put("NRawAppearances", num(raw.loc["appearances", "rows"], 0))
put("NRawTransfers", num(raw.loc["transfers", "rows"], 0))

tr = pd.read_parquet(PROC / "transfers.parquet", columns=["transfer_fee"])
put("NTransfersFee", num((tr.transfer_fee > 0).sum(), 0))

leak = r("table02_valuation_club_leakage", index_col=0)
put("NValRowsLeakPeriod", num(leak.iloc[:, 0].sum(), 0))
put("NValRowsNoTransfer", num(leak.loc["no earlier transfer"].iloc[0], 0))
put("PctValRowsNoTransfer", num(100 * leak.loc["no earlier transfer"].iloc[0] / leak.iloc[:, 0].sum(), 1))
put("PctNoTransferIsToday", num(leak.loc["no earlier transfer", "valuation club = 2026 club (%)"], 0))
put("PctTransferIsTransfer", num(leak.loc["has an earlier transfer", "valuation club = last transfer's club (%)"], 0))

# contract-date snapshot check (same computation as notebook 01, section 3b)
pl = pd.read_parquet(PROC / "players.parquet", columns=["player_id", "contract_expiration_date"])
va = pd.read_parquet(PROC / "player_valuations.parquet", columns=["player_id", "date"])
v = va.merge(pl, on="player_id")
v = v[v.contract_expiration_date.notna()]
years_left = (v.contract_expiration_date - v.date).dt.days / 365.25
put("PctContractFuture", num(100 * (years_left > 5).mean(), 1))

cov = r("table03_league_coverage")
put("NCoreLeagues", str((cov.group == "core").sum()))
put("NNewLeagues", str((cov.group == "new_leagues").sum()))
put("CoreLeagueList", ", ".join(sorted(cov.loc[cov.group == "core", "competition_id"])))

ds = pd.read_parquet(PROC / "dataset_v1.parquet", columns=["player_id", "league_group", "split"])
put("NRows", num(len(ds), 0))
put("NPlayers", num(ds.player_id.nunique(), 0))
core = ds[ds.league_group == "core"]
put("NCoreRows", num(len(core), 0))
put("NCorePlayers", num(core.player_id.nunique(), 0))

sizes = r("table07_main_split_sizes", index_col=0)
put("NTrain", num(sizes.loc["train", "rows"], 0))
put("NVal", num(sizes.loc["val", "rows"], 0))
put("NTest", num(sizes.loc["test", "rows"], 0))
put("NTestPlayers", num(sizes.loc["test", "players"], 0))
put("NTune", num(sizes.iloc[3]["rows"], 0))
put("NNewLeagueRows", num(sizes.loc["extra test: new leagues (17)", "rows"], 0))

lt = r("table06_leakage_cut_future_test")
real = lt[lt.feature != "career_max_log_value"]
put("NLeakChecks", str(len(real)))
put("NLeakPassed", str(int(real.passed.sum())))
put("NFeatures", str(real.feature.nunique()))

unk = r("table10_unknown_club_profile", index_col=0)
put("NUnknownRows", num(unk.loc["rows", "unknown club"], 0))
put("NUnknownTest", num(unk.loc["rows in test", "unknown club"], 0))

# ----------------------------------------------------------------------------- main results
res = r("table13_test_results_core", index_col=0)
ci16 = r("table16_bootstrap_model_ci", index_col=0)
order = [("Naive (no change)", "Naive", "Naive (no change)"), ("Ridge regression", "Ridge", "Ridge regression"),
         ("Random Forest", "Random Forest", "Random forest"), ("XGBoost", "XGBoost", "XGBoost"),
         ("CatBoost", "CatBoost", "CatBoost"), ("LightGBM", "LightGBM", "LightGBM"),
         ("GRU (mean of 3 seeds)", "GRU", "GRU (mean of 3 seeds)"),
         ("LightGBM + GRU average", "LightGBM + GRU", "LightGBM + GRU (average)")]
lines = []
for key, cikey, label in order:
    x = res.loc[key]
    rank = (lambda c, d: "--" if pd.isna(x[c]) else num(x[c], d))
    skill = "--" if key.startswith("Naive") else f"{num(x['skill_vs_naive_%'])} {ci_str(ci16.loc[cikey, 'skill 95% CI'])}"
    lines.append(" & ".join([label, f"{num(x.MAE_change, 3)} {ci_str(ci16.loc[cikey, 'MAE 95% CI'], 3)}",
                             num(x.R2_change, 2), rank("Spearman_change", 2), rank("top10_rise_prec_%", 1),
                             rank("top10_fall_prec_%", 1), num(x.MAE_eur / 1e6, 2), num(x["MdAPE_%"], 1),
                             num(x["within_20%"], 1), skill]) + r" \\")
(OUT / "tab_results.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{Test results on the 14 core leagues (valuation dates from July 2024; %s observations, %s players). MAE$_\Delta$: mean absolute error of the 12-month log change, with 95\%% player-level bootstrap CI. $\rho$: Spearman correlation of predicted and true change. P@10$\uparrow$/P@10$\downarrow$: share of the 10\%% largest predicted rises (falls) that truly rose (fell); base rates %s\%% and %s\%%. MdAPE: median absolute percentage error of the value in euros. $\pm$20\%%: share predicted within 20\%% of the true value. Skill: share of the naive MAE$_\Delta$ removed.}
\label{tab:results}
\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{lccccccccc}
\toprule
Model & MAE$_\Delta$ [95\%% CI] & $R^2_\Delta$ & $\rho$ & P@10$\uparrow$ & P@10$\downarrow$ & MAE (EUR m) & MdAPE (\%%) & $\pm$20\%% (\%%) & Skill (\%%) [95\%% CI] \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table*}
""" % (N["NTest"], N["NTestPlayers"], "{rose}", "{fell}", "\n".join(lines)))

br = pd.read_parquet(PROC / "dataset_v1.parquet", columns=["y", "log_value", "split", "league_group"])
br = br[(br.split == "test") & (br.league_group == "core")]
chg = (br.y - br.log_value).to_numpy()
put("BaseRose", num(100 * (chg > 0).mean(), 1))
put("BaseFell", num(100 * (chg < 0).mean(), 1))
put("BaseSame", num(100 * (chg == 0).mean(), 1))
t = (OUT / "tab_results.tex").read_text().replace("{rose}", N["BaseRose"]).replace("{fell}", N["BaseFell"])
(OUT / "tab_results.tex").write_text(t)

for key, short in [("Naive (no change)", "Naive"), ("Ridge regression", "Ridge"), ("Random Forest", "RF"),
                   ("XGBoost", "XGB"), ("CatBoost", "Cat"), ("LightGBM", "LGB"), ("GRU (mean of 3 seeds)", "GRU"),
                   ("LightGBM + GRU average", "Ens")]:
    x = res.loc[key]
    put(f"MAE{short}", num(x.MAE_change, 3))
    put(f"Within{short}", num(x["within_20%"], 1))
    put(f"MdAPE{short}", num(x["MdAPE_%"], 1))
    if not key.startswith("Naive"):
        put(f"Skill{short}", num(x["skill_vs_naive_%"], 1))
        put(f"Rho{short}", num(x.Spearman_change, 2))
        put(f"PRise{short}", num(x["top10_rise_prec_%"], 1))
        put(f"PFall{short}", num(x["top10_fall_prec_%"], 1))
        put(f"RsqChange{short}", num(x.R2_change, 2))
put("MAEeurNaive", num(res.loc["Naive (no change)", "MAE_eur"] / 1e6, 2))
put("MAEeurLGB", num(res.loc["LightGBM", "MAE_eur"] / 1e6, 2))
a1 = r("tableA1_value_level_r2", index_col=0)
put("RsqLevelNaive", num(a1.loc["Naive (no change)", "R2_value_level"], 2))
put("RsqLevelLGB", num(a1.loc["LightGBM", "R2_value_level"], 2))
gs = r("table12_gru_seeds", index_col=0)
put("GRUsd", num(gs.loc["std", "MAE_change"], 3))
best = pd.read_json(RES / "best_params.json")
put("GRUepochs", str(int(best.loc["epochs", "GRU"])))

pd17 = r("table17_bootstrap_paired_differences")
def pair(a, b):
    row = pd17[(pd17.A == a) & (pd17.B == b)].iloc[0]
    return row
for a, b, name in [("LightGBM", "CatBoost", "LGBCat"), ("LightGBM", "XGBoost", "LGBXGB"), ("CatBoost", "XGBoost", "CatXGB"),
                   ("LightGBM", "GRU", "LGBGRU"), ("LightGBM", "LightGBM + GRU", "LGBEns"),
                   ("LightGBM", "Random Forest", "LGBRF"), ("LightGBM", "Ridge", "LGBRidge")]:
    row = pair(a, b)
    put(f"Diff{name}", num(row["MAE(A) − MAE(B)"], 4).replace("-", "$-$"))
    put(f"Diff{name}CI", f"[{num(row['95% CI low'], 4)}, {num(row['95% CI high'], 4)}]".replace("-", "$-$"))

# ----------------------------------------------------------------------------- split comparison
t8 = r("table08_random_vs_time_split", index_col=[0, 1, 2])
t18 = r("table18_split_comparison_ci", index_col=[0, 1])
t9 = r("table09_random_split_leak_mechanism", index_col=0)
rows = [("Time split (ours)", "Time split", "all test rows", "all test rows", "all"),
        ("Random rows", "Random 80/20 (rows)", "all test rows", "all test rows", "all"),
        ("", "Random 80/20 (rows)", "test rows with t ≥ 2024-07", "t ≥ 2024-07", "$t\\geq$ 7/2024"),
        ("Random players", "Random 80/20 (by player)", "all test rows", "all test rows", "all"),
        ("", "Random 80/20 (by player)", "test rows with t ≥ 2024-07", "t ≥ 2024-07", "$t\\geq$ 7/2024")]
lines = []
for label, scheme, sub8, sub18, sub_label in rows:
    nv = t8.loc[(scheme, sub8, "Naive (no change)")]
    lg = t8.loc[(scheme, sub8, "LightGBM (untuned)")]
    s = t18.loc[(scheme, sub18)]
    lines.append(" & ".join([label, sub_label, num(nv.n, 0), num(nv.MAE_change, 3), num(lg.MAE_change, 3),
                             f"{num(s['skill_vs_naive_%'])} {ci_str(s['skill 95% CI'])}",
                             num(nv["within_20%"], 1), num(lg["within_20%"], 1)]) + r" \\")
(OUT / "tab_splits.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Same model (LightGBM, fixed settings) under three splits of the core-league data. Skill with 95\%% player-level bootstrap CI.}
\label{tab:splits}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{llrccccc}
\toprule
Split & Test rows & $n$ & \multicolumn{2}{c}{MAE$_\Delta$} & Skill (\%%) & \multicolumn{2}{c}{$\pm$20\%% (\%%)} \\
 & & & Naive & LGBM & [95\%% CI] & Naive & LGBM \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(lines))
for scheme, short in [("Time split", "Time"), ("Random 80/20 (rows)", "Rows"), ("Random 80/20 (by player)", "Players")]:
    put(f"SplitSkill{short}", num(t18.loc[(scheme, "all test rows"), "skill_vs_naive_%"], 1))
    put(f"SplitMAE{short}", num(t8.loc[(scheme, "all test rows", "LightGBM (untuned)"), "MAE_change"], 3))
put("SplitSkillRowsRecent", num(t18.loc[("Random 80/20 (rows)", "t ≥ 2024-07"), "skill_vs_naive_%"], 1))
put("SplitSkillPlayersRecent", num(t18.loc[("Random 80/20 (by player)", "t ≥ 2024-07"), "skill_vs_naive_%"], 1))
put("LeakLater", num(t9.loc["Random 80/20 (rows)", "player has a LATER row in train (%)"], 0))
put("LeakExact", num(t9.loc["Random 80/20 (rows)", "target valuation is a train row's input (%)"], 0))
t18b = r("table18b_split_comparison_differences")
for i, name in enumerate(["RowsTime", "PlayersTime", "RowsPlayers", "RowsTimeRecent", "PlayersTimeRecent", "RowsPlayersRecent"]):
    put(f"SplitDiff{name}", num(t18b.iloc[i]["skill(A) − skill(B), points"], 1))
    put(f"SplitDiff{name}CI", ci_str(t18b.iloc[i]["95% CI"]))

# ----------------------------------------------------------------------------- data table
lines = []
for key, label in [("train", "Train"), ("val", "Validation"), ("test", "Test"),
                   ("extra test: new leagues (17)", "Extra: 17 new leagues"),
                   ("extra test: OTHER (known club, uncovered league)", "Extra: other clubs"),
                   ("extra test: UNKNOWN club", "Extra: unknown club")]:
    x = sizes.loc[key]
    period = f"{str(x.first_t)[:7]} -- {str(x.last_t)[:7]}"
    lines.append(f"{label} & {period} & {num(x.rows, 0)} & {num(x.players, 0)}" + r" \\")
(OUT / "tab_data.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Observations per split. Main experiments use the 14 core leagues; the extra test sets cover the same test period.}
\label{tab:data}
\footnotesize
\resizebox{\columnwidth}{!}{\begin{tabular}{lcrr}
\toprule
Split & Valuation date $t$ & Rows & Players \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(lines))

# ----------------------------------------------------------------------------- groups, extra tests, unknown club
g = r("table19_error_by_group")
def gsk(grouping, group, col="skill_%"):
    return g[(g.grouping == grouping) & (g.group == group)].iloc[0][col]
put("SkillAgeU", num(gsk("age group", "<21"), 0))
put("SkillAgeA", num(gsk("age group", "21–25"), 0))
put("SkillAgeB", num(gsk("age group", "26–29"), 0))
put("SkillAgeC", num(gsk("age group", "30+"), 0))
put("WithinAgeUNaive", num(gsk("age group", "<21", "within20 naive %"), 1))
put("WithinAgeULGB", num(gsk("age group", "<21", "within20 LightGBM %"), 1))
put("SkillGK", num(gsk("position", "Goalkeeper"), 0))
put("SkillCheap", num(gsk("value level", "cheap (<€1m)"), 0))
put("SkillExpensive", num(gsk("value level", "expensive (≥€10m)"), 0))
put("MAEBigFallNaive", num(gsk("real change", "fell >33%", "MAE naive"), 2))
put("MAEBigFallLGB", num(gsk("real change", "fell >33%", "MAE LightGBM"), 2))
put("MAEBigRiseNaive", num(gsk("real change", "rose >50%", "MAE naive"), 2))
put("MAEBigRiseLGB", num(gsk("real change", "rose >50%", "MAE LightGBM"), 2))
put("MAEStableNaive", num(gsk("real change", "within about ±10%", "MAE naive"), 2))
put("MAEStableLGB", num(gsk("real change", "within about ±10%", "MAE LightGBM"), 2))
lg = g[g.grouping == "league"]
corel, newl = lg[lg.set == "core (in training)"], lg[lg.set == "new (not in training)"]
put("SkillCoreLeagueMin", num(corel["skill_%"].min(), 0))
put("SkillCoreLeagueMax", num(corel["skill_%"].max(), 0))
newl_pos = newl[newl["skill_%"] > 0]
put("SkillNewLeagueMin", num(newl_pos["skill_%"].min(), 0))
put("SkillNewLeagueMax", num(newl_pos["skill_%"].max(), 0))
neg = newl[newl["skill_%"] <= 0]
put("NewLeagueNegative", ", ".join(neg.group) if len(neg) else "none")

t14 = r("table14_test_results_extra", index_col=[0, 1])
put("SkillNewLeaguesLGB", num(t14.loc[("new leagues (17)", "LightGBM"), "skill_vs_naive_%"], 1))
put("SkillOtherLGB", num(t14.loc[("OTHER", "LightGBM"), "skill_vs_naive_%"], 1))
t15 = r("table15_with_without_unknown_club", index_col=[0, 1])
m_wo = [i for i in t15.index if i[0] == "core test" and "WITHOUT" in i[1]][0]
m_w = [i for i in t15.index if i[0] == "core test" and "WITH unknown" in i[1]][0]
put("UnknownModel", m_wo[1].split(" trained")[0])
put("MAEcoreWithoutUnknown", num(t15.loc[m_wo, "MAE_change"], 4))
put("MAEcoreWithUnknown", num(t15.loc[m_w, "MAE_change"], 4))

# ----------------------------------------------------------------------------- SHAP and ablation
sg = r("table22_shap_group_importance", index_col=0)
for grp, short in [("value history", "Hist"), ("club/league", "Club"), ("player", "Player"),
                   ("performance", "Perf"), ("transfers", "Transfers")]:
    put(f"Shap{short}", num(sg.loc[grp, "share_%"], 0))
si = r("table21_shap_importance")
READABLE = {"age": "age", "change_since_prev_valuation": "the change since the previous valuation",
            "value_vs_league_median": "the value relative to the league median",
            "n_valuations_so_far": "the number of past valuations",
            "days_since_last_appearance": "the days since the last match",
            "value_vs_club_mean": "the value relative to the club mean", "log_value": "the current value"}
top5 = [READABLE.get(f, f.replace("_", " ")) for f in si.feature.head(5)]
put("ShapTopFive", ", ".join(top5[:-1]) + " and " + top5[-1])
fin_rank = si.set_index("feature").loc[["club_log_spend_3y", "club_log_income_3y", "club_net_spend_3y_m"], "rank"]
put("ShapFinRankMin", str(int(fin_rank.min())))
put("ShapFinRankMax", str(int(fin_rank.max())))

ab = r("table26_feature_group_ablation", index_col=0)
ab2 = r("table26b_club_spending_ablation", index_col=0)
ov = r("table26c_club_spending_overlap", index_col=0)
put("SpendTotalValueCorr", num(ov.loc["club_log_spend_3y", "club_log_total_value"], 2))
rows = [("full model (50 features)", ab, "Full model"), ("without value history", ab, "w/o value history"),
        ("without performance", ab, "w/o performance"), ("without player", ab, "w/o player attributes"),
        ("without club/league", ab, "w/o club/league"), ("without transfers", ab, "w/o player transfer history"),
        ("without club finance (3 features)", ab2, "w/o club transfer finance (3)"),
        ("without club spending only", ab2, "w/o club spending only"),
        ("value history only", ab, "Value history only"), ("value history + age", ab, "Value history + age")]
lines = []
for key, src, label in rows:
    x = src.loc[key]
    if key.startswith("full"):
        d, verdict = "--", "--"
    else:
        d = f"{num(x['ΔMAE vs full'], 4)} [{num(x['Δ 95% CI low'], 4)}, {num(x['Δ 95% CI high'], 4)}]".replace("-", "$-$")
        verdict = {"no clear difference": "n.s.", "worse than full": "worse", "better than full": "better"}[x.verdict]
    lines.append(f"{label} & {int(x.features)} & {num(x.MAE_change, 4)} & {d} & {verdict}" + r" \\")
    if key.startswith("full"):
        lines.append(r"\midrule")
(OUT / "tab_ablation.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Feature ablation (LightGBM, tuned settings, test 2024/25). $\Delta$MAE: change versus the full model with 95\%% paired player-level bootstrap CI; n.s.: CI contains 0.}
\label{tab:ablation}
\footnotesize
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrcll}
\toprule
Variant & Feat. & MAE$_\Delta$ & $\Delta$MAE [95\%% CI] & \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(lines))
for key, src, short in [("without value history", ab, "Hist"), ("without performance", ab, "Perf"),
                        ("without player", ab, "Player"), ("without club/league", ab, "Club"),
                        ("without transfers", ab, "Transfers"), ("without club finance (3 features)", ab2, "Fin"),
                        ("value history only", ab, "HistOnly")]:
    put(f"Abl{short}", num(src.loc[key, "ΔMAE vs full"], 4))
put("SkillHistOnly", num(ab.loc["value history only", "skill_vs_naive_%"], 1))

# ----------------------------------------------------------------------------- rolling origin
f24 = r("table24_rolling_folds", index_col=0)
t25 = r("table25_rolling_results", index_col=[0, 1])
pair25 = r("table25b_rolling_lightgbm_vs_ridge", index_col=0)
lines = []
for y in f24.index:
    nv, rg, lb = t25.loc[(y, "Naive")], t25.loc[(y, "Ridge")], t25.loc[(y, "LightGBM")]
    lines.append(" & ".join([y.replace("/", "/"), num(f24.loc[y, "train rows"], 0), num(nv.n, 0), num(nv.MAE_change, 3),
                             f"{num(rg['skill_vs_naive_%'])} {ci_str(rg['skill 95% CI'])}",
                             f"{num(lb['skill_vs_naive_%'])} {ci_str(lb['skill 95% CI'])}",
                             num(nv["within_20%"], 1), num(lb["within_20%"], 1)]) + r" \\")
(OUT / "tab_rolling.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Rolling-origin evaluation (LightGBM with settings fixed in advance). Skill with 95\%% player-level bootstrap CI.}
\label{tab:rolling}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrcccrr}
\toprule
Test year & Train & Test & Naive & \multicolumn{2}{c}{Skill (\%%) [95\%% CI]} & \multicolumn{2}{c}{$\pm$20\%% (\%%)} \\
 & rows & rows & MAE$_\Delta$ & Ridge & LGBM & Naive & LGBM \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(lines))
sk = t25.xs("LightGBM", level="model")["skill_vs_naive_%"]
skr = t25.xs("Ridge", level="model")["skill_vs_naive_%"]
put("RollSkillMin", num(sk.min(), 1))
put("RollSkillMax", num(sk.max(), 1))
put("RollSkillMean", num(sk.mean(), 1))
put("RollRidgeMin", num(skr.min(), 1))
put("RollRidgeMax", num(skr.max(), 1))
put("RollYears", str(len(f24)))
put("RollLGBWins", str(int((pair25.verdict == "LightGBM better").sum())))
put("RollTrainFirst", num(f24["train rows"].iloc[0], 0))
put("RollTrainLast", num(f24["train rows"].iloc[-1], 0))

# ----------------------------------------------------------------------------- undervalued check
uv = r("table23_undervalued_check", index_col=0)
u, o, yg, al = (uv.loc["undervalued (top 10% predicted rise)"], uv.loc["overvalued (top 10% predicted fall)"],
                uv.loc["simple rule: youngest 10%"], uv.loc["all core test rows"])
put("UVRose", num(u["really rose %"], 1)); put("UVRoseCI", ci_str(u["rose 95% CI"]))
put("YoungRose", num(yg["really rose %"], 1)); put("YoungRoseCI", ci_str(yg["rose 95% CI"]))
put("AllRose", num(al["really rose %"], 1))
put("OVFell", num(o["really fell %"], 1))
put("UVSold", num(u["sold for a fee within 12m %"], 1)); put("AllSold", num(al["sold for a fee within 12m %"], 1))
put("UVFeeRatio", num(u["median fee / value at t (sold)"], 1)); put("AllFeeRatio", num(al["median fee / value at t (sold)"], 2))
put("YoungFeeRatio", num(yg["median fee / value at t (sold)"], 1))
put("UVFeeAbove", num(u["fee > value at t (% of sold)"], 0)); put("AllFeeAbove", num(al["fee > value at t (% of sold)"], 0))
put("YoungFeeAbove", num(yg["fee > value at t (% of sold)"], 0))
put("UVAge", num(u["median age"], 1))

# ----------------------------------------------------------------------------- ties at exactly 20%
p = pd.read_parquet(PROC / "predictions_test.parquet")
c = p[p.test_set == "core"]
ape = np.abs(c.value_t.astype(float) / c.value_target.astype(float) - 1)
put("PctTies", num(100 * np.isclose(ape, 0.2).mean(), 1))

# ----------------------------------------------------------------------------- feature list (appendix)
fl = r("table05_feature_list")
lines = [rf"\texttt{{{tex(x.feature)}}} & {tex(x.group)} & {tex(x.description)}" + r" \\" for x in fl.itertuples()]
(OUT / "tab_features.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{The %s features. Every feature uses only records dated on or before the valuation date $t$.}
\label{tab:features}
\scriptsize
\begin{tabular}{llp{11.5cm}}
\toprule
Feature & Group & Description \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (N["NFeatures"], "\n".join(lines)))
# single-column version for the supplement (may sit on the title page)
(OUT / "tab_features_inline.tex").write_text(
    (OUT / "tab_features.tex").read_text().replace(r"\begin{table*}[t]", r"\begin{table}[!h]")
    .replace(r"\end{table*}", r"\end{table}"))

# ----------------------------------------------------------------------------- write macros
body = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in N.items())
(OUT / "numbers.tex").write_text("% Generated by scripts/make_paper_assets.py from results/ - do not edit by hand.\n"
                                 "\\providecommand{\\xspace}{}\n" + body + "\n")
print(f"wrote {len(N)} numbers and {len(list(OUT.glob('tab_*.tex')))} tables to {OUT.relative_to(ROOT)}")
