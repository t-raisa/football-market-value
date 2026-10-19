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
                 ("–", "--"), ("€", "EUR "), ("|", r"$|$"), ("ρ", r"$\rho$"), ("…", r"\dots{}"), ("×", r"$\times$")]:
        s = s.replace(a, b)
    return s


def neg(s):
    """Proper LaTeX minus sign for a formatted number."""
    return s.replace("-", "$-$") if s.startswith("-") else s


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
import json
res = r("table13_test_results_core", index_col=0)
ci16 = r("table16_bootstrap_model_ci", index_col=0)
BEST = json.load(open(RES / "best_simple_baseline.json"))["best_simple_baseline"]
MAIN_ROWS = [("Naive (no change)", "Naive", "No change"), (BEST, BEST, r"Age $\times$ value curve"),
             ("Ridge regression", "Ridge", "Ridge"), ("Random Forest", "Random Forest", "Random forest"),
             ("XGBoost", "XGBoost", "XGBoost"), ("CatBoost", "CatBoost", "CatBoost"), ("LightGBM", "LightGBM", "LightGBM"),
             ("GRU (mean of 3 seeds)", "GRU", "GRU (small check)")]
EXTRA_ROWS = [("Age-value-league curve", "Age-value-league curve", r"Age $\times$ value $\times$ league curve"),
              ("Momentum", "Momentum", "Momentum"), ("Momentum (damped)", "Momentum (damped)", r"Momentum, damped ($\times$0.5)"),
              ("LightGBM + GRU average", "LightGBM + GRU", "LightGBM + GRU (average)")]


def result_line(key, cikey, label, full=False):
    x = res.loc[key]
    rank = (lambda c, d: "--" if pd.isna(x[c]) else num(x[c], d))
    s_naive = "--" if key.startswith("Naive") else neg(num(x["skill_vs_naive_%"], 1))
    s_curve = "--" if key == BEST else f"{neg(num(x['skill_vs_best_simple_%']))} {ci_str(ci16.loc[cikey, 'skill vs best simple 95% CI'])}"
    cells = [label, f"{num(x.MAE_change, 3)} {ci_str(ci16.loc[cikey, 'MAE 95% CI'], 3)}", s_naive, s_curve,
             rank("Spearman_change", 2), rank("top10_rise_prec_%", 1), rank("top10_fall_prec_%", 1),
             num(x["MdAPE_%"], 1), num(x["within_20%"], 1)]
    if full:
        cells += [num(x.R2_change, 2), num(x.MAE_eur / 1e6, 2)]
    return " & ".join(cells) + r" \\"


br = pd.read_parquet(PROC / "dataset_v1.parquet", columns=["y", "log_value", "split", "league_group"])
br = br[(br.split == "test") & (br.league_group == "core")]
chg = (br.y - br.log_value).to_numpy()
put("BaseRose", num(100 * (chg > 0).mean(), 1))
put("BaseFell", num(100 * (chg < 0).mean(), 1))
put("BaseSame", num(100 * (chg == 0).mean(), 1))

HEAD = (r"Model & MAE$_\Delta$ [95\% CI] & \multicolumn{2}{c}{Skill (\%) vs.} & $\rho$ & P@10$\uparrow$ & P@10$\downarrow$ "
        r"& MdAPE & $\pm$20\% ")
HEAD2 = r" & & no change & curve [95\% CI] & & (\%) & (\%) & (\%) & (\%) "
CAPTION = (r"Test results on the 14 core leagues (valuation dates from July 2024; %s observations, %s players). "
           r"MAE$_\Delta$: mean absolute error of the 12-month log change, with 95\%% player-level bootstrap CI. "
           r"Skill: share of the reference's MAE$_\Delta$ removed; \emph{curve} is the age $\times$ value curve, "
           r"the best simple baseline (chosen on validation). $\rho$: Spearman correlation of predicted and true change. "
           r"P@10$\uparrow$/P@10$\downarrow$: share of the 10\%% largest predicted rises (falls) that truly rose (fell); "
           r"base rates %s\%% and %s\%%. MdAPE: median absolute percentage error of the value in euros (main euro metric). "
           r"$\pm$20\%%: share within 20\%% of the true value (errors of exactly 20\%% count as within, because values are rounded).") % (
    N["NTest"], N["NTestPlayers"], N["BaseRose"], N["BaseFell"])
lines = [result_line(*row) for row in MAIN_ROWS]
lines.insert(2, r"\midrule")
(OUT / "tab_results.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{%s}
\label{tab:results}
\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}
\toprule
%s\\
%s\\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table*}
""" % (CAPTION, HEAD, HEAD2, "\n".join(lines)))

# full version for the supplement (all simple baselines, the ensemble, R^2 of the change, MAE in euros)
lines = [result_line(*row, full=True) for row in MAIN_ROWS + EXTRA_ROWS]
(OUT / "tab_s_results_full.tex").write_text(r"""\begin{table}[!h]
\centering
\caption{All models and baselines on the core-league test set (columns as in the main results table of the paper, plus $R^2$ of the change and the MAE in EUR million). Momentum sometimes predicts extreme changes, which explains its very large euro error.}
\label{tab:s-results}
\footnotesize
\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccccc}
\toprule
%s & $R^2_\Delta$ & MAE \\
%s & & (EUR m) \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % (HEAD, HEAD2, "\n".join(lines)))

for key, short in [("Naive (no change)", "Naive"), ("Ridge regression", "Ridge"), ("Random Forest", "RF"),
                   ("XGBoost", "XGB"), ("CatBoost", "Cat"), ("LightGBM", "LGB"), ("GRU (mean of 3 seeds)", "GRU"),
                   ("LightGBM + GRU average", "Ens"), (BEST, "Curve")]:
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
    if key != BEST and not key.startswith("Naive"):
        put(f"VsCurve{short}", num(x["skill_vs_best_simple_%"], 1))
for cikey, short in [("LightGBM", "LGB"), ("XGBoost", "XGB"), ("CatBoost", "Cat"), ("Ridge", "Ridge"),
                     ("Random Forest", "RF"), ("GRU", "GRU")]:
    put(f"VsCurve{short}CI", ci_str(ci16.loc[cikey, "skill vs best simple 95% CI"]))
put("SkillCurveCI", ci_str(ci16.loc[BEST, "skill 95% CI"]))
put("SkillLGBCI", ci_str(ci16.loc["LightGBM", "skill 95% CI"]))
put("SkillMomentum", num(res.loc["Momentum", "skill_vs_naive_%"], 1).replace("-", "$-$"))
put("SkillMomentumDamped", num(res.loc["Momentum (damped)", "skill_vs_naive_%"], 1).replace("-", "$-$"))
put("VsCurveLeagueCurve", num(res.loc["Age-value-league curve", "skill_vs_best_simple_%"], 1).replace("-", "$-$"))
put("VsCurveLeagueCurveCI", ci_str(ci16.loc["Age-value-league curve", "skill vs best simple 95% CI"]))
put("MAEeurNaive", num(res.loc["Naive (no change)", "MAE_eur"] / 1e6, 2))
put("MAEeurLGB", num(res.loc["LightGBM", "MAE_eur"] / 1e6, 2))
a1 = r("tableA1_value_level_r2", index_col=0)
put("RsqLevelNaive", num(a1.loc["Naive (no change)", "R2_value_level"], 2))
put("RsqLevelLGB", num(a1.loc["LightGBM", "R2_value_level"], 2))
gs = r("table12_gru_seeds", index_col=0)
put("GRUsd", num(gs.loc["std", "MAE_change"], 3))
best = pd.read_json(RES / "best_params.json")
put("GRUepochs", str(int(best.loc["epochs", "GRU"])))
t17b = r("table17b_vs_best_simple_baseline", index_col=0)
put("DiffLGBCurve", num(t17b.loc["LightGBM", "MAE(model) − MAE(best simple)"], 4).replace("-", "$-$"))
put("DiffLGBCurveCI", f"[{num(t17b.loc['LightGBM', '95% CI low'], 4)}, {num(t17b.loc['LightGBM', '95% CI high'], 4)}]".replace("-", "$-$"))

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

(OUT / "tab_data_inline.tex").write_text(                 # supplement version: no scaling, may stay on page 1
    (OUT / "tab_data.tex").read_text().replace(r"\begin{table}[t]", r"\begin{table}[!h]")
    .replace(r"\resizebox{\columnwidth}{!}{\begin{tabular}", r"\begin{tabular}").replace(r"\end{tabular}}", r"\end{tabular}"))

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
neg_lg = newl[newl["skill_%"] <= 0]
put("NewLeagueNegative", ", ".join(neg_lg.group) if len(neg_lg) else "none")

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
    if key == "without club spending only":            # snapshot attributes (notebook 06, D2)
        t36 = r("table36_without_snapshot_features", index_col=0)
        lo36, hi36 = (float(v) for v in re.findall(r"-?\d+\.\d+", t36.loc["difference (without − with)", "skill vs curve 95% CI"]))
        v36 = "n.s." if lo36 <= 0 <= hi36 else "worse"
        lines.append(f"w/o snapshot attributes (4) & 46 & {num(t36.iloc[1].MAE, 4)} & "
                     f"{num(t36.loc['difference (without − with)', 'MAE'], 4)} [{num(lo36, 5)}, {num(hi36, 4)}] & {v36}" + r" \\")
        put("SnapDiff", num(t36.loc["difference (without − with)", "MAE"], 4))
        put("SnapDiffCI", f"[{num(lo36, 5)}, {num(hi36, 4)}]")
(OUT / "tab_ablation.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Feature ablation (LightGBM, tuned settings, test 2024/25). $\Delta$MAE: change versus the full model with 95\%% paired player-level bootstrap CI; n.s.: CI contains 0. Snapshot attributes: position, sub-position, height and foot.}
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
t25c = r("table25c_rolling_lightgbm_vs_best_simple", index_col=0)
lines = []
for y in f24.index:
    nv, cv, lb = t25.loc[(y, "Naive")], t25.loc[(y, BEST)], t25.loc[(y, "LightGBM")]
    lines.append(" & ".join([y, num(f24.loc[y, "train rows"], 0), num(nv.n, 0), num(nv.MAE_change, 3),
                             num(cv["skill_vs_naive_%"], 1),
                             f"{num(lb['skill_vs_naive_%'])} {ci_str(lb['skill 95% CI'])}",
                             f"{num(t25c.loc[y, 'skill vs best simple %'], 1)} {ci_str(t25c.loc[y, 'skill 95% CI'])}"]) + r" \\")
(OUT / "tab_rolling.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Rolling-origin evaluation: six test years, each with its own training and validation period (LightGBM with settings fixed in advance). Skill with 95\%% player-level bootstrap CI.}
\label{tab:rolling}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrcccc}
\toprule
Test & Train & Test & No-change & \multicolumn{3}{c}{Skill (\%%)} \\
year & rows & rows & MAE$_\Delta$ & curve vs.\ & LGBM vs.\ & LGBM vs.\ \\
 & & & & no change & no change & curve \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{table}
""" % "\n".join(lines))
put("RollVsCurveMin", num(t25c["skill vs best simple %"].min(), 1))
put("RollVsCurveMax", num(t25c["skill vs best simple %"].max(), 1))
put("RollVsCurveMean", num(t25c["skill vs best simple %"].mean(), 1))
put("RollVsCurveWins", str(int((t25c.verdict == "LightGBM better").sum())))
cvs = t25.xs(BEST, level="model")["skill_vs_naive_%"]
put("RollCurveMin", num(cvs.min(), 1))
put("RollCurveMax", num(cvs.max(), 1))
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

# ----------------------------------------------------------------------------- review checks (notebooks 04-06)
def ci_parts(s):
    lo, hi = (float(v) for v in re.findall(r"-?\d+\.?\d*", str(s)))
    return lo, hi

# groups: skill vs the age x value curve
def gcurve(grouping, group):
    row = g[(g.grouping == grouping) & (g.group == group)].iloc[0]
    return row
for grouping, group, short in [("age group", "<21", "U"), ("age group", "21–25", "A"), ("age group", "26–29", "B"),
                               ("age group", "30+", "C"), ("real change", "fell >33%", "BigFall"),
                               ("real change", "fell 10–33%", "ModFall"), ("real change", "rose 10–50%", "ModRise"),
                               ("real change", "rose >50%", "BigRise"), ("value level", "cheap (<€1m)", "Cheap"),
                               ("value level", "expensive (≥€10m)", "Expensive")]:
    row = gcurve(grouping, group)
    put(f"GrpVsCurve{short}", neg(num(row["skill_vs_best_%"], 1)))
    put(f"GrpVsCurve{short}CI", f"[{num(row.skill_vs_best_lo, 1)}, {num(row.skill_vs_best_hi, 1)}]".replace("-", "$-$"))
put("GrpVsNaiveC", num(gcurve("age group", "30+")["skill_%"], 1))
pos = g[g.grouping == "position"]
put("GrpPosVsCurveMin", num(pos["skill_vs_best_%"].min(), 1))
put("GrpPosVsCurveMax", num(pos["skill_vs_best_%"].max(), 1))
lgc = g[(g.grouping == "league") & (g.set == "core (in training)")]
lgn = g[(g.grouping == "league") & (g.set == "new (not in training)")]
put("CoreLgPosCount", str(int((lgc.skill_vs_best_lo > 0).sum())))
put("CoreLgNeg", ", ".join(lgc.loc[lgc.skill_vs_best_hi < 0, "group"]) or "none")
put("NewLgPosCount", str(int((lgn.skill_vs_best_lo > 0).sum())))
put("NewLgNegCount", str(int((lgn.skill_vs_best_hi < 0).sum())))

# ranking vs simple age rules
rk = r("table23b_ranking_vs_age_rules", index_col=0)
put("RankRiseLGB", num(rk.loc["LightGBM: top 10% predicted rises", "precision %"], 1))
put("RankRiseYoung", num(rk.loc["Rule: youngest 10%", "precision %"], 1))
put("RankRiseDiff", num(rk.loc["LightGBM: top 10% predicted rises", "minus rule (points)"], 1))
put("RankRiseDiffCI", ci_str(rk.loc["LightGBM: top 10% predicted rises", "diff 95% CI"]))
put("RankFallLGB", num(rk.loc["LightGBM: top 10% predicted falls", "precision %"], 1))
put("RankFallOld", num(rk.loc["Rule: oldest 10%", "precision %"], 1))
put("RankFallDiff", num(rk.loc["LightGBM: top 10% predicted falls", "minus rule (points)"], 1))
put("RankFallDiffCI", ci_str(rk.loc["LightGBM: top 10% predicted falls", "diff 95% CI"]))
put("RankRiseCurve", num(rk.loc[f"{BEST}: top 10% predicted rises", "precision %"], 1))

# full-window sensitivity
fw = r("table31_sensitivity_full_window", index_col=0)
fwr = fw.loc["fully observable window only"]
put("FullWinRows", num(fwr.rows, 0))
put("FullWinVsNaive", num(fwr["LightGBM skill vs no change %"], 1))
put("FullWinVsNaiveCI", ci_str(fwr["  95% CI"]))
put("FullWinVsCurve", num(fwr["LightGBM skill vs curve %"], 1))
put("FullWinVsCurveCI", ci_str(fwr["  95% CI "]))

# forecast intervals: final uncalibrated (A) and calibrated by age group (D)
cal = r("table33_conformal_calibration")
def calrow(prefix, group):
    return cal[(cal.interval.str.startswith(prefix)) & (cal.group == group)].iloc[0]
A, D = calrow("A:", "all"), calrow("D:", "all")
B_ = calrow("B:", "all")
put("IntCovRaw", num(A["coverage %"], 1)); put("IntCovRawCI", ci_str(A["coverage 95% CI"]))
put("IntCovCal", num(D["coverage %"], 1)); put("IntCovCalCI", ci_str(D["coverage 95% CI"]))
put("IntWidthCal", num(D["median width / value at t (%)"], 0))
put("IntWidthIncrease", num(100 * (D["median width (log)"] / B_["median width (log)"] - 1), 0))
put("IntPinballCal", num(D["mean pinball"], 3))
U, O = calrow("D:", "age <21"), calrow("D:", "age 30+")
put("IntCovCalU", num(U["coverage %"], 1)); put("IntCovCalUCI", ci_str(U["coverage 95% CI"]))
put("IntCovCalC", num(O["coverage %"], 1))
put("IntBelowC", num(O["below q10 %"], 1)); put("IntAboveC", num(O["above q90 %"], 1))
put("IntAboveU", num(U["above q90 %"], 1))
iv = r("table32_forecast_intervals", index_col=0)
put("IntPinballRef", num(iv.loc["age × value percentiles", "mean pinball"], 3))
put("IntPinballRaw", num(iv.loc["LightGBM quantile", "mean pinball"], 3))
put("NCalRows", "12{,}799")
lines = []
for grp_, glabel in [("all", "all"), ("age <21", "$<$21"), ("age 30+", "30+")]:
    for prefix, label in [("A:", "final model, uncalibrated"), ("B:", "tuning-row model, uncalibrated"),
                          ("C:", "B + calibration (overall)"), ("D:", "B + calibration (by age group)")]:
        x = calrow(prefix, grp_)
        lines.append(f"{glabel} & {label} & {num(x['coverage %'], 1)} {ci_str(x['coverage 95% CI'])} & "
                     f"{num(x['below q10 %'], 1)} / {num(x['above q90 %'], 1)} & {num(x['median width (log)'], 2)} & "
                     f"{num(x['mean pinball'], 3)}" + r" \\")
    lines.append(r"\midrule")
(OUT / "tab_s_calibration.tex").write_text(r"""\begin{table}[!h]
\centering
\caption{80\%% forecast intervals before and after conformal calibration (core test 2024/25). Calibration rows: %s training observations whose targets fall between July 2023 and June 2024, i.e. before the test period, and unseen by the calibrated model.}
\label{tab:s-calibration}
\footnotesize
\begin{tabular}{llcccc}
\toprule
Age & Interval & Coverage (\%%) [95\%% CI] & Below / above (\%%) & Median width (log) & Mean pinball \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (N["NCalRows"], "\n".join(lines[:-1])))

# real-fee check
fee = r("table37_real_fee_check", index_col=0)
def fee_row(name, short):
    put(f"FeeRho{short}", num(fee.loc[name].iloc[0], 2))
    put(f"FeeRho{short}CI", ci_str(fee.loc[name].iloc[1], 2))
fee_row("LightGBM predicted change", "LGB")
fee_row("age × value curve predicted change", "Curve")
fee_row("minus age (reference)", "Age")
fee_row("true 12-month change of the market value (reference)", "True")
fee_row("difference LightGBM − curve", "DiffCurve")
fee_row("difference LightGBM − minus age", "DiffAge")
put("FeeN", "2{,}100"); put("FeePlayers", "1{,}121")

# dropped observations
dc = pd.read_csv(RES / "table27_dropped_rows_counts.csv", index_col=[0, 1])
tc = dc.loc[("14 core leagues", "test")]
put("DropRateTest", num(tc["genuine drop rate (% of complete-window rows)"], 1))
put("DropTest", num(tc["dropped: no valuation in window"], 0))
put("DropTestComplete", num(tc["kept"] + tc["dropped: no valuation in window"], 0))
put("KeptPartial", num(tc["kept (window partly after data end)"], 0))
lc = r("table29_dropped_rows_later_change", index_col=0)
k_, d_ = lc.loc["kept (12-month target), all"], lc.loc["genuine drops with a later valuation ≤18m, all"]
put("DropFell", num(d_["fell %"], 0)); put("DropRose", num(d_["rose %"], 0))
put("KeptFell", num(k_["fell %"], 0)); put("KeptRose", num(k_["rose %"], 0))
ba = pd.read_csv(RES / "table29b_dropped_rows_later_change_by_age.csv", header=[0, 1], index_col=0)
put("DropYoungMedian", num(ba.loc["<21", ("median", "dropped")], 2))
put("KeptYoungMedian", num(ba.loc["<21", ("median", "kept")], 2))
lines = []
for period in ["train", "val", "test", "all periods"]:
    x = dc.loc[("14 core leagues", period)]
    lines.append(f"{period} & {num(x['kept'], 0)} & {num(x['kept (window partly after data end)'], 0)} & "
                 f"{num(x['dropped: no valuation in window'], 0)} & {num(x['dropped: window after data end'], 0)} & "
                 f"{num(x['genuine drop rate (% of complete-window rows)'], 1)}" + r" \\")
(OUT / "tab_s_dropped.tex").write_text(r"""\begin{table}[!h]
\centering
\caption{Observations kept and dropped (14 core leagues, valuation dates from July 2014). A genuine drop has a fully observable 10--14-month window without a valuation in it; data-end drops have no target yet because the data end in June 2026.}
\label{tab:s-dropped}
\footnotesize
\begin{tabular}{lrrrrr}
\toprule
Period & Kept & Kept (window partly & Genuine & Dropped at & Genuine drop \\
 & & after data end) & drops & data end & rate (\%%) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(lines))

# 17 new leagues
nc = r("table34_new_league_feature_coverage", index_col=0)
put("NewWindowCov", num(nc.loc["17 new leagues (test)", "365-day window covered by match data (mean %)"], 1))
put("NewZeroMin", num(nc.loc["17 new leagues (test)", "minutes_365 = 0 (%)"], 1))
put("CoreZeroMin", num(nc.loc["core leagues (test)", "minutes_365 = 0 (%)"], 1))
put("NewNoApp", num(nc.loc["17 new leagues (test)", "no appearance on record (%)"], 0))
np_ = r("table35_new_leagues_without_performance")
def nlrow(test_set, model):
    return np_[(np_["test set"] == test_set) & (np_.model == model)].iloc[0]
full_nl, noperf_nl = nlrow("17 new leagues", "LightGBM (all 50 features)"), nlrow("17 new leagues", "LightGBM without performance")
put("NewVsCurve", num(full_nl["skill vs curve %"], 1)); put("NewVsCurveCI", ci_str(full_nl["  CI "]))
put("NewVsNaive", num(full_nl["skill vs no change %"], 1))
put("NewNoPerfVsCurve", num(noperf_nl["skill vs curve %"], 1)); put("NewNoPerfVsCurveCI", ci_str(noperf_nl["  CI "]))
put("NewCurveVsNaive", num(nlrow("17 new leagues", "age × value curve")["skill vs no change %"], 1))
lines = []
for ts in ["14 core leagues", "17 new leagues"]:
    for model in ["age × value curve", "LightGBM (all 50 features)", "LightGBM without performance"]:
        x = nlrow(ts, model)
        vc = "--" if model.startswith("age") else f"{num(x['skill vs curve %'], 1)} {ci_str(x['  CI '])}"
        lines.append(f"{ts} & {tex(model)} & {num(x.MAE, 4)} & {num(x['skill vs no change %'], 1)} {ci_str(x['  CI'])} & {vc}" + r" \\")
    lines.append(r"\midrule")
(OUT / "tab_s_newleagues.tex").write_text(r"""\begin{table}[!h]
\centering
\caption{Test results with and without the performance features. In the 17 new leagues, on average only %s\%% of the 365-day window has match data, and %s\%% of observations show zero minutes (core leagues: %s\%%).}
\label{tab:s-newleagues}
\footnotesize
\begin{tabular}{llccc}
\toprule
Test set & Model & MAE$_\Delta$ & Skill vs.\ no change (\%%) & Skill vs.\ curve (\%%) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (N["NewWindowCov"], N["NewZeroMin"], N["CoreZeroMin"], "\n".join(lines[:-1])))

# reproducibility counts
put("NResultTables", str(len(list((ROOT / "reference_results").glob("*.csv"))) + len(list((ROOT / "reference_results").glob("*.json")))))
put("NFigures", str(len(list((ROOT / "figures").glob("*.png")))))

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
