کد تست حساسیت (نسخهٔ تقویت‌شده)
Monte Carlo with Convergence Test + Surrogate + SHAP + Sensitivity Grid

- DB: selected (port 3307)
- Uses: sample_data_test, mode4, ErrorProb (for quality counts), ExpectedProfitLookup_Rebuilt
- Sampling: resampling from real rows (preserves correlations)
- Matching: Top-K from mode4 (bin-match then similarity), weighted aggregation of profit curves
- Convergence: grow N and stop when both max Δ|Spearman| < EPS_RHO AND/OR Δ(EONR/EFB) < EPS_VAL
- Surrogate: RandomForest (optionally XGBoost) to predict EONR and EFB
- SHAP: summary (beeswarm) + top dependence plots
- Sensitivity: run multiple configs (TOP_K, MIN_BIN_MATCH, ALPHA/BETA/GAMMA, CUMULATIVE)
Outputs per-config (CFG_ID):
  mc_samples_<CFG>.csv, mc_metrics_<CFG>.csv,
  shap_summary_EONR_<CFG>.png, shap_summary_EFB_<CFG>.png,
  shap_dependence_EONR_<feat>_<CFG>.png, shap_dependence_EFB_<feat>_<CFG>.png
"""

import os, ast, json, hashlib, warnings, gc
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hashlib import md5
from typing import Dict, List, Tuple, Optional

# ML + SHAP
from sklearn.ensemble import RandomForestRegressor
import shap
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)

# ========= SETTINGS (global defaults) =========
DB = dict(host="127.0.0.1", port=...., user="root", password="....",
          database="....", autocommit=False)

Q = 1
NUM_BINS = 5
SEED = 42
OUT_DIR = "."

# Convergence
BATCH_SIZES = [400, 800, 1600, 3200]  # if CUMULATIVE=False این‌ها تازه‌اند، اگر True تجمیع می‌شود
EPS_RHO = 0.02                         # آستانهٔ تغییر اسپیرمن
EPS_VAL = 1.0                          # آستانهٔ پایداری مقدار (EONR_mean در واحد kg/ha، EFB_mean در $/ha)
MIN_VALID_FOR_STATS = 50               # حداقل نمونهٔ معتبر برای محاسبهٔ آمار/SHAP

# Bin source: "sample" or "metadata" (placeholder)
BIN_SOURCE = "sample"

# Sensitivity grid (می‌توانی اضافه/کم کنی)
SENS_GRID = [
    dict(TOP_K=5,  MIN_BIN_MATCH=4, ALPHA=1.0, BETA=0.0, GAMMA=0.0, CUMULATIVE=True,  MODEL="RF"),
    dict(TOP_K=7,  MIN_BIN_MATCH=5, ALPHA=2.0, BETA=1.0, GAMMA=0.0, CUMULATIVE=True,  MODEL="RF"),
    dict(TOP_K=5,  MIN_BIN_MATCH=4, ALPHA=1.0, BETA=0.0, GAMMA=1.0, CUMULATIVE=True,  MODEL="RF"),
    dict(TOP_K=9,  MIN_BIN_MATCH=4, ALPHA=1.5, BETA=0.5, GAMMA=0.5, CUMULATIVE=False, MODEL="XGB"),
]

# Features & categories (یکپارچه با سیستم)
CONT_FEATS = ["ACLAY", "SOM", "CHU", "AWDR"]
CAT_PREV_LEVELS = ["Low nutrient", "Moderate nutrient", "High nutrient"]  # سه‌کلاسه
TILLAGE_LEVELS  = ["No till", "Conventionnel"]  # استاندارد و ثابت

PREV_SCORE = {"Low nutrient": 0.0, "Moderate nutrient": 0.75, "High nutrient": 1.0}
def prev_idx_for_key(prev_label): return CAT_PREV_LEVELS.index(prev_label) if prev_label in CAT_PREV_LEVELS else 1
def till_idx_for_key(till_label): return 0 if till_label == "No till" else 1

np.random.seed(SEED)

# ========= DB helpers =========
def connect():
    return mysql.connector.connect(**DB)

def parse_bins_u_data(s):
    try:
        d = ast.literal_eval(s)
        return list(map(int, d.get("bins", [])))
    except Exception:
        return []

def bin_feature_values(values, num_bins=NUM_BINS):
    vals = np.asarray(values, dtype=float)
    edges = np.linspace(np.min(vals), np.max(vals), num_bins + 1)
    means = [(edges[i] + edges[i+1]) / 2.0 for i in range(num_bins)]
    return means, edges

def closest_bin_idx(value, bin_means):
    return int(np.argmin([abs(value - m) for m in bin_means]))

def sha_bin_key(bin_indices):
    s = ",".join(map(str, bin_indices))
    return hashlib.sha256(s.encode()).hexdigest()

def standardize_tillage(x: str) -> str:
    """Normalize TILLAGE text to one of TILLAGE_LEVELS."""
    sx = str(x).strip().lower()
    if sx.startswith("no"):   # "no till", "no-till", "no_till"
        return "No till"
    return "Conventionnel"

def fetch_bins_from_metadata(conn) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    جایگزین احتمالی در آینده: لبه‌های بن را از جدول metadata بخوان.
    فعلاً به صورت placeholder به حالت sample برمی‌گردیم.
    """
    return {}

def fetch_sample_and_bins(conn):
    # 👇 ستون Prev Type را با alias برمی‌داریم تا راحت‌تر استفاده شود
    cols = ', '.join(CONT_FEATS + ['`Prev Type` AS PrevType', 'TILLAGE'])
    df = pd.read_sql(f"SELECT {cols} FROM sample_data_test", conn)

    # استانداردسازی TILLAGE
    df["TILLAGE_std"] = df["TILLAGE"].apply(standardize_tillage)

    # نگاشت Prev به سه سطح استاندارد
    def norm_prev(p):
        p = str(p).strip()
        return p if p in CAT_PREV_LEVELS else "Moderate nutrient"
    df["Prev_std"] = df["PrevType"].apply(norm_prev)   # 👈 دیگه space یا backtick نداریم

    # بن‌ها
    bin_means_cache, bin_edges_cache = {}, {}
    for f in CONT_FEATS:
        bin_means_cache[f], bin_edges_cache[f] = bin_feature_values(df[f].values, NUM_BINS)

    return df, bin_means_cache, bin_edges_cache


def user_bins_from_values(vals_dict, bin_means_cache):
    cont_idxs = [closest_bin_idx(vals_dict[f], bin_means_cache[f]) for f in CONT_FEATS]
    prev_idx  = prev_idx_for_key(vals_dict["PREV"])
    till_idx  = till_idx_for_key(vals_dict["TILLAGE"])
    return cont_idxs + [prev_idx, till_idx]

# ========= Top-K matching / data access =========
def get_mode4_topk_for_bins(conn, q_val, bin_indices, top_k, min_bin_match):
    key = sha_bin_key(bin_indices)
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT record_u_id, similarity, bins_u_data, record_j_id FROM mode4 WHERE q=%s AND bin_key=%s",
                (q_val, key))
    exact = cur.fetchall()
    if exact:
        cur.close()
        exact.sort(key=lambda r: float(r["similarity"]), reverse=True)
        for r in exact:
            r["bin_match_score"] = 6  # 4 cont + 2 cat
        return exact[:top_k], False

    cur.execute("SELECT record_u_id, similarity, bins_u_data, record_j_id FROM mode4 WHERE q=%s", (q_val,))
    pool = cur.fetchall()
    cur.close()

    def score(row):
        cand = parse_bins_u_data(row.get("bins_u_data", ""))
        return sum(1 for a, b in zip(cand, bin_indices) if a == b)

    for r in pool:
        r["bin_match_score"] = score(r)
    pool_f = [r for r in pool if r["bin_match_score"] >= min_bin_match] or pool
    pool_f.sort(key=lambda r: (r["bin_match_score"], float(r["similarity"])), reverse=True)
    return pool_f[:top_k], True

def count_errorprob_rows(conn, q_val, record_u_id):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ErrorProb WHERE q=%s AND record_u_id=%s", (q_val, record_u_id))
    c = cur.fetchone()[0]
    cur.close()
    return int(c or 0)

# کش ساده برای منحنی‌ها تا تردد DB کم شود
_CURVE_CACHE: Dict[int, pd.DataFrame] = {}
def fetch_profit_curve(conn, q_val, record_u_id) -> pd.DataFrame:
    if record_u_id in _CURVE_CACHE:
        return _CURVE_CACHE[record_u_id]
    df = pd.read_sql(
        "SELECT n_rate, expected_nrcf, efb FROM ExpectedProfitLookup_Rebuilt "
        "WHERE q=%s AND record_u_id=%s ORDER BY n_rate",
        conn, params=(q_val, record_u_id)
    )
    _CURVE_CACHE[record_u_id] = df
    return df

# ========= Weighted aggregation / decision =========
def compute_weights(similarities, bin_match_scores=None, backs=None,
                    alpha=1.0, beta=0.0, gamma=0.0) -> np.ndarray:
    sims = np.asarray(similarities, dtype=float)
    w = np.power(sims, alpha)
    if bin_match_scores is not None and beta != 0.0:
        w *= np.power(np.asarray(bin_match_scores, float), beta)
    if backs is not None and gamma != 0.0:
        w *= np.power(np.asarray(backs, float) + 1.0, gamma)  # کیفیت ErrorProb
    s = w.sum()
    return (w / s) if s > 1e-12 else np.ones_like(w) / len(w)

def weighted_aggregate_profit(curves: List[pd.DataFrame], weights: np.ndarray) -> Optional[pd.DataFrame]:
    if not curves: return None
    curves = [df for df in curves if (df is not None and not df.empty)]
    if not curves: return None
    dfs = []
    for i, df in enumerate(curves):
        d = df[['n_rate','expected_nrcf','efb']].copy()
        d.rename(columns={'expected_nrcf': f'exp_{i}', 'efb': f'efb_{i}'}, inplace=True)
        dfs.append(d)
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on='n_rate', how='inner')
    exp_cols = [c for c in base.columns if c.startswith('exp_')]
    efb_cols = [c for c in base.columns if c.startswith('efb_')]
    if not exp_cols or not efb_cols: return None
    m = min(len(exp_cols), len(efb_cols), len(weights))
    if m == 0: return None
    W = np.asarray(weights[:m], dtype=float)
    W = W / (W.sum() if W.sum() > 0 else 1.0)
    exp_mat = base[exp_cols[:m]].to_numpy(float)
    efb_mat = base[efb_cols[:m]].to_numpy(float)
    base['exp_weighted'] = (exp_mat * W).sum(axis=1)
    base['efb_weighted'] = (efb_mat * W).sum(axis=1)
    return base[['n_rate','exp_weighted','efb_weighted']]

def eonr_from_weighted_curve(wcurve: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
    if wcurve is None or wcurve.empty: return None, None
    idx = wcurve['efb_weighted'].idxmax()
    row = wcurve.loc[idx]
    return int(row['n_rate']), float(row['efb_weighted'])

# ========= Sampling (dependent via resampling from real rows) =========
def draw_sample_from_real(df_real: pd.DataFrame) -> Dict[str, float]:
    """Pick a real row for continuous; draw categorical from observed values."""
    row = df_real.sample(n=1, replace=True, random_state=None).iloc[0]
    s = {f: float(row[f]) for f in CONT_FEATS}
    prev = row['Prev_std'] if str(row['Prev_std']) in CAT_PREV_LEVELS else 'Moderate nutrient'
    till = row['TILLAGE_std']  # 'No till' یا 'Conventionnel'
    s['PREV'] = prev
    s['TILLAGE'] = till
    return s

# ========= Run MC up to N (per-config) =========
def run_monte_carlo(conn, df_real, bin_means_cache, N, cfg) -> pd.DataFrame:
    rows = []
    TOP_K = cfg["TOP_K"]; MIN_BIN_MATCH = cfg["MIN_BIN_MATCH"]
    ALPHA = cfg["ALPHA"]; BETA = cfg["BETA"]; GAMMA = cfg["GAMMA"]

    for _ in range(N):
        x = draw_sample_from_real(df_real)
        bins = user_bins_from_values(x, bin_means_cache)
        topk, used_fb = get_mode4_topk_for_bins(conn, Q, bins,
                                                top_k=TOP_K, min_bin_match=MIN_BIN_MATCH)
        if not topk:
            rows.append({**x,
                         "PREV_score": PREV_SCORE[x["PREV"]],
                         "PREV_bin": prev_idx_for_key(x["PREV"]),
                         "TILLAGE_bin": till_idx_for_key(x["TILLAGE"]),
                         "EONR": None, "EFB": None, "bin_match_used_fallback": True})
            continue

        sims = np.array([float(r["similarity"]) for r in topk], dtype=float)
        backs = []
        curves = []
        bin_match_scores = [int(r["bin_match_score"]) for r in topk]
        for r in topk:
            rid = int(r["record_u_id"])
            dfc = fetch_profit_curve(conn, Q, rid)
            if not dfc.empty:
                curves.append(dfc)
                backs.append(count_errorprob_rows(conn, Q, rid))
        if not curves:
            rows.append({**x,
                         "PREV_score": PREV_SCORE[x["PREV"]],
                         "PREV_bin": prev_idx_for_key(x["PREV"]),
                         "TILLAGE_bin": till_idx_for_key(x["TILLAGE"]),
                         "EONR": None, "EFB": None, "bin_match_used_fallback": bool(used_fb)})
            continue

        # وزن‌ها
        w = compute_weights(sims[:len(curves)],
                            bin_match_scores=bin_match_scores[:len(curves)],
                            backs=backs[:len(curves)],
                            alpha=ALPHA, beta=BETA, gamma=GAMMA)

        wcurve = weighted_aggregate_profit(curves, w)
        n_star, efb_star = eonr_from_weighted_curve(wcurve)
        rows.append({**x,
                     "PREV_score": PREV_SCORE[x["PREV"]],
                     "PREV_bin": prev_idx_for_key(x["PREV"]),
                     "TILLAGE_bin": till_idx_for_key(x["TILLAGE"]),
                     "EONR": n_star, "EFB": efb_star,
                     "bin_match_used_fallback": bool(used_fb)})

    df = pd.DataFrame(rows)
    return df.dropna(subset=["EONR","EFB"]).copy()

# ========= Convergence loop (per-config) =========
def convergence_run(cfg) -> pd.DataFrame:
    np.random.seed(SEED)
    conn = connect()
    df_real, bin_means_cache, _ = fetch_sample_and_bins(conn)

    last_spear = None
    last_stats = None
    all_df = None

    X_raw_cols = CONT_FEATS + ["PREV_score","TILLAGE_bin"]
    Y_cols = ["EONR","EFB"]

    for N in BATCH_SIZES:
        dfN = run_monte_carlo(conn, df_real, bin_means_cache, N, cfg)

        if cfg["CUMULATIVE"]:
            all_df = dfN if all_df is None else pd.concat([all_df, dfN], ignore_index=True)
        else:
            all_df = dfN

        if len(all_df) < MIN_VALID_FOR_STATS:
            print(f"N={N}: valid={len(all_df)} < {MIN_VALID_FOR_STATS}, continue …")
            continue

        spear = all_df[X_raw_cols + Y_cols].corr(method="spearman").loc[X_raw_cols, Y_cols]
        if last_spear is not None:
            delta_rho = (spear - last_spear).abs().max().max()
        else:
            delta_rho = np.nan

        # معیار پایداری مستقیمِ مقدار
        eonr_mean = float(all_df["EONR"].mean())
        efb_mean  = float(all_df["EFB"].mean())
        if last_stats is not None:
            d_eonr = abs(eonr_mean - last_stats["EONR_mean"])
            d_efb  = abs(efb_mean  - last_stats["EFB_mean"])
        else:
            d_eonr, d_efb = np.nan, np.nan

        print(f"N={N} | max Δ|Spearman|={delta_rho:.3f} | ΔEONR={d_eonr if not np.isnan(d_eonr) else 'NA'} | ΔEFB={d_efb if not np.isnan(d_efb) else 'NA'}")

        # شرط توقف
        stop_by_rho = (not np.isnan(delta_rho)) and (delta_rho < EPS_RHO)
        stop_by_val = (not np.isnan(d_eonr)) and (not np.isnan(d_efb)) and (max(d_eonr, d_efb) < EPS_VAL)
        if stop_by_rho or stop_by_val:
            print("Converged.", "rho" if stop_by_rho else "value")
            last_spear = spear
            last_stats = {"EONR_mean": eonr_mean, "EFB_mean": efb_mean}
            break

        last_spear = spear
        last_stats = {"EONR_mean": eonr_mean, "EFB_mean": efb_mean}

    conn.close()
    gc.collect()
    return all_df

# ========= Surrogate + SHAP (per-config) =========
def train_surrogate_and_shap(df_valid: pd.DataFrame, cfg_id: str, model_name: str):
    # Features با نام‌های خواناتر
    pretty_names = {
        "ACLAY": "Clay (%)",
        "SOM": "SOM (%)",
        "CHU": "CHU",
        "AWDR": "AWDR",
        "PREV_score": "Previous Crop",
        "TILLAGE_bin": "Tillage Type",
    }
    X_cols = ["ACLAY","SOM","CHU","AWDR","PREV_score","TILLAGE_bin"]
    X = df_valid[X_cols].copy()
    X.columns = [pretty_names[c] for c in X.columns]
    yE = df_valid["EONR"].values
    yF = df_valid["EFB"].values

    # مدل‌ها
    if model_name.upper() == "XGB" and HAS_XGB:
        model_E = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=4)
        model_F = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=4)
    else:
        model_E = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=SEED, n_jobs=-1)
        model_F = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=SEED, n_jobs=-1)

    model_E.fit(X, yE)
    model_F.fit(X, yF)

    # SHAP (TreeExplainer)
    expl_E = shap.TreeExplainer(model_E)
    expl_F = shap.TreeExplainer(model_F)
    shap_E = expl_E.shap_values(X)
    shap_F = expl_F.shap_values(X)

    # Summary beeswarm
    shap.summary_plot(shap_E, X, show=False)
    plt.title(f"SHAP Summary — EONR ({cfg_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_EONR_{cfg_id}.png"), dpi=150)
    plt.close()

    shap.summary_plot(shap_F, X, show=False)
    plt.title(f"SHAP Summary — EFB ({cfg_id})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"shap_summary_EFB_{cfg_id}.png"), dpi=150)
    plt.close()

    # Dependence plot برای 2 ویژگی برتر
    def top_features(shap_vals, columns, k=2):
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(-mean_abs)[:k]
        return [columns[i] for i in order]

    topE = top_features(shap_E, list(X.columns), k=2)
    topF = top_features(shap_F, list(X.columns), k=2)

    for feat in topE:
        shap.dependence_plot(feat, shap_E, X, show=False)
        plt.title(f"SHAP Dependence — EONR vs {feat} ({cfg_id})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_dependence_EONR_{feat.replace(' ','_')}_{cfg_id}.png"), dpi=150)
        plt.close()

    for feat in topF:
        shap.dependence_plot(feat, shap_F, X, show=False)
        plt.title(f"SHAP Dependence — EFB vs {feat} ({cfg_id})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_dependence_EFB_{feat.replace(' ','_')}_{cfg_id}.png"), dpi=150)
        plt.close()

    return model_E, model_F

# ========= Utils =========
def make_cfg_id(cfg: Dict) -> str:
    key = "|".join(f"{k}={cfg[k]}" for k in sorted(cfg.keys()))
    return md5(key.encode()).hexdigest()[:8]

def save_metrics_and_samples(df: pd.DataFrame, cfg_id: str,
                             delta_rho: Optional[float],
                             d_eonr: Optional[float], d_efb: Optional[float],
                             cfg: Dict):
    metrics = {
        "config_id": cfg_id,
        "Q": Q,
        "TOP_K": cfg["TOP_K"],
        "MIN_BIN_MATCH": cfg["MIN_BIN_MATCH"],
        "ALPHA": cfg["ALPHA"],
        "BETA": cfg["BETA"],
        "GAMMA": cfg["GAMMA"],
        "CUMULATIVE": cfg["CUMULATIVE"],
        "MODEL": cfg["MODEL"],
        "N_valid": len(df),
        "EONR_mean": float(df["EONR"].mean()) if len(df) else np.nan,
        "EONR_std":  float(df["EONR"].std())  if len(df) else np.nan,
        "EFB_mean":  float(df["EFB"].mean())  if len(df) else np.nan,
        "EFB_std":   float(df["EFB"].std())   if len(df) else np.nan,
        "rho_max_abs_change": float(delta_rho) if (delta_rho is not None and not np.isnan(delta_rho)) else np.nan,
        "delta_EONR_mean": float(d_eonr) if (d_eonr is not None and not np.isnan(d_eonr)) else np.nan,
        "delta_EFB_mean":  float(d_efb)  if (d_efb  is not None and not np.isnan(d_efb))  else np.nan,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, f"mc_metrics_{cfg_id}.csv"), index=False)
    df.assign(config_id=cfg_id).to_csv(os.path.join(OUT_DIR, f"mc_samples_{cfg_id}.csv"),
                                       index=False, encoding="utf-8-sig")

# ========= Main (grid over configs) =========
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for cfg in SENS_GRID:
        cfg_id = make_cfg_id(cfg)
        print("\n" + "="*80)
        print(f"Running sensitivity config {cfg_id}: {cfg}")
        print("="*80)

        # اجرای همگرایی برای این کانفیگ
        df_valid = convergence_run(cfg)
        print(f"[{cfg_id}] Final valid samples: {len(df_valid)}")

        # اگر ناکافی بود، فقط خروجی CSV خلاصه را بده و برو کانفیگ بعدی
        if len(df_valid) < MIN_VALID_FOR_STATS:
            save_metrics_and_samples(df_valid, cfg_id, np.nan, np.nan, np.nan, cfg)
            print(f"[{cfg_id}] Not enough valid samples for surrogate/SHAP.")
            continue

        # برای گزارش تغییرات، یکبار دیگر روی آخرین دو پله می‌توانستیم deltaها را ثبت کنیم؛
        # ولی اینجا deltaها در لاگ حلقه چاپ شده‌اند. در متریک نهایی مقدار NA می‌گذاریم.
        save_metrics_and_samples(df_valid, cfg_id, np.nan, np.nan, np.nan, cfg)

        # Surrogate + SHAP
        try:
            train_surrogate_and_shap(df_valid, cfg_id, cfg["MODEL"])
            print(f"[{cfg_id}] Saved SHAP plots and CSVs.")
        except Exception as e:
            print(f"[{cfg_id}] SHAP failed: {e}")

        # پاک‌سازی کش منحنی‌ها برای اجرای کانفیگ بعدی
        _CURVE_CACHE.clear()
        gc.collect()

if __name__ == "__main__":
    main()
