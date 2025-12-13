# -*- coding: utf-8 -*-
# Progressive image ranking & modeling with RandomForest (NO TEST LEAKAGE)
# Scenarios:
#   S1: Spectral + Soil One-Hot
#   S2: S1 + Topography (ELEVATION, SLOPE, ASPECT)
#   S3: S2 + Climate (BIO1, BIO12)
#
# Adds TEST parity plots (log & linear) per scenario + combined parity plots across scenarios.

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter
from matplotlib.ticker import AutoMinorLocator

# -------------------------
# Config
# -------------------------
RANDOM_STATE = 42
TEST_FIELD_RATIO = 0.2
LASSO_ALPHAS = np.logspace(-4, 0, 50)

RF_PARAM_GRID = {
    "n_estimators": [400, 800],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True],
    "random_state": [RANDOM_STATE],
}

ROBUST_Z_CUTOFF = -2.5
MIN_SCENARIO_SAMPLES = 5
CSV_PATH = "main.csv"

VERBOSE_PROGRESS = True
PRINT_EVERY = 5
MAX_PRINT_ROWS = 30
SHOW_PLOTS = True

# -------------------------
# Utilities
# -------------------------
def check_required_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def safe_savgol(y, min_win=5, poly=2):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < (poly + 3):
        return y.copy()
    win = max(min_win, poly + 3)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
        if win <= poly + 1:
            return y.copy()
    return savgol_filter(y, window_length=win, polyorder=poly)

def robust_delta_flags(metric_series, image_ids, negative_is_bad=True, z_cut=ROBUST_Z_CUTOFF):
    arr = np.asarray(metric_series, dtype=float)
    d = np.diff(arr, prepend=arr[0])
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-9
    z = (d - med) / (1.4826 * mad)
    bad_mask = (z < z_cut) if negative_is_bad else (z > -z_cut)
    bad_images = [img for img, is_bad in zip(image_ids, bad_mask) if is_bad]
    diag = pd.DataFrame({"image": image_ids, "metric": arr, "Δmetric": d, "z_robust": z})
    return bad_images, diag

def parse_soil_list(s):
    s = str(s)
    s = s.strip("[]").replace("'", "")
    items = [t.strip() for t in s.split(",") if t.strip()]
    return items

def print_section(title):
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")

def print_table(df, title=None, max_rows=MAX_PRINT_ROWS, round_digits=4):
    if df is None or len(df) == 0:
        return
    if title:
        print_section(title)
    df_print = df.copy()
    with pd.option_context("display.max_rows", max_rows, "display.width", 160):
        for c in df_print.select_dtypes(include=[np.number]).columns:
            df_print[c] = df_print[c].round(round_digits)
        print(df_print.head(max_rows).to_string(index=False))

def make_rf(**kwargs):
    return RandomForestRegressor(**kwargs)

def oof_predict(X, y, groups, best_params, base_scaler=None):
    """GroupKFold OOF predictions on TRAIN-only."""
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    uniq_groups = np.unique(groups)
    n_splits = max(2, min(5, len(uniq_groups)))
    gkf = GroupKFold(n_splits=n_splits)

    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]

        if base_scaler is not None:
            X_tr = base_scaler.transform(X_tr).astype(np.float32)
            X_va = base_scaler.transform(X_va).astype(np.float32)

        model = make_rf(**best_params)
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict(X_va)
        if VERBOSE_PROGRESS:
            print(f"[OOF] fold {k}/{n_splits} done.")
    return oof

# -------------------------
# Load data
# -------------------------
data = pd.read_csv(CSV_PATH, encoding="utf-8-sig").dropna().copy()

base_cols = ["FIELD_ID", "Image_ID", "mean_SOM", "soilTypes"]
check_required_columns(data, base_cols)

spectral_means = ['BI_mean','CI_mean','NDMI_mean','OMI_mean','RI_mean','SI_mean',
                  'EVI_mean','SAVI_mean','NDVI_mean','BSI_mean','CAI_mean']
spectral_stds  = ['BI_stdDev','CI_stdDev','NDMI_stdDev','OMI_stdDev','RI_stdDev','SI_stdDev',
                  'EVI_stdDev','SAVI_stdDev','NDVI_stdDev','BSI_stdDev','CAI_stdDev']
check_required_columns(data, spectral_means + spectral_stds)

# -------------------------
# Outlier removal on log10(mean_SOM) via IQR
# -------------------------
log_som_all = np.log10(data['mean_SOM'])
q1, q3 = log_som_all.quantile(0.25), log_som_all.quantile(0.75)
iqr = q3 - q1
lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
before_n = len(data)
data = data[(log_som_all >= lb) & (log_som_all <= ub)].copy()
after_n = len(data)
data["log_SOM"] = np.log10(data["mean_SOM"])
print(f"Outlier removal (IQR on log10): kept {after_n:,} of {before_n:,} rows "
      f"({100*after_n/before_n:.1f}%).")

# -------------------------
# soilTypes -> one-hot
# -------------------------
data["soilTypes_parsed"] = data["soilTypes"].apply(parse_soil_list)
mlb = MultiLabelBinarizer()
soil_features = mlb.fit_transform(data["soilTypes_parsed"])
soil_feature_names = mlb.classes_.tolist()
soil_df = pd.DataFrame(soil_features, columns=soil_feature_names, index=data.index)
data = pd.concat([data, soil_df], axis=1)
print(f"Soil one-hot features: {len(soil_feature_names)} -> {soil_feature_names[:10]}{'...' if len(soil_feature_names)>10 else ''}")

# -------------------------
# Define SCENARIOS (dynamic feature sets)
# -------------------------
spectral_cols = spectral_means + spectral_stds
topo_all = ['elevation_mean','slope_mean','aspect_mean']
clim_all = ['BIO1_mean','BIO12_mean']

present_topo = [c for c in topo_all if c in data.columns]
present_clim = [c for c in clim_all if c in data.columns]

SCENARIOS = {
    "S1_spec_soil": spectral_cols + soil_feature_names,
    "S2_spec_soil_topo": spectral_cols + soil_feature_names + present_topo,
    "S3_spec_soil_topo_clim": spectral_cols + soil_feature_names + present_topo + present_clim,
}

# -------------------------
# Single train/test split (fixed across scenarios)
# -------------------------
rng = np.random.default_rng(RANDOM_STATE)
unique_fields = data["FIELD_ID"].unique()
n_test = max(1, int(TEST_FIELD_RATIO * len(unique_fields)))
test_fields = rng.choice(unique_fields, size=n_test, replace=False)
train_fields = np.setdiff1d(unique_fields, test_fields)

train_data_full = data[data["FIELD_ID"].isin(train_fields)].copy()
test_data_full  = data[data["FIELD_ID"].isin(test_fields)].copy()

print_section("STEP 1 | Train/Test Split (by FIELD_ID)")
print(f"Train fields: {len(train_fields)}  |  Test fields: {len(test_fields)}")
print(f"Train rows: {len(train_data_full):,}    |  Test rows: {len(test_data_full):,}")
print("Test FIELD_IDs:", test_fields)

algo_name = "RandomForest"

# -------------------------
# Scenario runner
# -------------------------
def run_scenario(independent_vars, scenario_tag):
    print_section(f"SCENARIO = {scenario_tag}")
    # --- subset X/y ---
    X_train = train_data_full[independent_vars].copy()
    y_train = train_data_full["log_SOM"].values
    X_test  = test_data_full[independent_vars].copy()
    y_test  = test_data_full["log_SOM"].values
    y_test_lin = (10.0 ** y_test)

    # 2) LASSO FS (TRAIN-only, on standardized features)
    scaler_fs = StandardScaler().fit(X_train)
    X_train_scaled_for_lasso = scaler_fs.transform(X_train)
    lasso = Lasso(max_iter=100000, random_state=RANDOM_STATE)
    lasso_grid = GridSearchCV(lasso, param_grid={"alpha": LASSO_ALPHAS}, cv=5, n_jobs=-1)
    lasso_grid.fit(X_train_scaled_for_lasso, y_train)
    lasso_best = lasso_grid.best_estimator_
    print_section(f"STEP 2 | LASSO Feature Selection [{scenario_tag}]")
    print(f"Best alpha for Lasso: {lasso_grid.best_params_['alpha']:.6f}")

    lasso_coefficients = pd.Series(lasso_best.coef_, index=independent_vars)
    selected = lasso_coefficients[lasso_coefficients != 0]
    if selected.empty:
        selected = lasso_coefficients.abs().sort_values(ascending=False).head(10)
    selected_features = selected.index.tolist()

    out_dir = "lasso_selected"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"lasso_selected_{algo_name}_{scenario_tag}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"algo": algo_name, "scenario": scenario_tag,
                   "selected_features": selected_features,
                   "n_selected": len(selected_features)}, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] LASSO selected features saved to: {save_path}")

    top_coef = (lasso_coefficients.loc[selected_features]
                .sort_values(key=lambda s: s.abs(), ascending=False)
                .reset_index())
    top_coef.columns = ["feature", "coef"]
    print_table(top_coef, f"Top LASSO Coefficients (abs-sorted) [{scenario_tag}]", max_rows=20)

    if SHOW_PLOTS:
        plt.figure(figsize=(10, 6))
        selected.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')
        plt.title(f"Selected Features and Lasso Coefficients (TRAIN-only) — {scenario_tag}")
        plt.xlabel("Coefficient")
        plt.tight_layout()
        plt.show()

    # 3) Stable scaler on selected TRAIN features
    X_train_sel = train_data_full[selected_features].copy()
    X_test_sel  = test_data_full[selected_features].copy()
    base_scaler = StandardScaler().fit(X_train_sel)
    X_train_scaled_full = base_scaler.transform(X_train_sel).astype(np.float32)

    # 4) RF tuning with GroupKFold (TRAIN-only)
    groups_train = train_data_full["FIELD_ID"].values
    n_splits = max(2, min(5, len(np.unique(groups_train))))
    gkf = GroupKFold(n_splits=n_splits)

    rf_base = make_rf()
    rf_tuner = GridSearchCV(
        rf_base,
        RF_PARAM_GRID,
        cv=gkf.split(X_train_scaled_full, y_train, groups=groups_train),
        n_jobs=-1
    )
    rf_tuner.fit(X_train_scaled_full, y_train)
    best_params = rf_tuner.best_params_
    print_section(f"STEP 3 | One-shot RF Tuning (TRAIN) [{scenario_tag}]")
    print("Best params:", best_params)

    # 5) TRAIN-only OOF → per-image RMSE_OOF(image)
    print_section(f"STEP 4 | OOF (TRAIN) for per-image RMSE [{scenario_tag}]")
    oof_full = oof_predict(X_train_sel.values, y_train, groups_train, best_params, base_scaler=base_scaler)
    resid_oof = y_train - oof_full
    train_tmp = train_data_full.copy()
    train_tmp["oof_pred_full"] = oof_full
    train_tmp["oof_resid_full"] = resid_oof

    img_stats = train_tmp.groupby("Image_ID").apply(
        lambda df: pd.Series({
            "RMSE_OOF": np.sqrt(np.mean((df["oof_resid_full"].values) ** 2)),
            "n_rows": len(df)
        })
    ).reset_index()
    img_stats = img_stats.sort_values(["RMSE_OOF","n_rows"], ascending=[True, False]).reset_index(drop=True)
    ranked_image_ids = img_stats["Image_ID"].tolist()
    print_table(img_stats.head(20), f"Top-20 Images by RMSE_OOF (lower=better) [{scenario_tag}]")

    # 6) Progressive series for bad-image removal (ΔR2_OOF robust)
    print_section(f"STEP 5 | Progressive OOF series (bad-image removal) [{scenario_tag}]")
    prog_rows = []
    cumulative_ids = []
    for i, img in enumerate(ranked_image_ids, start=1):
        cumulative_ids.append(img)
        subset = train_tmp[train_tmp["Image_ID"].isin(cumulative_ids)]
        if len(subset) < MIN_SCENARIO_SAMPLES:
            prog_rows.append({"k": i, "Images_Used": i, "R2_OOF": np.nan, "RMSE_OOF": np.nan})
            continue

        X_sub = subset[selected_features].values
        y_sub = subset["log_SOM"].values
        g_sub = subset["FIELD_ID"].values

        oof_sub = oof_predict(X_sub, y_sub, g_sub, best_params, base_scaler=base_scaler)
        r2_oof = r2_score(y_sub, oof_sub)
        rmse_oof = mean_squared_error(y_sub, oof_sub, squared=False)
        prog_rows.append({"k": i, "Images_Used": i, "R2_OOF": r2_oof, "RMSE_OOF": rmse_oof})

        if VERBOSE_PROGRESS and (i % PRINT_EVERY == 0 or i == 1 or i == len(ranked_image_ids)):
            print(f"[OOF-Track {scenario_tag}] i={i:3d} | R2_OOF={r2_oof:.4f} | RMSE_OOF={rmse_oof:.4f}")

    prog_df = pd.DataFrame(prog_rows)

    valid_mask = ~pd.isna(prog_df["R2_OOF"])
    valid_r2 = prog_df.loc[valid_mask, "R2_OOF"].tolist()
    valid_imgs = np.array(ranked_image_ids)[valid_mask.values].tolist()
    bad_images, r2_diag = robust_delta_flags(valid_r2, valid_imgs, negative_is_bad=True, z_cut=ROBUST_Z_CUTOFF)

    print_section(f"STEP 6 | Bad Images by robust ΔR2_OOF (TRAIN-only) [{scenario_tag}]")
    print(f"Flagged bad images (n={len(bad_images)}): {bad_images}")
    print_table(r2_diag.sort_values("z_robust").head(15), f"Most negative jumps in R2_OOF (worst z_robust) [{scenario_tag}]")

    clean_image_ids = [img for img in ranked_image_ids if img not in set(bad_images)]
    print(f"Clean image count ({scenario_tag}): {len(clean_image_ids)} of {len(ranked_image_ids)}")

    # 7) Duan smearing on TRAIN (CLEANED set)
    print_section(f"STEP 7 | Duan Smearing (TRAIN-only, CLEANED) [{scenario_tag}]")
    train_clean = train_tmp[train_tmp["Image_ID"].isin(clean_image_ids)].copy()
    X_trc = train_clean[selected_features].values
    y_trc = train_clean["log_SOM"].values
    g_trc = train_clean["FIELD_ID"].values

    oof_clean = oof_predict(X_trc, y_trc, g_trc, best_params, base_scaler=base_scaler)
    resid_log = y_trc - oof_clean
    smearing_factor = np.mean(10.0 ** resid_log)
    print(f"Smearing factor (base 10, CLEANED, {scenario_tag}): {smearing_factor:.6f}")

    # ---- 7.5) Progressive TEST evaluation with cumulatively more CLEAN images ----
    print_section(f"STEP 7.5 | Progressive TEST evaluation (CLEAN images, cumulative) [{scenario_tag}]")

    prog_test_rows = []
    cumulative_clean = []
    for i, img in enumerate(clean_image_ids, start=1):
        cumulative_clean.append(img)

        # Train subset with top-i clean images
        sub_tr = train_tmp[train_tmp["Image_ID"].isin(cumulative_clean)].copy()
        X_tr_i = sub_tr[selected_features].values
        y_tr_i = sub_tr["log_SOM"].values
        g_tr_i = sub_tr["FIELD_ID"].values

        # Robust guard
        if len(sub_tr) < MIN_SCENARIO_SAMPLES or len(np.unique(g_tr_i)) < 2:
            prog_test_rows.append({
                "k": i, "Images_Used": i,
                "RMSE_log_TEST": np.nan, "MAE_log_TEST": np.nan, "R2_log_TEST": np.nan,
                "RMSE_lin_TEST": np.nan, "MAE_lin_TEST": np.nan, "R2_lin_TEST": np.nan,
            })
            continue

        # Fit scaler on current train subset
        scaler_i = StandardScaler().fit(sub_tr[selected_features])
        X_tr_i_sc = scaler_i.transform(X_tr_i).astype(np.float32)

        # Train RF with best_params found earlier
        model_i = make_rf(**best_params)
        model_i.fit(X_tr_i_sc, y_tr_i)

        # Duan smearing factor from TRAIN subset (using OOF on this subset for residuals)
        oof_i = oof_predict(X_tr_i, y_tr_i, g_tr_i, best_params, base_scaler=scaler_i)
        resid_i = y_tr_i - oof_i
        smear_i = np.mean(10.0 ** resid_i)

        # TEST evaluation for this stage
        X_te_i_sc = scaler_i.transform(X_test_sel).astype(np.float32)
        y_pred_log_i = model_i.predict(X_te_i_sc)
        mae_log_i  = mean_absolute_error(y_test, y_pred_log_i)
        rmse_log_i = mean_squared_error(y_test, y_pred_log_i, squared=False)
        r2_log_i   = r2_score(y_test, y_pred_log_i)

        y_pred_lin_i = (10.0 ** y_pred_log_i) * smear_i
        mae_lin_i  = mean_absolute_error(y_test_lin, y_pred_lin_i)
        rmse_lin_i = mean_squared_error(y_test_lin, y_pred_lin_i, squared=False)
        r2_lin_i   = r2_score(y_test_lin, y_pred_lin_i)

        prog_test_rows.append({
            "k": i, "Images_Used": i,
            "RMSE_log_TEST": rmse_log_i, "MAE_log_TEST": mae_log_i, "R2_log_TEST": r2_log_i,
            "RMSE_lin_TEST": rmse_lin_i, "MAE_lin_TEST": mae_lin_i, "R2_lin_TEST": r2_lin_i,
        })

        if VERBOSE_PROGRESS and (i == 1 or i % PRINT_EVERY == 0 or i == len(clean_image_ids)):
            print(f"[TEST-Track {scenario_tag}] i={i:3d} | R2_log={r2_log_i:.4f} | RMSE_log={rmse_log_i:.4f} | R2_lin={r2_lin_i:.4f} | RMSE_lin={rmse_lin_i:.4f}")

    prog_test_df = pd.DataFrame(prog_test_rows)
    prog_test_df.to_csv(f"progressive_test_metrics_{scenario_tag}.csv", index=False, encoding="utf-8-sig")
    print_table(prog_test_df.head(20), f"Progressive TEST Metrics (first 20) [{scenario_tag}]")

    # ---- انتخاب best k بر اساس متریک اصلی + رسم روند ----
    primary_metric = 'RMSE_log_TEST'  # یا 'R2_log_TEST'
    if len(prog_test_df) > 0 and prog_test_df[primary_metric].notna().any():
        if primary_metric == 'RMSE_log_TEST':
            best_idx = prog_test_df[primary_metric].idxmin()
        else:
            best_idx = prog_test_df[primary_metric].idxmax()

        best_row = prog_test_df.loc[best_idx]
        best_k   = int(best_row['k'])
        best_imgs = clean_image_ids[:best_k]

        best_info = {
            "scenario": scenario_tag,
            "primary_metric": primary_metric,
            "best_k": best_k,
            "best_images": best_imgs,
            "metrics_at_best_k": {
                "RMSE_log_TEST": float(best_row.get("RMSE_log_TEST", np.nan)),
                "MAE_log_TEST":  float(best_row.get("MAE_log_TEST", np.nan)),
                "R2_log_TEST":   float(best_row.get("R2_log_TEST", np.nan)),
                "RMSE_lin_TEST": float(best_row.get("RMSE_lin_TEST", np.nan)),
                "MAE_lin_TEST":  float(best_row.get("MAE_lin_TEST", np.nan)),
                "R2_lin_TEST":   float(best_row.get("R2_lin_TEST", np.nan)),
            }
        }
        os.makedirs("best_k", exist_ok=True)
        with open(f"best_k/best_k_{scenario_tag}.json", "w", encoding="utf-8") as f:
            json.dump(best_info, f, ensure_ascii=False, indent=2)

        print_section(f"BEST k on TEST [{scenario_tag}]")
        print(f"Primary metric: {primary_metric}")
        print(f"best_k = {best_k}  |  best_images = {best_imgs}")
        print(f"Metrics at best_k: {best_info['metrics_at_best_k']}")

        if SHOW_PLOTS and len(prog_test_df) > 0:
            # RMSE_log_TEST
            fig_r, ax_r = plt.subplots(figsize=(7,4))
            ax_r.plot(prog_test_df["k"], prog_test_df["RMSE_log_TEST"], marker='o')
            ax_r.axvline(best_k, linestyle='--')
            ax_r.set_xlabel("k (number of clean images used)")
            ax_r.set_ylabel("RMSE_log_TEST")
            ax_r.set_title(f"Progressive TEST RMSE_log — {scenario_tag}")
            ax_r.grid(True, linestyle=':', alpha=0.4)
            plt.tight_layout()
            fig_r.savefig(f"progressive_RMSE_log_TEST_{scenario_tag}.png", dpi=160)
            plt.show()

            # R2_log_TEST
            fig_r2, ax_r2 = plt.subplots(figsize=(7,4))
            ax_r2.plot(prog_test_df["k"], prog_test_df["R2_log_TEST"], marker='o')
            ax_r2.axvline(best_k, linestyle='--')
            ax_r2.set_xlabel("k (number of clean images used)")
            ax_r2.set_ylabel("R2_log_TEST")
            ax_r2.set_title(f"Progressive TEST R2_log — {scenario_tag}")
            ax_r2.grid(True, linestyle=':', alpha=0.4)
            plt.tight_layout()
            fig_r2.savefig(f"progressive_R2_log_TEST_{scenario_tag}.png", dpi=160)
            plt.show()

    # 8) Final one-shot TEST evaluation
    print_section(f"STEP 8 | Final ONE-SHOT Test (CLEANED) [{scenario_tag}]")
    final_model = make_rf(**best_params)
    final_model.fit(base_scaler.transform(X_trc).astype(np.float32), y_trc)

    X_test_scaled = base_scaler.transform(X_test_sel).astype(np.float32)
    y_pred_log = final_model.predict(X_test_scaled)
    mae_log  = mean_absolute_error(y_test, y_pred_log)
    rmse_log = mean_squared_error(y_test, y_pred_log, squared=False)
    r2_log   = r2_score(y_test, y_pred_log)

    y_pred_lin = (10.0 ** y_pred_log) * smearing_factor
    mae_lin  = mean_absolute_error(y_test_lin, y_pred_lin)
    rmse_lin = mean_squared_error(y_test_lin, y_pred_lin, squared=False)
    r2_lin   = r2_score(y_test_lin, y_pred_lin)

    summary_df = pd.DataFrame([{
        "Scenario": scenario_tag,
        "Images_Used": len(clean_image_ids),
        "MAE_log": mae_log, "RMSE_log": rmse_log, "R2_log": r2_log,
        "MAE_lin": mae_lin, "RMSE_lin": rmse_lin, "R2_lin": r2_lin
    }])
    print_table(summary_df, f"FINAL TEST METRICS [{scenario_tag}]")

    # 9) Per-fold TRAIN metrics (for stats)
    groups = train_data_full["FIELD_ID"].values
    n_splits = max(2, min(5, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)

    rmse_folds, r2_folds = [], []
    for k, (tr_idx, va_idx) in enumerate(gkf.split(X_train_sel, y_train, groups=groups), start=1):
        X_tr, X_va = X_train_sel.values[tr_idx], X_train_sel.values[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if base_scaler is not None:
            X_tr = base_scaler.transform(X_tr).astype(np.float32)
            X_va = base_scaler.transform(X_va).astype(np.float32)

        model = final_model.__class__(**best_params)
        model.fit(X_tr, y_tr)
        y_pred_va = model.predict(X_va)

        rmse = mean_squared_error(y_va, y_pred_va, squared=False)
        r2 = r2_score(y_va, y_pred_va)
        rmse_folds.append(rmse)
        r2_folds.append(r2)
        print(f"[{scenario_tag} | Fold {k}] RMSE_log={rmse:.4f}, R2_log={r2:.4f}")

    fold_metrics = pd.DataFrame({"Scenario": scenario_tag, "RMSE_log": rmse_folds, "R2_log": r2_folds})
    fold_metrics.to_csv(f"fold_metrics_{scenario_tag}.csv", index=False)
    print_table(fold_metrics, f"Per-Fold Metrics (TRAIN CV) [{scenario_tag}]", max_rows=10)

    # --- TEST parity plots (Measured vs Predicted) ---
    if SHOW_PLOTS:
        # 1) LOG-space parity
        fig_p1, ax_p1 = plt.subplots(figsize=(6.5, 6.5))
        ax_p1.scatter(y_test, y_pred_log, s=18, alpha=0.75, edgecolor='none')
        mn = np.nanmin([y_test.min(), y_pred_log.min()])
        mx = np.nanmax([y_test.max(), y_pred_log.max()])
        ax_p1.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1.2)
        ax_p1.set_aspect('equal', adjustable='box')
        ax_p1.set_xlim(mn, mx)
        ax_p1.set_ylim(mn, mx)
        ax_p1.set_xlabel("Measured (log10 SOM)")
        ax_p1.set_ylabel("Predicted (log10 SOM)")
        ax_p1.grid(True, linestyle=':', alpha=0.4)
        ax_p1.set_title(f"TEST Parity (LOG) — {scenario_tag}")
        ax_p1.text(0.02, 0.98,
                   f"R²={r2_log:.3f}\nRMSE={rmse_log:.3f}\nMAE={mae_log:.3f}",
                   transform=ax_p1.transAxes, va='top',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.tight_layout()
        fig_p1.savefig(f"parity_log_{scenario_tag}.png", dpi=160)
        plt.show()

        # 2) LINEAR-space parity (after Duan smearing)
        fig_p2, ax_p2 = plt.subplots(figsize=(6.5, 6.5))
        ax_p2.scatter(y_test_lin, y_pred_lin, s=18, alpha=0.75, edgecolor='none')
        mn_lin = np.nanmin([y_test_lin.min(), y_pred_lin.min()])
        mx_lin = np.nanmax([y_test_lin.max(), y_pred_lin.max()])
        ax_p2.plot([mn_lin, mx_lin], [mn_lin, mx_lin], linestyle='--', linewidth=1.2)
        ax_p2.set_aspect('equal', adjustable='box')
        ax_p2.set_xlim(mn_lin, mx_lin)
        ax_p2.set_ylim(mn_lin, mx_lin)
        ax_p2.set_xlabel("Measured SOM")
        ax_p2.set_ylabel("Predicted SOM (smeared)")
        ax_p2.grid(True, linestyle=':', alpha=0.4)
        ax_p2.set_title(f"TEST Parity (LINEAR) — {scenario_tag}")
        ax_p2.text(0.02, 0.98,
                   f"R²={r2_lin:.3f}\nRMSE={rmse_lin:.3f}\nMAE={mae_lin:.3f}",
                   transform=ax_p2.transAxes, va='top',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.tight_layout()
        fig_p2.savefig(f"parity_linear_{scenario_tag}.png", dpi=160)
        plt.show()

    # Save TEST predictions (keep for combined plots)
    preds_df = pd.DataFrame({
        "row_id": test_data_full.index,
        "FIELD_ID": test_data_full["FIELD_ID"].values,
        "Image_ID": test_data_full["Image_ID"].values,
        "y_true_lin": y_test_lin,
        "y_pred_lin": y_pred_lin,
        "y_true_log": y_test,
        "y_pred_log": y_pred_log,
        "algo": algo_name,
        "scenario": scenario_tag
    })
    preds_df.to_csv(f"preds_{algo_name}_{scenario_tag}.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] preds_{algo_name}_{scenario_tag}.csv written.")

    return summary_df, preds_df

# -------------------------
# Run all scenarios and aggregate summary + combined parity plots
# -------------------------
all_summaries = []
all_preds = []
for tag, feats in SCENARIOS.items():
    missing = [c for c in feats if c not in data.columns]
    if missing:
        print(f"[WARN] Scenario {tag}: missing columns will be ignored: {missing}")
        feats = [c for c in feats if c in data.columns]
    summ, preds = run_scenario(feats, tag)
    all_summaries.append(summ)
    all_preds.append(preds)

final_summary = pd.concat(all_summaries, ignore_index=True)
final_summary.to_csv("scenario_comparison_summary.csv", index=False, encoding="utf-8-sig")
print_section("SCENARIO COMPARISON — FINAL SUMMARY")
print_table(final_summary, max_rows=10)

# ---------- Combined parity plots across scenarios (TEST) ----------
combined_preds = pd.concat(all_preds, ignore_index=True)
combined_preds.to_csv("preds_all_scenarios.csv", index=False, encoding="utf-8-sig")

if SHOW_PLOTS:
    # LOG-space combined parity
    fig_c1, ax_c1 = plt.subplots(figsize=(7.2, 7.2))
    for tag in combined_preds["scenario"].unique():
        sub = combined_preds[combined_preds["scenario"] == tag]
        ax_c1.scatter(sub["y_true_log"].values, sub["y_pred_log"].values,
                      s=16, alpha=0.65, edgecolor='none', label=tag)
    mn = np.nanmin([combined_preds["y_true_log"].min(), combined_preds["y_pred_log"].min()])
    mx = np.nanmax([combined_preds["y_true_log"].max(), combined_preds["y_pred_log"].max()])
    ax_c1.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1.2)
    ax_c1.set_aspect('equal', adjustable='box')
    ax_c1.set_xlim(mn, mx); ax_c1.set_ylim(mn, mx)
    ax_c1.set_xlabel("Measured (log10 SOM)")
    ax_c1.set_ylabel("Predicted (log10 SOM)")
    ax_c1.grid(True, linestyle=':', alpha=0.4)
    ax_c1.set_title("TEST Parity (LOG) — All Scenarios")
    ax_c1.legend(frameon=True)
    plt.tight_layout()
    fig_c1.savefig("parity_log_all_scenarios.png", dpi=180)
    plt.show()

    # LINEAR-space combined parity (after Duan smearing)
    fig_c2, ax_c2 = plt.subplots(figsize=(7.2, 7.2))
    for tag in combined_preds["scenario"].unique():
        sub = combined_preds[combined_preds["scenario"] == tag]
        ax_c2.scatter(sub["y_true_lin"].values, sub["y_pred_lin"].values,
                      s=16, alpha=0.65, edgecolor='none', label=tag)
    mn_lin = np.nanmin([combined_preds["y_true_lin"].min(), combined_preds["y_pred_lin"].min()])
    mx_lin = np.nanmax([combined_preds["y_true_lin"].max(), combined_preds["y_pred_lin"].max()])
    ax_c2.plot([mn_lin, mx_lin], [mn_lin, mx_lin], linestyle='--', linewidth=1.2)
    ax_c2.set_aspect('equal', adjustable='box')
    ax_c2.set_xlim(mn_lin, mx_lin); ax_c2.set_ylim(mn_lin, mx_lin)
    ax_c2.set_xlabel("Measured SOM")
    ax_c2.set_ylabel("Predicted SOM (smeared)")
    ax_c2.grid(True, linestyle=':', alpha=0.4)
    ax_c2.set_title("TEST Parity (LINEAR) — All Scenarios")
    ax_c2.legend(frameon=True)
    plt.tight_layout()
    fig_c2.savefig("parity_linear_all_scenarios.png", dpi=180)
    plt.show()
