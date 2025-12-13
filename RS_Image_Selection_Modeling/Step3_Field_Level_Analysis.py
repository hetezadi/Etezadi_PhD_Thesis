# make_field_level.py
# Build FIELD-level predictions & metrics from existing image-level preds_*.csv
import os, glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

SHOW_PLOTS = True  # در صورت نیاز False کن

def field_level_from_preds(df):
    """گرفتن میانگین روی هر FIELD_ID در هر (algo, scenario)"""
    need_cols = {"FIELD_ID","Image_ID","y_true_log","y_pred_log","y_true_lin","y_pred_lin","algo","scenario"}
    missing = need_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in preds: {missing}")

    fld = (df
           .groupby(["algo","scenario","FIELD_ID"], as_index=False)
           .agg(y_true_log_field=("y_true_log","mean"),
                y_pred_log_field=("y_pred_log","mean"),
                y_true_lin_field=("y_true_lin","mean"),
                y_pred_lin_field=("y_pred_lin","mean"),
                n_images=("Image_ID","nunique")))
    return fld

def metrics_per_group(fld):
    rows = []
    for (algo, scen), g in fld.groupby(["algo","scenario"]):
        rmse_log = mean_squared_error(g["y_true_log_field"], g["y_pred_log_field"], squared=False)
        mae_log  = mean_absolute_error(g["y_true_log_field"], g["y_pred_log_field"])
        r2_log   = r2_score(g["y_true_log_field"], g["y_pred_log_field"])

        rmse_lin = mean_squared_error(g["y_true_lin_field"], g["y_pred_lin_field"], squared=False)
        mae_lin  = mean_absolute_error(g["y_true_lin_field"], g["y_pred_lin_field"])
        r2_lin   = r2_score(g["y_true_lin_field"], g["y_pred_lin_field"])

        rows.append({
            "Algo": algo, "Scenario": scen, "Level": "FIELD",
            "RMSE_log": rmse_log, "MAE_log": mae_log, "R2_log": r2_log,
            "RMSE_lin": rmse_lin, "MAE_lin": mae_lin, "R2_lin": r2_lin,
            "n_fields": g["FIELD_ID"].nunique(),
            "avg_images_per_field": g["n_images"].mean()
        })
    return pd.DataFrame(rows)

def plot_field_parity(fld, algo, scen):
    sub = fld[(fld["algo"]==algo) & (fld["scenario"]==scen)]
    if sub.empty: return
    # LOG
    fig, ax = plt.subplots(figsize=(6.2,6.2))
    ax.scatter(sub["y_true_log_field"], sub["y_pred_log_field"], s=28, alpha=0.85, edgecolor="none")
    mn = np.nanmin([sub["y_true_log_field"].min(), sub["y_pred_log_field"].min()])
    mx = np.nanmax([sub["y_true_log_field"].max(), sub["y_pred_log_field"].max()])
    ax.plot([mn,mx],[mn,mx],"--",lw=1.2)
    ax.set_aspect("equal","box")
    ax.set_xlim(mn,mx); ax.set_ylim(mn,mx)
    ax.set_xlabel("Measured (log10 SOM) — FIELD mean")
    ax.set_ylabel("Predicted (log10 SOM) — FIELD mean")
    ax.set_title(f"FIELD Parity (LOG) — {scen} — {algo}")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"parity_field_log_{algo}_{scen}.png", dpi=160)
    plt.show()

    # LINEAR
    fig2, ax2 = plt.subplots(figsize=(6.2,6.2))
    ax2.scatter(sub["y_true_lin_field"], sub["y_pred_lin_field"], s=28, alpha=0.85, edgecolor="none")
    mn2 = np.nanmin([sub["y_true_lin_field"].min(), sub["y_pred_lin_field"].min()])
    mx2 = np.nanmax([sub["y_true_lin_field"].max(), sub["y_pred_lin_field"].max()])
    ax2.plot([mn2,mx2],[mn2,mx2],"--",lw=1.2)
    ax2.set_aspect("equal","box")
    ax2.set_xlim(mn2,mx2); ax2.set_ylim(mn2,mx2)
    ax2.set_xlabel("Measured SOM — FIELD mean")
    ax2.set_ylabel("Predicted SOM (smeared) — FIELD mean")
    ax2.set_title(f"FIELD Parity (LINEAR) — {scen} — {algo}")
    ax2.grid(True, linestyle=":", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(f"parity_field_linear_{algo}_{scen}.png", dpi=160)
    plt.show()

def main():
    files = sorted(glob.glob("preds_*.csv"))
    if not files:
        raise FileNotFoundError("No preds_*.csv files found in current directory.")

    all_preds = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            # sanity: حداقل ستون‌های لازم
            need = {"FIELD_ID","Image_ID","y_true_log","y_pred_log","y_true_lin","y_pred_lin"}
            missing = need - set(df.columns)
            if missing:
                print(f"[WARN] Skip {f}: missing {missing}")
                continue
            # اگر algo/scenario در فایل نبود، از نام فایل حدس می‌زنیم
            if "algo" not in df.columns or "scenario" not in df.columns:
                base = os.path.splitext(os.path.basename(f))[0]  # preds_<ALGO>_<SCENARIO>
                parts = base.split("_", 2)
                algo = parts[1] if len(parts) > 1 else "UnknownAlgo"
                scen = parts[2] if len(parts) > 2 else "UnknownScenario"
                df["algo"] = algo
                df["scenario"] = scen
            all_preds.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")

    if not all_preds:
        raise RuntimeError("No valid preds files to process.")

    preds = pd.concat(all_preds, ignore_index=True)
    fld = field_level_from_preds(preds)
    fld.to_csv("field_level_preds_all.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] field_level_preds_all.csv written. ({len(fld)} rows)")

    summary = metrics_per_group(fld)
    summary.to_csv("field_level_summary_all.csv", index=False, encoding="utf-8-sig")
    print(f"[SAVE] field_level_summary_all.csv written.")
    print(summary)

    if SHOW_PLOTS:
        for (algo, scen) in summary[["Algo","Scenario"]].drop_duplicates().itertuples(index=False):
            plot_field_parity(fld, algo, scen)

if __name__ == "__main__":
    main()
