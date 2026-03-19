# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:55 2026

@author: NumericAg
"""

# -*- coding: utf-8 -*-
"""
QP Model-Probability Builder (Paper-aligned) ✅
WITHOUT touching your existing ErrorProb table

Goal (مثل مقاله):
For each (q, record_u_id):
  1) For ALL QP models i:
        Eerror_ij = (Yhat_i(N_j) - Y_j) * similarity_j
        avgerr_i  = mean(Eerror_i)
        sderr_i   = sample std(Eerror_i)   (ddof=1, closer to mother code)
        sderr_i   = max(sderr_i, STD_FLOOR)
        p0_i      = norm.pdf(0, avgerr_i, sderr_i)
  2) Normalize across models:
        Jprob_i = p0_i / sum(p0_all_models)
  3) Save (Y0,Ymax,Nymax, avgerr,sderr, p0, Jprob) into NEW table

⚠️ IMPORTANT PRACTICAL NOTE:
If you store ALL models for ALL record_u_id, row count can explode (very large).
So this script supports:
  - TOP_M = None  -> store ALL models
  - TOP_M = 50/100 -> store only the top-M models by Jprob
                      (computed from ALL models first)

Tables (lowercase):
  - qp_modelprob_paper
  - qp_paper_checkpoint
"""

import mysql.connector
import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import json


# ----------------------------
# QP yield model
# ----------------------------
def qp_yield(N, Y0, Ymax, Nymax):
    b = 2.0 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    return Y0 + b * N + c * N ** 2 if N <= Nymax else Ymax


def qp_yield_vec(N_arr, Y0, Ymax, Nymax):
    b = 2.0 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    y = Y0 + b * N_arr + c * (N_arr ** 2)
    return np.where(N_arr <= Nymax, y, Ymax)


# ----------------------------
# DB config
# ----------------------------
DB_CFG = dict(
    host="localhost",
    user="root",
    password="*******",
    port=3307,
    database="selected",
)


# ----------------------------
# Paper model grid
# NOTE:
# This version keeps your current grid unchanged.
# Only ddof/sample-std logic is corrected here.
# ----------------------------
Y0_range = np.arange(0, 20.5, 0.5)          # 0..20
Ymax_offsets = np.arange(0, 20.5, 0.5)      # +0..+20
Nymax_range = np.arange(10, 260, 10)    # 10..250


def generate_models():
    """
    returns list of tuples: (Y0, Ymax, Nymax)
    """
    models = []
    for Y0 in Y0_range:
        for off in Ymax_offsets:
            Ymax = Y0 + off
            if Ymax > 20:
                continue
            for Nymax in Nymax_range:
                models.append((float(Y0), float(Ymax), float(Nymax)))
    return models


MODELS = generate_models()
print(f"✅ Total QP models in grid: {len(MODELS)}")


# ----------------------------
# SETTINGS
# ----------------------------
TOP_M = 50          # set to None to store ALL models
STD_FLOOR = 0.1
BATCH_INSERT = 5000


# ----------------------------
# Safe sample std helper
# Mother-code closer behavior:
# - use sample std (ddof=1)
# - if too few samples or invalid std -> use STD_FLOOR
# ----------------------------
def safe_sample_std(arr: np.ndarray, std_floor: float = STD_FLOOR) -> float:
    arr = np.asarray(arr, dtype=np.float64)

    if arr.size < 2:
        return float(std_floor)

    s = float(arr.std(ddof=1))

    if not np.isfinite(s) or s < std_floor:
        return float(std_floor)

    return s


# ----------------------------
# Main
# ----------------------------
def main():
    conn = mysql.connector.connect(**DB_CFG, autocommit=False)
    cur = conn.cursor()

    # NEW tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS qp_modelprob_paper (
            id INT AUTO_INCREMENT PRIMARY KEY,
            q INT NOT NULL,
            bin_key VARCHAR(64) NOT NULL,
            record_u_id INT NOT NULL,
            Y0 FLOAT NOT NULL,
            Ymax FLOAT NOT NULL,
            Nymax FLOAT NOT NULL,
            avg_error FLOAT NOT NULL,
            std_error FLOAT NOT NULL,
            prob_zero_error FLOAT NOT NULL,
            jprob FLOAT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS qp_paper_checkpoint (
            q INT NOT NULL,
            record_u_id INT NOT NULL,
            status ENUM('done','in_progress') NOT NULL DEFAULT 'done',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (q, record_u_id)
        )
    """)

    # helpful indexes
    try:
        cur.execute("CREATE INDEX idx_qpmp_paper_qrid ON qp_modelprob_paper (q, record_u_id)")
    except mysql.connector.Error:
        pass

    try:
        cur.execute("CREATE INDEX idx_qpmp_paper_qrid_jprob ON qp_modelprob_paper (q, record_u_id, jprob DESC)")
    except mysql.connector.Error:
        pass

    conn.commit()

    # Load only what we need from mode4
    df = pd.read_sql(
        "SELECT q, bin_key, record_u_id, record_j_id, similarity, record_j_data FROM mode4",
        conn
    )
    print(f"📦 mode4 rows loaded: {len(df)}")

    # Done groups
    cur.execute("SELECT q, record_u_id FROM qp_paper_checkpoint WHERE status='done'")
    done_groups = set(cur.fetchall())
    print(f"🧭 Checkpoint: {len(done_groups)} groups already done")

    grouped = df.groupby(["q", "record_u_id"], sort=True)

    buffer = []
    pbar = tqdm(grouped, desc="🧠 Paper QP weights", unit="group")

    for (q, record_u_id), group in pbar:
        q = int(q)
        record_u_id = int(record_u_id)

        if (q, record_u_id) in done_groups:
            continue

        try:
            conn.start_transaction()

            # delete previous partial rows for this group
            cur.execute(
                "DELETE FROM qp_modelprob_paper WHERE q=%s AND record_u_id=%s",
                (q, record_u_id)
            )

            NTOT_vals = []
            YIELD_vals = []
            sim_vals = []

            for row in group.itertuples(index=False):
                rec = json.loads(row.record_j_data)
                NTOT_vals.append(float(rec["NTOT"]))
                YIELD_vals.append(float(rec["YIELD"]))
                sim_vals.append(float(row.similarity))

            N = np.asarray(NTOT_vals, dtype=np.float64)
            Y = np.asarray(YIELD_vals, dtype=np.float64)
            W = np.asarray(sim_vals, dtype=np.float64)

            bin_key = str(group["bin_key"].values[0])

            # ---- compute p0 for ALL models
            p0_list = []
            stats_list = []  # (Y0,Ymax,Nymax, avgerr, sderr, p0)

            for (Y0, Ymax, Nymax) in MODELS:
                preds = qp_yield_vec(N, Y0, Ymax, Nymax)
                Eerror = (preds - Y) * W

                avgerr = float(Eerror.mean()) if Eerror.size else 0.0

                # ✅ FIXED HERE:
                # use sample std (ddof=1) instead of population std (ddof=0)
                sderr = safe_sample_std(Eerror, std_floor=STD_FLOOR)

                p0 = float(norm.pdf(0.0, avgerr, sderr))

                stats_list.append((Y0, Ymax, Nymax, avgerr, sderr, p0))
                p0_list.append(p0)

            p0_arr = np.asarray(p0_list, dtype=np.float64)
            sump0 = float(p0_arr.sum())

            if sump0 <= 1e-18:
                jprob_arr = np.ones_like(p0_arr) / max(len(p0_arr), 1)
            else:
                jprob_arr = p0_arr / sump0

            # ---- OPTIONAL STORAGE REDUCTION (top-M)
            if TOP_M is not None:
                m = int(TOP_M)

                if m < len(jprob_arr):
                    idx = np.argpartition(-jprob_arr, m)[:m]
                    idx = idx[np.argsort(-jprob_arr[idx])]

                    kept = jprob_arr[idx]
                    s = float(kept.sum())
                    kept = kept / s if s > 0 else np.ones_like(kept) / max(len(kept), 1)

                    stats_to_store = []
                    for k, ii in enumerate(idx):
                        Y0, Ymax, Nymax, avgerr, sderr, p0 = stats_list[ii]
                        stats_to_store.append(
                            (Y0, Ymax, Nymax, avgerr, sderr, p0, float(kept[k]))
                        )
                else:
                    stats_to_store = [
                        (a, b, c, d, e, f, float(j))
                        for (a, b, c, d, e, f), j in zip(stats_list, jprob_arr)
                    ]
            else:
                stats_to_store = [
                    (a, b, c, d, e, f, float(j))
                    for (a, b, c, d, e, f), j in zip(stats_list, jprob_arr)
                ]

            # ---- buffered insert
            for (Y0, Ymax, Nymax, avgerr, sderr, p0, jp) in stats_to_store:
                buffer.append((
                    q, bin_key, record_u_id,
                    float(Y0), float(Ymax), float(Nymax),
                    float(avgerr), float(sderr),
                    float(p0), float(jp)
                ))

                if len(buffer) >= BATCH_INSERT:
                    cur.executemany("""
                        INSERT INTO qp_modelprob_paper (
                            q, bin_key, record_u_id,
                            Y0, Ymax, Nymax,
                            avg_error, std_error,
                            prob_zero_error, jprob
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """, buffer)
                    buffer.clear()

            # flush remainder for this group
            if buffer:
                cur.executemany("""
                    INSERT INTO qp_modelprob_paper (
                        q, bin_key, record_u_id,
                        Y0, Ymax, Nymax,
                        avg_error, std_error,
                        prob_zero_error, jprob
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, buffer)
                buffer.clear()

            # checkpoint done
            cur.execute("""
                INSERT INTO qp_paper_checkpoint (q, record_u_id, status)
                VALUES (%s, %s, 'done')
                ON DUPLICATE KEY UPDATE status='done', updated_at=CURRENT_TIMESTAMP
            """, (q, record_u_id))

            conn.commit()

        except Exception as e:
            conn.rollback()
            print(f"⚠️ Error in group (q={q}, record_u_id={record_u_id}): {e}")
            # next run will retry this group

    pbar.close()
    cur.close()
    conn.close()
    print("🎉 Done: qp_modelprob_paper built (paper-aligned, sample-std fixed).")


if __name__ == "__main__":
    main()