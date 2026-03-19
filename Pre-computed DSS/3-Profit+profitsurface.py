# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:20:55 2026

@author: NumericAg
"""

"""
Profit Lookup Builder (Paper-aligned, NON-DESTRUCTIVE)

This version builds TWO outputs:

1) expectedprofitlookup_paper
   - compact summary table for UI / results table
   - keeps ONLY the 4 displayed thresholds:
       1000, 1500, 2000, 2500

2) expectedprofit_surface_paper
   - dense threshold table for contour / heatmap
   - built from many profit thresholds, closer to mother-code logic

Main logic:
- Uses qp_modelprob_paper (QP model weights Jprob)
- FIXED: first mixes QP yields using Jprob, then computes profit
- Computes expected NRCF and EFB
- Computes probability of achieving NRCF >= threshold
- Uses >= (aligned with mother code)
- Explicitly normalizes price/cost probabilities and joint weights
- Keeps checkpointing and resume safety
"""

import mysql.connector
import numpy as np
from statistics import mean, stdev
from tqdm import tqdm
import time
import requests
from datetime import datetime


# ============================================================
# QP Yield Model
# ============================================================
def qp_yield(N, Y0, Ymax, Nymax):
    b = 2.0 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    return Y0 + b * N + c * (N ** 2) if N <= Nymax else Ymax


def qp_yield_vec(N_arr, Y0, Ymax, Nymax):
    b = 2.0 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    y = Y0 + b * N_arr + c * (N_arr ** 2)
    return np.where(N_arr <= Nymax, y, Ymax)


# ============================================================
# Corn price bins
# Simplified production version based on 5-year historical mean/std
# ============================================================
def fetch_corn_price_bins():
    api_key = "d5eac415442010f6a8567b47c69d4449"
    current_year = datetime.now().year
    start_year = current_year - 5

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=PMAIZMTUSDM"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&observation_start={start_year}-01-01"
    )

    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise Exception(f"❌ Failed to fetch corn prices from FRED. HTTP {r.status_code}")

    observations = r.json().get("observations", [])
    prices = [
        float(obs["value"])
        for obs in observations
        if obs.get("value") not in (None, ".", "")
    ]

    if not prices:
        raise Exception("❌ No valid corn price observations returned from FRED.")

    mu = mean(prices)
    sigma = stdev(prices) if len(prices) > 1 else 1.0

    print(f"🌽 Corn price stats → Mean: {mu:.2f}, Std: {sigma:.2f}")

    bins = [
        (mu - sigma, 0.20),
        (mu,         0.50),
        (mu + sigma, 0.30),
    ]
    return bins


# ============================================================
# Urea price bins
# Simplified production version
# ============================================================
def get_urea_price_bins():
    return [
        (0.30, 0.10),
        (0.45, 0.25),
        (0.60, 0.30),
        (0.90, 0.20),
        (1.20, 0.15),
    ]


# ============================================================
# Probability normalizer
# ============================================================
def normalize_probabilities(values, probs, name="distribution"):
    values = np.asarray(values, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    if values.size == 0 or probs.size == 0 or values.size != probs.size:
        raise ValueError(f"{name}: invalid values/probabilities")

    s = float(probs.sum())
    if s <= 1e-18:
        probs = np.ones_like(probs, dtype=np.float64) / len(probs)
    else:
        probs = probs / s

    return values, probs


# ============================================================
# DB connection
# ============================================================
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="*******",
        port=3307,
        database="selected",
        autocommit=False,
    )


# ============================================================
# Index helper
# ============================================================
def ensure_index(cursor, table, index_name, columns):
    cursor.execute(
        """
        SELECT COUNT(1)
        FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND index_name = %s
        """,
        (table, index_name),
    )
    exists = cursor.fetchone()[0]
    if not exists:
        cursor.execute(f"CREATE INDEX {index_name} ON {table} ({columns})")


# ============================================================
# Output tables
# ============================================================
def ensure_output_tables(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS expectedprofitlookup_paper (
            id INT AUTO_INCREMENT PRIMARY KEY,
            q INT NOT NULL,
            bin_key VARCHAR(64) NOT NULL,
            record_u_id INT NOT NULL,
            n_rate INT NOT NULL,
            prob_gt_1000 FLOAT,
            prob_gt_1500 FLOAT,
            prob_gt_2000 FLOAT,
            prob_gt_2500 FLOAT,
            expected_nrcf FLOAT,
            efb FLOAT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS expectedprofit_surface_paper (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            q INT NOT NULL,
            bin_key VARCHAR(64) NOT NULL,
            record_u_id INT NOT NULL,
            n_rate INT NOT NULL,
            profit_threshold INT NOT NULL,
            prob_ge_threshold FLOAT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS profit_paper_checkpoint (
            q INT NOT NULL,
            record_u_id INT NOT NULL,
            status ENUM('done','in_progress') NOT NULL DEFAULT 'done',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (q, record_u_id)
        )
    """)

    conn.commit()

    ensure_index(cur, "expectedprofitlookup_paper", "idx_eplp_q_rec", "q, record_u_id")
    ensure_index(cur, "expectedprofitlookup_paper", "idx_eplp_q_rec_n", "q, record_u_id, n_rate")
    ensure_index(cur, "expectedprofitlookup_paper", "idx_eplp_q_bin", "q, bin_key")

    ensure_index(cur, "expectedprofit_surface_paper", "idx_epsp_q_rec_n_t", "q, record_u_id, n_rate, profit_threshold")
    ensure_index(cur, "expectedprofit_surface_paper", "idx_epsp_q_bin", "q, bin_key")

    ensure_index(cur, "qp_modelprob_paper", "idx_qpmp_q_rec", "q, record_u_id")

    conn.commit()
    cur.close()


# ============================================================
# Resume helpers
# ============================================================
def pair_is_complete(cur, q, record_u_id, n_needed_summary, n_needed_surface):
    cur.execute(
        "SELECT COUNT(*) FROM expectedprofitlookup_paper WHERE q=%s AND record_u_id=%s",
        (int(q), int(record_u_id)),
    )
    summary_count = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM expectedprofit_surface_paper WHERE q=%s AND record_u_id=%s",
        (int(q), int(record_u_id)),
    )
    surface_count = cur.fetchone()[0]

    return (summary_count == n_needed_summary) and (surface_count == n_needed_surface)


def clear_pair(cur, q, record_u_id):
    cur.execute(
        "DELETE FROM expectedprofitlookup_paper WHERE q=%s AND record_u_id=%s",
        (int(q), int(record_u_id)),
    )
    cur.execute(
        "DELETE FROM expectedprofit_surface_paper WHERE q=%s AND record_u_id=%s",
        (int(q), int(record_u_id)),
    )


def fetch_pairs_batch(cur, last_q, last_rec, batch_size=5000):
    cur.execute(
        f"""
        SELECT q, record_u_id, MIN(bin_key) AS bin_key
        FROM qp_modelprob_paper
        WHERE (q > %s) OR (q = %s AND record_u_id > %s)
        GROUP BY q, record_u_id
        ORDER BY q, record_u_id
        LIMIT {int(batch_size)}
        """,
        (int(last_q), int(last_q), int(last_rec)),
    )
    return cur.fetchall()


# ============================================================
# Main
# ============================================================
def main():
    n_rates = list(range(0, 260, 10))   # 0..250

    table_thresholds = [1000, 1500, 2000, 2500]
    surface_thresholds = list(range(0, 3100, 100))  # 0..3000

    n_needed_summary = len(n_rates)
    n_needed_surface = len(n_rates) * len(surface_thresholds)

    conn = connect_db()
    ensure_output_tables(conn)

    # ------------------------------------------------
    # Economic bins
    # ------------------------------------------------
    corn_bins = fetch_corn_price_bins()
    urea_bins = get_urea_price_bins()

    p_vals_raw = [x[0] for x in corn_bins]
    p_prob_raw = [x[1] for x in corn_bins]
    c_vals_raw = [x[0] for x in urea_bins]
    c_prob_raw = [x[1] for x in urea_bins]

    p_vals, p_prob = normalize_probabilities(p_vals_raw, p_prob_raw, name="corn bins")
    c_vals, c_prob = normalize_probabilities(c_vals_raw, c_prob_raw, name="urea bins")

    # joint price-cost scenario weights
    Wsc = np.outer(p_prob, c_prob).reshape(-1)
    Wsc_sum = float(Wsc.sum())
    if Wsc_sum <= 1e-18:
        Wsc = np.ones_like(Wsc, dtype=np.float64) / len(Wsc)
    else:
        Wsc = Wsc / Wsc_sum

    # expand price/cost arrays to match joint scenarios
    P = np.repeat(p_vals, len(c_vals))
    C = np.tile(c_vals, len(p_vals))

    N = np.array(n_rates, dtype=np.float64)
    thr_table = np.array(table_thresholds, dtype=np.float64)
    thr_surface = np.array(surface_thresholds, dtype=np.float64)

    read_cur = conn.cursor(dictionary=True)
    write_cur = conn.cursor()

    last_q = -1
    last_rec = -1
    batch_size = 5000

    pbar = tqdm(desc="💰 Profit tables (summary + surface)", unit="pair")

    while True:
        batch = fetch_pairs_batch(read_cur, last_q, last_rec, batch_size=batch_size)
        if not batch:
            break

        for row in batch:
            q = int(row["q"])
            record_u_id = int(row["record_u_id"])
            bin_key = str(row["bin_key"])

            last_q, last_rec = q, record_u_id

            try:
                if pair_is_complete(write_cur, q, record_u_id, n_needed_summary, n_needed_surface):
                    pbar.update(1)
                    continue

                clear_pair(write_cur, q, record_u_id)

                read_cur.execute(
                    """
                    SELECT Y0, Ymax, Nymax, jprob
                    FROM qp_modelprob_paper
                    WHERE q=%s AND record_u_id=%s
                    """,
                    (q, record_u_id),
                )
                mrows = read_cur.fetchall()

                if not mrows:
                    conn.rollback()
                    pbar.update(1)
                    continue

                models = []
                w_models = []

                for mr in mrows:
                    models.append((
                        float(mr["Y0"]),
                        float(mr["Ymax"]),
                        float(mr["Nymax"])
                    ))
                    w_models.append(float(mr["jprob"]))

                w_models = np.asarray(w_models, dtype=np.float64)
                sw = float(w_models.sum())
                if sw <= 1e-18:
                    w_models = np.ones_like(w_models) / max(len(w_models), 1)
                else:
                    w_models = w_models / sw

                # ====================================================
                # FIXED LOGIC:
                # First mix yields across QP models using jprob,
                # then compute NRCF/probabilities on the mixed yield.
                # ====================================================

                # mixed yield over all N rates
                Y_mix = np.zeros(len(N), dtype=np.float64)

                for (Y0, Ymax, Nymax), wm in zip(models, w_models):
                    Y_mix += wm * qp_yield_vec(N, Y0, Ymax, Nymax)

                # mixed yield at N=0 for EFB base
                y0_mix = 0.0
                for (Y0, Ymax, Nymax), wm in zip(models, w_models):
                    y0_mix += wm * qp_yield(0, Y0, Ymax, Nymax)

                # NRCF for all N and all economic scenarios
                # shape = (n_rates, n_econ_scenarios)
                nrcf_mix = (Y_mix[:, None] * P[None, :]) - (N[:, None] * C[None, :])

                # expected NRCF under economic uncertainty
                expected_nrcf_mix = (nrcf_mix * Wsc[None, :]).sum(axis=1)

                # base NRCF at N=0 for EFB
                base_nrcf_mix = ((y0_mix * P) * Wsc).sum()

                # summary thresholds (4 shown in UI table)
                mask_table = (nrcf_mix[:, :, None] >= thr_table[None, None, :])
                prob_gt_table_mix = (mask_table * Wsc[None, :, None]).sum(axis=1) * 100.0

                # dense surface thresholds for contour / heatmap
                mask_surface = (nrcf_mix[:, :, None] >= thr_surface[None, None, :])
                prob_gt_surface_mix = (mask_surface * Wsc[None, :, None]).sum(axis=1) * 100.0

                efb_mix = expected_nrcf_mix - base_nrcf_mix

                prob_gt_table_mix = np.round(prob_gt_table_mix, 2)
                prob_gt_surface_mix = np.round(prob_gt_surface_mix, 2)

                # ----------------------------------------
                # Insert compact summary table
                # ----------------------------------------
                rows_summary = []
                for i, n_i in enumerate(n_rates):
                    rows_summary.append((
                        q,
                        bin_key,
                        record_u_id,
                        int(n_i),
                        float(prob_gt_table_mix[i, 0]),
                        float(prob_gt_table_mix[i, 1]),
                        float(prob_gt_table_mix[i, 2]),
                        float(prob_gt_table_mix[i, 3]),
                        float(expected_nrcf_mix[i]),
                        float(efb_mix[i]),
                    ))

                write_cur.executemany(
                    """
                    INSERT INTO expectedprofitlookup_paper (
                        q, bin_key, record_u_id, n_rate,
                        prob_gt_1000, prob_gt_1500, prob_gt_2000, prob_gt_2500,
                        expected_nrcf, efb
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    rows_summary,
                )

                # ----------------------------------------
                # Insert dense surface table
                # ----------------------------------------
                rows_surface = []
                for i, n_i in enumerate(n_rates):
                    for j, thr_j in enumerate(surface_thresholds):
                        rows_surface.append((
                            q,
                            bin_key,
                            record_u_id,
                            int(n_i),
                            int(thr_j),
                            float(prob_gt_surface_mix[i, j]),
                        ))

                write_cur.executemany(
                    """
                    INSERT INTO expectedprofit_surface_paper (
                        q, bin_key, record_u_id, n_rate,
                        profit_threshold, prob_ge_threshold
                    ) VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    rows_surface,
                )

                write_cur.execute(
                    """
                    INSERT INTO profit_paper_checkpoint (q, record_u_id, status)
                    VALUES (%s, %s, 'done')
                    ON DUPLICATE KEY UPDATE status='done', updated_at=CURRENT_TIMESTAMP
                    """,
                    (q, record_u_id),
                )

                conn.commit()

            except mysql.connector.errors.OperationalError as e:
                print(f"\n⚠️ Connection lost q={q}, record_u_id={record_u_id}: {e}")

                try:
                    conn.rollback()
                except Exception:
                    pass

                time.sleep(2)

                try:
                    read_cur.close()
                except Exception:
                    pass
                try:
                    write_cur.close()
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass

                conn = connect_db()
                ensure_output_tables(conn)
                read_cur = conn.cursor(dictionary=True)
                write_cur = conn.cursor()

            except Exception as e:
                print(f"\n❌ Error q={q}, record_u_id={record_u_id}: {e}")
                try:
                    conn.rollback()
                    clear_pair(write_cur, q, record_u_id)
                    conn.commit()
                except Exception:
                    pass

            pbar.update(1)

    pbar.close()

    try:
        read_cur.close()
    except Exception:
        pass
    try:
        write_cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

    print("🎉 Done:")
    print("   - expectedprofitlookup_paper (4 thresholds for table)")
    print("   - expectedprofit_surface_paper (dense thresholds for contour/heatmap)")


if __name__ == "__main__":
    main()