# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:06:44 2026

@author: NumericAg
"""

import numpy as np
import pandas as pd
import mysql.connector
import hashlib
import json
from tqdm import tqdm


# ============================================================
# Standard category levels (must stay consistent everywhere)
# ============================================================
CAT_PREV_LEVELS = ["Low nutrient", "Moderate nutrient", "High nutrient"]  # 0,1,2
TILLAGE_LEVELS = ["No till", "Conventionnel"]  # 0,1


def standardize_tillage(x: str) -> str:
    sx = str(x).strip().lower()
    if sx.startswith("no"):   # no till, no-till, no_till
        return "No till"
    return "Conventionnel"


def standardize_prev(x: str) -> str:
    sx = str(x).strip()
    return sx if sx in CAT_PREV_LEVELS else "Moderate nutrient"


def prev_idx(prev_label: str) -> int:
    if prev_label in CAT_PREV_LEVELS:
        return CAT_PREV_LEVELS.index(prev_label)
    return 1


def till_idx(till_label: str) -> int:
    return 0 if till_label == "No till" else 1


# ============================================================
# Fuzzy categorical similarity
# Aligned with mother-code logic:
# similarity = (base + (1 - normalized_diff) * (1 - base)) ^ q
# ============================================================
def categorical_similarity(value_j, value_u, base_penalty, denom, q):
    denom = denom if abs(denom) > 1e-12 else 1.0
    normalized_diff = abs((float(value_j) - float(value_u)) / denom)

    if normalized_diff < 0:
        normalized_diff = 0.0
    if normalized_diff > 1:
        normalized_diff = 1.0

    base_sim = base_penalty + (1.0 - normalized_diff) * (1.0 - base_penalty)

    if base_sim < 0:
        base_sim = 0.0
    if base_sim > 1:
        base_sim = 1.0

    sim = base_sim ** q

    if sim < 0:
        sim = 0.0
    if sim > 1:
        sim = 1.0

    return float(sim)


# ============================================================
# Similarity calculation
# Continuous part:
#   similarity = (1 - delta_lambda * normalized_diff) ^ q
#
# FIXED:
# - user side = bin mean
# - record j side = RAW VALUE from source table
# ============================================================
def calculate_similarity(
    features_j,
    features_u,
    features_max,
    features_min,
    delta_lambdas,
    q,
    num_continuous
):
    """
    features_j = [4 continuous RAW values] + [PREV_idx, TILL_idx]
    features_u = [4 continuous bin-means] + [PREV_idx, TILL_idx]

    Continuous variables:
        similarity = (1 - lambda * normalized_diff) ^ q

    Categorical variables:
        fuzzy logic aligned with mother code:
        - PrevCrop base = 0.5
        - Tillage  base = 0.9
        - then whole similarity ^ q
    """
    lambda_j_u = 1.0
    feature_similarities = []

    for k in range(len(features_j)):

        # ----------------------------------------------------
        # CONTINUOUS FEATURES
        # ----------------------------------------------------
        if k < num_continuous:
            denom = features_max[k] - features_min[k]
            denom = denom if abs(denom) > 1e-12 else 1.0

            normalized_diff = abs((features_j[k] - features_u[k]) / denom)

            base_sim = 1.0 - delta_lambdas[k] * normalized_diff

            if base_sim < 0:
                base_sim = 0.0
            if base_sim > 1:
                base_sim = 1.0

            feature_similarity = base_sim ** q

            if feature_similarity < 0:
                feature_similarity = 0.0
            if feature_similarity > 1:
                feature_similarity = 1.0

        # ----------------------------------------------------
        # CATEGORICAL FEATURES
        # ----------------------------------------------------
        else:
            cat_index = k - num_continuous

            if cat_index == 0:
                feature_similarity = categorical_similarity(
                    value_j=features_j[k],
                    value_u=features_u[k],
                    base_penalty=0.5,
                    denom=2.0,   # PREV_idx = 0,1,2
                    q=q
                )

            elif cat_index == 1:
                feature_similarity = categorical_similarity(
                    value_j=features_j[k],
                    value_u=features_u[k],
                    base_penalty=0.9,
                    denom=1.0,   # TILL_idx = 0,1
                    q=q
                )

            else:
                feature_similarity = 1.0

        lambda_j_u *= feature_similarity
        feature_similarities.append(float(feature_similarity))

    return float(lambda_j_u), feature_similarities


# ============================================================
# Binning helpers
# ============================================================
def build_bins(values: np.ndarray, num_bins: int):
    vals = np.asarray(values, dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    edges = np.linspace(vmin, vmax, num_bins + 1)
    means = [(edges[i] + edges[i + 1]) / 2.0 for i in range(num_bins)]
    return edges, means


def to_bin_index(value: float, edges: np.ndarray, num_bins: int) -> int:
    idx = int(np.digitize([value], edges)[0] - 1)

    if idx < 0:
        idx = 0
    if idx > num_bins - 1:
        idx = num_bins - 1

    return idx


def sha_bin_key(bin_indices):
    s = ",".join(map(str, bin_indices))
    return hashlib.sha256(s.encode()).hexdigest()


# ============================================================
# DB fetch
# ============================================================
def fetch_data_from_mysql(db_cfg, source_table):
    conn = mysql.connector.connect(**db_cfg)
    df = pd.read_sql(f"SELECT * FROM {source_table}", conn)
    conn.close()
    return df


# ============================================================
# Main builder for mode4
# ============================================================
def build_mode4(
    db_cfg,
    source_table="sample_data_test",
    out_table="mode4",
    q_values=(1, 2, 5),
    num_bins=5,
    batch_size=5000
):
    # ----------------------------
    # Load source table
    # ----------------------------
    df = fetch_data_from_mysql(db_cfg, source_table)

    # ----------------------------
    # Validate primary key column
    # ----------------------------
    if "id" not in df.columns:
        raise ValueError(
            "Column 'id' not found in source table. "
            "A real primary key column is required to store record_j_id correctly."
        )

    # ----------------------------
    # Standardize categorical columns
    # ----------------------------
    if "Prev Type" not in df.columns:
        raise ValueError("Column 'Prev Type' not found in source table.")
    if "TILLAGE" not in df.columns:
        raise ValueError("Column 'TILLAGE' not found in source table.")

    df["Prev_std"] = df["Prev Type"].apply(standardize_prev)
    df["TILLAGE_std"] = df["TILLAGE"].apply(standardize_tillage)

    df["PREV_idx"] = df["Prev_std"].apply(prev_idx).astype(int)     # 0,1,2
    df["TILL_idx"] = df["TILLAGE_std"].apply(till_idx).astype(int)  # 0,1

    # ----------------------------
    # Feature definitions
    # ----------------------------
    features_columns = ["ACLAY", "SOM", "CHU", "AWDR"]

    # ----------------------------
    # Build bins for continuous features
    # ----------------------------
    bin_edges = {}
    bin_means = {}

    for f in features_columns:
        edges, means = build_bins(df[f].values, num_bins)
        bin_edges[f] = edges
        bin_means[f] = means

    # ----------------------------
    # Store only j bin index for traceability
    # Similarity itself will use RAW continuous values
    # ----------------------------
    j_bin_idx = {}

    for f in features_columns:
        idxs = [to_bin_index(v, bin_edges[f], num_bins) for v in df[f].values]
        j_bin_idx[f] = np.array(idxs, dtype=int)

    # ----------------------------
    # Continuous normalization ranges
    # ----------------------------
    features_max = df[features_columns].max().values.astype(float)
    features_min = df[features_columns].min().values.astype(float)

    # continuous feature weights
    delta_lambdas = np.ones(len(features_columns), dtype=float)

    # ----------------------------
    # Total user combinations
    # 5^4 * 3 * 2
    # ----------------------------
    num_combinations = (
        (num_bins ** len(features_columns))
        * len(CAT_PREV_LEVELS)
        * len(TILLAGE_LEVELS)
    )
    print(f"✅ Total user input combinations: {num_combinations}")

    # ----------------------------
    # Create output table
    # ----------------------------
    conn = mysql.connector.connect(**db_cfg)
    cur = conn.cursor()

    cur.execute(f"DROP TABLE IF EXISTS {out_table}")
    cur.execute(f"""
        CREATE TABLE {out_table} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            q INT,
            bin_key VARCHAR(64),
            record_j_id BIGINT,
            record_u_id INT,
            record_j_data TEXT,
            record_u_data TEXT,
            similarity FLOAT,
            feature_similarities TEXT,
            bins_j_data TEXT,
            bins_u_data TEXT
        )
    """)
    conn.commit()

    # ----------------------------
    # Build user input combinations
    # order:
    # [ACLAY_bin, SOM_bin, CHU_bin, AWDR_bin, PREV_idx, TILL_idx]
    # ----------------------------
    cont_shapes = [num_bins] * len(features_columns)
    cat_shapes = [len(CAT_PREV_LEVELS), len(TILLAGE_LEVELS)]
    shapes = cont_shapes + cat_shapes

    # ----------------------------
    # Main insertion loop
    # ----------------------------
    for q in q_values:
        print(f"\n🚀 Building mode4 for q = {q}")
        buffer = []
        record_u_id = 0

        for user_case in tqdm(range(num_combinations), desc=f"q={q}"):
            idx_tuple = np.unravel_index(user_case, shapes)
            user_bins = list(map(int, idx_tuple))

            # User continuous values = bin means
            u_cont = [
                bin_means[features_columns[i]][user_bins[i]]
                for i in range(len(features_columns))
            ]

            # User categorical values
            u_cat = user_bins[len(features_columns):]   # [PREV_idx, TILL_idx]

            features_u = u_cont + u_cat
            bin_key = sha_bin_key(user_bins)

            record_u_data = {
                "cont_bin_means": dict(zip(features_columns, map(float, u_cont))),
                "PREV_idx": int(u_cat[0]),
                "TILL_idx": int(u_cat[1]),
                "bin_indices": user_bins
            }

            bins_u_data = {"bins": user_bins}

            # Loop over each database record j
            for j in range(len(df)):
                # RAW continuous values for record j
                j_cont = [float(df.iloc[j][f]) for f in features_columns]
                j_cat = [int(df.iloc[j]["PREV_idx"]), int(df.iloc[j]["TILL_idx"])]

                features_j = j_cont + j_cat

                sim, feat_sims = calculate_similarity(
                    features_j=features_j,
                    features_u=features_u,
                    features_max=features_max,
                    features_min=features_min,
                    delta_lambdas=delta_lambdas,
                    q=q,
                    num_continuous=len(features_columns)
                )

                # FIXED: use real DB primary key instead of dataframe row index
                record_j_id = int(df.iloc[j]["id"])

                record_j_data = df.iloc[j].to_dict()
                bins_j = [int(j_bin_idx[f][j]) for f in features_columns] + j_cat

                buffer.append((
                    int(q),
                    bin_key,
                    record_j_id,
                    int(record_u_id),
                    json.dumps(record_j_data, ensure_ascii=False),
                    json.dumps(record_u_data, ensure_ascii=False),
                    float(sim),
                    json.dumps(feat_sims, ensure_ascii=False),
                    json.dumps({"bins": bins_j}, ensure_ascii=False),
                    json.dumps(bins_u_data, ensure_ascii=False),
                ))

                if len(buffer) >= batch_size:
                    cur.executemany(f"""
                        INSERT INTO {out_table} (
                            q, bin_key, record_j_id, record_u_id,
                            record_j_data, record_u_data,
                            similarity, feature_similarities,
                            bins_j_data, bins_u_data
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, buffer)
                    conn.commit()
                    buffer.clear()

            record_u_id += 1

        # Flush remainder
        if buffer:
            cur.executemany(f"""
                INSERT INTO {out_table} (
                    q, bin_key, record_j_id, record_u_id,
                    record_j_data, record_u_data,
                    similarity, feature_similarities,
                    bins_j_data, bins_u_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, buffer)
            conn.commit()
            buffer.clear()

        print(f"✅ q = {q} done.")

    cur.close()
    conn.close()
    print("\n🎉 mode4 rebuilt successfully.")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    DB_CFG = {
        "host": "localhost",
        "user": "root",
        "password": "*******",
        "port": 3307,
        "database": "selected",
    }

    build_mode4(
        db_cfg=DB_CFG,
        source_table="sample_data_test",
        out_table="mode4",
        q_values=(1, 2, 5),
        num_bins=5,
        batch_size=5000
    )