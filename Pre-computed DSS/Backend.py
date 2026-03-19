# backend/app.py
# FINAL BACKEND - FALLBACK FIXED
# -----------------------------------------------------------
# Precomputed NumericAg DSS backend
#
# Uses:
#   - mode4
#   - qp_modelprob_paper
#   - expectedprofitlookup_paper
#   - expectedprofit_surface_paper
#   - sample_data_test
#
# Key design:
#   - single lookup by bin_key -> one record_u_id
#   - fallback is now based on nearest user-case bin distance (FIXED)
#   - summary profit table comes from expectedprofitlookup_paper
#   - contour surface comes from expectedprofit_surface_paper
#   - contour image is generated from precomputed surface (NOT recomputed live)
#   - email includes contour image
#   - base64 contour image returned to frontend
#
# This keeps frontend compatibility while preserving paper-aligned consistency.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from mysql.connector import pooling
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import hashlib
import math
import os
import io
import base64
import json

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import requests
from statistics import mean, stdev
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="NumericAg DSS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.numericag.com",
        "https://numericag.com",
    ],
    allow_origin_regex=r"https://[a-z0-9-]+\.ngrok-free\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)


# ============================================================
# Input schema
# Keep compatible with current frontend
# ============================================================
class UserInput(BaseModel):
    ACLAY: float
    SOM: float
    CHU: float
    AWDR: float
    PREV: str
    TILLAGE: str
    q: int

    Fertilizer: str
    Crop: str
    Season: int
    email: str

    yieldPriceMean: float
    yieldPriceStdDev: float
    fertilizerCostMean: float
    fertilizerCostStdDev: float

    lati: Optional[float] = None
    longti: Optional[float] = None
    PREV_VALUE: Optional[float] = None
    TILLAGE_VALUE: Optional[float] = None


# ============================================================
# Constants
# ============================================================
NUM_BINS = 5
VALID_Q_VALUES = {1, 2, 5}

CAT_PREV_LEVELS = ["Low nutrient", "Moderate nutrient", "High nutrient"]
TILLAGE_LEVELS = ["No till", "Conventionnel"]


# ============================================================
# DB pool
# ============================================================
cnxpool = pooling.MySQLConnectionPool(
    pool_name="numericag_pool",
    pool_size=5,
    host="127.0.0.1",
    port=3307,
    user="root",
    password="******",
    database="selected",
)

def get_conn():
    conn = cnxpool.get_connection()
    try:
        conn.ping(reconnect=True, attempts=2, delay=0)
    except Exception:
        pass
    return conn


# ============================================================
# Standardization helpers
# ============================================================
def standardize_tillage(x: str) -> str:
    sx = str(x).strip().lower()
    if sx.startswith("no"):
        return "No till"
    return "Conventionnel"

def standardize_prev(x: str) -> str:
    sx = str(x).strip()
    return sx if sx in CAT_PREV_LEVELS else "Moderate nutrient"

def prev_idx(prev_label: str) -> int:
    p = str(prev_label).strip()
    return CAT_PREV_LEVELS.index(p) if p in CAT_PREV_LEVELS else 1

def till_idx(till_label: str) -> int:
    t = standardize_tillage(till_label)
    return 0 if t == "No till" else 1


# ============================================================
# Optional numeric categorical resolution
# ============================================================
def is_finite_number(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

def prev_value_to_idx(value: float) -> int:
    v = clamp(value, 0.0, 2.0)
    if v < 0.5:
        return 0
    elif v < 1.5:
        return 1
    return 2

def till_value_to_idx(value: float) -> int:
    v = clamp(value, 0.0, 1.0)
    return 0 if v < 0.5 else 1

def resolve_prev_idx(user_input: UserInput) -> int:
    if is_finite_number(user_input.PREV_VALUE):
        return prev_value_to_idx(float(user_input.PREV_VALUE))
    return prev_idx(standardize_prev(user_input.PREV))

def resolve_till_idx(user_input: UserInput) -> int:
    if is_finite_number(user_input.TILLAGE_VALUE):
        return till_value_to_idx(float(user_input.TILLAGE_VALUE))
    return till_idx(standardize_tillage(user_input.TILLAGE))


# ============================================================
# Price helpers (informational only for UI/email)
# Profit itself is precomputed already
# ============================================================
def fetch_default_corn_price_stats():
    try:
        api_key = "d5eac415442010f6a8567b47c69d4449"
        current_year = datetime.now().year
        start_year = current_year - 5
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=PMAIZMTUSDM&api_key={api_key}&file_type=json"
            f"&observation_start={start_year}-01-01"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return 175.0, 43.0

        prices = [
            float(obs["value"])
            for obs in r.json().get("observations", [])
            if obs.get("value") not in (None, ".", "")
        ]
        if not prices:
            return 175.0, 43.0

        return float(mean(prices)), float(stdev(prices) if len(prices) > 1 else 1.0)
    except Exception:
        return 175.0, 43.0

def get_default_urea_cost_stats():
    return 0.6, 0.14


# ============================================================
# Email helpers
# ============================================================
SENDER_EMAIL = os.getenv("NUMERICAG_SENDER_EMAIL", "numericag.dss@gmail.com").strip()
SENDER_PASSWORD = os.getenv("NUMERICAG_SENDER_PASSWORD", "************").strip()
SMTP_HOST = os.getenv("NUMERICAG_SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("NUMERICAG_SMTP_PORT", "587"))

def send_email(email: str, subject: str, html_body: str, image_bytes: Optional[bytes] = None) -> bool:
    if not email:
        return False
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("⚠️ Email not sent: missing sender credentials.")
        return False

    msg = MIMEMultipart("related")
    msg["From"] = SENDER_EMAIL
    msg["To"] = email
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)
    alt.attach(MIMEText(html_body, "html", _charset="utf-8"))

    if image_bytes:
        try:
            mime_image = MIMEImage(image_bytes)
            mime_image.add_header("Content-ID", "<chart>")
            mime_image.add_header("Content-Disposition", "inline", filename="chart.png")
            msg.attach(mime_image)
        except Exception as e:
            print(f"⚠️ Could not attach image: {e}")

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# ============================================================
# Feature ranges endpoint
# ============================================================
@app.get("/feature_ranges/")
def get_feature_ranges():
    conn = None
    try:
        conn = get_conn()
        query = """
            SELECT
                ROUND(MIN(ACLAY), 1) AS ACLAY_min,
                ROUND(MAX(ACLAY), 1) AS ACLAY_max,
                ROUND(MIN(SOM), 1)   AS SOM_min,
                ROUND(MAX(SOM), 1)   AS SOM_max,
                ROUND(MIN(CHU), 1)   AS CHU_min,
                ROUND(MAX(CHU), 1)   AS CHU_max,
                ROUND(MIN(AWDR), 1)  AS AWDR_min,
                ROUND(MAX(AWDR), 1)  AS AWDR_max
            FROM sample_data_test
        """
        df = pd.read_sql(query, conn)
        if df.empty:
            raise HTTPException(status_code=500, detail="sample_data_test is empty.")
        return df.to_dict(orient="records")[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feature ranges: {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


# ============================================================
# Binning utilities
# MUST match mode4 builder
# ============================================================
def build_edges(vmin: float, vmax: float, num_bins: int) -> np.ndarray:
    return np.linspace(float(vmin), float(vmax), num_bins + 1)

def to_bin_index_mode4(value: float, edges: np.ndarray, num_bins: int) -> int:
    idx = int(np.digitize([float(value)], edges)[0] - 1)
    if idx < 0:
        idx = 0
    if idx > num_bins - 1:
        idx = num_bins - 1
    return idx

def sha_bin_key(bin_indices: List[int]) -> str:
    s = ",".join(map(str, bin_indices))
    return hashlib.sha256(s.encode()).hexdigest()


# ============================================================
# Fallback utilities
# ============================================================
def parse_bins_u_data(raw_value) -> Optional[List[int]]:
    """
    Expected shape in mode4:
      bins_u_data = {"bins": [ACLAY_bin, SOM_bin, CHU_bin, AWDR_bin, PREV_idx, TILL_idx]}
    """
    try:
        if raw_value is None:
            return None

        if isinstance(raw_value, dict):
            obj = raw_value
        else:
            obj = json.loads(raw_value)

        bins = obj.get("bins")
        if not isinstance(bins, list):
            return None

        bins = [int(x) for x in bins]
        if len(bins) != 6:
            return None

        return bins
    except Exception:
        return None

def manhattan_distance_bins(a: List[int], b: List[int]) -> int:
    return int(sum(abs(int(x) - int(y)) for x, y in zip(a, b)))


# ============================================================
# Lookup helper - FIXED FALLBACK
# ============================================================
def find_record_u_id(conn, q: int, user_bin_key: str, user_bin_indices: List[int]) -> Dict[str, Any]:
    """
    Returns:
      {
        "record_u_id": int,
        "matched_bin_key": str,
        "used_fallback": bool,
        "similarity": float,
        "fallback_distance": Optional[int]
      }
    """
    cur = conn.cursor(dictionary=True)

    try:
        # ----------------------------------------------------
        # 1) Exact match by bin_key
        # ----------------------------------------------------
        cur.execute(
            """
            SELECT record_u_id, similarity
            FROM mode4
            WHERE q=%s AND bin_key=%s
            ORDER BY similarity DESC
            LIMIT 1
            """,
            (int(q), str(user_bin_key)),
        )
        row = cur.fetchone()

        if row:
            return {
                "record_u_id": int(row["record_u_id"]),
                "matched_bin_key": str(user_bin_key),
                "used_fallback": False,
                "similarity": float(row["similarity"]) if row["similarity"] is not None else 0.0,
                "fallback_distance": None,
            }

        # ----------------------------------------------------
        # 2) Fallback by nearest user-case bins
        #    We read distinct user-cases from mode4 using bins_u_data
        # ----------------------------------------------------
        cur.execute(
            """
            SELECT
                record_u_id,
                bin_key,
                bins_u_data
            FROM mode4
            WHERE q=%s
            GROUP BY record_u_id, bin_key, bins_u_data
            """,
            (int(q),),
        )
        rows = cur.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No match found for this q (mode4 empty).")

        best_row = None
        best_dist = None

        for r in rows:
            bins = parse_bins_u_data(r.get("bins_u_data"))
            if bins is None:
                continue

            dist = manhattan_distance_bins(user_bin_indices, bins)

            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                best_row = r
            elif dist == best_dist:
                # stable tie-breaker: smaller record_u_id
                if best_row is not None and int(r["record_u_id"]) < int(best_row["record_u_id"]):
                    best_row = r
                    best_dist = dist

        if best_row is None:
            raise HTTPException(
                status_code=404,
                detail="Fallback failed: no valid bins_u_data found for this q."
            )

        return {
            "record_u_id": int(best_row["record_u_id"]),
            "matched_bin_key": str(best_row["bin_key"]),
            "used_fallback": True,
            # similarity from exact mode4 row is not meaningful in fallback mode
            "similarity": 0.0,
            "fallback_distance": int(best_dist),
        }

    finally:
        cur.close()


# ============================================================
# Surface data from precomputed table
# ============================================================
def fetch_contour_surface(conn, q: int, record_u_id: int):
    df = pd.read_sql(
        """
        SELECT
            n_rate,
            profit_threshold,
            prob_ge_threshold
        FROM expectedprofit_surface_paper
        WHERE q=%s AND record_u_id=%s
        ORDER BY profit_threshold, n_rate
        """,
        conn,
        params=(int(q), int(record_u_id)),
    )

    if df is None or df.empty:
        raise ValueError(f"No expectedprofit_surface_paper rows found for q={q}, record_u_id={record_u_id}")

    pivot = (
        df.pivot(index="profit_threshold", columns="n_rate", values="prob_ge_threshold")
          .sort_index(axis=0)
          .sort_index(axis=1)
          .fillna(0.0)
    )

    x = np.array(pivot.columns.tolist(), dtype=float)
    y = np.array(pivot.index.tolist(), dtype=float)
    z = pivot.values.astype(float)

    contour_data = {
        "x": [int(v) for v in x.tolist()],
        "y": [int(v) for v in y.tolist()],
        "z": z.tolist(),
    }

    return x, y, z, contour_data


# ============================================================
# Contour image generator from PRECOMPUTED surface
# ============================================================
def generate_probability_contour_from_surface(conn, q: int, record_u_id: int) -> bytes:
    x, y, Z, _ = fetch_contour_surface(conn, q, record_u_id)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    cf = ax.contourf(X, Y, Z, levels=np.linspace(0, 100, 21), cmap="viridis")
    ax.contour(X, Y, Z, levels=[25], colors="red", linewidths=1.2)
    ax.contour(X, Y, Z, levels=[50], colors="orange", linewidths=1.2)
    ax.contour(X, Y, Z, levels=[75], colors="yellow", linewidths=1.2)

    ax.text(95, 2400, "Unlikely (0% - 25%)", color="red", fontsize=11)
    ax.text(95, 2000, "Possibly (25% - 50%)", color="orange", fontsize=11)
    ax.text(95, 1600, "Most Likely (50% - 75%)", color="yellow", fontsize=11)
    ax.text(90, 500, "Certainly (75% - 100%)", color="lime", fontsize=11)

    ax.set_xlabel("Application Rate, kg/ha", fontsize=12, fontweight="bold")
    ax.set_ylabel("NRCF, $/ha", fontsize=12, fontweight="bold")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Probability of Achieving NRCF (equal or greater than)", fontsize=12)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


# ============================================================
# Main endpoint
# ============================================================
@app.post("/calculate_similarity/")
def calculate_similarity(user_input: UserInput):
    conn = None

    try:
        if int(user_input.q) not in VALID_Q_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid q={user_input.q}. Allowed values are {sorted(list(VALID_Q_VALUES))}."
            )

        # informational only for email/UI
        yield_mean, yield_std = (user_input.yieldPriceMean, user_input.yieldPriceStdDev)
        fert_mean, fert_std = (user_input.fertilizerCostMean, user_input.fertilizerCostStdDev)

        if yield_mean == 0 or yield_std == 0:
            yield_mean, yield_std = fetch_default_corn_price_stats()
        if fert_mean == 0 or fert_std == 0:
            fert_mean, fert_std = get_default_urea_cost_stats()

        conn = get_conn()

        # rebuild exact same bin structure as mode4
        mm = pd.read_sql(
            """
            SELECT
                MIN(ACLAY) AS ACLAY_min, MAX(ACLAY) AS ACLAY_max,
                MIN(SOM)   AS SOM_min,   MAX(SOM)   AS SOM_max,
                MIN(CHU)   AS CHU_min,   MAX(CHU)   AS CHU_max,
                MIN(AWDR)  AS AWDR_min,  MAX(AWDR)  AS AWDR_max
            FROM sample_data_test
            """,
            conn
        ).iloc[0].to_dict()

        edges_map = {
            "ACLAY": build_edges(mm["ACLAY_min"], mm["ACLAY_max"], NUM_BINS),
            "SOM":   build_edges(mm["SOM_min"],   mm["SOM_max"],   NUM_BINS),
            "CHU":   build_edges(mm["CHU_min"],   mm["CHU_max"],   NUM_BINS),
            "AWDR":  build_edges(mm["AWDR_min"],  mm["AWDR_max"],  NUM_BINS),
        }

        cont_feats = ["ACLAY", "SOM", "CHU", "AWDR"]
        user_values = [user_input.ACLAY, user_input.SOM, user_input.CHU, user_input.AWDR]

        bin_indices = [
            to_bin_index_mode4(v, edges_map[f], NUM_BINS)
            for f, v in zip(cont_feats, user_values)
        ]

        prev_index = resolve_prev_idx(user_input)
        till_index = resolve_till_idx(user_input)

        bin_indices += [int(prev_index), int(till_index)]
        user_bin_key = sha_bin_key(bin_indices)

        # single lookup with corrected fallback
        pick = find_record_u_id(
            conn=conn,
            q=int(user_input.q),
            user_bin_key=user_bin_key,
            user_bin_indices=bin_indices
        )

        record_u_id = int(pick["record_u_id"])
        matched_bin_key = str(pick["matched_bin_key"])
        used_fallback = bool(pick["used_fallback"])
        similarity = float(pick["similarity"])
        fallback_distance = pick.get("fallback_distance")

        # summary table from precomputed paper table
        curve = pd.read_sql(
            """
            SELECT
                n_rate,
                prob_gt_1000,
                prob_gt_1500,
                prob_gt_2000,
                prob_gt_2500,
                expected_nrcf,
                efb
            FROM expectedprofitlookup_paper
            WHERE q=%s AND record_u_id=%s
            ORDER BY n_rate ASC
            """,
            conn,
            params=(int(user_input.q), int(record_u_id)),
        )

        if curve is None or curve.empty:
            raise HTTPException(
                status_code=404,
                detail="No paper-aligned precomputed profit rows found for this match."
            )

        profit_rows: List[Dict[str, Any]] = []
        for r in curve.to_dict(orient="records"):
            profit_rows.append({
                "n_rate": int(r["n_rate"]),
                "prob_gt_1000": float(r["prob_gt_1000"]) if r["prob_gt_1000"] is not None else None,
                "prob_gt_1500": float(r["prob_gt_1500"]) if r["prob_gt_1500"] is not None else None,
                "prob_gt_2000": float(r["prob_gt_2000"]) if r["prob_gt_2000"] is not None else None,
                "prob_gt_2500": float(r["prob_gt_2500"]) if r["prob_gt_2500"] is not None else None,
                "expected_nrcf": float(r["expected_nrcf"]),
                "efb": float(r["efb"]),
            })

        max_efb_row = max(profit_rows, key=lambda row: row["efb"])
        eonr = int(max_efb_row["n_rate"])

        # display-only qp params
        qp_params = None
        try:
            rowp = pd.read_sql(
                """
                SELECT Y0, Ymax, Nymax, jprob
                FROM qp_modelprob_paper
                WHERE q=%s AND record_u_id=%s
                ORDER BY jprob DESC
                LIMIT 1
                """,
                conn,
                params=(int(user_input.q), int(record_u_id)),
            )
            if rowp is not None and not rowp.empty:
                qp_params = {
                    "Y0": float(rowp.iloc[0]["Y0"]),
                    "Ymax": float(rowp.iloc[0]["Ymax"]),
                    "Nymax": float(rowp.iloc[0]["Nymax"]),
                    "jprob": float(rowp.iloc[0]["jprob"]),
                }
        except Exception:
            qp_params = None

        # contour data + image from PRECOMPUTED surface table
        contour_data = {"x": [], "y": [], "z": []}
        contour_chart_base64 = None
        contour_png_bytes = None

        try:
            _, _, _, contour_data = fetch_contour_surface(conn, int(user_input.q), int(record_u_id))
            contour_png_bytes = generate_probability_contour_from_surface(
                conn, int(user_input.q), int(record_u_id)
            )
            contour_chart_base64 = base64.b64encode(contour_png_bytes).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Contour generation issue: {e}")

        # email html
        warning_msg = ""
        if used_fallback:
            warning_msg = (
                "<p style='color:red; font-weight:bold;'>"
                "⚠️ Exact bin match was not found. Results are shown using the nearest available precomputed scenario."
                "</p>"
            )

        def fmt(x):
            return "" if x is None else f"{x:.2f}"

        html_rows = ""
        for row in profit_rows:
            style = " style='background-color:#d1ffd1;font-weight:bold;'" if row["n_rate"] == eonr else ""
            html_rows += f"""
            <tr{style}>
                <td>{row['n_rate']}</td>
                <td>{fmt(row['prob_gt_1000'])}</td>
                <td>{fmt(row['prob_gt_1500'])}</td>
                <td>{fmt(row['prob_gt_2000'])}</td>
                <td>{fmt(row['prob_gt_2500'])}</td>
                <td>{row['expected_nrcf']:.2f}</td>
                <td>{row['efb']:.2f}</td>
            </tr>
            """

        qp_line = ""
        if qp_params is not None:
            qp_line = (
                f"<p>Top QP model for display: "
                f"<b>Y0={qp_params['Y0']:.2f}, "
                f"Ymax={qp_params['Ymax']:.2f}, "
                f"Nymax={qp_params['Nymax']:.2f}</b>"
                f" (Jprob={qp_params['jprob']:.4f})</p>"
            )

        fallback_line = ""
        if used_fallback and fallback_distance is not None:
            fallback_line = f"<p>Fallback bin distance: <b>{int(fallback_distance)}</b></p>"

        similarity_line = ""
        if used_fallback:
            similarity_line = "<p>Similarity: <b>Not applicable in fallback mode</b></p>"
        else:
            similarity_line = f"<p>Similarity: <b>{similarity:.4f}</b></p>"

        contour_html = ""
        if contour_png_bytes is not None:
            contour_html = '<h3>📈 Probability Contour</h3><img src="cid:chart" width="700" />'

        html_body = f"""
        <html><head><style>
        body {{ font-family: Arial; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        </style></head><body>

        <h3>📌 Precomputed Result</h3>
        {warning_msg}

        <p>Your bin key: <b>{user_bin_key}</b></p>
        <p>Matched bin key used: <b>{matched_bin_key}</b></p>
        {similarity_line}
        {fallback_line}
        <p>Resolved PREV index: <b>{prev_index}</b></p>
        <p>Resolved TILLAGE index: <b>{till_index}</b></p>
        {qp_line}
        <p>⭐ Economically Optimum N Rate (EONR): <b>{eonr} kg/ha</b></p>

        <h3>📊 Profit Curve</h3>
        <table>
          <tr>
            <th>N Rate</th>
            <th>P &gt; $1000</th>
            <th>P &gt; $1500</th>
            <th>P &gt; $2000</th>
            <th>P &gt; $2500</th>
            <th>Expected NRCF</th>
            <th>EFB</th>
          </tr>
          {html_rows}
        </table>

        {contour_html}

        </body></html>
        """

        email_sent = False
        try:
            if user_input.email:
                email_sent = send_email(
                    user_input.email,
                    "Profit Results (Precomputed)",
                    html_body,
                    image_bytes=contour_png_bytes,
                )
        except Exception as e:
            print(f"⚠️ Email issue: {e}")
            email_sent = False

        return {
            "bin_key": user_bin_key,
            "matched_bin_key": matched_bin_key,
            "used_fallback": used_fallback,
            "record_u_id": record_u_id,
            "similarity": similarity,
            "fallback_distance": fallback_distance,
            "resolved_prev_idx": int(prev_index),
            "resolved_till_idx": int(till_index),
            "eonr": eonr,
            "profit_results": profit_rows,
            "qp_params": qp_params,
            "used_prices": {
                "yield_mean": float(yield_mean),
                "yield_std": float(yield_std),
                "fert_mean": float(fert_mean),
                "fert_std": float(fert_std),
            },
            "email_sent": bool(email_sent),
            "contour_chart_base64": contour_chart_base64,
            "contour_data": contour_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


# ============================================================
# Optional health
# ============================================================
@app.get("/health/")
def health():
    return {"status": "ok"}


# Run:
# uvicorn app:app --host 0.0.0.0 --port 8010 --reload