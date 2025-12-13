# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from mysql.connector import pooling
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import json
import hashlib
import matplotlib.pyplot as plt
from statistics import mean, stdev
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple


# ----------------------------
# FastAPI & CORS
# ----------------------------
app = FastAPI()

# CORS FIX: اجازه فقط به مبدأهای مشخص (دامنه اصلی + الگوی ngrok)
# فقط بخش CORS را آپدیت کن
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.numericag.com",
        "https://numericag.com",
    ],
    allow_origin_regex=r"https://[a-z0-9-]+\.ngrok-free\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",                    # ← اضافه شد
        "ngrok-skip-browser-warning" # ← اضافه شد (علت اصلی خطا)
    ],
    expose_headers=["Content-Type", "Content-Length"],
    max_age=86400,
)

# ----------------------------
# Input schema
# ----------------------------
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
    # موقعیت از UI
    lati: Optional[float] = None
    longti: Optional[float] = None
    PREV_VALUE: Optional[float] = None
    TILLAGE_VALUE: Optional[float] = None

# ----------------------------
# Maps
# ----------------------------
previous_crop_map = {
    "Low nutrient": 0,
    "Moderate nutrient": 0.75,
    "High nutrient": 1,
}
tillage_type_map = {
    "No till": 0,
    "conventionnel": 1,
}

# ----------------------------
# DB pool
# ----------------------------
cnxpool = pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    host='127.0.0.1',
    port=....,
    user='root',
    password='....',
    database='....'
)

# ----------------------------
# Helpers: prices & bins
# ----------------------------
def fetch_default_corn_price_stats():
    """
    اگر کاربر میانگین/انحراف معیار قیمت را صفر داد، از FRED دریافت می‌کنیم؛
    در صورت خطا fallback ثابت می‌دهیم.
    """
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
        prices = [float(obs["value"]) for obs in r.json().get("observations", []) if obs["value"] != "."]
        if not prices:
            return 175.0, 43.0
        return float(mean(prices)), float(stdev(prices) if len(prices) > 1 else 1.0)
    except Exception:
        return 175.0, 43.0

def get_default_urea_cost_stats():
    return 0.6, 0.14

def price_bins_from_stats(mu: float, sigma: float):
    """سه سناریو مطابق منطق ساخت ExpectedProfitLookup_Rebuilt"""
    return [
        (mu - sigma, 0.20),
        (mu,         0.50),
        (mu + sigma, 0.30),
    ]

def urea_price_bins():
    """سناریوهای هزینهٔ اوره مطابق کد ساخت جدول ExpectedProfitLookup_Rebuilt"""
    return [
        (0.30, 0.10),
        (0.45, 0.25),
        (0.60, 0.30),
        (0.90, 0.20),
        (1.20, 0.15),
    ]

# ----------------------------
# QP yield model
# ----------------------------
def qp_yield(N: float, Y0: float, Ymax: float, Nymax: float) -> float:
    b = 2 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    return (Y0 + b * N + c * (N ** 2)) if N <= Nymax else Ymax

# ----------------------------
# Email helpers
# ----------------------------
def send_email(email: str, subject: str, html_body: str, image_path: Optional[str] = None):
    sender_email = "....."
    sender_password = "....."
    msg = MIMEMultipart('related')
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = subject
    alt = MIMEMultipart('alternative')
    msg.attach(alt)
    alt.attach(MIMEText(html_body, 'html', _charset='utf-8'))
    if image_path:
        try:
            with open(image_path, 'rb') as img:
                mime_image = MIMEImage(img.read())
                mime_image.add_header('Content-ID', '<chart>')
                mime_image.add_header('Content-Disposition', 'inline', filename='chart.png')
                msg.attach(mime_image)
        except Exception as e:
            print(f"⚠️ Could not attach image: {e}")
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"❌ Email failed: {e}")

def generate_chart(profit_results: List[Dict[str, Any]], filepath: str = "profit_chart.png"):
    x = [row['n_rate'] for row in profit_results]
    y1 = [row['expected_nrcf'] for row in profit_results]
    y2 = [row['efb'] for row in profit_results]
    plt.figure(figsize=(8, 4))
    plt.plot(x, y1, label="Expected NRCF", marker='o')
    plt.plot(x, y2, label="EFB", marker='o')
    plt.xlabel("N Rate (kg/ha)")
    plt.ylabel("$ per ha")
    plt.title("Expected NRCF and EFB vs. Nitrogen Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# ----------------------------
# Feature ranges
# ----------------------------
@app.get("/feature_ranges/")
def get_feature_ranges():
    try:
        conn = cnxpool.get_connection()
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
        conn.close()
        return df.to_dict(orient='records')[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feature ranges: {e}")

# ----------------------------
# Utils for binning (for key)
# ----------------------------
def bin_feature_values(values: np.ndarray, num_bins: int):
    bins = np.linspace(min(values), max(values), num_bins + 1)
    bin_means = [(bins[i] + bins[i + 1]) / 2 for i in range(num_bins)]
    return bin_means

def find_closest_bin(value: float, bin_means: List[float]) -> int:
    return min(range(len(bin_means)), key=lambda i: abs(bin_means[i] - value))

# ----------------------------
# Compute P(best N) distribution
# ----------------------------
def compute_p_best_by_n(
    conn,
    q: int,
    record_u_id: int,
    n_rates: List[int],
    yield_mean: float,
    yield_std: float
) -> Tuple[List[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    خروجی: (p_best_by_n, qp_params)
    - p_best_by_n: لیست {"n_rate": int, "p_best": float}
    - qp_params: {"Y0": float, "Ymax": float, "Nymax": float} یا None
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT Y0, Ymax, Nymax FROM ErrorProb WHERE q=%s AND record_u_id=%s LIMIT 1",
        (int(q), int(record_u_id))
    )
    row = cursor.fetchone()
    cursor.close()

    if not row:
        return [], None

    Y0 = float(row["Y0"]); Ymax = float(row["Ymax"]); Nymax = float(row["Nymax"])
    qp_params = {"Y0": Y0, "Ymax": Ymax, "Nymax": Nymax}

    def qp_yield_local(N: float) -> float:
        b = 2 * (Ymax - Y0) / Nymax
        c = (Y0 - Ymax) / (Nymax ** 2)
        return (Y0 + b * N + c * (N ** 2)) if N <= Nymax else Ymax

    price_bins = price_bins_from_stats(float(yield_mean), float(yield_std))   # [(price, p)]
    cost_bins  = urea_price_bins()                                            # [(cost,  p)]
    mass_by_n = {int(N): 0.0 for N in n_rates}

    for price, p_prob in price_bins:
        for cost, c_prob in cost_bins:
            joint = p_prob * c_prob
            if joint <= 0:
                continue
            best_n = None
            best_nrcf = -1e30
            for N in n_rates:
                yld = qp_yield_local(N)
                nrcf = yld * price - N * cost
                if nrcf > best_nrcf:
                    best_nrcf = nrcf
                    best_n = int(N)
            mass_by_n[best_n] += joint

    total = sum(mass_by_n.values())
    if total <= 0:
        return [], qp_params

    p_best_by_n = [{"n_rate": int(N), "p_best": float(mass_by_n[N] / total)} for N in n_rates]
    return p_best_by_n, qp_params


# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/calculate_similarity/")
def calculate_similarity(user_input: UserInput):
    try:
        # --- prices config (used or defaults)
        yield_mean, yield_std = (user_input.yieldPriceMean, user_input.yieldPriceStdDev)
        fert_mean, fert_std   = (user_input.fertilizerCostMean, user_input.fertilizerCostStdDev)
        if yield_mean == 0 or yield_std == 0:
            yield_mean, yield_std = fetch_default_corn_price_stats()
        if fert_mean == 0 or fert_std == 0:
            fert_mean, fert_std = get_default_urea_cost_stats()

        # --- read data for binning
        conn = cnxpool.get_connection()
        df = pd.read_sql("SELECT * FROM sample_data_test", conn)
        conn.close()

        # --- build bin_key from user inputs
        user_values = [user_input.ACLAY, user_input.SOM, user_input.CHU, user_input.AWDR]
        bin_indices = []
        for i, feature in enumerate(['ACLAY', 'SOM', 'CHU', 'AWDR']):
            bins = bin_feature_values(df[feature].values, 5)
            bin_indices.append(find_closest_bin(user_values[i], bins))

        prev_val = previous_crop_map.get(user_input.PREV)
        till_val = tillage_type_map.get(user_input.TILLAGE)
        if prev_val is None or till_val is None:
            raise HTTPException(status_code=400, detail="Invalid categorical input")
        bin_indices += [prev_val, till_val]

        bin_key_str = ",".join(map(str, bin_indices))
        bin_key = hashlib.sha256(bin_key_str.encode()).hexdigest()

        # --- find best match in mode4
        conn = cnxpool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM mode4 WHERE q = %s AND bin_key = %s", (user_input.q, bin_key))
        matches = cursor.fetchall()

        used_fallback = False
        if not matches:
            # fallback: از بین همین q بالاترین similarity
            cursor.execute("SELECT * FROM mode4 WHERE q = %s ORDER BY similarity DESC LIMIT 1", (user_input.q,))
            best_match = cursor.fetchone()
            used_fallback = True
        else:
            best_match = max(matches, key=lambda x: x['similarity'])

        if not best_match:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="No match found for this q.")

        record_u_id = best_match['record_u_id']

        # --- profit results for this (q, record_u_id)
        cursor.execute(
            "SELECT n_rate, prob_gt_1000, prob_gt_1500, prob_gt_2000, prob_gt_2500, expected_nrcf, efb "
            "FROM ExpectedProfitLookup_Rebuilt WHERE q = %s AND record_u_id = %s ORDER BY n_rate ASC",
            (user_input.q, record_u_id)
        )
        profit_rows = cursor.fetchall()
        cursor.close()

        if not profit_rows:
            conn.close()
            raise HTTPException(status_code=404, detail="No profit rows for this match.")

        # --- compute P(best N) based on QP model params in ErrorProb (if available)
        n_rates = [int(r["n_rate"]) for r in profit_rows]
        p_best_by_n, qp_params = compute_p_best_by_n(
            conn=conn,
            q=user_input.q,
            record_u_id=record_u_id,
            n_rates=n_rates,
            yield_mean=float(yield_mean),
            yield_std=float(yield_std),
        )
        conn.close()

        # --- email
        try:
            max_efb_row = max(profit_rows, key=lambda row: row['efb'])
        except Exception:
            max_efb_row = profit_rows[0]

        html_rows = "".join([
            ("<tr style='background-color: #d1ffd1; font-weight: bold;'>" if row['efb'] == max_efb_row['efb'] else "<tr>") +
            f"<td>{row['n_rate']}</td><td>{row['prob_gt_1000']}</td><td>{row['prob_gt_1500']}</td>" +
            f"<td>{row['prob_gt_2000']}</td><td>{row['prob_gt_2500']}</td>" +
            f"<td>{float(row['expected_nrcf']):.2f}</td><td>{float(row['efb']):.2f}</td></tr>"
            for row in profit_rows
        ])

        warning_msg = ""
        if used_fallback:
            warning_msg = "<p style='color:red'><b>⚠️ This result was based on the closest possible match. No exact match was found.</b></p>"

        html_body = f"""
        <html><head><style>
        table {{ border-collapse: collapse; width: 100%; font-family: Arial; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        </style></head><body>
        <h3>📌 Best Match Info</h3>
        {warning_msg}
        <p>Record ID: <b>{record_u_id}</b></p>
        <p>Similarity Score: <b>{best_match['similarity']:.4f}</b></p>
        <p>⭐ Economically Optimum N Rate (EONR): <b>{max_efb_row['n_rate']} kg/ha</b></p>
        <h3>📊 Profit Results</h3>
        <table><tr><th>N Rate</th><th>P &gt; $1000</th><th>P &gt; $1500</th><th>P &gt; $2000</th>
        <th>P &gt; $2500</th><th>Expected NRCF</th><th>EFB</th></tr>
        {html_rows}</table>
        <h3>📈 Chart</h3><img src="cid:chart" width="600" />
        </body></html>
        """

        chart_path = "profit_chart.png"
        try:
            generate_chart(profit_rows, chart_path)
            if user_input.email:
                send_email(user_input.email, "Similarity Match & Profit Results", html_body, image_path=chart_path)
        except Exception as e:
            print(f"⚠️ Chart/email issue: {e}")

        # --- final response
        return {
            "bin_key": bin_key,
            "record_u_id": record_u_id,
            "similarity": float(best_match['similarity']),
            "profit_results": [
                {
                    "n_rate": int(r["n_rate"]),
                    "prob_gt_1000": float(r["prob_gt_1000"]),
                    "prob_gt_1500": float(r["prob_gt_1500"]),
                    "prob_gt_2000": float(r["prob_gt_2000"]),
                    "prob_gt_2500": float(r["prob_gt_2500"]),
                    "expected_nrcf": float(r["expected_nrcf"]),
                    "efb": float(r["efb"]),
                } for r in profit_rows
            ],
            "p_best_by_n": p_best_by_n,  # ← NEW
            "qp_params": qp_params,
            "used_prices": {
                "yield_mean": float(yield_mean),
                "yield_std": float(yield_std),
                "fert_mean": float(fert_mean),
                "fert_std": float(fert_std)
            },
            "used_fallback": used_fallback,
            "email_sent": bool(user_input.email),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# // 29 Oct 2025
# Run: uvicorn backend.app:app --reload
