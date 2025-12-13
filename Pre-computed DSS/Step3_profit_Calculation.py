import mysql.connector
import pandas as pd
import numpy as np
from statistics import mean, stdev
from scipy.stats import norm
from tqdm import tqdm
import time
import requests
from datetime import datetime

# ----------------------------
# تابع مدل QP (بدون تغییر)
# ----------------------------
def qp_yield(N, Y0, Ymax, Nymax):
    b = 2 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    return Y0 + b * N + c * N ** 2 if N <= Nymax else Ymax

# ----------------------------
# دریافت قیمت ذرت از FRED (بدون تغییر)
# ----------------------------
def fetch_corn_price_bins():
    api_key = "d5eac415442010f6a8567b47c69d4449"
    current_year = datetime.now().year
    start_year = current_year - 5
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=PMAIZMTUSDM&api_key={api_key}&file_type=json&observation_start={start_year}-01-01"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("❌ Failed to fetch corn prices from FRED.")
    prices = [float(obs["value"]) for obs in response.json()["observations"] if obs["value"] != "."]
    μ, σ = mean(prices), stdev(prices)
    print(f"🌽 Corn price stats → Mean: {μ:.2f}, Std: {σ:.2f}")
    return [(μ - σ, 0.2), (μ, 0.5), (μ + σ, 0.3)]

# ----------------------------
# قیمت‌های اوره تعریف شده دستی (بدون تغییر)
# ----------------------------
def get_urea_price_bins():
    return [
        (0.3,  0.1),
        (0.45, 0.25),
        (0.6,  0.3),
        (0.9,  0.2),
        (1.2,  0.15)
    ]

# ----------------------------
# اتصال به دیتابیس (+ پایداری/ادامه)
# ----------------------------
def connect_db():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='....',
        port=....,
        database='....',
    )

conn = connect_db()
cursor = conn.cursor()

# خواندن داده‌های ورودی
df = pd.read_sql("SELECT * FROM ErrorProb", conn)
print(f"📦 Loaded ErrorProb records: {len(df)}")

# ----------------------------
# ساخت جدول خروجی اگر نبود (بدون DROP)
# ----------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ExpectedProfitLookup_Rebuilt (
        id INT AUTO_INCREMENT PRIMARY KEY,
        q INT,
        bin_key VARCHAR(64),
        record_u_id INT,
        n_rate INT,
        prob_gt_1000 FLOAT,
        prob_gt_1500 FLOAT,
        prob_gt_2000 FLOAT,
        prob_gt_2500 FLOAT,
        expected_nrcf FLOAT,
        efb FLOAT
    )
""")
conn.commit()

# --- ساخت ایندکس‌ها به‌صورت سازگار با نسخه‌های مختلف MySQL ---
def ensure_index(cursor, table, index_name, columns):
    """
    اگر ایندکس وجود نداشت، بساز (بدون استفاده از IF NOT EXISTS).
    columns باید به‌صورت 'col1, col2' پاس داده شود.
    """
    cursor.execute(
        """
        SELECT COUNT(1)
        FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND index_name = %s
        """,
        (table, index_name)
    )
    exists = cursor.fetchone()[0]
    if not exists:
        cursor.execute(f"CREATE INDEX {index_name} ON {table} ({columns})")

ensure_index(cursor, "ExpectedProfitLookup_Rebuilt", "idx_eplr_q_rec",   "q, record_u_id")
ensure_index(cursor, "ExpectedProfitLookup_Rebuilt", "idx_eplr_q_rec_n", "q, record_u_id, n_rate")
conn.commit()
print("📜 Table & indexes for ExpectedProfitLookup_Rebuilt are ready.")

# ----------------------------
# آماده‌سازی ثابت‌ها (بدون تغییر محاسبات)
# ----------------------------
corn_price_bins = fetch_corn_price_bins()
urea_price_bins = get_urea_price_bins()
n_rates = list(range(0, 260, 10))
thresholds = [1000, 1500, 2000, 2500]
n_needed = len(n_rates)

# گروه‌بندی ورودی
groups = df.groupby(["q", "record_u_id"])

# تابع کمکی: بررسی کامل بودن یک زوج (q, record_u_id)
def pair_is_complete(q, record_u_id):
    cursor.execute(
        "SELECT COUNT(*) FROM ExpectedProfitLookup_Rebuilt WHERE q=%s AND record_u_id=%s",
        (int(q), int(record_u_id))
    )
    cnt = cursor.fetchone()[0]
    return cnt == n_needed

# تابع کمکی: پاک‌کردن خروجی ناقص یک زوج (برای بازسازی همان زوج)
def clear_pair(q, record_u_id):
    cursor.execute(
        "DELETE FROM ExpectedProfitLookup_Rebuilt WHERE q=%s AND record_u_id=%s",
        (int(q), int(record_u_id))
    )

# ----------------------------
# حلقه محاسبات (منطق بدون تغییر) + قابلیت ادامه
# ----------------------------
for (q, record_u_id), group in tqdm(groups, desc="💰 Calculating profit"):
    try:
        # اگر قبلاً کامل ساخته شده، از این زوج عبور کن
        if pair_is_complete(q, record_u_id):
            # ✅ این زوج قبلاً کامل ذخیره شده است
            # برای سرعت بیشتر، محاسبات را تکرار نمی‌کنیم
            continue
        else:
            # اگر ناقص بود، همان زوج را پاک کن تا مجدداً از صفر برای همین زوج بسازیم
            clear_pair(q, record_u_id)
            conn.commit()

        bin_key = group["bin_key"].values[0]
        Y0 = group["Y0"].values[0]
        Ymax = group["Ymax"].values[0]
        Nymax = group["Nymax"].values[0]

        def model_yield(N):
            return qp_yield(N, Y0, Ymax, Nymax)

        # محاسبه پایه (بدون تغییر)
        base_nrcf = 0
        for price, p_prob in corn_price_bins:
            for cost, c_prob in urea_price_bins:
                for _, row in group.iterrows():
                    joint_prob = p_prob * c_prob * row["normalized_prob"]
                    pred_yld = model_yield(0)
                    revenue = pred_yld * price
                    base_nrcf += revenue * joint_prob

        # محاسبه برای همه نرخ‌های N (بدون تغییر)
        for N in n_rates:
            exp_profits = []
            prob_gt = {thr: [] for thr in thresholds}
            for price, p_prob in corn_price_bins:
                for cost, c_prob in urea_price_bins:
                    pred_yld = model_yield(N)
                    revenue = pred_yld * price
                    cost_val = N * cost
                    nrcf = revenue - cost_val
                    for _, row in group.iterrows():
                        joint_prob = p_prob * c_prob * row["normalized_prob"]
                        exp_profits.append(nrcf * joint_prob)
                        for thr in thresholds:
                            if nrcf > thr:
                                prob_gt[thr].append(joint_prob)

            expected_nrcf = sum(exp_profits)
            efb = expected_nrcf - base_nrcf
            prob_values = [round(sum(prob_gt[thr]) * 100, 2) for thr in thresholds]

            # درج رکورد (همان خروجی‌ها؛ بدون تغییر محاسبات)
            cursor.execute("""
                INSERT INTO ExpectedProfitLookup_Rebuilt (
                    q, bin_key, record_u_id, n_rate,
                    prob_gt_1000, prob_gt_1500, prob_gt_2000, prob_gt_2500,
                    expected_nrcf, efb
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                int(q), str(bin_key), int(record_u_id), int(N),
                prob_values[0], prob_values[1], prob_values[2], prob_values[3],
                float(expected_nrcf), float(efb)
            ))

        # کامیت در پایان هر زوج → قابلیت ازسرگیری
        conn.commit()
        print(f"✅ Profit results saved for q={q}, record_u_id={record_u_id}")

    except mysql.connector.errors.OperationalError as e:
        # اگر اتصال پرید، تلاش برای اتصال مجدد و ادامه همین زوج
        print(f"⚠️ DB connection lost. Reconnecting... ({e})")
        try:
            cursor.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        time.sleep(2)
        conn = connect_db()
        cursor = conn.cursor()
        # زوج فعلی ممکنه ناقص نوشته شده باشد؛ پاک و تکرار همان زوج در اجرای بعدی
        clear_pair(q, record_u_id)
        conn.commit()
        print(f"♻️ Cleared partial results for q={q}, record_u_id={record_u_id}. Will recompute on next run.")
    except Exception as e:
        # سایر خطاها را لاگ کن ولی اجازه بده حلقه ادامه یابد
        print(f"❌ Error in profit calculation for q={q}, record_u_id={record_u_id}: {e}")
        # پاک‌کردن خروجی ناقص این زوج تا در اجرای بعدی کامل ساخته شود
        try:
            clear_pair(q, record_u_id)
            conn.commit()
        except:
            pass

cursor.close()
conn.close()
print("🎉 All profit results saved (or will resume next run for any incomplete pairs).")
