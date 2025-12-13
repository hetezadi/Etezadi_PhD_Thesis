import mysql.connector
import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import json
import time

# ----------------------------
# QP yield model
# ----------------------------
def qp_yield(N, Y0, Ymax, Nymax):
    b = 2 * (Ymax - Y0) / Nymax
    c = (Y0 - Ymax) / (Nymax ** 2)
    return Y0 + b * N + c * N ** 2 if N <= Nymax else Ymax

# ----------------------------
# اتصال به دیتابیس
# ----------------------------
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='....',
    port=....,
    database='....',
    autocommit=False  # تراکنش‌ها را خودمان کنترل می‌کنیم
)
cursor = conn.cursor()

# ----------------------------
# ساخت امن جدول‌های لازم (بدون DROP)
# ----------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ErrorProb (
        id INT AUTO_INCREMENT PRIMARY KEY,
        q INT,
        bin_key VARCHAR(64),
        record_u_id INT,
        record_j_id INT,
        NTOT FLOAT,
        YIELD FLOAT,
        predicted_yield FLOAT,
        weighted_error FLOAT,
        avg_error FLOAT,
        std_error FLOAT,
        zero_error_prob FLOAT,
        normalized_prob FLOAT,
        similarity FLOAT,
        Y0 FLOAT,
        Ymax FLOAT,
        Nymax FLOAT
    )
""")

# جدول چک‌پوینت برای دنبال‌کردن گروه‌هایی که کامل شده‌اند
cursor.execute("""
    CREATE TABLE IF NOT EXISTS QP_Checkpoint (
        q INT NOT NULL,
        record_u_id INT NOT NULL,
        status ENUM('done','in_progress') NOT NULL DEFAULT 'done',
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (q, record_u_id)
    )
""")

# (اختیاری ولی مفید) تلاش برای جلوگیری از رکوردهای تکراری در ErrorProb
# ممکن است ایندکس از قبل وجود داشته باشد؛ در این‌صورت خطا را نادیده می‌گیریم.
try:
    cursor.execute("ALTER TABLE ErrorProb ADD UNIQUE KEY uq_err (q, record_u_id, record_j_id)")
except mysql.connector.Error:
    pass

conn.commit()

# ----------------------------
# بارگذاری داده‌ی ورودی
# ----------------------------
df = pd.read_sql("SELECT * FROM mode4", conn)
print(f"📦 Records loaded: {len(df)}")

# ----------------------------
# پارامترهای QP براساس پایان‌نامه
# ----------------------------
Y0_range = np.arange(0, 21, 1)
Ymax_offsets = np.arange(0, 21, 1)
Nymax_range = np.arange(10, 260, 10)

def generate_ymax_range(y0):
    return [y0 + offset for offset in Ymax_offsets if y0 + offset <= 20]

def find_best_qp_model_max_p0(N_vals, Y_vals, weights):
    best_params = None
    best_p0 = -1
    for Y0 in Y0_range:
        for Ymax in generate_ymax_range(Y0):
            for Nymax in Nymax_range:
                preds = np.array([qp_yield(n, Y0, Ymax, Nymax) for n in N_vals])
                residuals = preds - Y_vals
                weighted_errors = residuals * weights
                avg_error = np.average(weighted_errors)
                std_error = np.std(weighted_errors)
                std_error = max(std_error, 0.1)
                zero_probs = norm.pdf(weighted_errors, avg_error, std_error)
                p0 = float(np.sum(zero_probs))
                if p0 > best_p0:
                    best_p0 = p0
                    best_params = (float(Y0), float(Ymax), float(Nymax))
    return best_params

# ----------------------------
# بازیابی گروه‌های تکمیل‌شده از چک‌پوینت
# ----------------------------
cursor.execute("SELECT q, record_u_id FROM QP_Checkpoint WHERE status='done'")
done_groups = set(cursor.fetchall())  # مجموعه‌ای از (q, record_u_id)
print(f"🧭 Checkpoint: {len(done_groups)} groups already done")

# ----------------------------
# پردازش گروه‌ها با قابلیت ادامه از نقطهٔ قبل
# ----------------------------
grouped = df.groupby(["q", "record_u_id"])
for (q, record_u_id), group in tqdm(grouped, desc="🔄 Processing QP groups"):
    key = (int(q), int(record_u_id))
    if key in done_groups:
        # این گروه قبلاً کامل شده؛ رد شو
        continue

    try:
        # شروع تراکنش گروه
        conn.start_transaction()
        print(f"\n➡️ Group q={q}, record_u_id={record_u_id}, size={len(group)}")

        # اگر قبلاً این گروه تا حدی درج شده باشد، پاک می‌کنیم تا تمیز بسازیم
        cursor.execute(
            "DELETE FROM ErrorProb WHERE q=%s AND record_u_id=%s",
            (int(q), int(record_u_id))
        )

        NTOT_vals = []
        YIELD_vals = []
        similarity_vals = []

        for row in group.itertuples(index=False):
            record_data = json.loads(row.record_j_data)
            NTOT_vals.append(float(record_data["NTOT"]))
            YIELD_vals.append(float(record_data["YIELD"]))
            similarity_vals.append(float(row.similarity))

        NTOT_vals = np.array(NTOT_vals, dtype=float)
        YIELD_vals = np.array(YIELD_vals, dtype=float)
        similarity_vals = np.array(similarity_vals, dtype=float)

        bin_key = str(group["bin_key"].values[0])

        # انتخاب بهترین پارامترها
        Y0, Ymax, Nymax = find_best_qp_model_max_p0(NTOT_vals, YIELD_vals, similarity_vals)

        # پیش‌بینی و محاسبات خطا
        preds = np.array([qp_yield(n, Y0, Ymax, Nymax) for n in NTOT_vals])
        residuals = preds - YIELD_vals
        weighted_errors = residuals * similarity_vals

        avg_error = float(np.average(weighted_errors))
        std_error = float(max(np.std(weighted_errors), 0.1))

        zero_probs = norm.pdf(weighted_errors, avg_error, std_error)
        total_prob = float(np.sum(zero_probs))
        norm_probs = [float(p / total_prob) if total_prob > 0 else 0.0 for p in zero_probs]

        # آماده‌سازی درج‌های گروه به‌صورت batch
        rows_to_insert = []
        for idx, row in enumerate(group.itertuples(index=False)):
            rows_to_insert.append((
                int(q), bin_key, int(record_u_id), int(row.record_j_id),
                float(NTOT_vals[idx]), float(YIELD_vals[idx]),
                float(preds[idx]), float(weighted_errors[idx]),
                float(avg_error), float(std_error),
                float(zero_probs[idx]), float(norm_probs[idx]),
                float(similarity_vals[idx]), float(Y0), float(Ymax), float(Nymax)
            ))

        cursor.executemany("""
            INSERT INTO ErrorProb (
                q, bin_key, record_u_id, record_j_id, NTOT, YIELD, predicted_yield,
                weighted_error, avg_error, std_error, zero_error_prob,
                normalized_prob, similarity, Y0, Ymax, Nymax
            ) VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
        """, rows_to_insert)

        # علامت‌زدن این گروه به‌عنوان انجام‌شده در چک‌پوینت
        cursor.execute("""
            INSERT INTO QP_Checkpoint (q, record_u_id, status)
            VALUES (%s, %s, 'done')
            ON DUPLICATE KEY UPDATE status='done', updated_at=CURRENT_TIMESTAMP
        """, (int(q), int(record_u_id)))

        conn.commit()
        print(f"✅ Group {q}-{record_u_id} saved and checkpointed.")

    except Exception as e:
        # اگر مشکلی پیش آمد، همه چیز این گروه رول‌بک شود
        conn.rollback()
        print(f"⚠️ Error in group q={q}, record_u_id={record_u_id}: {e}")
        # ادامه بده به گروه بعدی؛ اجرای بعدی دوباره همین گروه را تمیز خواهد ساخت

# ----------------------------
# پایان
# ----------------------------
cursor.close()
conn.close()
print("🎉 All error probabilities calculated and saved (resume-safe).")
