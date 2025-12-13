import numpy as np
import pandas as pd
import mysql.connector
import hashlib

# مپینگ برای تبدیل متون به اعداد
previous_crop_map = {
    "Low nutrient": 0,
    "Moderate nutrient": 0.75,
    "High nutrient": 1,
}

tillage_type_map = {
    "No till": 0,
    "Conventionnel": 1,
}

# تابع محاسبه شباهت برای ویژگی‌های پیوسته و کتگوریکال
def calculate_similarity(features_j, features_u, features_max, features_min,
                         delta_lambdas, q, Np, Nt, num_continuous):
    lambda_j_u = 1.0
    feature_similarities = []
    
    for k in range(len(features_j)):
        if k < num_continuous:
            normalized_diff = abs((features_j[k] - features_u[k]) / (features_max[k] - features_min[k]))
            feature_similarity = 1 - delta_lambdas[k] * (normalized_diff ** q)
        else:
            feature_similarity = 1.0 if features_j[k] == features_u[k] else 0.0
        
        lambda_j_u *= feature_similarity
        feature_similarities.append(feature_similarity)
    
    lambda_j_u *= (Np / Nt)
    return lambda_j_u, feature_similarities

# تابع تقسیم داده‌های پیوسته به بین‌ها
def bin_feature_values(feature_values, num_bins):
    bins = np.linspace(min(feature_values), max(feature_values), num_bins + 1)
    bin_indices = np.digitize(feature_values, bins) - 1
    bin_means = [(bins[i] + bins[i+1]) / 2 for i in range(num_bins)]
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    return bin_indices, bin_means

# دریافت داده از دیتابیس
def fetch_data_from_mysql():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='.....',
            port=....,
            database='....'
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM ....")
        results = cursor.fetchall()
        df = pd.DataFrame(results)
        cursor.close()
        conn.close()
        return df
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# پردازش و ذخیره نتایج

def process_data_and_store_results(df, q_values, num_bins):
    try:
        df['PREV'] = df['Prev Type'].map(previous_crop_map)
        df['TILLAGE'] = df['TILLAGE'].map(tillage_type_map)

        features_columns = ['ACLAY', 'SOM', 'CHU', 'AWDR']
        categorical_columns = ['PREV', 'TILLAGE']
        delta_lambdas = np.ones(len(features_columns) + len(categorical_columns))
        features_max = df[features_columns].max().values
        features_min = df[features_columns].min().values
        Np = Nt = len(features_columns) + len(categorical_columns)

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='....',
            port=....,
            database='....'
        )
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS mode4")
        cursor.execute("""
            CREATE TABLE mode4 (
                id INT AUTO_INCREMENT PRIMARY KEY,
                q INT,
                bin_key VARCHAR(64),
                record_j_id INT,
                record_u_id INT,
                record_j_data TEXT,
                record_u_data TEXT,
                similarity FLOAT,
                feature_similarities TEXT,
                bins_j_data TEXT,
                bins_u_data TEXT
            )
        """)

        binned_features = {}
        for feature in features_columns:
            bin_indices, bin_means = bin_feature_values(df[feature].values, num_bins)
            binned_features[feature] = {'indices': bin_indices, 'means': bin_means}

        num_combinations = (num_bins ** len(features_columns)) * (2 ** len(categorical_columns))
        print(f"\n✅ Total user input combinations: {num_combinations}")

        for q in q_values:
            print(f"\n🚀 Processing for q = {q}")
            for i in range(len(df)):
                if i % 10 == 0:
                    print(f"🔄 Processing record {i+1}/{len(df)} for q = {q}", flush=True)
                for user_case in range(num_combinations):
                    user_bin_indices = list(np.unravel_index(
                        user_case,
                        [num_bins] * len(features_columns) + [2] * len(categorical_columns)
                    ))

                    features_u_cont = [binned_features[features_columns[j]]['means'][user_bin_indices[j]]
                                       for j in range(len(features_columns))]
                    features_u_cat = user_bin_indices[len(features_columns):]

                    features_j_cont = [binned_features[feature]['means'][binned_features[feature]['indices'][i]]
                                       for feature in features_columns]
                    features_j_cat = [df.iloc[i][cat] for cat in categorical_columns]

                    features_j_combined = features_j_cont + features_j_cat
                    features_u_combined = features_u_cont + features_u_cat

                    similarity, feature_similarities = calculate_similarity(
                        features_j_combined,
                        features_u_combined,
                        features_max,
                        features_min,
                        delta_lambdas,
                        q,
                        Np,
                        Nt,
                        num_continuous=len(features_columns)
                    )

                    record_j_data = df.iloc[i].to_json()
                    record_u_data = str(features_u_combined)

                    bins_j_data = {
                        "bins": [binned_features[feature]['indices'][i] for feature in features_columns] + features_j_cat
                    }
                    bins_u_data = {
                        "bins": user_bin_indices
                    }

                    bin_key_str = ",".join(map(str, user_bin_indices))
                    bin_key = hashlib.sha256(bin_key_str.encode()).hexdigest()

                    cursor.execute("""
                        INSERT INTO mode4 (q, bin_key, record_j_id, record_u_id, record_j_data, record_u_data,
                                           similarity, feature_similarities, bins_j_data, bins_u_data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        q, bin_key, i, user_case,
                        record_j_data, record_u_data,
                        similarity, str(feature_similarities), str(bins_j_data), str(bins_u_data)
                    ))

        conn.commit()
        cursor.close()
        conn.close()
        print("\n✅ Data processed and saved to mode4 successfully.")

    except mysql.connector.Error as err:
        print(f"Database error: {err}")

# اجرای کد
df = fetch_data_from_mysql()
if df is not None:
    process_data_and_store_results(df, q_values=[1, 2, 5], num_bins=5)
else:
    print("Failed to fetch data from MySQL.")