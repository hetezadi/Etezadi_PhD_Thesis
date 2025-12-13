import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import seaborn as sns
import matplotlib.pyplot as plt

df = '/content/fullMainData.xlsx'
data = pd.read_excel(df, sheet_name='mainData')
data.head()

data['log_Om'] = np.log10(data['Om_p'])
data['Field_number'] = 'F' + data['Field_no'].astype(str)
data['Field_number_SS'] =  data['Field_number'] + '  ' +data['TYPE_2']
data['Avelog_f'] = data.groupby('Field_number')['log_Om'].transform('mean')
data['Avelog_SS'] = data.groupby('Field_number_SS')['log_Om'].transform('mean')
'''F_avg = data.groupby('Field_number')['log_Om'].mean()
print(F_avg)'''
data['ij']=data['Avelog_SS']/data['Avelog_f']
data

# --- Summary statistics for Table ---
summary_raw = data['Om_p'].describe()[['min','max','mean','50%','std']]
summary_log = data['log_Om'].describe()[['min','max','mean','50%','std']]

# ساخت دیتافریم نهایی جدول
summary_table = pd.DataFrame({
    'Statistic': ['Minimum','Maximum','Mean','Median','Std. Dev.'],
    'Raw SOM (%)': [
        summary_raw['min'],
        summary_raw['max'],
        summary_raw['mean'],
        summary_raw['50%'],
        summary_raw['std']
    ],
    'Log(SOM)': [
        summary_log['min'],
        summary_log['max'],
        summary_log['mean'],
        summary_log['50%'],
        summary_log['std']
    ]
})

print(summary_table)

# =========================
# Figures for the manuscript
# =========================

import os
import matplotlib.ticker as mtick

# پوشه ذخیره خروجی شکل‌ها
os.makedirs("figs", exist_ok=True)

# --- (اختیاری) پاکسازی داده‌های کلیدی ---
plot_df = data.copy()
plot_df = plot_df.dropna(subset=['Om_p', 'log_Om', 'Field_number', 'TYPE_2', 'ij'])

# =========================
# Figure 4: Boxplots of SOM (%) by field
# =========================
# ترتیب فیلدها بر اساس میانه SOM برای نمایش مرتب
field_order = (
    plot_df.groupby('Field_number')['Om_p']
    .median()
    .sort_values(ascending=False)
    .index
    .tolist()
)

plt.figure(figsize=(14, 6))
sns.boxplot(
    data=plot_df,
    x='Field_number',
    y='Om_p',
    order=field_order,
    showfliers=False
)
sns.stripplot(
    data=plot_df,
    x='Field_number',
    y='Om_p',
    order=field_order,
    size=2, alpha=0.3, color='k'
)
plt.xlabel("Field")
plt.ylabel("SOM (%)")
plt.title("Distribution of SOM (%) across fields")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig("figs/figure4_boxplot_som_by_field.png", dpi=600)
plt.close()

# =========================
# Figure 5: Histogram + KDE of log(SOM)
# =========================
plt.figure(figsize=(7, 5))
sns.histplot(plot_df['log_Om'], bins=30, kde=True, edgecolor=None)
plt.xlabel("log10(SOM)")
plt.ylabel("Count")
plt.title("Distribution of log-transformed SOM")
plt.tight_layout()
plt.savefig("figs/figure5_hist_kde_logsom.png", dpi=600)
plt.close()

# =========================
# Figure 6: Normalized values by soil type (ij = Avelog_SS / Avelog_f)
# =========================
# ترتیب soil types بر اساس میانه ij برای خوانایی
soil_order = (
    plot_df.groupby('TYPE_2')['ij']
    .median()
    .sort_values(ascending=False)
    .index
    .tolist()
)

plt.figure(figsize=(14, 6))
sns.boxplot(
    data=plot_df,
    x='TYPE_2',
    y='ij',
    order=soil_order,
    showfliers=False
)
sns.stripplot(
    data=plot_df,
    x='TYPE_2',
    y='ij',
    order=soil_order,
    size=2, alpha=0.3, color='k'
)
plt.axhline(1.0, ls='--', lw=1, color='gray')  # خط مرجع (میانگین میدانی)
plt.xlabel("Soil type (TYPE_2)")
plt.ylabel("Normalized SOM (Avelog_SS / Avelog_f)")
plt.title("Normalized SOM by soil type")
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.savefig("figs/figure6_boxplot_normalized_by_soiltype.png", dpi=600)
plt.close()

# =========================
# (اختیاری) جدول خلاصه آماری برای مقاله (Table X)
# =========================
summary_raw = data['Om_p'].describe()[['min','max','mean','50%','std']]
summary_log = data['log_Om'].describe()[['min','max','mean','50%','std']]

summary_table = pd.DataFrame({
    'Statistic': ['Minimum','Maximum','Mean','Median','Std. Dev.'],
    'Raw SOM (%)': [
        summary_raw['min'],
        summary_raw['max'],
        summary_raw['mean'],
        summary_raw['50%'],
        summary_raw['std']
    ],
    'Log(SOM)': [
        summary_log['min'],
        summary_log['max'],
        summary_log['mean'],
        summary_log['50%'],
        summary_log['std']
    ]
})

# ذخیره جدول برای استفاده در مقاله
summary_table.to_csv("figs/TableX_SOM_summary_stats.csv", index=False)
print("Saved figures to figs/ and TableX_SOM_summary_stats.csv")

unique_Field_number_count = data['Field_number'].unique()
unique_Field_number_count

"""**Area Testing**"""

df_area = '/content/Area.xlsx'
area = pd.read_excel(df_area, sheet_name='1')

total_area_per_land = area.groupby('FIELD')['Shape_Area'].sum()
total_area = area['Shape_Area'].sum()
total_area

grouped_data = area.groupby(['FIELD', 'TYPE_2'])['Shape_Area'].sum().reset_index()

grouped_data

# Merge with the total area per land
grouped_data = pd.merge(grouped_data, total_area_per_land, left_on='FIELD', right_index=True, suffixes=('_soil', '_total'))
grouped_data.rename(columns={'FIELD': 'Field_number'}, inplace=True)
grouped_data['Field_number'] = grouped_data['Field_number'].astype(str)
grouped_data['Field_number'] = grouped_data['Field_number'].apply(lambda x: 'F' + x)

grouped_data

# Calculate the percentage of each soil type area in the total area for each land
grouped_data['Percentage_area'] = (grouped_data['Shape_Area_soil'] / grouped_data['Shape_Area_total'])
grouped_data

'''import pandas as pd

# تابع برای اسکیل کردن درصد مساحت‌ها
def scale_percentages(group):
    total_percentage = group['Percentage_area'].sum()
    if total_percentage != 1:
        group['Percentage_area'] = group['Percentage_area'] / total_percentage
    return group

# خواندن فایل اکسل
df_area = '/content/Area.xlsx'
area = pd.read_excel(df_area, sheet_name='1')

# محاسبه مساحت کل هر زمین
total_area_per_land = area.groupby('FIELD')['Shape_Area'].sum()

# گروه‌بندی داده‌ها بر اساس FIELD و TYPE_2 و محاسبه مجموع مساحت‌ها
grouped_data = area.groupby(['FIELD', 'TYPE_2'])['Shape_Area'].sum().reset_index()

# ادغام با مساحت کل هر زمین
grouped_data = pd.merge(grouped_data, total_area_per_land, left_on='FIELD', right_index=True, suffixes=('_soil', '_total'))

# تغییر نام ستون FIELD به Field_number
grouped_data.rename(columns={'FIELD': 'Field_number'}, inplace=True)

# تبدیل شماره‌های زمین به رشته و اضافه کردن پیشوند F
grouped_data['Field_number'] = grouped_data['Field_number'].astype(str)
grouped_data['Field_number'] = grouped_data['Field_number'].apply(lambda x: 'F' + x)

# محاسبه درصد مساحت هر نوع خاک نسبت به مساحت کل زمین
grouped_data['Percentage_area'] = grouped_data['Shape_Area_soil'] / grouped_data['Shape_Area_total']

# اسکیل کردن درصد مساحت‌ها برای هر زمین
grouped_data = grouped_data.groupby('Field_number').apply(scale_percentages).reset_index(drop=True)

print(grouped_data)'''

'''from google.colab import drive
drive.mount("/content/gdrive")
data.to_excel(excel_writer=r"/content/gdrive/MyDrive/ij.xlsx")'''

'''# Merge the DataFrames based on 'شماره زمین' and 'نوع خاک'
merged_df = pd.merge(data, grouped_data, on=['Field_number', 'TYPE_2'], how='inner')

print(merged_df)'''

import pandas as pd

# ادغام دو دیتافریم بر اساس ستون‌های مشترک
merged_data = pd.merge(data, grouped_data, on=['Field_number', 'TYPE_2'], how='left')

# استخراج مقادیر ستون Percentage_area
percentage_area_values = merged_data['Percentage_area']

# اضافه کردن مقادیر ستون Percentage_area به دیتافریم data
merged_data['Area%'] = percentage_area_values

# نمایش دیتافریم حاوی ستون جدید
print(merged_data)

'''# Merge the DataFrames based on 'شماره زمین' and 'نوع خاک'
merged_df = pd.merge(data, grouped_data, on=['Field_number', 'TYPE_2'], how='inner')

print(merged_df)'''

'''from google.colab import drive
drive.mount("/content/gdrive")
merged_data.to_excel(excel_writer=r"/content/gdrive/MyDrive/merged_data.xlsx")'''

'''mean_log_Om = merged_data.groupby(['Field_number', 'TYPE_2'])['log_Om'].transform('mean')

# اضافه کردن میانگین ماده آلی خاک به دیتافریم در یک ستون جدید
merged_data['mean_log_Om'] = mean_log_Om
merged_data'''

import pandas as pd

# ادغام دو دیتافریم بر اساس ستون‌های مشترک
merged_data = pd.merge(data, grouped_data, on=['Field_number', 'TYPE_2'], how='left')

# استخراج مقادیر ستون Percentage_area
percentage_area_values = merged_data['Percentage_area']

# اضافه کردن مقادیر ستون Percentage_area به دیتافریم data
merged_data['Area%'] = percentage_area_values

# نمایش دیتافریم حاوی ستون جدید
print(merged_data)
merged_data.columns

"""**در اینجا دیتا فریمی که می خوام روش کار کنم رو میسازم**"""

import pandas as pd

# فرض کنید df دیتافریم شما باشد

# ایجاد یک دیکشنری برای نگهداری اطلاعات مربوط به خاک‌های یونیک در هر زمین
unique_soils_dict = {}

# بررسی هر زمین برای جدا کردن خاک‌های یونیک
for field_number in merged_data['Field_number'].unique():
    temp_df = merged_data[merged_data['Field_number'] == field_number]
    unique_soils = temp_df['TYPE_2'].unique()
    unique_soils_dict[field_number] = unique_soils

# ایجاد یک لیست خالی برای ساخت دیتافریم جدید
new_rows = []

# ایجاد ردیف‌های جدید برای دیتافریم جدید با تعداد ردیف‌های متناسب با تعداد زمین‌ها
for field_number, unique_soils in unique_soils_dict.items():
    for soil in unique_soils:
        temp_df = merged_data[(merged_data['Field_number'] == field_number) & (merged_data['TYPE_2'] == soil)]
        new_row = {
            'Field_number': field_number,
            'ij': temp_df['ij'].iloc[0],  # مقدار ij مربوط به اولین ردیف
            'Area%': temp_df['Area%'].iloc[0],  # مقدار Area% مربوط به اولین ردیف
            'TYPE_2': soil
        }
        new_rows.append(new_row)

# ایجاد دیتافریم جدید با ردیف‌های ساخته شده
new_df = pd.DataFrame(new_rows)

# چاپ دیتافریم جدید
print(new_df)

"""### **With Validation**"""

import pandas as pd

def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land


# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نمایش نتایج
print("Soil Differences Per Land:")
print(soil_diffs_per_land)
num_lands = len(soil_diffs_per_land)
print("Number of lands:", num_lands)
for land, diffs in soil_diffs_per_land.items():
    num_diffs = len(diffs)
    print(f"Land '{land}' has {num_diffs} differences.")

soil_diffs_per_land

'''def find_differences(new_soils, soil_diffs_per_land):
    # ایجاد وکتور خالی
    difference_vector = []

    # بررسی هر زمین در دیکشنری
    for diffs in soil_diffs_per_land.values():
        # اختلاف‌های مربوط به نوع‌های خاک مشابه با زمین جدید
        common_diffs = []

        # بررسی هر اختلاف برای نوع‌های خاک مشابه
        for diff_key, diff_value in diffs.items():
            # اگر هر یک از نوع‌های خاک مشابه با زمین جدید باشد
            if all(soil in new_soils for soil in diff_key.split('-')):
                common_diffs.append(diff_value)

        # اضافه کردن اختلاف‌ها به وکتور
        difference_vector.extend(common_diffs)

    return difference_vector


# زمین جدید
#new_soils = ["shallow light sandy loam", "loamy sand","light sandy loam","clay loam"] #F12
#new_soils = ["clay loam", "fine sandy loam","silt loam","light sandy loam","loamy sand","sand"] #F110
#new_soils = ["loam", "clay loam"] #F14
new_soils = ["loam", "light sandy loam"] #F18

# جستجوی اختلاف‌ها و پر کردن وکتور
difference_vector = find_differences(new_soils, soil_diffs_per_land)


vertical_vector = np.array(difference_vector).reshape(-1, 1)

# نمایش وکتور
print("Difference Vector:")
print(vertical_vector)
'''

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        common_diffs = []
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                common_diffs.append(diff_value)
        difference_vector.extend(common_diffs)
    return difference_vector

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه وکتورهای اختلاف
all_difference_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    vertical_vector = np.array(difference_vector).reshape(-1, 1)
    all_difference_vectors[land] = vertical_vector

# نمایش وکتورهای اختلاف
for land, vector in all_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        common_diffs = []
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                common_diffs.append(diff_value)
        difference_vector.extend(common_diffs)
    return difference_vector

# تابع اضافه کردن 1 به اولین ردیف وکتور
def add_one_to_first_row(difference_vector):
    return np.insert(difference_vector, 0, 1).reshape(-1, 1)

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه وکتورهای اختلاف
all_difference_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    # اضافه کردن عدد 1 به اولین ردیف
    vertical_vector = add_one_to_first_row(difference_vector)
    all_difference_vectors[land] = vertical_vector

# نمایش وکتورهای اختلاف
for land, vector in all_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
    return difference_vector

# تابع پر کردن ماتریس اختلاف
def fill_difference_matrix(difference_vector, new_soils):
    num_rows = len(difference_vector)
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    for i, (soil1, soil2, diff_value) in enumerate(difference_vector):
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه ماتریس‌های اختلاف
all_difference_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix = fill_difference_matrix(difference_vector, new_soils)
    all_difference_matrices[land] = difference_matrix

# نمایش ماتریس‌های اختلاف
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)

'''def find_differences(new_soils, soil_diffs_per_land):
    difference_vector = []

    # بررسی هر زمین در دیکشنری
    for land, diffs in soil_diffs_per_land.items():
        # بررسی هر اختلاف برای نوع‌های خاک مشابه
        for diff_key, diff_value in diffs.items():
            # اگر هر یک از نوع‌های خاک مشابه با زمین جدید باشد
            if all(soil in new_soils for soil in diff_key.split('-')):
                # ذخیره نام زمین، نوع‌های خاک و تفاوت متناظر
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((land, soil1, soil2, diff_value))

    return difference_vector

# زمین جدید

#new_soils = ["fine sandy loam", "clay loam","loam"]
#new_soils = ["shallow light sandy loam", "loamy sand","light sandy loam","clay loam"]  #F12
# new_soils = ["clay loam", "fine sandy loam","silt loam","light sandy loam","loamy sand","sand"] #F10
#new_soils = ["loam", "clay loam"] #F14
new_soils = ["loam", "light sandy loam"] #F18

# جستجوی اختلاف‌ها و پر کردن وکتور
difference_vector = find_differences(new_soils, soil_diffs_per_land)

# نمایش وکتور
print("Difference Vector:")
for land, soil1, soil2, diff_value in difference_vector:
    print(f"Land: {land}, Soils: {soil1} and {soil2}, Difference: {diff_value}")
'''

'''import numpy as np

def fill_difference_matrix(difference_vector, new_soils):
    num_rows = len(difference_vector)
    num_cols = len(new_soils)

    # ساختن ماتریس خالی
    difference_matrix = np.zeros((num_rows, num_cols))

    # پرکردن سرریز با اطلاعات موجود در وکتور
    for i, (land, soil1, soil2, diff_value) in enumerate(difference_vector):
        # یافتن اندیس هر خاک در زمین جدید
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        # پرکردن سرریز با مقادیر متناظر
        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# مثال: خاک‌های زمین جدید

#new_soils = ["fine sandy loam", "clay loam","loam"]
#new_soils = ["shallow light sandy loam", "loamy sand","light sandy loam","clay loam"] #F12
#new_soils = ["clay loam", "fine sandy loam","silt loam","light sandy loam","loamy sand","sand"] #F10
#new_soils = ["loam", "clay loam"] #F14
new_soils = ["loam", "light sandy loam"] #F18

# پرکردن ماتریس با استفاده از تابع
difference_matrix = fill_difference_matrix(difference_vector, new_soils)

# نمایش ماتریس
print("Difference Matrix:")
print(difference_matrix)

'''

import pandas as pd
import numpy as np

# Function to calculate soil differences for each land
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# Function to find differences
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
    return difference_vector

# Function to fill the difference matrix
def fill_difference_matrix(difference_vector, new_soils, new_soil_areas):
    num_rows = len(difference_vector) + 1
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    for i, (soil, area) in enumerate(new_soil_areas.items()):
        soil_index = new_soils.index(soil)
        difference_matrix[0, soil_index] = area

    for i, (soil1, soil2, diff_value) in enumerate(difference_vector, start=1):
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# Suppose new_df is your main dataframe
# Calculate soil differences for each land
soil_diffs_per_land = calculate_soil_diffs(new_df)

# Store all difference matrices
all_difference_matrices = {}

# Create and store difference matrices for each land
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    new_soil_areas = dict(zip(new_df[new_df['Field_number'] == land]['TYPE_2'],
                              new_df[new_df['Field_number'] == land]['Area%']))
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix = fill_difference_matrix(difference_vector, new_soils, new_soil_areas)
    all_difference_matrices[land] = difference_matrix

# Display difference matrices
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
    return difference_vector

# تابع پر کردن ماتریس اختلاف
def fill_difference_matrix(difference_vector, new_soils, new_soil_areas):
    num_rows = len(difference_vector) + 1  # اضافه کردن یک ردیف برای مساحت خاک‌ها
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    # یافتن میانگین مساحت‌ها
    valid_areas = [area for area in new_soil_areas.values() if not np.isnan(area)]
    mean_area = np.mean(valid_areas) if valid_areas else 0

    # پر کردن ردیف اول با مساحت خاک‌ها
    for i, soil in enumerate(new_soils):
        soil_index = new_soils.index(soil)
        area = new_soil_areas.get(soil, np.nan)
        if np.isnan(area):
            difference_matrix[0, soil_index] = mean_area
        else:
            difference_matrix[0, soil_index] = area

    # پر کردن سرریز با اطلاعات موجود در وکتور
    for i, (soil1, soil2, diff_value) in enumerate(difference_vector, start=1):
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه ماتریس‌های اختلاف
all_difference_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    new_soil_areas = dict(zip(new_df[new_df['Field_number'] == land]['TYPE_2'],
                              new_df[new_df['Field_number'] == land]['Area%']))
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix = fill_difference_matrix(difference_vector, new_soils, new_soil_areas)
    all_difference_matrices[land] = difference_matrix

# نمایش ماتریس‌های اختلاف
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
    return difference_vector

# تابع پر کردن ماتریس اختلاف
def fill_difference_matrix(difference_vector, new_soils, new_soil_areas):
    num_rows = len(difference_vector) + 1  # اضافه کردن یک ردیف برای مساحت خاک‌ها
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    # یافتن میانگین مساحت‌ها و جایگزینی نال‌ها
    valid_areas = [area for area in new_soil_areas.values() if not np.isnan(area)]
    mean_area = np.mean(valid_areas) if valid_areas else 0

    for soil in new_soil_areas:
        if np.isnan(new_soil_areas[soil]):
            new_soil_areas[soil] = mean_area

    # محاسبه مجموع مساحت‌ها
    total_area = sum(new_soil_areas.values())

    # اسکیل کردن مساحت‌ها اگر مجموع آنها برابر با 1 نباشد
    if total_area != 1.0:
        scale_factor = 1.0 / total_area
        for soil in new_soil_areas:
            new_soil_areas[soil] *= scale_factor

    # پر کردن ردیف اول با مساحت خاک‌ها
    for i, soil in enumerate(new_soils):
        soil_index = new_soils.index(soil)
        area = new_soil_areas.get(soil, 0)
        difference_matrix[0, soil_index] = area

    # پر کردن سرریز با اطلاعات موجود در وکتور
    for i, (soil1, soil2, diff_value) in enumerate(difference_vector, start=1):
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه ماتریس‌های اختلاف
all_difference_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    new_soil_areas = dict(zip(new_df[new_df['Field_number'] == land]['TYPE_2'],
                              new_df[new_df['Field_number'] == land]['Area%']))
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix = fill_difference_matrix(difference_vector, new_soils, new_soil_areas)
    all_difference_matrices[land] = difference_matrix

# نمایش ماتریس‌های اختلاف
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)

'''import numpy as np

def fill_difference_matrix(difference_vector, new_soils, new_soil_areas):
    num_rows = len(difference_vector) + 1  # اضافه کردن یک ردیف برای مساحت خاک‌ها
    num_cols = len(new_soils)

    # ساختن ماتریس خالی
    difference_matrix = np.zeros((num_rows, num_cols))

    # پرکردن ردیف اول با مساحت خاک‌ها
    for i, (soil, area) in enumerate(new_soil_areas.items()):
        soil_index = new_soils.index(soil)
        difference_matrix[0, soil_index] = area

    # پرکردن سرریز با اطلاعات موجود در وکتور
    for i, (land, soil1, soil2, diff_value) in enumerate(difference_vector, start=1):
        # یافتن اندیس هر خاک در زمین جدید
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        # پرکردن سرریز با مقادیر متناظر
        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# مثال: خاک‌های زمین جدید و مساحت آنها
#new_soils = ["shallow over mineral deposit", "silt loam", "silt"]
#new_soils = ["fine sandy loam", "clay loam","loam"]
#new_soil_areas = {"fine sandy loam": 0.65, "clay loam": 0.2, "loam": 0.15}
new_soils = ["shallow light sandy loam", "loamy sand","light sandy loam","clay loam"] #F12
#new_soils = ["clay loam", "fine sandy loam","silt loam","light sandy loam","loamy sand","sand"] #F10
#new_soils = ["loam", "clay loam"] #F14
new_soils = ["loam", "light sandy loam"] #F18
new_soil_areas = {"loam": 0.3814,  "light sandy loam": 0.6186}

# پرکردن ماتریس با استفاده از تابع
difference_matrix = fill_difference_matrix(difference_vector, new_soils, new_soil_areas)

# نمایش ماتریس
print("Difference Matrix:")
print(difference_matrix)'''

'''import numpy as np

def find_differences(new_soils, soil_diffs_per_land):
    # ایجاد وکتور خالی
    difference_vector = []

    # بررسی هر زمین در دیکشنری
    for diffs in soil_diffs_per_land.values():
        # اختلاف‌های مربوط به نوع‌های خاک مشابه با زمین جدید
        common_diffs = []

        # بررسی هر اختلاف برای نوع‌های خاک مشابه
        for diff_key, diff_value in diffs.items():
            # اگر هر یک از نوع‌های خاک مشابه با زمین جدید باشد
            if all(soil in new_soils for soil in diff_key.split('-')):
                common_diffs.append(diff_value)

        # اضافه کردن اختلاف‌ها به وکتور
        difference_vector.extend(common_diffs)

    return difference_vector


# زمین جدید
#new_soils = ["shallow over mineral deposit", "silt loam", "silt"]
#new_soils = ["fine sandy loam", "clay loam","loam"]
#new_soils = ["shallow light sandy loam", "loamy sand","light sandy loam","clay loam"] #F12
#new_soils = ["clay loam", "fine sandy loam","silt loam","light sandy loam","loamy sand","sand"] #F10
#new_soils = ["loam", "clay loam"] #F14
new_soils = ["loam", "light sandy loam"] #F18

# جستجوی اختلاف‌ها و پر کردن وکتور
difference_vector = find_differences(new_soils, soil_diffs_per_land)

# اضافه کردن 1 به عنوان اولین عنصر وکتور
difference_vector.insert(0, 1)

# تبدیل وکتور به آرایه numpy و ساختن وکتور عمودی
vertical_vector = np.array(difference_vector).reshape(-1, 1)

# نمایش وکتور
print("Difference Vector:")
print(vertical_vector)
'''

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
    return difference_vector

# تابع ایجاد وکتور عمودی از انواع خاک‌ها
def create_soil_vector(new_soils):
    soil_array = np.array(new_soils)
    soil_vector = soil_array.reshape(-1, 1)
    return soil_vector

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری لیستی از وکتورهای خاک
all_soil_vectors = []

# ایجاد و ذخیره وکتورهای خاک برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    soil_vector = create_soil_vector(new_soils)
    all_soil_vectors.append(soil_vector)

# نمایش وکتورهای خاک
for land, vector in zip(new_df['Field_number'].unique(), all_soil_vectors):
    print(f"Soil Vector for Land {land}:")
    print(vector)

'''import numpy as np

def create_soil_vector(new_soils):
    # تبدیل لیست خاک‌های جدید به یک آرایه NumPy
    soil_array = np.array(new_soils)
    # تغییر شکل آرایه به وکتور عمودی
    soil_vector = soil_array.reshape(-1, 1)
    return soil_vector

# مثال: خاک‌های زمین جدید
#new_soils = ["shallow over mineral deposit", "silt loam", "silt"]
#new_soils = ["fine sandy loam", "clay loam","loam"]
#new_soils = ["shallow light sandy loam", "loamy sand","light sandy loam","clay loam"] #F12
#new_soils = ["clay loam", "fine sandy loam","silt loam","light sandy loam","loamy sand","sand"] #F10
#new_soils = ["loam", "clay loam"] #F14
new_soils = ["loam", "light sandy loam"] #F18

# ایجاد وکتور عمودی از خاک‌های زمین جدید
soil_vector = create_soil_vector(new_soils)

# نمایش وکتور
print("Soil Vector:")
print(soil_vector)
'''

# نگهداری همه میزان backslash برای هر زمین
all_solutions = {}

# محاسبه میزان backslash برای هر زمین
for land, matrix in all_difference_matrices.items():
    # وکتور اختلاف مربوط به هر زمین
    d = all_difference_vectors[land]

    # حل معادله خطی s = m \ d
    solution = np.linalg.lstsq(matrix, d, rcond=None)[0]

    # ذخیره میزان backslash
    all_solutions[land] = solution

# نمایش میزان backslash برای هر زمین
for land, solution in all_solutions.items():
    print(f"Backslash for Land {land}:")
    print(solution)

'''# وکتور D
D = vertical_vector

# ماتریس M
M = difference_matrix

# وکتور متغیرها S
S = soil_vector

# حل معادله خطی
solution = np.linalg.lstsq(M, D, rcond=None)[0]

# نمایش وکتور متغیرها
print("Solution:")
print(solution)'''

# نگهداری همه میزان backslash برای هر زمین
all_solutions = {}

# محاسبه میزان backslash برای هر زمین
for land, matrix in all_difference_matrices.items():
    # وکتور اختلاف مربوط به هر زمین
    d = all_difference_vectors[land]

    # چک کردن شرط تنها یک عضو بودن وکتور d و عضویت آن عضو برابر با یک
    if len(d) == 1 and d[0] == 1:
        # محاسبه میانگین عدد موجود در ستون 'ij' برای زمین مورد نظر
        mean_ij = merged_data.loc[merged_data['Field_number'] == land, 'ij'].mean()

        # قرار دادن میانگین محاسبه شده به جای backslash برای این زمین
        all_solutions[land] = mean_ij
    else:
        # حل معادله خطی s = m \ d
        solution = np.linalg.lstsq(matrix, d, rcond=None)[0]
        all_solutions[land] = solution

# نمایش میزان backslash برای هر زمین
for land, solution in all_solutions.items():
    print(f"Backslash for Land {land}:")
    print(solution)

# نگهداری همه میزان backslash برای هر زمین
all_solutions_with_Avelog_f = {}

# محاسبه میزان backslash برای هر زمین
for land, matrix in all_difference_matrices.items():
    # وکتور اختلاف مربوط به هر زمین
    d = all_difference_vectors[land]

    # حل معادله خطی s = m \ d
    solution = np.linalg.lstsq(matrix, d, rcond=None)[0]

    # Find the value of Avelog_f for the specific land
    avelog_f_value = merged_data.loc[merged_data['Field_number'] == land, 'Avelog_f'].values[0]

    # Multiply each element of the solution vector by the Avelog_f value
    solution_with_Avelog_f = [value * avelog_f_value for value in solution]
    #print(solution_with_Avelog_f)

    # Store the solution vector with Avelog_f value for the land
    all_solutions_with_Avelog_f[land] = solution_with_Avelog_f

# نمایش میزان backslash با در نظر گرفتن Avelog_f برای هر زمین
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    print(f"Backslash with Avelog_f for Land {land}:")
    print(solution_with_Avelog_f)

'''# Specify the name of the specific land (for example, "Test_Land") you're interested in
land_name = "F18"

# Find the value of Avelog_f for the specific land
avelog_f_value = merged_data.loc[merged_data['Field_number'] == land_name, 'Avelog_f'].values[0]
print(avelog_f_value)

# Multiply each element of the solution vector by the Avelog_f value
solution = [value * avelog_f_value for value in solution]

print(solution)'''

import pandas as pd

# لیستی از دیکشنری‌ها که هر دیکشنری اطلاعات یک ردیف از دیتافریم جدید را نشان می‌دهد
data = []

# بررسی هر زمین و مقادیر مربوط به آن
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    # بررسی هر نوع خاک و مقدار مربوط به آن
    for soil_type, value in zip(new_df[new_df['Field_number'] == land]['TYPE_2'].unique(), solution_with_Avelog_f):
        # ایجاد دیکشنری برای اطلاعات یک ردیف
        row = {'Field_number': land, 'TYPE_2': soil_type, 'Calculated_value': value[0]}
        # افزودن دیکشنری به لیست دیکشنری‌ها
        data.append(row)

# ساخت دیتافریم با استفاده از لیست دیکشنری‌ها
result_df = pd.DataFrame(data)

# نمایش اولین چند ردیف از دیتافریم جدید برای اطمینان از صحت عملیات
print(result_df.head())

import pandas as pd

# لیستی از دیکشنری‌ها که هر دیکشنری اطلاعات یک ردیف از دیتافریم جدید را نشان می‌دهد
data = []

# بررسی هر زمین و مقادیر مربوط به آن
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    # بررسی هر نوع خاک و مقدار مربوط به آن
    for soil_type, value in zip(new_df[new_df['Field_number'] == land]['TYPE_2'].unique(), solution_with_Avelog_f):
        # ایجاد دیکشنری برای اطلاعات یک ردیف
        row = {'Field_number': land, 'TYPE_2': soil_type, 'Calculated_value': value[0]}
        # افزودن دیکشنری به لیست دیکشنری‌ها
        data.append(row)

# ساخت دیتافریم با استفاده از لیست دیکشنری‌ها
result_df = pd.DataFrame(data)

# ساخت ستون جدید predicted در دیتافریم merged_data و پر کردن آن با مقادیر محاسبه شده
merged_data['predicted'] = merged_data.apply(lambda row: result_df[(result_df['Field_number'] == row['Field_number']) & (result_df['TYPE_2'] == row['TYPE_2'])]['Calculated_value'].values[0] if not result_df[(result_df['Field_number'] == row['Field_number']) & (result_df['TYPE_2'] == row['TYPE_2'])].empty else None, axis=1)

# نمایش چند ردیف از دیتافریم merged_data برای اطمینان از صحت عملیات
print(merged_data)

import numpy as np

# محاسبه RMSE بین دو ستون
def calculate_rmse(row):
    predicted_value = row['predicted']
    actual_value = row['log_Om']
    # بررسی برای مقادیر وجود دارد
    if pd.isnull(predicted_value) or pd.isnull(actual_value):
        return None
    # محاسبه مربعات اختلاف
    squared_diff = (predicted_value - actual_value) ** 2
    return np.sqrt(squared_diff)

# محاسبه RMSE برای هر ردیف دیتافریم
merged_data['RMSE'] = merged_data.apply(calculate_rmse, axis=1)

# نمایش چند ردیف از دیتافریم merged_data برای اطمینان از صحت عملیات
print(merged_data.head())

'''from google.colab import drive
drive.mount("/content/gdrive")
merged_data.to_excel(excel_writer=r"/content/gdrive/MyDrive/RMSE_Calc.xlsx")'''

import pandas as pd
import numpy as np

# فرض بر این است که دیتافریم شما به نام df است و ستون‌های predicted و log_Om را دارد
# محاسبه RMSE
rmse = np.sqrt(np.mean((merged_data['predicted'] - merged_data['log_Om']) ** 2))

print("RMSE Total:", rmse)

# محاسبه RMSE برای هر زمین
grouped = merged_data.groupby('Field_number').apply(lambda x: np.sqrt(np.mean((x['predicted'] - x['log_Om']) ** 2)))
grouped = grouped.reset_index(name='RMSE')

# نمایش نمودار
plt.figure(figsize=(10, 6))
plt.bar(grouped['Field_number'], grouped['RMSE'], color='skyblue')
plt.xlabel('Field Number')
plt.ylabel('RMSE')
plt.title('RMSE by Field Number')
plt.xticks(rotation=90)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# محاسبه RMSE برای هر زمین
grouped = merged_data.groupby('Field_number').apply(lambda x: np.sqrt(np.mean((x['predicted'] - x['log_Om']) ** 2)))
grouped = grouped.reset_index(name='RMSE')

# شناسایی زمین با RMSE بالا (فرض بر اینکه Field2 دارای RMSE بالاست)
field_with_high_rmse = grouped.loc[grouped['RMSE'].idxmax(), 'Field_number']
print("Field with highest RMSE:", field_with_high_rmse)

# استخراج داده‌های مربوط به این زمین
field_data = merged_data[merged_data['Field_number'] == field_with_high_rmse]

# نمایش داده‌ها برای تحلیل
print(field_data)

# نمایش نمودار تفاوت پیش‌بینی و مقدار واقعی برای این زمین
plt.figure(figsize=(10, 6))
plt.plot(field_data['predicted'], label='Predicted', marker='o')
plt.plot(field_data['log_Om'], label='Actual', marker='x')
plt.title(f'Differences between Predicted and Actual for {field_with_high_rmse}')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

import pandas as pd
import numpy as np


# محاسبه RMSE برای هر زمین
grouped = merged_data.groupby('Field_number').apply(lambda x: np.sqrt(np.mean((x['predicted'] - x['log_Om']) ** 2)))
grouped = grouped.reset_index(name='RMSE')

# شناسایی زمین با بدترین RMSE
worst_field = grouped.loc[grouped['RMSE'].idxmax(), 'Field_number']
print("Field with worst RMSE:", worst_field)

# حذف داده‌های مربوط به زمین با بدترین RMSE
df_filtered = merged_data[merged_data['Field_number'] != worst_field]

# محاسبه RMSE کلی بدون در نظر گرفتن زمین با بدترین RMSE
rmse_filtered = np.sqrt(np.mean((df_filtered['predicted'] - df_filtered['log_Om']) ** 2))

print("RMSE Worst field RMSE:", rmse_filtered)

# محاسبه RMSE برای هر زمین
grouped = merged_data.groupby('Field_number').apply(lambda x: np.sqrt(np.mean((x['predicted'] - x['log_Om']) ** 2)))
grouped = grouped.reset_index(name='RMSE')

# شناسایی دو زمین با بالاترین RMSE
worst_fields = grouped.nlargest(2, 'RMSE')['Field_number'].values
print("Fields with highest RMSE:", worst_fields)

# حذف داده‌های مربوط به این دو زمین از دیتافریم اصلی
df_filtered = merged_data[~merged_data['Field_number'].isin(worst_fields)]

# محاسبه مجدد RMSE کلی بدون در نظر گرفتن دو زمین با بالاترین RMSE
rmse_filtered = np.sqrt(np.mean((df_filtered['predicted'] - df_filtered['log_Om']) ** 2))

print("RMSE the two worst RMSE:", rmse_filtered)

from sklearn.metrics import mean_squared_error
from math import sqrt

# نگهداری مقادیر RMSE برای هر زمین
all_rmse_values = []

# محاسبه RMSE برای هر زمین
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    # مقادیر Avelog_SS برای هر نوع خاک در زمین مورد نظر
    avelog_ss_values = []
    for soil in new_df[new_df['Field_number'] == land]['TYPE_2'].unique():
        avelog_ss_values.append(merged_data.loc[(merged_data['Field_number'] == land) &
                                                (merged_data['TYPE_2'] == soil), 'Avelog_SS'].values[0])

    # محاسبه RMSE بین مقادیر محاسبه شده و مقادیر واقعی Avelog_SS
    rmse = sqrt(mean_squared_error(avelog_ss_values, solution_with_Avelog_f))

    # ذخیره مقدار RMSE برای زمین فعلی
    all_rmse_values.append(rmse)

    # چاپ مقدار RMSE برای هر زمین
    print(f"RMSE for Land {land}: {rmse}")

# محاسبه و چاپ میانگین RMSE برای همه زمین‌ها
mean_rmse = sum(all_rmse_values) / len(all_rmse_values)
print(f"Mean RMSE for all lands: {mean_rmse}")

'''#RMSE CALCULATION
from sklearn.metrics import mean_squared_error

# Define the land name
land_name = 'F18'

# Filter data for the specified land
land_data = merged_data[merged_data['Field_number'] == land_name]

# Initialize lists to store true and predicted values for each soil type
true_values = []
predicted_values = []

# Iterate over each soil type in Soil_Vector
for soil_type in soil_vector:
    soil_type = soil_type[0]  # Convert from 2D to 1D array
    # Find the Avelog_SS values for the current soil type in the specified land
    avelog_ss_values = land_data.loc[land_data['TYPE_2'] == soil_type, 'Avelog_SS'].values
    if len(avelog_ss_values) > 0:
        # Append the mean value of Avelog_SS for the current soil type to true_values
        true_values.append(np.mean(avelog_ss_values))
        # Find the corresponding predicted value for the current soil type
        predicted_value = solution[np.where(soil_vector == soil_type)[0][0]][0]
        predicted_values.append(predicted_value)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(true_values, predicted_values))

print("True values:", true_values)
print("Predicted values:", predicted_values)
print("RMSE:", rmse)
Performance=10**rmse
print("Performance:", Performance)
'''

'''from google.colab import drive
drive.mount("/content/gdrive")
new_df.to_excel(excel_writer=r"/content/gdrive/MyDrive/test4.xlsx")'''

'''import pandas as pd
import numpy as np

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    land_info = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append(diff_value)
                land_info.append((land, soil1, soil2))
    return difference_vector, land_info

def add_one_to_first_row(difference_vector):
    return np.insert(difference_vector, 0, 1).reshape(-1, 1)

# نگهداری همه وکتورهای اختلاف و اطلاعات مربوطه
all_difference_vectors = {}
all_land_info = {}

# ایجاد و ذخیره وکتورهای اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    # اضافه کردن عدد 1 به اولین ردیف
    vertical_vector = add_one_to_first_row(difference_vector)
    all_difference_vectors[land] = vertical_vector
    all_land_info[land] = land_info

# نمایش وکتورهای اختلاف و اطلاعات مربوطه
for land, vector in all_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)
    print(f"Land Info for Land {land}:")
    print(all_land_info[land])'''

"""### **تست صحت عملکرد کد**"""

'''import pandas as pd
import numpy as np

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    land_info = []
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
                land_info.append((land, soil1, soil2))
    return difference_vector, land_info
def fill_difference_matrix(difference_vector, new_soils):
    num_rows = len(difference_vector)
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))
    land_info_matrix = []

    for i, (soil1, soil2, diff_value) in enumerate(difference_vector):
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

        land_info_matrix.append((soil1, soil2, diff_value))

    return difference_matrix, land_info_matrix
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه ماتریس‌های اختلاف و اطلاعات مربوطه
all_difference_matrices = {}
all_land_info_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix, land_info_matrix = fill_difference_matrix(difference_vector, new_soils)
    all_difference_matrices[land] = difference_matrix
    all_land_info_matrices[land] = land_info_matrix

# نمایش ماتریس‌های اختلاف و اطلاعات مربوطه
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)
    print(f"Land Info for Land {land}:")
    print(all_land_info_matrices[land])'''

import pandas as pd
import numpy as np

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه مربعات تفاوت‌ها و اضافه کردن به ستون جدید
merged_data['normal_Pred'] = (merged_data['log_Om'] - merged_data['Avelog_f'])**2

# محاسبه میانگین مربعات تفاوت‌ها
mean_squared_error = merged_data['normal_Pred'].mean()

# محاسبه جذر میانگین مربعات تفاوت‌ها
root_mean_squared_error = np.sqrt(mean_squared_error)

print("Root Mean Squared Error:", root_mean_squared_error)

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# تابع هدف با استفاده از فاصلهٔ مطلق
def objective_function(beta, column1, column2):
    return np.mean(np.abs(column1 - beta * column2) - np.abs(column1 - column2))

# تابع برای یافتن بهترین بتاها برای هر ردیف با استفاده از SGD
def find_optimal_betas(data):
    betas = []
    for i in range(len(merged_data['log_Om'])):
        result = minimize(objective_function, x0=0, args=(merged_data['log_Om'][i], merged_data['Avelog_f'][i]), method='L-BFGS-B')
        betas.append(result.x[0])
    return betas


# یافتن بهترین بتاها با استفاده از SGD
merged_data['optimal_betas'] = find_optimal_betas(merged_data)

# نمایش نتایج
for i, beta in enumerate(merged_data['optimal_betas']):
    print(f"Optimal Beta for Row {i+1}: {beta}")

mean_SGD_by_label_abs = merged_data.groupby('TYPE_2')['optimal_betas'].mean()
merged_data['BetaSGD_mean_by_label_abs'] = merged_data['TYPE_2'].map(mean_SGD_by_label_abs).round(9)
merged_data['Beta_Pred'] = merged_data['BetaSGD_mean_by_label_abs'] * merged_data['Avelog_f']

S_G_D2=((merged_data['log_Om'] - merged_data['Beta_Pred'])**2).mean()
SGD2=np.sqrt(S_G_D2)
SGD2

"""### **Without Validation**"""

import pandas as pd

# فرض کنید df دیتافریم شما باشد

# ایجاد یک دیکشنری برای نگهداری اطلاعات مربوط به خاک‌های یونیک در هر زمین
unique_soils_dict = {}

# بررسی هر زمین برای جدا کردن خاک‌های یونیک
for field_number in merged_data['Field_number'].unique():
    temp_df = merged_data[merged_data['Field_number'] == field_number]
    unique_soils = temp_df['TYPE_2'].unique()
    unique_soils_dict[field_number] = unique_soils

# ایجاد یک لیست خالی برای ساخت دیتافریم جدید
new_rows = []

# ایجاد ردیف‌های جدید برای دیتافریم جدید با تعداد ردیف‌های متناسب با تعداد زمین‌ها
for field_number, unique_soils in unique_soils_dict.items():
    for soil in unique_soils:
        temp_df = merged_data[(merged_data['Field_number'] == field_number) & (merged_data['TYPE_2'] == soil)]
        new_row = {
            'Field_number': field_number,
            'ij': temp_df['ij'].iloc[0],  # مقدار ij مربوط به اولین ردیف
            'Area%': temp_df['Area%'].iloc[0],  # مقدار Area% مربوط به اولین ردیف
            'TYPE_2': soil
        }
        new_rows.append(new_row)

# ایجاد دیتافریم جدید با ردیف‌های ساخته شده
new_df = pd.DataFrame(new_rows)

# چاپ دیتافریم جدید
print(new_df)
import pandas as pd

def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land


# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نمایش نتایج
print("Soil Differences Per Land:")
print(soil_diffs_per_land)
num_lands = len(soil_diffs_per_land)
print("Number of lands:", num_lands)
for land, diffs in soil_diffs_per_land.items():
    num_diffs = len(diffs)
    print(f"Land '{land}' has {num_diffs} differences.")

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        common_diffs = []
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                common_diffs.append(diff_value)
            elif land == new_land and any(soil in diff_key.split('-') for soil in new_soils):
                common_diffs.append(diff_value)
        difference_vector.extend(common_diffs)
    return difference_vector

# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری وکتورهای اختلاف برای هر زمین با حفظ ترتیب
ordered_difference_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف برای هر زمین با حفظ ترتیب
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    vertical_vector = np.array(difference_vector).reshape(-1, 1)
    ordered_difference_vectors[land] = vertical_vector

# چاپ وکتورهای اختلاف به ترتیب زمین‌ها
for land, vector in ordered_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)

import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        common_diffs = []
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                common_diffs.append(diff_value)
            elif land == new_land and any(soil in diff_key.split('-') for soil in new_soils):
                common_diffs.append(diff_value)
        difference_vector.extend(common_diffs)
    return difference_vector

# تابع اضافه کردن 1 به اولین ردیف وکتور
def add_one_to_first_row(difference_vector):
    return np.insert(difference_vector, 0, 1).reshape(-1, 1)

# فرض کنید new_df دیتا فریم اصلی شما باشد
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری وکتورهای اختلاف برای هر زمین با حفظ ترتیب
ordered_difference_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف برای هر زمین با حفظ ترتیب
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    # اضافه کردن عدد 1 به اولین ردیف
    vertical_vector = add_one_to_first_row(difference_vector)
    ordered_difference_vectors[land] = vertical_vector

# چاپ وکتورهای اختلاف به ترتیب زمین‌ها
for land, vector in ordered_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)
import pandas as pd
import numpy as np

# تابع محاسبه اختلاف‌های خاک برای هر زمین
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

# تابع یافتن اختلافات
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    for land, diffs in soil_diffs_per_land.items():
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append((soil1, soil2, diff_value))
    return difference_vector

# تابع پر کردن ماتریس اختلاف
def fill_difference_matrix(difference_vector, new_soils):
    num_rows = len(difference_vector)
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    for i, (soil1, soil2, diff_value) in enumerate(difference_vector):
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix

# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه ماتریس‌های اختلاف
all_difference_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix = fill_difference_matrix(difference_vector, new_soils)
    all_difference_matrices[land] = difference_matrix

    # اضافه کردن اختلاف‌های خاک مشترک با زمین جدید
    for other_land, other_matrix in all_difference_matrices.items():
        if other_land == land:
            continue
        other_soils = list(new_df[new_df['Field_number'] == other_land]['TYPE_2'].unique())
        common_soils = list(set(new_soils) & set(other_soils))
        common_diff_vector = find_differences(common_soils, soil_diffs_per_land, land)
        common_diff_matrix = fill_difference_matrix(common_diff_vector, new_soils)
        all_difference_matrices[land] = np.vstack([difference_matrix, common_diff_matrix])

# نمایش ماتریس‌های اختلاف
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)
import pandas as pd
import numpy as np

# فرض کنید new_df دیتا فریم اصلی شما باشد
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    land_info = []

    # مقایسه خاک‌های داخل زمین جدید
    for i in range(len(new_soils)):
        for j in range(i + 1, len(new_soils)):
            soil1, soil2 = new_soils[i], new_soils[j]
            diff = new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil1)]['ij'].iloc[0] - new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil2)]['ij'].iloc[0]
            difference_vector.append(diff)
            land_info.append((new_land, soil1, soil2))

    # مقایسه خاک‌های زمین جدید با سایر زمین‌ها
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append(diff_value)
                land_info.append((land, soil1, soil2))

    return difference_vector, land_info
def fill_difference_matrix(difference_vector, new_soils):
    num_rows = len(difference_vector)
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))
    land_info_matrix = []

    for i, diff_value in enumerate(difference_vector):
        soil1, soil2 = land_info[i][1], land_info[i][2]
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

        land_info_matrix.append((soil1, soil2, diff_value))

    return difference_matrix, land_info_matrix
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه وکتورهای اختلاف و اطلاعات مربوطه
all_difference_vectors = {}
all_land_info_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    vertical_vector = np.array(difference_vector).reshape(-1, 1)
    all_difference_vectors[land] = vertical_vector
    all_land_info_vectors[land] = land_info

# نمایش وکتورهای اختلاف و اطلاعات مربوطه
for land, vector in all_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)
    print(f"Land Info for Land {land}:")
    print(all_land_info_vectors[land])

# نگهداری همه ماتریس‌های اختلاف و اطلاعات مربوطه
all_difference_matrices = {}
all_land_info_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix, land_info_matrix = fill_difference_matrix(difference_vector, new_soils)
    all_difference_matrices[land] = difference_matrix
    all_land_info_matrices[land] = land_info_matrix

# نمایش ماتریس‌های اختلاف و اطلاعات مربوطه
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)
    print(f"Land Info for Land {land}:")
    print(all_land_info_matrices[land])
import pandas as pd
import numpy as np

# فرض کنید new_df دیتا فریم اصلی شما باشد
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    land_info = []

    # مقایسه خاک‌های داخل زمین جدید
    for i in range(len(new_soils)):
        for j in range(i + 1, len(new_soils)):
            soil1, soil2 = new_soils[i], new_soils[j]
            diff = new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil1)]['ij'].iloc[0] - new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil2)]['ij'].iloc[0]
            difference_vector.append(diff)
            land_info.append((new_land, soil1, soil2))

    # مقایسه خاک‌های زمین جدید با سایر زمین‌ها بدون تکرار زمین جدید
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append(diff_value)
                land_info.append((land, soil1, soil2))

    return difference_vector, land_info
def fill_difference_matrix(difference_vector, land_info, new_soils):
    num_rows = len(difference_vector)
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    for i, diff_value in enumerate(difference_vector):
        soil1, soil2 = land_info[i][1], land_info[i][2]
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i, soil2_index] = -1

    return difference_matrix, land_info
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه وکتورهای اختلاف و اطلاعات مربوطه
all_difference_vectors = {}
all_land_info_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    vertical_vector = np.array(difference_vector).reshape(-1, 1)
    all_difference_vectors[land] = vertical_vector
    all_land_info_vectors[land] = land_info

# نمایش وکتورهای اختلاف و اطلاعات مربوطه
for land, vector in all_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)
    print(f"Land Info for Land {land}:")
    print(all_land_info_vectors[land])

# نگهداری همه ماتریس‌های اختلاف و اطلاعات مربوطه
all_difference_matrices = {}
all_land_info_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix, land_info_matrix = fill_difference_matrix(difference_vector, land_info, new_soils)
    all_difference_matrices[land] = difference_matrix
    all_land_info_matrices[land] = land_info_matrix

# نمایش ماتریس‌های اختلاف و اطلاعات مربوطه
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)
    print(f"Land Info for Land {land}:")
    print(all_land_info_matrices[land])
import pandas as pd
import numpy as np

# فرض کنید new_df دیتا فریم اصلی شما باشد
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                diff = group[group['TYPE_2'] == soil1]['ij'].iloc[0] - group[group['TYPE_2'] == soil2]['ij'].iloc[0]
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land
def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    land_info = []

    # مقایسه خاک‌های داخل زمین جدید
    for i in range(len(new_soils)):
        for j in range(i + 1, len(new_soils)):
            soil1, soil2 = new_soils[i], new_soils[j]
            diff = new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil1)]['ij'].iloc[0] - new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil2)]['ij'].iloc[0]
            difference_vector.append(diff)
            land_info.append((new_land, soil1, soil2))

    # مقایسه خاک‌های زمین جدید با سایر زمین‌ها بدون تکرار زمین جدید
    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append(diff_value)
                land_info.append((land, soil1, soil2))

    return difference_vector, land_info
def add_one_to_first_row(difference_vector):
    return np.insert(difference_vector, 0, 1).reshape(-1, 1)
def fill_difference_matrix(difference_vector, land_info, new_soils, new_df):
    num_rows = len(difference_vector) + 1  # +1 برای اضافه کردن ردیف مساحت
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))

    # محاسبه میانگین مساحت‌ها
    avg_area = new_df['Area%'].mean()

    # تنظیم مساحت برای هر خاک
    for j, soil in enumerate(new_soils):
        soil_area = new_df[(new_df['Field_number'] == land) & (new_df['TYPE_2'] == soil)]['Area%'].iloc[0]
        if pd.isnull(soil_area):
            soil_area = avg_area
        difference_matrix[0, j] = soil_area

    # نرمال‌سازی مساحت‌ها به طوری که مجموع آن‌ها برابر 1 شود
    difference_matrix[0, :] /= difference_matrix[0, :].sum()

    for i, diff_value in enumerate(difference_vector):
        soil1, soil2 = land_info[i][1], land_info[i][2]
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None

        if soil1_index is not None:
            difference_matrix[i + 1, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i + 1, soil2_index] = -1

    return difference_matrix, land_info
# محاسبه اختلاف‌های خاک برای هر زمین
soil_diffs_per_land = calculate_soil_diffs(new_df)

# نگهداری همه وکتورهای اختلاف و اطلاعات مربوطه
all_difference_vectors = {}
all_land_info_vectors = {}

# ایجاد و ذخیره وکتورهای اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = new_df[new_df['Field_number'] == land]['TYPE_2'].unique()
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    vertical_vector = add_one_to_first_row(difference_vector)
    all_difference_vectors[land] = vertical_vector
    all_land_info_vectors[land] = land_info

# نمایش وکتورهای اختلاف و اطلاعات مربوطه
for land, vector in all_difference_vectors.items():
    print(f"Difference Vector for Land {land}:")
    print(vector)
    print(f"Land Info for Land {land}:")
    print(all_land_info_vectors[land])

# نگهداری همه ماتریس‌های اختلاف و اطلاعات مربوطه
all_difference_matrices = {}
all_land_info_matrices = {}

# ایجاد و ذخیره ماتریس‌های اختلاف به همراه اطلاعات زمین برای هر زمین
for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    difference_matrix, land_info_matrix = fill_difference_matrix(difference_vector, land_info, new_soils, new_df)
    all_difference_matrices[land] = difference_matrix
    all_land_info_matrices[land] = land_info_matrix

# نمایش ماتریس‌های اختلاف و اطلاعات مربوطه
for land, matrix in all_difference_matrices.items():
    print(f"Difference Matrix for Land {land}:")
    print(matrix)
    print(f"Land Info for Land {land}:")
    print(all_land_info_matrices[land])
# نگهداری همه میزان backslash برای هر زمین
all_solutions = {}

# محاسبه میزان backslash برای هر زمین
for land, matrix in all_difference_matrices.items():
    # وکتور اختلاف مربوط به هر زمین
    d = all_difference_vectors[land]

    # حل معادله خطی s = m \ d
    solution = np.linalg.lstsq(matrix, d, rcond=None)[0]

    # ذخیره میزان backslash
    all_solutions[land] = solution

# نمایش میزان backslash برای هر زمین
for land, solution in all_solutions.items():
    print(f"Backslash for Land {land}:")
    print(solution)
# نگهداری همه میزان backslash برای هر زمین
all_solutions_with_Avelog_f = {}

# محاسبه میزان backslash برای هر زمین
for land, matrix in all_difference_matrices.items():
    # وکتور اختلاف مربوط به هر زمین
    d = all_difference_vectors[land]

    # حل معادله خطی s = m \ d
    solution = np.linalg.lstsq(matrix, d, rcond=None)[0]

    # Find the value of Avelog_f for the specific land
    avelog_f_value = merged_data.loc[merged_data['Field_number'] == land, 'Avelog_f'].values[0]

    # Multiply each element of the solution vector by the Avelog_f value
    solution_with_Avelog_f = [value * avelog_f_value for value in solution]
    #print(solution_with_Avelog_f)

    # Store the solution vector with Avelog_f value for the land
    all_solutions_with_Avelog_f[land] = solution_with_Avelog_f

# نمایش میزان backslash با در نظر گرفتن Avelog_f برای هر زمین
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    print(f"Backslash with Avelog_f for Land {land}:")
    print(solution_with_Avelog_f)
import pandas as pd

# لیستی از دیکشنری‌ها که هر دیکشنری اطلاعات یک ردیف از دیتافریم جدید را نشان می‌دهد
data = []

# بررسی هر زمین و مقادیر مربوط به آن
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    # بررسی هر نوع خاک و مقدار مربوط به آن
    for soil_type, value in zip(new_df[new_df['Field_number'] == land]['TYPE_2'].unique(), solution_with_Avelog_f):
        # ایجاد دیکشنری برای اطلاعات یک ردیف
        row = {'Field_number': land, 'TYPE_2': soil_type, 'Calculated_value': value[0]}
        # افزودن دیکشنری به لیست دیکشنری‌ها
        data.append(row)

# ساخت دیتافریم با استفاده از لیست دیکشنری‌ها
result_df = pd.DataFrame(data)

# نمایش اولین چند ردیف از دیتافریم جدید برای اطمینان از صحت عملیات
print(result_df.head())
import pandas as pd

# لیستی از دیکشنری‌ها که هر دیکشنری اطلاعات یک ردیف از دیتافریم جدید را نشان می‌دهد
data = []

# بررسی هر زمین و مقادیر مربوط به آن
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    # بررسی هر نوع خاک و مقدار مربوط به آن
    for soil_type, value in zip(new_df[new_df['Field_number'] == land]['TYPE_2'].unique(), solution_with_Avelog_f):
        # ایجاد دیکشنری برای اطلاعات یک ردیف
        row = {'Field_number': land, 'TYPE_2': soil_type, 'Calculated_value': value[0]}
        # افزودن دیکشنری به لیست دیکشنری‌ها
        data.append(row)

# ساخت دیتافریم با استفاده از لیست دیکشنری‌ها
result_df = pd.DataFrame(data)

# ساخت ستون جدید predicted در دیتافریم merged_data و پر کردن آن با مقادیر محاسبه شده
merged_data['predictedWithout'] = merged_data.apply(lambda row: result_df[(result_df['Field_number'] == row['Field_number']) & (result_df['TYPE_2'] == row['TYPE_2'])]['Calculated_value'].values[0] if not result_df[(result_df['Field_number'] == row['Field_number']) & (result_df['TYPE_2'] == row['TYPE_2'])].empty else None, axis=1)

# نمایش چند ردیف از دیتافریم merged_data برای اطمینان از صحت عملیات
print(merged_data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# نمایش چند ردیف از دیتافریم
print(merged_data[['predictedWithout', 'log_Om']].head())

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['predictedWithout']

# افزودن ثابت به مدل
X_with_const = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X_with_const).fit()

# محاسبه R²
r_squared = model.rsquared

# محاسبه RMSE
y_pred = model.predict(X_with_const)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# رسم نمودار بدون هاله
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='predictedWithout', data=merged_data,
            scatter_kws={'s': 10},
            line_kws={'color': 'red'},
            ci=None)  # <- حذف هاله با ci=None

plt.xlabel('Log_SOM')
plt.ylabel('Predicted Values')
plt.title('Pairwise')
plt.grid(True)

# افزودن فقط R² و RMSE به نمودار
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# نمایش چند ردیف از دیتافریم
print(merged_data[['predictedWithout', 'log_Om']].head())

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['predictedWithout']

# افزودن ثابت به مدل
X_with_const = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X_with_const).fit()

# محاسبه R²
r_squared = model.rsquared

# محاسبه RMSE
y_pred = model.predict(X_with_const)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# 🎯 رسم دقیقاً مشابه پلات predicted
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='predictedWithout', data=merged_data,
            scatter_kws={'s': 10},
            line_kws={'color': 'red'},
            ci=None)

plt.xlabel('Log_SOM')
plt.ylabel('Predicted Values')
plt.title('Pairwise_noValidation')
plt.grid(True)

# 🔒 تنظیم مقیاس دقیق محور
plt.xlim(0, 1.75)
plt.ylim(0, 1.75)

# افزودن فقط R² و RMSE به نمودار
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
ticks1 = np.arange(0, 1.76, 0.25)
ticks2 = np.arange(0, 1.76, 0.20)

plt.gca().set_xticks(ticks1)
plt.gca().set_yticks(ticks2)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values, method_name):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    print(f"Evaluation metrics for {method_name}:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}\n")

# ستون های شامل نتایج پیش بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']

# ارزیابی دقت پیش‌بینی برای هر روش
for method in methods:
    evaluate_predictions(merged_data['log_Om'], merged_data[method], method)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون های شامل نتایج پیش بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method
    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# نمایش نتایج
print(results_df)

# ایجاد نمودارها برای مقایسه بصری
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

# نمودار RMSE
ax[0, 0].bar(results_df['Method'], results_df['RMSE'], color='blue')
ax[0, 0].set_title('RMSE Comparison')
ax[0, 0].set_xlabel('Method')
ax[0, 0].set_ylabel('RMSE')

# نمودار MAE
ax[0, 1].bar(results_df['Method'], results_df['MAE'], color='green')
ax[0, 1].set_title('MAE Comparison')
ax[0, 1].set_xlabel('Method')
ax[0, 1].set_ylabel('MAE')

# نمودار R-squared
ax[1, 0].bar(results_df['Method'], results_df['R-squared'], color='red')
ax[1, 0].set_title('R-squared Comparison')
ax[1, 0].set_xlabel('Method')
ax[1, 0].set_ylabel('R-squared')

# نمودار True vs Predicted for one method as example
ax[1, 1].scatter(merged_data['log_Om'], merged_data['predicted'], color='purple', label='Predicted')
ax[1, 1].plot([merged_data['log_Om'].min(), merged_data['log_Om'].max()],
              [merged_data['log_Om'].min(), merged_data['log_Om'].max()], color='orange', lw=2, label='True')
ax[1, 1].set_title('True vs Predicted')
ax[1, 1].set_xlabel('True Values')
ax[1, 1].set_ylabel('Predicted Values')
ax[1, 1].legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون های شامل نتایج پیش بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method
    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# نمایش نتایج
print(results_df)

# ایجاد نمودارها برای مقایسه بصری
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# نمودار RMSE
axs[0, 0].bar(results_df['Method'], results_df['RMSE'], color='blue')
axs[0, 0].set_title('RMSE Comparison')
axs[0, 0].set_xlabel('Method')
axs[0, 0].set_ylabel('RMSE')

# نمودار MAE
axs[0, 1].bar(results_df['Method'], results_df['MAE'], color='green')
axs[0, 1].set_title('MAE Comparison')
axs[0, 1].set_xlabel('Method')
axs[0, 1].set_ylabel('MAE')

# نمودار R-squared
axs[1, 0].bar(results_df['Method'], results_df['R-squared'], color='red')
axs[1, 0].set_title('R-squared Comparison')
axs[1, 0].set_xlabel('Method')
axs[1, 0].set_ylabel('R-squared')

# نمودار True vs Predicted برای هر روش به صورت جداگانه
colors = ['purple', 'orange', 'blue', 'green']
for i, method in enumerate(methods):
    axs[1, 1].scatter(merged_data['log_Om'], merged_data[method], color=colors[i], label=method)
    # اضافه کردن خط رگرسیون
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    p = np.poly1d(coeffs)
    axs[1, 1].plot(merged_data['log_Om'], p(merged_data['log_Om']), color=colors[i], linestyle='--')

axs[1, 1].set_title('True vs Predicted')
axs[1, 1].set_xlabel('True Values')
axs[1, 1].set_ylabel('Predicted Values')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# نمایش چند ردیف از دیتافریم
print(merged_data[['predicted', 'log_Om']].head())

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['predicted']

# افزودن ثابت به مدل
X_with_const = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X_with_const).fit()

# محاسبه R²
r_squared = model.rsquared

# محاسبه RMSE
y_pred = model.predict(X_with_const)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# رسم نمودار بدون هاله
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='predicted', data=merged_data,
            scatter_kws={'s': 10},
            line_kws={'color': 'red'},
            ci=None)  # <- حذف هاله با ci=None

plt.xlabel('Log_SOM')
plt.ylabel('Predicted Values')
plt.title('Pairwise_valid')
plt.grid(True)

# افزودن فقط R² و RMSE به نمودار
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
ticks1 = np.arange(0, 1.76, 0.25)
ticks2 = np.arange(0, 1.76, 0.20)

plt.gca().set_xticks(ticks1)
plt.gca().set_yticks(ticks2)
plt.tight_layout()
plt.show()

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# نمایش چند ردیف از دیتافریم merged_data با ستون‌های predicted و log_Om
print(merged_data[['predicted', 'log_Om']].head())

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['predicted']

# افزودن ثابت (intercept) به مدل
X = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X).fit()

# استخراج مقدار R-squared
r_squared = model.rsquared

# استخراج ضرایب مدل
intercept, slope = model.params

# رسم نمودار رگرسیون
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='predicted', data=merged_data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('LSF_Validation')
plt.grid(True)

# افزودن فرمول خط رگرسیون و مقدار R-squared به نمودار
line_eq = f'y = {intercept:.2f} + {slope:.2f}x'
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\n{line_eq}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score

# نمایش چند ردیف از دیتافریم merged_data با ستون‌های predictedWithout و log_Om
print(merged_data[['predicted', 'log_Om']].head())

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['predicted']

# افزودن ثابت (intercept) به مدل
X = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X).fit()

# استخراج مقدار R-squared
r_squared_statsmodels = model.rsquared

# استخراج ضرایب مدل
intercept, slope = model.params

# محاسبه R-squared با استفاده از sklearn
y_pred = model.predict(X)
r_squared_sklearn = r2_score(y, y_pred)

# نمایش مقادیر R-squared از دو روش
print(f"R-squared (statsmodels): {r_squared_statsmodels:.2f}")
print(f"R-squared (sklearn): {r_squared_sklearn:.2f}")

# رسم نمودار رگرسیون
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='predicted', data=merged_data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xlabel('True Values')
plt.ylabel('predicted values')
plt.title('LSF_Validation')
plt.grid(True)

# افزودن فرمول خط رگرسیون و مقدار R-squared به نمودار
line_eq = f'y = {intercept:.2f} + {slope:.2f}x'
plt.text(0.05, 0.95, f'$R^2$ = {r_squared_statsmodels:.2f}\n{line_eq}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    # تغییر نام متدها بر اساس درخواست شما
    if method == 'Avelog_f':
        method_name = 'Normalized'
    elif method == 'Beta_Pred':
        method_name = 'Normalized_Optimized'
    elif method == 'predicted':
        method_name = 'Matrix_Validation'
    elif method == 'predictedWithout':
        method_name = 'Matrix_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name
    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# نمایش نتایج
print(results_df)

# ایجاد نمودارها برای مقایسه بصری
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# نمودار RMSE
axs[0, 0].bar(results_df['Method'], results_df['RMSE'], color='blue')
axs[0, 0].set_title('RMSE Comparison')
axs[0, 0].set_xlabel('Method')
axs[0, 0].set_ylabel('RMSE')

# نمودار MAE
axs[0, 1].bar(results_df['Method'], results_df['MAE'], color='green')
axs[0, 1].set_title('MAE Comparison')
axs[0, 1].set_xlabel('Method')
axs[0, 1].set_ylabel('MAE')

# نمودار R-squared
axs[1, 0].bar(results_df['Method'], results_df['R-squared'], color='red')
axs[1, 0].set_title('R-squared Comparison')
axs[1, 0].set_xlabel('Method')
axs[1, 0].set_ylabel('R-squared')

# نمودار True vs Predicted برای هر روش به صورت جداگانه
colors = ['purple', 'orange', 'blue', 'green']
for i, method in enumerate(methods):
    # اضافه کردن خط رگرسیون
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    p = np.poly1d(coeffs)

    axs[1, 1].scatter(merged_data['log_Om'], merged_data[method], color=colors[i], label=results_df.loc[i, 'Method'])
    axs[1, 1].plot(merged_data['log_Om'], p(merged_data['log_Om']), color=colors[i], linestyle='--')

axs[1, 1].set_title('True vs Predicted')
axs[1, 1].set_xlabel('True Values')
axs[1, 1].set_ylabel('Predicted Values')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['Avelog_f']

# افزودن ثابت (intercept) به مدل
X = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X).fit()

# استخراج R-squared
r_squared = model.rsquared

# محاسبه RMSE
y_pred = model.predict(X)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

# رسم نمودار رگرسیون
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='Avelog_f', data=merged_data,
            scatter_kws={'s':10},
            line_kws={'color':'red'},
            ci=None)

plt.xlabel('Log SOM')
plt.ylabel('Predicted Values')
plt.title('RMSE_Traditional')
plt.grid(True)

# ✅ افزودن R² و RMSE به نمودار
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}',
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['Beta_Pred']

# افزودن ثابت (intercept) به مدل
X = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X).fit()

# استخراج مقدار R-squared
r_squared = model.rsquared

# استخراج ضرایب مدل
intercept, slope = model.params

# رسم نمودار رگرسیون
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='Beta_Pred', data=merged_data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xlabel('True Values')
plt.ylabel('predicted values')
plt.title('RMSE_Optimized')
plt.grid(True)

# افزودن فرمول خط رگرسیون و مقدار R-squared به نمودار
line_eq = f'y = {intercept:.2f} + {slope:.2f}x'
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\n{line_eq}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.api as sm

# آماده‌سازی داده‌ها برای مدل رگرسیون
X = merged_data['log_Om']
y = merged_data['Beta_Pred']

# افزودن ثابت (intercept) به مدل
X = sm.add_constant(X)

# ایجاد و آموزش مدل رگرسیون خطی
model = sm.OLS(y, X).fit()

# استخراج مقدار R-squared از مدل statsmodels
r_squared_statsmodels = model.rsquared

# استخراج ضرایب مدل
intercept, slope = model.params

# محاسبه R-squared با استفاده از sklearn
y_pred = model.predict(X)
r_squared_sklearn = r2_score(y, y_pred)

# نمایش مقادیر R-squared از دو روش
print(f"R-squared (statsmodels): {r_squared_statsmodels:.2f}")
print(f"R-squared (sklearn): {r_squared_sklearn:.2f}")

# رسم نمودار رگرسیون
plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='Beta_Pred', data=merged_data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xlabel('True Values')
plt.ylabel('predicted values')
plt.title('RMSE_Optimized')
plt.grid(True)

# افزودن فرمول خط رگرسیون و مقدار R-squared به نمودار
line_eq = f'y = {intercept:.2f} + {slope:.2f}x'
plt.text(0.05, 0.95, f'$R^2$ = {r_squared_statsmodels:.2f}\n{line_eq}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    # تغییر نام متدها بر اساس درخواست شما
    if method == 'Avelog_f':
        method_name = 'RMSE_Base'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Matrix_Validation'
    elif method == 'predictedWithout':
        method_name = 'Matrix_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name
    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# مرتب‌سازی DataFrame براساس RMSE
results_df = results_df.sort_values(by='RMSE')

# نمایش نتایج
print(results_df)

# ایجاد نمودارها برای مقایسه بصری
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# نمودار RMSE
axs[0, 0].bar(results_df['Method'], results_df['RMSE'], color='blue')
axs[0, 0].set_title('RMSE Comparison')
axs[0, 0].set_xlabel('Method')
axs[0, 0].set_ylabel('RMSE')

# نمودار MAE
axs[0, 1].bar(results_df['Method'], results_df['MAE'], color='green')
axs[0, 1].set_title('MAE Comparison')
axs[0, 1].set_xlabel('Method')
axs[0, 1].set_ylabel('MAE')

# نمودار R-squared
axs[1, 0].bar(results_df['Method'], results_df['R-squared'], color='red')
axs[1, 0].set_title('R-squared Comparison')
axs[1, 0].set_xlabel('Method')
axs[1, 0].set_ylabel('R-squared')

# نمودار True vs Predicted برای هر روش به صورت جداگانه
colors = ['purple', 'orange', 'blue', 'green']
for i, method in enumerate(results_df['Method']):
    method_index = methods[results_df.index[i]]
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method_index], 1)
    p = np.poly1d(coeffs)

    axs[1, 1].scatter(merged_data['log_Om'], merged_data[method_index], color=colors[i], label=method)
    axs[1, 1].plot(merged_data['log_Om'], p(merged_data['log_Om']), color=colors[i], linestyle='--')

axs[1, 1].set_title('True vs Predicted')
axs[1, 1].set_xlabel('True Values')
axs[1, 1].set_ylabel('Predicted Values')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    # تغییر نام متدها بر اساس درخواست شما
    if method == 'Avelog_f':
        method_name = 'RMSE_Base'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Matrix_Validation'
    elif method == 'predictedWithout':
        method_name = 'Matrix_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name

    # محاسبه معادله رگرسیون خطی
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    metrics['Regression_Coefficients'] = coeffs

    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# مرتب‌سازی DataFrame براساس RMSE
results_df = results_df.sort_values(by='RMSE')

# نمایش نتایج
print(results_df)

# چاپ معادلات رگرسیون خطی
for index, row in results_df.iterrows():
    print(f"{row['Method']} Regression Equation: y = {row['Regression_Coefficients'][0]} * x + {row['Regression_Coefficients'][1]}")

# ایجاد نمودارها برای مقایسه بصری
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# نمودار RMSE
axs[0, 0].bar(results_df['Method'], results_df['RMSE'], color='blue')
axs[0, 0].set_title('RMSE Comparison')
axs[0, 0].set_xlabel('Method')
axs[0, 0].set_ylabel('RMSE')

# نمودار MAE
axs[0, 1].bar(results_df['Method'], results_df['MAE'], color='green')
axs[0, 1].set_title('MAE Comparison')
axs[0, 1].set_xlabel('Method')
axs[0, 1].set_ylabel('MAE')

# نمودار R-squared
axs[1, 0].bar(results_df['Method'], results_df['R-squared'], color='red')
axs[1, 0].set_title('R-squared Comparison')
axs[1, 0].set_xlabel('Method')
axs[1, 0].set_ylabel('R-squared')

# نمودار True vs Predicted برای هر روش به صورت جداگانه
colors = ['purple', 'orange', 'blue', 'green']
for i, method in enumerate(results_df['Method']):
    method_index = methods[results_df.index[i]]
    coeffs = results_df.loc[results_df['Method'] == method, 'Regression_Coefficients'].values[0]
    p = np.poly1d(coeffs)

    axs[1, 1].scatter(merged_data['log_Om'], merged_data[method_index], color=colors[i], label=method)
    axs[1, 1].plot(merged_data['log_Om'], p(merged_data['log_Om']), color=colors[i], linestyle='--')

axs[1, 1].set_title('True vs Predicted')
axs[1, 1].set_xlabel('True Values')
axs[1, 1].set_ylabel('Predicted Values')
axs[1, 1].legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    # تغییر نام متدها بر اساس درخواست شما
    if method == 'Avelog_f':
        method_name = 'RMSE_Base'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Matrix_Validation'
    elif method == 'predictedWithout':
        method_name = 'Matrix_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name

    # محاسبه معادله رگرسیون خطی
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    metrics['Regression_Coefficients'] = coeffs

    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# مرتب‌سازی DataFrame براساس RMSE
results_df = results_df.sort_values(by='RMSE')

# نمایش نتایج
print(results_df)

# چاپ معادلات رگرسیون خطی
for index, row in results_df.iterrows():
    print(f"{row['Method']} Regression Equation: y = {row['Regression_Coefficients'][0]} * x + {row['Regression_Coefficients'][1]}")

# ایجاد نمودارها برای مقایسه بصری
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

for i, method in enumerate(methods):
    method_name = results_df.iloc[i]['Method']
    coeffs = results_df.iloc[i]['Regression_Coefficients']
    p = np.poly1d(coeffs)

    axs[i].scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')
    axs[i].plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--', label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    axs[i].set_title(f'{method_name} R-squared: {results_df.iloc[i]["R-squared"]:.2f}')
    axs[i].set_xlabel('True Values')
    axs[i].set_ylabel('Predicted Values')
    axs[i].legend()

    # افزودن مقادیر RMSE و MAE به نمودار
    textstr = '\n'.join((
        f'RMSE: {results_df.iloc[i]["RMSE"]:.2f}',
        f'MAE: {results_df.iloc[i]["MAE"]:.2f}'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

plt.tight_layout()
ax.set_xlim(0, 1.75)
ax.set_ylim(0, 1.75)
ax.set_aspect('equal')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از سه تکنیک متفاوت
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش و ذخیره نتایج
for method in methods:
    # تغییر نام متدها بر اساس درخواست شما
    if method == 'Avelog_f':
        method_name = 'RMSE_Initial'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'LSF_Validation'
    elif method == 'predictedWithout':
        method_name = 'LSF_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name

    # محاسبه معادله رگرسیون خطی
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    metrics['Regression_Coefficients'] = coeffs

    results.append(metrics)

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)

# مرتب‌سازی DataFrame براساس RMSE
results_df = results_df.sort_values(by='RMSE')

# نمایش نتایج
print(results_df)

# ایجاد نمودارها برای مقایسه بصری
fig, axs = plt.subplots(1, len(methods), figsize=(20, 5))

for i, method in enumerate(methods):
    method_name = results_df.iloc[i]['Method']
    coeffs = results_df.iloc[i]['Regression_Coefficients']
    p = np.poly1d(coeffs)

    # رسم داده‌ها و خط رگرسیون
    axs[i].scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')
    axs[i].plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--', label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')

    # افزودن مقادیر RMSE و MAE به نمودار
    textstr = '\n'.join((
        f'RMSE: {results_df.iloc[i]["RMSE"]:.2f}',
        f'MAE: {results_df.iloc[i]["MAE"]:.2f}'))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[i].text(0.95, 0.05, textstr, transform=axs[i].transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # تنظیمات دیگر نمودار
    axs[i].set_title(f'{method_name} R-squared: {results_df.iloc[i]["R-squared"]:.2f}')
    axs[i].set_xlabel('True Values')
    axs[i].set_ylabel('Predicted Values')
    axs[i].legend()

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از چند تکنیک
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش
for method in methods:
    if method == 'Avelog_f':
        method_name = 'RMSE_preliminary'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Pairwise_Validation'
    elif method == 'predictedWithout':
        method_name = 'Pairwise_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name

    # محاسبه خط رگرسیون (فقط برای رسم)
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    metrics['Regression_Coefficients'] = coeffs

    results.append(metrics)

# ایجاد DataFrame و مرتب‌سازی بر اساس RMSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='RMSE')

# چاپ نتایج
print(results_df)

# رسم نمودارها
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for i, method in enumerate(methods):
    method_name = results_df.iloc[i]['Method']
    coeffs = results_df.iloc[i]['Regression_Coefficients']
    p = np.poly1d(coeffs)

    ax = axs[i // 2, i % 2]
    ax.scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')

    # رسم خط رگرسیون بدون افزودن label
    ax.plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--')

    # حذف عنوان بالا و تعیین عناوین محور
    ax.set_title('')
    ax.set_xlabel('Log_SOM')
    ax.set_ylabel('Predicted')
    ax.legend()
    ax.grid(True)


    # نمایش فقط Method، RMSE و R² در باکس داخل نمودار
    textstr = '\n'.join((
        f'Method: {method_name}',
        f'RMSE: {results_df.iloc[i]["RMSE"]:.2f}',
        f'R²: {results_df.iloc[i]["R-squared"]:.2f}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

plt.tight_layout()

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return {'RMSE': rmse, 'MAE': mae, 'R-squared': r2}

methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

for method in methods:
    if method == 'Avelog_f':
        method_name = 'RMSE_preliminary'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Pairwise_Validation'
    elif method == 'predictedWithout':
        method_name = 'Pairwise'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    metrics['Regression_Coefficients'] = coeffs
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='RMSE')
print(results_df)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for i, method in enumerate(methods):
    method_name = results_df.iloc[i]['Method']
    coeffs = results_df.iloc[i]['Regression_Coefficients']
    p = np.poly1d(coeffs)

    ax = axs[i // 2, i % 2]
    ax.scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')
    ax.plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--')

    ax.set_title('')
    ax.set_xlabel('Log_SOM')
    ax.set_ylabel('Predicted')
    ax.legend()
    ax.grid(True)

    # تنظیم محدوده و نسبت و تقسیمات مساوی

    ticks1 = np.arange(0, 1.76, 0.25)
    ticks2 = np.arange(0, 1.76, 0.20)

    ax.set_xticks(ticks1)
    ax.set_yticks(ticks2)

    textstr = '\n'.join((
        f'Method: {method_name}',
        f'RMSE: {results_df.iloc[i]["RMSE"]:.2f}',
        f'R²: {results_df.iloc[i]["R-squared"]:.2f}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

for method in methods:
    if method == 'Avelog_f':
        method_name = 'RMSE_Field Average'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Pairwise_Validation'
    elif method == 'predictedWithout':
        method_name = 'Pairwise'

    true = merged_data['log_Om']
    predicted = merged_data[method]

    # برازش مدل فقط برای خط رگرسیون
    X = sm.add_constant(true)
    model = sm.OLS(predicted, X).fit()
    intercept, slope = model.params

    # ❗ محاسبه RMSE به صورت دستی با فرمول اصلی
    residuals = predicted - true
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    r2 = model.rsquared  # از OLS برای زیبایی، ولی میشه دستی هم محاسبه کرد

    results.append({
        'Method': method_name,
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2,
        'intercept': intercept,
        'slope': slope,
        'true': true,
        'predicted': predicted
    })

# نمایش و رسم نتایج
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for i, result in enumerate(results):
    ax = axs[i // 2, i % 2]

    true = result['true']
    predicted = result['predicted']
    intercept = result['intercept']
    slope = result['slope']
    line = intercept + slope * true

    ax.scatter(true, predicted, color='blue', label='Data')
    ax.plot(true, line, color='red', linestyle='--', label='OLS fit')

    ax.set_xlim(0, 1.75)
    ax.set_ylim(0, 1.75)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, 1.76, 0.25))
    ax.set_yticks(np.arange(0, 1.76, 0.25))
    ax.set_xlabel('Log_SOM')
    ax.set_ylabel('Predicted')
    ax.grid(True)
    ax.legend()

    textstr = '\n'.join((
        f'Method: {result["Method"]}',
        f'RMSE: {result["RMSE"]:.3f}',
        f'R²: {result["R-squared"]:.3f}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ستون‌های شامل نتایج پیش‌بینی شده از چند تکنیک
methods = ['Avelog_f', 'Beta_Pred', 'predicted', 'predictedWithout']
results = []

# ارزیابی دقت پیش‌بینی برای هر روش
for method in methods:
    if method == 'Avelog_f':
        method_name = 'RMSE_preliminary'
    elif method == 'Beta_Pred':
        method_name = 'RMSE_Optimized'
    elif method == 'predicted':
        method_name = 'Pairwise_Validation'
    elif method == 'predictedWithout':
        method_name = 'Pairwise_NoValidation'

    metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
    metrics['Method'] = method_name

    # محاسبه خط رگرسیون (فقط برای رسم)
    coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
    metrics['Regression_Coefficients'] = coeffs

    results.append(metrics)

# ایجاد DataFrame و مرتب‌سازی بر اساس RMSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='RMSE')

# چاپ نتایج
print(results_df)

# رسم نمودارها
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for i, method in enumerate(methods):
    method_name = results_df.iloc[i]['Method']
    coeffs = results_df.iloc[i]['Regression_Coefficients']
    p = np.poly1d(coeffs)

    ax = axs[i // 2, i % 2]
    ax.scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')
    ax.plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--')

    ax.set_title('')
    ax.set_xlabel('Log_SOM')
    ax.set_ylabel('Predicted')
    ax.grid(True)
    ax.legend()

    # 🔧 تنظیم مقیاس محورها و مربع‌سازی نمودار
    ax.set_xlim(0, 1.75)
    ax.set_ylim(0, 1.75)
    #ax.set_aspect('equal')

    # نمایش باکس اطلاعات داخل نمودار
    textstr = '\n'.join((
        f'Method: {method_name}',
        f'RMSE: {results_df.iloc[i]["RMSE"]:.2f}',
        f'R²: {results_df.iloc[i]["R-squared"]:.2f}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ارزیابی دقت پیش‌بینی برای روش Avelog_f
method = 'Avelog_f'
method_name = 'RMSE_Initial'

metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
metrics['Method'] = method_name

# محاسبه معادله رگرسیون خطی
coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
metrics['Regression_Coefficients'] = coeffs

# نمایش نتایج
print(metrics)

# رسم نمودار
fig, ax = plt.subplots(figsize=(7, 5))

p = np.poly1d(coeffs)

# رسم داده‌ها و خط رگرسیون
ax.scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')
ax.plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--', label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')

# افزودن مقادیر RMSE و MAE به نمودار
textstr = '\n'.join((
    f'RMSE: {metrics["RMSE"]:.2f}',
    f'MAE: {metrics["MAE"]:.2f}'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# تنظیمات دیگر نمودار
ax.set_title(f'{method_name} R-squared: {metrics["R-squared"]:.2f}')
ax.set_xlabel('log_Om')
ax.set_ylabel(method)
ax.legend()
ticks1 = np.arange(0, 1.76, 0.25)
ticks2 = np.arange(0, 1.76, 0.20)

ax.set_xticks(ticks1)
ax.set_yticks(ticks2)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# فرض کنیم merged_data از قبل تعریف شده باشد
# محاسبه معیارهای ارزیابی برای هر ستون پیش‌بینی

def evaluate_predictions(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# ارزیابی دقت پیش‌بینی برای روش Beta_Pred
method = 'Beta_Pred'
method_name = 'RMSE_Optimized'

metrics = evaluate_predictions(merged_data['log_Om'], merged_data[method])
metrics['Method'] = method_name

# محاسبه معادله رگرسیون خطی
coeffs = np.polyfit(merged_data['log_Om'], merged_data[method], 1)
metrics['Regression_Coefficients'] = coeffs

# نمایش نتایج
print(metrics)

# رسم نمودار
fig, ax = plt.subplots(figsize=(7, 5))

p = np.poly1d(coeffs)

# رسم داده‌ها و خط رگرسیون
ax.scatter(merged_data['log_Om'], merged_data[method], color='blue', label='Data')
ax.plot(merged_data['log_Om'], p(merged_data['log_Om']), color='red', linestyle='--', label=f'Fit: y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')

# افزودن مقادیر RMSE و MAE به نمودار
textstr = '\n'.join((
    f'RMSE: {metrics["RMSE"]:.2f}',
    f'MAE: {metrics["MAE"]:.2f}'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

# تنظیمات دیگر نمودار
ax.set_title(f'{method_name} R-squared: {metrics["R-squared"]:.2f}')
ax.set_xlabel('log_Om')
ax.set_ylabel(method)
ax.legend()

plt.tight_layout()
plt.show()

# نمایش چند ردیف از دیتافریم merged_data با ستون‌های Field_number، TYPE_2، log_Om و Avelog_f
print(merged_data[['Field_number', 'TYPE_2', 'log_Om', 'Avelog_f']].head())

# استخراج داده‌های مربوطه و نمایش آنها
data_subset = merged_data[['Field_number', 'TYPE_2', 'log_Om', 'Avelog_f']]

# نمایش داده‌ها
print(data_subset)

# اگر مایل هستید داده‌ها را ذخیره کنید یا به صورت جدولی نمایش دهید
data_subset.to_csv('data_comparison.csv', index=False)

"""# **تست روش بدست آوردن بتا از مقادیر میانگین ماده آلی هر نوع خاک در هر زمین**"""

'''
import numpy as np
import pandas as pd

# فرض: merged_data شامل ستون‌های Field_number، TYPE_2، log_Om است

def build_J_and_S(merged_data):
    # همه انواع خاک موجود
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    J_rows = []
    S_values = []

    # مرتب‌سازی بر اساس تعداد زمین‌های دارای هر نوع خاک
    soil_field_counts = merged_data.groupby('TYPE_2')['Field_number'].nunique().sort_values(ascending=False)
    sorted_soils = soil_field_counts.index.tolist()

    for soil in sorted_soils:
        # پیدا کردن تمام field‌هایی که این نوع خاک را دارند
        fields_with_soil = merged_data[merged_data['TYPE_2'] == soil]['Field_number'].unique()

        for field in fields_with_soil:
            # میانگین log_Om آن نوع خاک در این زمین
            w = merged_data[(merged_data['Field_number'] == field) &
                            (merged_data['TYPE_2'] == soil)]['log_Om'].mean()

            row = np.zeros(len(soil_types))
            row[soil_type_to_index[soil]] = w
            J_rows.append(row)

            # مقدارهای واقعی log_Om نمونه‌های این خاک در این زمین → به عنوان S
            samples = merged_data[(merged_data['Field_number'] == field) &
                                  (merged_data['TYPE_2'] == soil)]['log_Om'].values
            S_values.extend(samples)

    J = np.array(J_rows)
    S = np.array(S_values).reshape(-1, 1)
    return J, S, soil_types

# مثال استفاده:
J, S, soil_types = build_J_and_S(merged_data)

# حل معادله J * B = S
# چون J و S ابعاد متفاوت دارند، باید ابتدا J را به تعداد ردیف‌های S تکرار کنیم.
# (به این معنا که برای هر sample مقدار J مربوطه را برداریم)
J_expanded = np.repeat(J, [len(merged_data[(merged_data['TYPE_2'] == soil)]['log_Om'])
                           for soil in soil_types for field in merged_data[merged_data['TYPE_2'] == soil]['Field_number'].unique()], axis=0)

# حل least squares
B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)

# نمایش ضرایب به دست آمده برای هر soil type
for soil, b_value in zip(soil_types, B.flatten()):
    print(f"Soil Type: {soil}, B: {b_value:.4f}")
'''

import numpy as np
import pandas as pd

def build_J_and_S(merged_data):
    """
    Corrected version based on real model:
    - J = zone matrix (n_zones x n_soil_types)
    - S = sample vector (n_samples x 1)
    - sample → zone mapping for J_expanded
    """
    # همه انواع خاک به ترتیب
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    J_rows = []
    zone_keys = []

    # ساخت J (zone matrix)
    zones = merged_data.groupby(['Field_number', 'TYPE_2'])
    for (field, soil), group in zones:
        row = np.zeros(len(soil_types))
        row[soil_type_to_index[soil]] = group['log_Om'].mean()
        J_rows.append(row)
        zone_keys.append((field, soil))
    J = np.array(J_rows)

    # ساخت S (sample values) + mapping sample → zone
    S_values = []
    sample_zone_index = []
    for i, (field, soil) in enumerate(zone_keys):
        samples = merged_data[(merged_data['Field_number'] == field) &
                              (merged_data['TYPE_2'] == soil)]['log_Om'].values
        S_values.extend(samples)
        sample_zone_index.extend([i] * len(samples))
    S = np.array(S_values).reshape(-1, 1)

    return J, S, soil_types, sample_zone_index

# مثال استفاده:
J, S, soil_types, sample_zone_index = build_J_and_S(merged_data)

# ساخت J_expanded با mapping صحیح
J_expanded = J[sample_zone_index, :]

# حل least squares
B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)

# نمایش ضرایب به دست آمده برای هر soil type
for soil, b_value in zip(soil_types, B.flatten()):
    print(f"Soil Type: {soil}, B: {b_value}")

import numpy as np

# فرض: J_expanded و S همین الآن ساخته شده‌اند
# مطابق مدل واقعی مقاله

# حل B (همان J\S)
B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)

# محاسبه error
predicted_S = J_expanded @ B   # معادل J * B
error = np.sqrt(np.sum((predicted_S - S) ** 2))

print(f"Total error = {error:.12f}")

import numpy as np

# فرض: J_expanded و S همین الآن ساخته شده‌اند
# مطابق مدل واقعی مقاله

# حل B (همان J\S)
B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)

# محاسبه error
predicted_S = J_expanded @ B   # معادل J * B
RSSR = np.sqrt(np.sum((predicted_S - S) ** 2))
print(f"Total error (RSSR) = {RSSR:.12f}")

# محاسبه RMS error
n_samples = S.shape[0]
RMS_error = RSSR / np.sqrt(n_samples)
print(f"RMS error per sample = {RMS_error:.12f}")
#Root Sum of Squared Residuals (RSSR)

# محاسبه R-squared
SS_res = np.sum((S - predicted_S) ** 2)
SS_tot = np.sum((S - np.mean(S)) ** 2)
R_squared = 1 - (SS_res / SS_tot)
print(f"R-squared = {R_squared:.6f}")

import pandas as pd

# J_expanded را به DataFrame تبدیل کن برای نمایش جدولی
J_expanded_df = pd.DataFrame(J_expanded, columns=soil_types)

# index را به sample number تغییر بده
J_expanded_df.index = [f"Sample_{i+1}" for i in range(J_expanded.shape[0])]

# نمایش جدول به شکل کامل
print("Schematic of J_expanded (Table Form):")
print(J_expanded_df.to_string())

import pandas as pd
import numpy as np

# تعریف کلی برای جدول نمادین مقاله
n_samples = 8  # تعداد نمونه‌های نمایشی
n_soil_types = 6  # تعداد خاک‌های نمایشی

# تعریف ستون‌ها → نمادین
soil_types = ['Type 1', 'Type 2', '.', '.', 'Type m']
samples = ['Sample 1', 'Sample 2', '.', '.', 'Sample n']

# DataFrame با صفر
J_expanded_schematic = pd.DataFrame('0', index=samples, columns=soil_types)

# نمونه‌های نمادین (فقط مثال، فقط برای نمایش مدل)
# به چند ردیف اولیه X̄_ij بدهیم
J_expanded_schematic.loc['Sample 1', 'Type 1'] = 'X̄_ij'
J_expanded_schematic.loc['Sample 2', 'Type 2'] = 'X̄_ij'
J_expanded_schematic.loc['.', '.'] = 'X̄_ij'
J_expanded_schematic.loc['Sample n', 'Type m'] = 'X̄_ij'

# نمایش جدول
print("✅ Generalized Schematic of J_expanded (Paper-ready symbolic form):")
print(J_expanded_schematic.to_string(index=True))

import numpy as np
import pandas as pd

def build_J_and_S_real_model(merged_data):
    """
    Final corrected model:
    - S = raw sample values (X_ijk)
    - J = zone matrix = X̄_ij at sample level + intercept
    """
    # Define soil types in fixed order
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    # Step 1: calculate zone means (X̄_ij)
    zones = merged_data.groupby(['Field_number', 'TYPE_2'])
    zone_means = {}
    for (field, soil), group in zones:
        zone_means[(field, soil)] = group['log_Om'].mean()

    # Step 2: build S and J
    S_values = []
    J_rows = []
    for index, row in merged_data.iterrows():
        field = row['Field_number']
        soil = row['TYPE_2']
        sample_value = row['log_Om']
        zone_mean = zone_means[(field, soil)]

        S_values.append(sample_value)

        # row for J → zone mean only for soil type of sample
        J_row = np.zeros(len(soil_types))
        J_row[soil_type_to_index[soil]] = zone_mean
        J_rows.append(J_row)

    S = np.array(S_values).reshape(-1, 1)
    J = np.array(J_rows)

    # Step 3: add intercept column
    intercept_column = np.ones((J.shape[0], 1))
    J_expanded = np.hstack([intercept_column, J])
    coef_names = ['Intercept'] + soil_types

    return J_expanded, S, coef_names

# ==================
# Example usage:
# ==================
# merged_data must exist: DataFrame with columns ['Field_number', 'TYPE_2', 'log_Om']

J_expanded, S, coef_names = build_J_and_S_real_model(merged_data)

# Solve least squares
B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)

# Show results
print("\nCoefficients (final model + intercept):")
for name, value in zip(coef_names, B.flatten()):
    print(f"{name}: {value:.6f}")

# Calculate error
predicted_S = J_expanded @ B
RSSR = np.sqrt(np.sum((predicted_S - S) ** 2))
RMS_error = RSSR / np.sqrt(S.shape[0])

print(f"\nTotal error (RSSR) = {RSSR:.12f}")
print(f"RMS error per sample = {RMS_error:.12f}")

import numpy as np
import pandas as pd

def build_J_and_S_normalized_model(merged_data):
    """
    Build J and S matrices for zonal regression with min-max normalized zone means.
    """
    # Step 1: fixed soil type order
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    # Step 2: compute zone means X̄_ij
    zone_means = {}
    for (field, soil), group in merged_data.groupby(['Field_number', 'TYPE_2']):
        zone_means[(field, soil)] = group['log_Om'].mean()

    # Step 3: normalize the zone means using min-max scaling
    mean_values = np.array(list(zone_means.values()))
    min_val, max_val = mean_values.min(), mean_values.max()
    normalized_zone_means = {
        key: (value - min_val) / (max_val - min_val)
        for key, value in zone_means.items()
    }

    # Step 4: construct J and S
    S_values = []
    J_rows = []
    for _, row in merged_data.iterrows():
        field = row['Field_number']
        soil = row['TYPE_2']
        sample_value = row['log_Om']
        norm_mean = normalized_zone_means[(field, soil)]

        S_values.append(sample_value)

        J_row = np.zeros(len(soil_types))
        J_row[soil_type_to_index[soil]] = norm_mean
        J_rows.append(J_row)

    S = np.array(S_values).reshape(-1, 1)
    J = np.array(J_rows)

    # Step 5: add intercept column
    intercept_column = np.ones((J.shape[0], 1))
    J_expanded = np.hstack([intercept_column, J])
    coef_names = ['Intercept'] + soil_types

    return J_expanded, S, coef_names

# ============================
# Example usage
# ============================

# فرض بر این است که merged_data شامل ستون‌های زیر است:
# 'Field_number', 'TYPE_2', 'log_Om'

# ⚠️ اگر ستون log_Om هنوز ساخته نشده:
# merged_data['log_Om'] = np.log10(merged_data['Om_p'])  ← اگر Om_p دارید

# ساخت ماتریس J و بردار S
J_expanded, S, coef_names = build_J_and_S_normalized_model(merged_data)

# حل مدل رگرسیون: β = J \ S
B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)

# پیش‌بینی و محاسبه خطا
predicted_S = J_expanded @ B
RSSR = np.sqrt(np.sum((predicted_S - S) ** 2))
RMS_error = RSSR / np.sqrt(S.shape[0])

# نمایش ضرایب
print("\nCoefficients (Normalized Model + Intercept):")
for name, value in zip(coef_names, B.flatten()):
    print(f"{name}: {value:.6f}")

# نمایش خطا
print(f"\nTotal error (RSSR) = {RSSR:.6f}")
print(f"RMS error per sample = {RMS_error:.6f}")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ========= Helper Function for Zonal Normalized Model =========
def build_J_and_S_normalized_model(data):
    soil_types = sorted(data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    zone_means = data.groupby(['Field_number', 'TYPE_2'])['log_Om'].mean().to_dict()
    mean_values = np.array(list(zone_means.values()))
    min_val, max_val = mean_values.min(), mean_values.max()
    normalized_zone_means = {
        key: (value - min_val) / (max_val - min_val)
        for key, value in zone_means.items()
    }

    S_values, J_rows = [], []
    for _, row in data.iterrows():
        field = row['Field_number']
        soil = row['TYPE_2']
        if (field, soil) not in normalized_zone_means:
            continue
        norm_mean = normalized_zone_means[(field, soil)]
        row_vector = np.zeros(len(soil_types))
        row_vector[soil_type_to_index[soil]] = norm_mean
        J_rows.append(row_vector)
        S_values.append(row['log_Om'])

    S = np.array(S_values).reshape(-1, 1)
    J = np.array(J_rows)
    intercept = np.ones((J.shape[0], 1))
    J_expanded = np.hstack([intercept, J])
    coef_names = ['Intercept'] + soil_types

    return J_expanded, S, coef_names, normalized_zone_means, min_val, max_val

# ========= Scenario A: Full Dataset Evaluation =========
def evaluate_full_dataset(merged_data):
    J_expanded, S, coef_names, _, _, _ = build_J_and_S_normalized_model(merged_data)
    B, residuals, rank, s = np.linalg.lstsq(J_expanded, S, rcond=None)
    predicted_S = J_expanded @ B
    rmse = np.sqrt(mean_squared_error(S, predicted_S))

    print("\n🟢 Scenario A: Full Dataset Evaluation")
    for name, value in zip(coef_names, B.flatten()):
        print(f"{name}: {value:.6f}")
    print(f"\nFull Dataset RMSE: {rmse:.6f}")
    return rmse

# ========= Scenario B: LOFO-CV =========
def zonal_normalized_lofo_cv(merged_data):
    all_preds = []
    all_truths = []
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    for test_field in merged_data['Field_number'].unique():
        train_data = merged_data[merged_data['Field_number'] != test_field]
        test_data = merged_data[merged_data['Field_number'] == test_field]

        # Zone means from training data
        zone_means = train_data.groupby(['Field_number', 'TYPE_2'])['log_Om'].mean().to_dict()
        if len(zone_means) == 0:
            continue
        mean_values = np.array(list(zone_means.values()))
        min_val, max_val = mean_values.min(), mean_values.max()
        normalized_zone_means = {
            key: (value - min_val) / (max_val - min_val)
            for key, value in zone_means.items()
        }

        # Build training J and S
        S_train, J_train_rows = [], []
        for _, row in train_data.iterrows():
            field, soil = row['Field_number'], row['TYPE_2']
            if (field, soil) not in normalized_zone_means:
                continue
            norm_mean = normalized_zone_means[(field, soil)]
            row_vector = np.zeros(len(soil_types))
            row_vector[soil_type_to_index[soil]] = norm_mean
            J_train_rows.append(row_vector)
            S_train.append(row['log_Om'])

        if len(S_train) == 0:
            continue

        S_train = np.array(S_train).reshape(-1, 1)
        J_train = np.array(J_train_rows)
        J_train_expanded = np.hstack([np.ones((J_train.shape[0], 1)), J_train])
        B, _, _, _ = np.linalg.lstsq(J_train_expanded, S_train, rcond=None)

        # Build test J and S
        J_test_rows, S_test = [], []
        for _, row in test_data.iterrows():
            soil = row['TYPE_2']
            test_mean = test_data[test_data['TYPE_2'] == soil]['log_Om'].mean()
            norm_test_mean = (test_mean - min_val) / (max_val - min_val)
            row_vector = np.zeros(len(soil_types))
            row_vector[soil_type_to_index[soil]] = norm_test_mean
            J_test_rows.append(row_vector)
            S_test.append(row['log_Om'])

        if len(J_test_rows) == 0:
            continue

        S_test = np.array(S_test).reshape(-1, 1)
        J_test = np.array(J_test_rows)
        J_test_expanded = np.hstack([np.ones((J_test.shape[0], 1)), J_test])
        preds = J_test_expanded @ B

        all_preds.extend(preds.flatten())
        all_truths.extend(S_test.flatten())

    rmse = np.sqrt(mean_squared_error(all_truths, all_preds))
    print(f"\n🟡 Scenario B: LOFO-CV Evaluation")
    print(f"LOFO-CV RMSE: {rmse:.6f}")
    return rmse

# ========= Run Both Scenarios =========
def run_zonal_validation(merged_data):
    print("Running zonal regression model with normalized zone means...")
    rmse_full = evaluate_full_dataset(merged_data)
    rmse_lofo = zonal_normalized_lofo_cv(merged_data)
    print("\n✅ Summary:")
    print(f"RMSE (Full Dataset): {rmse_full:.6f}")
    print(f"RMSE (LOFO-CV):       {rmse_lofo:.6f}")

# ========= Example Call =========
# Make sure 'log_Om' column exists in merged_data
# merged_data['log_Om'] = np.log10(merged_data['Om_p'])
run_zonal_validation(merged_data)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_matrix_table(J_expanded, S, B, coef_names):
    # J_expanded → DataFrame
    J_df = pd.DataFrame(J_expanded, columns=coef_names)
    J_df['S'] = S.flatten()
    J_df.index = [f"Sample_{i+1}" for i in range(J_df.shape[0])]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(min(20, 1 + J_df.shape[1]), min(1 + J_df.shape[0]*0.5, 20)))
    ax.axis('off')
    ax.axis('tight')

    # Create table
    table = ax.table(cellText=J_df.values,
                     colLabels=J_df.columns,
                     rowLabels=J_df.index,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    plt.title("Matrix J_expanded with Sample Values (S)", fontsize=12, weight='bold')
    plt.show()

    # Plot B vector separately
    B_df = pd.DataFrame(B.T, columns=coef_names)
    fig2, ax2 = plt.subplots(figsize=(min(20, 1 + B_df.shape[1]), 2))
    ax2.axis('off')
    ax2.axis('tight')

    table2 = ax2.table(cellText=B_df.values,
                       colLabels=B_df.columns,
                       rowLabels=["B (Coefficients)"],
                       loc='center',
                       cellLoc='center')

    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.2, 1.2)

    plt.title("Soil Type Coefficients (B)", fontsize=12, weight='bold')
    plt.show()

# ==================
# Example usage:
# ==================
# فرض کن J_expanded, S, B, coef_names آماده هستند

plot_matrix_table(J_expanded, S, B, coef_names)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ========= ماتریس و ضرایب =========
def build_J_and_S_mean_model(data):
    soil_types = sorted(data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    zone_means = data.groupby(['Field_number', 'TYPE_2'])['log_Om'].mean().to_dict()

    S_values, J_rows = [], []
    for _, row in data.iterrows():
        field = row['Field_number']
        soil = row['TYPE_2']
        if (field, soil) not in zone_means:
            continue
        zone_mean = zone_means[(field, soil)]
        row_vector = np.zeros(len(soil_types))
        row_vector[soil_type_to_index[soil]] = zone_mean
        J_rows.append(row_vector)
        S_values.append(row['log_Om'])

    S = np.array(S_values).reshape(-1, 1)
    J = np.array(J_rows)
    intercept = np.ones((J.shape[0], 1))
    J_expanded = np.hstack([intercept, J])
    coef_names = ['Intercept'] + soil_types

    return J_expanded, S, coef_names

# ========= Scenario A: Full Dataset =========
def run_full_dataset_model(merged_data):
    J_expanded, S, coef_names = build_J_and_S_mean_model(merged_data)
    B, _, _, _ = np.linalg.lstsq(J_expanded, S, rcond=None)
    preds = J_expanded @ B
    rmse = np.sqrt(mean_squared_error(S, preds))

    print("\n🟢 Scenario A: Full Dataset")
    for name, value in zip(coef_names, B.flatten()):
        print(f"{name}: {value:.6f}")
    print(f"\nFull Dataset RMSE: {rmse:.6f}")
    return J_expanded, S, B, coef_names, rmse

# ========= Scenario B: LOFO-CV =========
def run_lofo_cv(merged_data):
    all_preds = []
    all_truths = []
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    for test_field in merged_data['Field_number'].unique():
        train_data = merged_data[merged_data['Field_number'] != test_field]
        test_data = merged_data[merged_data['Field_number'] == test_field]

        # zone mean از train
        zone_means = train_data.groupby(['Field_number', 'TYPE_2'])['log_Om'].mean().to_dict()

        # ساخت مدل روی train
        S_train, J_train_rows = [], []
        for _, row in train_data.iterrows():
            field = row['Field_number']
            soil = row['TYPE_2']
            if (field, soil) not in zone_means:
                continue
            zone_mean = zone_means[(field, soil)]
            row_vector = np.zeros(len(soil_types))
            row_vector[soil_type_to_index[soil]] = zone_mean
            J_train_rows.append(row_vector)
            S_train.append(row['log_Om'])

        if len(S_train) == 0:
            continue

        S_train = np.array(S_train).reshape(-1, 1)
        J_train = np.array(J_train_rows)
        J_train_expanded = np.hstack([np.ones((J_train.shape[0], 1)), J_train])
        B, _, _, _ = np.linalg.lstsq(J_train_expanded, S_train, rcond=None)

        # پیش‌بینی روی test
        J_test_rows, S_test = [], []
        for _, row in test_data.iterrows():
            field = row['Field_number']
            soil = row['TYPE_2']
            # zone mean برای test field
            test_mean = test_data[test_data['TYPE_2'] == soil]['log_Om'].mean()
            row_vector = np.zeros(len(soil_types))
            row_vector[soil_type_to_index[soil]] = test_mean
            J_test_rows.append(row_vector)
            S_test.append(row['log_Om'])

        if len(J_test_rows) == 0:
            continue

        S_test = np.array(S_test).reshape(-1, 1)
        J_test = np.array(J_test_rows)
        J_test_expanded = np.hstack([np.ones((J_test.shape[0], 1)), J_test])
        preds = J_test_expanded @ B

        all_preds.extend(preds.flatten())
        all_truths.extend(S_test.flatten())

    rmse = np.sqrt(mean_squared_error(all_truths, all_preds))
    print(f"\n🟡 Scenario B: LOFO-CV RMSE: {rmse:.6f}")
    return rmse

# ========= Visualization =========
def plot_matrix_table(J_expanded, S, B, coef_names):
    J_df = pd.DataFrame(J_expanded, columns=coef_names)
    J_df['S'] = S.flatten()
    J_df.index = [f"Sample_{i+1}" for i in range(J_df.shape[0])]

    fig, ax = plt.subplots(figsize=(min(20, 1 + J_df.shape[1]), min(1 + J_df.shape[0]*0.5, 20)))
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=J_df.values,
                     colLabels=J_df.columns,
                     rowLabels=J_df.index,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title("Matrix J_expanded with Sample Values (S)", fontsize=12, weight='bold')
    plt.show()

    B_df = pd.DataFrame(B.T, columns=coef_names)
    fig2, ax2 = plt.subplots(figsize=(min(20, 1 + B_df.shape[1]), 2))
    ax2.axis('off')
    ax2.axis('tight')

    table2 = ax2.table(cellText=B_df.values,
                       colLabels=B_df.columns,
                       rowLabels=["B (Coefficients)"],
                       loc='center',
                       cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.2, 1.2)
    plt.title("Soil Type Coefficients (B)", fontsize=12, weight='bold')
    plt.show()

# ========= اجرا =========
def run_both_scenarios_mean_model(merged_data):
    print("📘 Running Mean Model (No Normalization)")
    J_expanded, S, B, coef_names, rmse_full = run_full_dataset_model(merged_data)
    rmse_lofo = run_lofo_cv(merged_data)
    plot_matrix_table(J_expanded, S, B, coef_names)
    print("\n✅ Summary:")
    print(f"RMSE (Full Dataset): {rmse_full:.6f}")
    print(f"RMSE (LOFO-CV):       {rmse_lofo:.6f}")
run_both_scenarios_mean_model(merged_data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ========= ماتریس و ضرایب =========
def build_J_and_S_mean_model(data):
    soil_types = sorted(data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}
    zone_means = data.groupby(['Field_number', 'TYPE_2'])['log_Om'].mean().to_dict()

    S_values, J_rows = [], []
    for _, row in data.iterrows():
        field = row['Field_number']
        soil = row['TYPE_2']
        if (field, soil) not in zone_means:
            continue
        zone_mean = zone_means[(field, soil)]
        row_vector = np.zeros(len(soil_types))
        row_vector[soil_type_to_index[soil]] = zone_mean
        J_rows.append(row_vector)
        S_values.append(row['log_Om'])

    S = np.array(S_values).reshape(-1, 1)
    J = np.array(J_rows)
    intercept = np.ones((J.shape[0], 1))
    J_expanded = np.hstack([intercept, J])
    coef_names = ['Intercept'] + soil_types

    return J_expanded, S, coef_names

# ========= Scenario A: Full Dataset =========
def run_full_dataset_model(merged_data):
    J_expanded, S, coef_names = build_J_and_S_mean_model(merged_data)
    B, _, _, _ = np.linalg.lstsq(J_expanded, S, rcond=None)
    preds = J_expanded @ B
    rmse = np.sqrt(mean_squared_error(S, preds))
    r2 = r2_score(S, preds)

    print("\n🟢 Scenario A: Full Dataset")
    for name, value in zip(coef_names, B.flatten()):
        print(f"{name}: {value:.6f}")
    print(f"\nFull Dataset RMSE: {rmse:.6f}")
    print(f"Full Dataset R²:   {r2:.6f}")
    return J_expanded, S, B, coef_names, rmse, r2

# ========= Scenario B: LOFO-CV =========
def run_lofo_cv(merged_data):
    all_preds = []
    all_truths = []
    soil_types = sorted(merged_data['TYPE_2'].unique())
    soil_type_to_index = {soil: idx for idx, soil in enumerate(soil_types)}

    for test_field in merged_data['Field_number'].unique():
        train_data = merged_data[merged_data['Field_number'] != test_field]
        test_data = merged_data[merged_data['Field_number'] == test_field]
        zone_means = train_data.groupby(['Field_number', 'TYPE_2'])['log_Om'].mean().to_dict()

        S_train, J_train_rows = [], []
        for _, row in train_data.iterrows():
            field = row['Field_number']
            soil = row['TYPE_2']
            if (field, soil) not in zone_means:
                continue
            zone_mean = zone_means[(field, soil)]
            row_vector = np.zeros(len(soil_types))
            row_vector[soil_type_to_index[soil]] = zone_mean
            J_train_rows.append(row_vector)
            S_train.append(row['log_Om'])

        if len(S_train) == 0:
            continue

        S_train = np.array(S_train).reshape(-1, 1)
        J_train = np.array(J_train_rows)
        J_train_expanded = np.hstack([np.ones((J_train.shape[0], 1)), J_train])
        B, _, _, _ = np.linalg.lstsq(J_train_expanded, S_train, rcond=None)

        J_test_rows, S_test = [], []
        for _, row in test_data.iterrows():
            soil = row['TYPE_2']
            test_mean = test_data[test_data['TYPE_2'] == soil]['log_Om'].mean()
            row_vector = np.zeros(len(soil_types))
            row_vector[soil_type_to_index[soil]] = test_mean
            J_test_rows.append(row_vector)
            S_test.append(row['log_Om'])

        if len(J_test_rows) == 0:
            continue

        S_test = np.array(S_test).reshape(-1, 1)
        J_test = np.array(J_test_rows)
        J_test_expanded = np.hstack([np.ones((J_test.shape[0], 1)), J_test])
        preds = J_test_expanded @ B

        all_preds.extend(preds.flatten())
        all_truths.extend(S_test.flatten())

    rmse = np.sqrt(mean_squared_error(all_truths, all_preds))
    r2 = r2_score(all_truths, all_preds)

    print(f"\n🟡 Scenario B: LOFO-CV RMSE: {rmse:.6f}")
    print(f"LOFO-CV R²:                {r2:.6f}")
    return rmse, r2

# ========= Visualization =========
def plot_matrix_table(J_expanded, S, B, coef_names):
    J_df = pd.DataFrame(J_expanded, columns=coef_names)
    J_df['S'] = S.flatten()
    J_df.index = [f"Sample_{i+1}" for i in range(J_df.shape[0])]

    fig, ax = plt.subplots(figsize=(min(20, 1 + J_df.shape[1]), min(1 + J_df.shape[0]*0.5, 20)))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=J_df.values,
                     colLabels=J_df.columns,
                     rowLabels=J_df.index,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title("Matrix J_expanded with Sample Values (S)", fontsize=12, weight='bold')
    plt.show()

    B_df = pd.DataFrame(B.T, columns=coef_names)
    fig2, ax2 = plt.subplots(figsize=(min(20, 1 + B_df.shape[1]), 2))
    ax2.axis('off')
    ax2.axis('tight')
    table2 = ax2.table(cellText=B_df.values,
                       colLabels=B_df.columns,
                       rowLabels=["B (Coefficients)"],
                       loc='center',
                       cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.2, 1.2)
    plt.title("Soil Type Coefficients (B)", fontsize=12, weight='bold')
    plt.show()

# ========= اجرا =========
def run_both_scenarios_mean_model(merged_data):
    print("📘 Running Mean Model (No Normalization)")
    J_expanded, S, B, coef_names, rmse_full, r2_full = run_full_dataset_model(merged_data)
    rmse_lofo, r2_lofo = run_lofo_cv(merged_data)
    plot_matrix_table(J_expanded, S, B, coef_names)

    print("\n✅ Summary:")
    print(f"RMSE (Full Dataset): {rmse_full:.6f}")
    print(f"R²   (Full Dataset): {r2_full:.6f}")
    print(f"RMSE (LOFO-CV):      {rmse_lofo:.6f}")
    print(f"R²   (LOFO-CV):      {r2_lofo:.6f}")
run_both_scenarios_mean_model(merged_data)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- مرحله 1: خواندن و آماده‌سازی داده‌ها ---
df_path = '/content/fullMainData.xlsx'
data = pd.read_excel(df_path, sheet_name='mainData')
data['log_Om'] = np.log10(data['Om_p'])
data['Field_number'] = 'F' + data['Field_no'].astype(str)
data['Field_number_SS'] = data['Field_number'] + '  ' + data['TYPE_2']
data['Avelog_SS'] = data.groupby('Field_number_SS')['log_Om'].transform('mean')
data['Avelog_f'] = data.groupby('Field_number')['log_Om'].transform('mean')

# خواندن اطلاعات مساحت
df_area = '/content/Area.xlsx'
area = pd.read_excel(df_area, sheet_name='1')
total_area_per_land = area.groupby('FIELD')['Shape_Area'].sum()
grouped_data = area.groupby(['FIELD', 'TYPE_2'])['Shape_Area'].sum().reset_index()
grouped_data = pd.merge(grouped_data, total_area_per_land, left_on='FIELD', right_index=True, suffixes=('_soil', '_total'))
grouped_data.rename(columns={'FIELD': 'Field_number'}, inplace=True)
grouped_data['Field_number'] = grouped_data['Field_number'].astype(str)
grouped_data['Field_number'] = grouped_data['Field_number'].apply(lambda x: 'F' + x)
grouped_data['Percentage_area'] = grouped_data['Shape_Area_soil'] / grouped_data['Shape_Area_total']

# ادغام داده‌هاتم
merged_data = pd.merge(data, grouped_data, on=['Field_number', 'TYPE_2'], how='left')
merged_data['Area%'] = merged_data['Percentage_area']

# --- مرحله 2: ساخت دیتافریم نهایی با Avelog_SS ---
unique_soils_dict = {}
for field_number in merged_data['Field_number'].unique():
    temp_df = merged_data[merged_data['Field_number'] == field_number]
    unique_soils = temp_df['TYPE_2'].unique()
    unique_soils_dict[field_number] = unique_soils

new_rows = []
for field_number, unique_soils in unique_soils_dict.items():
    for soil in unique_soils:
        temp_df = merged_data[(merged_data['Field_number'] == field_number) & (merged_data['TYPE_2'] == soil)]
        new_row = {
            'Field_number': field_number,
            'TYPE_2': soil,
            'Avelog_SS': temp_df['Avelog_SS'].iloc[0],
            'Area%': temp_df['Area%'].iloc[0]
        }
        new_rows.append(new_row)
new_df = pd.DataFrame(new_rows)

# --- مرحله 3: محاسبه اختلاف میانگین‌ها ---
def calculate_soil_diffs(data_frame):
    soil_diffs_per_land = {}
    for land, group in data_frame.groupby('Field_number'):
        soils = group['TYPE_2'].unique()
        soil_diffs = {}
        for i in range(len(soils)):
            for j in range(i + 1, len(soils)):
                soil1, soil2 = soils[i], soils[j]
                avg1 = group[group['TYPE_2'] == soil1]['Avelog_SS'].iloc[0]
                avg2 = group[group['TYPE_2'] == soil2]['Avelog_SS'].iloc[0]
                diff = avg1 - avg2
                key = f"{soil1}-{soil2}"
                soil_diffs[key] = diff
        soil_diffs_per_land[land] = soil_diffs
    return soil_diffs_per_land

def find_differences(new_soils, soil_diffs_per_land, new_land):
    difference_vector = []
    land_info = []
    for i in range(len(new_soils)):
        for j in range(i + 1, len(new_soils)):
            soil1, soil2 = new_soils[i], new_soils[j]
            diff = new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil1)]['Avelog_SS'].iloc[0] - \
                   new_df[(new_df['Field_number'] == new_land) & (new_df['TYPE_2'] == soil2)]['Avelog_SS'].iloc[0]
            difference_vector.append(diff)
            land_info.append((new_land, soil1, soil2))

    for land, diffs in soil_diffs_per_land.items():
        if land == new_land:
            continue
        for diff_key, diff_value in diffs.items():
            if all(soil in new_soils for soil in diff_key.split('-')):
                soil1, soil2 = diff_key.split('-')
                difference_vector.append(diff_value)
                land_info.append((land, soil1, soil2))

    return difference_vector, land_info

def add_one_to_first_row(difference_vector):
    return np.insert(difference_vector, 0, 1).reshape(-1, 1)

def fill_difference_matrix(difference_vector, land_info, new_soils, new_df):
    num_rows = len(difference_vector) + 1
    num_cols = len(new_soils)
    difference_matrix = np.zeros((num_rows, num_cols))
    avg_area = new_df['Area%'].mean()

    for j, soil in enumerate(new_soils):
        soil_area = new_df[(new_df['Field_number'] == land) & (new_df['TYPE_2'] == soil)]['Area%'].iloc[0]
        if pd.isnull(soil_area):
            soil_area = avg_area
        difference_matrix[0, j] = soil_area
    difference_matrix[0, :] /= difference_matrix[0, :].sum()

    for i, diff_value in enumerate(difference_vector):
        soil1, soil2 = land_info[i][1], land_info[i][2]
        soil1_index = new_soils.index(soil1) if soil1 in new_soils else None
        soil2_index = new_soils.index(soil2) if soil2 in new_soils else None
        if soil1_index is not None:
            difference_matrix[i + 1, soil1_index] = 1
        if soil2_index is not None:
            difference_matrix[i + 1, soil2_index] = -1
    return difference_matrix, land_info

# --- مرحله 4: ساخت ماتریس‌ها و حل سیستم ---
soil_diffs_per_land = calculate_soil_diffs(new_df)
all_difference_vectors = {}
all_land_info_vectors = {}
all_difference_matrices = {}
all_land_info_matrices = {}
all_solutions_with_Avelog_f = {}

for land in new_df['Field_number'].unique():
    new_soils = list(new_df[new_df['Field_number'] == land]['TYPE_2'].unique())
    difference_vector, land_info = find_differences(new_soils, soil_diffs_per_land, land)
    vertical_vector = add_one_to_first_row(difference_vector)
    matrix, info_matrix = fill_difference_matrix(difference_vector, land_info, new_soils, new_df)

    d = vertical_vector
    solution = np.linalg.lstsq(matrix, d, rcond=None)[0]

    avelog_f_value = merged_data.loc[merged_data['Field_number'] == land, 'Avelog_f'].values[0]
    solution_with_Avelog_f = [value * avelog_f_value for value in solution]
    #solution_with_Avelog_f = solution
    all_solutions_with_Avelog_f[land] = solution_with_Avelog_f

# --- مرحله 5: ساخت دیتافریم نهایی و تحلیل ---
data = []
for land, solution_with_Avelog_f in all_solutions_with_Avelog_f.items():
    for soil_type, value in zip(new_df[new_df['Field_number'] == land]['TYPE_2'].unique(), solution_with_Avelog_f):
        row = {'Field_number': land, 'TYPE_2': soil_type, 'Calculated_value': value[0]}
        data.append(row)
result_df = pd.DataFrame(data)

# ادغام با merged_data
merged_data['predictedWithout'] = merged_data.apply(
    lambda row: result_df[(result_df['Field_number'] == row['Field_number']) & (result_df['TYPE_2'] == row['TYPE_2'])]['Calculated_value'].values[0]
    if not result_df[(result_df['Field_number'] == row['Field_number']) & (result_df['TYPE_2'] == row['TYPE_2'])].empty else None,
    axis=1
)

# حذف ردیف‌هایی که NaN دارند
filtered = merged_data.dropna(subset=['log_Om', 'predictedWithout'])

# اگر ردیفی باقی نماند، هشدار بده
if filtered.empty:
    print("⚠️ All rows have NaN values after filtering.")
else:
    # ادامه محاسبه
    X = filtered['log_Om']
    y = filtered['predictedWithout']
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    r_squared = model.rsquared
    y_pred = model.predict(X_with_const)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print("R² =", r_squared, "| RMSE =", rmse)


# رسم نمودار و محاسبه خطا
X = merged_data['log_Om']
y = merged_data['predictedWithout']
X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
r_squared = model.rsquared
y_pred = model.predict(X_with_const)
rmse = np.sqrt(np.mean((y - y_pred) ** 2))

plt.figure(figsize=(10, 6))
sns.regplot(x='log_Om', y='predictedWithout', data=merged_data, scatter_kws={'s': 10}, line_kws={'color': 'red'}, ci=None)
plt.xlabel('Log_SOM')
plt.ylabel('Predicted Values')
plt.title('Pairwise (with Avelog_SS differences)')
plt.grid(True)
plt.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nRMSE = {rmse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel, wilcoxon

# فرض: merged_data موجود است و دارای ستون‌های زیر:
# Field_number, log_Om, predictedWithout (مدل تفاوت جفتی)

# ایجاد پیش‌بینی‌های مدل زونی (میانگین هر zone برای هر نوع خاک در زمین)
zonal_means = merged_data.groupby(['Field_number', 'TYPE_2'])['log_Om'].transform('mean')
merged_data['predictedZonal'] = zonal_means

# محاسبه RMSE برای هر زمین و هر مدل
field_rmse = []

for field in merged_data['Field_number'].unique():
    subset = merged_data[merged_data['Field_number'] == field]
    rmse_without = np.sqrt(np.mean((subset['log_Om'] - subset['predictedWithout']) ** 2))
    rmse_zonal = np.sqrt(np.mean((subset['log_Om'] - subset['predictedZonal']) ** 2))
    field_rmse.append({
        'Field_number': field,
        'RMSE_Pairwise': rmse_without,
        'RMSE_Zonal': rmse_zonal
    })


rmse_df = pd.DataFrame(field_rmse)

# تست t-زوجی برای بررسی تفاوت میان دو مدل
t_stat, p_ttest = ttest_rel(rmse_df['RMSE_Pairwise'], rmse_df['RMSE_Zonal'])

# تست Wilcoxon برای بررسی مقاوم‌تر تفاوت‌ها
w_stat, p_wilcoxon = wilcoxon(rmse_df['RMSE_Pairwise'], rmse_df['RMSE_Zonal'])

# نمایش نتایج
print("RMSE Summary by Model:")
print(rmse_df.describe())

print("\nT-Test (Paired):")
print(f"t = {t_stat:.3f}, p-value = {p_ttest:.4f}")

print("\nWilcoxon Test:")
print(f"statistic = {w_stat:.3f}, p-value = {p_wilcoxon:.4f}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) خواندن فایل (اسم فایل خودت را بگذار)
df = pd.read_excel("/content/fullMainData.xlsx", sheet_name="mainData")
print(df.columns.tolist())


# 2) فقط رکوردهایی که SOM (ستون Om_p) دارند
df = df.dropna(subset=["Om_p"])

# 3) اگر دوست داری اسم فیلدها مثل F26, F84 باشد:
df["Field"] = "F" + df["Field_no"].astype(int).astype(str)

# 4) مرتب‌کردن فیلدها بر اساس میانه SOM (برای شبیه شدن به شکل شما)
order = (
    df.groupby("Field")["Om_p"]
      .median()
      .sort_values(ascending=False)
      .index
)

# 5) رسم نمودار جعبه‌ای با فونت بزرگ‌تر
plt.figure(figsize=(20, 6))
sns.boxplot(data=df, x="Field", y="Om_p", order=order)

plt.xlabel("Field", fontsize=16)
plt.ylabel("SOM (%)", fontsize=16)
plt.title("Distribution of SOM (%) across fields", fontsize=18)

plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

