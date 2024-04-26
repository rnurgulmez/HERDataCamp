##############################################
# Çok Değişkenli Aykırı Değer Analizi
##############################################

# 17 yaşında olmak aykırı değildir, 3 kere evlenmiş olmak aykırı değildir
# ancak bu 2 faktör bir aradaysa yani 17 yaşında 3 kere evlenmiş olmak aykırı değerdir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor  # çok değişkenli aykırı değer yakalama yöntemi
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler  # dönüştürme fonksiyonları


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outliers(dataframe, col_name):
    # eğer burada q1 ve q3'ü seçebilmek istiyorsak parametre olarak girilmeli, fakat istemiyoruz.
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != 0]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        # örnek sayısı 10'dan fazlasysa hepsini değil headini getirir.
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe["Fare"] < low) | (dataframe["Fare"] > up))]
    return df_without_outliers


def replace_with_thereshold(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] > up), col_name] = up
    dataframe.loc[(dataframe[col_name] < low), col_name] = low


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outliers(df, col))

low, up = outlier_thresholds(df, "carat")

df.shape

df[(df["carat"] < low) | (df["carat"] > up)].shape # değişkenlere tek tek bakıldığında çok fazla sayıda aykırı değer var
# ancak bir de çok değişkenli bakalım

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_ # skorları tutuyoruz
df_scores[0:5]
# df_scores = -df_scoreces
np.sort(df_scores)[0:5] # -1'e yakın olanlar en iyi, -1'den uzaklaştıkça outlier anlamına gelir

# eşik değeri belirlemek için : elbow yöntemi
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()
# en sert değişimin olduğu nokta eşik değeri belirlenir.

th = np.sort(df_scores)[3] # eşik değeri belirlendi

df[df_scores < th]

df[df_scores < th].shape # yalnızca 3 tane aykırı değer kaldı lof'tan sonra

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
# gözlem sayısı çok olduğunda baskılama yapmak mantıklı değil
