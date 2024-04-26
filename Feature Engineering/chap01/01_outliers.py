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

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)


def load_application_train():
    # büyük ölçekli veri seti
    data = pd.read_csv("datasets/application_train.csv")
    return data


df = load_application_train()
df.head()


def load():
    # küçük ölçekli veri seti
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

# Aykırı Değerleri Yakalama

# Grafik Teknikle Aykırı Değer : boxplot
# 2.olarak sayısal değişkenlerde aykırı değerleri histogram ile görebiliriz.

sns.boxplot(x=df["Age"])  # bir sayısal değişkenin dağılım grafiğini verir
plt.show()

# Aykırı değerler nasıl yakalanır?
q1 = df["Age"].quantile(0.25)
q1
q3 = df["Age"].quantile(0.75)
q3

iqr = q3 - q1
iqr

up = q3 + 1.5 * iqr  # 64.81
low = q1 - 1.5 * iqr  # -6.68

df[(df["Age"] < low) | (df["Age"] > up)].shape

# Aykırı değer var mı yok mu?
df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)

df[~((df["Age"] > up) | (df["Age"] < low))].any(axis=None)


# İşlemleri Fonksiyonlaştırmak
# Eşik değerleri hesaplayan fonksiyonumuz
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Age")


# Aykırı değer olup olmadığını kontrol eden fonksiyon
def check_outliers(dataframe, col_name):
    # eğer burada q1 ve q3'ü seçebilmek istiyorsak parametre olarak girilmeli, fakat istemiyoruz.
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


check_outliers(df, "Age")

#####################
# grab_col_names
#####################

dff = load_application_train()
dff.head()


# Çok fazla değişken olduğunda numerik, kategorik olan ama
# aslında numerik, numerik ama aslında cardinalleri yakalayamalıyız
# onun için bir fonksiyon yazacağız.

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


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outliers(dff, col))


# aykırı değerleri yakalma
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


grab_outliers(df, "Age")
age_index = grab_outliers(df, "Age", True)

#################################
# Aykırı Değer Problemini Çözme
#################################

# Silme

low, up = outlier_thresholds(df, "Fare")
df.shape
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe["Fare"] < low) | (dataframe["Fare"] > up))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outliers(df, col)

df.shape[0] - new_df.shape[0]

# Baskılama : silme yöntemindeki gibi veri kaybı olmaması için tercih edilir.
# (re-assignment) eşik değerlerin üzerinde kalan değerler eşik değeri ile değiştirilir.

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]
# aynı şeyi loc ile de yapabiliriz
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]
# loc'u kullanarak hem satırlardan hem sütunlardan seçme yaptık

df.loc[(df["Fare"] > up), "Fare"] = up  # up değeleri aykırıların yeni değerleri olarak değiştirdik

df.loc[(df["Fare"] < low), "Fare"] = low


def replace_with_thereshold(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] > up), col_name] = up
    dataframe.loc[(dataframe[col_name] < low), col_name] = low

df = load()
df.shape

for col in num_cols: # aykırı değer var
    print(col, check_outliers(df, col))

for col in num_cols:
    replace_with_thereshold(df, col)

for col in num_cols:
    print(col, check_outliers(df, col)) # artık aykırı değer yok