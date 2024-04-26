# eksikliğin rassallığı : eksik verinin rastsal olup olmaması önemli
# rastgele ise silebiriliz, ancak diğer değişkenler ile ilişkili ise silinmemeli.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


# Eksik Değerlerin Yakalanması / Analizi

def load():
    # küçük ölçekli veri seti
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

# eksik değer var mı yok mu sorgusu
df.isnull().values.any()

# değişkenlere göre kaç adet eksik değer var
df.isnull().sum()

# değişkenlerdeki tam değer sayısı
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()  # kendisinde en az 1 tane eksik değer olan tüm satırları aldığı için sonuç çok yüksek

# en az 1 tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().any(axis=1)]

# azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# eksikliğin bütün veri setine oranı
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# eksik değere sahip olanlar
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


# analiz ihtiyacımız olan 3 şey : eksikliğin frekansı, yüzdeleri, eksik olan değişkenlerin seçilmesi

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # concat ile bir df oluşturduk concat = birleştirme
    # axis=1 sütunlara göre birleştirme
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_cols


missing_values_table(df, na_name=True)

# Eksik Değer Problemini Çözme
#####################
# 1.Yöntem Silmek
####################
df.dropna().shape  # en az 1 tane bile null varsa o satırı siler

##############################################
# 2. Basit Atama Yöntemleri ile Doldurmak
##############################################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
# mean ile geçici olarak doldurduk ve sonuç 0 verdi yani eksik değer kalmadı

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()  # axis = 0 sütun

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

# Kategorik değişkenler mode ile doldurulabilir
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("mssing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# 10 ile sınırlama sebebimiz kategorik olduğu halde cardinal olabilir

# Kategorik Değişken Kırılımında Değer Atama

df.groupby("Sex")["Age"].mean()  # cinsiyete göre yaş ortalaması

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]  # cinsiyete göre yaş kırılımında kadınalrın yaş ortalaması

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female")]  # kadın olup yaş değişkeni eksik olanları getirdik

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
# loc[satır, sütun]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

######################################################################
# 3. Yöntem Tahmine Dayalı Atama ile Eksik Verilerin Doldurulması
######################################################################
# eksikliğe sahip olan değişken = bağımlı değişken, eksikliğe sahip olmayan değişkenler = bağımsız değişken

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True, dtype=int)

# 2 veya daha fazla sınıfı olan değişkenleri numerik olarak ifade etmek
# female male gibi olanlardan bir kategoriyi düşürüp(binary problemlerde)
# ilk sınıfını atacak 2.yi tutacak binary şekilde temsil edilir.

dff.head()

# değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn uyguluyoruz
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

# Eksik Değerlerin Yapısını İncelemek
msno.bar(df)  # tam değerlerin sayısını verir
plt.show()

msno.matrix(df)  # birbirine bağlı eksiklikler var mı bunu gözlemleyebiliriz.
plt.show()

msno.heatmap(df)
# 1 veya -1 yüksek ilişki 1 pozitif yönlü -1 negaitf daha düşük değerler anlamlı bir ilişki yok demektir
plt.show()

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
# bağımlı değişkenden kasıt hedef değişken hayatta kalıp kalamama hangi sebeplere bağlı gibi
missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_targeted(dataframe, target, na_cols):
    temp_df = dataframe.copy()
    for col in na_cols:
        # eksiklik olana 1, olmayana 0 yazar
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_targeted(df, "Survived", na_cols)
