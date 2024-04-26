##################################################################
# Özellik Çıkarımı
##################################################################
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


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

# Özellik çıkarımı, ham veriden değişken türetmek demektir.
# Yapısal verilerden türetmek/ yapısal olmayan(görüntü, ses) verilerden türetmek

# Yapısal -> timestamp gibi var olan bir değişkenden ay, yıl, gün çıkarmak gibi
# Yapısal olmayan -> bir resmin piksellere ayrılması gibi


###############################################
# Binary Features: Flag, Bool, True-False
###############################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype(int) # dolu olanlara 1, boş olanlara 0 yazdık
df.head()

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"}) # kabin numarası olanların hayatta kalma oranı

# oran testi
from statsmodels.stats.proportion import proportions_ztest
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                       nobs=[df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

# yeni NEW_IS_ALONE değişkeni için oran testi
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                               df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                       nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
