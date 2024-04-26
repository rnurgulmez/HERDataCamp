##################################################################
# Özellik Ölçeklendirme
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

# Özellik Ölçeklendirme : modellerin değişkenlere eşit şartlar altında yaklaşması

############################################
# StardardScaler : Klasik standartlaştırma.Normalleştirme. Ortalamayı çıkar, standart sapmaya böl z = (x - u) / s
###########################################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

############################################
# RobustScaler : Medyanı çıkar iqr'a böl. Aykırı değerlere karşı daha dirençli
###########################################

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

############################################
# MinMaxScaler : Verilen 2 değer arasında değişken dönüşümü
###########################################

mms = MinMaxScaler() # 0 ile 1 arasında
df["Age_minmax_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

# sadece gözlemlemek amacıyla num_summary fonk yazıldı
# ve dağılımın değil sadece ölçeklendirme tarzının değiştiğini gördük
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)

############################################
# Numeric to Categorical
# Binning
###########################################

df["Age_qcut"] = pd.qcut(df["Age"], q=5) # istersek bu aralıklara label da girebiliriz
# qcut methodu değişkenin değerlerini küçükten büyüğe sıralar ve 5 parçaya (istediğimiz q sayısına bağlı olarak) böler.
df.head()