##################################################################
# Özellik Etkileşimleri
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


df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
# yaşı küçük olduğu halde iyi bir sınıfa sahipse refah seviyesi hakkında bilgi verebilir yaptığımız bu işlem

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1 # aile üye sayısı

# Kategorik ve sayısal değişkenlerin de etkileşim noktalarına bakılarak flag atılabilir.
# erkek olanlar
df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

# kadınlar
df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()