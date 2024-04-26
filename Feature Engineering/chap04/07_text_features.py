##################################################################
# Metinler Üzerinden Özellik Türetme
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


df = load()
df.head()

####################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

####################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(x.split(" ")))
df.head()

##########################
# Özel Yapıları Yakalamak
#########################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

#################################
# Regex ile Değişken Türetmek
################################
# Düzenli ifadeler ile değişken türetmek bir pattern yakalamaya çalışıyoruz.

df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
df.head()
# .str.extract() fonksiyonu, bir dize içinde belirli bir desene göre eşleşen kısımları ayıklamak için kullanılır.

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})
