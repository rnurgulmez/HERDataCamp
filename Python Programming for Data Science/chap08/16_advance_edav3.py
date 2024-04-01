# 3.Sayısal Değişken Analizi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

####################################################################################################################
# kategorik olanlar
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
####################################################################################################################

####################################################################################################################
# sayısal olanlar
num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
# numerik görünüp numerik olmayanlar vardı
num_cols = [col for col in num_cols if col not in cat_cols]
####################################################################################################################

# Sayısal Değişkenleri Bulmak İçin Analiz Fonksiyonu
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)