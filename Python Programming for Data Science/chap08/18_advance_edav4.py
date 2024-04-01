# 4.Hedef Değişken Analizi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtype == "bool":
        df[col] = df[col].astype("int64")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkneler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

   Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_car cat_cols'un içerisinde
    """
    ####################################################################################################################
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    ####################################################################################################################
    # sayısal olanlar
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    # numerik görünüp numerik olmayanlar
    num_cols = [col for col in num_cols if col not in cat_cols]
    ####################################################################################################################
    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))
    print("##########################################")

# Hedef değişken : survived
cat_summary(df, "survived")

##########################################################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
##########################################################
df.groupby("sex")["survived"].mean()
# aynı şeyi fonksiyon ile yaparsak
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df,"survived", "sex")
target_summary_with_cat(df,"survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

##########################################################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
##########################################################
# kategoriğin tersi olarak groupby'a bağımlı değişkeni getirir, agg'a sayısal değişkeni getiririz
# bu sayede kadın mı erkek mi hayatta kaldı sorusu yerine
# hayatta kalanlar kaç yaşındaydı gibi bir soruya cevap bulunmuş olur
df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age": "mean"})
# aynısını fonksiyon ile yapalım
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)