# 2.Kategorik Değişken Analizi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df['embarked'].value_counts()
df['sex'].unique()
df['sex'].nunique()

df.info()

# Kategorik görünmeyen survived, pclass gibi sınıfları temsil eden değişkenler de kategoriktir.
# çünkü kadın ve erkek gibi yaşama veya yaşamama sınıfını temsil eder.

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
# str(df[col].dtypes bu kısmı önemli

# survived, pclass gibi aslında kategorik olanlara karşı belli bir unique sınıftan azsa kategoriktir yaklaşımı
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype in ["int64", "float64"]]

# kardiniltesi yüksek değerler var mı diye bakıyoruz. yani anlam taşımayacak kadar unique olan isim+soyisim gibi
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtype) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

# fonksiyon hali yaptıklarımız

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)  # sınıfların yüzdelik karşılığı


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(df)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)
# plot özelliği ekledik ama int olmayanlarla çalışmıyor
for col in cat_cols:
    if df[col].dtype == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

# int olmayanlarla olan sorunu halletmeye çalışıyoruz
df["adult_male"].astype(int)
