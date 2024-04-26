##################################################################
# Encoding : değişkenlerin temsil şekillerinin değiştirilmesi
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


# Label Encoding : male female labellerı kadın veya değil yani 0 veya 1'e çevirmek.
# One Hot Encoding : aralarında büyüklük küçüklük ilişkisi olmayan durumlarda gerekir.
# Rare Encoding

############################
# Label (Binary) Encoding
############################

df = load()
df.head()

df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]  # ilk gördüğüne alfabetik sıraya göre 0, diğerine 1 verir
# fit = encoding uygula, transform = dönüştür
le.inverse_transform([0, 1])  # sınıfların ne olduğunu öğrenmek için


def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()
# label encoder'ı get_dummies() ile de yapabiliriz drop_first parametresi ile
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]
# len(df[col].unique) yaparsak NaN olanları da sınıf sayar bu hataya sebep olur nunique kullanılmalı

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()  # EMERGENCYSTATE_MODE değişkeninde NaN değerlere 2 atadı buna dikkat etmeliyiz

df = load()
df.value_counts()
df["Embarked"].nunique()  # nunique'te eksik değer saymaz 3 sınıf var der
df["Embarked"].unique()  # NaN'la beraber 4 sınıf var der

############################
# One Hot Encoding
############################

# GS -> 1 0 0 0 0 0
# FB -> 0 1 0 0 0 0
# BJK -> 0 0 1 0 0 0

df = load()
df.head()
df["Embarked"].value_counts()  # S C Q sınıfları var ve bu sınıflar arasında fark yok bu sebeple OHE uyguluyoruz

pd.get_dummies(df, columns=["Embarked"], dtype=int).head()
# dummy değişken tuzağı = değişkenlerin birbiri üzerinden üretilebiliyor olması
# kurtulmak için drop_first=True ilk sınıfı (alfabeye göre) düşürür C Q S'den C düşer
pd.get_dummies(df, columns=["Embarked"], dtype=int, drop_first=True).head()
pd.get_dummies(df, columns=["Embarked"], dtype=int, dummy_na=True).head()  # NaN da sınıf olarak gelsin seçeneği

# drop_first sayesinde label yani 2 sınıflı işlemlerden birini düşüreceği için hem OHE hem LE yapmış oluruz
pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int, drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dtype=int, drop_first=drop_first)
    return dataframe


df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# bütün kategorik değşkenleri sokmak tehlikeli survived hedef değişkeni o girmemeli

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()  # kalıcı olarak df'e atmadık

############################
# Rare Encoding
############################

# çok az gözlemlenen değerler a'dan 56 b'den 100 c'den 2 tane varsa bu bir sorun
# rare değerleri (belirli bir eşiğe göre) bir araya getirilir.

# Adımlar
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmes
# 3. Rare encoder yazılması

# 1
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# 2
# kategorik değişkenleri target açısından değerlendirme
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", cat_cols)

# 3


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp > rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()