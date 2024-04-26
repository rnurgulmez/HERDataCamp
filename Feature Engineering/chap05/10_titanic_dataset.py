##################################################################
# Titanic ile Uçtan Uca Feature Engineering & Data Preprocessing
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
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].empty == False:
        return True
    else:
        return False


def replace_with_thereshold(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] > up), col_name] = up
    dataframe.loc[(dataframe[col_name] < low), col_name] = low


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


def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = dataframe[var].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[var] = np.where(dataframe[var].isin(rare_labels), "Rare", dataframe[var])

    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dtype=int, drop_first=drop_first)
    return dataframe


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns] # bütün değişken isimlerini büyüttük

###################################################
# 1.Feature Engineering (Değişken Mühendisliği)
###################################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype(int)
# Name Count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# Name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(x.split(" ")))
# Name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# Name title
df["NEW_TITLE"] = df.NAME.str.extract(" ([A-Za-z]+)\.", expand=False)
# Family Size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# Age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# Is alone
df.loc[(df["SIBSP"] + df["PARCH"] > 0), "NEW_IS_ALONE"] = "NO"
df.loc[(df["SIBSP"] + df["PARCH"] == 0), "NEW_IS_ALONE"] = "YES"
# Age level
df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] = "senior"
# Sex x Age
# erkek olanlar
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & (df["AGE"] <= 50)), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"

# kadınlar
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & (df["AGE"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

##################################
# 2.Outliers (Aykırı Değerler)
##################################

for col in num_cols:                                # Aykırı değer var
    print(col, check_outliers(df, col))

for col in num_cols:
    replace_with_thereshold(df, col)

for col in num_cols:                               # Aykırı değer kalmadı
    print(col, check_outliers(df, col))

####################################
# 3.Missing Values (Eksik Değerler)
###################################

df.loc[df["NEW_AGE_CAT"] == "nan", "NEW_AGE_CAT"] = np.NaN # sonradan ekledik

missing_values_table(df)

# cabin değişkeni yerine cabin_bool değişkeni oluşturduğumuz için bunu düşürebiliriz.
df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, axis=1, inplace=True)
# yaşa bağlı olarak oluşan tüm eksik değerler tekrar oluşturup doldurmamızla gitmiş oldu.

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df["AGE"] > 56), "NEW_AGE_CAT"] = "senior"

# erkek olanlar
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df["SEX"] == "male") & ((df["AGE"] > 21) & (df["AGE"] <= 50)), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniormale"

# kadınlar
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df["SEX"] == "female") & ((df["AGE"] > 21) & (df["AGE"] <= 50)), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

missing_values_table(df)
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

####################################
# 4.Label Encoding
###################################

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "int32", "float64"]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

####################################
# 5.Rare Encoding
###################################

rare_analyser(df, "SURVIVED", cat_cols)
df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

####################################
# 6.One-Hot Encoding
###################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols) # burada 2 sınıfın oranları birbirine yakın mı bunu görmek istiyoruz.

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

####################################
# 7.Standart Scaler
###################################
# bu problemde gerekli değil

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#################
# 8.Model
################

y = df["SURVIVED"] # bağımlı değişken
X = df.drop(["SURVIVED"], axis=1) # bağımsız değişkenler.
# passenger_id yukarıda zaten silindiği için bir daha silmedik.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)