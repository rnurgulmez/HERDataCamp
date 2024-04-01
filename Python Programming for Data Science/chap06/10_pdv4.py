#############################
# Apply & Lambda
#############################
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

# Apply : satır ya da sütunlarda otomatik olarak fonk. çalıştırmayı sağlar.
# Lambda : kullan at fonksiyon

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5
# değişkenlere fonk. uygulamak istiyoruz ama bu şekilde tek tek yapamayız çok fazla değişken var
(df["age"] / 10).head()
(df["age2"] / 10).head()
(df["age3"] / 10).head()

for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df.head()

df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()
# standartlaştırma işlemi
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()


# df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.head()

#############################
# Birleştirme İşlemleri
#############################
import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
# 1.arguman veri yapısı(liste, sözlük, np array), 2.arguman değişken isimleri
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)
# ön tanımlı olarak index=0'dır satır bazında 1 yaparsak yan yana birleştirir

# Merge ile Birleştirme İşlemleri
df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"], # kaç farklı veri yapısı var? 5
                    "start_date": [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees") # özellikle nereden birleşmesi gerektiğini söylersek

df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({"group": ["accounting", "hr", "engineering"],
                    "manager": ["Caner", "Mustafa", "Berkcan"]})

pd.merge(df3, df4, on="group")

dict = {"Paris": [10], "Berlin": [20]}
pd.DataFrame(dict)