# Koşullu Seçimler
import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()  # yaşı 50'den büyük olan 5 kişiyi getirir
df[df["age"] > 50]["age"].count()  # yaşı 50'den büyük olan kaç kişi var sorusunun cevabı
# df[df["age"] > 50'den sonra ["age"] demezsek tüm özellikleri sayar

df.loc[df["age"] > 50, ["age", "class"]].head()  # yaşı 50'den büyük olanların sınıf bilgisi döner
# 1 koşul 2 sınıf seçtik

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50)
                & (df["sex"] == "male")
                & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")), ["age", "class",
                                                                                                "embark_town"]]  # birden fazla koşul

df_new["embark_town"].value_counts()

# Toplulaştırma ve Gruplama : hep bir arada olur

# Toplulaştırma : özet istatistikler veren fonksiyonlardır
# group by ile hepsi kullanılabilir. pivot table hariç
# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns

pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()
# yaş ortalaması
df["age"].mean()
# cinsiyete göre yaş ortalaması
df.groupby("sex")["age"].mean()  # df'i cinsiyete göre grupla ve yaş değişkeninin ortalamasını al demektir.

df.groupby("sex").agg({"age": "mean"})  # bunu kullanmak daha iyi buna alış

df.groupby("sex").agg({"age": ["mean", "sum"]})  # çünkü bu şekilde birden fazla işlemi liste olarak gönderebiliriz.

df.groupby("sex").agg({"age": ["mean", "sum"],
                       # "embark_town": "count",
                       # bu kısım olmaz cinsiyete göre sayı verdi anlamlı değil pivot table ihtiyacı
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": "mean",
                                        "survived": "mean"})  # 2 seviyeden groupby yaptık daha fazla da olabilir.

df.groupby(["sex", "embark_town", "class"]).agg({"age": "mean",
                                                 "survived": "mean",
                                                 "sex": "count"})

