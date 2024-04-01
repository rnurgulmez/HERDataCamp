import pandas as pd
import seaborn as sns

# kütüphane tanımlama
df = sns.load_dataset("titanic")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
# kadın ve erkek sayısını bulma
df["sex"].value_counts()
# her sütuna ait unique değerler sayısını bulma
df.nunique(dropna=False)  # number of unique, dropna=False nulları da saymasını sağladık
# pclass içindeki unique değerler sayısı
df["pclass"].nunique()
df["pclass"].unique() # array döner unique değerlerim içeriğine
# pclass ve parch içindeki unique değerler sayısı
df[['pclass', 'parch']].nunique()
# embarked değişken tipi
print(df["embarked"].dtype)  # object
# embarked değişkenin tipini category olarak değiştir
df["embarked"] = df["embarked"].astype("category")
print(df["embarked"].dtype)  # category
# embarked C olanların tüm bilgileri
df[df["embarked"] == "C"]
# loc ile de yapılabilir
df.loc[df["embarked"] == "C"]
# embarked'ı S olmayanların tüm bilgileri
df[df["embarked"] != "S"]
# yaşı 30'dan küçük kadınların tüm bilgileri
df[(df["age"] < 30) & (df["sex"] == "women")]  # parantezler önemli ()
# fare>500 ve yaşı>70
df[(df["age"] > 70) | (df["fare"] > 500)]
# her değişkendeki null değerlerin toplamı
df.isnull().sum()
# who değişkenini df'den çıkarma
df.drop("who", axis=1, inplace=True) # 1 kolon, 0 satır demek
# deck değişkenindeki nulları en çok tekrar eden değer ile doldurma
mode_of_deck = df["deck"].mode()[0]
df["deck"].fillna(mode_of_deck, inplace=True)
df
# age değişkenindeki boş değerleri medyanı ile doldurma
median_of_age = df["age"].median()
df["age"].fillna(median_of_age, inplace=True)
df
# survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerleri
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})


# 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın
# Yazdığımız fonksiyonu kullarak age_flag değişkeni oluştur(apply ve lambda kullanılacak)
def age_30(age):
    if age < 30:
        return 1
    else:
        return 0

# int(15 < 30)

# 2.yol
# def age_30_v2(age):
#   return int(age < 30)

df["age_flag"] = df["age"].apply(lambda x: int(x < 30))
# apply for döngüsüne ihtiyaç kalmadan age'in içerisinde gezmeyi sağlar
# 2.yol
# df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)
# 3.yol
# df["age_flag"] = df["age"].apply(lambda x: age_30(x))

# map ile
# df["age_flag"] = df["age"].map(age_30)

# apply ve fonksiyon ile
# df["age_flag"] = df["age"].apply(age_30)

# tips veri setini tanımlama
df = sns.load_dataset("tips")
df.head()

# time değişkeninin kategorilerine göre total_bill değerlerinin sum,min,max ve mean bul
df.groupby("time").agg({"total_bill": ["sum","min", "max", "mean"]})
# pivot table ile de yapabiliriz
df.pivot_table("total_bill", "time", aggfunc=["sum", "mean", "min", "max"])

# günlere ve time değişkeninin kategorilerine göre total_bill değerlerinin sum,min,max ve mean bul
df.groupby(["time", "day"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

# lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre min,max,sum ve mean bul
# 1.adım : lunch ve female filtrelemek
df[(df["sex"] == "Female") & (df["time"] == "Lunch")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                          "tip": ["sum", "min", "max", "mean"]})

#pivotla aynı işlemi yapmak için
df[(df["sex"] == "Female") & (df["time"] == "Lunch")].pivot_table(["total_bill", "tip"],
                                                                  "day",
                                                                  aggfunc=["sum", "min", "max", "mean"])

# size'ı 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması? (loc ile çözülmeli)
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean() # loc : label satır, sütun
# filtrelere uyan total_bill satırlarını getirir

# total_bill_tip_sum değişkeni oluştur, her müşterinin ödediği tip + sum gelsin
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

# total_bill_tip_sum'a göre büyükten küçüğe sırala ve ilk 30 kişiyi yeni bir df'ye ata
new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape