# Veri setinin okunması ve genel özellikler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/persona.csv")
# veri seti hakkında genel bilgiler
df.head()
df.info()
df.describe()
df.columns
df.isnull().sum()
# unique "source" adedi ve frekansları
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# kaç unique "PRICE" vardır?
df["PRICE"].nunique()
# hangi price'tan kaçar tane satış gerçekleşti
df["PRICE"].value_counts()
# hangi ülkeden kaçar tane satış olmuştur?
df["COUNTRY"].value_counts()
# ülkelere göre satışlardan toplam ne kadar kazanılmıştır?
df.groupby("COUNTRY")["PRICE"].sum()
# source türlerine göre satış sayıları?
df["SOURCE"].value_counts()
# ülkelere göre price ortalamaları
df.groupby("COUNTRY")["PRICE"].mean()
# sourcelara göre price ortalamaları
df.groupby("SOURCE")["PRICE"].mean()
# country-source kırılımında price ortalamaları
df.groupby(["SOURCE", "COUNTRY"])["PRICE"].mean()
# country-source-sex-age kırılımda ortalama kazançlar
df.groupby(["SOURCE", "COUNTRY", "SEX", "AGE"]).agg({"PRICE": "mean"}).head()
# çıktıyı price'a göre azalan olacak şekilde sırala ve agg_df'e kaydet
agg_df = df.groupby(["SOURCE", "COUNTRY", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)
agg_df.head()
# index'te yer alan isimleri değişken ismine çevir.
# Bir önceki sorudaki price hariç diğerleri index isimleri bu isimleri değişken isimlerine çevir.
# reser_index()
agg_df.shape # 348 değişken 1 kolon var aslında index olanlar değişken gibi kaldı
agg_df.reset_index(inplace=True)
agg_df.head()
agg_df.shape
# age değişkenini kategorik değişkene çevir ve agg_df'e ekle
agg_df["AGE"].min()
agg_df["AGE"].max()
# pd.cut() : veriyi belli sayılarda aralıklara böler.
# aralıkları belirleyelim
bin = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
# bölünen noktalara göre isimlendirmeler
mylabels = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())]
# age'i bölelim
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bin, mylabels)
agg_df.head()
pd.crosstab(agg_df["AGE"], agg_df["AGE_CAT"]) # hangi yaş aralığında kaç yaşında olan kaç kişi var bunu verir
# yeni seviye tabanlı müşterileri tanımlama ve customer_level_based değişkenini oluşturma
# ve tekilleştirmek gerekli örneğin USA_ANDROID_MALE_0_18 birden fazla iste groupby'a alıp ort. alınmalı
agg_df.head()
agg_df.columns
# gözlem değerlerine eriştik
for row in agg_df.values:
    print(row)
# birleştirme
agg_df["customer_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + str(row[5]) for row in agg_df.values]
# 2.yöntem : itterows
[row["COUNTRY"].upper() + "_" + row["SOURCE"].upper() + "_" + row["SEX"].upper() + "_" + str(row["AGE_CAT"]) for index, row in agg_df.iterrows()]
agg_df.head()

# artık diğerlerine ihtiyacımız olmadığı için yeni oluşturduğumuz ve price sütunu haricini düşürdük
agg_df1 = agg_df[["customer_level_based", "PRICE"]]
agg_df1.head()
agg_df1.shape
# aynı persone birden fazla kere geçiyor bunu düzeltmemiz lazım
agg_df1["customer_level_based"].value_counts()
# bu sebeple segmentelere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz
agg_df1 = agg_df1.groupby("customer_level_based").agg({"PRICE": "mean"})
# groupby yaptığımız değişken indexe geliyor bunu düzeltmeliyiz
agg_df1.reset_index(inplace=True) # indexten kolona çıktı tekrar
agg_df1.head()
# her personadan 1 tane olmalı
agg_df1["customer_level_based"].value_counts()
agg_df1.shape # 348 değişken vardı 109 tane kaldı tekilleştirmeden sonra

# yeni müşterileri segmentlere ayırın(price'a göre), segmentleri "SEGMENT" olarak agg_df'e ekle
agg_df1["SEGMENT"]=pd.qcut(agg_df1["PRICE"], 4, labels=["D", "C", "B", "A"]) # 4 segmente ayırdık müşterileri
# qcut : belirlenen sayıda bölmeyi sağlar ancak küçükten büyüğe böler o sebeple d c b a diye isimlendirdik
agg_df1.groupby("SEGMENT").agg({"PRICE": ["mean", "min", "max", "sum", "count"]}).sort_values(by="SEGMENT", ascending=False)
# yeni gelen müşterileri sınıflandırınız ve ne kadar gelir getirebileceğini tahmin ediniz
# 33 yaşında android kullanan bir Türk kadını hangi segmente ait ve ne kadar gelir kazandırabilir?
new_user = "TUR_ANDROID_FEMALE_(30_40]"
agg_df1[agg_df1["customer_level_based"] == new_user]
# 35 yaşında Fransız kadın
new_user2 = "FRA_IOS_FEMALE_(30_40]"
agg_df1[agg_df1["customer_level_based"] == new_user2]
