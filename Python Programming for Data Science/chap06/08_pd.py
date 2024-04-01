# PANDAS : veri analizi, manipülasyonu

# Pandas Series :  tek boyutlu ve index bilgisi barındırır.
# Pandas Dataframe : çok boyutlu ve index bilgisi barındır.

# Pandas Series
import pandas as pd
import seaborn as sns

s = pd.Series([1, 2, 3, 4, 5]) # solda index var
type(s)
s.index
s.dtype # içindeki elemanların türü
s.size # içindeki eleman sayısı
s.ndim # seriesler tek boyutludur
s.values # index bilgisini göz ardı ettik dolayısıyla bu bir ndarraya dönüşmüş oldu
type(s.values) # ndarray yazar
s.head(3) # baştan 3 gözlem
s.tail(3) # sondan 3 gözlem

# Veri Okuma
df = pd.read_csv("datasets\Advertising.csv") # pd üzerine ctrl basarak gelerek diğer formatlara da bakabiliriz.
df.head()

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info() # object de category de kategorik değişkendir.
df.columns
df.index
df.describe().T
df.isnull().values.any() # en az 1 tane bile olsa isnull var mı?
# pd series de data frame de values denince numpy arraye döner. değişkenlerden ve indexten arındığı için.
df.isnull().sum() # her değişkende kaç eksik olduğunu söyler
df["sex"].head()
df["sex"].value_counts() # bir dfden değişken seçip sınıfları ve sınıflarında kaçar tane var bilgisini verir.

# Pandas'ta Seçim İşlemleri (ÖNEMLİ KONU)

df # titanic veri seti
df.index
df[0:13]
df.drop(0, axis=0).head() # axis=0 satır, axis=1 sütun

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10) # df'e atayarak yapmadığımız sürece kalıcı bir işlem değildir.
# df = df.drop(delete_indexes, axis=0).head(10) # kalıcı silme 1.yol
# df.drop(delete_indexes, axis=0, inplace=True).head(10) # kalıcı silme 2.yol
# inplace genel olarak kalıcı olmasını istediğimiz durumlarda kullanılır bir çok method ile kullanılır.

# Değişkeni Index'e Çevirmek
df["age"].head()
df.age.head() # aynı kullanım başka yolu

df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)
df.head()

# Index'i Değişkene Çevirme
df.index

df["age"] = df.index # df'in içinde age değişkeni varsa döner yoksa yeni değişken eklenir
df.head()
df.drop("age", axis=1, inplace=True)
df = df.reset_index().head() # indexte yer alan değeri siler yeni sütun olarak ekler oradaki bilgiyi
df.head()

# Değişkenler Üzerinde İşlemler(Sütun Indexi)

pd.set_option("display.max_columns", None)
df

"age" in df # age sütunu df'de var mı? demektir
df["age"].head() # 1.yol
df[["age"]] # 2.yol, liste girelibiliriz araya birden fazla sütun seçmek için. ortadaki aslında bir liste
df.age.head() # 3.yol
# ÇOK ÖNEMLİ
type(df[["age"]]) # df olarka döner DİKKAT!!
type(df["age"].head()) # pandas series, df bekleyen yere verirsek hata alırız. DİKKAT!!

df[["age", "adult_male", "survived"]]

col_names = ["sex", "embark_town", "alone", "age"]
df[col_names]

df["age2"] = df["age"]**2 # df'te olmayan bir isim girersek yeni sütun oluşturur.
df["age3"] = df["age"]/df["age2"]
df.drop("age3", axis=1).head() # axis=1 sütun siler, inplace=True verirsek kalıcı olarak silinir
df.drop(col_names, axis=1).head() # birden fazla sütunu silme liste yoluyla
# loc label based seçim yapmak için kullanılır
df.loc[:, ~df.columns.str.contains("age")].head() # ~ değildir demektir.
# burada satırları elleme : , sütunlarla işimiz var demiş olduk

# Loc Iloc :  df'de seçim işlemleri için kullanılır. Iloc(integer based selection) loc(label based selection)

# iloc

df.iloc[0:3] # 0'dan 3'E KADAR indexe göre seçer
df.iloc[0, 0] # satır, sütun

# loc : mutlak olarak yazılanı seçer

df.loc[0:3] # 0 ve 3 ikisi de dahil label based 3.labelı da seçer

# df.iloc[0:3, "age"] diyemeyiz hatalı çünkü iloc index alır
df.iloc[0:3, 0:3]
df.loc[0:3, "age"] # labela göre işlem yapılacaksa loc kullanılır

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]