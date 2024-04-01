#############################
# Pivot Table
############################
# group by'a benzer veri setini kırılımlar açısından değerlendirir ve
# ilgilendiğimiz özet istatistiği bu kırılımlar açısından görmemizi sağlar.

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked") # values(satır), index(satır), calumns(sütun)
# pivot_table ön tanımlı olarak mean yani ortalama alır. bunu değiştirmek için aggfunc= belirtmeli

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head() # yaş değişkeni int onu değiştirmek istiyoruz
# sayısal değişknei kategorik değişkene çevirmek istiyoruz
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
# cut : bölmek istediğimiz aralıkları biliyorsak, qcut: bilmiyorsak yüzdelik çeyreğe göre böler.
df.head()

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option("display.width", 500) # yan yana gelmesi için tüm tablonun çıktıda