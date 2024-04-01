# Seaborn : görselleştirme için kullanılır yüksek seviyeli
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("titanic")
df.head()

# Kategorik Değişkenleri Görselleştirme
df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

# Sayısal Değişkenleri Görselleştirme
sns.boxplot(x=df["survived"])
plt.show()

# pandas'ta yer alan hist da kullanılacak
df["age"].hist()
plt.show()