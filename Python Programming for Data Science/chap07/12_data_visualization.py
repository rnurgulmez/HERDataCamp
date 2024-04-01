# Veri Görselleştirme :  Matplotlib & Seaborn
import numpy as np
# Matplotlib : low level veri görselleştirme yapar

# Kategorik değişken: sütun grafik. countplot bar
# Sayısal değişken: hist, boxplot

# Kategorik Değişken Görselleştirme
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show() # grafiği oluşturduktan sonra print etmek gibi yazdırmalıyız

# Sayısal Değişken Görselleştirme
plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"]) # aykırı değerleri tespit eder
plt.show()

# Matplotlib Özellikleri : katmanlı şekilde veri görselleştirme sağlar.

# plot özelliği
x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o")
plt.show()

# marker özelliği
y = np.array([13, 28, 11, 100])

plt.plot(y, marker="h") # o, *, ., ,, x, X, P, s, D, d, p, H, h, + bu sembollerin tümü kullanılabilir
plt.show()

# line özelliği
y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashed", color="r") # dashed, line, dashdot, dotted
plt.show()

# multiplelines özelliği
x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x)
plt.plot(y)
plt.show()

# labes özelliği
x = np.array([80, 85, 90, 95, 100, 105])
y = np.array([240, 250, 260, 270, 280, 290])
plt.plot(x, y)
# Başlık
plt.title("Bu ana başlık")
# X ekseninde isimlendirme
plt.xlabel("x ekseni")
# Y ekseninde isimlendirme
plt.ylabel("y ekseni")
plt.grid()
plt.show()

# Subplots
x = np.array([80, 85, 90, 95, 100, 105])
y = np.array([240, 250, 260, 270, 280, 290])
plt.subplot(1, 2, 1) # 1'e 2'lik grafik ve bunlardan 1.yi gösteriyorum demek
# plt.subplot(1, 2, 2) # 1'e 2'lik grafik ve bunlardan 2.yi gösteriyorum demek
plt.title("1")
plt.plot()
plt.show()