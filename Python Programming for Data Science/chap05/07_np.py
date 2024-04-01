# NUMPY : matematik ve istatistik kütüphanesi
# Neden numpy?
# Hız : sabit tipte veri tutar bu sebeple hızlı.
# Yüksek seviyeden işlem yapmayı sağlar.

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

# numpy vektörel seviyede işlem yapmayı sağlar bu sebeple hızlıdır.

# normal yol
ab = []
for i in range(0, len(a)):
    ab.append(a[i] * b[i])

# numpy ile: vektörel işlem
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

# Numpy Array Oluşturma
# ndarray de bir veri yapısıdır. Int,float gibi

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))  # ndarray

np.zeros(10, dtype=int)
np.random.randint(0, 10, size=10)  # 0 ile 10 arasında rastgele 10 tane int üretir.
np.random.normal(10, 4, (3, 4))  # ortalamasını 10 standart sap. 4 olan 3x4lük bir array oluşturur.

# Numpy Array Özellikleri

a = np.random.randint(10, size=5)  # başlangıca bir şey girmediğimizde 0'dan başlar 10'a kadar 5 adet
a
a.ndim  # boyut sayısı
a.shape  # boyut bilgisi
a.size  # toplam eleman sayısı
a.dtype  # tip bilgisi

# Reshapping

b = np.random.randint(10, size=9)
b
b.reshape(3, 3)  # eleman sayısı önemli size 10 olan bir arrayı 3x3 yapamayız

# Index Seçimi

a = np.random.randint(10, size=10)
a
a[0]
# Slicing
a[0:5]  # 0 dahil 5 değil
a[0] = 999
a

m = np.random.randint(10, size=(3, 5))
m[1, 1]  # satır, sütun 2 boyutlu dizilerde indexleme
m[2, 3] = 999
m
m[
    2, 3] = 2.9  # float girsek bile sadece 2 kısmı gelir çünkü numpy tek bir tip tutar bu da onu hızlı kılan şeydir zaten.
m
m[:, 0]
m[1, :]
m[0:2, 0:3]  # 2 ve 3 dahil değil :sol taraf dahil değil unutuma slicing kuralı

# Fancy Index
v = np.arange(0, 30, 3)  # arange : belli bir adım boyunca array oluşturma
v
v[1]
v[4]

catch = [1, 2, 3]

v[catch]  # bir dizi index

# Numpy'da koşullu işlemler

import numpy as np

v = np.array([1, 2, 3, 4, 5])

v < 3 #bütün elemanları için tek tek sorgular

v[v < 3]

# Matematiksel İşlemler
v

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)

# Numpy ile 2 bilinmeyenli denklem

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]]) # önce x0'ın kat sayıları sonra x1'in
b = np.array([12, 10]) # sonuçlar

np.linalg.solve(a, b)