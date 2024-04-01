# Veri Yapıları

# Sayılar

# Integer
x = 46
type(x)

# Float
x = 10.3
type(x)

# Complex
x= 2j + 1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)
5 == 4

# Liste
x = ["btc", "eth", "xrp"]
type(x)

# Sözlük
x = {"name": "peter", "age": 36}
type(x)

# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)

# Liste, set, tuple, sözlük aynı zamanda python collections(arrays) olarak da geçer.

# Sayılar
a = 5
b = 10.5
# PEP8, python yazarken kurallar
a * 3
a * b / 10

# Tip Değiştirme
int(b)
float(a)

int(a * b / 10) # önce parantez içini yapar daha sonra dönüşüm gerçekleşir

# String(Karakter Dizileri)
print("John")
name = "John"

# çok satırlı karakter dizisi
long_str = """ buraya çok satırlı
bir string
yazabiliriz ve bunu da bir değişkene atayabiliriz."""
long_str

# stringlerde eleman seçme
name
name[0]
name[0:2]

# string içinde eleman sorgulama
long_str
"bunu" in long_str # case sensitive olduğuna dikkat Bunu deseydik False dönerdi.

# String Metodları
dir(int) # intlerle kullanılabilecek methodları görme
dir(str) # stringlerle kullanılabilecek methodları görme

name = "john"

len(name)
len("rabia nur")

# class içerisinde tanımlanan fonksiyona "method" denir. Class yapısı içinde değilse fonksiyondur.

"miuul".upper() # upper bir methoddur.

hi = "hello ai era"
hi.replace("l", "p")

hi.split() # boşluklara göre böler ya da nasıl bölmek istediğimizi bir belirtiriz arguman olarak.

" ofofof ".strip() #boşluklara göre kırpma işlemi yapar
" ofofof ".strip("o")

"foo".capitalize()

"foo".startswith("o")

# Liste
# değiştirilebilir,sıralıdır(index işlemleri yapılabilir.),kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]
not_nam[5]
not_nam[6][1]

not_nam[0] = 99
not_nam

not_nam[0:4]

# Liste Methodları
dir(notes)

len(notes)
len(not_nam)

notes.append(100) # listenin sonuna ekler.
notes

notes.pop(0) # indexe göre eleman siler.
notes

notes.insert(2, 88) # indexe göre eleman ekler. hem index hem yeni eleman belirtilir.
notes

# Sözlük
# değiştirilebilir, sırasız(3.7 itibariyle sıralı), kapsayıcı
# key-value

dict = {"reg": "regression",
        "log": "logistic regression",
        "cart": "classification and regression"}

dict["reg"]

dict = {"reg": ["RMSE", 10],
        "log": ["LOG", 20],
        "cart": ["SSE", 30]}

dict["log"][1]

# Key Sorgulama

"reg" in dict

# Keye göre valueye erişme

dict["log"]
dict.get("log") # bu türlü de yapılabilir.

dict["reg"] = ["YSA", 88]
dict

# Sözlük Methodları

dict.keys()
dict.values()
dict.items() # bir liste içinde tuple cinsinden hem key hem valueları verir.
dict.update({"abc": 13}) # yeni bir çift de eklenebilir olan çift de değiştirilebilir.
dict

# Tuple
# değiştirilemez, sıralıdır(elemanlarına erişilebilir), kapsayıcı(birden fazla veri yapısını tutabilir.)

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99 # yapmaz hata

# değiştirmek istersek listeye çevirmeliyiz.

# Set
# değiştirilebilir, sırasız+eşsiz, kapsayıcıdır.
# setler kümeler gibidir. kesişimleri nedir birleşimleri nelerdir gibi alanlarda kullanılır.

# difference(): iki kümenin farkı
set1 = set([1, 3, 5]) # normal {} ile ifade edilir ama  bu da 2.yoludur.
set2 = set([1, 2, 3])

# set1de olup set2de olmayanlar
set1.difference(set2)

# set2de olup set1de olmayanlar
set2.difference(set1)

# symmetric_difference(): iki kümede de birbirlerine göre olmayanlar
set1.symmetric_difference(set2)
set1 - set2

# intersection(): iki kümenin kesişimi
set1.intersection(set2)
set1 & set2 # aynı şeyi yapar

# union(): iki kümenin birleşimi
set1.union(set2)

# isdisjoint(): iki kümenin kesişimi boş mu?
set1.isdisjoint(set2)

# issubset(): alt kümesi mi?
set1.issubset(set2)
set2.issubset(set1)

# issuperset(): kapsıyor mu?
set1.issuperset(set2)