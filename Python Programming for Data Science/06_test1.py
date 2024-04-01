##########################################################
x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1, 2, 3, 4]
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}
##########################################################
text = "The goal is to turn data information and information into insight"
text.upper().split()
##########################################################
lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

len(lst)

lst[0]
lst[10]

lst[0:4]

lst.append("R")
lst

lst.insert(8, "N")
lst
##########################################################
dict = {"Christian": ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

dict.keys()
dict.values()
dict["Daisy"][1] = 13
dict.values()
dict.update({"Ahmet": ["Turkey", 24]})
dict.values()
dict.pop("Antonio")
dict.values()
##########################################################
li = [2, 13, 18, 93, 22]

even_list = []
odd_list = []


def func(l):
    for i in l:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list


even_list, odd_list = func(li)
even_list
odd_list
##########################################################
ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for index, ogrenci in enumerate(ogrenciler, 1):
    if index < 4:
        print(f"Mühendislik Fakültesi {index}. öğrenci: {ogrenci}")
    else:
        print(f"Tıp Fakültesi {index}. öğrenci: {ogrenci}")
##########################################################
ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

list_of_zip = list(zip(ders_kodu, kredi, kontenjan))

for kod, kredi, kontenjan in list_of_zip:
    print(f"Kredisi {kredi} olan {kod} kodlu dersin kontenjanı {kontenjan} kişidir.")
##########################################################
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

if kume1.issuperset(kume2):
    print(kume1.intersection(kume2))
else:
    print(kume2.difference(kume1))
##########################################################
################ List Comprehension ######################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

{"NUM_" + col.upper() if df[col].dtype != "object" else col.upper() for col in df.columns}
##########################################################
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
{col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns}
##########################################################
og_liste = ["abbrev", "no_previous"]

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

new_cols = []

{new_cols.append(col) for col in df.columns if col not in og_liste}

new_df = df[new_cols]

import seaborn as sns
df=sns.load_dataset("titanic")
sns.countplot(x="class",data=df)
plt.show()