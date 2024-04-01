# COMPREHENSION : fazla satırlı kodları tek satırda gerçekleştirmek

# List Comprehension

salaries = [100, 200, 300, 400, 500]


def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary(salary))


null_list = []

for salary in salaries:
    if salary > 300:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary) * 2)


[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary > 300]

[salary * 2  if salary > 300 else salary * 0 for salary in salaries]
# sadece if varsa for'un sağına if else varsa soluna yazılır.

[new_salary(salary * 2)  if salary > 300 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students] # çok önemli!

[student.upper() if student not in students_no else student.lower() for student in students] # aynı şeyi yapar not in kullandık

# Dict Comprehension

dict = {"a": 1,
        "b": 2,
        "c": 3,
        "d": 4}

dict.keys()
dict.values()
dict.items()
# liste formunda ama her eleman tuple olarak ifade edilerek döner ("a",1) gibi [] liste içinde tuple çiftleri döner

{k: v ** 2 for (k, v) in dict.items()}

{k.upper(): v for (k, v) in dict.items()}

{k.upper(): v ** 2 for (k, v) in dict.items()}

# UYGULAMA - MÜLAKAT SORUSU

# Amaç : çift sayıların karesini alarak bir sözlüğe eklemek

numbers = range(10)
# 0'dan 10'a kadar sayıları ifade eder
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2 # bu mantık önemli
        # new_dict sözlük yapısı olduğu için new_dict[n] dediğimizde
        # otomatik olarak key olarak karşılığını da value olarak ekler.

print(new_dict)

{n: n ** 2 for n in numbers if n % 2 == 0} # çok önemli

# List & Dic Comprehension Uygulamalar

# Bir veri setindeki değiken isimlerini değiştirmek

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

A = []

for column in df.columns:
    A.append(column.upper())

df.columns = A

# isminde ins olan değişkenlerin başına flog olmalaranlara no_flog eklemek
# sadece sayısal değişkenler için bu işlemi yapmak istiyoruz.

df.columns = {"FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns}

# Amaç : key'i string, value'su liste olan bir sözlük oluşturmak
# Şöyle olmalı {"total": ["mean", "max", "min", "var"],
#                "speeding": ["mean", "max", "min", "var"]}

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [col for col in df.columns if df[col].dtype != "O"] # buradaki O object'in baş harfi

dict = {}
agg_list = ["mean", "max", "min", "sum"]

for col in num_cols:
    dict[col] = agg_list

# kısa yol
{col: agg_list for col in num_cols}

# bu örneğin kullanım alanı
new_dict = {col: agg_list for col in num_cols}

df[num_cols].agg(new_dict)
