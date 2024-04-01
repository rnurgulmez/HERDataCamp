# Koşullar
1 == 1
1 == 2

# if
if 1 == 1:
    print("something")

number = 10
if number == 11:
    print("Number is 11")


# DRY (don't repeat yourself)

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(8)

students = ["John", "Mark", "Venessa", "Mariam"]
# içinde gezilen elemanı iç içe döngü vs olursa diye onu takip etmek için geçici student adını veriyoruz
for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


new_salary(1500, 10)
new_salary(2000, 15)

for salary in salaries:
    print(new_salary(salary, 10))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))

# UYGULAMA MÜLAKAT SORUSU

# Amaç : hi my name is john cümlesini Hi My NaMe iS JoHn yazmak

range(len("miuul"))  # 0'dan başla demesek de default olarak 0'dan başlar
range(0, 5)

for i in range(0, 5):
    print(i)


def alternating(string):
    new_string = ""
    # girilen stringin indexlerinde gez
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçük harfe çevir
        else:
            new_string += string[string_index].lower()
    return new_string


alternating("rabia")

# Break, continue, while

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

for salary in salaries:
    # 3000'i atlayıp devam eder
    if salary == 3000:
        continue
    print(salary)

# while
number = 1
while number < 5:
    print(number)
    number += 1

# Enumarate : Otomatik counter/indexer ile for loop

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)
# enumerate(students, 1) dersek indexleme 1'den başlar
for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

# UYGULAMA MÜLAKAT SORUSU
# Amaç : divide_students fonk. yaz çift indexteki öğrencileri bir listeye tek indexteki öğrencileri başka listeye al
# fakat bu 2 liste tek bir liste olarak return olsun

students = ["John", "Mark", "Venessa", "Mariam"]


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups


st = divide_students(students)
st[0]
st[1]


def alternating_with_enumarate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumarate("hi my name is john and i am learning python")

# zip

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

# 3 farklı listeyi bir arada değerlendirmek

list(zip(students, departments, ages))


# lambda, map, filter, reduce

def summer(a, b):
    return a + b

# lambda

summer(1, 3) * 9
# kullan at fonksiyonlar. değişkenler bölümünde yer tutmaz. map ve apply ile kullanılır.
new_sum = lambda a, b: a + b

new_sum(1, 5)


# map
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries)) # döngü yazmaya gerek kalmadan yapmak istediğimiz fonksiyonu ve iteratif listeyi verdik.

# lambda-map ilişkisi
list(map(lambda x: x ** 2, salaries))

# filter : sorgu gibidir. iteratif nesnede sorgu yapar
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9]
list(filter(lambda x: x % 2 == 0, list_store))

# reduce
from functools import reduce
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9]
reduce(lambda a, b: a + b, list_store)

string = "abracadabra"
group = []

for index, letter in enumerate(string, 1):
    if index * 2 % 2 == 0:
        group.append(letter)
    print(group)