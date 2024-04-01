# Fonksiyon
# parametre fonksiyon tanımlanırken girilmesi istenen değişkenler, arguman fonksiyonu kullanırken verdiğimiz değer.

# Fonksiyon Okur-Yazarlığı
# ?print konsola yazarak parametrelerine ve kullanımına bakabiliriz.
# docstring : kullandığımız fonksiyonların dokumantasyonları

print("a", "b", sep="__")


# Fonksiyon Tanımlama
def calculate(x):
    print(x * 2)


calculate(5)


def summer(arg1, arg2):
    """
    Sum of two numbers

    Args:
        arg1: int, float
        arg2: int, float

    Returns:
        int, float

    Examples:

    Notes:

    """
    print(arg1 + arg2)


# Docstring : fonksiyonlara girilen bilgi notu. help yazarak ulaşılabilir.
# ?summer yazarak consoledan ulaşılabilir. Ya da help(summer) ile ulaşılabilir bilgiye.

# Fonksiyonların Statement/Body Bölümü

# def function_name(parameters/arguments):
#   statements(function body)

def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")


say_hi("rabia")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# girilen değerleri bir liste içinde saklayacak

list_store = []


def add_element(a, b):
    c = a * b  # c: local scope'da, list_store: global scope'da
    list_store.append(c)  # kalıcı olarak ekler append
    print(list_store)


add_element(1, 8)
add_element(14, 2)


# Ön Tanımlı Argümanlar/Parametreler:

def divide(a, b=1):
    print(a / b)


divide(5)


# ön tanımlı argüman vermek fonksiyonun kolay kullanılması içindir.

def say_hi(string="hello"):
    print(string)
    print("merhaba")


say_hi("rabia")


# Ne zaman fonksiyon yazılır? : kendimizi tekrar etmemek için (DRY prensibi)

def calculate(warm, moisture, charge):
    print((warm + moisture) / charge)


calculate(45, 23, 12)


# Return : fonksiyon çıktısını girdi olarak kullanmayı sağlar


def calculate(warm, moisture, charge):
    warm = warm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (warm + moisture) / charge

    return warm, moisture, charge, output


warm, moisture, charge, output = calculate(98, 12, 78)  # tuple olarak döner?? neden tuple olarak çıkıyor
# bütün çıktıları bir değişkene atadık

type(calculate(98, 12, 78))


# Fonksiyon içinden fonksiyon çağırmak


def calculate(warm, moisture, charge):
    return int((warm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


def all_calculations(warm, moisture, charge, p):
    # misafir fonksiyonların parametrelerini de ana fonksiyonda karşılamak zorundayız.
    a = calculate(warm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculations(1, 3, 5, 12)
