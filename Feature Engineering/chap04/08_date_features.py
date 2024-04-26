#################################
# Date ile Değişken Türetmek
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor  # çok değişkenli aykırı değer yakalama yöntemi
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler  # dönüştürme fonksiyonları

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info() # timestamp değişkeni object bunu değiştirmeliyiz.

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"])
dff.dtypes

# year
dff["year"] = dff["Timestamp"].dt.year

# month
dff["month"] = dff["Timestamp"].dt.month

# year diff
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

# month diff(iki tarih arasındaki ay farkı) : yıl farkı + ay farkı
dff["month_diff"] = (date.today().year - dff["Timestamp"].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

# day name
dff["day_name"] = dff["Timestamp"].dt.day_name()
dff.head()