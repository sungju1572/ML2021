import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv("accidents_2017.csv")

data.describe()

data.columns

data_upper.dtypes


#df 모두 대문자로 바꾸기
data_upper = data.apply(lambda x: x.astype(str).str.upper())

#형변환
data_upper["Day"] = pd.to_numeric(data_upper["Day"])
data_upper["Hour"] = pd.to_numeric(data_upper["Hour"])
data_upper["Mild injuries"] = pd.to_numeric(data_upper["Mild injuries"])
data_upper["Serious injuries"] = pd.to_numeric(data_upper["Serious injuries"])
data_upper["Victims"] = pd.to_numeric(data_upper["Victims"])
data_upper["Vehicles involved"] = pd.to_numeric(data_upper["Vehicles involved"])
data_upper["Longitude"] = pd.to_numeric(data_upper["Longitude"])
data_upper["Latitude"] = pd.to_numeric(data_upper["Latitude"])

#id 행 삭제
data_upper.drop(["Id"],axis=1)

data_upper.isna().sum()

#unknown 행삭제
idx = data_upper[data_upper["District Name"]== "Unknown"].index
data_1 = data_upper.drop(idx)


#District Name 더미변수
dummy = pd.get_dummies(data_1["District Name"])

data_dummy = data_1.join(dummy.add_prefix("District_"))

data_dummy = data_dummy.drop(["District Name"], axis=1)

#Neighborhood Name 더미변수
dummy = pd.get_dummies(data_dummy["Neighborhood Name"])

data_dummy = data_dummy.join(dummy.add_prefix("Neighborhood_"))

data_dummy = data_dummy.drop(["Neighborhood Name"], axis=1)


#Street 더미변수
street_iter = (set(x.split('/')) for x in data_dummy.Street)
street_set = sorted(set.union(*street_iter))

indicator_mat = DataFrame(np.zeros((len(data_dummy), len(street_set))),columns=street_set)


for i, street in enumerate(data_dummy.Street):indicator_mat.loc[i, street.split('/')] = 1

data_dummy_mat = data_dummy.join(indicator_mat.add_prefix('street_'))

data_dummy_mat = data_dummy_mat.drop(["Street"], axis=1)

#Weekday 더미변수
dummy = pd.get_dummies(data_dummy_mat["Weekday"])

data_dummy_mat = data_dummy_mat.join(dummy.add_prefix("Weekday_"))

data_dummy_mat = data_dummy_mat.drop(["Weekday"], axis=1)

#Month 더미변수
dummy = pd.get_dummies(data_dummy_mat["Month"])

data_dummy_mat = data_dummy_mat.join(dummy.add_prefix("Month_"))

data_dummy_mat = data_dummy_mat.drop(["Month"], axis=1)


#Part of the day 더미변수
dummy = pd.get_dummies(data_dummy_mat["Part of the day"])

data_dummy_mat = data_dummy_mat.join(dummy.add_prefix("Part_"))

data_dummy_mat = data_dummy_mat.drop(["Part of the day"], axis=1)


#Id 열삭제
data_dummy_mat = data_dummy_mat.drop(["Id"],axis=1)


#데이터 준비
accidents = data_dummy_mat.drop(["Victims"],axis=1)
accidents_label = data_dummy_mat.Victims


#스케일링
scaler = StandardScaler()
scaler.fit(accidents)
scaled = scaler.transform(accidents)
scaled_accidents = pd.DataFrame(scaled, columns=accidents.columns)

#선형회귀
lin_reg = LinearRegression()
lin_reg.fit(scaled_accidents, accidents_label)

#평가
some_data = accidents.iloc[:5]
some_labels = accidents_label.iloc[:5]

lin_reg.predict(scaled_accidents)

from sklearn.metrics import mean_squared_error
accidents_pred = lin_reg.predict(scaled_accidents)
lin_mse = mean_squared_error(accidents_label, accidents_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse 
