import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("accidents_2017.csv")

#df 새로 생성
data.columns

data["add"] = data["Mild injuries"]+data["Serious injuries"]

accidents = data[["add", "Victims"]]


accidents_data = accidents[["add"]]
accidents_label = accidents[["Victims"]]

#선형회귀
lin_reg = LinearRegression()



#10개 폴드세트를 분리하고, 각 폴드별 정확도를 담을 리스트를 생성
kfold = KFold(n_splits=10, shuffle=True)
lin_accuracy =[]



n_iter = 0


for train_idx, test_idx in kfold.split(accidents_data):
    X_train, X_test = accidents_data.iloc[train_idx], accidents_data.iloc[test_idx]
    
    y_train, y_test = accidents_label.iloc[train_idx], accidents_label.iloc[test_idx]

    #학습진행
    lin_reg.fit(X_train, y_train)
    
    #예측
    fold_pred = lin_reg.predict(X_test)
    
    #MAE 측정
    n_iter += 1
    mae = mean_absolute_error(y_test, fold_pred)
    print("\n {} MAE : {} 학습데이터 크기 : {} 검증 데이터 크기 : {}".format(n_iter,mae, X_train.shape[0],X_test.shape[0]))
    




