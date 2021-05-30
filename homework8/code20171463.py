#20171463 유성주

""" 
problem
목적 : 바로셀로나 시에서 현지 경찰이 처리한 데이터를 가지고 희생자수를 예측하는 회귀모델 만들기
예측을 위한 regression을 수행할 것이며, 데이터의 칼럼은 15개이고 Victims 칼럼이 target이다
"""

"""
Feature
데이터를 훑어 보았을때 우리의 타겟인 Victims 칼럼이  Mild injuries칼럼과 Serious injuries 칼럼을 합한것과 
매우 유사하다는것을 확인했다. 실제로 두 칼럼을 합쳐 Victims 칼럼과 어느정도 일치하는지 확인했더니, 2개 빼고 전부 일치하는것을 확인했다
따라서 회귀분석을 위한 feature는 두 칼럼을 합친것만 사용해도 충분하다고 결정했다
"""

""" 
Model
feature가 한개이므로 선형회귀분석(LinearRegression)을 사용하였다. 
"""

""" 
Measure
10개의 fold 교차검증을 하였으며, 평가지표는 MAE로 하였다

"""


""" 
Model parameter engineering
성능이 매우 좋아 파라미터는 더이상 조절할 필요가 없었다
"""


import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

#데이터 읽기
data = pd.read_csv("accidents_2017.csv")

#데이터를 훑어 보았을때 우리의 타겟인 Victims 칼럼이  Mild injuries칼럼과 Serious injuries 칼럼을 합한것과 매우 유사하다는것을 확인했다 

#data 데이터 프레임에 두 칼럼을 합친 "add" 칼럼 새로 생성
data["add"] = data["Mild injuries"]+data["Serious injuries"]

#새로만든 add 칼럼과 Victims 칼럼이 어느 정도 일치하는지 확인
a = data["add"] == data["Victims"] 

# 두번 빼고 전부 일치하는것을 확인가능 -> target인 Victims에 대한 데이터는 add 칼럼만 사용해도 되겠다는 결론 도출
a.describe()


#새로운 데이터 프레임 생성
accidents = data[["add", "Victims"]]

#feature
accidents_data = accidents[["add"]]
#label
accidents_label = accidents[["Victims"]]


#feature가 한개이므로 선형회귀 사용
lin_reg = LinearRegression()



#10-fold 
kfold = KFold(n_splits=10, shuffle=True, random_state=123)

#카운트용 변수 생성
n_iter = 0


#train / test 로 나눠 폴드별 선형회귀에 따른 MAE값이 얼마인지 출력
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
    




