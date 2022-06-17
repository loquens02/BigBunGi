#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part3-모델링-예측(연속형)
# Regressor

# ### 전처리

# In[1]:


### 시간 측정, 데이터 로드, 종속-독립변수 분리
import pandas as pd
import time
start_time= time.time()
data=pd.read_csv('bigData/mtcars.csv')
# print(data.info(), data.head(),'-----'))
Y= data['mpg']
X= data.drop(columns=['mpg','Unnamed: 0'])
pd.set_option('display.max_columns',None) # s!
X.head()

### null 찾고 채우기, 이상한 값 찾고 바꾸기, 이상한 타입 바꾸기, 인코딩
# print(X.loc[X.isnull().sum(1) != 0], '-----'))
X['cyl'].fillna(X.describe().loc['mean','cyl'], inplace=True)
X['qsec'].fillna(X.describe().loc['mean','cyl'], inplace=True)
# print(X, '-----'))
# print(X.loc[:,X.dtypes==object].iloc[:,1].unique(), '-----')) #'gear'
# print(X.info(), '-----')) 
X['gear']= X['gear'].replace('*3','3').replace('*5','5').astype('int64') # min 이상치에 'gear' 도 포함된다
# print(pd.get_dummies(X, drop_first=True), '-----')) # help(pd.get_dummies)
# print(X['am'].unique(), '-----'))
# print(X, '-----'))
X= pd.get_dummies(X, drop_first=True)
# print(X, '-----')) # am_auto 뿐만 아니라, 기존 am 컬럼도 자동으로 지워진다.
# print(X.info(), '-----'))

### 이상치 교체(meanStd or IQR)
## describe -> mean +- 1.5*std , Q3 + 1.5*IQR 및 Q1 - 1.5*IQR -> def 컬럼 별. maxOver minOver. 찾는 것부터
# True 중에 중복 처리없게 골라내기. int64는 되도록 유지하기
# print(X.loc[:,X.dtypes=='int64'].nunique(), '-----')) # hp22, vs2, gear3, carb6 종
# print(X.loc[:,X.dtypes=='int64'], '-----')) # vs 01, gear 345, carb 123468 
# 컬럼 조건만 알아본 것
def outlierCheck(data):
    desc= data.describe()
    max1= desc.loc['max']
    min1= desc.loc['min']
    
    mean= desc.loc['mean']
    std= desc.loc['std']
    minBms= mean - 1.5*std
    maxBms= mean + 1.5*std
    minOverMs= min1 < minBms # Boundary 보다 작으면 이상치
    maxOverMs= max1 > maxBms # Boundary 보다 크면 이상치
#     print(minOverMs, '-----')) # drat, wt 만
#     print(maxOverMs, '-----')) # vs, am_maunal 빼고 다 => disp, drat 만. gear 제외
    
    Q1= desc.loc['25%']
    Q3= desc.loc['75%']
    IQR= Q3-Q1
    minBiqr= Q1 - 1.5*IQR
    maxBiqr= Q3 + 1.5*IQR
    minOverIQR= min1 < minBiqr
    maxOverIQR= max1 > maxBiqr
#     print(minOverIQR, '-----')) # qsec 만
#     print(maxOverIQR, '-----')) # cyl, hp, wt, qsec, carb => carb 유지. 기화기는 엔진 부품. 8개면 좀 이상하긴하다
    
    #     print(minBms, maxBms, minBiqr, maxBiqr, '-----'))
    return minBms, maxBms, minBiqr, maxBiqr

# outlierCheck(X)

# 뭘 바꿀지 이미 정했으므로 행 조건만 알면 된다.
def outlierReplace(data):
    minBms, maxBms, minBiqr, maxBiqr= outlierCheck(data)
    #행 조건- 각 열 내의 값이 Boundary Over 인지. data[col]<minBms[col]
    #열 조건
    minMScols= ['drat','wt']
    maxMScols= ['disp','drat']
    minIQRcols= ['qsec']
    maxIQRcols= ['cyl', 'hp', 'wt', 'qsec', 'carb']
    
    #     print(type(minBms), '-----'))
    #     print(minBms[column], '-----'))
    for col in minMScols:
#         print(f"BeforeMinMS: {data.loc[ data[col]<minBms[col],col]}, minBms: {minBms[col]:.7}") 
        data.loc[ data[col]<minBms[col], col]= minBms[col]
    for col in maxMScols:
#         print(f"BeforeMaxMS: {data.loc[ data[col]>maxBms[col],col]}, maxBms: {maxBms[col]:.7}")
        data.loc[ data[col]>maxBms[col], col]= maxBms[col]
    for col in minIQRcols:
#         print(f"BeforeMinIQR: {data.loc[ data[col]<minBiqr[col],col]}, minBiqr: {minBiqr[col]:.7}")
        data.loc[ data[col]<minBiqr[col], col]= minBiqr[col]
    for col in maxIQRcols:
#         print(f"BeforeMaxIQR: {data.loc[ data[col]>maxBiqr[col],col]}, maxBiqr: {maxBiqr[col]:.7}")
        data.loc[ data[col]>maxBiqr[col], col]= maxBiqr[col]
    return data

X= outlierReplace(X) # inplace 될 때도 있고, 안 될 때도 있어서 확실히 해두기.
# print(f"로드~이상치 처리 시간: {time.time()-start_time:.7} sec") #  0.095 sec
# print(X, '-----'))

### 범위 맞추기
# from sklearn import preprocessing
# help(preprocessing)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
# print(X.describe(), '-----'))
def rangeScaler(scalers, columnDataFrame): # 이 인자 외워야. X[[ ]]
    if scalers=='MinMax':
        scaler= MinMaxScaler()
    elif scalers=='Standard':
        scaler= StandardScaler()
    elif scalers=='Robust':
        scaler= RobustScaler()
    #     help(scaler.fit_transform)
    scaledDf= pd.DataFrame(scaler.fit_transform(columnDataFrame)) # 이 형태 외워야.
    return scaledDf

# print(X[['qsec']], '-----'))
# ret2= rangeScaler('MinMax', X[['qsec']]) # min 0, max 1
# ret2= rangeScaler('Standard', X[['qsec']]) # mean 0 근접, std 1 근접
# ret2= rangeScaler('Robust', X[['qsec']]) # mid 0, IQR= Q3-Q1= 1
# print(ret2.describe(), '-----'))
# print(ret2.describe().loc['75%']-ret2.describe().loc['25%'])
# print(ret2, '-----'))

## 범주형 명시- category
# help(X['am_manual'].astype) #Convert to categorical type
X['am_manual']= X['am_manual'].astype('category')
X['vs']= X['vs'].astype('category')
# print(X.info(), '-----'))

## LabelEncoder 연습
# face= ['happy','sad','soso']
# from sklearn import preprocessing
# help(preprocessing)
# from sklearn.preprocessing import LabelEncoder
# encoder= LabelEncoder()
# print(face, encoder.fit_transform(face), '-----'))

### 파생변수
## 무게등급
condition= X['wt'] < 3.3
X.loc[condition, 'wt_class']= 0 # 작
X.loc[~condition, 'wt_class']= 1 #크거나 같
X.drop(columns='wt',inplace=True)
# print(X, '-----')
X['wt_class']= X['wt_class'].astype('int64').astype('category') # float 에 category 는 deprecated
# print(X, '-----')
## 1mile
X['qsec_4']= X['qsec']*4
# print(X[['qsec_4','qsec']], '-----')
X.drop(columns='qsec', inplace=True)
# print(X, '-----')

### 표준화 적용- 파생변수 뒤에 와야
targetCols= X.columns[(X.dtypes=='float64')]
for col in targetCols:
    X[col]= rangeScaler('MinMax', X[[col]]) # MinMax 종속 연속형

print(X, '-----')
print(f"실행시간: {time.time()-start_time:.7} sec") # 0.130 sec (6.858 sec 는 최초 module 로드탓)


# ## 모델링
# p.280

# ### 1. 학습용과 검증용 데이터 나누기
# - 입력: 독립변수, 종속변수, 검증 데이터 비율
# - 나누지 않으면 주어진 데이터에 과적합되어 새 데이터는 못 맞추는 일이 발생할 것

# In[2]:


# import sklearn
# help(sklearn) # package/ model_selection
# from sklearn import model_selection
# help(model_selection) #train_test_split . 적어도 train_ 은 알고있어야
from sklearn.model_selection import train_test_split
# help(train_test_split) # examples
# 인자: 독립변수X set, 종속변수Y set, test 비율, [random_state]
# 반환: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.3, random_state=10) 
# random_state 동일결과 확인용. 시험때는 안 넣어도 된다.
print(X_train.shape, X_train.head())
print(X_test.shape, X_test.head())
print(y_train.shape, y_train.head())
print(y_test.shape, y_test.head())


# ### 2-1. 모델 학습- 선형회귀 모델로 - LinearRegression
# p.287 / 얘는 끝이 ssion 인데, 랜포는 끝이 ssor 다. 주의!
# 1. from sklearn.모듈 import 모델함수
# > 사용할 모델의 함수 가져오기 / 모듈: linear_model, 
# 2. model= 모델함수()
# > 학습 모델 만들기
# 3. model.fit(X_train, y_train)
# > 학습 데이터로 모델 학습시키기
# 4. y_train 예측값= model.predict(X_train); y_test 예측값= model.predict(y_test)
# > 학습된 모델로 값 예측하기

# In[3]:


# import sklearn
# help(sklearn) # package/ linear_model
# from sklearn import linear_model
# help(linear_model) # classes 6째 단락에 있긴한데 잘 안 보임. LinearRegression
from sklearn.linear_model import LinearRegression
# help(LinearRegression) #reg = LinearRegression().fit(X, y) #reg.score(X, y), reg.coef_ , reg_intercept_ #reg.predict()

# 학습할 학생= 모델
model= LinearRegression() 
# 학생에게 문제집 풀게하기= 학습
model.fit(X_train, y_train)
# 학생 시험치기- 문제집에 있는 문제로= 예측
y_train_predict= model.predict(X_train)
print(y_train_predict) # 예측값

# X_train 으로 만든걸 y_train_predict 라 이름 붙인다. > score 낼 때 헷갈리는데, 여하튼 predict 결과로 보고 싶은건 y 종속변수라 그렇다

# 학생 시험치기- 문제집에 없는 문제로= 예측
y_test_predict= model.predict(X_test)
print(y_test_predict)


# plot 은 그릴 수 없지만, 선형회귀식 => 선

# In[4]:


print(f"y절편: {model.intercept_}")
print(X.columns)
print(f"각 독립변수의 기울기: {model.coef_}")


# ##### 모델 평가 
# 예측한 값이 믿을 수 있는지
# 1. from sklearn.metrics import 평가함수
# > 평가할 함수 가져오기
# 2. print(평가함수(y_train, y_train의 예측값); print(평가함수(y_test, y_test의 예측값)
# > 모델 평가하기

# p.288
# - model.score(). 결정계수. 실제분산과 예측분산의 비율이 1에 가까울수록 정확도가 높음. r2_score() 와 같음
# 
# ##### 이건 predict 수행여부와는 별개더라
# 왜지. 이미 한 번 실행해서 먹히는 것일듯?

# In[5]:


# help(model.score) #score(X, y, sample_weight=None) : R^2
# 문제집에 있는 문제 기준으로 학생의 공부결과 평가
print(f"train: {model.score(X_train, y_train)}")
# 문제집에 없는 새로운 문제로 학생의 공부결과 평가
print(f"test: {model.score(X_test, y_test)}")


# 종속변수 실제값과 종속변수 예측값 간의 이러저런 차이 > y 랑 y_predict 간 차이
# - MAE. mean_absolute_error. mean(|실제값-예측값|)
# - MSE. mean_squared_error. mean(|실제값-예측값|^2)
# - RMSE. root MSE. sqrt(MSE)

# In[137]:


# import sklearn # help(sklearn) # metrics
# from sklearn import metrics #help(metrics) # 한 중간에 있어서 보기어렵. mean_a mean_s r2_검색. 
# r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#제곱근 계산
import numpy as np
# help(np) # np.sqrt

print('분산 비율 > 1에 가까울수록 좋다') 
print('R^2 train: ',r2_score(y_train, y_train_predict)) 
print('R^2 test: ',r2_score(y_test, y_test_predict))

print('\n차이 > 0에 가까울수록 좋다')
# print('MAE train: ',mean_absolute_error(y_train, y_train_predict))
print('MAE test: ',mean_absolute_error(y_test, y_test_predict))  # 종속변수 실제값과 종속변수 예측값 간의 이러저런 차이

# 차이 > 0에 가까울수록 좋다
# MSEtrain=mean_squared_error(y_train, y_train_predict)
MSEtest= mean_squared_error(y_test, y_test_predict)
# print('MSE train: ',MSEtrain)
print('MSE test: ',MSEtest)

# 차이 > 0에 가까울수록 좋다
# print('RMSE train: ',np.sqrt(MSEtrain))
print('RMSE test: ',np.sqrt(MSEtest))


# In[ ]:





# ### 2-2. 모델학습 및 평가- 랜덤 포레스트 회귀분석
# p.290

# In[7]:


# import sklearn # help(sklearn) #ensemble. 앙상블. 이건 외우고 있어야.
# from sklearn import ensemble # help(ensemble) #randomforest/ RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# '22.6.16.목 23:17 

# '22.6.17.금 12:15
# ~297 ~321

# In[8]:


from sklearn.ensemble import RandomForestRegressor


# In[9]:


print(X.columns)
# 학습자(모델) 생성
model= RandomForestRegressor(random_state=10)
# 학습자에게 문제집70% 제공해서 학습하도록
model.fit(X_train, y_train) # y는 Series 이어야 한다. df ㄴㄴ
# 학습자에게 문제70% 중 아무거나 줘서 잘 학습했는지 확인
y_train_predict= model.predict(X_train)
print(y_train_predict)
# 학습자에게 나머지 문제 30% 중 아무거나 줘서 학습한 거 외에도 잘 푸는 지 확인
y_test_predict= model.predict(X_test)
print(y_test_predict)


# In[10]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 학습문제70% 로 보는 점수. 주어진 문제 잘 공부했는지. 이게 낮으면 아직 덜 공부한 것.
print('train: ',r2_score(y_train, y_train_predict)) 

# R^2. 결정계수. 종속변수 실제값과 예측값 비율. 1에 가까울 수록 정확
print('test: ',r2_score(y_test, y_test_predict)) # 나머지 문제 30%로 보는 점수. 모델 실전 평가

# MAE. 종속변수 |실제값과 예측값 차이|
print('test: ',mean_absolute_error(y_test, y_test_predict))

# MSE. 종속변수 |실제값과 예측값 차이|^2
print('test: ',mean_squared_error(y_test, y_test_predict))


# 시험에 R2, MAE, MSE, RMSE 중 뭐로 평가하라고 할지 모른다.
# - 안 정해주면 score 높은 (R2 면 1에 가깝고, 나머지는 0에 가까운) 것 골라서 넣고
# - 정해준다면 모델을 조정하여 점수를 높이자

# ####  하이퍼 파라미터
# 모델을 조정하는 방법 / p.292

# In[69]:


# help(RandomForestRegressor)
# (n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0
# ,max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None
# , bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)


# In[23]:


import time
import numpy as np
start_time= time.time()
from sklearn.ensemble import RandomForestRegressor
# 투표할 트리 1천개, 트리 분할기준은 MAE
# model= RandomForestRegressor(n_estimators=1000, criterion='mae', random_state= 10)
model= RandomForestRegressor(n_estimators=1000, criterion='mse', random_state= 10)

# 모델 학습(70%)
model.fit(X_train, y_train)
# 모델 예측(70%. 시험범위)
y_train_predict= model.predict(X_train)
# 모델 예측(나머지 30%. 범위 밖)
y_test_predict= model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
# 모델 평가(70%. 시험범위)
print('R2 train: ',r2_score(y_train, y_train_predict))
# 모델 평가(나머지 30%)- R2
print('R2 test: ',r2_score(y_test, y_test_predict))
# 모델 평가(나머지 30%)- MAE
print('MAE test: ',mean_absolute_error(y_test, y_test_predict))
# 모델 평가(나머지 30%)- MSE
print('MSE test: ',mean_squared_error(y_test, y_test_predict))
# 모델 평가(나머지 30%)- RMSE
print('RMSE test: ', np.sqrt(mean_squared_error(y_test, y_test_predict)))
#
print(f"랜포 실행시간 {time.time()-start_time:.7} sec") #0.9798 ~1.9


# In[ ]:





# ### 2-3. 모델학습 및 평가- 그래디언트 부스팅 회귀
# GradientBoostingRegressor. p.293
# - decision tree 묶기

# In[29]:


# from sklearn import ensemble # help(ensemble) # regressor 검색. GradientBoostingRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# help(GradientBoostingRegressor) 
#(*, loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse'
# , min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0
# , min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None
# , warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0) 


# In[55]:


import time
start_time= time.time()

from sklearn.ensemble import GradientBoostingRegressor
# 학습자. 70% 학습, 70% 예측, 나머지 30% 예측
model= GradientBoostingRegressor(random_state=10)
model.fit(X_train, y_train)
y_train_predict= model.predict(X_train) # 결과가 y라서
y_test_predict= model.predict(X_test)

# Train 평가, test 평가[R2, MAE, MSE]
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("R2 train기본: ",model.score(X_train, y_train)) # <=> r2_score(y_train, model.fit(X_train, y_train) )
print("R2 test: ",r2_score(y_test, y_test_predict)) # <=> model.score(X_test, y_test)
print("MAE test: ",mean_absolute_error(y_test, y_test_predict))
print("MSE test: ",mean_squared_error(y_test, y_test_predict))
print(f"그래디언트부스팅 수행시간: {time.time()-start_time:.7} sec")


# In[ ]:





# ### 2-4. 모델학습 및 평가 - 익스트림 그래디언트 부스팅
# XGB Regressor. 흔히 말하는 xgboost. 성능 끝판왕이지만 시험때는 자제
# - 다수의 성능 떨어지는 분류기를 합쳤더니 성능이 좋아지더라

# In[ ]:


# !pip install xgboost # Anaconda 기본에도 미설치인데 시험장에 설치가 되어있을까?


# In[61]:


# import xgboost # help(xgboost) #class/ XGBRegressor, XGBClassifier. 특이한 게 package에 sklearn 이 있다
# from xgboost import XGBRegressor
# help(XGBRegressor) 
# (XGBModel, sklearn.base.RegressorMixin)  |  XGBRegressor(*
#, objective: Union[str, Callable[[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]], NoneType] 
#= 'reg:squarederror', **kwargs: Any) -> None


# 이걸로 안 됨 [xgboost DMatrix parameter `enable_categorical` must be set to `True`.](https://velog.io/@gibonki77/EX6-ray-tune-%EC%9C%BC%EB%A1%9C-XGBoost-%ED%95%98%EC%9D%B4%ED%8D%BC%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D%ED%95%98%EA%B8%B0)
# - model.fit() 에 넣는 설명이 없는데, 시험장에서는 category 없애는 방법으로 해야지

# [깊은 복사](https://wikidocs.net/16038) 원본 영향 없이 전체 복사
# - import copy
# - b = copy.deepcopy(a)

# In[79]:


print(X.columns[X.dtypes=='category'])
import copy
X_xgb= copy.deepcopy(X)
X_xgb['vs']= X_xgb['vs'].astype('int64')
X_xgb['am_manual']= X_xgb['am_manual'].astype('int64')
X_xgb['wt_class']= X_xgb['wt_class'].astype('int64')
print(X_xgb.columns[X_xgb.dtypes=='category'])


# import sklearn # help(sklearn) #model_selection
# from sklearn import model_selection # help(model_selection) # see also/ train_test_split
from sklearn.model_selection import train_test_split
# help(train_test_split) # example/ X_train, X_test, y_train, y_test= train_test_split(
#    ...     X, y, test_size=0.33, random_state=42)

xgboost_X_train, xgboost_X_test, xgboost_y_train, xgboost_y_test= train_test_split(X_xgb, Y, test_size= 0.3, random_state=10)


# In[134]:


import time
start_time= time.time()
import xgboost
from xgboost import XGBRegressor
# 학습자. 70% 학습, 70%에서 예측. 나머지 30%에서 예측
model= XGBRegressor()

model.fit(xgboost_X_train, xgboost_y_train)
# help(model.fit) # DMatrix 설명 없고, enable_category 에 대한 언급도 없다
# When categorical type is supplied, DMatrix parameter `enable_categorical` must be set to `True`. 
# Invalid columns:vs, am_manual, wt_class
y_train_predict= model.predict(xgboost_X_train)
y_test_predict= model.predict(xgboost_X_test)
# 70% 모델 평가, 나머지 30% 로 모델 평가[R2, MAE, MSE]
print('train R2: ',r2_score(xgboost_y_train, y_train_predict))
print('test R2: ',r2_score(xgboost_y_test, y_test_predict))
print('test MAE: ',mean_absolute_error(xgboost_y_test, y_test_predict))
print('test MSE: ',mean_squared_error(xgboost_y_test, y_test_predict))
print(f"XGB Regressor 수행시간: {time.time()-start_time:.7} sec")


# ### Regressor 결론
# 

# ##### Linear Regression
# - R^2 test:  0.2905203422802105
# - 차이 > 0에 가까울수록 좋다
# - MAE test:  2.209756946563721
# - MSE test:  7.047970919788388
# - RMSE test:  2.6548014840639946
# 
# ##### Random Forest
# - R2 train:  0.9809794570146911
# - R2 test:  0.45810313267565417
# - MAE test:  1.7127200000000264
# - MSE test:  5.383203480000051
# - RMSE test:  2.3201731573311615
# - 랜포 실행시간 1.788558 sec
# 
# ##### Gradient Boosting
# - R2 train r2_:  0.9999940057279647
# - R2 test:  0.2191733401187025
# - MAE test:  2.007699537572006
# - MSE test:  7.756732039260808
# - 그래디언트부스팅 수행시간: 0.08700275 sec
# 
# ##### XG Boosting
# - train R2:  0.9999999861571242
# - test R2:  0.2905203422802105
# - test MAE:  2.209756946563721
# - test MSE:  7.047970919788388
# - XGB Regressor 수행시간: 0.1259995 sec
