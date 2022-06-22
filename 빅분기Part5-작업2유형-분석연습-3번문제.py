#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part5-작업2유형-분석연습-3번문제
# p.445~/ 자전거 대여량

# - 목적: 시간당 자전거 대여량. bike_y_train [datetime, count]. 10886개
# - 데이터: 고객의 자전거 대여 속성. bike_x_train, bike_x_test. 10886, 6493 개
# - 계절[1봄 2여름 3가을 4겨울], 공휴일[1공휴일 0아님], 근무일[1근무일 0아님], 날씨[1깨끗 2구르미 3눈비 4폭우우박]
# - 평가: R2 score

# 22.6.22.수 10:00

# 데이터로드, y_train 에서 datetime 떼기
# // null, 상관관계, 뭔가이상한거, 파생변수
# // 모델링, R2 score

# In[75]:


import pandas as pd
x_train= pd.read_csv('bigData/bike_x_train.csv', encoding='CP949')
y_train= pd.read_csv('bigData/bike_y_train.csv')
x_test= pd.read_csv('bigData/bike_x_test.csv', encoding='CP949')

def viewData(n=2):
    display('----  ---- x_train ----  ----  ')
    display(x_train.info(), x_train.head(n), x_train.tail(n))
    display('----  ---- y_train ----  ----  ')
    display(y_train.info(), y_train.head(n), y_train.tail(n))
    display('----  ---- x_test ----  ----  ')
    display(x_test.info(), x_test.head(n), x_test.tail(n))
    
# viewData() #null 이 없고 다 숫자. 그래도 혹시 모르니 value_counts()
x_train_date= x_train.iloc[:,0] # 2011-1-1 ~ 2012-12-19 23:00
x_test_date= x_test.iloc[:,0] # 2011-1-20 ~ 2012-12-31 23:00

## 시간당이니까 시간을 쓰면 더 좋지 않을까

## 책- pd.to_todatetime p.453 
## 일단 다 쪼개고, count 랑 결합해서 groupby(['기준시간'])['count'].sum() 으로 유의미한 변수인지 본다
## 쓸만한 것만 남기고 버림. 그 결과를 x_test 에도 적용
x_train['datetime']= pd.to_datetime(x_train['datetime'])
x_train['year']= x_train['datetime'].dt.year
x_train['month']= x_train['datetime'].dt.month
# 영향적음 # x_train['day']= x_train['datetime'].dt.day
x_train['hour']= x_train['datetime'].dt.hour
# 영향적음 # x_train['dayofweek']= x_train['datetime'].dt.dayofweek
# help(x_train['datetime' ] 없 .dt.dayofweek) # 오프라인에서 dayofweek 결과 못 찾음

# print(x_train['year'])
# print(x_train['year'].unique()) # 2011, 2012
# print(x_train['month'].unique()) # 1~12
# print(x_train['day'].unique()) # 19일 까지만
# print(x_train['hour'].unique()) # 0~23
# print(x_train['dayofweek'].unique()) # 0~6. 월~일

# featureDate= pd.concat([x_train, y_train], axis=1)
# display(featureDate)
# featureDate.columns[featureDate.dtypes]
# date 아니고 그냥 int64 # display(featureDate.dtypes) 
# dateCols= ['year','month','day','hour','dayofweek']
# for col in dateCols:
#     display(featureDate.groupby([col])['count'].sum())
## year: 2012년 우세. month: 5~10월 우세. day: 균일, hour: 17~19시 우세, dayofweek: 토요일이 근소하게 앞서나 대체로 균일
# 균일[day, dayofweek] 한 것은 영향이 적다고 판단하고 제외

x_test['datetime']= pd.to_datetime(x_test['datetime'])
x_test['year']= x_test['datetime'].dt.year
x_test['month']= x_test['datetime'].dt.month
x_test['hour']= x_test['datetime'].dt.hour

x_train.drop(columns='datetime', inplace=True)
x_test.drop(columns='datetime', inplace=True)
y_train.drop(columns='datetime', inplace=True) # 2011-1-1 ~ 2012-12-19 23:00

# viewData()

## 상관관계
# (x_train.corr() > 0.6) | (x_train.corr() < -0.6) # 온도~ 체감온도
x_train.describe()
def temperatureCate(data, col, newCol):
    desc= data[col].describe()
    q1= desc.loc['25%']
    q2= desc.loc['50%']
    q3= desc.loc['75%']
    low= data[col] < q1
    mid= (data[col]>=q1) & (data[col]<q2)
    high= (data[col]>=q2) & (data[col]<q3)
    veryHigh= data[col]>=q3
    
    data.loc[low, newCol]= 0
    data.loc[mid, newCol]= 1
    data.loc[high, newCol]= 2
    data.loc[veryHigh, newCol]= 3
    
    data[newCol]= data[newCol].astype('int64').astype('category')
    data.drop(columns=col, inplace=True)
    return data

x_train.drop(columns='온도', inplace=True)
x_test.drop(columns='온도', inplace=True)
x_train= temperatureCate(x_train, '체감온도', '체감온도구간')
x_test= temperatureCate(x_test, '체감온도', '체감온도구간')
x_train= temperatureCate(x_train, '습도', '습도구간')
x_test= temperatureCate(x_test, '습도', '습도구간')
x_train= temperatureCate(x_train, '풍속', '풍속구간')
x_test= temperatureCate(x_test, '풍속', '풍속구간')

# viewData() #구간 확인
def categorize(data, col):
    data[col]= data[col].astype('category')

categorize(x_train, '계절')
categorize(x_test, '계절')
categorize(x_train, '공휴일')
categorize(x_test, '공휴일')
categorize(x_train, '근무일')
categorize(x_test, '근무일')
categorize(x_train, '날씨')
categorize(x_test, '날씨')

# viewData()

## 스케일링 할 것도 없고, 이상한거 있나 확인만- 깔끔
## 책- 독립변수들이 대여량에 미치는 영향 보기
# cols= x_train.columns
# varnCount= pd.concat([x_train, y_train], axis=1)
# for col in cols:
    #display(x_train[col].value_counts())
    #display(varnCount.groupby([col])['count'].sum()) # 값에 따라 현저한 차이가 난다. => 유의미

## 모델링
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
model.fit(x_train, y_train.iloc[:,0])
y_train_predict= model.predict(x_train) # score
y_test_predict= model.predict(x_test) # csv

from sklearn.metrics import r2_score
print(r2_score(y_train, pd.Series(y_train_predict))) 
## R2 는 1에 가까울수록 좋다. 
## datetime 빼고 돌렸을 때: 0.3968
## datetime 적용 후: 0.9861

## 제출
# help(pd.concat)
resDF= pd.concat([x_test_date, pd.Series(y_test_predict.round(0)).astype('int64')], axis=1) #원본 y_train 과 비교
# print(resDF.columns)
resDF.rename(columns= {0:"count"}, inplace=True) # columns= 붙여야 적용
display(resDF)

# resDF.to_csv('data/bike-khj.csv', index=False)


# 22.6.22.수 10:48

# 더 해야 할 것
# - datetime 뜯어보기. cf p.453
# - y_test_predict 에 음수가 있는지 확인

# In[ ]:




