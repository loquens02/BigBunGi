#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part5-작업2유형-분석연습-2번문제
# p.421~ / 분류모델

# - 데이터: titanic_y_train, titanic_x_train, titanic_x_test
# - 설명: 티켓등급=객실등급, 형제자매배우자수=해당 승객과 같이 탑승한 형제/자매/배우자인원수, 부모자식수=해당 승객과 같이 탑승한 부모와 자식의 인원수, 선착장[C 프랑스 셰르부르, Q 영국 퀸스타운, S 영국 사우샘프턴]
# - 결과: 생존여부
# - 제출형식: PassengerId, Survived
# 
# > 892, 0
# 
# > 893, 1

# 22.6.21 ~ / 22.6.22 08:40~

# 데이터로드,
# 탐색[y_train, NULL, 이상치, 상관관계], 
# 파생변수, 스케일링, 
# 모델링[test roc score, proba], 
# 파일로 내보내기

# In[146]:


import time
start_time= time.time()

import pandas as pd

#### 전처리
## 데이터로드 및 탐색[y_train id포함, null은 train-test 상이]
x_train= pd.read_csv('bigData/titanic_x_train.csv', encoding='CP949')
y_train= pd.read_csv('bigData/titanic_y_train.csv')
x_test= pd.read_csv('bigData/titanic_x_test.csv', encoding='CP949')

## diplay 덩어리
pd.set_option('display.max_columns',None)
def viewData(n=2):
    print('----------- x_train --------')
    display(x_train.info(), x_train.head(n), x_train.tail(n))
    print('----------- y_train --------')
    display(y_train.info(), y_train.head(n), y_train.tail(n))
    print('----------- x_test --------')
    display(x_test.info(), x_test.head(n), x_test.tail(n)) 
    
# viewData()
# x_train: #(891,11) null[나이 714, 객실번호 204, 선착장 889], object[승객이름, 성별, 티켓번호, 객실번호, 선착장]
# y_train: # (891,2) PassengerId, Survived
# x_test: # (418,11) null[나이 332, 운임요금 417, 객실번호 91]

## ID 분리
# print(x_train.tail()['PassengerId'])
y_train_ID= x_train['PassengerId'] #1,2,3, ..., 889,890,891 <=> y_train['PassengerId']
y_test_ID= x_test['PassengerId'] # 892, 893, ... , 1308, 1309
x_train.drop(columns='PassengerId', inplace=True)
x_test.drop(columns='PassengerId', inplace=True)
y_train.drop(columns='PassengerId', inplace=True)

# viewData()
# ID 빠진 것 확인

## 파생변수
# 승객이름 첫 어절(= 성= 가족) 만 남기고 나머지 삭제. LabelEncoder
# 부모자식수 0 있나? #print(x_train['부모자식수'].unique()) # 0 1 2 3 4 5 6 # 혼자 탄 사람 vs 같이 탄사람

## NULL 및 좀색다른값들 처리 - train-test 같이 및 겸사겸사 파생변수도
## 나이: 다른 거보다가 28살 자꾸 보이길래 봤더니 28살 뿐이다. 컬럼 버리자. (원안.100개쯤 없는데, median 으로 대체)
## 객실번호: 거의 대부분 없어서 "0"로 채우고, value_counts, 글자+숫자 중에 글자만 떼서 파생변수, LabelEncoder
## 선착장: x_train 에만 2개 없는데, value_counts 중에 많은 걸로 채우던지
## 운임요금: x_test 에만 1개 없는데, 같은 티켓등급 중에서 운임요금 median 으로 채우던지. 근데 1개뿐이라 대충해도 될듯
# x_train['나이'].value_counts() # 죄다 28살로 채워져있다. 컬럼 버리자. 이게 상수지 변수냐
x_train.drop(columns='나이', inplace=True)
x_test.drop(columns='나이', inplace=True)
# viewData() # 나이 빠진 것 확인
# display(x_train['객실번호'].fillna("").value_counts()) # 객실번호가 여러개인 게 있네. 첫 한글자만 떼고 파생변수 만들기
# x_train[x_train['객실번호']=='C23 C25 C27'] # 4명이 가족(승객이름 성이 같다). 티켓번호, 운임요금 같다. 나이 이상함.
# split ',' [0] 으로 성 떼면 될듯

x_train['객실번호'].fillna("0", inplace=True) # 예외처리하기 싫어서 "" 대신 "0"으로
# viewData() # display(x_train['객실번호'].value_counts()) # fillna 적용된거 확인
# print(x_train['객실번호'].map(lambda x:x[0])) # 첫글자 확인
x_train['객실번호']= x_train['객실번호'].map(lambda x:x[0])
# viewData() # 객실번호 바뀐거 확인
x_test['객실번호'].fillna("0", inplace=True)
x_test['객실번호']= x_test['객실번호'].map(lambda x:x[0])
# viewData() # 객실번호 바뀐거 확인

# print(x_train['선착장'].value_counts()) # S가 많다.
# x_train[x_train['선착장'].isnull()] #혼자왔고 티켓등급1, 운임요금 80
# x_train.groupby(['티켓등급', '선착장'])['선착장'].count() #티켓 1~3등급 전부 S가 많다. 부유여부 관계없이 대도시인듯
x_train['선착장'].fillna("S", inplace=True)
# viewData() # 이제 train 에는 null 없다

## 티켓등급~운임요금 관계보려면 운임요금을 구간별로 나눠야
# print(x_train.describe().loc[:,'운임요금']) # 0~512. 0?? 직원인가.
# display(x_train[x_train['운임요금']==0]) # 15명. 전원 남성. 간혹 티켓번호'LINE'. 선착장 S로 동일.
# display(x_train[x_train['티켓번호']=='LINE']) # 1티켓번호'LINE'. 4명. 전부 운임요금==0 에 해당
# descLoc1= x_train[x_train['운임요금']!=0].describe().loc[:,'운임요금']
# print(descLoc1) # 직원 빼고. 4.0 ~ 512. 사분위수[7.9, 14.5, 31.27]
# print(descLoc1.loc['25%']) # 25% 뽑히는거 확인

def conditionFee(data): # train-test 다 적용해야 하니
    descLoc= data[data['운임요금']!=0].describe().loc[:,'운임요금']
    q1= descLoc.loc['25%']
    q2= descLoc.loc['50%']
    q3= descLoc.loc['75%']
    
    # () 없으면 에러. or 아니다 and 로 해야!. value_counts() 로 보고 깨달음
    free= data['운임요금']<=0.0
    low= (data['운임요금']>0) & (data['운임요금']<q1)
    mid= (data['운임요금']>=q1) & (data['운임요금']<q2)
    high= (data['운임요금']>=q2) & (data['운임요금']<q3)
    veryHigh= data['운임요금']>=q3
    
    data.loc[free, '운임요금구간']= 0
    data.loc[low, '운임요금구간']= 1
    data.loc[mid, '운임요금구간']= 2
    data.loc[high, '운임요금구간']= 3
    data.loc[veryHigh, '운임요금구간']= 4
    
    # 운임요금구간은 대소비교가 가능해서 category 는 아니다
    data['운임요금구간']= data['운임요금구간'].astype('int64')
    data.drop(columns='운임요금', inplace=True)
    return data

x_train= conditionFee(x_train)

## 티켓등급 1이 비싼거. 고오오급
# display(x_train.groupby(['티켓등급','운임요금구간'])['운임요금구간'].count())
# 티켓1에는 요금구간4가 많고, 티켓2에는 요금구간2~3이 많고, 티켓3에는 요금구간1이 많다
# 요금구간 1,2 어디감? - conditionFee에서 조건을 & 로 줄 것을 | 로 잘못 줬었다
# x_train['운임요금구간'].value_counts()

# display(x_test[x_test['운임요금'].isnull()]) # 혼자왔고 티켓등급 3
# display(x_test['운임요금'].describe()) #티켓3등급이면 운임요금mid [q1,q2) 7.89~14.45
testFeeFill= (x_test['운임요금'].describe().loc['25%'] + x_test['운임요금'].describe().loc['50%'])/2
x_test['운임요금'].fillna(testFeeFill, inplace=True)
x_test= conditionFee(x_test) # 먼저 null 처리해야 가능 
# viewData() # 이제 train, test 둘다 null 없다
    
## 범주화1
## 성별. replace
## 선착장. replace

def replaceVar(data, col, target1, target2, target3=0):
    #display(data[col].value_counts())
    data[col]= data[col].replace(target1,0).replace(target2,1).astype('category')
    if target3 != 0:
        data[col]= data[col].replace(target1,0).replace(target2,1).replace(target3,2).astype('category')
    #display(data[col].value_counts())

replaceVar(x_train, '성별', 'male', 'female') # 혹시 제3의 성? 없다.
replaceVar(x_test, '성별', 'male', 'female')
# display(x_train['선착장'].value_counts()) # 3개. S C Q
replaceVar(x_train, '선착장', 'S', 'C', 'Q')
replaceVar(x_test, '선착장', 'S', 'C', 'Q')

# viewData() # 바뀐거 확인

## 범주화2 및 파생변수
## 객실번호. LabelEncoder
## 승객이름 -> 성 -> LabelEncoder
# 이거까지 했다간 죄다 category 될 것 같아서 제외
## 형제자매배우자수 -> 여부
## 부모자식수 -> 여부

# display(x_train['객실번호'].value_counts()) #많다
# import sklearn # help(sklearn) # LabelEncoder 는 어디에 있을까
## from sklearn import preprocessing # Sacler, Encoder # dir(preprocessing)
# from sklearn import model_selection # split
# from sklearn import metrics #score
# from sklearn import ensemble #분류 및 예측

## replace 할 시간에 이거 만들고 넣는게 훨씬 빠르겠다.
from sklearn.preprocessing import LabelEncoder
# x_train['객실번호']= encoder.fit_transform(x_train['객실번호'])

def labelEncodeVar(data, col, col2=''):
    encoder= LabelEncoder()
    if col2=='':
        col2=col
    data[col2]= encoder.fit_transform(data[col])
    data[col2]= data[col2].astype('category')
    #display(data[col2].unique()) #범주화 제대로 되었나 확인
    
labelEncodeVar(x_train, '객실번호')
labelEncodeVar(x_test, '객실번호')

# 승객이름 첫 어절 -> 성
x_train['승객이름']= x_train['승객이름'].map(lambda x:x.split(',')[0])
x_test['승객이름']= x_test['승객이름'].map(lambda x:x.split(',')[0])
labelEncodeVar(x_train, '승객이름')
labelEncodeVar(x_test, '승객이름')
# viewData() 확인

## 물론 친구끼리 왔을수도 있는데, 배가 침몰할 때 가족친지보다 진한 게 있을까.
# x_train['형제자매배우자수'].value_counts() # 형제자매 배우자 없는 사람이 대단히 많다
# x_train['부모자식수'].value_counts() #부모자식 없는 사람이 대단히 많다.
# x_train[(x_train['형제자매배우자수']==0) & (x_train['부모자식수']==0)] # 537명: 혼자 온 사람
conditionAlone= (x_train['형제자매배우자수']==0) & (x_train['부모자식수']==0)
x_train.loc[conditionAlone, '혼자왔니']= 1
x_train.loc[~conditionAlone, '혼자왔니']= 0
x_test.loc[conditionAlone, '혼자왔니']= 1
x_test.loc[~conditionAlone, '혼자왔니']= 0
x_train['혼자왔니']= x_train['혼자왔니'].astype('int64').astype('category')
x_test['혼자왔니']= x_test['혼자왔니'].astype('int64').astype('category')
# viewData()

## 티켓번호? '문자 숫자'. 티켓번호가 같은 사람들끼리 유의미
## 1. 파생변수 -> 문자만 떼서 만들기. 숫자만 있는건 0 같은거 채우고
## 2. 티켓번호 원본을 LabelEncoder
# print(x_train['티켓번호'].unique()) #죄다 문자열
labelEncodeVar(x_train, '티켓번호', '티켓번호전체범주화')
labelEncodeVar(x_test, '티켓번호', '티켓번호전체범주화')

import re #help(re) #sub(pattern, replacement결과물, string원본, count=0, flags=0)
# x_train['티켓번호'].map(lambda x:x.split(' ')[0]) # .replace('[0-9]+','0') 안 먹힌다
x_train['티켓번호문자범주화']= x_train['티켓번호'].map(lambda x: re.sub('\d+', '0', x.split(' ')[0])).astype('category')
labelEncodeVar(x_train, '티켓번호문자범주화')
# x_train['티켓번호'].value_counts() #숫자만 있는 게 661개고 나머지는 많지 않은데, 일단.
x_test['티켓번호문자범주화']= x_test['티켓번호'].map(lambda x: re.sub('\d+', '0', x.split(' ')[0])).astype('category')
labelEncodeVar(x_test, '티켓번호문자범주화')

x_train.drop(columns='티켓번호', inplace=True)
x_test.drop(columns='티켓번호', inplace=True)

# viewData()

## 스케일링 - 할 게 없다. 전부 int64 아니면 category

#### 모델링
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
model.fit(x_train, y_train.iloc[:,0]) # 1D array 이길 기대하길래
y_train_predict= model.predict(x_train) # score
y_test_predict= model.predict(x_test) # 제출

from sklearn.metrics import roc_auc_score
display(type(roc_auc_score(y_train, y_train_predict)))
display(roc_auc_score(y_train, y_train_predict)) # 0.9956

## 파일 내보내기. 꼭 df 형태 확인하기!
# display(pd.DataFrame(y_test_predict)) # id 없다
# help(pd.concat)
resDF= pd.concat([y_test_ID, pd.Series(y_test_predict)], axis=1)
## 컬럼만 콕 집어 바꾸는 방법: 숫자면 숫자를 key로 넣고, inplace 넣고, 대상 DF 컬럼이 원래 뭐였는지 확인해두기
# resDF.rename(columns={0:'Survived'}, inplace=True)
# help(resDF.rename)
resDF.columns=['PassengerId', 'Survived']
# display(resDF)
# pd.DataFrame(resDF).to_csv('data/titanic-survived-value-khj.csv', index=False) # 한번만

print(f"수행시간 {time.time()-start_time} sec") # 0.55 sec


# In[ ]:





# In[147]:


######  ######  ######  ######  ######  ######  ######  ######  


# 책에 있는데 하지 않은 것
# 1. train_test_split 하지 않음
# - test data를 학습에 섞는 게 몹쓸짓이라 생각했는데 이점이 있다
# - Y_TRAIN, Y_TEST 의 열에서 알아서 id 가 빠진다
# - 없던 Y_TEST 를 생성해서 모델 외 데이터로 검증할 수 있다
# 
# 2. xgboost 를 사용하지 않았다
# - 시험에서 쓰면 작성할 때는 될지라도 제출한 뒤 돌리면 1분이 넘어간다길래
# 
# 3. 선착장 get_dummies 대신 LabelEncoder 사용
# - 컬럼을 늘리면 메모리를 많이 먹더라. 노트북에서 안 돌아갈까봐. => 시험장에서도 걱정 
# - 자연어도 아닌데 굳이 컬럼을 쪼갤 필요가 있을까? (성능차이는 미확인)

# In[ ]:


randomState= 10
from sklearn.model_selection import train_test_split
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST= train_test_split(x_train, y_train, test_size=0.2,                                                    random_state= randomState)

from xgboost import XGBClassifier
model= XGBClassifier(n_estimators=100, max_depth=5, eval_metric= 'error',                      random_state= randomState)
model.fit(X_TRAIN, Y_TRAIN)


# 책에 더한 것
# 1. (나이 빼고) 다 drop 시킨 열에도 전처리를 수행했다
# - 시험때는 이러면 시간 초과할 듯. 아니면 전처리를 완전 숙달하거나.
# 
