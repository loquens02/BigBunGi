#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part5-작업2유형-분석연습
# p.389~ / 22.6.19.일 22:22
# - 22.6.20.월 14:30~16:57 
# - predict 에서 Memory부족 에러나고 막힘. 8GB 넘게 먹어서 이 놋북에서 실행불가. 혹은 램 덜 먹게 조치??

# ### 문제1
# 고객 3500명에 대한 학습용 데이터(x_train, y_train)을 이용하여 성별 예측 모형을 만든 후, 이를 평가용 데이터(x_test)에 적용하여 얻은 2482명 고객의 성별 예측값(남자일 확률)을 다음과 같은 형식의 csv 파일로 생성하시오 (제출한 모델의 성능은 ROC-AUC 평가지표에 따라 채점)
# - y_train: 고객의 성별 데이터. 학습 데이터. 3500명
# - x_train, x_test: 고객의 상품 구매 속성. 학습 및 평가용
# 
# ##### 제출 형식
# - custid, gender
# - 3500, 0.267
# - 3501, 0.578
# - 3502, 0.885

# 1. loc에 컬럼 여럿 print(x_train.describe().loc[:,('총구매액', '환불금액')])
# 2. print 대신 display 로 보면 값-컬럼 정렬되있어서 보기 좋다
# 3. [column rename python](https://rfriend.tistory.com/468) 특정 컬럼만 콕 집어 바꿀 수 있다.  df.rename(columns = {"old": "new"}, inplace = True)
# 4. 메모리 부족 에러: 코랩이든 남의 컴이든 똑같을 것. 변수 개수 줄이고 각종 중간 변수들 생략하고, 에러나도 메모리 해제하게끔 조치해보자

# In[17]:


## 데이터 로드, null, dtypes 체크 #이상치 체크, 인코딩, 파생변수 # 스케일링 # 모델링, train score, test score # 형식에 맞게 출력
## gender 일 확률. proba


# !ls
import time
start_time= time.time()
import pandas as pd
pd.set_option('display.max_columns',None)

# help(pd.read_csv) # encoding #UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 8: invalid continuation byte
x_train= pd.read_csv('bigData/x_train.csv', encoding='CP949')
y_train= pd.read_csv('bigData/y_train.csv', encoding='CP949')
x_test= pd.read_csv('bigData/x_train.csv', encoding='CP949')
# display(type(x_train))
# display(x_train.shape, x_train.info(). x_train.head(2)) # AttributeError: 'NoneType' object has no attribute 'x_train'
# info() 뒤에 , 대신 . 을 찍었다
# display(x_train.shape, x_train.info(), x_train.head(2)) # 총 3500개. 환불금액 1205개. 주구매상품 주구매지점 object
# display(y_train.shape, y_train.info(), y_train.head(2)) # cust_id, gender. 제출형식에는 _ 없음에 유의
# display(x_test.shape, x_test.info(), x_test.head(2)) # 총 3500개. train-test 가 반반이네.

## null 처리. 환불 null 은 환불이 없었다고 가정. fillna(0)
# display(x_train['환불금액'].min() ,'---') # min: 5600. null 이 아닌 것 중 0인 것은 없다
# 틀림 x_train.loc[x_train.isnull().sum()!=0]
# display(x_train.loc[x_train.isnull().sum(1) !=0],'---')
# display(x_train.describe().loc[:,('총구매액', '환불금액')],'---')
# 환불금액이 null 인 것의 의미? 가정1. 적지 않았다. 가정2. 환불이 없었다. - 2295 개나 적지 않았다는걸 받아들이기 어려움
# 환불금액을 mean 값으로 대체할 경우 문제점: 총구매액보다 커질 수 있다.
x_train['환불금액']= x_train['환불금액'].fillna(0)
# display(x_train.loc[x_train.isnull().sum(1)!=0].describe(),'---')
# display(y_train.loc[x_train.isnull().sum(1)!=0],'---') # y_train 에는 null 이 없다

## dtypes 체크 - 숫자인데 문자로 되어있는 건 아니 보인다
# display(x_train.columns[x_train.dtypes==object])
# display(x_train.loc[:,x_train.dtypes==object].iloc[:,0].value_counts().count()) # 42종 카테고리
# display(x_train.loc[:,x_train.dtypes==object].iloc[:,1].value_counts().count()) # 24종 카테고리
# display(x_train.columns[x_train.dtypes!=object])
# display(x_train.loc[:,x_train.dtypes!=object])
# display(x_train.describe())

## 이상치 보기. 구매액이 매우 크다고 해서 절삭하는 건 말이 안 된다고 본다.
# 대신 boundary 바깥의 값을 파생변수에 이용할 수도?
def outlierCheck(data):
    dataNum= data.loc[:,data.dtypes!=object] # 숫자만
    
    desc= dataNum.describe()
    min1= desc.loc['min']
    max1= desc.loc['max']
    std= desc.loc['std']
    mean= desc.loc['mean']
    maxBoundary= mean+1.5*std
    minBoundary= mean-1.5*std
#     print("maxBoundary", maxBoundary, '---')
#     print("minBoundary", minBoundary, '---')
    
#     print((dataNum>maxBoundary).sum(0), '---') # 큰 이상치 수백개
#     print((dataNum<minBoundary).sum(0), '---') # 작은 이상치 없음
    
    return minBoundary, maxBoundary

minB, maxB= outlierCheck(x_train)

## 파생변수. 주말방문 경험 유무? 1회 최대구매액 구간- boundary 밖 & median? 환불여부?
# 주말방문여부, 환불여부, 최대구매액많은편
# display(x_train.head(3))
conditionWeekend= x_train['주말방문비율']>0
x_train.loc[conditionWeekend, '주말방문여부']= 1
x_train.loc[~conditionWeekend, '주말방문여부']= 0
# display(x_train['내점일수'].value_counts(), '---')
# display(x_train['구매주기'].count(), '---')
# display(x_train[x_train['구매주기']<=4].count(), '---') # 기준값 모르겠다
# display(x_train.describe().loc[:,('총구매액','최대구매액','환불금액')], '---')
conditionRefund= x_train['환불금액']>0
x_train.loc[conditionRefund, '환불여부']= 1
x_train.loc[~conditionRefund, '환불여부']= 0
# print(maxB)
conditionMaxBuyOutlier= x_train['최대구매액']> maxB['최대구매액']
conditionMaxBuy= (x_train['최대구매액']<= maxB['최대구매액']) & (x_train['최대구매액'] > x_train['최대구매액'].median())
conditionMaxBuySmall= x_train['최대구매액'] <= x_train['최대구매액'].median()
x_train.loc[conditionMaxBuyOutlier, '최대구매액많은편']=2
x_train.loc[conditionMaxBuy, '최대구매액많은편']=1
x_train.loc[conditionMaxBuySmall, '최대구매액많은편']=0

# display(x_train.head(), '---')
# display(x_train['최대구매액많은편'].value_counts()) # 2는 207개

## 메모리 관리. 변수 지우기
x_train.drop(columns='환불금액', inplace=True)

## 책- 상관관계
# display((x_train.corr() < -0.7) |(x_train.corr()>0.7))
# => (총구매액~최대구매액) , (최대구매액많은편~최대구매액) 의 상관관계가 0.7 초과이므로, 최대구매액을 없애자
x_train.drop(columns='최대구매액', inplace=True)
# display((x_train.corr() < -0.7) |(x_train.corr()>0.7)) # 더이상 다중공선성 문제있는 것은 없다
# display(x_train.head(), '---')

## 인코딩. 범주화
# display(x_train.columns[x_train.dtypes==object], '---') # ['주구매상품', '주구매지점']
# import sklearn # help(sklearn)
# from sklearn import preprocessing # dir(preprocessing) # LabelEncoder
from sklearn.preprocessing import LabelEncoder
# help(LabelEncoder) # examples #fit_transform(self, y)
encoder= LabelEncoder()
# print(type(encoder.fit_transform(x_train['주구매상품']))) # ndarray. inplace 없다
x_train['주구매상품']= encoder.fit_transform(x_train['주구매상품']) # astype 바로 못 붙인다
x_train['주구매지점']= encoder.fit_transform(x_train['주구매지점'])
x_train['주구매상품']= x_train['주구매상품'].astype('category') # inplace 없다
x_train['주구매지점']= x_train['주구매지점'].astype('category')
# 파생변수 범주화
x_train['주말방문여부']= x_train['주말방문여부'].astype('category')
x_train['환불여부']= x_train['환불여부'].astype('category')
x_train['최대구매액많은편']= x_train['최대구매액많은편'].astype('category')


# display(x_train.dtypes, x_train.head(), '---')

## cust_id 를 custid 로 변경하고 index 로 만들기
# print(y_train.head())
# y_train 은 cust_id 로 되어 있어서, 미리 변경하지 말고 to_csv 하기 직전에 바꾸자 <<
## 나중에 되돌리는 건 df.reset_index(drop=False, inplace=True)
# y_train 으로 보내야 하니 버릴 수는 없고, 
# help(x_train) # rename( . df.rename(columns={"A": "a", "B": "c"}, inplace=Ture)
# x_train.rename(columns={"cust_id":"custid"}, inplace=True) # 콜론!
# print(x_train.columns[0])
# print(x_train.columns[1::])
# print(x_train.columns, '---')
x_train.index= x_train['cust_id']
x_train.drop(columns='cust_id', inplace=True)

# display(x_train.head(), x_train.dtypes, '---')


## 스케일링 - 시험때는 조건말고 눈으로 보고 컬럼 골라내서 for 문 돌리자
# + 조건으로 거니까 앞에 코드에서 변경이 있어도, 최소 변경만으로 수월하게 진행가능
# from sklearn import preprocessing # dir(preprocessing) # gender는 범주형이니, StandardScaler
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_trainNumCols= x_train.columns[(x_train.dtypes!=object) &(x_train.dtypes!='category')]
# print(x_trainNumCols)
# print(x_trainNumCols[x_train.describe().loc['max']> 10000])
bigNumCols= x_trainNumCols[x_train.describe().loc['max']> 30] # 30 근거: 범주컬럼 냅두고 싶어서 눈으로 정한 값
# scaler.fit_transform()
for col in bigNumCols:
    x_train[col]= scaler.fit_transform(x_train[[col]])
    
# display(x_train.head(), x_train.dtypes, '---')

beforeModeling= time.time()
print(f"모델링전까지 수행시간: {beforeModeling-start_time:.6} sec")

## 모델링
# import sklearn # help(sklearn) #package. ensemble
# from sklearn import ensemble # dir(ensemble) #RandomForestClassifier - gender
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators= 50) # help(model) #fit # 메모리 줄이기. n_estimators=100 default
# print(y_train.head()) # y_train 이 확률값이 아니다 ! 모델은 값으로 만들고 나중에 제출할 때만 _proba 해야!

## 메모리 이슈
# 에러! MemoryError: could not allocate 458752000 byte= 437MB. 여유분 약 3GB
# MemoryError: could not allocate 229376000 bytes= 218MB
# model.fit_transform(x_train, y_train) # RandomForestClassifier 에 fit_transform 이 없다.

model.fit(x_train, y_train) # <<<

# 틀림. proba는 제출할때. y_train_proba= model.predict_proba(x_train) 
# 여태 만진 게 train인데 왜 갑자기 test를. 굳이 train ㄴㄴ. 바로 test ㄱㄱ. y_train_predict= model.predict(x_train)
# 에러 y_test_predict= model.predict(x_test) # ValueError: could not convert string to float: '기타'

y_train_predict= model.predict(x_train) # <<<
display(pd.DataFrame(y_train_predict).head()) # <<<

print(model.score(y_train, y_train_predict)) # <<<



afterModeling= time.time()
print(f"모델링 시간: {afterModeling-beforeModeling:.6} sec")


# [Memory Error 해결방법](https://bskyvision.com/799)
# 1. 재실행 및 재부팅 후 실행 -> 마찬가지
# 2. batch 사이즈 줄이기 -> n_estimators, max_depth 정도?
# 3. 도중에 중단되도 메모리 해제하기 -> 파이썬에는 메모리 관리 명령어가 없다 (!!)
# > [python try except finally memory leak](https://stackoverflow.com/a/60454634)
# 4. 변수를 줄인다 -> 파생변수를 지우거나 원본변수를 지우거나
# 5. 페이징 파일을 증가시킨다 -> 내 PC에서는 가능하더라도, 시험장에서는 .?

# In[3]:


from sklearn.ensemble import RandomForestClassifier
# help(RandomForestClassifier) 
#(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, 
# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
# min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, 
#n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, 
# max_samples=None)

