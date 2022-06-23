#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part5-작업2유형-분석연습-1번문제
# p.389~ 

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

# - 22.6.19.일 22:22 데이터 로드만.
# - 22.6.20.월 14:30~16:57 predict 에서 Memory부족 에러나고 막힘. 8GB 넘게 먹어서 이 놋북에서 실행불가. 혹은 램 덜 먹게 조치??
# > n_estimators= 10 으로 해결
# - 22.6.20.월 18:00~22:30 ValueError: could not broadcast input array from shape (3500,3500) into shape (3500,) 및 #ValueError: multiclass-multioutput format is not supported
# > y_test_proba 의 형태가 (2,3500,3500) 꼴로 나온다. 왜지

# y_train에 index가 포함된 경우
# #### predict 는 슬라이싱 하지 말아야 하고
# - model.fit(x_train, y_train) #display(pd.DataFrame(y_train_predict).head()) # <<< 2열 (인덱스,예측 결과값)
# - y_test_predict= model.predict(x_test) 
# 
# #### proba 는 슬라이싱 해서 gender 만 내도록 해야 한다.
# - model.fit(x_train, y_train.iloc[:,1]) # <<< 없는 cust_id 가 방해가 되나 싶어서 1열만 내도록 학습
# - y_test_proba= model.predict_proba(x_test) # <<< train 으로 만든 모델에 x_test 넣기. 결과 제출용으로 필수
# 
# #### 애초에 y_train 에서 index 부분은 drop 하고 따로 저장했어야?

# ##### error 회피하려다 초가삼간 태운다
# 그렇게 추정한 각 정보는 누구 id 것인데?
# > id 붙이면 해결!. train 으로 model 만들적부터 오름차순으로만 되어있으면 가능

# ### 주석 싹싹 지우고 다시2
# 22.6.21 15:44/ y_train 에서 id 빼고 학습시켜야 한다. concat 은 Series 끼리만 []로 가능

# In[20]:


## 데이터 로드, null, dtypes 체크 #이상치 체크, 인코딩, 파생변수, 상관관계, 변수 드랍, x_test에도 전처리 
## 스케일링 # 모델링, train score, test score # id붙이기, to_csv

## gender 일 확률. proba
import time
start_time= time.time()
import pandas as pd
pd.set_option('display.max_columns',None)
x_train= pd.read_csv('bigData/x_train.csv', encoding='CP949')
y_train= pd.read_csv('bigData/y_train.csv', encoding='CP949')
x_test= pd.read_csv('bigData/x_test.csv', encoding='CP949') # Wow! x_test 넣어야할 것을 x_train 넣었었다.

## null 처리. 환불 null 은 환불이 없었다고 가정. fillna(0)
x_train['환불금액']= x_train['환불금액'].fillna(0)
x_test['환불금액']= x_test['환불금액'].fillna(0)

## dtypes 체크 - 숫자인데 문자로 되어있는 건 아니 보인다

## 이상치 보기. 구매액이 매우 크다고 해서 절삭하는 건 말이 안 된다고 본다.
def outlierCheck(data):
    dataNum= data.loc[:,data.dtypes!=object] # 숫자만
    desc= dataNum.describe()
    min1= desc.loc['min']
    max1= desc.loc['max']
    std= desc.loc['std']
    mean= desc.loc['mean']
    maxBoundary= mean+1.5*std
    minBoundary= mean-1.5*std
    
    return minBoundary, maxBoundary

minB, maxB= outlierCheck(x_train)
minB_test, maxB_test= outlierCheck(x_test)

## 파생변수. 주말방문 경험 유무? 1회 최대구매액 구간- boundary 밖 & median? 환불여부?
conditionWeekend= x_train['주말방문비율']>0
x_train.loc[conditionWeekend, '주말방문여부']= 1
x_train.loc[~conditionWeekend, '주말방문여부']= 0

conditionRefund= x_train['환불금액']>0
x_train.loc[conditionRefund, '환불여부']= 1
x_train.loc[~conditionRefund, '환불여부']= 0

conditionMaxBuyOutlier= x_train['최대구매액']> maxB['최대구매액']
conditionMaxBuy= (x_train['최대구매액']<= maxB['최대구매액']) & (x_train['최대구매액'] > x_train['최대구매액'].median())
conditionMaxBuySmall= x_train['최대구매액'] <= x_train['최대구매액'].median()
x_train.loc[conditionMaxBuyOutlier, '최대구매액많은편']=2
x_train.loc[conditionMaxBuy, '최대구매액많은편']=1
x_train.loc[conditionMaxBuySmall, '최대구매액많은편']=0

## 파생변수- x_test
conditionWeekend= x_test['주말방문비율']>0
x_test.loc[conditionWeekend, '주말방문여부']= 1
x_test.loc[~conditionWeekend, '주말방문여부']= 0

conditionRefund= x_test['환불금액']>0
x_test.loc[conditionRefund, '환불여부']= 1
x_test.loc[~conditionRefund, '환불여부']= 0

conditionMaxBuyOutlier= x_test['최대구매액']> maxB_test['최대구매액']
conditionMaxBuy= (x_test['최대구매액']<= maxB_test['최대구매액']) & (x_test['최대구매액'] > x_test['최대구매액'].median())
conditionMaxBuySmall= x_test['최대구매액'] <= x_test['최대구매액'].median()
x_test.loc[conditionMaxBuyOutlier, '최대구매액많은편']=2
x_test.loc[conditionMaxBuy, '최대구매액많은편']=1
x_test.loc[conditionMaxBuySmall, '최대구매액많은편']=0

## 메모리 관리. 변수 지우기
x_train.drop(columns='환불금액', inplace=True)
x_test.drop(columns='환불금액', inplace=True)

## 책- 상관관계
x_train.drop(columns='최대구매액', inplace=True)
x_test.drop(columns='최대구매액', inplace=True)

## 인코딩. 범주화
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
x_train['주구매상품']= encoder.fit_transform(x_train['주구매상품']) # astype 바로 못 붙인다
x_train['주구매지점']= encoder.fit_transform(x_train['주구매지점'])
x_train['주구매상품']= x_train['주구매상품'].astype('category') # inplace 없다
x_train['주구매지점']= x_train['주구매지점'].astype('category')
x_train['주말방문여부']= x_train['주말방문여부'].astype('category')
x_train['환불여부']= x_train['환불여부'].astype('category')
x_train['최대구매액많은편']= x_train['최대구매액많은편'].astype('category')

## 인코딩. 범주화- x_test
x_test['주구매상품']= encoder.fit_transform(x_test['주구매상품'])
x_test['주구매지점']= encoder.fit_transform(x_test['주구매지점'])
x_test['주구매상품']= x_test['주구매상품'].astype('category')
x_test['주구매지점']= x_test['주구매지점'].astype('category')
x_test['주말방문여부']= x_test['주말방문여부'].astype('category')
x_test['환불여부']= x_test['환불여부'].astype('category')
x_test['최대구매액많은편']= x_test['최대구매액많은편'].astype('category')


## 그냥 drop 시키고 cust_id 만 따로 저장
x_train_custid= x_train['cust_id']
x_train.drop(columns='cust_id', inplace=True)

x_test_custid= x_test['cust_id']
x_test.drop(columns='cust_id', inplace=True)

## 스케일링 - 시험때는 조건말고 눈으로 보고 컬럼 골라내서 for 문 돌리자
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x_trainNumCols= x_train.columns[(x_train.dtypes!=object) &(x_train.dtypes!='category')]
bigNumCols= x_trainNumCols[x_train.describe().loc['max']> 30] # 30 근거: 범주컬럼 냅두고 싶어서 눈으로 정한 값
for col in bigNumCols:
    x_train[col]= scaler.fit_transform(x_train[[col]])
    
## 스케일링 - x_test
x_testNumCols= x_test.columns[(x_test.dtypes!=object) &(x_test.dtypes!='category')]
bigNumCols= x_testNumCols[x_train.describe().loc['max']> 30] # 30 근거: 범주컬럼 냅두고 싶어서 눈으로 정한 값
for col in bigNumCols:
    x_test[col]= scaler.fit_transform(x_test[[col]])
    
beforeModeling= time.time()
print(f"모델링전까지 수행시간: {beforeModeling-start_time:.6} sec")

## 모델링
from sklearn.ensemble import RandomForestClassifier
## 메모리 이슈 -  여분이 7.9GB 인데 랜포 40으로 돌리면 8.5GB 를 드신다. 랜포말고 DecisionTree는?
model= RandomForestClassifier(n_estimators= 10) # help(model) #fit # 메모리 줄이기. n_estimators=100 default. 10으로 해결.


## id 가 포함된 채로 y_train 목표로 학습(fit)시킬 경우, y_test_predict 의 id 에는 중복값이 들어가있다. y_test_proba는 (2,2438,3500)이 되고.
model.fit(x_train, y_train.iloc[:,1]) # y_test_predict 용 모델도 gender 만 넣고
y_test_predict= model.predict(x_test) 
y_test_proba= model.predict_proba(x_test) # <<< 결과 제출용


## help(pd.concat) # pd.concat([series1, series2], ignore_index=True, axis=1) # y_test_predict 는 ndarray
# display(pd.concat([x_test_custid, pd.Series(y_test_predict)], ignore_index=True, axis=1)) 

## x_test_predict 와 x_test_proba 를 비교하여, proba 의 어느 열을 남길지 결정
## > predict 값이 1(남자) 인 게, proba의 2번째 컬럼[1]
## 데이터 설명에서 gender==1 이 남자.

# print(f"y_test_proba {y_test_proba} \ny_test_proba[0] {y_test_proba[0]}")
# display(pd.DataFrame(y_test_proba).head()) # <<< columns=[0,1], 첫 행= [0.5, 0.5]
# display(pd.DataFrame(y_test_proba).info())
# display(pd.concat([ x_test_custid, pd.DataFrame(y_test_proba).iloc[:,1] ], axis=1).rename(columns={"cust_id":"custid", 1:"gender"}))
genderManProba= pd.concat([ x_test_custid, pd.DataFrame(y_test_proba).iloc[:,1] ], axis=1).rename(columns={"cust_id":"custid", 1:"gender"})
# print(y_train)

# help(pd.DataFrame.to_csv) # index
genderManProba.to_csv('data/김형준-1번-proba.csv', index=False)

## score 는 필수사항은 아니므로 일단 보류
from sklearn.metrics import roc_auc_score

afterModeling= time.time()
print(f"모델링 시간: {afterModeling-beforeModeling:.6} sec")
print(f"전체 수행 시간: {afterModeling-start_time:.6} sec") # 노트북 빠를때 0.313999 sec


# In[74]:


# help(pd.concat) # pd.concat([s1, s2], ignore_index=True, axis=1)


# In[ ]:





# In[ ]:




####  ####  ####  ####  ####  ####  ####  ####  
####  ####  ####  ####  ####  ####  ####  ####  


# In[ ]:





# # 실제 시험환경에서 친 코드
# - https://dataq.goorm.io/exam/116674/%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/3
# > X_test['주구매지점']= encoder.fit_transform(X_test['주구매지점']) # 여기서 상품으로 바꿔야되는데 지점으로 냅뒀다. ctrl CV의 폐해
# 
# - lightgbm 사용
# > 랜포로 돌렸을 때: 0.313999 sec <p>
# lightgbm 으로 돌렸을 때: 0.3529975 sec

# In[22]:


#####
## 원핫 vs 레이블인코더 - 원핫 쓰면 train과 test 컬럼수가 달라지는 경우가 생긴다. 메모리 소모가 훨씬 많아지기도 하고. 안전하게 레이블인코더 ㄱㄱ

## 여러 컬럼 한번에 인코딩하는 방법
floatCols= X_train.select_dtypes(include='float64').columns
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
X_train[floatCols]= scaler.fit_transform(X_train[floatCols]) # return type ndarray

## 단, scaler.fit_transform 인자는 DF만 받아서 컬럼 하나 넣을 때도 이렇게 해야 한다
X_train['최대구매액']= scaler.fit_transform(X_train[['최대구매액']])


## 파일 출력
## 제출. 방법1
# print(pd.concat([X_test_custid ,pd.Series(y_test_predict)], axis=1).rename(columns={0:'gender'}))

## 제출. 방법2. 시험장에서 제시한 코드. 깔끔 간단 에러안남.
print(pd.DataFrame({'cust_id': X_test_custid, 'gender': y_test_predict}))


# viewData() 만들었으면 train 만 보지 말고 test 까지 보렴
# ![image.png](attachment:image.png)

# In[2]:


# !pip install lightgbm


# In[18]:


import time
start_time= time.time()
# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
pd.set_option('display.max_columns',None)
X_train= pd.read_csv('bigData/x_train.csv', encoding='CP949')
y_train= pd.read_csv('bigData/y_train.csv', encoding='CP949')
X_test= pd.read_csv('bigData/x_test.csv', encoding='CP949')

## 데이터로드, null체크, 뭔가이상한거체크, 이상치~파생변수, 상관관계, 변수선택
## 범주화, 스케일링
## 모델링. lightgbm 써보기. 하이퍼파라미터 튜닝
pd.set_option('display.max_columns',None) # 먹히는거 확인
# help(pd)
def viewData(n=2):
	print('----  ----  X_train  ----  ----  ')
	print(X_train.info(), X_train.head(n), X_train.tail(n))
	print('----  ----  X_test  ----  ----  ')
	print(X_test.info(), X_test.head(n), X_test.tail(n))
	print('----  ----  y_train  ----  ----  ')
	print(y_train.info(), y_train.head(n), y_train.tail(n))

X_train_custid= X_train['cust_id'] #3500
X_test_custid= X_test['cust_id'] #2482
X_train.drop(columns='cust_id',inplace=True)
X_test.drop(columns='cust_id',inplace=True)
y_train.drop(columns='cust_id',inplace=True)

# viewData()

# print(X_train[X_train.isnull().sum(1)!=0])
# print(X_train['환불금액'].describe())
# print(X_train[X_train.isnull().sum(1)==0].describe()) # null 아닌 것중 min 이 0이 아니다
X_train['환불금액'].fillna(0, inplace=True)
X_test['환불금액'].fillna(0, inplace=True)

# viewData()

# print(X_train.columns[X_train.dtypes==object])
# print(X_train.select_dtypes(include=object).columns)
# print((X_train.corr()>0.6) | (X_train.corr()< -0.6)) # 총구매액이랑 공선성 걸리는 게 많다. 총구를 버리자
X_train.drop(columns='총구매액', inplace=True)
X_test.drop(columns='총구매액', inplace=True)

# viewData()

# from sklearn import preprocessing # print(dir(preprocessing))
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
# help(encoder.fit_transform) # 정보가 거의 없다
# print(type(encoder.fit_transform(X_train['주구매지점'])))
X_train['주구매지점']= encoder.fit_transform(X_train['주구매지점'])
X_test['주구매지점']= encoder.fit_transform(X_test['주구매지점'])

# print(type(pd.get_dummies(X_train, drop_first=True))) # dataframe
# X_train= pd.get_dummies(X_train, drop_first=True) #인자는 통째 넣어도 object만 골라주고, 반환은 DF.

X_train['주구매상품']= encoder.fit_transform(X_train['주구매상품'])
X_test['주구매상품']= encoder.fit_transform(X_test['주구매상품'])

# X_test= pd.get_dummies(X_test, drop_first=True)

# viewData() #train 과 test 컬럼수가 다르다. get_dummies 문제가 좀 많은데?

## 스케일링
# from sklearn import preprocessing #print(dir(preprocessing))
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
# print(type(scaler.fit_transform(X_train[floatCols]))) #ndarray
# print(scaler.fit_transform(X_train[floatCols]))
# print(X_train.select_dtypes(include='float64').columns) #['환불금액', '내점당구매건수', '주말방문비율']
floatCols= X_train.select_dtypes(include='float64').columns
X_train[floatCols]= scaler.fit_transform(X_train[floatCols])
X_train['최대구매액']= scaler.fit_transform(X_train[['최대구매액']]) # 최대구매액 int64 인데 빠트림

floatCols= X_test.select_dtypes(include='float64').columns
X_test[floatCols]= scaler.fit_transform(X_test[floatCols])
X_test['최대구매액']= scaler.fit_transform(X_test[['최대구매액']]) # 최대구매액 int64 인데 빠트림

# viewData() 

## test 도 섞어서 만들건지. model_selection. train_test_split
from sklearn.model_selection import train_test_split
# help(train_test_split)

## 모델링
# import lightgbm # print(dir(lightgbm)) # LGBMClassifier
# 좀더 큰 친구다 model= lightgbm() # help(lightgbm) #잘린다
from lightgbm import LGBMClassifier
# help(LGBMClassifier) #  max_depth: int = -1, learning_rate: float = 0.1, n_estimators: int = 100, random_state
model= LGBMClassifier() # random_state= 10
# viewData()
model.fit(X_train, y_train.iloc[:,0]) # 1d array 를 원하네. 여기까지 정상 실행
# help(model)
y_train_predict= model.predict(X_train) # score 알아보는 용도
y_test_predict= model.predict(X_test) # 제출용

from sklearn.metrics import roc_auc_score
# help(roc_auc_score) #roc_auc_score(y, y_pred, average=None)
# print(roc_auc_score(y_train, y_train_predict)) # 0.8589938430363963

# print(type(y_test_predict)) # ndarray
## 제출. 방법1
# print(pd.concat([X_test_custid ,pd.Series(y_test_predict)], axis=1).rename(columns={0:'gender'}))

## 제출. 방법2. 시험장에서 제시한 코드
print(pd.DataFrame({'cust_id': X_test_custid, 'gender': y_test_predict}))


# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)

print(f"실행시간 {time.time() - start_time:.7} sec") # 노트북 빠를 때 0.3529975 sec # 노트북 느릴때. 0.8639154 sec


# In[90]:


###############################################################################