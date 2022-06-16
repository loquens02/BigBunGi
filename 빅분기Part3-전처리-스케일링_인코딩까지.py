#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part3-전처리-스케일링
# p.252 / '22.6.15
# 1. 표준 크기변환. standard Scaling. 평균0 표준편차1
# 2. 최소최대 크기변환. Min-Max Scaling. 최소0 최대1
# 3. 로버스트 크기변환. Robust Scaling. 중앙값0 IQR 1

# In[4]:


# !ls


# In[130]:


import pandas as pd
import time


# 앞에서 해온 데이터로드 및 전처리

# In[129]:


start_time= time.time()
### 데이터 로드
data= pd.read_csv('bigData/mtcars.csv')
pd.set_option('display.max_columns', None) # head() 컬럼 다 보기
# data.head(2)
### 종속-독립변수 분리 및 문자열변수 제거(일단 시험용)
# data.info()
Y= data['mpg']
#Y.shape
X= data.drop(columns=['mpg','Unnamed: 0']) # Error X= data.drop(['mpg','Unnamed: 0']) # help(data.drop)
### null 채우기
# print(X.loc[X.isnull().sum(1)!=0])
# X.describe()
mean_cyl= X.describe().loc['mean']['cyl']
median_qsec=X.describe().loc['50%']['qsec']
X['cyl'].fillna(mean_cyl, inplace=True)# help(X['cyl'].fillna)
X['qsec'].fillna(median_qsec, inplace=True)
# print(X)
### 숫자형 같은데 문자형으로 둔갑한 것 찾기. 데이터타입 변경은 p.262 인코딩에서
### 뭔가 이상한 게 껴있는 것 찾고 바꾸기
# print(X.loc[:,X.dtypes==object].iloc[:,0].unique()) #문제없음
# print(X.loc[:,X.dtypes==object].iloc[:,1].unique()) #gear
# print(X['gear'].replace('*3','3').replace('*5','5'))#의도대로인지 확인
X['gear']= X['gear'].replace('*3','3').replace('*5','5')
#print(X['gear'])
# print(X.info())
# print(X)
### 이상치
#print(X.describe())
def outlierCheckByMeanStd(data):
    data_desc= data.describe()
    data_mean= data_desc.loc['mean']
    data_std= data_desc.loc['std']
    data_max= data_desc.loc['max']
    data_min= data_desc.loc['min']
    maxBoundary= mean+ 1.5*std
    minBoundary= mean- 1.5*std
    # print(data_max > maxBoundary)
    # print(data_min < minBoundary)
    # maxOverCols= data_desc.columns[data_max > maxBoundary]
    # minOverCols= data_desc.columns[data_min < minBoundary]
    overCols= data_desc.columns[(data_max > maxBoundary)|(data_min < minBoundary)]
    # print(f"maxOverCols: {maxOverCols}")
    # print(f"minOverCols: {minOverCols}")
    # print(f"overCols: {overCols}\n")
    return overCols

def outlierReplaceByMeanStd(data, column):
    data_desc= data.describe()
    mean= data_desc.loc['mean',column]
    std= data_desc.loc['std',column]
    # print(column,std)
    maxBoundary= mean+ 1.5*std
    minBoundary= mean- 1.5*std
    print(maxBoundary, minBoundary)
    maxOver= data[column] > maxBoundary
    minOver= data[column] < minBoundary
    # print(minOver, maxOver)
    # print(minOver.sum(), maxOver.sum())
    print(f"before: {data.loc[minOver|maxOver, column]}")
    if maxOver.sum() !=0:
        data.loc[maxOver, column]= maxBoundary
    if minOver.sum() !=0:
        data.loc[minOver, column]= minBoundary
    print()
    print(f"after: {data.loc[minOver|maxOver, column]}")
    return data

def outlierReplaceByIQR(data, column):
    data_desc= data.describe()
    data_Q3= data_desc.loc['75%', column]
    data_Q1= data_desc.loc['25%', column]
    data_IQR= data_Q3 - data_Q1
    maxBoundaryIQR= data_Q3 + 1.5*data_IQR
    minBoundaryIQR= data_Q1 - 1.5*data_IQR
    print(maxBoundaryIQR, minBoundaryIQR)
    maxOverIQR= data[column] > maxBoundaryIQR
    minOverIQR= data[column] < minBoundaryIQR
    print(f"before: {data.loc[minOverIQR | maxOverIQR , column]}")
    # print(maxOverIQR.sum())
    if maxOverIQR.sum() != 0:
        data.loc[maxOverIQR , column]= maxBoundaryIQR
    if minOverIQR.sum() != 0:
        data.loc[minOverIQR , column]= minBoundaryIQR
    print(f"After: {data.loc[minOverIQR | maxOverIQR , column]}")
    return data

# 일단 시험용이니.
# 문제1- 정수로 있어야 할 컬럼이 float 이 되었다. 책에서는 그냥 진행.
# 문제2- carb 기화기 개수가 6개나 8개인 차가 있을수도.
outlierCols= outlierCheckByMeanStd(X)
print(outlierCols)
for outlierCol in outlierCols:
    if outlierCol=='qsec': # byIQR 에서
        continue
    X= outlierReplaceByMeanStd(X, outlierCol) # 한 번만 실행! 

X= outlierReplaceByIQR(X, 'qsec')
    
print(X)
print(f"전처리 시간: {time.time()-start_time:.7} sec")


# In[126]:


# 바꾸는 함수 만드는 용
# column= 'wt'
# X_desc= X.describe()
# mean= X_desc.loc['mean',column] # 컬럼은 하나씩만 보자
# std= X_desc.loc['std',column]
# print(std)
# maxBoundary= mean+ 1.5*std
# minBoundary= mean- 1.5*std
# # 컬럼은 하나씩만 보자 maxOver= X_desc.loc['max'] > maxBoundary # minOver= X_desc.loc['min'] < minBoundary
# maxOver= X[column] > maxBoundary
# minOver= X[column] < minBoundary
# # print(minOver, maxOver)
# # print(minOver.sum(), maxOver.sum())
# if maxOver.sum() !=0:
#     print(X.loc[maxOver, column])
# if minOver.sum() !=0:
#     print(X.loc[minOver, column])

#체크용 함수 재료
# X_desc= X.describe()
# X_mean= X_desc.loc['mean']
# X_std= X_desc.loc['std']
# X_max= X_desc.loc['max']
# X_min= X_desc.loc['min']
# maxBoundary= mean+ 1.5*std
# minBoundary= mean- 1.5*std
# # print(X_max > maxBoundary)
# # print(X_min < minBoundary)
# maxOverCols= X_desc.columns[X_max > maxBoundary]
# minOverCols= X_desc.columns[X_min < minBoundary]
# print(f"maxOverCols: {maxOverCols}")
# print(f"minOverCols: {minOverCols}")

# IQR
# column= 'qsec'
# X_desc= X.describe()
# X_Q3= X_desc.loc['75%', column]
# X_Q1= X_desc.loc['25%', column]
# X_IQR= X_Q3 - X_Q1
# maxBoundaryIQR= X_Q3 + 1.5*X_IQR
# minBoundaryIQR= X_Q1 - 1.5*X_IQR
# print(maxBoundaryIQR, minBoundaryIQR)
# maxOverIQR= X[column] > maxBoundaryIQR
# minOverIQR= X[column] < minBoundaryIQR
# print(f"before: {X.loc[minOverIQR | maxOverIQR , column]}")
# # print(maxOverIQR.sum())
# if maxOverIQR.sum() != 0:
#     pass
#     #X.loc[maxOverIQR , column]= maxBoundaryIQR
# if minOverIQR.sum() != 0:
#     pass
#     #X.loc[minOverIQR , column]= minBoundaryIQR
# print(f"After: {X.loc[minOverIQR | maxOverIQR , column]}")


# In[ ]:





# ### 표준크기 변환. Standard Scaling
# p.254 / 평균0 표준편차1 / 주로 종속변수가 범주형

# In[15]:


# help(StandardScaler) # import 해야 비로소 보인다.

# 첫줄로 힌트 삼아 떠올려야. in module sklearn.preprocessing._data:
# class StandardScaler(sklearn

# from 패키지.module import Class - 에 있는 함수들


# In[5]:


from sklearn.preprocessing import StandardScaler


# In[68]:


print(type(X['qsec']))
print(type(X[['qsec']]))


# In[131]:


temp= X[['qsec']]


# In[132]:


# 표준변환 기능이 있는 scaler 객체 생성
scaler= StandardScaler()


# In[154]:


# scaler 에게 변수의 크기변환 요청 
# print(scaler.fit_transform(temp))
# print(type(scaler.fit_transform(temp))) # numpy.ndarray
# print(pd.DataFrame(scaler.fit_transform(temp))) # DataFrame
qsec_s_scaler= pd.DataFrame(scaler.fit_transform(temp))
print(qsec_s_scaler)

# qsec_s_scaler 변수의 기초통계량 확인
print(qsec_s_scaler.describe()) # std 가 1에 근접 => 표준크기변환이 잘 되었다.


# In[ ]:


# 기존 컬럼을 대체하고 싶다면.
# X['qsec']=pd.DataFrame(scaler.fit_transform(temp)) #qsec_s_scaler


# ### 최소-최대 크기변환: MinMaxScaler
# p.256 / 최소0 최대1 / 주로 종속변수가 연속형

# In[140]:


# Y # mpg: 연속형으로 보임


# In[146]:


from sklearn.preprocessing import MinMaxScaler
# help(MinMaxScaler) # import 전에는 못 찾는다


# In[151]:


# 특정 컬럼만 추출
temp= X[['qsec']]

# 최소최대변환 기능이 있는 scaler 객체 만들기
scaler= MinMaxScaler()


# In[153]:


# scaler 객체에게 temp의 크기변환 요청 & DataFrame으로 변환
qsec_m_scaler= pd.DataFrame(scaler.fit_transform(temp))
print(qsec_m_scaler)


# In[157]:


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
temp= X[['qsec']]
print(pd.DataFrame(scaler.fit_transform(temp)).describe())
# min 0 max 1 - MinMaxScaler 가 잘 적용되었다.


# ### 로버스트 크기변환. RobustScaler
# p.258 / 중앙값0 사분범위IQR 1 / 이상치 영향을 덜 받고 싶을 때

# In[160]:


from sklearn.preprocessing import RobustScaler
# help(RobustScaler)


# In[165]:


# 열 하나 추출해서 임시변수 만들고, RobustScaler 객체 만들기
temp= X[['qsec']]
scaler= RobustScaler()
# RobustScaler 로 하여금 temp 값 크기변환하도록 함
qsec_r_scaler= pd.DataFrame(scaler.fit_transform(temp))
print(qsec_r_scaler)


# In[166]:


# 중앙값 50% 0, IQR 1
print(qsec_r_scaler.describe())
qsec_r_scaler.describe().loc['75%'] - qsec_r_scaler.describe().loc['25%']


# In[ ]:





# ## 데이터 타입 변경
# p.261

# ### 수치형1 => 수치형2. astype()
# Series.astype(타입)
# - 타입: 'float64', 'int64' 등

# In[167]:


print(X.info())


# In[170]:


# origin= pd.read_csv('bigData/mtcars.csv')
# origin.info()

# hp, carb 는 원래 int64 였다. 
# cyl 은 엔진 기통수인데 원래부터 float64 인걸 보니 개수라고 꼭 int 일 것은 없을수도.


# In[171]:


X.head(2)


# In[178]:


# 개수가 적다고 꼭 범주형인 것은 아니다! # X.loc[:,X.dtypes==object].nunique()
X['gear'].value_counts() # gear: 전진기어 개수. 기어가 3~5개


# In[181]:


temp_time= time.time()
X['gear']= X['gear'].astype('int64')
print(f"type변경 {time.time()-temp_time}")


# In[183]:


X.dtypes


# ### 범주형=>수치형. 인코딩 encoding
# - one-hot encoding(100 010 001), label encoding(0,1,2,...)
# - 문자열로 된 것을 컴퓨터가 이해할만하도록 수치형으로 변경

# ##### one-hot encoding
# p.264

# In[197]:


# X.loc[:,X.dtypes==object].value_counts() # 컬럼 하나뿐이라
print(X.loc[:, X.dtypes==object].nunique()) # am 컬럼에 2 종류의 값
print(X.loc[:,X.dtypes==object].iloc[:,0].unique())


# In[194]:


# 간단한 방법- 2종류라서 2개 컬럼에 01 10 생성
print(pd.get_dummies(X['am']))


# In[200]:


# 간단 + 공간효율적인 방법. 하나가 0이면 다른건 1일 것이므로
# 종류가 3개라면 2개 컬럼 생성. 001 010 100 => 010 100. 2개만 있어도 남은 하나를 구분할 수 있다.
print(pd.get_dummies(X['am'], drop_first= True))


# ##### 따로 범주형 안 골라내고도 알아서 범주형만 골라서 one-hot encoding

# In[208]:


# help(pd.get_dummies) # drop_first
# print(X['am'].unique()) 
print(X['am'].head(4)) # ['manual' 'auto'] => 1 0
pd.get_dummies(X, drop_first=True)


# In[ ]:





# ##### 라벨 인코딩. Label Encoding
# p.269 / 트리 계열 분석에 좋다 - 대소비교로 쪼개기가 쉬울테니

# In[211]:


X.loc[:,X.dtypes==object].head(4)


# In[217]:


# import sklearn
# help(sklearn) # PACKAGE CONTENTS 에 preprocessing 있다. Examples 에 from sklearn.ㅁㅁ import ㄹㄹ 형식도 있고


# In[220]:


# from sklearn import preprocessing
# help(preprocessing) # CLASSES 에 다 들어있다. 클래스 이름을 다 안 외워도 된다! 외우는 게 3시간 안에 문제 푸는 데 좋겠지만
# StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


# In[221]:


from sklearn.preprocessing import LabelEncoder
# help(LabelEncoder)


# In[222]:


# LabelEncoder 객체에게 'am'컬럼 인코딩 시키기
encoder= LabelEncoder()
print(encoder.fit_transform(X['am'])) # manual 1, auto 0. 2종이라 01


# 임의로 3종 만들기 => 012 로 인코딩

# In[224]:


from sklearn.preprocessing import LabelEncoder
fruit= ['apple','banana','grape']
encoder= LabelEncoder()
fruitEnc= encoder.fit_transform(fruit)
print(fruit, fruitEnc)


# 복습 - '22.6.16 오전
# - 이상치 제거 시행착오에 3h 가량

# In[227]:


import pandas as pd
import time
# from sklearn import preprocessing
# help(preprocessing)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


# [python type to factor](https://stackoverflow.com/a/27023500) dtype="category"

# In[381]:


### 시간 측정, 데이터 로드, 종속-독립변수 분리
start_time= time.time()
import pandas as pd
data=pd.read_csv('bigData/mtcars.csv')
# print(data.info(), '-----'))
# print(data.head(), '-----'))
Y= data[['mpg']]
X= data.drop(columns=['mpg','Unnamed: 0'])
X.head()
### null 찾고 채우기, 이상한 값 찾고 바꾸기, 이상한 타입 바꾸기, 인코딩
# print(X.loc[X.isnull().sum(1) != 0], '-----'))
X['cyl'].fillna(X.describe().loc['mean','cyl'], inplace=True)
X['qsec'].fillna(X.describe().loc['mean','cyl'], inplace=True)
# print(X, '-----'))
# X.loc[:,X.dtypes==object].iloc[:,0].unique()
# print(X.loc[:,X.dtypes==object].iloc[:,1].unique(), '-----')) #'gear'
# print(X.info(), '-----')) 
X['gear']= X['gear'].replace('*3','3').replace('*5','5').astype('int64') # min 이상치에 'gear' 도 포함된다
# help(pd.get_dummies)
# print(pd.get_dummies(X, drop_first=True), '-----'))
# print(X['am'].unique(), '-----'))
# print(X, '-----'))
X= pd.get_dummies(X, drop_first=True) # min/max 이상치에 am_manual 도 포함된다. manual 1 , auto 0
# print(X, '-----')) # am_auto 뿐만 아니라, 기존 am 컬럼도 자동으로 지워진다.

# print(X.info(), '-----'))
### 이상치 교체(meanStd or IQR)
## describe -> mean +- 1.5*std , Q3 + 1.5*IQR 및 Q1 - 1.5*IQR -> def 컬럼 별. maxOver minOver. 찾는 것부터
# 같은 컬럼을 2번 돌리면 안 되기 때문에 True 라고 전부 loop 돌리면 곤란하다. 
# int64 를 float로? 눈으로 데이터보며 이상치 제거할지 말지 결정하자.
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
    minOverMs= min1 < minBms#조건을 잘못 걸었다. Boundary 보다 더 작은 게 이상치. minOverMs= min1 > minBms
    maxOverMs= max1 > maxBms #Boundary 보다 더 크면 이상치. maxOverMs= max1 < maxBms
#     print(minOverMs, '-----')) # drat, wt 만
#     print(maxOverMs, '-----')) # vs, am_maunal 빼고 다 => disp, drat 만. gear 제외
    # 여기부터 의심했어야 했는데. vs 01인데 왜 이상치로 잡나에 대해.
    
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

# 여긴 조건 맞게함. 서로 등호를 반대로 했으니 겹치는 게 없어서 아무것도 안 바꾼것.
# 뭘 바꿀지 이미 정했으므로 행 조건만 알면 된다.
def outlierReplace(data):
    minBms, maxBms, minBiqr, maxBiqr= outlierCheck(data)
    #행 조건
    # 각 열 내의 값이 Boundary Over 인지
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
        pass
        # print(f"BeforeMaxMS: {data.loc[ data[col]>maxBms[col],col]}, maxBms: {maxBms[col]:.7}")
        # data.loc[ data[col]>maxBms[col], col]= maxBms[col] # 대상 컬럼이 없다
    for col in minIQRcols:
#         print(f"BeforeMinIQR: {data.loc[ data[col]<minBiqr[col],col]}, minBiqr: {minBiqr[col]:.7}")
        data.loc[ data[col]<minBiqr[col], col]= minBiqr[col]
    for col in maxIQRcols:
#         print(f"BeforeMaxIQR: {data.loc[ data[col]>maxBiqr[col],col]}, maxBiqr: {maxBiqr[col]:.7}")
        data.loc[ data[col]>maxBiqr[col], col]= maxBiqr[col]
    return data

X= outlierReplace(X) # inplace 될 때도 있고, 안 될 때도 있어서 확실히 해두기.
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
ret2= rangeScaler('MinMax', X[['qsec']]) # min 0, max 1
ret2= rangeScaler('Standard', X[['qsec']]) # mean 0 근접, std 1 근접
ret2= rangeScaler('Robust', X[['qsec']]) # mid 0, IQR= Q3-Q1= 1
# print(ret2.describe(), '-----'))
# ret2.describe().loc['75%']-ret2.describe().loc['25%']
# print(ret2, '-----'))

## 범주형은 안 건드리고 싶다.
# help(X['am_manual'].astype) #Convert to categorical type
X['am_manual']= X['am_manual'].astype('category')
X['vs']= X['vs'].astype('category')
# print(X.info(), '-----'))

## 기존 데이터에 범위 적용
# print(X.describe(), '-----')) # 나중에는 print 가 잘 안 보인다. 지울 예정인건 표시해두자.
# targetCols= X.columns[(X.dtypes=='float64') | (X.dtypes=='int64')] # 탐탁지 않음.
targetCols= X.columns[(X.dtypes=='float64')] # 정수형, 범주형은 안 건드리고 싶었다.
# print(targetCols)
for col in targetCols:
    X[col]= rangeScaler('MinMax', X[[col]]) # 종속변수 연속형이면 주로 MinMax 
    # (범주형이면 Standard, 이상치 안 잡고 싶다면 Robust)

print(X, '-----')

## LabelEncoder 연습
face= ['happy','sad','soso']
# from sklearn import preprocessing
# help(preprocessing)
from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
# print(face, encoder.fit_transform(face), '-----'))


# In[327]:


# col= 'cyl'
# minBms, maxBms, minBiqr, maxBiqr= outlierCheck(X)
# print(minBms[col])
# X[col]
# print(f"BeforeMinMS: {data.loc[ data[col]<minBms[col],col]}")
# print(X['cyl']<minBms['cyl'])
# print(data.loc[minOverMs['cyl'], 'cyl'])


# In[325]:


# col= 'qsec'
# X[col]==X.describe().loc['max',col]


# In[ ]:





# In[ ]:





# ### 파생변수
# p.272 / '22.6.16 16:20

# In[ ]:




