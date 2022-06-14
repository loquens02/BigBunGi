#!/usr/bin/env python
# coding: utf-8
# # 데이터 분석 절차 체득하기2- 전처리
# ## 종속변수(결과,효과)와 독립변수(원인) 분리
# p.233

# In[234]:
import time
start_time= time.time()
# print(f"{time.time()-start_time} sec")

# In[235]:
import pandas as pd

# In[236]:
get_ipython().system('ls')

# In[237]:
data= pd.read_csv('bigData/mtcars.csv')

# In[238]:
data.info()

# In[239]:
X= data.drop(columns= 'mpg')
Y= data['mpg']

# In[240]:
print(X.shape)
print(Y.shape)

# In[241]:
X['cyl'].shape # null 껴있어도 shape 는 전체 개수 출력! - 혹시나 종속변수에 null 껴있어서 shape 맞출 걱정 ㄴㄴ

# In[242]:
start_time= time.time()
print("\n----= ----= real data ----=")
pd.set_option('display.max_columns', None)
print(X.head(2))
print("\n----= ----= null check ----=")
print(X[X.isnull().sum(1) != 0])
print("\n----= ----= 수치형 변수 ----=")
print(X.describe().columns)
print("\n----= ----= object에 수치형 껴있나 ----=")
objCols= X.loc[:,X.dtypes == object].columns
for idx, _ in enumerate(objCols):
    if idx==0:
        continue # 맨앞 것은 명백히 문자열이더라
    print(f"{_}\n{X[_].unique()}\n")
print("\n----= ----= 이상치: mean + 4*std < max 면 거의 확실. min도 보고----= ") # 그래도 눈으로 보는 게 특이 케이스를 더 잘 거를듯
print(X.describe().loc[['mean','min','max','std'],:]) # 400m 가는데 0.1초도 이상하지만 100초도 너무 이상한데?
print("\n----= ----= time check ----=")
print(f"{time.time()-start_time:.7f} sec")

# In[243]:
start_time= time.time()
pd.set_option('display.max_columns',None)
print(X.head())
print(X[X.isnull().sum(1) != 0])
print(X.describe().columns)
objCols= X.loc[:, X.dtypes == object].columns
for idx,_ in enumerate(objCols):
    if idx==0:
        continue
    print(f"{_}\n{X[_].unique()}\n")
print(X.describe().loc[['mean','min','max','std'], :])
print(f"{time.time() - start_time:.7f} sec")
# start_time= time.time()
# print("\n----= ----= real data ----=")
# pd.set_option('display.max_columns', None)
# print(X.head(2))
# print("\n----= ----= null check ----=")
# print(X[X.isnull().sum(1) != 0])
# print("\n----= ----= 수치형 변수 ----=")
# print(X.describe().columns)
# print("\n----= ----= object에 수치형 껴있나 ----=")
# objCols= X.loc[:,X.dtypes == object].columns
# for idx, _ in enumerate(objCols):
#     if idx==0:
#         continue # 맨앞 것은 명백히 문자열이더라
#     print(f"{_}\n{X[_].unique()}\n")
# print("\n----= ----= 이상치: mean + 4*std < max 면 거의 확실. min도 보고----= ") # 그래도 눈으로 보는 게 특이 케이스를 더 잘 거를듯
# print(X.describe().loc[['mean','min','max','std'],:]) # 400m 가는데 0.1초도 이상하지만 100초도 너무 이상한데?
# print("\n----= ----= time check ----=")
# print(f"{time.time()-start_time:.7f} sec")

# In[244]:
Y.head(5)
# ### 데이터 관찰 및 가공
# p.234
# 과정
# 1. 종속변수에 영향없는 불필요 변수(컬럼) 제거
# 2. 누락된 값 수정 및 제거
# 3. 잘못된 값 수정 및 제거
# 4. 이상값 조정
# 5. 각 숫자 범위 동일하게 맞추기
# 6. 적절한 데이터 타입으로 변경
# 7. 문자로 구성된 범주형 데이터를 숫자형으로 변경
# 8. 분석에 필요한 새로운 열(파생변수) 만들기
# ##### 영향없는 변수 제거

# In[245]:
print(X.head())

# In[246]:
X.drop(columns='Unnamed: 0', inplace= True)
print(X.head())

# In[247]:
# 혹은
# X= X.iloc[:, 1:] # 두 번째 열부터 끝까지 다시 저장
# ##### 결측값 처리

# In[248]:
print(X.isnull().sum())

# In[249]:
nullCheck= X.loc[X.isnull().sum(1)!=0, :]
print(nullCheck)
#print(type(nullCheck)) # DataFrame
# ##### 결측값 처리- 평균값 대치
# p.238

# In[250]:
# cyl
X_cyl_mean= X['cyl'].mean()
print(X_cyl_mean)

# In[251]:
X['cyl'].fillna(X_cyl_mean, inplace=True)

# In[252]:
nullCheck

# In[253]:
X['cyl']

# In[254]:
print(X.isnull().sum())
# ##### 결측값 처리- 중위값으로 대체

# In[255]:
X_qsec_median= X['qsec'].median()
print(X_qsec_median)

# In[256]:
# 지금은 하나 남았으니 괜찮지만, 컬럼 단위로 넣자!
# X.fillna(X_qsec_median)

# In[257]:
X['qsec'].fillna(X_qsec_median, inplace=True)
print(X.iloc[10,:])

# In[258]:
print(X.loc[X.isnull().sum(1)!=0, :]) # null 없어진 걸 확인

# In[259]:
print(X.isnull().sum())
# ##### 결측치 행 삭제
# 실전/시험 모두에서 이러면 안 되지만, 시험에서 따로 명시했을 때

# In[260]:
# X.dropna()
# ### 잘못된 값 올바르게 변경
# p.240

# In[261]:
objCols= X.loc[:,X.dtypes==object].columns
for _ in objCols:
    print(f"{_}: {X[_].unique()}")

# In[262]:
#print(X.replace('*3','3').replace('*5','5')) # replace 같은 건 DataFrame 에 함부로 적용하면 원복이 안 된다
print(X['gear'].replace('*3','3').replace('*5','5')) 

# In[263]:
# 위에서 출력을 확인하고, 의도대로 되었으면 바꾼다
X['gear']= X['gear'].replace('*3','3').replace('*5','5')
print(X['gear'])

# In[264]:
X['gear'].unique()
# ##### 이상치 처리
# p.242
# 1. 데이터 스케일링
# 2. 다른 값으로 교체
# - 사분범위(IQR= Q3 - Q1) 활용: Q3 + 1.5*IQR= Q3 + 1.5Q3 - 1.5Q1= 2.5Q3 - 1.5Q1

# In[265]:
X_describe= X.describe()
print(X_describe)
# #### 책 안 보고 하고 싶은대로 한 것
# 복잡해서 기억하기 어려울 듯

# In[266]:
X_Q3= X_describe.loc['75%']; print(X_Q3)
X_Q1= X_describe.loc['25%']; print(X_Q1)
X_IQR= X_describe.loc['75%'] - X_describe.loc['25%']; print(X_IQR)
# 이상치: Q3 + 1.5 * IQR 을 초과하는 것
# > 2.5*Q3 - 1.5Q1 을 초과하는 것. 왜 값이 다르지?? . Q3 +  ++++++++++++++++ !! 더하기다!
# 빼기 아니고 더하기! print(X_describe.loc['75%'] - 1.5* X_IQR)
print(X_describe.loc['75%'] + 1.5* X_IQR)
print(2.5*X_Q3 - 1.5*X_Q1)
outBoundary= 2.5*X_Q3 - 1.5*X_Q1 # 최대 경계값
# X_describe.loc[:,X_describe.loc['max'] >= outBoundary] # 굳이?
X_describe.loc['max'] >= outBoundary # 무슨 컬럼이 이상치가 있는지만 알면 충분
# 실데이터 어디에 이상치가 있는지 확인
# [dataframe compare series](https://stackoverflow.com/a/40889125) df.gt(s, axis=0)
# - lt
# gt
# le
# ge
# ne
# eq
# - axis 1 0 뭔지?
# X > outBoundary # deprecated 예정
X.ge(outBoundary, axis=1) # 컬럼 살리고 행을 기준으로 값 비교. isnull().sum(1) 에서도 컬럼 살리고 행 합할 때 '1' 쓴다.
# 전부 False 로 나옴. 의도한 게 아니다. X.ge(outBoundary, axis=0)
# axis=1 확인용
print(outBoundary['wt'])
X.iloc[16:18]
# help(X.ge) #axis : {0 or 'index', 1 or 'columns'}, default 'columns'
# X.ge(outBoundary, axis='columns')
X.ge(outBoundary)
X.loc[X.ge(outBoundary, axis='columns').sum(axis='columns') != 0]
# 뭘.. 하려고 했더라 >> [ max > 2.5Q3 - 1.5Q1 ]
# - 이상치 있어보이는 컬럼만 보기
# - 개별 값 말고 max 값만 Q3+1.5IQR 넘는 것
X_describe= X.describe()
outlierColsTF= X_describe.loc['max'] > 2.5*X_describe.loc['75%'] - 1.5*X_describe.loc['25%']
print(outlierColsTF)
# X.dtypes == object
# X.loc[:,X.dtypes == object] # 열 조건 되는거 재확인
outlierCols= X_describe.columns[outlierColsTF] # !!! T/F Columns 에서 Columns 이름 뽑아내는 방법
print(outlierCols)
# 안 되는 이유: X는 컬럼이 10개인데, X_describe는 8개라서. X.loc[:,outlierCols]
# 내가 뭘.. 하려고 했더라2
# 1. max 값이 최대경계값보다 큰 컬럼 알기
# 2. 그 컬럼 중에서
# 3. max 값이 들어있는 행만 보기
outBoundary= 2.5*X_describe.loc['75%'] - 1.5*X_describe.loc['25%']; print(outBoundary) # 최대 경계값
print()
outlierCols= X_describe.columns[X_describe.loc['max'] >= outBoundary]; print(outlierCols)
print()
X_out= X[outlierCols]
X_outlier_max= X_out[X_out.eq(X_describe.loc['max']).sum(1) !=0]
print(X_outlier_max)
# 이걸 시험장에서 어떻게 기억하누
# X[outlierCols][X[outlierCols].eq(X_describe.loc['max'], axis='columns').sum(axis='columns') !=0]
# 이렇게 간추리고도, 각 열에서 max 값이 뭔지 찾아야 한다.
# ### 책 대로 + 간편 계산
# p.246
# 1. 사분위수로 이상치 찾기
# - 최대경계값 = Q3 + 1.5IQR = 2.5Q3 - 1.5Q1
# - 각 열 max 값 >= 최대경계값. 각각 보기!
# ##### 이상치가 있을 것 같은 컬럼 확인

# In[267]:
X_desc= X.describe()
outBoundaryMax= 2.5*X_desc.loc['75%'] - 1.5*X_desc.loc['25%']
print(outBoundaryMax)
outColsTF= X_desc.loc['max'] >= outBoundaryMax
print(outColsTF)
# cyl, hp, wt, qsec, carb

# In[268]:
# Max 를 굳이 찾을 필요가 없다
# float equal 계산을 믿니?
# outColsMax= X_desc.loc['max'][outColsTF]
# print(outColsMax)
#print(type(outColsMax)) # Series
# for idx, _ in zip(outColsMax.index, outColsMax):
# 컬럼 별로 각각 최대경계값 이상인 행 찾기

# In[269]:
print(X.loc[X['cyl'] >= outBoundaryMax['cyl']])

# In[270]:
outCols= X_desc.columns[outColsTF]; print(outCols)
for col in outCols:
    print(f"{col}:\n{X.loc[X[col] >= outBoundaryMax[col]]}\n")
# ### 이상치 처리- 사분위수- cyl, hp
# p.246. 바로 위 결과를 눈으로 보며.

# In[271]:
X.loc[14,'cyl']

# In[272]:
outBoundaryMax['cyl']

# In[273]:
#outBoundaryMax= 2.5*X_desc.loc['75%'] - 1.5*X_desc.loc['25%'] #X_desc= X.describe()
X.loc[14,'cyl']= outBoundaryMax['cyl']

# In[274]:
X.loc[14,'cyl'] # 확인

# In[275]:
X.loc[30,'hp']

# In[276]:
outBoundaryMax['hp']

# In[277]:
X.loc[30,'hp']= outBoundaryMax['hp']
print(X.loc[30,'hp'])
# (최대 경계값: Q3 + 1.5*IQR) <= Max
# (최소 경계값: Q1 - 1.5*IQR) >= min

# In[278]:
outBoundaryMin= X_desc.loc['25%'] - 1.5 * (X_desc.loc['75%'] - X_desc.loc['25%'])
print(outBoundaryMin)

# In[279]:
X_desc.loc['min'] <= outBoundaryMin
# ### 이상치 처리- 평균 표준편차- qsec, carb
# p.249
# - 최대경계값: mean + 1.5*std. 이것보다 크면 이상치
# - 최소경계값: mean - 1.5*std. 보다 작으면 이상치

# In[280]:
# help(data.mean()) # numeric_only default True is deprecated. 추후 None 으로 바뀔 것
data.mean(numeric_only=True)
# data.loc[:,data.dtypes != object].mean()

# In[281]:
#Error! Series 에는 이런 인자 없다. data['mpg'].std(numeric_only=True)

# In[282]:
def outlierMeanStd(data, column):
    mean= data[column].mean()
    std= data[column].std()
    minBoundary= mean - 1.5*std
    maxBoundary= mean + 1.5*std
    print(f"최소경계값: {minBoundary}, 최대경계값: {maxBoundary}")
    outlier_index= data[column][ (data[column] < minBoundary) | (maxBoundary < data[column]) ].index
    return outlier_index

# In[283]:
def outlierMeanStdReplace(data, column):
    mean= data[column].mean()
    std= data[column].std()
    minBoundary= mean - 1.5*std
    maxBoundary= mean + 1.5*std
    print(f"최소경계값: {minBoundary}, 최대경계값: {maxBoundary}")
    
    minBoundaryOver= data[column] < minBoundary
    maxBoundaryOver= maxBoundary < data[column]
    outlier= data[column][ minBoundaryOver | maxBoundaryOver ] # 자체가 value. 가져온다면 .index
    print(f"이상치 idx 값: {outlier}")
    
    # 둘 다 걸릴 수 있다
    if minBoundaryOver.sum():
        data.loc[outlier.index, column]= minBoundary
    if maxBoundaryOver.sum():
        data.loc[outlier.index, column]= maxBoundary
    
    return data

# In[284]:
print((data[column] < minBoundary).sum())
(maxBoundary < data[column]).sum()

# In[285]:
outlierMeanStd(X, 'qsec')

# In[286]:
X.describe()

# In[287]:
X.loc[outlierMeanStd(X, 'qsec'), 'qsec']

# In[288]:
# 한 번만 실행해야 한다.
outlierMeanStdReplace(X, 'qsec')
# 한 번만 실행해야 한다.

# In[301]:
X.loc[outlierMeanStd(X, 'qsec'), 'qsec'] # outlierMeanStdReplace 실행 후 모습. 경계값이 바뀌었다.

# In[289]:
def swap(a,b):
    temp= b
    b=a
    a=temp
    return (a,b) #a,b 로 보내서 a,b 로 받았더니 unpack Error. 원래 되지 않았나???

# In[290]:
a= 5
b= 10
swap(a,b)

# In[291]:
a
# 인자 전달 방식은 얕은 복사. 바뀌길 바란다면 바꾼걸 받아줘야.
# 얕은 복사인줄 알았는데, 함수 안에서 data.loc[outlier.index, column]= minBoundary 를 하면 바깥도 바뀌더라!
# > !!

# In[292]:
# 2번 실행하면 경계값이 바뀌어서 안 된다. 그냥 index 눈으로 찾아서 상수로 넣자.
# X.loc[outlierMeanStd(X, 'qsec'), 'qsec']

# In[293]:
# 2번 실행하면 경계값이 바뀌어서 안 된다
# outlierMeanStdReplace(X, 'qsec')

# In[294]:
X

# In[295]:
a,b= swap(a,b)
a

# In[296]:
column= 'qsec'
mean= data[column].mean()
std= data[column].std()
minBoundary= mean - 1.5*std
maxBoundary= mean + 1.5*std

# In[297]:
outlier= data[column][ (data[column] < minBoundary) | (maxBoundary < data[column]) ]
# help(outlier)
print(outlier > 99)
outlier # 이거 자체가 100.0
# outlier_index= outlier.index
# print(outlier_index)

# index 를 굳이 반환하는 이유: 바꾸려고
# > index 가 없어도 바꾸는 방법?

# In[298]:
data[column][(data[column] < minBoundary) | (maxBoundary < data[column])]

# In[299]:
# X.dtypes == object
# X.loc[:,X.dtypes == object] # 열 조건 되는거 재확인
# carb

# In[303]:
X['carb'].describe()

# In[304]:
outlierMeanStd(X,'carb')

# In[306]:
X.loc[29:30,'carb']

# In[307]:
outlierMeanStdReplace(X, 'carb')

# In[300]:
print(f"{time.time()-start_time} sec") # 1.59 sec