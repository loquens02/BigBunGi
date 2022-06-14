#!/usr/bin/env python
# coding: utf-8
# setting: p.213~
# - [학습 데이터](https://github.com/7ieon/bigData) git clone https://github.com/7ieon/bigData.git
# 실행시간이 1s 를 넘기면 안된다. 제출 전에 확인하자
# - [python runtime check](https://seulcode.tistory.com/245)

# In[164]:
import time # 내장함수
start_time = time.time()
for idx, a in enumerate(range(100000)):
    if idx % 1000 == 0:
        pass
    
print(f"{time.time()-start_time} sec") # 0.032~0.051 매번 다르다
# # 데이터 분석 절차 체득하기
# 실습: p.222~

# In[3]:
get_ipython().system('ls')
# bigData
# - bike_x_test.csv
# - bike_x_train.csv
# - bike_y_train.csv
# - boston.csv
# - mtcars.csv
# - titanic_x_test.csv
# - titanic_x_train.csv
# - titanic_y_train.csv
# - x_test.csv
# - x_train.csv
# - y_train.csv
# 
# ### 데이터 준비하기: 데이터 로드
# p.225

# In[4]:
import pandas as pd

# In[5]:
data= pd.read_csv('bigData/mtcars.csv')

# In[6]:
data.info()
# dtype 이 int64 이면 방심하면 안 된다. 범주형변수 일수도 있다. (p.229)
# > value_counts() 로 알아보기

# In[42]:
#help(data)
print(type(data.nunique()))
data.nunique() # value_counts() 보다 먼저 보면 좋을. 손이 덜 간다

# In[ ]:
# ## 데이터의 모양과 구조, 특징 파악
# p.226~231

# In[52]:
factorDoubt= data.loc[:,data.nunique() <10].columns; factorDoubt

# In[37]:
#data.unique() # Error ! unique() 는 Series 로만 가능
#data[['cyl']].unique() # Error! 여전히 DataFrame 이다 
data['cyl'].unique()

# In[38]:
data[['cyl','vs','am','gear','carb']].value_counts() # 종류가 몇 개 안 된다면 이렇게도 가능한데, 직관적이진 않다.

# In[54]:
#factorDoubt= ['cyl','vs','am','gear','carb']
for _ in factorDoubt:
    print(data[_].unique())

# In[58]:
print(data[factorDoubt].info())

# In[ ]:

# In[28]:
#help(data)
# data[['hp','vs','carb']].value_counts()
# data[['vs','carb']].value_counts()
data[['vs']].value_counts() # 0 1
# data[['carb']].value_counts() # 1 2 3 4 6 8
# data 설명: p.222
# - mtcars: 32종 자동차의 디자인/ 성능 특성/ 연비 정보. mpg 가 알고 싶은 종속변수, 나머지가 독립변수
# - hp: 마력
# - vs: V형 엔진(V) or 직렬 엔진(S)
# - carb: 기화기 개수

# In[9]:
# print(data[data.isnull().sum(1) !=0]) # 시험 환경
data[data.isnull().sum(1) !=0]

# In[29]:
print(data.head()) # 실제로 어떻게 생겼는지도 봐두면 좋다.

# In[19]:
print(data.shape)
print(type(data.shape))
print(data.shape[1]) # typle index => length 대용으로 사용 가능할듯

# In[21]:
print(type(data))

# In[59]:
print(data.columns)

# In[60]:
data.head(1)

# In[62]:
print(data.describe())
# ### describe() 해석
# ##### 숫자형 변수가 8개이다
# 1. describe() 로 결과를 낼 수 있는 게 9열
# 2. vs 는 사분위수가 너무 깔끔 => 0,1 로 된 factor
# 3. 나머지 변수 => 사전에 변수 설명을 읽고 describe() 를 보니 factor 가 아니라 실제 값일 것
# 
# ##### 뭔가 이상한 것 찾기
# 1. min max 를 주의 깊게 보자. mean 과 비교하며
# 2. cyl 은 엔진 기통수인데 50개가 달린 차가 있다 ??? 항공모함인가..
# 3. psec 은 1/4 mile(약 402m) 가는데 걸리는 시간인데, 0.1초만에 가는 차가 있다??? 초속 4km..

# In[ ]:
# ##### 수치형 데이터가 뭐뭐있나
# describe 로 본거랑 실제 데이터랑 개수가 다르네?

# In[73]:
data.describe().shape[1] # 9열. 수치형 데이터인 변수 개수. 느려보이지만 확실.

# In[87]:
# help(data)
# data.loc[:,data.dtypes != 'object'].shape[1]# 수치형 데이터인 변수 개수2. 여기서는 맞았지만 확신은 X

# In[88]:
data.head(2) # 10개로 보임
# 컬럼이 많아서 생략(...) 될 경우
# [python head all columns](https://towardsdatascience.com/how-to-show-all-columns-rows-of-a-pandas-dataframe-c49d4507fcf)

# In[ ]:
pd.set_option('display.max_columns', None)

# In[118]:
# help(pd.set_option) - 이걸로 위 형태를 알아낼 수 없다
# 클라우드 환경에서 먹힐지도 미지수

# In[ ]:
# ##### 눈으로 대조 
# 컬럼 20개 넘어가면 어려운 방법

# In[92]:
data.dtypes # 눈으로 대조 => gear 가 object
# ##### 수치형인데 문자형으로 되어있는 경우- 방법2: factor 찾다가 우연히 얻어걸린 방법
# gear에 이상한 데이터가 껴있구나!

# In[100]:
# data.nunique()

# In[99]:
factorDoubt= data.loc[:,data.nunique() < 10].columns
for _ in factorDoubt:
    print(f"{_}: {data[_].unique()}")
# ##### 수치형인데 문자형으로 되어있는 경우- 방법1
# 문자형 데이터에 value_counts() 로 구성요소 뜯어보기

# In[114]:
numberDoubt= data.loc[:,data.dtypes == object].columns
for _ in numberDoubt:
    print(f"{_}:\n{data[_].value_counts()}\n")

# In[ ]:
# ### 상관관계
# 너무 밀접(1)하면 동일하다 보고 하나만 선택하고, 너무 무관(0)하면 관계없다고 보고 변수에서 제외

# In[167]:
#type(data.corr()) # DataFrame
corr= data.corr(); corr

# In[179]:
(0.8 <= corr) # 양의 상관관계. 대각행렬 제외하고 True 인 경우: wt무게 ~ disp배기량

# In[180]:
(corr <= -0.8) # 음의 상관관계. mpg연비 ~ disp배기량, mpg연비~wt무게
# [python dataframe or](https://ponyozzang.tistory.com/608)
# - df[(df['age'] < 20) | (df['point'] > 90)]
# - any 는 동일 Series (혹은 동일 컬럼, 동일 DataFrame) 안에서 True/False 판별하는 것

# In[189]:
# 동시에 보면서도 양/음 확인 가능한 방법
corr[(corr <= -0.8) | (0.8 <= corr)]
# mpg 연비, disp 배기량, wt 무게
# ##### [mpg 와 disp], [mpg 와 wt] 음의 상관관계 강함 => mpg는 종속변수이니 독립변수로 disp 나 wt 가 유력
# ##### [disp 와 wt] 양의 상관관계 강함 => 둘 중 하나만 써도 될? 것

# In[184]:
# T-F T-T 뭐일때 T 로 반환할지 의도가 불분명하므로 Error (0.8 <= corr) or (corr <= -0.8)

# In[173]:
# help(corr.any) # Examples > >>> pd.Series([False, False]).any()    # False

# In[183]:
# 시행착오. any는 각 Series 내부에서 계산하는 용도. pd.Series([0.8<=corr, corr<=-0.8]).any() 

# In[ ]:

# In[190]:
corr[((-0.1<=corr) & (corr<=0.1))]
# mpg 연비(종속변수), qsec 400m 가는데 걸리는 시간, cyl 엔진 기통 수, hp 마력, wt 무게, // drat 뒤 차축비(rear axle ratio), carb 기화기 개수
# ##### mpg연비 와 qsec 은 별 상관이 없다 => 종속변수 알아보는데 qsec 은 빼도 될 것 같다.
# ##### qsec 과 mpg,cyl,hp, wt 는 별 상관이 없다 => 다 독립변수라서 분석에 영향은 없을 것
# ##### drat 와 carb 는 별 상관이 없다 => 다 독립변수라서 분석에 영향은 없을 것

print(f"{time.time()-start_time} sec")