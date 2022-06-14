#!/usr/bin/env python
# coding: utf-8
# ### cf. 빅데이터 분석기사- 프리렉
# p.168~

import time # 내장함수
start_time = time.time()

# for idx, a in enumerate(range(100000)):
#     if idx % 1000 == 0:
#         pass
# 0.032~0.051 매번 다르다
    
# print(f"{time.time()-start_time} sec") 

# In[2]:
import pandas as pd

# In[3]:
a= ['길동','순신','관순','봉길']

# In[4]:
b= pd.Series(a)

# In[6]:
a

# In[5]:
b

# In[7]:
type(b)

# In[9]:
help(b)

# In[ ]:

# In[10]:
import pandas as pd

# In[41]:
a= (['관순',19,'여'], ['순신',30,'남'],['길동',52,'남'],['봉길',25,'남']); a

# In[42]:
df= pd.DataFrame(a); df

# In[43]:
type(df)

# In[ ]:

# In[19]:
help(df)

# In[44]:
df.columns

# In[45]:
df

# In[22]:
df.index

# In[25]:
df.dtypes

# In[46]:
df.index=['1번','2번','3번','4번']

# In[47]:
df

# In[48]:
df.columns=['이름','나이','성별']; df

# In[ ]:

# In[49]:
df.나이

# In[50]:
df['나이']

# In[37]:
df['나이','성별']

# In[51]:
df[['나이','성별']]

# In[52]:
print(type(df['나이']))
print(type(df[['나이','성별']]))

# In[53]:
df[:]

# In[56]:
df[:,2]

# In[60]:
df

# In[58]:
df[1:3] # [ ) 이상~미만. 0부터 시작

# In[59]:
df['2번':'3번'] # [ ] 이상~이하

# In[ ]:

# In[61]:
get_ipython().system('ls')

# In[62]:
import pandas as pd

# In[65]:
data= pd.read_csv('data/train.csv')

# In[66]:
data

# In[69]:
data.isnull()
# 하나라도 null 인 행을 추출하는 방법?

# In[71]:
data[data['holiday'].isnull()]

# In[114]:
data= pd.read_csv('data/train.csv')
data['holiday'].sum(0)
data['holiday'].sum(1) # Series 는 안 되고 DataFrame 에만 적용 가능

# In[73]:
data.isnull().sum(0) 
# 0: 열 기준 연산
# 1: 행 기준 연산

# In[75]:
data[data['registered'].isnull()]

# In[ ]:
# # Lib 소개
# ### sklearn
# p.173
# - 전처리, 데이터 변환, 모델 선택, 검증
# - from sklearn.모델 import 함수

# In[76]:
from sklearn.model_selection import train_test_split

# In[77]:
from sklearn.tree import DecisionTreeClassifier
# ### numpy
# p.176. 수치 연산
# - 빅분기에서 활용도는 낮다

# In[78]:
import numpy as np

# In[80]:
np.zeros((2,3)) # 행, 열

# In[81]:
np.ones((3,2))

# In[82]:
np.full((2,3), 10)

# In[83]:
np.array(range(20)) # [0,20)

# In[84]:
np.array(range(20)).reshape(4,5)

# In[ ]:
# ### 유용한 내장함수
# p.178

# In[85]:
import pandas as pd
import numpy as np

# In[90]:
import random

# In[99]:
ages= [random.randrange(i,i+20) for i in range(20,30)]; ages

# In[101]:
genders= ['남','여']*5
genders[9]= np.nan
genders

# In[102]:
data= {      '나이': ages,       '성별': genders
      }

# In[103]:
data

# In[104]:
df= pd.DataFrame(data); df

# In[105]:
type(df)

# In[107]:
print(type(df.나이))
print(type(df['나이']))

# In[108]:
df.나이.sum()

# In[115]:
sum(df.나이)

# In[116]:
df.max()

# In[117]:
df.나이.max()

# In[118]:
max(df.나이)

# In[120]:
df.나이.min()

# In[121]:
min(df.나이)

# In[122]:
df.나이.mean()

# In[123]:
mean(df.나이)

# In[124]:
# 뒤에 () 를 꼭 넣어줘야
df.나이.median

# In[125]:
df.나이

# In[126]:
df.나이.median() # 중앙값

# In[127]:
df.나이.var() # 분산

# In[129]:
var(df.나이)

# In[131]:
df.나이.std() # 표준편차

# In[134]:
pow(df.나이, 0.5)

# In[135]:
df.나이.quantile() # default: 0.5

# In[136]:
df.나이.quantile(0.25)

# In[137]:
df.나이.quantile(1)

# In[138]:
df.나이.quantile(0.5)

# In[139]:
help(df.나이)

# In[140]:
df.quantile(0.5)

# In[141]:
print(type(df.나이.quantile(0.5)))
print(type(df.quantile(0.5)))
# p.183
# - round 반올림은 내장함수라서 막바로 쓸 수 있다
# - 올림 ceil, 내림 floor 는 math 것이라 불러와야 하고

# In[142]:
f1= df.나이/7; f1

# In[143]:
f1.round()

# In[146]:
f1.round(2) # 2째 자리의 결과를 보고 싶음. 3째에서 반올림

# In[147]:
f1.round(-1) # 10의 자리의 결과를 보고 싶음

# In[148]:
round(f1, -1)

# In[149]:
(df.나이/7).round()

# In[150]:
f1.ceil()

# In[163]:
math.ceil(f1)

# In[151]:
import math

# In[165]:
map(math.ceil, f1)

# In[159]:
[i for i in map(math.ceil, f1)]

# In[164]:
[i for i in map(math.ceil, df.나이/7)]

# In[166]:
[i for i in map(math.floor, f1)]

# In[ ]:

# In[167]:
df
# ### 데이터 겉모습
# p.185

# In[168]:
df.shape # 10행 2열

# In[172]:
df.나이.shape # 10행 짜리 열 1개 => Series 구나

# In[173]:
len(df) #10행

# In[175]:
df.len()

# In[180]:
len('01234')

# In[181]:
len(df.나이)

# In[182]:
len(123)

# In[183]:
str(123)

# In[184]:
len(str(123))

# In[188]:
import numpy as np

# In[189]:
a= np.arange(10).reshape(5,2); a

# In[191]:
len(a) # 행 개수

# In[194]:
a.size # 요소 총 개수

# In[203]:
size(pd.DataFrame(a)) # df 도 이런 형태는 안 된다

# In[195]:
a.size()
size(a) # 다 안 된다

# In[199]:
a.head() # np 에 head 안 된다

# In[200]:
df.head()

# In[204]:
print(type(df.head()))
print(type(df.head))

# In[205]:
df.head(2)

# In[208]:
df.head(-1) # 마지막 하나 빼고 다

# In[210]:
df.head(-2) # 마지막 두개 빼고 다

# In[211]:
df

# In[213]:
df.tail() # 마지막 5개

# In[214]:
df.tail(3)

# In[216]:
df.tail(-2) # 처음 2개 빼고 다

# In[217]:
df.columns

# In[221]:
df.names # 이런거 아니고

# In[224]:

df.index # index 범위 0~10, index 넘버링 차분값= 1

# In[227]:
df.index[0]

# In[230]:
df.index[-1]

# In[229]:
df.index(1)
# ### 데이터 요약 관련
# p.190
# - value_counts(), unique(), isnull()
# - count(), info(), describe()

# In[231]:
df.describe()

# In[232]:
df.columns

# In[233]:
df.value_counts()

# In[236]:
df.성별[0]='여'
# [slice 더 좋은 방법](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)
# -  df['one']['second'] 보다는 df.loc[:, ('one', 'second')] 방법이 좋다
# -  [][] 는 순차적으로 실행되고 중간에 변수가 생성되는데
# - .loc[:, (,)] 는 튜플로 동시에 실행이 되니 더 빠르고 리소스도 적게 든다

# In[251]:
df.loc[0,'성별']= '여' # df.loc[행,열]

# In[252]:
df

# In[253]:
df.성별.value_counts()
# #### df.성별 과 df['성별'] 차이
# 만에 하나 df에 '성별'이라는 이름의 모듈이 있으면, df.성별 호출시 컬럼대신 해당 모듈이 호출된다.
# > 그러니 항상 컬럼을 받고 싶다면 df['성별'] 과 같이 부르자

# In[254]:
df['성별'].unique() # 중복 제거

# In[257]:
df.isnull()

# In[260]:
df.isnull().sum()
# ##### 하나라도 null 이 들어있는 행 출력

# In[264]:
df[df.isnull().sum(1)!=0]

# In[271]:
df.count() # null 빼고 출력. 열 별 값의 수

# In[273]:
df.count(1) # 행 별 값의 수

# In[267]:
len(df)

# In[268]:
'I am superman!!'.count('a')

# In[270]:
'You r young'.count('y') # 대소문자 구별한다

# In[ ]:

# In[275]:
df.info() # 데이터 생김새 종합선물세트. type, index, columns, count, dtypes, memory!

# In[276]:
df.describe() # 기초 통계

# In[ ]:
# ### 데이터 변형 관련
# p.194
# - transpose(), T
# - .loc[행,열], iloc[행,열], sort_values('열')
# - fillna('열'), dropna(), drop()
# - replace(현재, 바꾸려는값)

# In[278]:
df

# In[280]:
df.transpose()

# In[282]:
df.T()

# In[283]:
df.T

# In[286]:
df.loc[:,'성별']

# In[298]:
df.loc[2:4,'성별']

# In[297]:
df.loc[,'성별']

# In[287]:
df.loc['성별']

# In[296]:
df.loc[3,:]

# In[293]:
df.loc[3,]

# In[292]:
#type(df.loc[3])
df.loc[3]

# In[ ]:

# In[299]:
df['성별']=='남'

# In[300]:
df[df['성별']=='남']

# In[301]:
df.loc[df['성별']=='남']

# In[ ]:
# p.199 할 차례

# In[303]:
df.iloc[0,:]

# In[304]:
df.iloc[:,0]

# In[305]:
df.iloc[:,1]

# In[306]:
df.iloc[2:3,0:1]

# In[307]:
df.sort_values()

# In[308]:
df.sort_values('나이')

# In[309]:
df.sort_values(df.columns[0])

# In[312]:
help(df.sort_values)

# In[315]:
df.sort_values('나이',ascending =False) # 내림차순. F FALSE 아니고 False

# In[318]:
df.sort_values('나이', ascending=True) # default. 오름차순

# In[319]:
df.sort_values(['성별','나이']) # 성별 먼저 오름차순 ㄱㄴㄷ 정렬하고, 그 안에서 나이를 오름차순 정렬

# In[ ]:

# In[320]:
df.info()

# In[323]:
df[df.isnull().sum(1) != 0]

# In[325]:
df['성별'].fillNa('남')

# In[326]:
df['성별'].fillna('남') # 무지성 채우기. 원본 변화 없음

# In[328]:
df[df.isnull().sum(1) != 0]

# In[329]:
help(df['성별'].fillna('남'))

# In[331]:
df['성별'].fillna('남', inplace= True) # 원본 바꿈

# In[332]:
df

# In[333]:
df.iloc[-1,-1]

# In[335]:
# 원복

# In[334]:
import numpy as np
df.iloc[-1,-1]= np.nan
df

# In[336]:
df['성별채움']=df['성별'].fillna('남')
df
# p.206

# In[337]:
df[df.isnull().sum(1) != 0]

# In[338]:

df.dropna() # inplace= False

# In[339]:
df[df.isnull().sum(1) != 0]

# In[340]:
df.dropna(inplace= True)

# In[341]:
df[df.isnull().sum(1) != 0]

# In[342]:
df
# DataFrame 에 행 추가 방법
# [df add row](https://www.statology.org/pandas-add-row-to-dataframe/)

# In[345]:
len(df.index)

# In[346]:
import numpy as np
df.loc[len(df.index)]= [29,np.nan,'남']

# In[347]:
df

# In[348]:
df.head()

# In[349]:

df.drop(index=[0,2,4]) # inplace= False. index 가 비어있는 채로 남는다.

# In[350]:
df

# In[354]:
df.drop(columns='성별')

# In[352]:
df.drop(columns=['성별','성별채움'])

# In[353]:
help(df.drop)
# p.209 replace

# In[356]:
df['성별'].replace('남','M').replace('여','F')

# In[357]:
df['Gender']= df['성별'].replace('남','M').replace('여','F')

# In[358]:
df

# In[361]:
df[['성별','Gender']]

# In[365]:
df.drop('Gender', inplace= True)

# In[366]:
df.drop(columns='Gender', inplace= True)

# In[368]:
# 이렇게는 절대 하지 말자. df[df['성별']=='여']='W' # 원본 망함.

# In[364]:
df.drop(columns='Gender', inplace= True)

# In[369]:
df

# In[ ]:

# In[372]:
import random

# In[373]:
ages= [random.randrange(i,i+20) for i in range(20,30)]; ages

# In[374]:
genders= ['남','여']*5
genders[9]= np.nan
genders[0]= '여'
genders

# In[378]:
data= {      '나이': ages,        '성별': genders
      }

# In[379]:
data

# In[380]:
df= pd.DataFrame(data); df

# In[ ]:
# ##### replace 보다 여러 조건을 쉽게 넣을 수 있는 방법
# isin() 대신에 범위조건 이용한다면 봄여름가을겨울 나눌 수도 있겠지
# - test.loc[test["weather"].isin([1,2]),"weather12"]= 1
# - test.loc[test["weather"].isin([3,4]),"weather34"]= 0
# 
# ##### 특정 조건에 해당하는 값만 컬럼 분리. 여기선 따로 X
# - train["atemp_LL"]= (train["atemp"]<10)
# - train["atemp_L"]= ((train["atemp"]>=10) & (train["atemp"]<20))

# In[383]:
df.loc[df['성별'].isin(['남']), 'Gender']= 'M'
df.loc[df['성별'].isin(['여']), 'Gender']= 'W'

# In[384]:
df

# In[ ]:
print(f"{time.time()-start_time} sec") # 