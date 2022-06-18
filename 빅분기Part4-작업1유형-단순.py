#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part4-작업1유형-단순한 데이터 분석
# p.315~ / 되도록 문제보고 바로 풀어보자

# 22.6.17.금 / ~p.321 권장

# 22.6.18.토 / ~p.321 ~p.345 권장

# ### 1.1 Top 10 구하기
# boston. MEDV. 작은값부터 오름차순. 10개 행

# 주의: 따로 말은 없었지만 MEDV 컬럼만 출력해야 한다.

# In[5]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# print(data.info, data.head())
# help(data) # ascending. df.sort_values(columns, ascending=False).head(n)
data.sort_values('MEDV', ascending=True).head(10)['MEDV']


# ### 1.2 결측치 확인
# boston. RM 컬럼. 결측치. |(평균값 대치 한 후 표준편차 값) - (결측치 삭제 후 표준편차 값)|

# 1. std 계산은 결측치 빼고 계산한다
# 2. 양수 조건이 있으면 안전하게 abs() 는 넣어주자.
# 3. abs() 는 별도 모듈 없이도 기본으로 계산가능!
# 4. std() 도 describe() 없이 컬럼에다 직접 계산할 수 있다!
# 4. 마지막에 변수에 넣어 출력하면 안 된다. 꼭 print 안에 계산식을 넣어주자

# In[19]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# print(data.info())
# print(data.describe().loc['std','RM']) # <=> data.dropna().describe().loc['std','RM']
# help(data.fillna)
fillData= data.fillna(data.describe().loc['mean'])
# print(fillData.info())
# print(fillData.describe().loc['std','RM'])

# import numpy as np
# help(np) #abs. np.abs
# help(data) #abs. 기본으로 abs() 가 들어있다!

print(abs(data.describe().loc['std','RM'] - fillData.describe().loc['std','RM']))


# In[23]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# print(data.fillna(data['RM'].mean())['RM'].std())
# print(data['RM'].std())
print(abs(data.fillna(data['RM'].mean())['RM'].std()-data['RM'].std()))


# In[24]:


# # 원본  data에 영향을 주지 않는다. 깊은 복사가 기본제공!
# data2= data.copy()
# data2.fillna(0, inplace=True) 
# print(data.info()) 
# print(data2.info())


# ### 1.3 이상값 확인하기
# p.325/ boston. ZN 컬럼 대상. ZN 평균값에서 표준편차의 1.5배보다 크거나 작은 ZN 값의 합계

# ZN mean() +- 1.5*std()

# In[32]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
stdZN= data['ZN'].std()
meanZN= data['ZN'].mean()
# print(meanZN- 1.5*stdZN, meanZN+ 1.5*stdZN)
print(data.loc[(data['ZN'] < meanZN- 1.5*stdZN) | (data['ZN'] > meanZN+ 1.5*stdZN), 'ZN'].sum())


# ### 1.4 사분위수 구하기
# p.331/ boston. CHAS와 RAD 컬럼을 제외하고 컬럼별 IQR 값 구하기. 출력구조는 2열. 1열은 보스턴 데이터 세트의 컬럼이름 표시

# 주의
# 1. columns에 set 쓰지 말자. 순서 바뀌어서 오답처리될 수도 있다.
# 2. '출력구조 2열'은 Series 안 된다. 책 p.335 이 틀렸다

# In[93]:


# 합리적이면서 옳은 방법
import pandas as pd
data= pd.read_csv('bigData/boston.csv')
data.drop(columns=['CHAS','RAD'], inplace=True)
cols= data.columns
iqr= data.describe().loc['75%'] - data.describe().loc['25%']
print(pd.DataFrame(data=zip(cols, iqr)).shape)
print(pd.DataFrame(data=zip(cols, iqr)))


# In[88]:


# ## 어차피 Series 결과 가능하면 합리적으로
# import pandas as pd
# data= pd.read_csv('bigData/boston.csv')
# # print(data.columns)
# data.drop(columns=['CHAS','RAD'], inplace=True)
# # print(data.columns)
# # print((data.describe().loc['75%']-data.describe().loc['25%']).shape) # (12,)
# print(data.describe().loc['75%']-data.describe().loc['25%'])


# In[89]:


# ## 책이랑 똑같이 = 결과물 Series 네
# import pandas as pd
# data= pd.read_csv('bigData/boston.csv')
# # print(data.columns)
# data.drop(columns= ['CHAS', 'RAD'], inplace=True)
# desc= data.describe()
# # print(desc.iloc[[4,6]].shape)
# # print(desc.iloc[[4,6]])
# # print(desc.iloc[[4,6]].T.shape)
# # print(desc.iloc[[4,6]].T)
# descQ13= desc.iloc[[4,6]].T
# print((descQ13['75%']-descQ13['25%']).shape)
# print(descQ13['75%']-descQ13['25%'])


# [python set to list](https://www.geeksforgeeks.org/python-convert-set-into-a-list/) list(set())

# In[90]:


# ## 책 => (12,) 틀림
# import pandas as pd
# data= pd.read_csv('bigData/boston.csv')
# # print(data.columns)
# data.drop(columns= ['CHAS', 'RAD'], inplace=True)
# # print(data.columns)
# # print(data.describe().loc[['25%', '75%']])
# descQ1Q3= data.describe().loc[['25%', '75%']].T
# # print(descQ1Q3)
# print((descQ1Q3['75%'] - descQ1Q3['25%']).shape)
# print(descQ1Q3['75%'] - descQ1Q3['25%'])


# In[95]:


# ## 더 귀찮은 방법. 특히 set 하면 순서 바뀌어서 시험 때 답과 다르다고 오답처리할 수도 있다.
# # help(boston)
# import pandas as pd
# boston= pd.read_csv('bigData/boston.csv')
# # print(boston.columns)
# # targetCols= list(set(boston.columns) - set(['CHAS', 'RAD']))
# targetCols= set(boston.columns) - set(['CHAS', 'RAD'])
# desc= boston[targetCols].describe()
# # print(desc.loc['75%']- desc.loc['25%'])

# # 이게 끝이 아니다! 출력구조 => shape 을 보자. 아닌가??
# # help(boston) # len
# # print(len(desc.loc['75%']- desc.loc['25%'])) # 12

# print(pd.DataFrame(data=zip(targetCols, desc.loc['75%']- desc.loc['25%'])).shape)
# print(pd.DataFrame(data=zip(targetCols, desc.loc['75%']- desc.loc['25%'])))


# In[ ]:





# ### 1.5 순위 구하기
# p.336/ boston. MEDV 컬럼에서 30번째 큰 값을 1~29번째로 큰 값에 적용한다. 그리고 MEDV 컬럼의 평균값, 중위값, 최솟값, 최댓값 순으로 한 줄에 출력하기

# 주의
# 1. .sort_values() 로 index 가 뒤섞인 것은 [] 로 접근할 수 없고, .iloc[] 만 가능하다
# 2. .iloc[0:29] 이면 .iloc[0] 부터 iloc[28] 까지라는 의미다. [이상 ~ 미만) 임에 유의!!

# In[146]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# help(data['MEDV']) # ascending. reindex(Series에 없음) # print(len(data['MEDV'])) # help(data) # ascending
# print(data['MEDV'].sort_values(ascending=False)) # 큰 수 => False
# print(data['MEDV'].sort_values(ascending=False).head(30)) #.head()

# big30= data['MEDV'].sort_values(ascending=False).head(30)
MEDVdescending= data['MEDV'].sort_values(ascending=False)
# print(big30.iloc[0], big30.iloc[29])

# big30.iloc[0:28]= big30.iloc[29]
#틀렸다! MEDVdescending.iloc[0:28]= MEDVdescending.iloc[29]
# print(MEDVdescending.iloc[0:1]) # 1개
# print(MEDVdescending.iloc[0:2]) # 2개
# print(len(MEDVdescending.iloc[0:28])) #28개
# print(MEDVdescending.iloc[0:32])
MEDVdescending.iloc[0:29]= MEDVdescending.iloc[29]
print(MEDVdescending.mean(), MEDVdescending.median(), MEDVdescending.min(), MEDVdescending.max())


# In[ ]:




