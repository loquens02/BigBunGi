#!/usr/bin/env python
# coding: utf-8

# # 빅분기Part4-작업1유형-복잡한 데이터 분석
# 상세한 요구조건

# 22.6.18.토 / 권장 ~p.345

# [이상치 하한, 상한 일괄적용](https://pandas.pydata.org/docs/reference/api/pandas.Series.clip.html) df['..'].clip(하한, 상한)

# ### 2.1 그룹별 집계/요약하기
# p.341 / boston. TAX컬럼이 TAX 중위값보다 큰 데이터 중. CHAS 컬럼과 RAD 컬럼 순으로 그룹 짓고. 각 그룹의 데이터 개수 구하기. 단, CHAS 와 RAD 컬럼별 데이터 개수는 COUNT 라는 컬럼으로 출력한다.

# In[35]:


# 책 참고
import pandas as pd
data= pd.read_csv('bigData/boston.csv')
newdata= data[data['TAX']> data['TAX'].median()][['CHAS','RAD']] # 각 그룹 => 얘네만 본다.
# print(newdata) 
# print(newdata['CHAS'].unique())
# print(newdata['RAD'].unique())
# help(newdata.groupby) # example 에도 없는 용법. groupby(['기준컬럼'])['대상컬럼'].작업()n

## 2개로 기준잡은 group 이라서 둘중 뭐로 해도 count() 결과는 같다
## count 세는 값 위에 '컬럼명'이 비어있다. 이걸 COUNT 로 채우는 것
# print(newdata.groupby(['CHAS', 'RAD'])['CHAS'].count())
# 같음 print(newdata.groupby(['CHAS', 'RAD'])['RAD'].count()) 
# 틀림 print(newdata.groupby(['CHAS', 'RAD'])['CHAS','RAD'].count()) # 같은 count 열이 2개 나온다. deprecated

groupCntCR= newdata.groupby(['CHAS','RAD'])['CHAS'].count()
# 안된다 groupCntCR.columns= 'COUNT'
# 에러! Series에는 columns가 없다 print(groupCntCR.columns) 

# print(type(groupCntCR)) # Series
# help(groupCntCR) # rename
# 안된다. 밑에 이름이 바뀜 groupCntCR.rename('COUNT', inplace=True)
# print(groupCntCR)

# 발상의 전환. 없으면 만들라
dfCR= pd.DataFrame(groupCntCR)
#Index(...) must be called with a collection of some kind  #dfCR.columns= 'COUNT'
dfCR.columns= ['COUNT']
print(dfCR)


# In[10]:


# 헤멤
import pandas as pd
data=pd.read_csv('bigData/boston.csv')
# help(data.groupby) # exam. df.groupby(['Animal']).mean()
print(data[data['TAX']> data['TAX'].median()].groupby(['CHAS','RAD']).count())
# ???


# In[ ]:





# ### 2.2 오름차순/내림차순 정렬하기
# p.349 / boston 데이터. TAX 컬럼 오름차순 정렬 결과랑 내림차순 정렬 결과 각각 구하기. 각 순번에 맞는 오름차순 값과 내림차순 값의 차이를 구하여 분산값 출력

# In[57]:


# 책 (p.352) 참조. index 가 뒤섞였으면 다시 만들면 된다!
import pandas as pd
data= pd.read_csv('bigData/boston.csv')
ascTAX= data['TAX'].sort_values(ascending=True); #print(ascTAX) # 오름
desTAX= data['TAX'].sort_values(ascending=False); #print(desTAX)# 내림
# help(ascTAX) #_index. reset_index(self, level=None, drop=False, name=None, inplace=False)
# drop: False: 새 DataFrame에 열로 삽입하지 않고 인덱스를 재설정하기만 하면 됩니다.
# drop: True: (책) 기존 index 정보를 남기지 않고 삭제하겠다
ascTAX.reset_index(drop=True, inplace=True)
desTAX.reset_index(drop=True, inplace=True)
# print(ascTAX) # 오름
# print(desTAX)# 내림
# print(ascTAX-desTAX) # 의도대로 뺄셈이 잘 되었다. index 가 같아야 연산이 제대로 되는 구나!

## 분산 구하기
# print((ascTAX-desTAX).var()) # 101954.72475247525
# print((desTAX-ascTAX).var()) # 101954.72475247525
print(abs(desTAX-ascTAX).var()) # 101954.72475247525


## 그냥 concat 써보기
# 오름/내림차순 합치기
# help(pd.concat) # exam. pd.concat([s1, s2])
#(objs: 'Iterable[NDFrame] | Mapping[Hashable, NDFrame]', axis=0, join='outer', ignore_index: 'bool' = False,
# keys=None, levels=None, names=None, verify_integrity: 'bool' = False, sort: 'bool' = False, copy: 'bool' = True) 
# taxCon= pd.concat([ascTAX, desTAX])
taxCon= pd.concat([ascTAX, desTAX], axis= 1) # 컬럼을 붙이려면 axis=1 컬럼
# print(taxCon)


# 22.6.18.토 p.357

# In[43]:


# # 헤멤. index 뒤섞인 걸 그대로 하려고 하니까 진전을 못 했다.
# import pandas as pd
# data= pd.read_csv('bigData/boston.csv')
# # help(data) #sort
# ascTAX= data['TAX'].sort_values(ascending=True) # df, Series 에 둘다 sort 있는데, 인자 차이뿐.
# desTAX= data['TAX'].sort_values(ascending=False)
# len(ascTAX)
# # 분산 sum(diff^2)
# print(ascTAX.iloc[0:3]) # index 353 123 122
# print(desTAX.iloc[0:3]) # index 492 491 490

# print(ascTAX.iloc[0:3]-desTAX.iloc[0:3]) 
#>>>>>>> # 전혀 예상밖으로 NAN 6개가 나온다. index 122 123 353 490 491 492
# 서로 index 가 다른걸 연산해서.


# In[ ]:





# In[ ]:


# import pandas as pd
# data= pd.read_csv('bigData/mtcars.csv')
# clip 잘 되는지 확인하고 싶음


# In[ ]:





# ### 2.3 최소최대 변환하기
# boston 데이터. MEDV 컬럼 MinMaxScaler 변환. 0.5 보다 큰 값을 갖는 레코드 수

# 22.6.19.일 16:20 / 권장 ~p.382

# In[74]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# import sklearn # help(sklearn) # pacakge. 둘러봄. preprocessing
from sklearn.preprocessing import MinMaxScaler
# dir(MinMaxScaler) # fit_transform
model= MinMaxScaler()
after= model.fit_transform(data[['MEDV']]) # input 2D array
# print(len(after)) # 506
print(len(after[after >  0.5])) # 106
print(type(after[after >  0.5])) # ndarray
# ndarray 에는 count 없다. print((after[after >  0.5]).count()) # ndarray


# In[71]:


# 책: ndarray 를 pd.DataFrame 으로 변경. columns= data.columns 로 컬럼 유지. df.count() 로 셈
book= model.fit_transform(data)
print(type(book))
book= pd.DataFrame(book, columns= data.columns)
print(type(book))
book.loc[book['MEDV']>0.5, 'MEDV'].count()


# In[ ]:





# ### 2.4 빈도값 구하기
# boston 데이터. AGE 컬럼 소수 첫째 자리에서 반올림. 가장 많은 비중을 차지하는 AGE 값과 그 개수를 차례대로 출력(AGE 최빈값과 그 개수)

# ##### (91,) Series 를 (91,2) DataFrame 으로 변경하는 방법
# pd.DataFrame(시리즈).reset_index(drop=False, inplace=True)
# > 새 index 를 만들건데, 기존 index 는 보존하겠다. 어디에? 새 컬럼에! - p.367

# In[122]:


# my
import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# help(data['AGE'].round) # decimal param
# print(data['AGE'].round(0)) # 0 이 소수 첫째에서. -1 이 일의자리에서, 1이 소수 둘째에서 (0 일의자리로, 1 소수첫째자리로) 반올림
# print(data['AGE'].round(0).value_counts())
print(data['AGE'].round(0).value_counts().index[0], data['AGE'].round(0).value_counts().iloc[0])
# print(data['AGE'].round(0).value_counts().iloc[0]) # 그냥 [0]은 안 된다.


# In[146]:


## 책2- scipy 를 익혀보자
# import scipy # help(scipy) # pacakge. stats기초통계 fft io signal신호처리 sparse희소행렬 등
# from scipy import stats
# dir(stats) # mode, 피어슨, 베이즈, binom, nbinom, boxcox, 카이, 코사인, iqr, mstats, 푸아송, randint
# 반원(semicircular), skey, t, uniform, wilcoxon, zscore 등

from scipy.stats import mode
# help(mode) #(a, axis=0, nan_policy='propagate') #from scipy import stats # stats.mode(a)
# 컬럼 다 있는 거 print(data.columns, mode(data))
# print(data2.columns, mode(data2)) # data2: 아래칸 먼저 실행. 일의자리까지 반올림하고 'AGE'만 있는 DF.
# ModeResult(mode=array([[100.]]), count=array([[43]])) : 최빈값 100. , 개수 43

# print(mode(data2)[0][0][0])
print(int(mode(data2)[0]), int(mode(data2)[1]))
# print(int(mode(data2)[1]))


# In[123]:


# 책1- 효율적인 길은 아니지만 복습겸 새 함수 사용법 배울겸
data2= round(data['AGE'],0)
# print(type(data2)) # Series

## groupby 를 써보기 위해 DF 로 변환
data2= pd.DataFrame(data2)
data3= data2.groupby(['AGE'])['AGE'].count()
# print(data2.groupby(['AGE'])['AGE'].count())
# type(data2.groupby(['AGE'])['AGE'].count()) #Series
# 요상함 print(data2.groupby(['AGE']).count()) # groupby를 대상 컬럼 지정없이 그냥 쓰면 뭔가 생소한 결과가 나온다.
# 요상함 type(data2.groupby(['AGE']).count()) # DataFrame. 91 rows 0 columns (???)

## index 를 컬럼으로 바꾸기 위해 DataFrame 으로 전환
data3= pd.DataFrame(data3)
data3.columns= ['COUNT']
# print(data3.tail(2)) # AGE 가 index, COUNT 가 컬럼
# print(type(data3), data3.shape) # DataFrame (91,1)

data3.reset_index(drop=False, inplace=True)
# print(data3.tail(2))
# print(type(data3), data3.shape) # DataFrame (91,2) <<<<<< !!!
# print(data3.iloc[-1])
print(data3.iloc[-1,0], data3.iloc[-1,1]) # 이러면 끝이지만

## sort_values 를 써보기 위해
data3.sort_values(by= 'COUNT', ascending=False, inplace=True)
# print(data3.head(3))
print(data3.iloc[0,0], data3.iloc[0,1])


# In[ ]:





# ### 2.5 표준 변환하기
# boston 데이터. DIS 컬럼을 표준화척도(Standard Scale)로 변환 후, 0.4보다 크면서 0.6보다 작은 값들에 대해 평균 구하기. 소수 셋째자리에서 반올림하여 소수 둘째자리까지 출력하시오

# In[182]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
# data.columns
# from sklearn import preprocessing # dir(preprocessing)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
dataDIS= scaler.fit_transform(data[['DIS']])
print(type(dataDIS), dataDIS[0:3]) # ndarray
# 틀림. 값이 중복으로 더 들어가더니 고장난다. -8.4e-17. dataDIS[(dataDIS > 0.4) | (dataDIS < 0.6)].mean()
dataDIS[(dataDIS > 0.4) & (dataDIS < 0.6)].mean().round(2)


# In[ ]:





# ### 2.6 유니크한 값 구하기
# boston 데이터. 중복제거. 컬럼 별로 유니크한 값의 개수를 기준으로 평균값을 구하시오

# In[188]:


import pandas as pd
data= pd.read_csv('bigData/boston.csv')
print(data.nunique().mean()) # 0행


# In[256]:


## nunique() 를 모른다고 할 때
# help(data.apply)
# print(data.dtypes) # float64 or int64
# print(data.info()) # 'RM' 에만 null 이 있다.
# data.applymap(lambda x: pd.unique(x))  #'float' object is not iterable

# data[['CRIM','ZN','INDUS']].applymap(lambda x: pd.unique(x))
# 이 시도가 잘못된 이유: 개별 요소마다 unique() 를 적용하려고 하는데, unique(32.5) 하면 당연히 에러!

# 원래 의도: 개별 요소가 아니라 각 열마다 unique() 적용
# data.unique(axis=1) # DF 에 unique 없다

cols= data.columns
uniqueSum= 0
# print(len(data['CRIM'].unique()))
# print(len(data['CRIM']))
for col in cols:
#     print(data[col].unique().count())
    uniqueSum+= len(data[col].unique())
print(uniqueSum/len(cols))
print(uniqueSum, len(cols))
print(218 *14)

# print(data.info()) # 'RM' 에만 null 이 있다.
print(len(data['RM'].unique()))
print(pd.DataFrame(data['RM'].unique()).count())


# [dataframe apply unique](https://stackoverflow.com/a/48409827)
# df.apply(lambda x: pd.unique(x).tolist())
# - unique() 는 전역함수가 아니기에, 그냥 쓰면 "name 'unique' is not defined"

# [applymap 'float' object is not iterable](https://stackoverflow.com/a/58742977)
# - null 이 있으면 적용이 안 된다

# ##### [공식  applymap](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.applymap.html)
# 
# 
# ##### [map, apply, applymap](http://www.leejungmin.org/post/2018/04/21/pandas_apply_and_map/)
# 예제가 어렵. 공식을 보자
# - map: Series 에만. df['winning_rate']  = df['team'].map(lambda x : 커스텀함수return(x)). 요소하나하나에 적용. 함수뿐 아니라 딕셔너리 및 Series 도 가능
# - apply: DF, Series 둘다. 
# - applymap: DF. 각 요소에 적용.

# In[212]:


# import pdb
# a= 5
# pdb.set_trace() 
# b 숫자. 브레이크포인트 대상함수에다가 # c continue 다음 브레이크까지 # n next # 변수명 값보기
# print(5, "이게 뭐지. python debugger")

