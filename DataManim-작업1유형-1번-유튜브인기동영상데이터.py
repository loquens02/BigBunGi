#!/usr/bin/env python
# coding: utf-8

# # [DataManim-작업1유형](https://www.datamanim.com/dataset/03_dataq/typeone.html#id2)

# 1. 유튜브 인기동영상 데이터 << 
# 2. 유튜브 공범컨텐츠 동영상 데이터
# 3. 월드컵 출전선수 골기록 데이터
# 4. 서울시 따릉이 이용정보 데이터
# 5. 전세계 행복도 지표 데이터
# 6. 지역구 에너지 소비량 데이터
# 7. 포켓몬 정보 데이터
# 8. 대한민국 체력장 데이터
# 9. 기온 강수량 데이터
# 10. 서비스 이탈예측 데이터
# 11. 성인 건강검진 데이터
# 12. 자동차 보험가입 예측데이터
# 13. 핸드폰 가격 예측데이터
# 14. 비행탑승 경험 만족도 데이터
# 15. 수질 음용성 여부 데이터
# 16. 의료 비용 예측 데이터
# 17. 킹카운티 주거지 가격예측문제 데이터
# 18. 대학원 입학가능성 데이터
# 19. 레드 와인 퀄리티 예측 데이터
# 20. 약물 분류 데이터
# 21. 사기회사 분류 데이터
# 22. 센서데이터 동작유형 분류 데이터
# 23. 현대 차량 가격 분류문제 데이터
# 24. 당뇨여부판단 데이터
# 25. 넷플릭스 주식 데이터
# 26. 220510추가

# ## 유튜브 인기동영상 데이터
# - 데이터: [출처-캐글](https://www.kaggle.com/rsrishav/youtube-trending-video-dataset?select=KR_youtube_trending_data.csv)
# - 데이터 설명 : 유튜브 데일리 인기동영상 (한국)
# - 가공 dataurl : https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv

# In[2]:


import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv",index_col=0)
df.head()


# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# ### 인기동영상 제작횟수가 많은 채널 상위 10개명을 출력하라 (날짜기준, 중복포함)

# In[2]:


df.columns


# In[9]:


# print(df.channelTitle.value_counts().head(10).index)
print(list(df.channelTitle.value_counts().head(10).index))


# 답: ['장삐쭈', '총몇명', '파뿌리', '짤툰', '런닝맨 - 스브스 공식 채널', '엠뚜루마뚜루 : MBC 공식 종합 채널', 'SPOTV', '채널 십오야', '이과장', 'BANGTANTV']
# > answer =list(df.loc[df.channelId.isin(df.channelId.value_counts().head(10).index)].channelTitle.unique());
# print(answer)
# 
# 뭐가 더 긴데, 답은 같다. list 붙여주면 정답 형식에 더 알맞다

# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# ### 논란으로 인기동영상이 된 케이스를 확인하고 싶다. dislikes수가 like 수보다 높은 동영상을 제작한 채널을 모두 출력하라

# In[18]:


print(list(df.loc[df['dislikes'] > df['likes'], 'channelTitle'].unique()))


# 답: ['핫도그TV', 'ASMR 애정TV', '하얀트리HayanTree', '양팡 YangPang', '철구형 (CHULTUBE)', '왜냐맨하우스', '(MUTUBE)와꾸대장봉준', '오메킴TV', '육지담', 'MapleStory_KR', 'ROAD FIGHTING CHAMPIONSHIP', '사나이 김기훈', '나혼자산다 STUDIO', 'Gen.G esports']
# > answer =list(df.loc[df.likes < df.dislikes].channelTitle.unique()) <p>
# print(answer)

# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# ### 채널명을 바꾼 케이스가 있는지 확인하고 싶다. channelId의 경우 고유값이므로 이를 통해 채널명을 한번이라도 바꾼 채널의 개수를 구하여라

# In[46]:


# print(df.columns)
# df.groupby(['channelId','channelTitle'])['channelTitle'].count()
# print(len(df['channelId'].unique())) # 1770 개. 만약 시험이었으면 1770개 for문 돌렸을듯
df.loc[df['channelId']=='UC-0C8yVGJy-cS4FGlYKelWw','channelTitle'].unique()

## channelId 당 딸린 channelTitle 의 고윳값이 여러 개인 걸 찾고 싶다.
# df[['channelId','channelTitle']].value_counts()


# 모르겠다. 정답 참조:
# #### 중복제거 : df[['컬럼1', '컬럼2']].drop_duplicates()

# In[45]:


change = df[['channelTitle','channelId']].drop_duplicates().channelId.value_counts()
target = change[change>1]
print(len(target))


# In[90]:


# df[['channelTitle','channelId']]
# df[['channelTitle','channelId']].drop_duplicates() # 인기 동영상에 여럿 올라갔어도 1개씩만 체크
changeNameList= df[['channelTitle','channelId']].drop_duplicates()['channelId'].value_counts() > 1
# 1개씩만 남겼는데도 channelId 가 여럿(1개 초과) 보이는 것 == 채널명 바꾼 것
print(changeNameList.sum(0))

## 무슨 채널일까.
# df.loc[df['channelId'] in changeNameList]
# changeNameList 에 있는건 df['channelId'].count() 와 달라서 바로 넣을 수 없다.
each1= df[['channelTitle','channelId']].drop_duplicates()['channelId'].value_counts()
changeChannelList= each1[changeNameList].index# 확인 .count()
changeChannelDFs= df[df['channelId'].map(lambda x: x in changeChannelList)]
# 뭔가 이상하다. 심하게 명령어 중복인데.
resDF= changeChannelDFs[['channelTitle','channelId']].drop_duplicates().sort_values('channelId') #확인
resDF.to_csv('data/channelNameChanges.csv', index=False, encoding='utf-8-sig')


# In[ ]:





# '22.6.22 16:52 피곤../ 19:14 운동하고 저녁먹고 옴

# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# ### 일요일에 인기있었던 영상들중 가장많은 영상 종류(categoryId)는 무엇인가?

# In[107]:


df.loc[((pd.to_datetime(df['trending_date2'])).dt.dayofweek==6), 'categoryId'].value_counts().iloc[[0]].index[0]


# 답
# - dt.dayofweek==6 (일요일) 을 외우지 않아도 된다
# > dt.day_name() == 'Sunday' 를 기억한다면
# - index 여럿 있을 때 첫 번째 것만 뽑고 싶다면 .iloc[[0]].index[0] 하지 않아도 된다
# > index[0] 만 해도 충분하다

# In[108]:


df['trending_date2'] = pd.to_datetime(df['trending_date2'])
answer =df.loc[df['trending_date2'].dt.day_name() =='Sunday'].categoryId.value_counts().index[0]
print(answer)


# In[149]:


######  ######  ######  ######  ######  ######  ######  ######  


# ### 각 요일별 인기 영상들의 categoryId는 각각 몇 개 씩인지 하나의 데이터 프레임으로 표현하라

# In[115]:


df['dayName']= (pd.to_datetime(df['trending_date2'])).dt.day_name()
# df.columns
display(pd.DataFrame(df.groupby(['dayName'])['categoryId'].count()))


# 답:
# - 문제를 잘못 이해했다. "요일의" "인기 영상들의" "categoryId는 각각" 이라서 3차원을 내었어야 했는데
# - groupby( ,as_index=False) 옵션을 주면 만들다 만 것처럼 생긴 index 대신 값이 채워져있는 DF 를 얻을 수 있다
# > df.groupby([df['trending_date2'].dt.day_name(), 'categoryId'], as_index=False).size()

# In[124]:


# df['trending_date2'].dt.day_name() # 앞에서 이미 to_datetime 해서
group= df.groupby([df['trending_date2'].dt.day_name(), 'categoryId'], as_index=False).size()
display(group)


# 파이썬에도 엑셀에서 보면서 만들던 피벗이 있다. 
# - 컬럼3개로 이루어진 group
# - (원래 index 중 하나였던) categoryId를 index 삼고
# - (원래 index 중 하나였던) trendig_date2 라는 허물을 뒤집어쓴 요일을 column 삼아서
# - pivot 을 만든다
# > group.pivot(index='categoryId', columns='trending_date2')

# In[116]:


group = df.groupby([df['trending_date2'].dt.day_name(),'categoryId'],as_index=False).size()
answer= group.pivot(index='categoryId',columns='trending_date2')
display(answer)


# (다시) 각 요일별 인기 영상들의 categoryId는 각각 몇 개 씩인지 하나의 데이터 프레임으로 표현하라

# [python count size difference](https://stackoverflow.com/a/33346694) 
# - size 는 null 을 포함
# - 댓글: .size() 함수는 모든 열에 대해 .column()이 사용되는 동안 특정 열의 집계 값만 가져옵니다
# - .count() 는 모든 열의 집계를 가져오지만, 같은DF.groupby([]).집계() 환경에서는 어차피 같은 결과를 가져오기에 이 상황에서는 size()가 적절하다

# In[147]:


# print(df.columns)

### Step1. 재료 데이터: 요일, categoryId, 개수(size) 를 포함한 DF 를 만든다
## 각 요일별 > trending_date2 의 dt.day_name()
## 재료1: 요일
df['trending_date2']= pd.to_datetime(df['trending_date2'])
df['trending_date2'].dt.day_name() 
## 재료2: categoryId
'categoryId'

## 대상: count() 말고 size() < 다른가?
# 이상스레 여럿 나온다. 왜? display(df.groupby([df['trending_date2'].dt.day_name(), 'categoryId']).count())
df.groupby([df['trending_date2'].dt.day_name(), 'categoryId']).size()

# 1. groupby () 안은 전부 [] 대괄호로 둘려야 한다.
# 2. count() 랑 size() 가 뭔차이인지는 몰라도 결과가 완전 딴판이다.
# - count: groupby 로 준 기준에 대해 값이 컬럼 개수만큼 똑같은 게 여럿 나오고
# - size: groupby 로 준 기준에 대해 값이 컬럼 한 줄만 나온다.
# => 결과 하나씩 쳐보고 눈으로 봐도 count() 가 쓸모가 없고 size() 가 괜찮아 보인다.

type(df.groupby([df['trending_date2'].dt.day_name(), 'categoryId']).size())
# .groupby([]).size() 결과물은 Series. 이걸 DF 형태로 만들면서 빈칸을 채우려면, as_index=False
ingredient= df.groupby([df['trending_date2'].dt.day_name(), 'categoryId'], as_index=False).size()
# display(ingredient)

### Step2. 가공: 요일 별, categoryId 별, 개수 를 표현한다
# help(ingredient.pivot) # pivot(index=None, columns=None, values=None) # column 아니고 columns
ingredient.pivot(index='trending_date2', columns='categoryId')
# index-columns 뭐가 되든 상관없을 것 같긴 한데, 상수가 x축에 있으면 좀더 예쁘니까
display(ingredient.pivot(index='categoryId', columns='trending_date2'))


# In[ ]:





# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# # 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다. viewcount대비 댓글수가 가장 높은 영상을 확인하라 (view_count값이 0인 경우는 제외한다)

# In[6]:


# df.columns
# comment_count / view_count 값이 가장 높은 것을 찾고, 소수점 4자리 밑으로 적당히 절삭해서 비교 기준값으로 삼았다.
nonZeroDF= df[df['view_count']!=0]
maxRate= round(max(nonZeroDF['comment_count'] / nonZeroDF['view_count']), 4)
# print(maxRate)
# display(nonZeroDF[((nonZeroDF['comment_count'] / nonZeroDF['view_count'])>maxRate)] )
display(nonZeroDF.loc[((nonZeroDF['comment_count'] / nonZeroDF['view_count'])>maxRate), 'title'].values[0] )


# 답
# - 영상 줄을 뽑는 게 아니라 영상 이름을 뽑는 거였네

# In[164]:


target2= df.loc[df.view_count!=0]
t = target2.copy()
t['ratio'] = (target2['comment_count']/target2['view_count']).dropna()
result = t.sort_values(by='ratio', ascending=False).iloc[0].title
print(result)


# In[ ]:





# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# # 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다.viewcount대비 댓글수가 가장 낮은 영상을 확인하라 (view_counts, ratio값이 0인경우는 제외한다.)

# In[87]:


import math
# print(df.columns)
## 댓글0, view_counts0 제외 # 
nonzeroDF= df[(df['view_count']!=0) & (df['comment_count']!=0)]
rate= nonzeroDF['comment_count'] / nonzeroDF['view_count']
boundary= round(min(rate), 12) # 가장 작은값보다 좀더 작은데, 어떻게 답이 나왔지???. 대소비교에도 부동소수점 에러가 있나??
# print(boundary)
nonzeroDF.loc[rate < boundary, 'title'].values[0]


# 답: (간단해 보인다)

# In[39]:


ratio = (df['comment_count'] / df['view_count']).dropna().sort_values()
ratio[ratio!=0].index[0]

result= df.iloc[ratio[ratio!=0].index[0]].title
print(result)


# In[56]:


df[(df['comment_count'] / df['view_count']).isnull()] # view_count 가 0인 컬럼만 나온다. !! Inf 도 null 로 치나봄?
# df['comment_count'] / df['view_count']


# In[ ]:





# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# # like 대비 dislike의 수가 가장 적은 영상은 무엇인가? (like, dislike 값이 0인경우는 제외한다)

# In[86]:


# df.columns
# help(df.align)
# (/df['likes']).dropna()
dl= df[df['dislikes']!=0]
l= df[df['likes']!=0]
dl, l= dl.align(l, join= 'inner')
# print(dl.shape, l.shape) # 일치!
dlRate= dl['dislikes']/l['likes']
boundary= round(dlRate.min(), 8)+ 1e-8 # 가장 작은 값보다 더 작아서 임의의 작은 값을 더해주었다. 하나 나올 때까지

# print(dlRate.min(), boundary)
dl.loc[dlRate < boundary, 'title'].values[0]


# In[91]:


# help(pd.qcut) # 같은 개수로 쪼개는 건데, 사분위수로 나누는 것 대비 설득력이 떨어져 보인다


# In[92]:


target = df.loc[(df.likes !=0) & (df.dislikes !=0)]
num = (target['dislikes']/target['likes']).sort_values().index[0]

answer = df.iloc[num].title
print(answer)


# In[99]:


target= df[((df.likes !=0) & (df.dislikes !=0))] # df.align 안 써도 충분
num= (target['dislikes']/target['likes']).sort_values().index[0] # 어차피 열 하나 뿐이라 sort 해도 메모리 얼마 안 들 것
# index 를 살릴 수 있어서 비교없이도 바로 찾을 수 있다.
print(df.iloc[num].title) # print 안에 넣으면 '' 없앨 수 있다.


# In[ ]:





# In[148]:


######  ######  ######  ######  ######  ######  ######  ######  


# # 가장많은 트렌드 영상을 제작한 채널의 이름은 무엇인가? (날짜기준, 중복포함)
# 

# In[100]:


df.columns


# 날짜 [python day equal](https://stackoverflow.com/a/6407393) dt.date, dt.time, dt.datetime

# In[127]:


## 트렌드 영상이 뭐지?

df['trending_date2']= pd.to_datetime(df['trending_date2'])
year= df['trending_date2'].dt.year
month= df['trending_date2'].dt.month
day= df['trending_date2'].dt.day
# df.groupby(year)['channelTitle'].count() # 2021 년 뿐 
df.groupby(month)['channelTitle'].count().sort_values(ascending=False).index[0] # 6월
# print(df.groupby(day)['channelTitle'].count().sort_values(ascending=False).index[0]) # 2일
## 이건 어느 날에 (채널 관계없이)가장 많은 영상이 인기 영상에 있었는지 말 하는 것.
# + 특정 날짜만 보는 건 어떻게 하지?
date1= df['trending_date2'].dt.date
bestDate= df.groupby(date1)['channelTitle'].count().sort_values(ascending=False).index[0] # 여긴 1일이네 ㄷㄷ
bestDateDF= df[df['trending_date2'].dt.date==bestDate]
bestDateDF.groupby(['trending_date2', 'channelTitle'])['channelTitle'].count().sort_values(ascending=False).index[0][1]


# 답

# In[128]:


answer = df.loc[df.channelId ==df.channelId.value_counts().index[0]].channelTitle.unique()[0]
print(answer)


# In[136]:


# 트렌드 영상 == 인기 영상
# 날짜 기준이라고 써있어서 헷갈림.
# df.channelId.value_counts()
# df[df.channelId ==df.channelId.value_counts().index[0]].channelTitle.unique()
# df.loc[df.channelTitle ==df.channelTitle.value_counts().index[0], 'channelTitle'].unique()
df.channelTitle.value_counts().index[0] # 차피 title 볼거면 왜 굳이 channelId ? 채널 바뀌었던거 고려하려고?

