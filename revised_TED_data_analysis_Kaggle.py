
# coding: utf-8
【手把手Python数据分析】看看Ted Talks最火的视频和哪些因素相关？

作者：吕天旻
日期：2018年6月

数据来源：TED官方在Kaggle（著名数据竞赛平台）上发布了自成立以来至2017年9月视频（2550个）数据。其中包括了：
comments, 评论数， （举例：4553）
description, 介绍， （Sir Ken Robinson makes an entertaining and pro...）
duration, 时长， （1164秒）
event, 活动， （TED2006）
film_date, 拍摄时间， （1140825600, Unixstamp格式）
languages, 字幕语言数， （60种）
main_speaker, 主要演说者， （Ken Robinson）
name, 视频名， （Ken Robinson: Do schools kill creativity?）
num_speaker, 演说人数， （1）
published_date, 发布时间， （1151367060, Unixstamp格式）
ratings, 评价， （[{'id': 7, 'name': 'Funny', 'count': 19645}, {...）
related_talks, 相关视频， （[{'id': 865, 'hero': 'https://pe.tedcdn.com/im...）
speaker_occupation, 演说者职业， （Author/educator）
tags, 标签， （['children', 'creativity', 'culture', 'dance',...）
title, 标题， （Do schools kill creativity?）
url, 视频url地址， （https://www.ted.com/talks/ken_robinson_says_sc...）
views，播放数， （47227110）
以及视频的transcript（字幕）

传送门：https://www.kaggle.com/rounakbanik/ted-talksTomer Eldor 曾于今年1月发表了一篇对于上述数据的分析文章，Data Reveals: What Makes a Ted Talk Popular?
得出以下基本结论，供参考：

高播放量（views）视频的特征有：
1. 高评论数(comments)，不奇怪
2. 被翻译成多种语言（languages），不奇怪
3. 1和2结合所产生的效应，比1，2单独的效应更强
4. 不能太短，时长（duration）与流行度没什么关系，但最火的视频长度在8～18分钟之间
5. 更多的标签数（tag），最好在3～8个之间
6. 最好在一个周五上传（published_date）

详细的文章内容，可以通过链接进入：
https://towardsdatascience.com/data-reveals-what-makes-a-ted-talk-popular-6bc15540b995

介于Eldor主要在数值类变量间，寻找与播放量相关的因素，本文主要从文本类变量（e.g. title，ratings，tags，main_speaker）来寻找与播放量相关的因素。

先抛出问题：
问题一：title的长短与视频播放量之间是否存在相关？
问题二：title中的负面情绪是否与视频播放量之间是否存在相关？
问题三：流行视频常贴哪些tag？
问题四：视频播放量与ratings数量是否存在相关？涉及的领域知识：
1.Python：
2.自然语言处理包
3.matplotlib数据可视化
4.统计学相关（correlation）分析# Loading dataset and libraries
import numpy as np
import pandas as pd # pandas
import csv
import matplotlib.pyplot as plt # module for plotting 
from matplotlib.patches import Polygon
from matplotlib import gridspec
from matplotlib import rcParams
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats
from textblob import TextBlob
from textstat.textstat import textstat
import gender_guesser.detector as gender

wordSyl = nltk.corpus.cmudict.dict()

td = pd.read_csv('ted_main.csv')
td.head()fig, ax1 = plt.subplots(figsize = (10,7))
ax1.grid(zorder=1)
ax1.xaxis.grid(False)
plt.hist(td['views'],zorder=0,color = "#000080")
plt.xlabel('Views')
plt.ylabel('Count')
plt.title('TED View Distribution')
plt.axvline(x=td['views'].mean(),linestyle='--')
plt.axvline(x=td['views'].median(),color = '#FFFF00',linestyle='-.')
plt.legend(['mean of views','median of views'], loc='upper right')
plt.show()td_view_rank = td.sort_values('views',ascending=False)
title_rank = td_view_rank[['views','tags','title','ratings','main_speaker']]
title_rank.head()# 性别比例
gender_speaker = []
names = []
for name in title_rank['main_speaker']:
    name = name.split()[0]
    names.append(name)
d = gender.Detector()
for n in names:
    gender_speaker.append(d.get_gender(n))
title_rank['gender_speaker'] = gender_speaker
title_rank.head(100)
gender_count = title_rank['gender_speaker'].value_counts()
gender_count# title 单词数与views是否相关
length_title = []
for t in title_rank['title']:
    t = word_tokenize(t)
    length_title.append(len(t))
title_rank['title_leng'] = length_title
c = title_rank['title_leng'].corr(title_rank['views'])
#plt.scatter(title_rank.title_leng, title_rank.views, alpha=0.35,color='grey')
plt.hist(title_rank['title_leng'],zorder=0,color = "#FFA500")
plt.axvline(x=title_rank['title_leng'].mean(),linestyle='--')
plt.axvline(x=title_rank['title_leng'].median(),color = '#AEFD8E',linestyle='-.')# title中有问号与观看数是否存在相关
ques_mark = []
for t in title_rank['title']:
    t = word_tokenize(t)
    if '?' in t:
        ques_mark.append(True)
    else:
        ques_mark.append(False)
title_rank['ques_mark'] = ques_mark
a = title_rank['ques_mark']
b = title_rank['views']
ques_views_ave = title_rank.groupby(['ques_mark'])['views'].mean()
c = stats.pointbiserialr(a,b)
# PointbiserialrResult(correlation=0.032033285169904391, pvalue=0.1058300461339059)# title中有感叹号与观看数是否存在相关
exc_mark = []
for t in title_rank['title']:
    t = word_tokenize(t)
    if '!' in t:
        exc_mark.append(True)
    else:
        exc_mark.append(False)
title_rank['exc_mark'] = exc_mark
a = title_rank['exc_mark']
b = title_rank['views']
ques_views_ave = title_rank.groupby(['exc_mark'])['views'].mean()
c = stats.pointbiserialr(a,b)
# PointbiserialrResult(correlation=-0.014297140204136572, pvalue=0.47050779285729782)# 有question words与views之间的相关
ques_words=['what','when','why','who','how','which','whose','where']
ques_start=[]
for t in title_rank['title']:
    t = word_tokenize(t)
    if set(t).intersection(ques_words) != set():
        ques_start.append(True)
    else:
        ques_start.append(False)

title_rank['ques_start'] = ques_start
a = title_rank['ques_start']
b = title_rank['views']
ques_views_ave = title_rank.groupby(['ques_start'])['views'].mean()
c = stats.pointbiserialr(a,b)
plt.scatter(title_rank.ques_start, title_rank.views, alpha=0.35,color='grey')
plt.show()
# PointbiserialrResult(correlation=0.074441732813935999, pvalue=0.00016822244638525606)# title 的中的情感倾向与views
title_pol = []
for t in title_rank['title']:
    t = TextBlob(t)
    title_pol.append(t.sentiment.polarity)
title_rank['title_pol'] = title_pol
#pol_pos=title_rank[title_rank.title_pol>0]
#pol_neg =title_rank[title_rank.title_pol<0]
#pol_neg.head()
#pol_pos.head()
#p = stats.ttest_ind(pol_neg['views'],pol_pos['views'])
#p
c = title_rank['title_pol'].corr(title_rank['views'])
print(c)
# Result: 0.0556131305115
#fig, ax1 = plt.subplots(figsize = (10,7))
#ax1.grid(zorder=1)
#ax1.xaxis.grid(False)
#plt.hist(title_rank['title_pol'],zorder=0,color = "#000080")
#plt.xlabel('Views')
#plt.ylabel('Count')
#plt.title('TED View Distribution')
#plt.axvline(x=title_rank['title_pol'].mean(),linestyle='--')
#plt.axvline(x=title_rank['title_pol'].median(),color = '#FFFF00',linestyle='-.')
#plt.legend(['mean of views','median of views'], loc='upper right')
#plt.show()# title 易懂性与views
title_ease = []
for t in title_rank['title']:
    title_ease.append(textstat.flesch_reading_ease(t))
title_rank['title_ease']=title_ease
#title_rank.sort_values('title_ease')
c = title_rank['title_ease'].corr(title_rank['views'])
#result: 0.283477600857 title_ease & title_leng corr
plt.scatter(title_rank.title_ease, title_rank.views, alpha=0.35,color='grey')
fig, ax1 = plt.subplots(figsize = (10,7))
ax1.grid(zorder=1)
ax1.xaxis.grid(False)
plt.hist(title_rank['title_ease'],zorder=0,color = "#000080")# 排名前10的tag
m = ['[',"'",',',']']
tags_split = []
indi_tag = []
for t in title_rank['tags']:
    t = t.split("'")
    #print(t)
    for i in t:
        if i[0] in m:
            t.remove(i)
    tags_split.append(t)
title_rank['tags_split'] = tags_split
for row in tags_split:
    for w in row:
        if w in indi_tag:
            continue
        else:
            indi_tag.append(w)
tags_count = []
for t in title_rank['tags_split']:
    tags_count.append(len(t))
title_rank['tags_count']=tags_count

indi_tag_view = {}
view_tag = dict(zip(title_rank.views, title_rank.tags_split))
#for v in view_tag.values():

indi_tag_view = indi_tag_view.fromkeys(indi_tag)
indi_tag_view
#for k,v in view_tag.items():
#    sum += k
#print(sum)
sum = 0
for k,v in indi_tag_view.items():
    indi_tag_view[k] = 0
for key in indi_tag_view:
    for k,v in view_tag.items():
        if key in v:
            indi_tag_view[key] += k

#tag_view_count = pd.DataFrame(indi_tag_view, orient='index')
#tag_view_count.columns = ['tag']
#tag_view_count
list(indi_tag_view.keys())
list(indi_tag_view.values())
tag_view_count_dict = {'tag':list(indi_tag_view.keys()),'view_count':list(indi_tag_view.values())}
tag_view_count = pd.DataFrame.from_dict(tag_view_count_dict)
tag_view_count=tag_view_count.sort_values('view_count',ascending = False)
#plt.bar(tag_view_count['tag'], tag_view_count['view_count'], color='green', width=0.8)
#plt.barh(np.arange(1,len(tag_view_count['tag'])+1),tag_view_count['view_count'],align='center',color='blue')
#plt.yticks(range(1,len(AM_FullFreq.emptyfullfreq)+1),AM_Full_top.station,size='small')
#plt.show()
tag_rank_10 = tag_view_count.head(10)
plt.barh(np.arange(1,len(tag_rank_10['tag'])+1),tag_rank_10['view_count'],align='center',color='blue')
plt.yticks(range(1,len(tag_rank_10.tag)+1),tag_rank_10.tag,size='small')
plt.show()# ratings的情感倾向与views数量 1
rat_cal = []
for r in title_rank['ratings']:
    rat_split = r.split("'")
    sum= 0
    for i in range(14):
    #print (rat_split[i*8+5],TextBlob(rat_split[i*8+5]).sentiment.polarity*int(rat_split[i*8+8].split('}')[0].split(' ')[1]))
        sum += TextBlob(rat_split[i*8+5]).sentiment.polarity*int(rat_split[i*8+8].split('}')[0].split(' ')[1])
    rat_cal.append(sum)
title_rank['rat_cal'] = rat_cal
title_rank['ratings']# ratings的情感倾向与views数量 2
c = title_rank['views'].corr(title_rank['rat_cal'])
print (c)
plt.scatter(title_rank.views, title_rank.rat_cal, alpha=0.50,color='grey')
plt.show()
title_rank.sort_values('rat_cal',ascending = False)# rating中各adj的比例与views的相关
#adjs = [Funny,Beautiful,Ingenious,Courageous,Longwinded,
#        Confusing,Informative,Fascinating,Unconvincing,
#        Persuasive,Jaw_dropping,OK,Obnoxious,Inspiring]
adjs ={'Beautiful':[],'Confusing':[],'Courageous':[],
      'Fascinating':[],'Funny':[],'Informative':[],
      'Ingenious':[],'Inspiring':[],'Jaw-dropping':[],
      'Longwinded':[],'OK':[],'Obnoxious':[],
      'Persuasive':[],'Unconvincing':[]}

for text in title_rank['ratings']:
    
    for i in range(14):
        adj = text.split("'")[i*8+5]
        adj_count = text.split("'")[i*8+8]
        adj_count_clean = int(adj_count.split("}")[0].split(" ")[1])
        adjs[adj].append(adj_count_clean)

adj_count = pd.DataFrame.from_dict(adjs)
#adj_count['views'] = title_rank['views']
adj_count['title'] = list(title_rank['title'])
adj_count['views'] = list(title_rank['views'])
#c = adj_count['Inspiring'].corr(adj_count['Informative'])

#adj_count['title'] = views
#title_rank = pd.merge(title_rank,adj_count)
#result = pd.merge(left, right, on='k')
        #print(adj,adj_count_clean,count)