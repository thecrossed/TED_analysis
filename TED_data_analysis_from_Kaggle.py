
# coding: utf-8
# Title
# Author
# Date# Question 1: What are the most popular tags in talks?
# Question 2: What are the common features(ratings) of popular talks?
# Question 3: Does the length of title influence the popularity?
# In[1]:


# Loading dataset and libraries
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


wordSyl = nltk.corpus.cmudict.dict()

td = pd.read_csv('ted_main.csv')
td.head()

# Summary Statistics 
# Tables and graphs that describe the data.1 - Histograms
Let’s examine the distributions of the key parameters that we’ll be using. 
To the right are histograms of most of our numerical variables. Below are some more detailed histograms, with a yellow line for the median and blue line for the mean.fig, ax1 = plt.subplots(figsize = (10,7))
ax1.grid(zorder=1)
ax1.xaxis.grid(False)
plt.hist(td['views'],zorder=0,color = "#000080")
plt.xlabel('Views')
plt.ylabel('Count')
plt.title('TED View Distribution')
plt.axvline(x=td['views'].mean(),linestyle='--')
plt.axvline(x=td['views'].median(),color = '#FFFF00',linestyle='-.')
plt.legend(['mean of views','median of views'], loc='upper right')
plt.show()
# In[245]:


td_view_rank = td.sort_values('views',ascending=False)
title_rank = td_view_rank[['views','tags','title','ratings']]
title_rank

# title中有问号与观看数是否存在相关
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
# PointbiserialrResult(correlation=-0.014297140204136572, pvalue=0.47050779285729782)# title 单词数与views是否相关
length_title = []
for t in title_rank['title']:
    t = word_tokenize(t)
    length_title.append(len(t))
title_rank['title_leng'] = length_title
c = title_rank['title_leng'].corr(title_rank['views'])
#plt.scatter(title_rank.title_leng, title_rank.views, alpha=0.35,color='grey')
plt.hist(title_rank['title_leng'],zorder=0,color = "#FFA500")
plt.axvline(x=title_rank['title_leng'].mean(),linestyle='--')
plt.axvline(x=title_rank['title_leng'].median(),color = '#AEFD8E',linestyle='-.')# 有question words与views之间的相关
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
# PointbiserialrResult(correlation=0.074441732813935999, pvalue=0.00016822244638525606)# title 的中的情感倾向与views
title_pol = []
for t in title_rank['title']:
    t = TextBlob(t)
    title_pol.append(t.sentiment.polarity)
title_rank['title_pol'] = title_pol
#pol_pos=title_rank[title_rank.title_pol>0]
#pol_neg =title_rank[title_rank.title_pol<0]
3pol_neg.head()
#pol_pos.head()
#p = stats.ttest_ind(pol_neg['views'],pol_pos['views'])
#p
#c = pol_neg['views'].corr(pol_pos['views'])
#print(c)
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
c = title_rank['title_ease'].corr(title_rank['title_leng'])
#result: 0.283477600857 title_ease & title_leng corr
plt.scatter(title_rank.title_ease, title_rank.title_leng, alpha=0.35,color='grey')
fig, ax1 = plt.subplots(figsize = (10,7))
ax1.grid(zorder=1)
ax1.xaxis.grid(False)
plt.hist(td['comments'],range(0,4000,250),zorder=0,color = "#FFA500")
plt.xlabel('comments')
plt.ylabel('Count')
plt.title('TED comments Distribution')
plt.axvline(x=td['comments'].mean(),linestyle='--')
plt.axvline(x=td['comments'].median(),color = '#AEFD8E',linestyle='-.')
plt.legend(['mean of comments','median of comments'], loc='upper right')
plt.show()fig, ax1 = plt.subplots(figsize = (10,7))
ax1.grid(zorder=1)
ax1.xaxis.grid(False)
plt.hist(td['duration'],range(0,4000,250),zorder=0,color = "#66B266")
plt.xlabel('duration(seconds)')
plt.ylabel('How many talks in that duration')
plt.title('TED duration Distribution')
plt.axvline(x=td['duration'].mean(),linestyle='--')
plt.axvline(x=td['duration'].median(),color = '#FFFF7F',linestyle='-.')
plt.legend(['mean of duration','median of duration'], loc='upper right')
plt.show()
# [',]# 排名前10的tag
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
title_rank['ratings']
# In[234]:




# ratings的情感倾向与views数量 2
c = title_rank['views'].corr(title_rank['rat_cal'])
print (c)
plt.scatter(title_rank.views, title_rank.rat_cal, alpha=0.50,color='grey')
plt.show()
title_rank.sort_values('rat_cal',ascending = False)
# In[263]:


# rating中各adj的比例与views的相关
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
adj_count
#adj_count['title'] = views
#title_rank = pd.merge(title_rank,adj_count)
#result = pd.merge(left, right, on='k')
        #print(adj,adj_count_clean,count)

