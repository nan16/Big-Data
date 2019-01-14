
# coding: utf-8

# ### Merging CSVs

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filenames = ['tweets_4-6Nov.csv','tweets_7-9Nov.csv', 'tweets_10Nov.csv','tweets_11Nov.csv']
with open('merged_file', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)



# In[7]:


df = pd.read_csv("merged_file", names = ["time", "text"])

df.head()


# ### Data Cleaning

# In[3]:


df['length'] = [len(t) for t in df.text]
df.head(50)


# ### Finding Tweet Length

# In[4]:


import re

#df['text'] = df['text'].map(lambda x:str(x)[2:])
#df['text'] = df['text'].map(lambda x:str(x)[:-1])
df['text'] = df['text'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
df['text'] = df['text'].apply(lambda x: re.sub('  ', ' ', x))

df.head()


# In[5]:


LATIN_1_CHARS = (
    ('\xe2\x80\x99', "'"),
    ('\xc3\xa9', 'e'),
    ('\xe2\x80\x90', '-'),
    ('\xe2\x80\x91', '-'),
    ('\xe2\x80\x92', '-'),
    ('\xe2\x80\x93', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x94', '-'),
    ('\xe2\x80\x98', "'"),
    ('\xe2\x80\x9b', "'"),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9c', '"'),
    ('\xe2\x80\x9d', '"'),
    ('\xe2\x80\x9e', '"'),
    ('\xe2\x80\x9f', '"'),
    ('\xe2\x80\xa6', '...'),
    ('\xe2\x80\xb2', "'"),
    ('\xe2\x80\xb3', "'"),
    ('\xe2\x80\xb4', "'"),
    ('\xe2\x80\xb5', "'"),
    ('\xe2\x80\xb6', "'"),
    ('\xe2\x80\xb7', "'"),
    ('\xe2\x81\xba', "+"),
    ('\xe2\x81\xbb', "-"),
    ('\xe2\x81\xbc', "="),
    ('\xe2\x81\xbd', "("),
    ('\xe2\x81\xbe', ")"),
)


def clean_latin1(data):
    try:
        return data.encode('utf-8')
    except UnicodeDecodeError:
        data = data.decode('iso-8859-1')
        for _hex, _char in LATIN_1_CHARS:
            data = data.replace(_hex, _char)
        return data.encode('utf8')
    
    df['text'] = df['text'].apply(lambda x: clean_latin1(x))
    
df.head(25)


# In[6]:


df1=df


# In[7]:


# import re
# df['text'] = df['text'].map(lambda line: re.sub("[^a-zA-Z]","", line))
# df.head()


# ### Average Tweet Length

# In[8]:


mean = np.mean(df['length'])

print("Average Tweet Length : {} Characters".format(mean))


# In[9]:


tlen = pd.Series(data=df['length'].values, index=df['time'])
tlen.plot(figsize=(16,4), color='r');

#df1.groupby(['time_hour']).SA.sum().plot(figsize=(16,4), color='b')


# In[10]:


from textblob import TextBlob
from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


# In[11]:


df['SA'] = np.array([ analize_sentiment(tweet) for tweet in df['text'] ])
# We display the updated dataframe with the new column:
display(df.head(10))


# In[12]:


pos_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(df['text']) if df['SA'][index] < 0]


# ### Ratio of Positive, Neutral, and Negative Tweets

# In[13]:



print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(df['text'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(df['text'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(df['text'])))


# In[14]:


df_pos = df.filter(items=['SA==1','text'])
df_nt = df.filter(items=['SA==0','text'])
df_neg = df.filter(items=['SA==-1','text'])


# In[15]:


df1 = df


# In[16]:


df1.head()


# In[17]:


df1['time'] = pd.to_datetime(df1['time'])


# In[18]:


df1['time_hour'] = df1.time.apply(lambda x: x.hour)


# In[19]:


df1.head()


# ### Visualizing No of Tweets vs TIme

# In[20]:


df1.index=df1.time_hour


# In[21]:


times = pd.to_datetime(df1.time)


# In[22]:


df1.groupby(['time_hour']).SA.sum()


# In[23]:


df1.groupby(['time_hour']).SA.sum().plot(figsize=(16,4), color='b')


# In[24]:


df2= df1.groupby(['time_hour']).SA.sum()


# In[25]:


df2.head()


# In[26]:


from pandas import Series
from matplotlib import pyplot
series = Series(df2)


# In[27]:


pyplot.plot(series)


# In[28]:


from statsmodels.tsa.seasonal import seasonal_decompose
from random import randrange


# ### Plotting Time Series of Tweets Posted

# In[29]:


series = [i+randrange(10) for i in range(1,100)]
result = seasonal_decompose(series, model='additive',freq=2)
result.plot()
pyplot.show()


# In[30]:


df['text'] = df['text'].apply(lambda x: re.sub('rt', ' ',x))


# In[31]:


df['text'] = df['text'].map(lambda x:str(x)[2:])


# In[32]:


df['text'] = df['text'].apply(lambda x:  re.sub(r'\b\w{1,3}\b', '', x))


# In[33]:


df['text'].head(5)


# In[34]:



from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output

dwordcloud= df['text']            


# In[35]:


dwordcloud.head(5)


# In[36]:


def wordcloud_draw(dwordcloud, color = 'black'):
    words = ' '.join(dwordcloud)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                 and not word.startswith('x')
                                and not word.startswith('#')
                                and word != 'rt'
                             
                             
                               
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(155, 135))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
print("Total Words")    
wordcloud_draw(dwordcloud,'white')


# In[37]:


df1['time'] = pd.to_datetime(df1['time'])


# In[38]:


df1['time_date'] = df1.time.apply(lambda x: x.date)


# In[39]:


df1['time_date'].head(10)


# In[40]:


from datetime import datetime
d = datetime.now()


# In[41]:


only_date, only_time = d.date(), d.time()
only_date


# In[42]:


df1['only_date'] = [d.date() for d in df1['time']]


# In[43]:


df1['only_date'].head(10)


# In[44]:


df1.groupby(['only_date']).SA.sum().plot(figsize=(16,4), color='b')


# In[45]:


df3= df1.groupby(['only_date']).SA.sum()


# In[46]:


df3.head()
df3.to_csv('Stock', sep=',')


# ### Creating Stock Price Data Frame to find correlation

# In[47]:


d = {'date': ['11/4/2018','11/5/2018','11/6/2018','11/7/2018','11/8/2018','11/9/2018','11/10/2018','11/11/2018'], 
     'Stock Price': [6391.873333,6436.965,6445.354167,6538.79,6486.251667,6411.280833,6399.033333,6378.268333]}
df4 = pd.DataFrame(data=d)
df4.head()

