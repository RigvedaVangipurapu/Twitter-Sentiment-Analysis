
# coding: utf-8

# In[1]:


import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups


# In[2]:


consumer_key = 'Sri1ChSCMuYs16x48Z7YbgDJX'
consumer_secret = 'f4gCuO4VqhtFITDUVvgKW7HDribREgpGBUl4JgRWsaH3XjSfuP'
access_token = '1128352344595283968-7gmltStBsYtpHV19Cgs3dftZLf9w1s'
access_token_secret = 'zZNO9vkT1h6yrOKQ9Ur2oZwk9y8T2sQrPhDKsqDyz3zuM'


# In[3]:


auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)


# In[4]:


api=tweepy.API(auth)


# In[5]:



twenty_train = fetch_20newsgroups(subset='train', shuffle=True,categories=['comp.graphics','comp.windows.x','sci.space','rec.sport.baseball','sci.electronics','talk.politics.misc','talk.religion.misc'])


# In[6]:


twenty_train.target_names


# In[ ]:


'''print (twenty_train.data[0],"\n\n\n\n")
print(type(twenty_train.data[0]))
print(":::::::::::::")
print("\n".join(twenty_train.data[0].split("\n")[:3]))'''


# In[8]:


# Extracting features from text files
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# In[9]:


# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[10]:


# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


# In[11]:


number_of_tweets=500
tweets=api.user_timeline(screen_name='elonmusk')
temp1=[]
p=2
tweets_for_csv=[tweet.text for tweet in tweets]
for j in tweets_for_csv:
    temp1.append(j)
    print ('-',j,'-')
print (len(temp1))
print(temp1[p])


# In[12]:


# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


# In[13]:


temp=[]
for j in temp1:
    new=j.split()
    j=[]
    for x in new:
        if '@' not in x and 'http' not in x and 'www.' not in x:
            j.append(x)


    j=' '.join(j)
    temp.append(j)
tweets_for_csv=temp


# In[14]:


# Performance of NB Classifier

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
#print(twenty_train.target_names[text_clf.predict("Php sucks , python rules")])
predicted = text_clf.predict(tweets_for_csv)
print(type(predicted))
p=0
dict={'comp.graphics':"Technology",'comp.windows.x':"Technology",'sci.space':'Science','rec.sport.baseball':'General','sci.electronics':'Science','talk.politics.misc':'Politics','talk.religion.misc':'Religion'}
#for p in range(20):
 #   print(predicted[p],tweets_for_csv[p],'------->',dict[twenty_train.target_names[predicted[p]]],"\n------\n")
    
#np.mean(predicted == twenty_test.target)



# In[15]:


def getSenti(polarity):
    if polarity >0.1:
        return "Positive"
    elif polarity <-0.1 :
        return "Negative"
    else :
        return "Neutral"


# In[16]:



df = pd.DataFrame(tweets_for_csv, columns=['tweets'])

df['Predicted'] = np.array([dict[twenty_train.target_names[predicted[p]]] for p in range (len(tweets_for_csv))])

df['Sentiment']=np.array([getSenti(TextBlob(j).sentiment.polarity) for j in tweets_for_csv])
df['Polarity ']=np.array([(TextBlob(j).sentiment.polarity) for j in tweets_for_csv])
df

