#!/usr/bin/env python
# coding: utf-8

# In[181]:


import numpy as np
import pandas as pd
from datetime import datetime
import tqdm
import config

checkin_file = config.dataset
df = pd.read_csv(checkin_file, sep='\t', header=None)
df.columns = ["user", "time", "latitude", "longitude", "poi"]
df = df[['user', 'time', 'poi']]


# In[182]:


len(df)


# In[183]:


prev_cnt = 0
curr_cnt = len(df)
while prev_cnt != curr_cnt:
    prev_cnt = curr_cnt
    df = df[df.groupby('user').user.transform(len) > 5]
    df = df[df.groupby('poi').poi.transform(len) > 5]
    curr_cnt = len(df)


# In[184]:


len(df)


# In[186]:


poi2id = np.load("./npy/poi2id.npy", allow_pickle=True).item()
df['poi'] = df['poi'].apply(lambda x: poi2id[x] if poi2id.get(x) != None else 0)
df = df[df['poi'] != 0]
df['time'] = df['time'].apply(lambda x: (datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")-datetime(2009,1,1)).total_seconds()                              /360)  # hour


# In[187]:


np.max(df.poi)


# In[188]:


user2id = {'unk':0}
id2user = [0]
for target_idx in tqdm.tqdm(range(len(df))):
    (user, time, poi) = df.iloc[target_idx]
    if user2id.get(user) == None:
        user2id[user] = len(id2user)
        id2user.append(user)
    if len(id2user) == 4627:
        break
df['user'] = df['user'].apply(lambda x: user2id[x] if user2id.get(x) != None else 0)
df = df[df['user'] != 0]


# In[189]:


len(df.groupby('user')),len(df)


# In[190]:


train_user = []
train_context = []
train_target = []
valid_user = []
valid_context = []
valid_target = []
test_user = []
test_context = []
test_target = []

tow = 6
prev_user = df['user'][0]
user_user = []
user_context = []
user_target = []
for target_idx in tqdm.tqdm(range(len(df))):
    (user, time, poi) = df.iloc[target_idx]
    if prev_user != user:
        prev_user = user
        if len(user_user) > 20:
            train_thr = int(len(user_user)*0.9)
            valid_thr = int(len(user_user)*0.95)
            train_user += user_user[:train_thr]
            train_context += user_context[:train_thr]
            train_target += user_target[:train_thr]
            valid_user += user_user[train_thr:valid_thr]
            valid_context += user_context[train_thr:valid_thr]
            valid_target += user_target[train_thr:valid_thr]
            test_user += user_user[valid_thr:]
            test_context += user_context[valid_thr:]
            test_target += user_target[valid_thr:]
        elif len(user_user) > 0:
            train_user += user_user
            train_context += user_context
            train_target += user_target    
        user_user = []
        user_context = []
        user_target = []
        #print train_user, train_context, train_target
    
    context = []
    for context_idx in range(target_idx+1, len(df)):
        (c_user, c_time, c_poi) = df.iloc[context_idx]
        if user == c_user and (time-tow) < c_time:
                context.append(c_poi)
        else:
            break
    if context:
        user_user.append(user)
        user_context.append(context)
        user_target.append(poi)
        
if len(user_user) > 20:
    train_thr = int(len(user_user)*0.8)
    valid_thr = int(len(user_user)*1.0)
    train_user += user_user[:train_thr]
    train_context += user_context[:train_thr]
    train_target += user_target[:train_thr]
    valid_user += user_user[train_thr:valid_thr]
    valid_context += user_context[train_thr:valid_thr]
    valid_target += user_target[train_thr:valid_thr]
    test_user += user_user[valid_thr:]
    test_context += user_context[valid_thr:]
    test_target += user_target[valid_thr:]
elif len(user_user) > 0:
    train_user += user_user
    train_context += user_context
    train_target += user_target


# In[191]:


len(train_user), len(train_context), len(train_target), len(valid_user), len(valid_context), len(valid_target), len(test_user), len(test_context), len(test_target)


# In[192]:


len_context = []
for i, context in enumerate(train_context):
    len_context.append(len(context))
print (np.max(len_context), np.mean(len_context), np.median(len_context), np.min(len_context))
for i, context in enumerate(valid_context):
    len_context.append(len(context))
for i, context in enumerate(test_context):
    len_context.append(len(context))
len_context.sort()
print (np.max(len_context), np.mean(len_context), np.median(len_context), np.min(len_context))


# In[193]:


print (len(len_context))
print (len_context[int(len(len_context)*0.99)])


# In[194]:


maxlen_context = 32
for i, context in enumerate(train_context):
    if len(context) < maxlen_context:
        train_context[i] += ([0]*(maxlen_context-len(context)))
    elif len(context) > maxlen_context:
        train_context[i] = context[:maxlen_context]
for i, context in enumerate(valid_context):
    if len(context) < maxlen_context:
        valid_context[i] += ([0]*(maxlen_context-len(context)))
    elif len(context) > maxlen_context:
        valid_context[i] = context[:maxlen_context]
for i, context in enumerate(test_context):
    if len(context) < maxlen_context:
        test_context[i] += ([0]*(maxlen_context-len(context)))
    elif len(context) > maxlen_context:
        test_context[i] = context[:maxlen_context]


# In[195]:


len_context = []
for context in test_context:
    len_context.append(len(context))
print (np.max(len_context), np.mean(len_context), np.median(len_context), np.min(len_context))


# In[196]:


np.save('./npy/train_context.npy', train_context)
np.save('./npy/valid_context.npy', valid_context)
np.save('./npy/test_context.npy', test_context)
np.save('./npy/user2id.npy', user2id)
np.save('./npy/id2user.npy', id2user)


# In[197]:


np.save('./npy/train_user.npy', train_user)
np.save('./npy/valid_user.npy', valid_user)
np.save('./npy/test_user.npy', test_user)
np.save('./npy/train_target.npy', train_target)
np.save('./npy/valid_target.npy', valid_target)
np.save('./npy/test_target.npy', test_target)


# In[101]:


len(id2user)

