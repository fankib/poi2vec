#!/usr/bin/env python
# coding: utf-8

# In[146]:


import os
import tqdm
import numpy as np

route_files = []
lr_files = []
prob_files = []

for (path, dir, files) in os.walk("./npy/splited_file/"):
    for filename in files:
        if 'id2route' in filename:
            route_files.append('./npy/splited_file/'+filename)
        if 'id2lr' in filename:
            lr_files.append('./npy/splited_file/'+filename)
        if 'id2prob' in filename:
            prob_files.append('./npy/splited_file/'+filename)


# In[147]:


route_files.sort()
lr_files.sort()
prob_files.sort()


# In[149]:


pad = [0]*17
id2route = [[pad[1:], pad[1:], pad[1:], pad[1:]]]
id2route_cnt = [0]
max_route_cnt = 4
max_node_idx = 0

for filename in tqdm.tqdm(route_files):
    routes_list = np.load(filename, allow_pickle=True)
    for routes in routes_list:
        id2route_cnt.append(len(routes))
        
        batch_max = np.max([node for route in routes
                                for node in route[1:]])
        if batch_max > max_node_idx:
            max_node_idx = batch_max
            
        if len(routes) < max_route_cnt:
            for _ in range(max_route_cnt - len(routes)):
                routes.append(pad)
        
        routes = np.asarray([l[1:] for l in routes])
        id2route.append(routes)


# In[150]:


print (np.asarray(id2route).shape)
print (np.asarray(id2route_cnt).shape)
print (max_node_idx)


# In[151]:


pad = [0]*16
id2lr = [[pad, pad, pad, pad]]

for filename in tqdm.tqdm(lr_files):
    lrs_list = np.load(filename, allow_pickle=True)
    for lrs in lrs_list:
        if len(lrs) < max_route_cnt:
            for _ in range(max_route_cnt - len(lrs)):
                lrs.append(pad)
        id2lr.append(lrs)


# In[152]:


print (np.asarray(id2lr).shape)


# In[153]:


pad = 0
id2prob = [[0,0,0,0]]

for filename in tqdm.tqdm(prob_files):
    probs_list = np.load(filename, allow_pickle=True)
    for probs in probs_list:
        probs = list(probs)
        if len(probs) < max_route_cnt:
            for _ in range(max_route_cnt - len(probs)):
                probs.append(pad)
        id2prob.append(probs)       


# In[154]:


print (np.asarray(id2prob).shape)


# In[155]:


np.save('./npy/id2route.npy', np.asarray(id2route))
np.save('./npy/id2route_cnt.npy', np.asarray(id2route_cnt))
np.save('./npy/id2lr.npy', np.asarray(id2lr))
np.save('./npy/id2prob.npy', np.asarray(id2prob))


# In[158]:


print (id2route[1])
print (id2prob[1])

