#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# ## Task 1

# In[2]:


xlsx = pd.ExcelFile('Analyst_Excel_Test.xlsx')
population = pd.read_excel(xlsx, 'CA population data')


# In[3]:


population.head()


# In[4]:


from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[5]:


X_train = population.loc[:,['Population']]


# In[6]:


km = KMeans(n_clusters=3)


# In[7]:


cluster_assignments = km.fit_predict(X_train)
cluster_assignments[:10]


# In[8]:


fig = plt.figure(figsize=(10,10))
for i in range(3):
    X_subset = X_train[cluster_assignments == i]
    plt.scatter(pd.Series(np.array(['0'] * len(X_subset))), X_subset.iloc[:,0],s = 80,alpha = 0.8, label = 'cluster '+str(i))
    plt.plot(km.cluster_centers_[i][0],marker='x',c='k', ms=5, mew=5, label=None);
_ = plt.legend();
_ = plt.ylabel('Population')
_ = plt.title('Clustering', fontdict = {'fontsize':30})


# In[9]:


population['City Type'] = cluster_assignments
population.loc[population.Population > 600000, 'City Type'] = 1
population.loc[population['City Type'] == 1, 'City Type'] = 'big'
population.loc[population['City Type'] == 0, 'City Type'] = 'small'
population.loc[population['City Type'] == 2, 'City Type'] = 'medium'


# In[10]:


len(population.loc[population['City Type'] == 'small'])


# In[11]:


len(population.loc[population['City Type'] == 'medium'])


# In[12]:


len(population.loc[population['City Type'] == 'big'])


# In[13]:


population.head()


# In[14]:


len(population)


# In[15]:


population.to_excel(r'Population.xlsx', index = None, header=True)


# In[16]:


# Name has blank in the tail


# ### Kmeans is sensitive to outliers. So we cut off the largest cities.

# In[17]:


extra_big = population.loc[population.Population > 600000]
extra_big


# In[18]:


population_cut = population.loc[population.Population < 600000].reset_index(drop = True)[['Name ', 'Population']]
X_train = population.loc[population.Population < 600000].reset_index(drop = True)[['Population']]


# In[19]:


cluster_assignments = km.fit_predict(X_train)
cluster_assignments[:10]


# In[20]:


fig = plt.figure(figsize=(10,10))
for i in range(3):
    X_subset = X_train[cluster_assignments == i]
    plt.scatter(pd.Series(np.array(['0'] * len(X_subset))), X_subset.iloc[:,0],s = 80,alpha = 0.8, label = 'cluster '+str(i))
    plt.plot(km.cluster_centers_[i][0],marker='x',c='k', ms=5, mew=5, label=None);
_ = plt.legend();
_ = plt.ylabel('Population')
_ = plt.title('Clustering', fontdict = {'fontsize':30})


# In[21]:


population_cut['City Type'] = cluster_assignments
population_cut.loc[population_cut['City Type'] == 1, 'City Type'] = 'medium'
population_cut.loc[population_cut['City Type'] == 0, 'City Type'] = 'small'
population_cut.loc[population_cut['City Type'] == 2, 'City Type'] = 'big'


# In[22]:


len(population_cut.loc[population_cut['City Type'] == 'small'])


# In[23]:


len(population_cut.loc[population_cut['City Type'] == 'medium'])


# In[24]:


len(population_cut.loc[population_cut['City Type'] == 'big'])


# In[25]:


extra_big


# In[26]:


# keep the order as original form
line1 = pd.DataFrame(extra_big.iloc[0]).T
line2 = pd.DataFrame(extra_big.iloc[1]).T
line3 = pd.DataFrame(extra_big.iloc[2]).T
line4 = pd.DataFrame(extra_big.iloc[3]).T
line5 = pd.DataFrame(extra_big.iloc[4]).T
population_final = pd.concat([population_cut.iloc[:392], line1, population_cut.iloc[392:753], line2,                           population_cut.iloc[753:1134], line3, population_cut.iloc[1134:1136], line4,                           population_cut.iloc[1136:1142], line5, population_cut.iloc[1142:]]).reset_index(drop=True)


# In[27]:


len(population_final.loc[population_final['City Type'] == 'small'])


# In[28]:


len(population_final.loc[population_final['City Type'] == 'medium'])


# In[29]:


len(population_final.loc[population_final['City Type'] == 'big'])


# In[30]:


population_final.groupby(['City Type'])['Population'].max()


# In[31]:


population_final.groupby(['City Type'])['Population'].min()


# In[32]:


population_final.to_excel(r'Population_final.xlsx', index = None, header=True)


# ## Task 2

# In[33]:


info = pd.read_excel(xlsx, 'RAW real estate data')
info.head()


# In[34]:


info.columns


# In[35]:


len(info)


# In[36]:


population_final.head()


# In[37]:


population_final.columns


# In[38]:


# delete ,CA and uppercase
for i in range(len(population_final)):
    if 'CA' in population_final.at[i, 'Name '].split(',')[-1]:
        population_final.at[i, 'Name '] = ''.join(population_final.at[i, 'Name '].split(',')[:-1])
    population_final.at[i, 'Name '] = population_final.at[i, 'Name '].upper()
population_final.head()


# In[39]:


new_info = pd.merge(info, population_final[['Name ', 'City Type']], left_on = 'city', right_on = 'Name ', how = 'left')
new_info.head()


# In[40]:


# check if there is null
new_info.isnull().sum()


# In[41]:


new_info.loc[new_info['Name '].isnull()]


# In[42]:


# delete the null rows and drop duplicate columns
new_info = new_info.loc[new_info['Name '].notnull()].reset_index(drop = True)
len(new_info)


# In[43]:


new_info.head()


# In[44]:


new_info = new_info.drop(['Name '], axis = 1)
new_info.head()


# In[45]:


new_info.to_excel(r'Real Estate Data with City Type.xlsx', index = None, header=True)


# In[46]:


# assumption: one bed in one bedroom
cal_metric = new_info[['City Type', 'type', 'beds', 'sq__ft', 'price']].groupby(['City Type', 'type']).mean()
cal_metric


# In[137]:


cal_metric.to_excel(r'Average Data.xlsx', index = True, header=True)


# In[48]:


len(new_info.loc[new_info['type'] == 'Unkown'])


# In[ ]:




