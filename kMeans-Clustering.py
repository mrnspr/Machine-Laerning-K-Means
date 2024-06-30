#!/usr/bin/env python
# coding: utf-8

# 
# # K Means Clustering with Python
# 
# 
# 
# ## Method Used
# 
# K Means Clustering is an unsupervised learning algorithm that tries to cluster data based on their similarity. Unsupervised learning means that there is no outcome to be predicted, and the algorithm just tries to find patterns in the data. In k means clustering, we have the specify the number of clusters we want the data to be grouped into. The algorithm randomly assigns each observation to a cluster, and finds the centroid of each cluster. Then, the algorithm iterates through two steps:
# Reassign data points to the cluster whose centroid is closest. Calculate new centroid of each cluster. These two steps are repeated till the within cluster variation cannot be reduced any further. The within cluster variation is calculated as the sum of the euclidean distance between the data points and their respective cluster centroids.

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Create some Data

# In[22]:


from sklearn.datasets import make_blobs


# In[23]:


# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)


# In[24]:


data


# In[25]:


type(data)


# In[26]:


len(data)


# In[27]:


data[0]


# In[28]:


data[1]


# In[29]:


data[0][1]


# In[30]:


data[0][1,0]


# In[31]:


data[0][1,1]


# In[32]:


data[0][:,0]


# In[33]:


data[0][:,1]


# ## Visualize Data

# In[34]:


sns.scatterplot(x=data[0][:,0],y=data[0][:,1],hue=data[1],palette='tab10')


# In[35]:


sns.scatterplot(x=data[0][:,0],y=data[0][:,1],hue=data[1],palette='Paired')


# In[36]:


sns.scatterplot(x=data[0][:,0],y=data[0][:,1],hue=data[1],palette='rainbow')


# In[37]:


plt.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='rainbow')


# ## Creating the Clusters

# In[38]:


from sklearn.cluster import KMeans


# In[39]:


kmeans = KMeans(n_clusters=4)


# In[40]:


kmeans.fit(data[0])


# In[41]:


kmeans.cluster_centers_


# In[42]:


kmeans.labels_


# In[43]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(x=data[0][:,0],y=data[0][:,1],c=kmeans.labels_,cmap='tab10')
ax2.set_title("Original")
ax2.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='tab10')


# In[44]:


kmeans = KMeans(n_clusters=3)


# In[45]:


kmeans.fit(data[0])


# In[46]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(x=data[0][:,0],y=data[0][:,1],c=kmeans.labels_,cmap='tab10')
ax2.set_title("Original")
ax2.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='tab10')

