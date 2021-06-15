#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation GRIP Task-2
# 
# ## Prediction Using Unsupervised Machine Learning
# 
# ##### Author - Prabhat Kumar
# 
# ###### Objective - Predict the optimum number of clusters from the given 'Iris' dataset and represent it visually.
# 
# 

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[2]:


df = pd.read_csv("iris.csv",sep=",")
df.head()


# In[3]:


# Checking information
df.info()


# In[4]:


# checking missing values
df.isnull().sum()


# In[5]:


# Counting each species present in the dataset
df.Species.value_counts()


# In[6]:


df.describe()


# # Modelling
# # Rescaling the Data

# In[7]:


import sklearn
from sklearn.preprocessing import StandardScaler


# In[8]:


scaler = StandardScaler()
scaled_column = scaler.fit_transform(df.drop(['Id','Species'],axis=1))
scaled_column.shape


# # Finding Optimal Number of Clusters

# In[9]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[10]:


# The Elbow Curve
x = df.iloc[:,[0, 1, 2, 3]].values
wcss= []
for i in range (1, 11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.show()


# In[11]:


# silhouette analysis
range_n_clusters = [2,3,4,5,6,7,8]

for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(scaled_column)
    
    cluster_labels = kmeans.labels_
    
    silhouette_avg = silhouette_score(scaled_column, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# From both the analysis we can choose the number of clusters as 3.

# # Hierarchical Clustering
# 
# # Reconfirming the number of clusters

# In[12]:


import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[13]:


mergin = linkage(scaled_column, method='complete', metric='euclidean')
dendrogram(mergin)
plt.show()


# Now this dendrogram clearly indicates that there are 3 distinct set of clusters.

# In[14]:


kmeans = KMeans(n_clusters = 3)
y_kmeans = kmeans.fit(scaled_column)
y_kmeans_predict = y_kmeans.predict(scaled_column)
y_kmeans_predict


# In[15]:


centers = kmeans.cluster_centers_


# # Visualising the Clusters

# In[16]:


# Representing the Clusters for Visualisation
plt.figure(figsize=(8,6))
plt.scatter(scaled_column[y_kmeans_predict ==0,0], scaled_column[y_kmeans_predict ==0,1], s=100, marker='*', c='blue', 
label='Iris-setosa')
plt.scatter(scaled_column[y_kmeans_predict ==1,0], scaled_column[y_kmeans_predict ==1,1], s=100, marker='*', c='red',
label = 'Iris-versicolour')    
plt.scatter(scaled_column[y_kmeans_predict ==2,0], scaled_column[y_kmeans_predict ==2,1], s=100, marker='*', c='orange',
label = 'Iris-verginica')  
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, marker=',', c='black', label= 'centroids')
plt.title('Number of Clusters')
plt.legend()
plt.show()


# # CONCLUSION

# The optimal number of clusters for this data is 3.
