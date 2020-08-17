
# coding: utf-8

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[3]:


from sklearn.model_selection import train_test_split


# In[15]:


data1 = np.genfromtxt('xf.csv', delimiter=',')
data2 = np.genfromtxt('yf.csv', delimiter=',')
print(len(data1))


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(data1,data2, test_size=0.2)


# In[17]:


model.fit(X_train, y_train)


# In[18]:


model.score(X_test, y_test)


# In[58]:


from matplotlib.image import imread

img = imread('F-121.png')

a = np.array(img)

b = a.ravel()
b=b.reshape(1, -1)





print("Character found is "+chr(int(model.predict(b)+65)))

