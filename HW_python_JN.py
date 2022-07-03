#!/usr/bin/env python
# coding: utf-8

# # HW Assignment - Jupyter Notebook
# Plotting using Python (linear regression)

# # Load Dependencies

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

# # Import Data

# In[2]:


#Importing the dataset

dataset = pd.read_csv(sys.argv[1])

# # Scatter Plot

# In[3]:


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('Y vs X')
plt.xlabel('X')
plt.ylabel('Y')



# # Fit Data

# In[4]:


# Fitting Linear Regression to the Dataset

model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# # Get R-squared Values

# In[5]:


#Adjusted R-squared
model.score(dataset[['x']], dataset[['y']])


# # Linear Regression

# In[6]:


#Visualizing the Linear Regression results
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('Y vs X')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Linear Model')
plt.clf()
