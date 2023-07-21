#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[2]:


iris=load_iris()
X, y = iris.data, iris.target


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )


# In[4]:


#Scale the features using StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[5]:


#Create decision tree classifier with ID3 algorithm
clf = DecisionTreeClassifier(random_state=42)


# In[6]:


#Define hyperparameters and their possible values for tuning
param_grid = {
    'criterion':['gini','entropy'],
    'max_depth':[None, 5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4] 
}


# In[8]:


#usr GridSearchCV to find the best hyperparameters
grid_search=GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)


# In[9]:


#get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


# In[10]:


#create the decision tree classifier with the best hyperparameters
best_clf = DecisionTreeClassifier(**best_params,random_state=42)


# In[11]:


#train the classifier on the training data
best_clf.fit(X_train,y_train)


# In[12]:


#make predictions on the test data
y_pred=best_clf.predict(X_test)


# In[13]:


#calculate the accuracy of the model
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:",accuracy)


# In[14]:


target_names = iris.target_names
print("Classification Report:")
print(classification_report(y_test, y_pred,target_names=target_names))


# In[15]:


plt.figure(figsize=(6,4))
sns.countplot(x=y, palette = 'coolwarm')
plt.xticks(ticks=np.unique(y),labels=target_names,rotation=45)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# In[16]:


conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='coolwarm',xticklabels=target_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




