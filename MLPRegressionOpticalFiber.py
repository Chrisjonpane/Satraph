#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
OpticalFiber_data = pd.read_csv('OpticalFiber.csv')
OpticalFiber_data = OpticalFiber_data.sample(frac=1).reset_index(drop=True)
OpticalFiber_data.head(10)


# In[2]:


OpticalFiber_data.shape


# In[3]:


OpticalFiber_data[OpticalFiber_data.isnull().any(axis=1)].count()


# In[4]:


OpticalFiber_data.describe()


# In[5]:


OpticalFiber_data_corr=OpticalFiber_data.corr()
OpticalFiber_data_corr


# In[6]:


fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(OpticalFiber_data_corr, annot=True)


# In[7]:


from sklearn.model_selection import train_test_split
X = OpticalFiber_data.drop('OUTPUT DEMAND', axis=1)
Y = OpticalFiber_data['OUTPUT DEMAND']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.20)


# In[8]:


x_train.sample(5)


# In[9]:


#from sklearn.preprocessing import StandardScaler


# In[10]:


#scaler = StandardScaler()
#scaler.fit(x_train)

#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)


# In[11]:


from sklearn.neural_network import MLPClassifier


# In[12]:


mlp_clf = MLPClassifier(hidden_layer_sizes=(10,) ,
                       max_iter=1000,
                       activation = 'relu',
                       alpha=0.0001,
                       solver = 'lbfgs',
                       verbose=True,
                       )


# In[13]:


mlp_clf.fit(x_train, y_train)


# In[14]:


y_pred = mlp_clf.predict(x_test)


# In[15]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[16]:


#mlp_reg.score(x_train, y_train)


# In[17]:


#r2_score(y_test, y_pred )


# In[18]:


#plt.plot(y_pred, label='Predicted')
#plt.plot(y_test.values, label='Actual')

#plt.ylabel('change in weight')

#plt.legend()
#plt.show()


# In[22]:


pred_results = pd.DataFrame({'y_test':y_test,
                             'y_pred':y_pred})
pred_results.sample(10)


# In[23]:


OpticalFiber_data_crosstab = pd.crosstab(pred_results.y_test, pred_results.y_pred)
OpticalFiber_data_crosstab


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix


# In[26]:


print(confusion_matrix(y_test,y_pred))


# In[27]:


print(classification_report(y_test,y_pred))


# In[39]:


mlp_clf.predict([[0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,2,1,0]])


# In[45]:


import pickle

# Save the trained model as a pickle string.
saved_model = pickle.dumps(mlp_clf)
  
# Load the pickled model
mlp_from_pickle = pickle.loads(saved_model)
  
# Use the loaded pickled model to make predictions
mlp_from_pickle.predict([[0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,1,0,0]])


# In[ ]:




