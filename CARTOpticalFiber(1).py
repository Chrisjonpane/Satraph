#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

OpticalFiber_data = pd.read_csv('OpticalFiber.csv')
OpticalFiber_data = OpticalFiber_data.sample(frac=1).reset_index(drop=True)
OpticalFiber_data.head()


# In[2]:


X = OpticalFiber_data.drop('OUTPUT DEMAND', axis=1)
Y = OpticalFiber_data['OUTPUT DEMAND']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[9]:


def summarize_classification(y_test, y_pred):
    
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)
    
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
   #print ("Test data count:  ", (y_test))
   #print("accuracy_count  :  ", num_acc)
    print("accuracy_score  :  ", acc)
    print("precision_score :  ", prec)
    print("recall_score    :  ", recall)
    print()
    
    


# In[4]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth' : [2, 3, 4, 5, 6, 7, 8, 9, 10]}

grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

grid_search.best_params_


# In[5]:


for i in range(9) :
    print('Parameters: ', grid_search.cv_results_['params'][i])
    
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])
    


# In[6]:


decision_tree_model = DecisionTreeClassifier(     max_depth = grid_search.best_params_['max_depth']).fit(x_train, y_train)


# In[7]:


y_pred = decision_tree_model.predict(x_test)


# In[10]:


summarize_classification(y_test, y_pred)


# In[13]:


feature_importance = decision_tree_model.tree_.compute_feature_importances(normalize=False)
print("feature_importance = " + str(feature_importance))


# In[15]:


pip install graphviz


# In[90]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure("Decision Tree", figsize=[14, 7])
plot_tree(decision_tree_model, fontsize=10, filled=True, precision = 2)
plt.tight_layout()

plt.show()


# In[ ]:




