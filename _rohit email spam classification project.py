#!/usr/bin/env python
# coding: utf-8

# # Email Spam Classification

# ### Importing Libraries

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ### Importing Dataset

# In[38]:


df = pd.read_csv('C:\\Users\\05roh\\OneDrive\\Desktop\\spam (1) (4).csv')   


# In[39]:


df.head()


# ### 'Ham' means Genuine email

# ### 'Spam' means spam

# In[40]:


df.shape


# In[41]:


df.dtypes


# In[42]:


df.describe()


# In[43]:


df.Category.unique()


# ### Removing Null Values

# In[44]:


df.isna().sum()


# In[45]:


df.dropna(inplace=True)


# In[46]:


df.isna().sum()


# ### Removing Duplicate Values

# In[47]:


df.duplicated().sum()


# In[48]:


df=df.drop_duplicates()


# In[49]:


df.duplicated().sum()


# In[50]:


df.shape


# In[51]:


df.groupby('Category').size()


# In[52]:


(df['Category'].value_counts()/len(df['Category']))*100


# In[53]:


plt.figure(figsize=(4,3))
sns.countplot(x=df.Category)
plt.title('Ham vs Spams')
plt.show()


# ### Encoding target variable

# In[54]:


df.loc[df['Category'] == 'spam' , 'Category'] = 0
df.loc[df['Category'] == 'ham' , 'Category'] = 1


# In[55]:


X = df['Message']
Y = df['Category']


# In[56]:


print(X)


# In[57]:


print(Y)


# In[58]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=10)


# In[59]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[60]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# ### Transform the test data to feature vectors so that it can be used in ML models

# In[61]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
# stop words are the words which dont have meaning and can be removed without sacrificing the meaning of the sentence

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# In[62]:


Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[63]:


print(X_train)


# In[64]:


print(X_train_features)


# ## Model Building

# ### Logistic Regression 

# In[65]:


lr = LogisticRegression()


# In[66]:


lr.fit(X_train_features, Y_train)  #training the model


# In[67]:


lr_prediction = lr.predict(X_test_features)
lr_accuracy = accuracy_score(Y_test, lr_prediction)


# In[68]:


print('Logistic Regression Accuracy: ', lr_accuracy)


# In[69]:


print(classification_report(Y_test,lr_prediction))


# ### Decision Tree 

# In[70]:


dtc = DecisionTreeClassifier()


# In[71]:


dtc.fit(X_train_features, Y_train)  #training the model


# In[72]:


dtc_prediction = dtc.predict(X_test_features)
dtc_accuracy = accuracy_score(Y_test, dtc_prediction)


# In[73]:


print('Decision Tree Accuracy: ', dtc_accuracy)


# In[74]:


print(classification_report(Y_test,dtc_prediction))


# ### Random Forest 

# In[75]:


rf= RandomForestClassifier(n_estimators=200)


# In[76]:


rf.fit(X_train_features, Y_train)  #training the model


# In[77]:


rf_prediction = rf.predict(X_test_features)
rf_accuracy = accuracy_score(Y_test, rf_prediction)


# In[78]:


print('Random Forest Accuracy: ', rf_accuracy)


# In[79]:


print(classification_report(Y_test,rf_prediction))


# In[80]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [lr_accuracy, dtc_accuracy, rf_accuracy]

plt.figure(figsize=(10, 3))
sns.barplot(x=accuracies, y=classifiers, width=0.5, orient='h')  # Set orient to 'h' for horizontal
plt.title('Best Accuracy Selection')
plt.xlabel('Accuracy')
plt.xlim(0.95, 0.98)  # Set x-axis limits for accuracy values
plt.show()


# In[ ]:





# In[ ]:




