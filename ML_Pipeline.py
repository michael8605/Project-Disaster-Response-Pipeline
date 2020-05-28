#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle


# In[2]:





# In[10]:


# load data from database
engine = create_engine('sqlite:///disaster.db')
df = pd.read_sql_table('messages_disaster', con = engine)


# In[11]:


df.shape


# In[12]:


X = df['message'] 
Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
#X = X.iloc[0:10]
Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
#Y = df[['related', 'request']]
#Y = Y.iloc[0:10]


# In[13]:


# Tokenization function to process text data.
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


# In[14]:


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# In[15]:


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


# Split data into train and test tests.
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 45)


# In[16]:


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[9]:


# Train the model.
pipeline.fit(X_train, y_train)


# In[17]:


# Test the model and print the classification report for each of the 36 categories.
def performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))
        # print raw accuracy score 
        print('Accuracy Score: {}'.format(np.mean(y_test.values == y_pred)))


# In[19]:


# In[12]:


performance(pipeline, X_test, y_test)


# In[20]:


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[13]:


# Hyperparameter grid
parameters = {  
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 
# Create model
cv = GridSearchCV(pipeline, param_grid=parameters)


# In[21]:


# Train the tuned model.
cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[15]:


# Test the tuned model and print the classification reports.
performance(cv, X_test, y_test)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[9]:


# Improve the pipeline.
pipeline2 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('best', TruncatedSVD()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])




pipeline2.get_params()





# Train the adjusted pipeline.
pipeline2.fit(X_train, y_train)



performance(pipeline2, X_test, y_test)


# In[22]:


parameters2 = { #'vect__ngram_range': ((1, 1), (1, 2)), 
              #'vect__max_df': (0.5, 1.0), 
              #'vect__max_features': (None, 5000), 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }
              #'clf__estimator__min_samples_split': [2, 4]# 


# In[23]:


cv2 = GridSearchCV(pipeline2, param_grid=parameters2)
cv2


# In[19]:


cv2.fit(X_train, y_train)


# In[20]:


performance(cv2, X_test, y_test)


# ### 9. Export your model as a pickle file

# In[23]:


# In[24]:


# Save the model as a pickle file.
with open('model.pkl', 'wb') as f:
    pickle.dump(cv2, f)


# In[ ]:




