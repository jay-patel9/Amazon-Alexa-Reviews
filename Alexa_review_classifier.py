# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 23:26:56 2018

@author: JAY PATEL
"""

#Model Accuracy 0.9428
# Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('amazon_alexa.tsv', delimiter = '\t', quoting = 3) # quoting for removing ""

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # Remove words like (the, is, that)
from nltk.stem.porter import PorterStemmer # Stemming Words (loves == love)

# List of Words
data = [] 
for i in range(0, 3150):
    # Using regular expression to display alphabets only
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i])
    # Convert to Lower Case
    review = review.lower()
    # Create List of Words for Stemming
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Combine list to from Word
    review = ' '.join(review)
    data.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(data).toarray()
Y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train, y_train)

## Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Model Accuracy (26+568)/630 = 0.942

# Save your Model
from sklearn.externals import joblib
joblib.dump(classifier,'model_joblib')

