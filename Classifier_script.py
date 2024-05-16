# Libraries for data loading and data ccleaning
import numpy as np
import pandas as pd

# Libraries for modelling
#Models from Scikit-Learn
from xgboost.sklearn import XGBClassifier

# Model Evaluations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# from the nltk library.
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import string
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
from sklearn.utils import resample

# Other neeeded libraries
import re
import pickle

# loading the data. Using an excel file here.
# Change the file name to that of your file
file = pd.ExcelFile('excel_file_name.xlsx')
data = file.parse('sheet_name')

# Keeping the username column and the comments column
usernames = data['username']
data['comments_to_use'] = data['comments']

# Converting all entries of the comment column to lower case
data['comments'] = data['comments'].str.lower()
data['comments'] = data['comments'].astype(str)

# Removing punctuations from the comments
string.punctuation

# Function that removes punctuations from texts
def clean_punct(text):
    """
    The function clean_punction: It takes in a text as input and loops through
    the text to select only characters that are not in string.punctuation.
    
    """
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

# Removing all website urls and replacing them with the text 'web-url'
pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
replace_url = r'web-url'
data['comments'] = data['comments'].replace(to_replace = pattern_url, value = replace_url, regex = True)

# Applying the clean_punct function to the comments column
data['comments'] = data['comments'].apply(lambda x: clean_punct(x))

# Tokenizing the comments
tokeniser = TreebankWordTokenizer()
data['comments'] = data['comments'].apply(tokeniser.tokenize)

# Applying Lammetization
lemmatizer = WordNetLemmatizer()

# Defining a function that will handle the lematization process
def lemma(words, lemmatizer):
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

# Applying the lemma function
data['comments'] = data['comments'].apply(lemma, args=(lemmatizer, ))

# Vectorizing the data using countVectorizer
# The pickle file for our vectorizer will be used here instead of creating a new model.
vect = pickle.load(open("vectorizer.pkl", "rb"))
data_vect = vect.transform(data['comments'].values.astype(str))

# Converting the vectorized data to array
x_vect = data_vect.toarray()

# Initializing our models for fitting. The pickle file for our model will be used here instead of creating a new model.
xgb_classifier = pickle.load(open('xgb_model.pkl', 'rb'))

# Generating predictions
prediction = xgb_classifier.predict(x_vect)

# Converting the result to a column of our test dataset
data['label'] = prediction

# Converting the label column back to text
data['label'] = data['label'].replace({0:'Other', 1:'Veterinarian', 2:'Medical Doctor'})

# Dropping the duplicate column created for comments
data = data.drop(['comments'], axis=1)

# Bringing back the username column
data['username'] = usernames

data = data[['username', 'comments_to_use', 'label']]

# Converting our result to excel
# You can change the file name and sheet name to names of your choice.
data.to_excel('classifier_testing.xlsx', 'predictions', index=False)


