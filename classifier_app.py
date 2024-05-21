# Streamlit dependencies
import streamlit as st
from streamlit_option_menu import option_menu

# Data handling dependencies
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize, TreebankWordTokenizer

img_1 = Image.open('image1.PNG')
img_2 = Image.open('image2.PNG')
img_3 = Image.open('image3.PNG')

img1 = img_1.resize((1000,300))
img2 = img_2.resize((1150,300))
img3 = img_3.resize((1800,700))

col1, col2, col3 = st.columns(3)
with col1:
    st.image(img1, caption=None, use_column_width=True)
with col2:
    st.image(img3, caption=None, use_column_width=True)
with col3:
    st.image(img2, caption=None, use_column_width=True)


# st.image('image1.PNG',use_column_width=True)
st.title('REDDIT COMMENTS CLASSIFIER')

# Data loading
no_link = pd.DataFrame({'username':['demo user1','demo user2','demo user3'], 'comments':['sample, invalid comment', 'sample, invalid comment', 'sample, invalid comment']})
no_link['username'] = no_link['username'].astype(str)
no_link['comments'] = no_link['comments'].astype(str)

url = st.text_input('Input google sheet link for your data and press the Enter button of your keyboard to apply the link')
if url == '':
    df = no_link
else:
    link = url.split('/')[0]+'/'+url.split('/')[1]+'/'+url.split('/')[2]+'/'+url.split('/')[3]+'/'+url.split('/')[4]+'/'+url.split('/')[5]
    df = pd.read_csv(link+'/export?format=csv')
    df = df[['username', 'comments']]

# Duplicating the comments column
df['comments_to_use'] = df['comments'].copy()

# Converting all entries of the comment column to lower case
df['comments_to_use'] = df['comments_to_use'].str.lower()
df['comments_to_use'] = df['comments_to_use'].astype(str)

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
df['comments_to_use'] = df['comments_to_use'].replace(to_replace = pattern_url, value = replace_url, regex = True)

# Applying the clean_punct function to the comments column
df['comments_to_use'] = df['comments_to_use'].apply(lambda x: clean_punct(x))

# Tokenizing the comments
tokeniser = TreebankWordTokenizer()
df['comments_to_use'] = df['comments_to_use'].apply(tokeniser.tokenize)

# Applying Lammetization
lemmatizer = WordNetLemmatizer()

# Defining a function that will handle the lematization process
def lemma(words, lemmatizer):
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

# Applying the lemma function
df['comments_to_use'] = df['comments_to_use'].apply(lemma, args=(lemmatizer, ))

# Vectorizing the data using countVectorizer
# The pickle file for our vectorizer will be used here instead of creating a new model.
vect = pickle.load(open("vectorizer.pkl", "rb"))
data_vect = vect.transform(df['comments_to_use'].values.astype(str))

# Converting the vectorized data to array
x_vect = data_vect.toarray()

# Initializing our models for fitting. The pickle file for our model will be used here instead of creating a new model.
xgb_classifier = pickle.load(open('xgb_model.pkl', 'rb'))

# Generating predictions
prediction = xgb_classifier.predict(x_vect)

# Converting the result to a column of our test dataset
df['label'] = prediction

# Converting the label column back to text
df['label'] = df['label'].replace({0:'Other', 1:'Veterinarian', 2:'Medical Doctor'})

# Dropping the duplicate column created for comments
df = df.drop(['comments_to_use'], axis=1)

df = df[['username', 'comments', 'label']]


st.write('View predictions and scroll down to download result')
st.write(df)

st.write(f"Download predictions")
csv = df.to_csv(index=False)
st.download_button('Download', csv, 'labels prediction.csv', key = 'download-csv')
if 'labels prediction.csv' in os.listdir():
    os.remove('labels prediction.csv')
