import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def remove_special_characters(text):
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text.lower()

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stop words, and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(word) 
                      for word in tokens if word.isalnum() and word not in stop_words]
    
    return ' '.join(cleaned_tokens)

def preprocess_textinput(df, text_column):    
    vectorizer = CountVectorizer(max_features=5000)  
    df['cleaned_text'] = df[text_column].apply(remove_special_characters)    
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    return X

def preprocess_category(df, column):
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes
    return df

def preprocess_onehot(df, column):
    df = pd.get_dummies(df, columns=[column])
    return df