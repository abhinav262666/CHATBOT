import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import re
import gensim
import matplotlib.pyplot as plt
from gensim.test.utils import get_tmpfile
import sys


def stem_words(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


if __name__ == '__main__':
    
    # Load description features
    df = pd.read_csv('Data/data1.csv')

    
    df['description'] = df.description.apply(func=make_lower_case)
    df['description'] = df.description.apply(func=remove_stop_words)
    df['description'] = df.description.apply(func=remove_punctuation)
    df['description'] = df.description.apply(func=stem_words)
    
    df['title'] = df.title.apply(func=make_lower_case)
    df['title'] = df.title.apply(func=remove_punctuation)
    df['title'] = df.title.apply(func=stem_words)

    df['full_document'] = df['title'] + ' '   +  df['title'] + ' ' + df['title'] + ' ' + df['description']    
    df.to_csv('Data/data_processed.csv', index = False)
