from logging import error
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import pickle
from textwrap import wrap
import re

#import matplotlib.pyplot as plt
#from skimage import io

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
DOLLAR_RATE = 75

def remove_dollar(text):
    text = re.sub("[^0-9.]", "", text)
    if(len(text) > 10):
        text = "10000000000000000000000"
    return text

class Retrieval_Model():

    def __init__(self, maxprice = None):
        self.dv = Doc2Vec.load("models/doc2vec_model")
        self.tf = pickle.load(open("models/tfidf_model.pkl", "rb"))
        self.svd = pickle.load(open("models/svd_model.pkl", "rb"))
        self.svd_feature_matrix = pickle.load(open("models/lsa_embeddings.pkl", "rb"))
        self.doctovec_feature_matrix = pickle.load(open("models/doctovec_embeddings.pkl", "rb"))
        self.df = df = pd.read_csv("Data/data_processed.csv")
        
        if(maxprice):
            self.df = self.df.dropna(subset=['price'])
            self.df.price = self.df.price.apply(func=remove_dollar)
            self.df.price = self.df.price.astype(np.float64,errors = 'ignore')
            self.df = self.df[self.df['price'] <= (maxprice/DOLLAR_RATE)]
            
        self.hal = sia()



    @staticmethod
    def stem_words(text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text


    @staticmethod
    def make_lower_case(text):
        return text.lower()


    @staticmethod
    def remove_stop_words(text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text


    @staticmethod
    def remove_punctuation(text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        text = " ".join(text)
        return text


    def get_message_sentiment(self, message):
        sentences = re.split('\.|\but',message)
        sentences = [x for x in sentences if x != ""]
        love_message = ""
        hate_message = ""
        for s in sentences:
            sentiment_scores = self.hal.polarity_scores(s)
            if sentiment_scores['neg'] > 0:
                hate_message = hate_message + s
            else:
                love_message = love_message + s
        return love_message, hate_message


    def preprocess_message(self, message):
        message = self.make_lower_case(message)
        message = self.remove_stop_words(message)
        message = self.remove_punctuation(message)
        message = self.stem_words(message)
        return message


    def get_message_tfidf_embedding_vector(self, message):
        message_array = self.tf.transform([message]).toarray()
        message_array = self.svd.transform(message_array)
        message_array = message_array[:,0:25].reshape(1, -1)
        return message_array


    def get_message_doctovec_embedding_vector(self, message):
        message_array = self.dv.infer_vector(doc_words=message.split(" "), epochs=200)
        message_array = message_array.reshape(1, -1)
        return message_array


    @staticmethod
    def get_similarity_scores(message_array, embeddings):
        cosine_sim_matrix = pd.DataFrame(cosine_similarity(X=embeddings,
                                                           Y=message_array,
                                                           dense_output=True))
        cosine_sim_matrix.set_index(embeddings.index, inplace=True)
        cosine_sim_matrix.columns = ["cosine_similarity"]
        return cosine_sim_matrix


    def get_ensemble_similarity_scores(self, message):
        message = self.preprocess_message(message)
        bow_message_array = self.get_message_tfidf_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

        bow_similarity = self.get_similarity_scores(bow_message_array, self.svd_feature_matrix) # for bag of words matching 
        semantic_similarity = self.get_similarity_scores(semantic_message_array, self.doctovec_feature_matrix) # for semantics

        ensemble_similarity = pd.merge(semantic_similarity, bow_similarity, left_index=True, right_index=True)
        ensemble_similarity.columns = ["semantic_similarity", "bow_similarity"]
        # combine bag of words similarity and semantic similarity to find matching item
        ensemble_similarity['ensemble_similarity'] = (ensemble_similarity["semantic_similarity"] + ensemble_similarity["bow_similarity"])/2
        ensemble_similarity.sort_values(by="ensemble_similarity", ascending=False, inplace=True)
        return ensemble_similarity


    def get_dissimilarity_scores(self, message):
        message = self.preprocess_message(message)
        bow_message_array = self.get_message_tfidf_embedding_vector(message)
        semantic_message_array = self.get_message_doctovec_embedding_vector(message)

        dissimilarity = self.get_similarity_scores(bow_message_array, self.svd_feature_matrix)
        dissimilarity.columns = ["dissimilarity"]
        dissimilarity.sort_values(by="dissimilarity", ascending=False, inplace=True)
        return dissimilarity


    def get_similar_items(self, message, n):

        pos_msg, neg_msg = self.get_message_sentiment(message)
        similar_items = self.get_ensemble_similarity_scores(pos_msg)
        dissimilar_items = self.get_dissimilarity_scores(neg_msg)
        dissimilar_items = dissimilar_items.query('dissimilarity > .3')
        # remove items which customer do not want
        similar_items = similar_items.drop(dissimilar_items.index)
        similar_items = similar_items[similar_items.index.isin(set(self.df['title']))]
        similar_items = similar_items.reindex()
        
        return similar_items.head(n)

    def view_recommendations(self, recs):
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15,10))
        ax = axes.ravel()

        for i in range(len(recs)):
            single_title = recs.index.tolist()[i]
            cur_item = self.df.query('title==@single_title')
            name = cur_item.title.values[0]

            if(cur_item.image_url != ""):
                cur_img = cur_item.image_url.values[0]
                image = io.imread(cur_img)
                ax[i].imshow(image)
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])
            ax[i].set_title("\n".join(wrap(name, 20)))
            ax[i].axis('off')

        plt.show()
