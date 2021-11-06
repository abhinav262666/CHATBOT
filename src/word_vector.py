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
from sklearn.decomposition import TruncatedSVD



if __name__ == '__main__':
    # Load description features
    df = pd.read_csv('Data/data_processed.csv')

    #Fit TFIDF 
    #Learn vocabulary and tfidf from all style_ids.
    tf = TfidfVectorizer(analyzer='word', 
                        min_df=10,
                        ngram_range=(1, 2),
                        max_features=5000,
                        stop_words='english')
    tf.fit(df['full_document'])

    #Transform style_id products to document-term matrix.
    tfidf_matrix = tf.transform(df['full_document'])
    pickle.dump(tf, open("models/tfidf_model.pkl", "wb"))

    print (tfidf_matrix.shape)
    # Compress with SVD
    
    svd = TruncatedSVD(n_components=500)
    latent_matrix = svd.fit_transform(tfidf_matrix)
    pickle.dump(svd, open("models/svd_model.pkl", "wb"))

    print (latent_matrix.shape)

    n = 25 #pick components
    #Use elbow and cumulative plot to pick number of components. 
    #Need high ammount of variance explained. 
    doc_labels = df.title
    svd_feature_matrix = pd.DataFrame(latent_matrix[:,0:n] ,index=doc_labels)
    print(svd_feature_matrix.shape)
    svd_feature_matrix.head()

    pickle.dump(svd_feature_matrix, open("models/lsa_embeddings.pkl", "wb"))
    