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

    #Use reviews, descriptions, and notes for vocabulary 
    titles = df.title.values.tolist()
    descriptions = df.description.values.tolist()
    #notes = df.notes.values.tolist() #not using notes because sematics and order of list is not meaningfull. 

    documents = []
    for i in range(len(df)):
        mystr = str(titles[i])
        mystr = mystr + str(descriptions[i])
        documents.append(re.sub("[^\w]", " ",  mystr).split())
    

    formatted_documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]

    model = gensim.models.doc2vec.Doc2Vec(vector_size=25, min_count=5, epochs=20, seed=0, window=3, dm=1)
    model.build_vocab(formatted_documents)
    model.train(formatted_documents, total_examples=model.corpus_count, epochs=model.epochs)
    fname = get_tmpfile("models/doc2vec_model")
    model.save("models/doc2vec_model")
    model = gensim.models.doc2vec.Doc2Vec.load("./models/doc2vec_model")
    vector = model.infer_vector(doc_words=["this", "is", "a", "test"], epochs=50)
    print(vector)
    doctovec_feature_matrix = pd.DataFrame(model.docvecs.vectors_docs, index=df.title)
    pickle.dump(doctovec_feature_matrix, open("models/doctovec_embeddings.pkl", "wb"))