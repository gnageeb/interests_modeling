from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
ngrams = 2
n_topics = 100
n_top_words = 10
import pandas as pd

def perform_tfidf(documents):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=0, ngram_range=(1, ngrams),stop_words='english',)


    tfidf = tfidf_vectorizer.fit_transform(documents.tweet)
    return tfidf, tfidf_vectorizer

def perform_tf(documents):
    tf_vectorizer = CountVectorizer(max_df=0.5, stop_words='english', min_df=0, ngram_range=(1, ngrams),
                                    max_features=3000)


    tf = tf_vectorizer.fit_transform(documents.tweet)
    return tf, tf_vectorizer

def get_nmf_topics(tfidf, tfidf_vectorizer):
    feature_names = tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
    nmf.fit(tfidf)
    topics_words = []
    for topic_idx, topic in enumerate(nmf.components_):
        topics_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topics_words

def get_lda_topics(tf, tf_vectorizer):
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(tf)
    feature_names = tf_vectorizer.get_feature_names()
    topics_words = []
    for topic_idx, topic in enumerate(lda.components_):
        topics_words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topics_words

data_path = 'data/training.1600000.processed.noemoticon.csv'
#modify nrows to include more data
documents = pd.read_csv(data_path,nrows=200000, error_bad_lines=False, sep=',',encoding='ISO-8859-1', quotechar='"', names= ["sentiment", "ID","date","type","user","tweet"])
# tf, vect = perform_tf(documents)
# topics_words = get_lda_topics(tf, vect)
# [print(topics_words[i]) for i in range(len(topics_words))]

tfidf, vect = perform_tfidf(documents)
topics_words = get_nmf_topics(tfidf, vect)
[print(topics_words[i]) for i in range(len(topics_words))]
