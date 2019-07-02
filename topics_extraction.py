from utils import *

import pandas as pd


import dill as pickle
#reading data
data_path = 'data/training.1600000.processed.noemoticon.csv'
documents = pd.read_csv(data_path, error_bad_lines=False, sep=',',encoding='ISO-8859-1', quotechar='"', names= ["sentiment", "ID","date","type","user","tweet"])

tokenized = get_tokenized(documents.tweet)
documents["tokenized"] = tokenized
with open("documents","wb") as f:
    pickle.dump(documents, f)
tf, vect = perform_tf(tokenized)
with open("tf","wb") as f:
    pickle.dump(tf, f)
with open("vecttf","wb") as f:
    pickle.dump(vect, f)
topics_words = get_lda_topics(tf, vect)
# print("LDA:")
[print(topics_words[i]) for i in range(len(topics_words))]
with open("topics_words_lda","wb") as f:
    pickle.dump(topics_words, f)

tfidf, vect = perform_tfidf(tokenized)
with open("tfidf","wb") as f:
    pickle.dump(tfidf, f)
with open("vecttfidf","wb") as f:
    pickle.dump(vect, f)
topics_words = get_nmf_topics(tfidf, vect)
print("NMF:")
[print(topics_words[i]) for i in range(len(topics_words))]

with open("topics_words_nmf","wb") as f:
    pickle.dump(topics_words, f)
