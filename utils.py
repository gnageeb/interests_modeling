
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
ngrams = 2

n_topics = 20
n_top_words = 100

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english')).union(set(['nothing','omg,''something','thing','things',
                                                                        'thank','someone','one','oh','lt', 'sorry',
                                                                        'everything','everyone','haha','yay','dont',
                                                                        'ok','u','im','quot','http','lol','x','wow',
                                                                        'yes','yeah','amp','hey','hi','thanks','aww',
                                                                        'awww']))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates


def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda word, pos, chunk: chunk != 'O') if key]

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def get_tokenized(documents):
    tokenized = []
    for text in documents:
        tokenized.append(extract_candidate_words(text))
    return tokenized

def perform_tfidf(tokenized):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.05, min_df=0.00001, ngram_range=(1, ngrams),
                                       analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x: x,token_pattern=None)

    tfidf = tfidf_vectorizer.fit_transform(tokenized)
    return tfidf, tfidf_vectorizer

def perform_tf(tokenized):
    tf_vectorizer = CountVectorizer(max_df=0.05, min_df=0.00001, ngram_range=(2, ngrams),
                                    analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x: x,token_pattern=None)
    tf = tf_vectorizer.fit_transform(tokenized)
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


def get_ngrams(n,tokenized):
    from nltk import ngrams
    grams = ngrams(tokenized, n)
    return [' '.join(gram) for gram in grams]
