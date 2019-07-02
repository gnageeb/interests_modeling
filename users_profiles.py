import pickle
import pandas as pd
import numpy as np
import utils
import csv
import matplotlib.pyplot as plt
import random

sentiments = {0:"negative",4:"positive",2:"neutral"}
with open("documents","rb") as f:
    documents = pickle.load(f)

with open("topics_words_lda","rb") as f:
    topics_words_lda = pickle.load(f)
with open("topics_words_nmf","rb") as f:
    topics_words_nmf = pickle.load(f)

with open("output/lda_topics.csv","w") as f:
    writer = csv.writer(f)
    writer.writerows(topics_words_lda)

with open("output/nmf_topics.csv","w") as f:
    writer = csv.writer(f)
    writer.writerows(topics_words_nmf)

#merge both topics types



def match_topics_hybrid(x):
    import operator
    topics = topics_words_lda + topics_words_nmf
    tweets_topics = {}

    for t in range(len(x.tokenized)):
        tweet = x.tokenized._values[t]+ utils.get_ngrams(2,x.tokenized._values[t])
        num_words = {}
        for i in range(len(topics)):
            topic = topics[i]
            num_words.setdefault(topic[0], 0)
            #tweets_topics.setdefault(topic[0],0)
            for j in range(len(topic)):
                w = topic[j]
                if w in tweet:
                    # weight by invers index in the topic list to give lower weight to less relevant words
                    num_words[topic[0]] += 1/(j+1)
        if all(value == 0 for value in num_words.values()):
            current_topic = "uncategorized"
        else:
            current_topic = max(num_words.items(), key=operator.itemgetter(1))[0]
        tweets_topics.setdefault(current_topic+'_'+sentiments[x.sentiment._values[t]], 0)
        tweets_topics[current_topic+'_'+sentiments[x.sentiment._values[t]]] += 1
    #tweets_topics["uncategorized"] = len(x.tokenized) - sum(tweets_topics.values())
    return tweets_topics

def match_topics_nmf(x):
    import operator
    topics = topics_words_nmf
    tweets_topics = {}

    for t in range(len(x.tokenized)):
        tweet = x.tokenized._values[t]+ utils.get_ngrams(2,x.tokenized._values[t])
        num_words = {}
        for i in range(len(topics)):
            topic = topics[i]
            num_words.setdefault(topic[0], 0)
            #tweets_topics.setdefault(topic[0],0)
            for j in range(len(topic)):
                w = topic[j]
                if w in tweet:
                    # weight by invers index in the topic list to give lower weight to less relevant words
                    num_words[topic[0]] += 1/(j+1)
        if all(value == 0 for value in num_words.values()):
            current_topic = "uncategorized"
        else:
            current_topic = max(num_words.items(), key=operator.itemgetter(1))[0]
        tweets_topics.setdefault(current_topic+'_'+sentiments[x.sentiment._values[t]], 0)
        tweets_topics[current_topic+'_'+sentiments[x.sentiment._values[t]]] += 1
    #tweets_topics["uncategorized"] = len(x.tokenized) - sum(tweets_topics.values())
    return tweets_topics

def match_topics_lda(x):
    import operator
    topics = topics_words_lda
    tweets_topics = {}

    for t in range(len(x.tokenized)):
        tweet = x.tokenized._values[t]+ utils.get_ngrams(2,x.tokenized._values[t])
        num_words = {}
        for i in range(len(topics)):
            topic = topics[i]
            num_words.setdefault(topic[0], 0)
            #tweets_topics.setdefault(topic[0],0)
            for j in range(len(topic)):
                w = topic[j]
                if w in tweet:
                    # weight by invers index in the topic list to give lower weight to less relevant words
                    num_words[topic[0]] += 1/(j+1)
        if all(value == 0 for value in num_words.values()):
            current_topic = "uncategorized"
        else:
            current_topic = max(num_words.items(), key=operator.itemgetter(1))[0]
        tweets_topics.setdefault(current_topic+'_'+sentiments[x.sentiment._values[t]], 0)
        tweets_topics[current_topic+'_'+sentiments[x.sentiment._values[t]]] += 1
    #tweets_topics["uncategorized"] = len(x.tokenized) - sum(tweets_topics.values())
    return tweets_topics





def func2(label,pct, allvals):
    return "{:.1f}%({:d} tweet)".format(round((pct/np.sum(allvals))*100,1),int(pct))+"\n"+label



groups = documents.groupby("user")["tweet","tokenized","sentiment"]
users = groups.apply(match_topics_hybrid)
all_users = pd.DataFrame(list(users), index=users.index)
df = all_users[all_users.sum(axis=1) > 5]

selected_users = random.choices(range(df.shape[0]),k=5)


for u in selected_users:

    fig, ax = plt.subplots(figsize=(6,3), subplot_kw=dict(aspect="equal"))

    data = df.iloc[u].dropna().iloc[df.iloc[u].dropna().to_numpy().nonzero()]
    percentage = [func2(lbl,d,data) for d,lbl in list(zip(data,data.index))]
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(percentage[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)



    plt.savefig('output/users_interests_hybrid/'+df.index[u]+'.png')
    with open('output/'+df.index[u]+'.csv','w') as f:
        groups.get_group(df.index[u]).to_csv(f)


users = groups.apply(match_topics_nmf)
all_users = pd.DataFrame(list(users), index=users.index)
df = all_users[all_users.sum(axis=1) > 5]

#selected_users = random.choices(range(df.shape[0]),k=5)


for u in selected_users:

    fig, ax = plt.subplots(figsize=(6,3), subplot_kw=dict(aspect="equal"))

    data = df.iloc[u].dropna().iloc[df.iloc[u].dropna().to_numpy().nonzero()]
    percentage = [func2(lbl,d,data) for d,lbl in list(zip(data,data.index))]
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(percentage[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)



    plt.savefig('output/users_interests_nmf/'+df.index[u]+'.png')


users = groups.apply(match_topics_lda)
all_users = pd.DataFrame(list(users), index=users.index)
df = all_users[all_users.sum(axis=1) > 5]

#selected_users = random.choices(range(df.shape[0]),k=5)


for u in selected_users:

    fig, ax = plt.subplots(figsize=(6,3), subplot_kw=dict(aspect="equal"))

    data = df.iloc[u].dropna().iloc[df.iloc[u].dropna().to_numpy().nonzero()]
    percentage = [func2(lbl,d,data) for d,lbl in list(zip(data,data.index))]
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(percentage[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)



    plt.savefig('output/users_interests_lda/'+df.index[u]+'.png')

