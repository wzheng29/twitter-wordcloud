import sys,tweepy
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
#nltk.download('stopwords')
#nltk.download('punkt')

#Authentication
def twitter_auth():
    try:
        consumer_key = '---'
        consumer_secret = '---'
        access_token = '---'
        access_secret = '---'
    except KeyError:
        sys.stderr.write("Enviornment variable not set\n")
        sys.exit(1)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth

def get_twitter_client():
    auth = twitter_auth()
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

#Get tweets from user since May 1st
def processTweets():
    user = input("Enter Username: ")
    api = get_twitter_client()
    status = tweepy.Cursor(api.user_timeline, id = user, q='-filter:retweets', tweet_mode = "extended", lang = 'en', since = '2021-05-10').items()

    tweet_list = [tweets.full_text for tweets in status]
    word_list = []
    for tweet in tweet_list:
        for word in tweet.split():
            if word.isalpha():
                word_list.append(word)
    group_words = []
    split_words = np.array_split(word_list, 10)
    for group in split_words:
        group_words.append(' '.join(group))

    return group_words

def tokenizer(keyword):
    #stemmer = PorterStemmer()
    return [word for word in keyword.split(' ')]

if __name__ == "__main__":
    keywords = processTweets()
    tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'))
    X = pd.DataFrame(tfidf.fit_transform(keywords).toarray(), index=keywords, columns=tfidf.get_feature_names())
    features = tfidf.fit_transform(keywords)
    k = 5
    model = KMeans(n_clusters=k, init='k-means++', n_init=2, max_iter=100)
    model.fit(features)
    order_centroids = model.cluster_centers_.argsort()[:,::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("cluster ID %d: " % i)
        for j in order_centroids[i, :10]:
            print(' %s' % terms[j])
        print('--------------------------------')
