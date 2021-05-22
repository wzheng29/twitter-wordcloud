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

if __name__ == "__main__":
    print(processTweets())
    
