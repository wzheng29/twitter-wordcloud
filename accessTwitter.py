import sys,tweepy

#Authentication
def twitter_auth():
    try:
        consumer_key = '-----------------'
        consumer_secret = '---------------------'
        access_token = '---------------------'
        access_secret = '-------------------'
    except KeyError:
        sys.stderr.write("enviornment variable not set\n")
        sys.exit(1)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth

def get_twitter_client():
    auth = twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client

#Write tweet to file
if __name__ == '__main__':
    file = open("tweetFile.txt","w")
    user = input("Enter Username: ")
    client = get_twitter_client()
    status = tweepy.Cursor(client.user_timeline, id = user, tweet_mode = "extended").items(1)
    for tweets in status:
        file.write(tweets.full_text)
    file.close