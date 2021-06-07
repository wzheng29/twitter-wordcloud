# Twitter Wordcloud
## Introduction
Generate wordclouds with tweets from user input of a Twitter account.
![output](https://user-images.githubusercontent.com/36235827/120943503-45806500-c6fd-11eb-998a-22e5b8bf849c.png)

## Setup
1. In order to run the Twitter API, you must first acquire a set of keys/ tokens from the [Twitter Developer Platform](https://developer.twitter.com/en). You will need the `consumer_key`, `consumer_secret`, `access_token` and `access_secret`.
2. Clone the repo:
```
git clone https://github.com/wzheng29/twitter-wordcloud.git
cd twitter-wordcloud
```
3. Run script:
```
python tweet_clusters.py
```
4. Enter a Twitter Username (without the @ symbol):
```
Enter Username: 
```

## Topics
Concepts applied in this project include:
- Python Twitter API
- K-Means Clustering
- TF-IDF Vectorization
- Natural Language Processing
- Matplotlib

## References
- [Tweepy Documentation](https://docs.tweepy.org/en/latest/index.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [nltk.corpus Documentation](https://www.nltk.org/api/nltk.corpus.html)
- Sklearn: [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) ([Reference](https://towardsdatascience.com/k-means-clustering-8e1e64c1561c))
- WordClouds: [Documentation](https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html) and [Reference](https://www.datacamp.com/community/tutorials/wordcloud-python)
