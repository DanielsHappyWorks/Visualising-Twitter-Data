import tweepy as tp  # pip install tweepy
from tweet import Tweet
import os

# Fill in before use
API_KEY = 'Key'
API_SECRET = 'Secret'
ACCESS_TOKEN = 'AccessKey'
ACCESS_TOKEN_SECRET = 'AccessSecret'

auth = tp.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tp.API(auth, wait_on_rate_limit=True)


def get_tweets(query):
    query = query + " -filter:retweets"
    tweets_array = []
    for page in tp.Cursor(api.search,
                          q=query,
                          count=100,
                          tweet_mode='extended',
                          lang="en").pages(15):
        tweets_array = tweets_array + process_page(page)
    return tweets_array


def process_page(page):
    tweets_array = []
    for SearchResults in page:
        tweet = Tweet(SearchResults)
        tweets_array.append(tweet)
    return tweets_array


def generate_csv(tweets_array, path):
    if not os.path.exists(path):
        f = open(path, "a", encoding="utf-8")
        for tweet in tweets_array:
            f.write(tweet.get_csv_line() + "\n")
        f.close()
    else:
        print(f"file already exists {path}")


if __name__ == '__main__':
    tweets = get_tweets("Animal Crossing")
    generate_csv(tweets, "data.csv")
