import tweepy as tp  # pip install tweepy
from tweet import Tweet
import os

# Fill in before use
API_KEY = 'Key'
API_SECRET = 'Secret'
ACCESS_TOKEN = 'AccessKey'
ACCESS_TOKEN_SECRET = 'AccessSecret'

API_KEY = 'NEBAA5K4NoLeXR7KMpdpBNQHu'
API_SECRET = 'rdywZEKPocCBidZjT7BS9HWu8KDC0Mi65rnY7joZmctJCAwYfD'
ACCESS_TOKEN = '839486666-nO1lMZ7nde8ZlJnni3CODMjf0UT2upudNSwuFKU6'
ACCESS_TOKEN_SECRET = '0pfbZNjjhWeKISZ92h0ypu5nVQfysNM3Mpyg1uj7yoMg1'

auth = tp.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tp.API(auth, wait_on_rate_limit=True)


def get_tweets(query, count, pages):
    query = query + " -filter:retweets"
    tweets_array = []
    for page in tp.Cursor(api.search,
                          q=query,
                          count=count,
                          tweet_mode='extended',
                          lang="en").pages(pages):
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
    tweets = get_tweets("Animal Crossing", 100, 15)
    generate_csv(tweets, "data.csv")
