import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # pip install vaderSentiment
from code.tweet_scraper import get_tweets
from code.tweet import Tweet

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def get_data_frame_raw():
    columns = ['id', 'creation_time', 'retweet_count', 'favorite_count', 'source', 'user_name', 'user_location',
               'content', 'basic_content', 'emotion', 'polarity']
    return pd.read_csv("./data.csv", names=columns, encoding='utf8', delimiter="~@#@~", engine='python')


def get_data_frame_processed(df, unseen_data):
    df = do_tdf(df, 'basic_content', unseen_data)

    # Altering any col here will create a model that contains more/less inputs
    # remove all unecessary excluding the tweet text cols for model as they effect performance
    df = df.drop('creation_time', axis=1)
    df = df.drop('content', axis=1)
    df = df.drop('source', axis=1)
    df = df.drop('user_name', axis=1)
    df = df.drop('user_location', axis=1)
    df = df.drop('retweet_count', axis=1)
    df = df.drop('favorite_count', axis=1)
    df = df.drop('emotion', axis=1)
    return df


def do_tdf(data_frame, col, unseen_data):
    if unseen_data:
        tfidf_matrix = tf.transform(data_frame[col].values)
    else:
        tweet = data_frame[col].values  # df is your dataframe
        tfidf_matrix = tf.fit_transform(tweet)
    matrix_as_array = list(tfidf_matrix.toarray())

    columns = []
    for i in range(len(matrix_as_array[0])):
        columns.append(col + str(i))

    matrix_as_df = pd.DataFrame(data=matrix_as_array, columns=columns)
    data_frame = data_frame.drop([col], axis=1)
    return pd.concat([data_frame, matrix_as_df], axis=1)


def do_label_encoding(data_frame, col):
    lb_encoder = LabelEncoder()
    data_frame[col] = lb_encoder.fit_transform(data_frame[col])
    return data_frame


def tokenize(text):
    # remove these characters as they get tokenised and appear very often:
    text = text.replace("#", '')
    text = text.replace("@", '')
    text = text.replace(".", '')
    stemmer = nltk.stem.PorterStemmer()
    filtered_sentence = [w for w in nltk.tokenize.word_tokenize(text) if not w in set(stopwords.words('english'))]
    return [stemmer.stem(w) for w in filtered_sentence]


def run_classifier(x, y, feature_range, classifier):
    x_train, x_test, y_train, y_test = get_test_data(x, y)

    pipeline = Pipeline(
        [("scaler", MinMaxScaler(feature_range=feature_range)),
         ('classifier', classifier)])

    pipeline.fit(x_train, y_train)

    print("Confusion Matrix on all data")
    print_prediction_results(y, pipeline.predict(x))
    print("Confusion Matrix on untested data")
    print_prediction_results(y_test, pipeline.predict(x_test))

    print("Vader comparison: all data")
    print_prediction_results(vader_full, pipeline.predict(x))
    print("Vader comparison: test data")
    print_prediction_results(vader_test, pipeline.predict(x_test))

    return pipeline


def print_prediction_results(y, y_pred):
    print("Accuracy", accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred, labels=['positive', 'neutral', 'negative']))
    print(classification_report(y, y_pred, labels=['positive', 'neutral', 'negative']))


def run_vader_full(x, y):
    output = run_vader(x['content'])
    print("Vader vs Manual Predictions on Full Data")
    print_prediction_results(y, output)
    return output


def run_vader_test(x, y):
    x_train, x_test, y_train, y_test = get_test_data(x, y)
    output = run_vader(x_test['content'])
    print("Vader vs Manual Predictions on Test Data")
    print_prediction_results(y_test, output)
    return output


def run_vader(text_array):
    output = []
    for text in text_array:
        output.append(sentiment_analyzer_scores(text))
    return output


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    category = "neutral"
    if score['compound'] >= 0.05:
        category = "positive"
    elif score['compound'] <= -0.05:
        category = "negative"

    return category


def get_test_data(x, y):
    return train_test_split(x, y, test_size=test_size,
                            random_state=random_state)


def scrape_and_classify_tweets(query, count, pages, classifier):
    tweets_df = pd.DataFrame(columns=Tweet.get_columns())
    tweets = get_tweets(query, count, pages)

    for tweet in tweets:
        tweets_df.append(tweet.get_as_data_frame())

    tweets_df = pd.DataFrame([tweet.get_as_data_touple() for tweet in tweets], columns=Tweet.get_columns())

    vader_results = run_vader(tweets_df['content'])

    tweets_df = get_data_frame_processed(tweets_df, True)
    tweets_df = tweets_df.drop('polarity', axis=1)
    results = classifier.predict(tweets_df)

    print(f"Vader comparison: {query}")
    print("Accuracy", accuracy_score(vader_results, results))
    print(confusion_matrix(vader_results, results, labels=['positive', 'neutral', 'negative']))


# Define globally used params
print("*****************************SETTING GLOBAL VALUES*****************************")
test_size = 0.3
random_state = 10
output_dir = "../output"
tf = TfidfVectorizer(max_features=2000, stop_words=stopwords.words('english'), tokenizer=tokenize)
print("\n")

print("*****************************GETTING DATAFRAME FROM CSV*****************************")
# Running all functions in correct order
data_frame = get_data_frame_raw()
print(data_frame.head())
data_frame_x = data_frame.drop(['polarity'], axis=1)
data_frame_y = data_frame['polarity']
print("\n")

print("*****************************RUNNING VADER ANALYSIS ON FULL CONTENT OF TWEET*****************************")
analyser = SentimentIntensityAnalyzer()
vader_full = run_vader_full(data_frame_x, data_frame_y)
vader_test = run_vader_test(data_frame_x, data_frame_y)

print("*****************************CONVERTING DATA FRAME TO FORMAT SUPPORTED BY CLASSIFIERS*****************************")
data_frame = get_data_frame_processed(data_frame, False)
print(data_frame.head())
data_frame_x = data_frame.drop(['polarity'], axis=1)
data_frame_y = data_frame['polarity']
print("\n")

print("*****************************RUNNING ANALYSIS ON DIFFERENT CLASSIFIERS*****************************")
print("Running MLPClassifier analysis")
run_classifier(data_frame_x, data_frame_y, (-1, 1),
               MLPClassifier(max_iter=300, hidden_layer_sizes=60, activation='identity', solver='lbfgs',
                             random_state=random_state))
print("\n")
print("Running SVC analysis")
run_classifier(data_frame_x, data_frame_y, (-1, 1),
               SVC(class_weight='balanced', kernel='poly', C=1, degree=3, gamma='scale', random_state=random_state))
print("\n")
print("Running RandomForestClassifier analysis")
model = run_classifier(data_frame_x, data_frame_y, (-1, 1),
                       RandomForestClassifier(n_estimators=100, random_state=0))
print("\n")
print("Running Linear SVC analysis")
run_classifier(data_frame_x, data_frame_y, (-1, 1), LinearSVC(C=2, dual=False, random_state=random_state))
print("\n")
print("Running MultinomialNB analysis")
run_classifier(data_frame_x, data_frame_y, (0, 1), MultinomialNB(alpha=0.05))
print("\n")
print("*****************************TESTING ACCURACY AGAINST VADER USING RANDOM FOREST CLASSIFIER ON UNSEEN DATA*****************************")
# Take a classifier and fit it to new tweets comparing to vader results here (max 100 per 1 page -> api limitation)
scrape_and_classify_tweets("Donald Trump", 100, 5, model)
scrape_and_classify_tweets("Boris Jonson", 100, 5, model)
scrape_and_classify_tweets("Covid-19", 100, 5, model)
scrape_and_classify_tweets("Isolation", 100, 5, model)
scrape_and_classify_tweets("virtual reality", 100, 5, model)
scrape_and_classify_tweets("ps5", 100, 5, model)
scrape_and_classify_tweets("Xbox Series X", 100, 5, model)
