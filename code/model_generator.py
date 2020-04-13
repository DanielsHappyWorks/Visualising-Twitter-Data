import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #pip install vaderSentiment


def get_data_frame():
    columns = ['id', 'creation_time', 'retweet_count', 'favorite_count', 'source', 'user_name', 'user_location',
               'content', 'basic_content', 'emotion', 'polarity']
    data_frame = pd.read_csv("./data.csv", names=columns, encoding='utf8', delimiter="~@#@~", engine='python')
    data_frame = data_frame.drop(['creation_time'], axis=1)
    do_label_encoding(data_frame, 'source')
    do_label_encoding(data_frame, 'user_name')
    data_frame['user_location'].fillna("Undefined", inplace=True)
    do_label_encoding(data_frame, 'user_location')
    data_frame['emotion'].fillna("Undefined", inplace=True)
    do_label_encoding(data_frame, 'emotion')
    data_frame = do_tdf(data_frame, 'basic_content')
    # data_frame = do_tdf(data_frame, 'content')
    data_frame = data_frame.drop('content', axis=1)
    return data_frame


def do_tdf(data_frame, col):
    tweet = data_frame[col].values  # df is your dataframe
    tf = TfidfVectorizer(stop_words='english')
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
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(w) for w in nltk.tokenize.word_tokenize(text)]


def run_classifier(x, y, test_size, random_state, feature_range, classifier):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    pipeline = Pipeline(
        [("scaler", MinMaxScaler(feature_range=feature_range)),
         ('classifier', classifier)])

    pipeline.fit(x_train, y_train)

    print("Accuracy", pipeline.score(x, y))
    print("Confusion Matrix on all data")
    print_confusion_matrix(y, pipeline.predict(x), labels=['positive', 'neutral', 'negative'])
    print("Accuracy", pipeline.score(x_test, y_test))
    print("Confusion Matrix on untested data")
    print_confusion_matrix(y_test, pipeline.predict(x_test), labels=['positive', 'neutral', 'negative'])

    print("Vader comparison")
    print("Accuracy", accuracy_score(vader_full, pipeline.predict(x)))
    print("Vader Confusion Matrix on all data")
    print_confusion_matrix(vader_full, pipeline.predict(x), labels=['positive', 'neutral', 'negative'])
    print("Accuracy", accuracy_score(vader_test, pipeline.predict(x_test)))
    print("Vader Confusion Matrix on untested data")
    print_confusion_matrix(vader_test, pipeline.predict(x_test), labels=['positive', 'neutral', 'negative'])

def print_confusion_matrix(y, x, labels):
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(y, x, labels=labels))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(confusion_matrix_df)
    confusion_matrix_df.to_csv("confusion_matrix.csv", sep='\t', encoding='utf-8')


def run_vader_full():
    columns = ['id', 'creation_time', 'retweet_count', 'favorite_count', 'source', 'user_name', 'user_location',
               'content', 'basic_content', 'emotion', 'polarity']
    data_frame = pd.read_csv("./data.csv", names=columns, encoding='utf8', delimiter="~@#@~", engine='python')
    x = data_frame['content']
    y = data_frame['polarity']

    output = []
    for row in x:
        output.append(sentiment_analyzer_scores(row))

    print("Vader Accuracy", accuracy_score(y, output))
    return output

def run_vader_test(test_size, random_state):
    columns = ['id', 'creation_time', 'retweet_count', 'favorite_count', 'source', 'user_name', 'user_location',
               'content', 'basic_content', 'emotion', 'polarity']
    data_frame = pd.read_csv("./data.csv", names=columns, encoding='utf8', delimiter="~@#@~", engine='python')
    data_frame_x = data_frame.drop(['polarity'], axis=1)
    data_frame_y = data_frame['polarity']
    x_train, x_test, y_train, y_test = train_test_split(data_frame_x, data_frame_y, test_size=test_size, random_state=random_state)

    x = x_test['content']

    output = []
    for row in x:
        output.append(sentiment_analyzer_scores(row))

    print("Vader Accuracy", accuracy_score(y_test, output))
    return output

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    category = "neutral"
    if score['compound'] >= 0.05:
        category = "positive"
    elif score['compound'] <= -0.05:
        category = "negative"

    #print("{:-<40} {}".format(sentence, str(score)))
    return category


# Running all functions in correct order
data_frame = get_data_frame()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data_frame)
print(data_frame.head())
data_frame_x = data_frame.drop(['polarity'], axis=1)
data_frame_y = data_frame['polarity']

print("Running Vader Sentiment analysis")
analyser = SentimentIntensityAnalyzer()
vader_full = run_vader_full()
vader_test = run_vader_test(0.3, 10)

print("Running NN analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (-1, 1),
               MLPClassifier(max_iter=300, hidden_layer_sizes=50, shuffle=True, activation='relu', solver='lbfgs'))
print("Running SVC analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (-1, 1),
               SVC(class_weight='balanced', kernel='rbf', C=1, degree=2, gamma=0.1))
print("Running Linear SVC analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (-1, 1), LinearSVC(C=1, dual=False))
print("Running Naive bays analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10, (0, 1), MultinomialNB(alpha=0.005))
