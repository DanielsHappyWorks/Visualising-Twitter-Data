import nltk
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def get_data_frame():
    columns = ['id', 'creation_time', 'retweet_count', 'favorite_count', 'source', 'user_name', 'user_location', 'content', 'basic_content']
    return pd.read_csv("./data.csv", names=columns, encoding='utf8', delimiter="~@#@~", engine='python')


def tokenize(text):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(w) for w in nltk.tokenize.word_tokenize(text)]


def run_classifier(x, y, test_size, random_state, classifier):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    nb_clf = Pipeline(
        [('vectorizer', TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), tokenizer=tokenize)),
         ('classifier', classifier)])

    nb_clf.fit(x_train, y_train)

    print("Accuracy", nb_clf.score(x_test, y_test))
    print("Confusion Matrix")
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, nb_clf.predict(x_test)))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(confusion_matrix_df)
    confusion_matrix_df.to_csv("confusion_matrix.csv", sep='\t', encoding='utf-8')


# Running all functions in correct order
data_frame = get_data_frame()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data_frame)
print(data_frame.head())
data_frame_x = data_frame.drop(['expression'], axis=1)
data_frame_y = data_frame['expression']

print("Running NN analysis")
run_classifier(data_frame_x, data_frame_y, 0.3, 10,
               MLPClassifier(max_iter=300, hidden_layer_sizes=50, shuffle=True, activation='relu', solver='lbfgs'))
