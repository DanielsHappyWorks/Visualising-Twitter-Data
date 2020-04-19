import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os


def create_path(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed")


def graph_dataframe_group_as_stacked_bar_percentage(df, a, b, dir, prefix):
    df.groupby([a, b]).size().groupby(level=0).apply(
        lambda x: 100 * x / x.sum()
    ).unstack().plot(kind='bar', stacked=True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    current_handles, _ = plt.gca().get_legend_handles_labels()
    reversed_handles = reversed(current_handles)

    labels = reversed(df[a].unique())

    plt.legend(reversed_handles, labels, loc='lower right', title=b)
    plt.suptitle(a + " vs " + b + " as a % for the" + prefix + " data set")

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    plt.savefig(path + prefix + "_bar_pcnt.png")
    plt.close()

def graph_tweet_data(df, dir, prefix):
    srtd = df.sort_values(['retweet_count', 'favorite_count'])
    ax = srtd.plot.line(x='retweet_count', y='favorite_count')

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    ax.figure.savefig(path + prefix + "_ret_vs_fav_lin.png")

def graph_model_data(df, dir, prefix, hasPolarity):
    if hasPolarity:
        graph_dataframe_group_as_stacked_bar_percentage(df, 'polarity', 'Vader Pred', dir, prefix)
        graph_dataframe_group_as_stacked_bar_percentage(df, 'polarity', 'Classifier Pred', dir, prefix)
    graph_dataframe_group_as_stacked_bar_percentage(df, 'Vader Pred', 'Classifier Pred', dir, prefix)

    fc = df['polarity'].value_counts()
    print(fc.keys())
    keys, val = zip(*fc.items())
    plot = plt.subplot(131, label="Manual")
    plot.set_title("Manual")
    plt.bar(keys, val)
    plot = plt.subplot(132, label="Classifier Pred")
    plot.set_title("Classifier Pred")
    fc =df['Classifier Pred'].value_counts()
    print(fc.keys())
    keys, val = zip(*fc.items())
    plt.bar(keys, val)
    plot = plt.subplot(133, label="Vader Pred")
    plot.set_title("Vader Pred")
    fc =df['Vader Pred'].value_counts()
    print(fc.keys())
    keys, val = zip(*fc.items())
    plt.bar(keys, val)
    plt.suptitle('Tweet predictions for the '+ prefix + ' data set')

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    plt.savefig(path + prefix + "_bar_counters.png")
    plt.close()


def run_graph_original_data():
    dir = os.listdir("../output/model_data")[0]
    print(dir)
    # graph for full data
    df = pd.read_pickle("../output/model_data/" + dir + "/full_df.pkl")
    graph_tweet_data(df, dir, "original")
    # graph for test data
    df = pd.read_pickle("../output/model_data/" + dir + "/test_df.pkl")
    graph_tweet_data(df, dir, "test")


def run_graph_model_data():
    for dir in os.listdir("../output/model_data"):
        print(dir)
        df = pd.read_pickle("../output/model_data/" + dir + "/full_df.pkl")
        graph_model_data(df, dir, "original", True)
        df = pd.read_pickle("../output/model_data/" + dir + "/test_df.pkl")
        graph_model_data(df, dir, "test", True)


def run_graph_unseen_data():
    for dir in os.listdir("../output/unseen_data"):
        print(dir)
        df = pd.read_pickle("../output/unseen_data/" + dir + "/df.pkl")
        graph_tweet_data(df, dir, "unseen")
        graph_model_data(df, dir, "unseen", False)


# run through original data plots output files
run_graph_original_data()
# run through all model output files
run_graph_model_data()
# run through all unseen data output files
run_graph_unseen_data()
