import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import plotly.express as px
from wordcloud import WordCloud,STOPWORDS #pip install wordcloud
import numpy as np


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

    #labels = reversed(df[a].unique())

   # plt.legend(reversed_handles, labels, loc='lower right', title=b)
    plt.suptitle(a + " vs " + b + " as a % for the " + prefix + " data set")

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    plt.savefig(path + prefix + "_" + a + "_vs_" + b + "_bar.png", bbox_inches='tight')
    plt.close()


def graph_dataframe_group_as_stacked_bar(df, a, b, dir, prefix):
    df.groupby([a, b]).size().unstack().plot(kind='bar', stacked=True)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    current_handles, _ = plt.gca().get_legend_handles_labels()
    reversed_handles = reversed(current_handles)

    labels = reversed(df[a].unique())

    plt.legend(reversed_handles, labels, loc='lower right', title=b)
    plt.suptitle(a + " vs " + b + " as a % for the " + prefix + " data set")

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    plt.savefig(path + prefix + "_" + a + "_vs_" + b + "_bar.png", bbox_inches='tight')
    plt.close()


def graph_tweet_data(df, dir, prefix):
    srtd = df.sort_values(['retweet_count', 'favorite_count'])
    ax = srtd.plot.line(x='retweet_count', y='favorite_count')

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    ax.figure.savefig(path + prefix + "_ret_vs_fav_lin.png", bbox_inches='tight')
    plt.close()


def graph_model_data(df, dir, prefix, hasPolarity):
    if hasPolarity:
        graph_dataframe_group_as_stacked_bar_percentage(df, 'polarity', 'Vader Pred', dir, prefix)
        graph_dataframe_group_as_stacked_bar_percentage(df, 'polarity', 'Classifier Pred', dir, prefix)
    graph_dataframe_group_as_stacked_bar_percentage(df, 'Vader Pred', 'Classifier Pred', dir, prefix)

    fc = df['polarity'].value_counts()
    keys, val = zip(*fc.items())
    if not hasPolarity:
        keys = 'Total'
    plt.figure(figsize=(10, 5))
    plot = plt.subplot(131, label="Manual")
    plot.set_xlabel("Manual")
    if not hasPolarity:
        plot.set_xlabel("")
    plt.bar(keys, val)
    plot = plt.subplot(132, label="Classifier Pred")
    plot.set_xlabel("Classifier Pred")
    fc = df['Classifier Pred'].value_counts()
    keys, val = zip(*fc.items())
    plt.bar(keys, val)
    plot = plt.subplot(133, label="Vader Pred")
    plot.set_xlabel("Vader Pred")
    fc = df['Vader Pred'].value_counts()
    keys, val = zip(*fc.items())
    plt.bar(keys, val)
    plt.suptitle('Tweet predictions for the ' + prefix + ' data set', y=1)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    path = '../output/graphs/' + dir + "/"
    create_path(path)
    plt.savefig(path + prefix + "_bar_counters.png", bbox_inches='tight')
    plt.close()


def graph_tree_map(df, col, cols, dir, prefix):
    df_rpl_epth = df.replace(r'^\s*$', "Undefined", regex=True)
    df_rpl_epth = df_rpl_epth.replace(np.nan, 'Undefined', regex=True)
    df_rpl_epth["all"] = col
    fig = px.treemap(df_rpl_epth, path=cols)
    #fig.show()

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    fig.write_image(path + prefix + "_" + col + "_treemap.svg")
    fig.write_image(path + prefix + "_" + col + "_treemap.png")

    fig = px.sunburst(
        df_rpl_epth,
        path=cols
    )
    path = '../output/graphs/' + dir + "/"
    create_path(path)
    fig.write_image(path + prefix + "_" + col + "_sunburst.svg")
    fig.write_image(path + prefix + "_" + col + "_sunburst.png")
    #fig.show()


def graph_word_cloud(df, col, dir, prefix):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=set(STOPWORDS),
                          min_font_size=12).generate(" ".join(df[col].to_list()))
    plt.imshow(wordcloud)
    plt.axis('off')

    path = '../output/graphs/' + dir + "/"
    create_path(path)
    plt.savefig(path + prefix + "_" + col + "_wordcloud.png", dpi=300)
    plt.close()


def run_graph_original_data():
    dir = os.listdir("../output/model_data")[0]
    # graph for full data
    df = pd.read_pickle("../output/model_data/" + dir + "/full_df.pkl")
    graph_tweet_data(df, "original_data/", "original")
    graph_word_cloud(df, 'basic_content', "original_data/", "original")
    graph_tree_map(df, 'source', ['all', 'polarity', 'source'], "original_data/", "original")
    graph_tree_map(df, 'user_location', ['all', 'polarity', 'user_location'], "original_data/", "original")
    # graph for test data
    df = pd.read_pickle("../output/model_data/" + dir + "/test_df.pkl")
    graph_tweet_data(df, "original_data/", "test")
    graph_word_cloud(df, 'basic_content', "original_data/", "test")
    graph_tree_map(df, 'source', ['all', 'polarity', 'source'], "original_data/", "test")
    graph_tree_map(df, 'user_location', ['all', 'polarity', 'user_location'], "original_data/", "test")


def run_graph_model_data():
    frames_full = []
    frames_test = []
    for dir in os.listdir("../output/model_data"):
        df = pd.read_pickle("../output/model_data/" + dir + "/full_df.pkl")
        graph_model_data(df, "model/" + dir, "original", True)
        frames_full.append(df)
        df = pd.read_pickle("../output/model_data/" + dir + "/test_df.pkl")
        graph_model_data(df, "model/" + dir, "test", True)
        frames_test.append(df)
    result_full = pd.concat(frames_full)
    result_test = pd.concat(frames_test)
    graph_model_data(result_full, "model/all_model", "all_model_full", True)
    graph_model_data(result_test, "model/all_model", "all_model_test", True)


def run_graph_unseen_data():
    frames = []
    for dir in os.listdir("../output/unseen_data"):
        df = pd.read_pickle("../output/unseen_data/" + dir + "/df.pkl")
        graph_tweet_data(df, "unseen/" + dir, "unseen")
        graph_model_data(df, "unseen/" + dir, "unseen", False)
        graph_word_cloud(df, 'basic_content', "unseen/" + dir, "unseen")
        graph_tree_map(df, 'source', ['all', 'source'], "unseen/" + dir, "unseen")
        graph_tree_map(df, 'user_location', ['all', 'user_location'], "unseen/" + dir, "unseen")
        df['Topic'] = dir
        frames.append(df)
    result = pd.concat(frames)

    graph_tree_map(result, 'source', ['all', 'Topic', 'source'], "unseen/all_unseen", "all_unseen")
    graph_tree_map(result, 'user_location', ['all', 'Topic', 'user_location'], "unseen/all_unseen", "all_unseen")

    graph_tweet_data(result, "unseen/all_unseen", "all_unseen")
    graph_model_data(result, "unseen/all_unseen", "all_unseen", False)

    graph_word_cloud(result, 'basic_content', "unseen/all_unseen", "all_unseen")


# run through original data plots output files
run_graph_original_data()
# run through all model output files
run_graph_model_data()
# run through all unseen data output files
run_graph_unseen_data()
