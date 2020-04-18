import re
import pandas as pd


class Tweet:
    id = "NA"
    creation_time = "NA"
    retweet_count = 0
    favorite_count = 0
    source = "NA"
    user_name = "NA"
    user_location = "NA"
    content = "NA"
    delimiter = "~@#@~"

    def __init__(self, tweet):
        self.id = tweet.id_str
        self.creation_time = str(tweet.created_at)
        self.retweet_count = tweet.retweet_count
        self.favorite_count = tweet.favorite_count
        self.source = self.process_source(tweet.source)
        self.user_name = tweet.user.screen_name
        self.user_location = tweet.user.location
        self.content = Tweet.process_content(tweet.full_text)
        self.basic_content = Tweet.process_content_to_basic_text(tweet.full_text)

    def get_csv_line(self):
        return self.id + self.delimiter + self.creation_time + self.delimiter + str(self.retweet_count) \
               + self.delimiter + str(self.favorite_count) + self.delimiter + self.source + self.delimiter \
               + self.user_name + self.delimiter + self.user_location + self.delimiter + self.content \
               + self.delimiter + self.basic_content

    def print_tweet(self):
        print(self.id + ", " + self.creation_time + ", " + str(self.retweet_count) \
              + ", " + str(self.favorite_count) + ", " + self.source + ", " \
              + self.user_name + ", " + self.user_location + ", " + self.content + ", " + self.basic_content)

    @staticmethod
    def process_content(text):
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        return text

    @staticmethod
    def process_content_to_basic_text(text):
        text = Tweet.process_content(text)
        # remove links
        text = re.sub(r"http\S+", "", text)
        # replace all special html characters
        for char in [("&", "&amp;"), ('"', "&quot;"), ("'", "&apos;"), (">", "&gt;"), ("<", "&lt;")]:
            text = text.replace(char[1], char[0])
        # keep characters that are necessary for proper sentences
        text = re.sub('[^A-Za-z0-9 /!@?.#/"&\'<>]+', '', text)
        return text

    @staticmethod
    def process_source(text):
        text = text.replace('Twitter for ', '')
        text = text.replace('\t', '')
        return text

    @staticmethod
    def get_columns():
        return ['id', 'creation_time', 'retweet_count', 'favorite_count', 'source', 'user_name', 'user_location',
                'content', 'basic_content', 'emotion', 'polarity']

    def get_as_data_frame(self):
        df = pd.DataFrame([(self.id, self.creation_time, self.retweet_count, self.favorite_count, self.source,
                            self.user_name, self.user_location, self.content, self.basic_content, "NA", "NA")],
                          columns=Tweet.get_columns())
        return df

    def get_as_data_touple(self):
        return (self.id, self.creation_time, self.retweet_count, self.favorite_count, self.source,
                            self.user_name, self.user_location, self.content, self.basic_content, "NA", "NA")
