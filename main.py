import csv
from collections import defaultdict

TRAIN_FILEPATH = "./mock_datasets/generated_tweets_time_series.csv"

# Nota bene: reading from a CSV will be a pain because user text will often contain commas
def parse_user_dict(data_source_csv):
    raise NotImplementedError

def parse_tweet_dict(data_source_csv):
    raise NotImplementedError

def preprocess(tweet_data):
    raise NotImplementedError

def get_temporal_analysis(data_source_csv):
    raise NotImplementedError

def get_user_text_dict(csv_filepath):
    user_tweets = defaultdict(list)
    
    with open(csv_filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = row['user_id']
            tweet = row['tweet']
            user_tweets[user_id].append(tweet)
    
    return user_tweets

def sort_by_timestamp(user_text_dict):
    raise NotImplementedError

def add_sentiment_scores(user_text_dict):
    raise NotImplementedError

def set_anxiety_bit(user):
    raise NotImplementedError

def get_anxiety_score(user_text_dict):
    raise NotImplementedError

def run_sig_tester(user_text_dict):
    for user in user_text_dict:
        anxiety_timestamps, anxiety_score_over_time = get_anxiety_score(user_text_dict)
        import sentiment_tester

        val = sentiment_tester.run_correlation_test(anxiety_timestamps, anxiety_score_over_time)

        if val:
            set_anxiety_bit(user)


def get_sentiment_classification(data_source_csv):
    # TODO: get time-series data of anxious text sentiment for each person 
    # ...then run the statistical correlation test to check if the person 
    # ...is actually becoming more anxious over time

    user_text_dict = get_user_text_dict(data_source_csv)
    sort_by_timestamp(user_text_dict)
    add_sentiment_scores(user_text_dict)
    run_sig_tester(user_text_dict)

def get_anamoly_layer(data_source_csv):
    raise NotImplementedError

def get_risk_accumulation(data_source_csv):
    raise NotImplementedError

def display_major_threats(risk_accumulation_system):
    raise NotImplementedError

def main(data_source_csv):
    user_dict = parse_user_dict(data_source_csv=TRAIN_FILEPATH)
    tweet_dict = parse_tweet_dict(data_source_csv=TRAIN_FILEPATH)

    tweet_dict = preprocess(tweet_dict)

    temporal_layer = get_temporal_analysis(user_dict, tweet_dict)
    sentiment_layer = get_sentiment_classification(user_dict, tweet_dict)
    anomaly_layer = get_anamoly_layer(user_dict, tweet_dict)

    risk_accumulation_system = get_risk_accumulation(temporal_layer, sentiment_layer, anomaly_layer)

    #TODO: add looping fine tuning from validaiton layer to the temp, sent, and anaom. layers
    
    display_major_threats(risk_accumulation_system)