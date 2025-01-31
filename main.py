def parse_user_dict(data_source_csv):
    raise NotImplementedError

def parse_tweet_dict(data_source_csv):
    raise NotImplementedError

def preprocess(tweet_data):
    raise NotImplementedError

def get_temporal_analysis(data_source_csv):
    raise NotImplementedError

def get_sentiment_classification(data_source_csv):
    raise NotImplementedError

def get_anamoly_layer(data_source_csv):
    raise NotImplementedError

def get_risk_accumulation(data_source_csv):
    raise NotImplementedError

def display_major_threats(risk_accumulation_system):
    raise NotImplementedError

def main(data_source_csv):
    user_dict = parse_user_dict(data_source_csv)
    tweet_dict = parse_tweet_dict(data_source_csv)

    tweet_dict = preprocess(tweet_dict)

    temporal_layer = get_temporal_analysis(user_dict, tweet_dict)
    sentiment_layer = get_sentiment_classification(user_dict, tweet_dict)
    anomaly_layer = get_anamoly_layer(user_dict, tweet_dict)

    risk_accumulation_system = get_risk_accumulation(temporal_layer, sentiment_layer, anomaly_layer)

    #TODO: add looping fine tuning from validaiton layer to the temp, sent, and anaom. layers
    
    display_major_threats(risk_accumulation_system)