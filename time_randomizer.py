import pandas as pd
import numpy as np

import os

current_directory = os.path.dirname(os.path.realpath(__file__))

TRAIN_DATA_FILEPATH = os.path.join(current_directory, "finetune", "data", "generated_tweets_time_series_training_REFORMATTED.csv")

df = pd.read_csv(filepath_or_buffer=TRAIN_DATA_FILEPATH, sep=",", quotechar='"')

def randomize_timestamps(df:pd.DataFrame):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    min_time = df["Timestamp"].min()
    max_time = df["Timestamp"].max()

    random_timestamps = pd.to_datetime(np.random.uniform(min_time.value, max_time.value, size=len(df)))

    df["Timestamp"] = random_timestamps

    return df
