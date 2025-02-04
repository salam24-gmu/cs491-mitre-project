#simple class that simply just holds important values for sentiment analysis and temporal analysis
class Tweet:
    def __init__(self, tid, tweet, timestamp) -> None:
        self.tid=tid
        self.tweet=tweet
        self.threat=False
        self.timestamp = timestamp
    def __str__(self):
        return f"timestamp: '{self.timestamp}', tid: '{self.tid}' Tweet contents: '{self.tweet}', is threat?: {self.threat}"