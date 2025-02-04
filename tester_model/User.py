#simple class that simply just holds important values for sentiment analysis and temporal analysis
class User:
    def __init__(self, uid, uname) -> None:
        self.uid=uid
        self.uname=uname
        self.score=0
        self.tweets = []
    def __str__(self):
        return f"uid: '{self.uid}', uname: '{self.uname}', threat: {self.score} \n"
    def getTweets(self):
        return self.tweets
        
