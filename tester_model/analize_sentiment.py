import filter2
from transformers import BertForPreTraining, BertTokenizer, AutoModel
import torch
import torch.nn.functional as F
class analize_sentiment:
    def run_sent(filename, id, val):
        model = AutoModel.from_pretrained("vinai/bertweet-base")
        f = filter2.filter2()
        tweets = f.preprocess(filename, id, val)
        for user in tweets:
            for tweet in tweets.get(user):                
                with torch.no_grad(): #still figuring out how to get sentiment from this data. Tried something from internet and didn't work and IDK why
                    outputs = model(tweet)
                    features = model(tweet)
                    print(outputs)
                    print(features)
                    





analize_sentiment.run_sent("mock_datasets/data_gen_example.csv", 0, 1)

    