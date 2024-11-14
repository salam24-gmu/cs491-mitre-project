import filter2
from transformers import BertForPreTraining, BertTokenizer, AutoModel, BertForSequenceClassification, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
class analize_sentiment:
    def run_sent(filename, id, val):
        model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base")
        f = filter2.filter2()
        tweets = f.preprocess(filename, id, val)
        for user in tweets:
            for tweet in tweets.get(user):                
                with torch.no_grad():
                    print("TWEET ", tweet)
                    outputs = model(tweet)
                    print("OUTPUTS FROM MODEL ", outputs)
                    predictions = F.softmax(outputs.logits, dim=1)
                    print("predictions: ", predictions)
                    labels = torch.argmax(predictions, dim=1)
                    print("labels: ", labels)
                    
                    





analize_sentiment.run_sent("/home/logan/Desktop/Mitre_Project/cs491-mitre-project/mock_datasets/data_gen_example.csv", 0, 1)

    