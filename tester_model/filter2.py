
from transformers import  AutoTokenizer
from transformers import BertTokenizer, BertForMaskedLM
import csv
import torch
from TweetNormalizer import normalizeTweet #useing a pre trained model for analyzing sentiment from tweets.
#credit goes to https://github.com/VinAIResearch/BERTweet?tab=readme-ov-file#usage2
#using the model to get values from tweets to train our own model on detecting insider threats
class filter2:
    #returns a dictionary of twitter usernames mapped to a 2d array of a preprocessed tweet
    #plan, make dict, map usernames to a 2d array, 2d array is the processed tokens of each tweet that the username has made   
    def preprocess(self, filename, id, val):
        user_dict = dict()
        f = open(filename)
        #o = open("output.txt", 'w')
        with f as file_obj: 
            reader_obj = csv.reader(file_obj) 
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            total = 0
            for row in reader_obj:
                #if statement just to control how much we do, this is a slow process. 
                if(total==100):
                    break
                tweets = [] #2d list
                if row[id] in user_dict: #if the item already exist, change the 2d array to the existing one that it points to
                    tweets = user_dict.get(row[id])

                line = normalizeTweet(row[val])
                ids = torch.tensor([tokenizer.encode(line)])
                #print(ids) 
                tweets.append(ids)
                user_dict.update({row[id] : tweets})
                total+=1
        return user_dict

#print(filter2.preprocess("mock_datasets/twitter_training.csv", 1, 3).get("Borderlands"))
#print(filter2.preprocess("mock_datasets/data_gen_example.csv", 0, 1).get("insider_ops414"))