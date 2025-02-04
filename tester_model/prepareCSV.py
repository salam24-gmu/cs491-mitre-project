import csv
import User
import Tweet
from TweetNormalizer import normalizeTweet #useing a pre trained model for analyzing sentiment from tweets.
#credit goes to https://github.com/VinAIResearch/BERTweet?tab=readme-ov-file#usage2
#using the model to get values from tweets to train our own model on detecting insider threats
class prepareCSV:
    #returns a dictionary of the user id mapped to a user class. each user class contains an array of "Tweet" class  
    def prepare(self, filename, id, val):
        user_dict = dict()
        f = open(filename)
        #o = open("output.txt", 'w')
        with f as file_obj: 
            reader_obj = csv.reader(file_obj) 
            #code that was used for tokenizing. Not sure how we will impliment the preprocessing
            #tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
            total = 0
            for row in reader_obj:
                #if statement just to control how much we do, this is a slow process. 
                if(total==100):
                    break
                if row[id] in user_dict: #if the item already exist, change the 2d array to the existing one that it points to
                    user = user_dict.get(row[id])
                else:
                    #in the provided csv file, 9 is where the user name is stored
                    user = User.User(row[id], row[9])
                print(user)
                #code used from the tweetnormalizer.py file to do preprocess the information for bertweet.
                #line = normalizeTweet(row[val])
                #ids = torch.tensor([tokenizer.encode(line)])

                #row[1] in the inputted file is the id, and row 0 is the time. For the actual tweet, it is in row 3
                tweet = Tweet.Tweet(row[1], row[val], row[0])
                print(tweet)
                user.tweets.append(tweet)
                user_dict.update({row[id] : user})
                total+=1
        return user_dict
f = prepareCSV()

#just some code to verify that the correct stuff is getting stored
d = f.prepare("../mock_datasets/generated_tweets_time_series_training.csv",8, 3 )
for a in d:
    print(d.get(a))
    print(d.get(a).getTweets()[0])