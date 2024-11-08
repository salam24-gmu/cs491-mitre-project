

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk  import sent_tokenize, word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import csv

#methods used in this are slower but more accurate. Can be chaged if needed pretty easily at point
#example, lemmination is slower then stemming, and nltk is slower than SpaCy
class filter:
    #returns a dictionary of twitter usernames mapped to a 2d array of a preprocessed tweet
    #plan, make dict, map usernames to a 2d array, 2d array is the processed tokens of each tweet that the username has made   
    def preprocess(filename):
        user_dict = dict()
        f = open(filename)
        lem = WordNetLemmatizer()

        #o = open("output.txt", 'w')
        analyzer = SentimentIntensityAnalyzer()
        with f as file_obj: 
            reader_obj = csv.reader(file_obj) 
            stop_words = set(stopwords.words('english'))
            total = 0
            for row in reader_obj:
                #if statement just to control how much we do, this is a slow process. 
                if(total==100):
                    break
                
                lemd = [] #list of words after being lemmatized
                tweets = [] #2d list
                if row[1] in user_dict: #if the item already exist, change the 2d array to the existing one that it points to
                    tweets = user_dict.get(row[1])

                for s in word_tokenize(row[3]):
                    l = lem.lemmatize(s)
                    #this way of handling stop words is for now. need to look into what stopwords to keep and what to get rid of
                    #and from there modify this accordingly 
                    if s not in stop_words:
                        lemd.append(l)
                tweets.append(lemd)
                user_dict.update({row[1] : tweets})
                total+=1
        return user_dict

print(filter.preprocess("../mock_datasets/twitter_training.csv").get("Borderlands"))