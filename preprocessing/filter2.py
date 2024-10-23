from transformers import BertTokenizer, BertForMaskedLM
import csv

#methods used in this are slower but more accurate. Can be chaged if needed pretty easily at point
#example, lemmination is slower then stemming, and nltk is slower than SpaCy
class filter2:
    #returns a dictionary of twitter usernames mapped to a 2d array of a preprocessed tweet
    #plan, make dict, map usernames to a 2d array, 2d array is the processed tokens of each tweet that the username has made   
    def preprocess(filename):
        user_dict = dict()
        f = open(filename)

        #o = open("output.txt", 'w')
        with f as file_obj: 
            reader_obj = csv.reader(file_obj) 
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            total = 0
            for row in reader_obj:
                #if statement just to control how much we do, this is a slow process. 
                if(total==100):
                    break
                
                lemd = [] #list of words after being lemmatized
                tweets = [] #2d list
                if row[1] in user_dict: #if the item already exist, change the 2d array to the existing one that it points to
                    tweets = user_dict.get(row[1])

                tokens = tokenizer.tokenize(row[3]) #converts the sentence into subwords
                lemd = tokenizer.convert_tokens_to_ids(tokens) #converts the subword into numbers that bert understands
                
                
                tweets.append(lemd)
                user_dict.update({row[1] : tweets})
                total+=1
        return user_dict

print(filter2.preprocess("../mock_datasets/twitter_training.csv").get("Borderlands"))