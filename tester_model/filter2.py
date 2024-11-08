from transformers import BertTokenizer, BertForMaskedLM
import csv
#methods used in this are slower but more accurate. Can be chaged if needed pretty easily at point
#example, lemmination is slower then stemming, and nltk is slower than SpaCy
class filter2:
    #returns a dictionary of twitter usernames mapped to a 2d array of a preprocessed tweet
    #plan, make dict, map usernames to a 2d array, 2d array is the processed tokens of each tweet that the username has made   
    def preprocess(self, filename, id, val):
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
                if row[id] in user_dict: #if the item already exist, change the 2d array to the existing one that it points to
                    tweets = user_dict.get(row[id])

                tokens = tokenizer.tokenize(row[val]) #converts the sentence into subwords
                lemd = tokenizer.convert_tokens_to_ids(tokens) #converts the subword into numbers that bert understands
                
                
                tweets.append(lemd)
                user_dict.update({row[id] : tweets})
                total+=1
        return user_dict

#print(filter2.preprocess("mock_datasets/twitter_training.csv", 1, 3).get("Borderlands"))
#print(filter2.preprocess("mock_datasets/data_gen_example.csv", 0, 1).get("insider_ops414"))