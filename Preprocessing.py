import pandas as pd #Import pandas for data manipulation, data processing
import string 
import re
import nltk
from nltk.stem import  wordnet
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as token

#downloading necessary modules
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
stemmer = PorterStemmer()
lem = wordnet.WordNetLemmatizer()

#To remove punctuation from the string
def checkTranslation(stringInput, translator):
    if isinstance(stringInput, str): 
        return stringInput.translate(translator)
    else:
        return stringInput
#To remove any special characters
def checkText(stringInput, number):
    if isinstance(stringInput, str):
        if(number):
            return re.sub('[^A-Za-z]', ' ', stringInput)
    else:
        return stringInput
#Tokenizing the data and removing stopwords and stemming/lemmatization
def tokenizeData(stringInput, sw):
    if isinstance(stringInput, str):
        tokenz = token(stringInput) 
        tokenizedList = []
        tokenizedRemoved = ''
        for i in tokenz:
            if i not in sw:
                tokenizedList.append(lem.lemmatize(i, pos = 'v'))
        tokenizedRemoved = ' '.join(tokenizedList) #Join the array
        return tokenizedRemoved
    else:
        return stringInput
def checkTwitterHandles(stringInput):
    twitterText = ''
    if isinstance(stringInput, str):
        #Removes oldstyle retweet
        twitterText = re.sub(r'^RT[\s]+', '', stringInput)
        #Removes hyperlinks
        twitterText = re.sub(r'https?:\/\/.*[\r\n]*', '', stringInput)
        #Removes hashtags
        twitterText = re.sub(r'#', '', stringInput)
        return twitterText
    else:
        return stringInput
    
    
    
#Main function to do preprocessing ==> using pandas and data frame for this component
def preprocessTrainData(df):
    df["Text"] = df["Text"].apply(lambda s: checkText(s, False))
    df["Text"] = df["Text"].str.lower()
    removed = str.maketrans('', '', string.punctuation)
    df["Text"] = df["Text"].apply(lambda s: checkTranslation(s, removed))
    df["Text"] = df['Text'].apply(lambda s: checkTwitterHandles(s))
    removed = str.maketrans('', '', string.punctuation)
    line = checkTranslation(line, removed)
    return df


 
  