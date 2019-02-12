from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from pathlib import Path
from micro_influencer_utilities import topic_selection
import numpy as np
import datasetUtils as dsu
import embeddings
import sys
import os
import re

dataset_path = "FastText/dataset.vec"
tweet_threshold = 3

vectorizer = CountVectorizer(stop_words="english", analyzer="word")
analyzer = vectorizer.build_analyzer()

topic_selected = topic_selection()
pathToUserList = "./Data/"+topic_selected+"/00_potential_micro_influencers_users/user_list.csv"
f = open(pathToUserList, "r")
usernames = f.read().split(",")

print("Loading embeddings dataset...")
wordDictionary = dsu.parseFastText(dataset_path)
print("Data successfully loaded.")

outfile = []
for username in usernames:
    print("\nUser:",username)
    tweet_file_path = "Data/"+topic_selected+"/02_users_tweets/"+username
    if not os.path.isfile(tweet_file_path):
        print("Error. Cannot find tweets for username",username)
        continue

    filteredTweets = []
    #preprocess tweets
    for tweet in open(tweet_file_path, "r"):

        if re.match(r'^(RT)', tweet):
            #ingore pure retweets (= with no associated text)
            continue
        #remove links starting with "http"
        tweet = re.sub(r'((http)([^\s]*)(\s|$))|((http)([^\s]*)$)', "", tweet)
        #remove links with no http (probably unnecessary)
        tweet = re.sub(r'(\s([^\s]*)\.([^\s]*)\/([^\s]*)\s)|(^([^\s]*)\.([^\s]*)\/([^\s]*)(\s|$))|(\s([^\s]*)\.([^\s]*)\/([^\s]*)$)', " ", tweet)
        #remove mentions
        tweet = re.sub(r'(\s(@)([^\s]*)\s)|((^@)([^\s]*)(\s|$))|(@([^\s]*)$)', " ", tweet)
        #hashtags are removed by countvectorizer

        filteredTweets.append(tweet)

        if len(filteredTweets) == 0:
            print("Not enough tweets for prediction.")
            continue

    #now we can process the tweet using embeddings.transofrmTextForTraining
    try:
        tweetEmbeddings = embeddings.transformTextForTesting(wordDictionary, tweet_threshold, filteredTweets, "conc")
        print("Embeddings computed.")
    except:
        #most of tweets are ingored for brevity/no embedding correspondence
        print("Not enough tweets for prediction.")
        continue

    scores = {}
    #load the saved ML models
    for trait in ["O","C","E","A","N"]:
        # model = joblib.load("Models/SVM_fasttext_conc_"+trait+".pkl")
        model = joblib.load("Models/SVM/SVM_"+trait+".pkl")
        mean = np.mean(tweetEmbeddings, axis = 0)
        score = model.predict([mean])
        scores[trait] = float(str(score[0])[0:5])
        print("\tScore for",trait,"is:",str(score[0])[0:5])

    jung = ""
    if scores["E"] > 3:
        jung = "E"
    else:
        jung = "I"
    if scores["O"] > 3:
        jung = jung + "N"
    else:
        jung = jung + "S"
    if scores["A"] > 3:
        jung = jung + "F"
    else:
        jung = jung + "T"
    if scores["C"] > 3:
        jung = jung + "J"
    else:
        jung = jung + "P"

    print("\tJungian type is",jung)

    outfile.append(username+","+str(scores["O"])+","+str(scores["C"])+","+str(scores["E"])+","+str(scores["A"])+","+str(scores["N"]))

pathToResults = "./Data/"+topic_selected+"/03_users_parameters/big5/"
filename = pathToResults + "big5.csv"
fp = open(filename,"w")
for elem in outfile:
    fp.write(elem+"\n")
fp.close()