# This Python file uses the following encoding: utf-8
#!/usr/bin/

import tweepy
import os
import time
import sys
from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from scipy.spatial import distance


# pathToTwitterAuthData = "./../../twitterAccess.txt"
pathToDataFolder = "./../../Data"
# pathToDevKeyAndSecret = "../../access.txt"
pathToSchwartzCentroids = "../../Schwartz_10"
gloveWordVecFile = "../../../GloVe-1.2/glove.6B.300d.txt"

#-------------------------------------------------#
#--Creating schwartz folder path if not existing--#
#-------------------------------------------------#
if not os.path.exists(pathToDataFolder+"/03_users_parameters/schwartz"):  #here we will store data collection after tweets retrieval
   os.makedirs(pathToDataFolder+"/03_users_parameters/schwartz")

selfdirection = ["creativity", "freedom", "goal-setting", "curious", "independent", "self-respect", "intelligent", "privacy"]
stimulation = ["excitement", "novelty", "challenge", "variety", "stimulation", "daring"]
hedonism = ["pleasure", "sensuous",  "gratification", "enjoyable", "self-indulgent"]
achievement = ["ambitious", "successful", "capable", "influential", "intelligent", "self-respect"] 
power = ["authority", "wealth", "power", "reputation", "notoriety"]
security = ["safety", "harmony", "stability", "order", "security", "clean", "reciprocation", "healthy", "moderate", "belonging"]
conformity = ["obedient", "self-discipline", "politeness", "honoring" , "loyal", "responsible"]
tradition = ["tradition", "humble", "devout", "moderate", "spiritualist"]
benevolence = ["helpful", "honest", "forgiving", "responsible", "loyal", "friendship", "love", "meaningful"]
universalism = ["broadminded", "justice", "equality", "peace", "beauty", "environment-friendly", "wisdom", "environmentalist", "harmony"]

schwartzBasicHumanValues = [selfdirection, stimulation, hedonism, achievement, power, security, conformity, tradition, benevolence, universalism]
schwartzNames = ["selfdirection", "stimulation", "hedonism", "achievement", "power", "security", "conformity", "tradition", "benevolence", "universalism"]


def loadGloveModel(gloveFile):     ##this model works and doesn't kill my cpu
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

localModel = loadGloveModel(gloveWordVecFile)
#schwartzNCentroidFile = Path(pathToSchwartzCentroids+"/"+"schwartzCentroids.txt") #here we find key and secret of the user using the app on Twitter

pos = 0
schwartzCentroids = {}

for humanValue in schwartzBasicHumanValues:
	count_elements = 0.0
	schwartzNCentroid = [0.0]
	schwartzNCentroid = schwartzNCentroid*300
	schwartzNCentroid = np.asarray(schwartzNCentroid)
	for representativeWord in humanValue:
		schwartzNCentroid = schwartzNCentroid + np.asarray(localModel[representativeWord])
		count_elements +=1
	schwartzCentroids[schwartzNames[pos]] = schwartzNCentroid/count_elements
	#f.write((schwartzNames[pos] + " " + str(schwartzNCentroid/count_elements) +"\n").encode("utf-8"))
	pos +=1
#f.close()
print "Centroids computed!"

unique_users_returned = []
fmil = open("./../../Data/00_potential_micro_influencers_users/potential_mi.txt", "r") # file micro influencer list
for line in fmil.readlines():
	line = line.rstrip('\n')
	unique_users_returned.append(line)
fmil.close()
print "User list retrieved"




stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

NON_BMP_RE = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)

for user in unique_users_returned:
	#user = "2_off_the_grid"
	#total_words = {"selfdirection" : 0, "stimulation":0, "hedonism":0, "achievement":0, "power":0, "security":0, "conformity":0, "tradition":0, "benevolence":0, "universalism":0}
	total_words = {}
	cumulative_vectors ={}
	for category in schwartzNames:
		total_words[category] = 0
		cumulative_vectors[category] = np.asarray([0.0]*300)

	print "Working on " + user
	fuss = open(pathToDataFolder+"/03_users_parameters/schwartz/"+user, "w")#file user schwartz scores
	ftun = open(pathToDataFolder + "/" + "02_users_tweets/" + user, "r") #file tweets user name
	doc = ftun.read().decode('utf-8') 
	doc_complete = doc.split('\n')
	doc_cleaned = [clean(doc).split() for doc in doc_complete]

	#doc_further_cleaned = []
	for line in doc_cleaned:
		for word in line:
			if word.startswith('@') or word.isdigit() or ("http" in word):
		 		continue
		 	else:
		 		word = NON_BMP_RE.sub('', word)
		 		if len(word)>0: 
		 			#doc_further_cleaned.append(word)
		 			if word in localModel: 
		 				min_distance = sys.float_info.max
		 				which_schwartz = ""
		 				for pos in schwartzNames: 
		 					now_distance = distance.euclidean(np.asarray(localModel[word]), schwartzCentroids[pos]) #the second is already a numpy array
		 					if now_distance<min_distance:
		 						min_distance = now_distance
		 						which_schwartz = pos
		 				total_words[which_schwartz] += 1 
		 				cumulative_vectors[which_schwartz] += np.asarray(localModel[word])
	ftun.close()

	for category in schwartzNames: 
		if total_words[category] != 0:
			now_centroid = cumulative_vectors[category]/total_words[category]
			dist = distance.euclidean(now_centroid, schwartzCentroids[pos])
			if dist != 0:
				fuss.write(category + " " +str(now_centroid) + " " + str(total_words[category]) + " " + str(dist) + " " + str(total_words[category]*(1/dist)) +"\n")  #300d vector centroid of all tweet for that category, number of words, distance, score=n*1/dist
			else:
				fuss.write(category + " " +str(now_centroid) + " " + str(total_words[category]) + " " + str(dist) + " " + "max_value")
		else:
			fuss.write(category + "  no words in this category \n")
	fuss.close()

	# for line in ftun.readlines():
	# 	line = line.rstrip("\n")
	# 	words = []
	# 	words = line.split(" ")
	# 	for word in words:
	# 		if word.startswith('@') or word.isdigit() or ("http" in word) :
	# 			continue

#print schwartzCentroids["universalism"]

print "process ended successfully"


# def vec(w):
#   return words.loc[w].as_matrix()


# f = open(gloveWordVecFile,'r')
# words = pd.read_csv(f, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# print vec("authority")

# hh = numpy.asarray([82.5, 168.5])
# aa = numpy.asarray([2.5, 8.5])

# cc = hh - aa
# ff = cc * 2
# ss = cc / cc.size
# print cc
# print ff
# print ss