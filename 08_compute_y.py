# This Python file uses the following encoding: utf-8
#!/usr/bin/

from micro_influencer_utilities import *
import tweepy
import os
import time
import sys
from pathlib import Path
import re

pathToDataFolder = "Data"
potentialMiPath = "/00_potential_micro_influencers_users/"
parametersPath = "/03_users_parameters/"

topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserCsv = pathToTopic + potentialMiPath + "user_list.csv"
unique_users_returned = retrieve_user_list(pathToUserCsv)
pathToUserParameters = pathToTopic + parametersPath

threshold_score_embeddness = 1.0
avg_score_recall = 0.0
avg_score_interest = 0.0
total_user = len(unique_users_returned)

#compute averages
for user in unique_users_returned:
	fr = open(pathToUserParameters+"recall/"+user, "r")
	recall = float(fr.read())
	avg_score_recall += recall
	fr.close()
	fi = open(pathToUserParameters+"interest/"+user, "r")
	interest = float(fi.read())
	avg_score_interest += interest
	fi.close()

avg_score_recall = avg_score_recall/total_user
avg_score_interest = avg_score_interest/total_user

for user in unique_users_returned:
	fy = open(pathToUserParameters+"y/"+user, "w")
	fe = open(pathToUserParameters+"embeddness/"+user, "r")
	embeddness = float(fe.read())
	if embeddness < threshold_score_embeddness:
		fy.write("0")
		fy.close()
		continue
	else:
		fr = open(pathToUserParameters+"recall/"+user, "r")
		recall = float(fr.read())
		fr.close()
		fi = open(pathToUserParameters+"interest/"+user, "r")
		interest = float(fi.read())
		fi.close()
		if ((recall > avg_score_recall) and (interest > avg_score_interest)):
			fy.write("1")
			fy.close()
		else:
			fy.write("0")
			fy.close()

print ("y computed")