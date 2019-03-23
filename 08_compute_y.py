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

if len(sys.argv)== 2:
	topic_selected = sys.argv[1]
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
else:
	topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserCsv = pathToTopic + potentialMiPath + "user_list.csv"
unique_users_returned = retrieve_user_list(pathToUserCsv)
pathToUserParameters = pathToTopic + parametersPath

if not os.path.exists(pathToUserParameters+"y2"):  
		   os.makedirs(pathToUserParameters+"y2")

threshold_score_embeddness = 1
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

total_influencers = 0
total_influencers_2 = 0
avg_threshold = 3
threshold_score_embeddness_2 = 0.1

for user in unique_users_returned:
	fy = open(pathToUserParameters+"y/"+user, "w")
	fy2 = open(pathToUserParameters+"y2/"+user, "w")
	fe = open(pathToUserParameters+"embeddness/"+user, "r")
	embeddness = float(fe.read())
	if embeddness < threshold_score_embeddness:
		fy.write("0")
		fy.close()
		#--------new------------------#
		if ((embeddness>= threshold_score_embeddness_2) and(recall > avg_score_recall/avg_threshold) and (interest > avg_score_interest/avg_threshold)):
			fy2.write("1")
			fy2.close()
			total_influencers_2+=1
		else:
			fy2.write("0")
			fy2.close()
		#--------new------------------#
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
			total_influencers+=1
			fy.close()
		else:
			fy.write("0")
			fy.close()
		#--------new------------------#
		if ((recall > avg_score_recall/avg_threshold) and (interest > avg_score_interest/avg_threshold)):
			#fy.write("1")
			total_influencers_2+=1
			fy2.write("1")
			fy2.close()
			#fy.close()
		else:
			#fy.write("0")
			#fy.close()
			total_influencers_2+=0
			fy2.write("0")
			fy2.close()
		#--------new------------------#	

print ("y computed, there are: " + str(total_influencers) + "/"+ str(len(unique_users_returned))+" micro-influencers")
print("for the topic: " +topic_selected)
print("total micro-influencers with new threshold:", total_influencers_2)