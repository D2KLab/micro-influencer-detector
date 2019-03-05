# This Python file uses the following encoding: utf-8
#!/usr/bin/

from micro_influencer_utilities import *
import tweepy
import os
import time
import sys
from pathlib import Path
from datetime import datetime
import json
import re

pathToDataFolder = "Data2"
potentialMiPath = "/00_potential_micro_influencers_users/"
followersPath = "/01_followers_list/"
tweetsPath = "/02_users_tweets/"

if len(sys.argv)== 2:
	topic_selected = sys.argv[1]
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
else:
	topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserCsv = pathToTopic + potentialMiPath + "user_list.csv"
unique_users_returned = retrieve_user_list(pathToUserCsv)

total_tweets_topic = 0.0

for user in unique_users_returned:
	f = open(pathToTopic+tweetsPath+"/"+user, "r")
	tweets = f.read().split("\n")
	print("tweets of", user, str(len(tweets)))
	total_tweets_topic +=len(tweets)

mean_tweets_per_user = total_tweets_topic/len(unique_users_returned)
print("total_tweets_topic", total_tweets_topic)
print("mean_tweets_per_user", mean_tweets_per_user)
print("number of user retrieved", len(unique_users_returned))