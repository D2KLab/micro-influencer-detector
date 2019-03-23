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

pathToTwitterAuthData = "twitterAccess.txt"
pathToDataFolder = "Data"
pathToDevKeyAndSecret = "consumer_api_keys.txt"
potentialMiPath = "/00_potential_micro_influencers_users/"
followersPath = "/01_followers_list/"
tweetsPath = "/02_users_tweets/"
parametersPath = "/03_users_parameters/"

api = authentication(pathToDevKeyAndSecret, pathToTwitterAuthData)
if len(sys.argv)== 2:
	topic_selected = sys.argv[1]
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
else:
	topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserCsv = pathToTopic + potentialMiPath + "user_list.csv"
unique_users_returned = retrieve_user_list(pathToUserCsv)
pathToFollowerList = pathToTopic + followersPath
pathToUserTweets = pathToTopic + tweetsPath
pathToUserParameters = pathToTopic + parametersPath
compute_and_store_recall(topic_selected, pathToFollowerList, pathToUserTweets, pathToUserParameters, unique_users_returned, api)