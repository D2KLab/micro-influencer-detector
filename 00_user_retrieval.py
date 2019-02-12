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

api = authentication(pathToDevKeyAndSecret, pathToTwitterAuthData)

if len(sys.argv)== 2:
	topic_selected = sys.argv[1]
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
else:
	topic_selected = topic_selection()
pathToTopic = create_all_necessary_folders(pathToDataFolder, topic_selected)
unique_users_retrieved = user_list_from_topic_selected(topic_selected, api)
print ("We have found " + str(len(unique_users_retrieved)))
store_user_list_csv(pathToTopic + potentialMiPath + "user_list.csv", unique_users_retrieved)