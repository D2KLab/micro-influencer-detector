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
tweetsPath = "/02_users_tweets/"
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
compute_and_store_interest(topic_selected, pathToTopic + tweetsPath , pathToUserParameters, unique_users_returned)