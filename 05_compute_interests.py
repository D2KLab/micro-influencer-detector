# This Python file uses the following encoding: utf-8
#!/usr/bin/

from micro_influencer_utilities import *
import tweepy
import os
import time
import sys
from pathlib import Path
import re

#pathToTwitterAuthData = "./../../twitterAccess.txt"
pathToDataFolder = "./../../Data"
#pathToDevKeyAndSecret = "../../access.txt"
potentialMiPath = "/00_potential_micro_influencers_users/"
followersPath = "/01_followers_list/"
tweetsPath = "/02_users_tweets/"
parametersPath = "/03_users_parameters/"

#api = authentication(pathToDevKeyAndSecret, pathToTwitterAuthData)
topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserCsv = pathToTopic + potentialMiPath + "user_list.csv"
unique_users_returned = retrieve_user_list(pathToUserCsv)
pathToFollowerList = pathToTopic + followersPath
pathToUserParameters = pathToTopic + parametersPath
compute_and_store_interest(topic_selected, pathToTopic + tweetsPath , pathToUserParameters, unique_users_returned)