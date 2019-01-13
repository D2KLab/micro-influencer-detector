# This Python file uses the following encoding: utf-8
#!/usr/bin/

import tweepy
import pyperclip
import os
import time
import sys
from pathlib import Path
from datetime import datetime
import json
import re

pathToTwitterAuthData = "./../../twitterAccess.txt"
pathToDataFolder = "./../../Data"
pathToDevKeyAndSecret = "../../access.txt"

try:
    f = open(pathToDevKeyAndSecret, "r")  #retrieving key and secret in a local file, not available on github
    									  #ask this info to the developer of the app
except IOError:
    print ("file with key and secret of Twitter app not found, ask to the developer\n")
    exit()
else:
    print ("file opening and information retrieving")

#read my developer app key and secret from local file .gitignore
consumer_key = f.readline().rstrip('\n') 
print(consumer_key)
consumer_secret = f.readline().rstrip('\n') 
print(consumer_secret)
f.close()

#create the root folder if not exists
if not os.path.exists(pathToDataFolder):  #here we will store data collection after tweets retrieval
   os.makedirs(pathToDataFolder)			

twitterAuthData = Path(pathToTwitterAuthData) #here we find key and secret of the user using the app on Twitter

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

if not twitterAuthData.is_file() or os.stat(pathToTwitterAuthData).st_size == 0:  
	#no previous authentication data, need to autenthicate via browser
	try:
	    redirect_url = auth.get_authorization_url()
	    print("Redirect url:", redirect_url)
	    #copy redirect url in clipboard
	    #pyperclip.copy(redirect_url)
	except tweepy.TweepError:
	    print ('Error! Failed to get request token.')

	verifier = raw_input('Verifier:')

	try:
	    auth.get_access_token(verifier)
	except tweepy.TweepError:
	    print ('Error! Failed to get access token.')

	access_token = auth.access_token
	access_token_secret = auth.access_token_secret
	twitterAuthData = open(pathToTwitterAuthData, "w") 
	twitterAuthData.write(auth.access_token+"\n"+auth.access_token_secret+"\n");
	twitterAuthData.close();

else:
	#already got auth data, read it from file
	twitterAuthData = open(pathToTwitterAuthData, "r") 
	access_token = twitterAuthData.readline().rstrip('\n')
	access_token_secret = twitterAuthData.readline().rstrip('\n')
	twitterAuthData.close()


auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


topic_selected = raw_input('What topic are you looking micro-influencers for?\n')
if not topic_selected.startswith('#'):
	topic_selected = "#"+topic_selected
	print topic_selected

users_returned = []
flag = 0
for tweet in tweepy.Cursor(api.search,q="#offgrid", count = 100, lang = "en").items(200):
	if flag == 0:
		flag = 1
		print("\n\n\n")
		print("users with at least 1k and at most 20k followers,")
		print("having recently spoke about topic selected")
	if (tweet.user.followers_count>1000 and tweet.user.followers_count<20000):
		#print (tweet.user.screen_name)
		users_returned.append(tweet.user.screen_name)

unique_users_returned = set(users_returned)
print(unique_users_returned)



	
#public_tweets = api.home_timeline()
#for tweet in public_tweets:
#    print(tweet.text)

#user = api.get_user('Agenzia_Ansa')
# print(user.screen_name)
# print(user.followers_count)

# count_friends = 0

# for friend in user.friends():
# 	if count_friends<10:
# 		count_friends +=1
#    		print(friend.screen_name)
#    	else:
#    		break