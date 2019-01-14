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

pathToTwitterAuthData = "./../../twitterAccess.txt"
pathToDataFolder = "./../../Data"
pathToDevKeyAndSecret = "../../access.txt"

#------------------------------------------------------#
#-------------authentication phase started-------------#
#------------------------------------------------------#
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
consumer_secret = f.readline().rstrip('\n') 
f.close()
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)	

twitterAuthData = Path(pathToTwitterAuthData) #here we find key and secret of the user using the app on Twitter
if not twitterAuthData.is_file() or os.stat(pathToTwitterAuthData).st_size == 0:  
	#no previous authentication data, need to autenthicate via browser
	try:
	    redirect_url = auth.get_authorization_url()
	    print("Redirect url:", redirect_url)
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
api = tweepy.API(auth)#, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#-------------------------------------------------------#
#------------authentication phase terminated------------#
#-------------------------------------------------------#

#---------------------------------------------------#
#--Create the Data collection folder if not exists--#
#---------------------------------------------------#
if not os.path.exists(pathToDataFolder):  #here we will store data collection after tweets retrieval
   os.makedirs(pathToDataFolder)

#------------------------------------------------------------------------------------------------#
#--Selection of the topic, just one for the moment, in which you want to find micro influencers--#
#------------------------------------------------------------------------------------------------#
topic_selected = raw_input('What topic are you looking micro-influencers for?\n')
if not topic_selected.startswith('#'):
	topic_selected = "#"+topic_selected
	#print topic_selected

#----------------------------------------------------------------------------#
#--Here we look for random Twitter users that have used that topic recently--#
#----------------------------------------------------------------------------#
users_returned = []
flag = 0
for tweet in tweepy.Cursor(api.search,q=topic_selected, count = 100, lang = "en").items(200):
	if flag == 0:
		flag = 1
		print("\n")
		print("Looking for users with at least 1k and at most 20k followers,")
		print("having recently spoke about topic selected", topic_selected)
	if (tweet.user.followers_count>1000 and tweet.user.followers_count<20000):
		#print (tweet.user.screen_name)
		if tweet.user.friends_count < tweet.user.followers_count:
			users_returned.append(tweet.user.screen_name)

unique_users_returned = set(users_returned)
unique_users_returned = list(unique_users_returned)

if not os.path.exists(pathToDataFolder+"/potential_micro_influencers_users"):  
   os.makedirs(pathToDataFolder+"/potential_micro_influencers_users")

fp1 = open(pathToDataFolder+"/potential_micro_influencers_users"+"/"+"potential_mi.txt", "w")
for mi_username in unique_users_returned:
	fp1.write((str(mi_username)+"\n").encode("utf-8"))
fp1.close()
print "Searching users phase completed."

#-----------------------------------------------------------------------------------------#
#--Searching and saving followers lists ids of potential micro influencers on that topic--#
#-----------------------------------------------------------------------------------------#
print("\n")
if not os.path.exists(pathToDataFolder+"/followers_list"):  
	   os.makedirs(pathToDataFolder+"/followers_list")
if not os.path.exists(pathToDataFolder+"/users_tweets"):  
	   os.makedirs(pathToDataFolder+"/users_tweets")

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15*60)


####work in progress retweets list
for username in unique_users_returned:
	#get tweets
	print "Searching tweets of " + username
	fp3 = open(pathToDataFolder+"/users_tweets"+"/"+username+"_tweets.txt", "w")
	for page in limit_handled(tweepy.Cursor(api.user_timeline, username, count=100).pages()):
		for tweet in page:
			#----------->>>>>>#results = api.retweets(firstTweet.id)
			fp3.write((str(tweet.id) +" : "+tweet.text+"\n").encode("utf-8"))
	fp3.close()

print "tweets printed"

###work in progress2 rate limit 
for i in unique_users_returned:
	print "potential micro influencer " + i
	fp2 = open(pathToDataFolder+"/followers_list"+"/"+i+"_followers_ids.txt", "w")
	for follower_id in limit_handled(tweepy.Cursor(api.followers_ids, screen_name=i).items()):
		fp2.write((str(follower_id)+"\n").encode("utf-8"))
	fp2.close()
	# try:
	# 	array_of_user_friends_id = api.followers_ids(screen_name=i)
	# except:
	# 	print  "Having some trouble"
	# #print array_of_user_friends_id[1]
	# fp2 = open(pathToDataFolder+"/followers_list"+"/"+i+"_followers_ids.txt", "w")
	# for follower in array_of_user_friends_id:
	# 	fp2.write((str(follower)+"\n").encode("utf-8"))
	# fp2.close()

print "Storing users followers phase completed."

	#id_tweet - tweet_text 1 file, id_tweet list of user that have retweets 2 file, foreach user for each tweet  
	
