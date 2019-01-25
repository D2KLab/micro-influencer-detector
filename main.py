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
print "[0] authentication completed with success"

#----------------------------------------------------------------------------------------------------------#
#--this function will be used to react to Twitter api rate limit error/limitation, wait on it and restart--#
#----------------------------------------------------------------------------------------------------------#
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15*60)

#---------------------------------------------------#
#--Create the Data collection folder if not exists--#
#---------------------------------------------------#
if not os.path.exists(pathToDataFolder):  #here we will store data collection after tweets retrieval
   os.makedirs(pathToDataFolder)

#-------------------------------------------------------------------#
#--Create the potential micro influencer list folder if not exists--#
#-------------------------------------------------------------------#
if not os.path.exists(pathToDataFolder+"/00_potential_micro_influencers_users"):  
   os.makedirs(pathToDataFolder+"/00_potential_micro_influencers_users")

#-------------------------------------------------------------------------------------#
#--Create the follower list folder for all potential micro influencers if not exists--#
#-------------------------------------------------------------------------------------#
if not os.path.exists(pathToDataFolder+"/01_followers_list"):  
	   os.makedirs(pathToDataFolder+"/01_followers_list")
	
#----------------------------------------------------------------------------------------------------#
#--Create the selected and filtered tweets folder for all potential micro influencers if not exists--#
#----------------------------------------------------------------------------------------------------#   
if not os.path.exists(pathToDataFolder+"/02_users_tweets"):  
	   os.makedirs(pathToDataFolder+"/02_users_tweets")

#-----------------------------------------------------------------------------#
#--Create users_parameters folder of potential micrp infuencer if not exists--#
#-----------------------------------------------------------------------------#   
if not os.path.exists(pathToDataFolder+"/03_users_parameters/recall"):  
	   os.makedirs(pathToDataFolder+"/03_users_parameters/recall")
if not os.path.exists(pathToDataFolder+"/03_users_parameters/embeddness"):  
	   os.makedirs(pathToDataFolder+"/03_users_parameters/embeddness")

print "[1] all folders created or checked"

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
for tweet in tweepy.Cursor(api.search,q=topic_selected, count = 100, lang = "en").items(10000):  #now 10000
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

#----------------------------------------------------------#
#--Here we write the potential micro influencers selected--#
#----------------------------------------------------------#
fp1 = open(pathToDataFolder+"/00_potential_micro_influencers_users"+"/"+"potential_mi.txt", "w")
for mi_username in unique_users_returned:
	fp1.write((str(mi_username)+"\n").encode("utf-8"))
fp1.close()
print "[2] Searching users potential micro influencers phase completed."

#-----------------------------------------------------------------------------------------#
#--Searching and saving followers lists ids of potential micro influencers on that topic--#
#-----------------------------------------------------------------------------------------#

for i in unique_users_returned:
	while True:
		try:
			print "retrieving followers of:  " + i 
			fp2 = open(pathToDataFolder+"/01_followers_list"+"/"+i, "w")
			for follower_id in limit_handled(tweepy.Cursor(api.followers_ids, screen_name=i).items()):
				fp2.write((str(follower_id)+"\n").encode("utf-8"))
			fp2.close()
			break #exiting infinite while loop
		except tweepy.TweepError:
			time.sleep(30)

print "[3] Storing users followers phase completed."


for username in unique_users_returned:
	username_followers_list = []
	significative_tweets_counter = 0.0
	total_retweets_performed_by_followers = 0.0
	fp2 = open(pathToDataFolder+"/01_followers_list"+"/"+username, "r")
	for line in fp2.readlines():
		line.rstrip('\n')
		username_followers_list.append(line)
	fp2.close()
	while True:
		try:
			#get tweets
			print "Searching tweets of " + username
			fp3 = open(pathToDataFolder+"/02_users_tweets"+"/"+username, "w")
			for page in limit_handled(tweepy.Cursor(api.user_timeline, username, count=100).pages()):  #all tweets
				for tweet in page:
					fp3.write((str(tweet.id) +" : "+tweet.text+"\n").encode("utf-8"))					
			fp3.close()
			break #exiting infinite while loop
		except tweepy.TweepError as e:
			print(e)


print "[4]tweets retrieved and stored"
 
#### passiamo alla fase di embeddness score
#### ovvero la sovrapposizione dei followers tra potenziali micro influencer
#### seguiamo la teoria secondo cui l'informazione si propaga a cascata se almeno due o 
#### più utenti la suggeriscono
#### sommiamo su un contatore quante volte i followers di un utente si ripresentano negli altri della
#### lista e alla fine dividiamo per il numero di follower di quell'utente

compare_follows_dict = {}

for username in unique_users_returned:
	username_followers_list = []
	fp2 = open(pathToDataFolder+"/01_followers_list"+"/"+username, "r")
	for line in fp2.readlines():
		line.rstrip('\n')
		username_followers_list.append(line)
	fp2.close()
	compare_follows_dict[username] = username_followers_list

print "[5] dictionary created"

embeddnessScore = 0.0
for user in compare_follows_dict:
	total_overlapping = 0.0  #sum up all followers of a mi when compare in other mi followers list
	followers_count = len(compare_follows_dict[user])
	for user2 in compare_follows_dict:
		if user != user2 :
			same_followers_list = set(compare_follows_dict[user]) & set(compare_follows_dict[user2])
			total_overlapping += len(same_followers_list)

	if followers_count > 0:
		embeddnessScore = total_overlapping/followers_count
	else:
		embeddnessScore = 0.0
	fp4 = open(pathToDataFolder+"/03_users_parameters/embeddness"+"/"+user+"_embeddnessScore.txt", "w");
	fp4.write(str(embeddnessScore).encode("utf-8"))
	fp4.close()

print "[6] embeddness score computed and stored"


###computing recall and interes## 
for username in unique_users_returned:
	print username
	username_followers_list = []
	fp2 = open(pathToDataFolder+"/01_followers_list"+"/"+username, "r")
	for line in fp2.readlines():
		username_followers_list.append(line.rstrip('\n'))
	fp2.close()
	significative_tweets_counter = 0.0
	total_retweets_performed_by_followers = 0.0
	user_tweets_counter = 0.0 

	fp3 = open(pathToDataFolder+"/02_users_tweets"+"/"+username, "r")
	for line in fp3.readlines():
		#print line
		user_tweets_counter += 1 
		if topic_selected in line:
			significative_tweets_counter +=1
			informations = []
			informations = line.split(" ")
			#print informations[0]
			if (informations[0].rstrip("\n")).isdigit():
				#print "info : " + informations[0].rstrip("\n")				
				try:
					statuses = api.retweets(informations[0].rstrip("\n"))
					for status in statuses:
						#print "status user id :" + str(status.user.id)
						if str(status.user.id).rstrip("\n") in username_followers_list:
							print "status inside if: " + str(status.user.id)
							time.sleep(10)
							total_retweets_performed_by_followers +=1
				except tweepy.RateLimitError:
					time.sleep(15*60)
			#for status in tweepy.Cursor(api.retweets, id=str(informations[0])).items():
			#	if status.user.id_str in username_followers_list:
			#		total_retweets_performed_by_followers +=1
	fp3.close()
	if significative_tweets_counter > 0:
		recallScore = (total_retweets_performed_by_followers/significative_tweets_counter)#/len(username_followers_list) rimosso per la non pagination quindi 100 retweet per tutti
	else:
		recallScore = 0.0
	if user_tweets_counter > 0:
		interest_in_that_topic = significative_tweets_counter/user_tweets_counter
	else:
		interest_in_that_topic = 0.0
	fp4 = open(pathToDataFolder+"/03_users_parameters/recall"+"/"+username+"_recallScore.txt", "w");
	fp4.write((str(recallScore)+ " " +str(interest_in_that_topic)).encode("utf-8"))
	fp4.close()

print "[7] filtered by topic tweets printed (no pure retweets) and recall score calculated"
print "[8]end of main"