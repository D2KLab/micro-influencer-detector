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
#------------------------------------------------------#
#-------------authentication phase started-------------#
#------------------------------------------------------#
def authentication(pathToDevKeyAndSecret, pathToTwitterAuthData):
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
	print "[0] authentication completed with success"
	return api
def create_all_necessary_folders(pathToDataFolder, topic_selected):
	#--Create the Data collection folder if not exists--#
	if not os.path.exists(pathToDataFolder):  #here we will store data collection after tweets retrieval
	   os.makedirs(pathToDataFolder)
	#--One folder per topic, in order to do not overwrite folders--#
	pathToDataFolder = pathToDataFolder+"/"+topic_selected
	#--Create the potential micro influencer list folder if not exists--#
	if not os.path.exists(pathToDataFolder+"/00_potential_micro_influencers_users"):  
	   os.makedirs(pathToDataFolder+"/00_potential_micro_influencers_users")
	#--Create the follower list folder for all potential micro influencers if not exists--#
	if not os.path.exists(pathToDataFolder+"/01_followers_list"):  
		   os.makedirs(pathToDataFolder+"/01_followers_list")
	#--Create the selected and filtered tweets folder for all potential micro influencers if not exists--#
	if not os.path.exists(pathToDataFolder+"/02_users_tweets"):  
		   os.makedirs(pathToDataFolder+"/02_users_tweets")
	#--Create users_parameters folder of potential micr0 infuencer if not exists--#
	if not os.path.exists(pathToDataFolder+"/03_users_parameters/recall"):  
		   os.makedirs(pathToDataFolder+"/03_users_parameters/recall")
	if not os.path.exists(pathToDataFolder+"/03_users_parameters/embeddness"):  
		   os.makedirs(pathToDataFolder+"/03_users_parameters/embeddness")
	if not os.path.exists(pathToDataFolder+"/03_users_parameters/interest"):  
		   os.makedirs(pathToDataFolder+"/03_users_parameters/interest")
	print "[1] all folders created or checked" 
	return pathToDataFolder 
def topic_selection():
	topic_selected = raw_input('What topic are you looking micro-influencers for?\n')
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
		#print topic_selected 
	return topic_selected         
def user_list_from_topic_selected(topic_selected, api):
	users_returned = []
	print("Looking for users with at least 1k and at most 20k followers,")
	print("having recently spoke about topic selected", topic_selected)
	for tweet in tweepy.Cursor(api.search,q=topic_selected, count = 100, lang = "en").items(1000): #now 1000, we'll exec on more topics			
		if (tweet.user.followers_count>1000 and tweet.user.followers_count<20000):
			#print (tweet.user.screen_name)
			if tweet.user.friends_count < tweet.user.followers_count:
				users_returned.append(tweet.user.screen_name)
	unique_users_returned = set(users_returned)
	unique_users_returned = list(unique_users_returned)
	return unique_users_returned
def store_user_list_csv(pathToStore, unique_users_returned):
	fp1 = open(pathToStore, "w")
	for mi_username in unique_users_returned:
		if mi_username == unique_users_returned[-1]:
			fp1.write(str(mi_username).encode("utf-8"))
		else:
			fp1.write((str(mi_username)+",").encode("utf-8"))
	fp1.close()
	print "[2] List of potential micro influencers stored."
def retrieve_user_list(pathToUserList):
	unique_users_returned = []
	f = open(pathToUserList, "r")
	content = f.read()
	unique_users_returned = content.split(",")
	return unique_users_returned 
def limit_handled(cursor):
	while True:
		try:
			yield cursor.next()
		except tweepy.RateLimitError:
			time.sleep(15*60) 
def retrieve_and_store_followers_csv(pathToFollowerList, unique_users_returned, api):
	for i in unique_users_returned:
		count = 0
		while True:
			try:
				print "retrieving followers of:  " + i 
				fp2 = open(pathToFollowerList + i +".csv", "w")
				for follower_id in limit_handled(tweepy.Cursor(api.followers_ids, screen_name=i).items()):
					if count == 0:
						fp2.write(str(follower_id).encode("utf-8"))
						count +=1
					else:
						fp2.write((","+str(follower_id)).encode("utf-8"))
						count +=1
				fp2.close()
				break #exiting infinite while loop
			except tweepy.TweepError:
				time.sleep(15)
		print i + "'s followers stored. They are " + str(count)
	print "[3] Storing users followers phase completed." 
def retrieve_and_store_tweet_tab_back(pathToUserTweets, unique_users_returned, api):
	for username in unique_users_returned:
		while True:
			try:
				#get tweets
				print "Searching tweets of " + username
				#fp3 = open(pathToDataFolder+"/02_users_tweets"+"/"+username, "w")
				fp3 = open(pathToUserTweets+username, "w")
				for page in limit_handled(tweepy.Cursor(api.user_timeline, username, count=100, lang = "en").pages()):  #all tweets
					for tweet in page:
						fp3.write((str(tweet.id)+"\t").encode("utf-8"))
						new_tweet = ""
						tweet_cleaned = tweet.text.split("\n")
						for sintagma in tweet_cleaned:
							new_tweet = new_tweet + " " + sintagma
						new_tweet2 = ""
						tweet_cleaned2 = new_tweet.split("\t")
						for sintagma2 in tweet_cleaned2:
							new_tweet2 = new_tweet2 + " " + sintagma2
						fp3.write((new_tweet2 + "\n").encode("utf-8"))
						#at the end of the story we have  ---->  TweetId\tTweetText\n				
				fp3.close()
				break #exiting infinite while loop
			except tweepy.TweepError as e:
				print(e)
	print "[4]tweets retrieved and stored" 
def compute_and_store_embeddeness(pathToFollowerList, pathToUserParameters, unique_users_returned):
	compare_follows_dict = {}
	for username in unique_users_returned:
		username_followers_list = []
		fp2 = open(pathToFollowerList+username+".csv", "r")
		username_followers_list = fp2.read().split(",")
		fp2.close()
		compare_follows_dict[username] = username_followers_list
	print "[5] dictionary created"

	for user in compare_follows_dict:
		embeddnessScore = 0.0
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
		#fp4 = open(pathToDataFolder+"/03_users_parameters/embeddness"+"/"+user+"_embeddnessScore.txt", "w");
		fp4 = open(pathToUserParameters+"embeddness/"+user+"_Semb.txt", "w")
		fp4.write(str(embeddnessScore).encode("utf-8"))
		fp4.close()
	print "[6] embeddness score computed and stored" 
def compute_and_store_interest(topic_selected, pathToUserTweets, pathToUserParameters, unique_users_returned):
	for user in unique_users_returned:
		#print topic_selected
		#print topic_selected[1:]
		significative_tweets_counter = 0.0
		total_tweets = 0.0 
		f = open(pathToUserTweets+user, "r")
		for line in f.readlines():
			if topic_selected in line or topic_selected[1:] in line:
				significative_tweets_counter +=1
			total_tweets += 1
		f.close()
		if total_tweets > 0:
			Sint = (significative_tweets_counter/total_tweets)
		else:
			Sint = 0.0
		fout = open(pathToUserParameters+"interest/"+user, "w")
		fout.write(str(Sint))
		fout.close()
def compute_and_store_recall(topic_selected, pathToFollowerList, pathToUserTweets, pathToUserParameters, unique_users_returned, api):
	for username in unique_users_returned:
		print username
		username_followers_list = []
		fp2 = open(pathToFollowerList+username+".csv", "r")
		username_followers_list = fp2.read().split(",")
		fp2.close()
		significative_tweets_counter = 0.0
		total_retweets_performed_by_followers = 0.0
		user_tweets_counter = 0.0
		fp3 = open(pathToUserTweets+username, "r")
		for line in fp3.readlines():
			user_tweets_counter += 1 
			if topic_selected in line or topic_selected[1:] in line:
				significative_tweets_counter +=1
				informations = []
				informations = line.split("\t")
				if informations[0].isdigit():			
					try:
						statuses = api.retweets(informations[0])
						for status in statuses:
							#print "status user id :" + str(status.user.id)
							if str(status.user.id).rstrip("\n") in username_followers_list:
								total_retweets_performed_by_followers +=1
					except tweepy.RateLimitError:
						time.sleep(15*60)
		fp3.close()
		if significative_tweets_counter > 0:
			recallScore = (total_retweets_performed_by_followers/significative_tweets_counter)#/len(username_followers_list) rimosso per la non pagination quindi 100 retweet per tutti
		else:
			recallScore = 0.0
		if user_tweets_counter > 0:
			interest_in_that_topic = significative_tweets_counter/user_tweets_counter
		else:
			interest_in_that_topic = 0.0
		fp4 = open(pathToUserParameters+"recall/"+username+"_Srec.txt", "w");
		fp4.write(str(recallScore).encode("utf-8"))
		fp4.close()
	print "[7] filtered by topic tweets printed and recall score calculated"   