# micro-influencer-detector
Framework to detect micro influencers of a given topic using computational linguistics and Twitter platform

### SETUP
It is needed to register you and your application on Twitter platform to use this technology.
https://developer.twitter.com/en/apply-for-access.html

You will obtain two keys: 
1) Consumer API keys: two strings 

  Create a file called: consumer_api_keys.txt and paste them like this:
  
    string1\n
    
    string2
    
2) Access token & access token secret: two strings

  Create a file called: twitterAccess.txt and paste them like this:
  
    string3\n
    
    string4

Whole code runs with python3.x version under Ubuntu 18.04 distribution, 
if you want to run under Windows OS it is possible you have to check and change 
folders' path. 

### Twitter data retrieval from TOPIC
Follow this logical sequence:
00_user_retrieval.py

01_follower_list_retrieval.py

02_tweets_retrieval.py

03_compute_embeddedness.py 

04_compute_recall.py 

05_compute_interests.py 

Before going on use Train_SVM_model.py in order to create SVM model to perform
big5 evaluation on next step. This program is useful just the first time you'll run
this process, if you need to perform analysis over other topic, you can skip this one,
because you've already the models trained. 

06_big5_csv.py

07_scheartz_csv.py

08_compute_y.py

09_create_table.py

10_ten_fold_cross_validation.py
