# micro-influencer-detector
Framework to detect micro influencers of a given topic using tweets

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


