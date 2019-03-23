#!/bin/bash          
topic="#opioid"
python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic

topic="#womenintech"
python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic

topic="#fintech"
python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic

topic="#freelance"
python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic

topic="#AI"
python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic

topic="#AugmentedReality"
python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic