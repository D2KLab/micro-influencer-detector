#!/bin/bash          
topic="#tinyhouse"
echo $topic

python3 ./00_user_retrieval.py $topic
python3 ./01_follower_list_retrieval.py $topic
python3 ./02_tweets_retrieval.py $topic
python3 ./03_compute_embeddedness.py $topic
python3 ./04_compute_recall.py $topic
python3 ./05_compute_interests.py $topic
python3 ./07_big5_csv.py $topic
python3 ./08_schwartz_csv.py $topic
python3 ./09_compute_y.py $topic
python3 ./10_create_table.py $topic
python3 ./11_ten_fold_cross_validation.py $topic


