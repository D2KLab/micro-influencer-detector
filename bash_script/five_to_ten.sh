#!/bin/bash          
topic="#mechatronics"
echo $topic

python3 ./05_compute_interests.py $topic
python3 ./06_big5_csv.py $topic
python3 ./07_schwartz_csv.py $topic
python3 ./08_compute_y.py $topic
python3 ./09_create_table.py $topic


topic="#greenliving"
echo $topic

python3 ./05_compute_interests.py $topic
python3 ./06_big5_csv.py $topic
python3 ./07_schwartz_csv.py $topic
python3 ./08_compute_y.py $topic
python3 ./09_create_table.py $topic

topic="#robotics"
echo $topic

python3 ./05_compute_interests.py $topic
python3 ./06_big5_csv.py $topic
python3 ./07_schwartz_csv.py $topic
python3 ./08_compute_y.py $topic
python3 ./09_create_table.py $topic

