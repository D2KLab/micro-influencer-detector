# This Python file uses the following encoding: utf-8
#!/usr/bin/

from micro_influencer_utilities import *
import tweepy
import os
import time
import sys
from pathlib import Path
import re

pathToDataFolder = "Data"
potentialMiPath = "/00_potential_micro_influencers_users/"
parametersPath = "/03_users_parameters/"

if len(sys.argv)== 2:
	topic_selected = sys.argv[1]
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
else:
	topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserCsv = pathToTopic + potentialMiPath + "user_list.csv"
unique_users_returned = retrieve_user_list(pathToUserCsv)
pathToUserParameters = pathToTopic + parametersPath

fbig5 = open(pathToUserParameters+"/big5/big5.csv", "r")
fout = open(pathToUserParameters+"table/"+topic_selected[1:]+".csv", "w")

#table :username, Semb, Srec, Sint, Big5 *5 , (Schwartz*300 + n + dist + score) * 10 , y : total 3040 columns 

fout.write("user_screen_name,Semb,Srec,Sint")
for i in ["O", "C", "E", "A", "N"]:
	fout.write(","+"big5_"+i)

schwartzNames = ["selfdirection", "stimulation", "hedonism", "achievement", "power", "security", "conformity", "tradition", "benevolence", "universalism"]

for i in schwartzNames:
	for j in range(300):
		fout.write(","+str(i)+"_"+str(j))
	fout.write(",num_of_words"+str(i)+",distance"+str(i)+",score"+str(i))
fout.write(",y\n")

for user in unique_users_returned:
	fout.write(str(user))
	fin = open(pathToUserParameters+"embeddness/"+user, "r")
	Semb = float(fin.read())
	fin.close()
	fout.write(","+str(Semb))
	fin = open(pathToUserParameters+"recall/"+user, "r")
	Srec = float(fin.read())
	fin.close()
	fout.write(","+str(Srec))
	fin = open(pathToUserParameters+"interest/"+user, "r")
	Sint = float(fin.read())
	fin.close()
	fout.write(","+str(Sint))
	big5 = fbig5.readline().split(",")
	big5[5] = big5[5].rstrip("\n")
	for big5Value in big5[1:]:
		fout.write(","+str(big5Value))
	fin = open(pathToUserParameters+"schwartz/"+user, "r")
	for line in fin.readlines():
		line = line.rstrip("\n")
		elements = line.split(",")
		for element in elements[1:]:
			fout.write(","+str(element))
	fin.close()
	fin = open(pathToUserParameters+"y/"+user, "r")
	y = int(fin.read())
	fin.close()
	fout.write(","+str(y))
	fout.write("\n")
fout.close()


fout = open(pathToUserParameters+"table/"+topic_selected[1:]+".csv", "r")
line = fout.readline().split(",")
fout.close()
print ("columns are: "+str(len(line)))
print ("table created")


