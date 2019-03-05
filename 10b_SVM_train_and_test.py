# This Python file uses the following encoding: utf-8
#!/usr/bin/


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from micro_influencer_utilities import *
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Reading the data set
pathToDataFolder = "Data0"
parametersPath = "/03_users_parameters/"

if len(sys.argv)== 2:
	topic_selected = sys.argv[1]
	if not topic_selected.startswith('#'):
		topic_selected = "#"+topic_selected
else:
	topic_selected = topic_selection()
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserParameters = pathToTopic + parametersPath

#filtering out not confoming data i.e max-value (oor) or no words in category (anomaly)
f = open(pathToUserParameters+"table/"+topic_selected[1:]+".csv", "r")
f2 = open(pathToUserParameters+"table/"+topic_selected[1:]+"2.csv", "w")
for line in f.readlines():
	if "max" in line:
		continue
	elements = line.split(",")
	if len(elements)<3040:
		continue
	else:
		f2.write(line)
f.close()
f2.close()

#read csv from the filtered and clean csv(just created), not the old one (dirt one)
dataset = pd.read_csv(pathToUserParameters+"table/"+topic_selected[1:]+"2.csv")

#Pre-processing data set to data frame by choosing features (X) and output (y)
#print (dataset.head()) #debug 
y = dataset.y #micro influencer yes = 1, no = 0
X = dataset.drop(["user_screen_name", "Semb", "Srec", "Sint","y"],axis=1)# "num_of_words","distance", "y"], axis=1)

#choose features of interest in prevision excluding scores that preaviously otuput yes/no values
X = X.loc[:, ["big5_O", "big5_C", "big5_E", "big5_A", "big5_N","scoreselfdirection","scorestimulation", "scorehedonism", 
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", "scoretradition", "scorebenevolence", "scoreuniversalism"]] 
# y = dataset.iloc[:, 3039] #old version
#print (X.head()) #debug

#X = X.to_numpy()
X = X.as_matrix()
#y = y.to_numpy()
y = y.as_matrix()
# array = dataset.to_numpy()  #old version
# X = array[:, 1:3038] #old version
# y = array[:, 3039] #old version
# scaler = MinMaxScaler(feature_range=(0, 1)) #old version
# X = scaler.fit_transform(X) #old version
# print (X) #debug
# print(y) #debug
# This technique re-scales the data between a specified range(in this case, between 0–1), 
# to ensure that certain features do not affect the final prediction more than the other features.


rbf_svc = svm.SVC(kernel='rbf', gamma='scale')
# rbf_svc = svm.SVC(kernel='rbf', gamma='scale')
# rbf_svc = svm.SVC(kernel='linear', C=1, gamma=1)
# rbf_svc = svm.SVC(kernel='linear', C=1, gamma=100)
# rbf_svc = svm.SVC(kernel='rbf', C=1, gamma=100)
# rbf_svc = svm.SVC(kernel='rbf', C=100, gamma=100)
# cv = KFold(n_splits=10, random_state=42, shuffle=False) 
# for train_index, test_index in cv.split(X):
#     # print("Train Index: ", train_index, "\n")  #debug
#     # print("Test Index: ", test_index) #debug
#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     clf = rbf_svc.fit(X_train, y_train)
clf = rbf_svc.fit(X, y)

# -----------------------------collect test dataset----------------------------#
pathToDataFolder = "Data"
parametersPath = "/03_users_parameters/"
pathToTopic = pathToDataFolder+"/"+topic_selected
pathToUserParameters = pathToTopic + parametersPath
f = open(pathToUserParameters+"table/"+topic_selected[1:]+".csv", "r")
f2 = open(pathToUserParameters+"table/"+topic_selected[1:]+"2.csv", "w")
for line in f.readlines():
	if "max" in line:
		continue
	elements = line.split(",")
	if len(elements)<3040:
		continue
	else:
		f2.write(line)
f.close()
f2.close()
dataset_test = pd.read_csv(pathToUserParameters+"table/"+topic_selected[1:]+"2.csv")

#Pre-processing data set to data frame by choosing features (X) and output (y)
#print (dataset.head()) #debug 
y_test = dataset_test.y #micro influencer yes = 1, no = 0
X_test = dataset_test.drop(["user_screen_name", "Semb", "Srec", "Sint","y"],axis=1)# "num_of_words","distance", "y"], axis=1)

#choose features of interest in prevision excluding scores that preaviously otuput yes/no values
X_test = X_test.loc[:, ["big5_O", "big5_C", "big5_E", "big5_A", "big5_N","scoreselfdirection","scorestimulation", "scorehedonism", 
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", "scoretradition", "scorebenevolence", "scoreuniversalism"]] 
# y = dataset.iloc[:, 3039] #old version
#print (X.head()) #debug

X_test = X_test.as_matrix()
y_test = y_test.as_matrix()
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_test = scaler.fit_transform(X_test)

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred,).ravel()
print("tn, fp, fn, tp", (tn, fp, fn, tp) )
# print("Second Scoring system")  
print("Classification report: ")
# y_true = y
# y_pred = clf.predict(X)
target_names = ['not_micro_influencer', 'micro_influencer']
print(classification_report(y_test, y_pred, target_names=target_names))
# print("y_test", y_test)

# print("average_precision_score", cross_val_score(clf, X, y,cv=10, scoring='precision'))
#print("recall", cross_val_score(clf, X, y,  cv=10, scoring='recall'))
#print("accuracy", cross_val_score(clf, X, y, scoring='accuracy', cv=10))
#print("f1", cross_val_score(clf, X, y, scoring='f1', cv=10))

#F1 = 2 * (precision * recall) / (precision + recall)


# clf = svm.SVC(gamma='scale', random_state=0)
# cross_val_score(clf, X, y, scoring='average_precision_score', cv=10)

