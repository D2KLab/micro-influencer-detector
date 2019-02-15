# This Python file uses the following encoding: utf-8
#!/usr/bin/


import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from micro_influencer_utilities import *
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import svm

#Reading the data set
pathToDataFolder = "Data"
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
f = open(pathToUserParameters+"/table/"+topic_selected[1:]+".csv", "r")
f2 = open(pathToUserParameters+"/table/"+topic_selected[1:]+"2.csv", "w")
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
dataset = pandas.read_csv(pathToUserParameters+"/table/"+topic_selected[1:]+"2.csv")

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

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) #i.e. example from documentation
# clf.score(X_test, y_test) 
scores = []

rbf_svc = svm.SVC(kernel='rbf', gamma='scale')
# documentation on the following function can be found at
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
cv = KFold(n_splits=10, random_state=42, shuffle=False) 
for train_index, test_index in cv.split(X):
    # print("Train Index: ", train_index, "\n")  #debug
    # print("Test Index: ", test_index) #debug
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    clf = rbf_svc.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))  ##this is accuracy

   

my_scores = cross_val_score(clf, X, y, cv=10)  
print("Accuracy: %0.2f (+/- %0.2f)" % (my_scores.mean(), my_scores.std() * 2)) 
y_pred = rbf_svc.predict(X)
y_true = y
print("precision: ",metrics.precision_score(y_true, y_pred))
print("recall: ",metrics.recall_score(y_true, y_pred))
print("f1: ",metrics.f1_score(y_true, y_pred))
#print("Score Cross aka accuracy: ", np.mean(scores))

fout = open(pathToUserParameters+"cv_results.txt", "w")
fout.write("Accuracy +-: " + str(my_scores.mean()) +" " + str(my_scores.std() * 2) + "\n")
fout.write("precision: " + str(metrics.precision_score(y_true, y_pred)) +"\n")
fout.write("recall: " + str(metrics.recall_score(y_true, y_pred)) +"\n")
fout.write("f1: " + str(metrics.f1_score(y_true, y_pred)) +"\n")
fout.close()


#print("average_precision_score", cross_val_score(clf, X, y,cv=10, scoring='precision' ))
#print("recall", cross_val_score(clf, X, y,  cv=10, scoring='recall'))
#print("accuracy", cross_val_score(clf, X, y, scoring='accuracy', cv=10))
#print("f1", cross_val_score(clf, X, y, scoring='f1', cv=10))

#F1 = 2 * (precision * recall) / (precision + recall)


# clf = svm.SVC(gamma='scale', random_state=0)
# cross_val_score(clf, X, y, scoring='average_precision_score', cv=10)

# We are using the RBF kernel of the SVM model, implemented using the sklearn library 
# First, we indicate the number of folds we want our data set to be split into. 
# Here, we have used 10-Fold CV (n_splits=10), where the data will be split into 10 folds.
# If you want, removing comment on print lines inside for loop, you can print out the indexes of
# the training and the testing sets in each iteration to clearly see the process of K-Fold CV 
# where the training and testing set changes in each iteration.
# We specify the training and testing sets to be used in each iteration. 
# For this, we use the indexes(train_index, test_index) specified in the K-Fold CV process.
# Then, we train the model in each iteration using the train_index of each iteration of the K-Fold process 
# and append the error metric value to a list(scores )

# The error metric computed using the best_svr.score() function is the r2 score. 
# Each iteration of F-Fold CV provides an r2 score. We append each score to a list 
# and get the mean value in order to determine the overall accuracy of the model.



