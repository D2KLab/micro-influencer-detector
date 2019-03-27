# This Python file uses the following encoding: utf-8
#!/usr/bin/
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import sys
from micro_influencer_utilities import topic_selection
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

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
dataset = pd.read_csv(pathToUserParameters+"/table/"+topic_selected[1:]+"2.csv")

#Pre-processing data set to data frame by choosing features (X) and output (y)
#print (dataset.head()) #debug 
y = dataset.y #micro influencer yes = 1, no = 0
X = dataset.drop(["user_screen_name", "Semb", "Srec", "Sint","y"],axis=1)# "num_of_words","distance", "y"], axis=1)
X = X.loc[:, ["big5_O", "big5_C", "big5_E", "big5_A", "big5_N","scoreselfdirection","scorestimulation", "scorehedonism", 
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", 
"scoretradition", "scorebenevolence", "scoreuniversalism"]] 
#"big5_O", "big5_C", "big5_E", "big5_A", "big5_N",

X = X.as_matrix()
y = y.as_matrix()
Xmat = X
ymat = y
X = np.array(X)
y = np.array(y)
print("y: ", y)
scaler = MinMaxScaler(feature_range=(0, 1)) #old version
X = scaler.fit_transform(X) #old version
Xmat =scaler.fit_transform(Xmat)

# kfold = model_selection.KFold(n_splits=10, random_state=42)
skf = StratifiedKFold(n_splits=5)

#model
rbf_svc = svm.SVC(kernel='rbf', gamma='scale', class_weight={0:1,1:5})#dai peso 10 alla classe 1
# print("cross_val_rec_micro", cross_val_score(rbf_svc, X, y, scoring='recall_micro', cv=10))

#scoring metrics
accuracy = []
f1 = []
recall = []
precision = []
mtn = []
mtp = []
mfn = []
mfp = []
print("Scoring parameters with StratifiedKfold, Classification Binary Case")
for train_index, test_index in skf.split(X, y):
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	y_pred = rbf_svc.fit(X_train, y_train).predict(X_test)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	mtn.append(tn)
	mfp.append(fp)
	mfn.append(fn)
	mtp.append(tp)
	accuracy.append(rbf_svc.fit(X_train, y_train).score(X_test, y_test))
	f1.append(f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	recall.append(recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	precision.append(precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
print("svc mean accuracy: ",np.mean(accuracy))
print("svc mean recall: ", np.mean(recall))
print("svc mean precision: ", np.mean(precision))
print("svc mean f1: ", np.mean(f1))

print("svc mean tn:", np.mean(mtn))
print("svc mean fp:", np.mean(mfp))
print("svc mean fn:", np.mean(mfn))
print("svc mean tp:", np.mean(mtp))

print("Random forest phase following")
clf1 = RandomForestClassifier(n_estimators=100, random_state=0, class_weight={1:10}, max_features=15)
clf = clf1
accuracy = []
f1 = []
recall = []
precision = []
mtn = []
mtp = []
mfn = []
mfp = []

skf = KFold(n_splits=3)
print("Scoring parameters with StratifiedKfold, Classification Binary Case")
for train_index, test_index in skf.split(X, y):
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	y_pred = clf.fit(X_train, y_train).predict(X_test)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	mtn.append(tn)
	mfp.append(fp)
	mfn.append(fn)
	mtp.append(tp)
	accuracy.append(clf.fit(X_train, y_train).score(X_test, y_test))
	f1.append(f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	recall.append(recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	precision.append(precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
print("rf mean accuracy: ",np.mean(accuracy))
print("rf mean recall: ", np.mean(recall))
print("rf mean precision: ", np.mean(precision))
print("rf mean f1: ", np.mean(f1))

print("\nrf mean tn:", np.mean(mtn))
print("rf mean fp:", np.mean(mfp))
print("rf mean fn:", np.mean(mfn))
print("rf mean tp:", np.mean(mtp))




# print(X)
# print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# #model
# rbf_svc = svm.SVC(kernel='rbf', gamma='scale', class_weight={0:1,1:10})#dai peso 10 alla classe 1
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# print(y_pred)
# print(y_test)
# print("conf mat:", confusion_matrix(y_test, y_pred))
# print("f1:", f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
# print("recall:", recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
# print("precision:", precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))

# f = open( "results_scorer/" + topic_selected[1:] + ".txt", "w")
# f.write(str(results))
# f.write("\ntest_acc_mean: " + str(np.mean(results["test_acc"])))
# f.write("\ntest_rec_mean: " + str(np.mean(results["test_rec"])))
# f.write("\ntest_f1_mean: " + str(np.mean(results["test_f1"])))
# f.write("\ntest_prec_mean: " + str(np.mean(results["test_prec"])))
# f.close()






