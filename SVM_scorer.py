# This Python file uses the following encoding: utf-8
#!/usr/bin/


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from micro_influencer_utilities import *
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
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", "scoretradition", "scorebenevolence", "scoreuniversalism"]] 

X = X.as_matrix()
y = y.as_matrix()

Xmat = X
ymat = y

X = np.array(X)
y = np.array(y)
# print("y: ", y)

scaler = MinMaxScaler(feature_range=(0, 1)) #old version
X = scaler.fit_transform(X) #old version

Xmat =scaler.fit_transform(Xmat)
# kfold = model_selection.KFold(n_splits=10, random_state=42)
skf = StratifiedKFold(n_splits=10)
rbf_svc = svm.SVC(kernel='rbf', gamma='scale', class_weight={1:10}) #dai peso 10 alla classe 1
# print("cross_val_rec_micro", cross_val_score(rbf_svc, X, y, scoring='recall_micro', cv=10))
# print("cross_val_rec_macro", cross_val_score(rbf_svc, X, y, scoring='f1_micro', cv=10))

# rbf_svc.fit(X,y)
# y_pred = rbf_svc.predict(X)

# scoring = {'recall': make_scorer(recall_score, labels=None, pos_label=1, average='binary')}

# #scoring = ['precision','recall','f1']
# results = model_selection.cross_validate(rbf_svc, Xmat, ymat, cv=10, scoring=scoring)
# #print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# print("results", results)


# print("Classification metrics")
# from sklearn.metrics import classification_report
# y_true = y
# target_names = ['not_micro_infl', 'micro_infl']
# print(classification_report(y_true, y_pred, target_names=target_names))
# print("balanced_accuracy_score: ",balanced_accuracy_score(y_true,y_pred))

accuracy = [];
f1 = [];
recall = [];
precision = [];

print("Scoring parameters with StratifiedKfold with 10 fold, Classification Binary Case")
for train_index, test_index in skf.split(X, y):
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	clf = rbf_svc.fit(X_train, y_train)
	y_pred = rbf_svc.fit(X_train, y_train).predict(X_test)
	accuracy.append(clf.score(X_test, y_test))
	f1.append(f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	# print("f1",f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	recall.append(recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	# print("recall", recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	precision.append(precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	# print("precision", precision_score(y_test, y_pred, labels=None, pos_label=1, average='binary'))
	# print("y_test",y_test)
	# print("y_pred", y_pred)

print("mean accuracy: ",np.mean(accuracy))
print("mean recall: ", np.mean(recall))
print("mean precision: ", np.mean(precision))
print("mean f1: ", np.mean(f1))

# print("f1 for binary case: ",cross_val_score(rbf_svc, X, y, scoring='precision_micro', cv=4))
# print("f1", f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None))
# scoring = { 
# 'prec': make_scorer(precision_score, average='binary', pos_label=1)} 
# 'f1' : make_scorer(f1_score, average=None), 
# 'rec' : make_scorer(recall_score, average=None)}
# scoring = {"prec": 'precision'}
# results = model_selection.cross_validate(rbf_svc, X, y, cv=skf, scoring=scoring)
# print(results)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# cnf_matrix_tra = confusion_matrix(y_true, y_pred)
# class_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
# plt.show()


# print("y_pred:" , y_pred)
# scoring = {'acc': make_scorer(accuracy_score), 
# 'prec': make_scorer(precision_score,average=None), 
# 'f1' : make_scorer(f1_score, average=None), 
# 'rec' : make_scorer(recall_score, average=None)}
# scoring = {'acc': make_scorer(accuracy_score, y, y_pred, 'prec': make_scorer(precision_score,y, y_pred, pos_label=1 ,average='binary'), 
# 'f1' : make_scorer(f1_score,y,y_pred, pos_label=1 ,average='binary'), 'rec' : make_scorer(recall_score,y, y_pred, pos_label=1 ,average='binary')}





# print("test_acc mean: ", str(np.mean(results["test_acc"])))

# f = open( "results_scorer/" + topic_selected[1:] + ".txt", "w")
# f.write(str(results))
# f.write("\ntest_acc_mean: " + str(np.mean(results["test_acc"])))
# f.write("\ntest_rec_mean: " + str(np.mean(results["test_rec"])))
# f.write("\ntest_f1_mean: " + str(np.mean(results["test_f1"])))
# f.write("\ntest_prec_mean: " + str(np.mean(results["test_prec"])))
# f.close()






