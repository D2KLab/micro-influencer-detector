# This Python file uses the following encoding: utf-8
#!/usr/bin/

from micro_influencer_utilities import *
import sys
import os  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from random import sample  
from sklearn import tree  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.model_selection import cross_val_score  
from sklearn import metrics  
from IPython.display import Image  
from pydotplus import graph_from_dot_data  
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
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
df = pd.read_csv(pathToUserParameters+"/table/"+topic_selected[1:]+"2.csv")

#Pre-processing data set to data frame by choosing features (X) and output (y)
#print (dataset.head()) #debug 
y = df.y #micro influencer yes = 1, no = 0
X = df.drop(["user_screen_name", "Semb", "Srec", "Sint","y"],axis=1)# "num_of_words","distance", "y"], axis=1)

#choose features of interest in prevision excluding scores that preaviously otuput yes/no values
X = X.loc[:, ["big5_O", "big5_C", "big5_E", "big5_A", "big5_N","scoreselfdirection","scorestimulation", "scorehedonism", 
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", "scoretradition", "scorebenevolence", "scoreuniversalism"]] 
# y = dataset.iloc[:, 3039] #old version
#print (X.head()) #debug
Xcorr = df.drop(["user_screen_name", "Semb", "Srec", "Sint"],axis=1)
Xcorr = Xcorr.loc[:, ["big5_O", "big5_C", "big5_E", "big5_A", "big5_N","scoreselfdirection","scorestimulation", "scorehedonism", 
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", "scoretradition", "scorebenevolence", "scoreuniversalism", "y"]]
X = X.as_matrix()
y = y.as_matrix()

print(df.shape)
print("y=1: ", len(df[df['y'] == 1]))

correlation = Xcorr.corr()  
print(correlation)



# working with smote



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("\nNumber X_train dataset: ", X_train.shape)
print("Number y_train dataset: ", y_train.shape)
print("Number X_test dataset: ", X_test.shape)
print("Number y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))



parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression()
clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=1)
clf.fit(X_train_res, y_train_res.ravel())

print("clf.best_params_:  ", clf.best_params_)

lr1 = LogisticRegression(C=9,penalty='l1', verbose=5)
lr1.fit(X_train_res, y_train_res.ravel())


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

y_train_pre = lr1.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
plt.show()

y_pre = lr1.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pre)

print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Confusion matrix')
plt.show()

tmp = lr1.fit(X_train_res, y_train_res.ravel())

y_pred_sample_score = tmp.decision_function(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(roc_auc)
# scores_accuracy = []
# predicts_accuracy = []


# cv = KFold(n_splits=10, random_state=42, shuffle=False) 
# for train_index, test_index in cv.split(X):
#     # print("Train Index: ", train_index, "\n")  #debug
#     # print("Test Index: ", test_index) #debug
#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     logit_model = LogisticRegression() 
#     logit_model = logit_model.fit(X_train, y_train) 
#     scores_accuracy.append(logit_model.score(X_train, y_train))  ##this is accuracy
#     predicted = pd.DataFrame(logit_model.predict(X_test))
#     probs = pd.DataFrame(logit_model.predict_proba(X_test))
#     predicts_accuracy.append(metrics.accuracy_score(y_test, predicted))
#     print(metrics.classification_report(y_test, predicted))

# print(scores_accuracy)
# print(predicts_accuracy)
