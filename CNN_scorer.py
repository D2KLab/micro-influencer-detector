import numpy 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from micro_influencer_utilities import *


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

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

# load dataset
# split into input (X) and output (Y) variables

#Pre-processing data set to data frame by choosing features (X) and output (y)
#print (dataset.head()) #debug 
Y = dataset.y #micro influencer yes = 1, no = 0
X = dataset.drop(["user_screen_name", "Semb", "Srec", "Sint","y"],axis=1)# "num_of_words","distance", "y"], axis=1)

X = X.loc[:, ["big5_O", "big5_C", "big5_E", "big5_A", "big5_N","scoreselfdirection","scorestimulation", "scorehedonism", 
"scoreachievement","scorepower" ,"scoresecurity","scoreconformity", "scoretradition", "scorebenevolence", "scoreuniversalism"]].astype(float) 


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(15, input_dim=15, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# evaluate baseline model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = cross_val_predict(pipeline, X, encoded_Y, cv=kfold)
y_pred = numpy.asarray(y_pred)
#print("y_pred", y_pred)
#print("y_true", encoded_Y)

tn, fp, fn, tp = confusion_matrix(encoded_Y, y_pred).ravel();
precision = tp/(tp+fp);
recall = tp/(tp+fn)
f1_score = 2*(precision*recall)/(precision+recall);
print("precision: ", precision)
print("recall: ", recall)
print("f1_score:", f1_score)
# import keras
# import keras_metrics as km
# import sklearn.metrics as sklm


# class Metrics(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.confusion = []
#         self.precision = []
#         self.recall = []
#         self.f1s = []
#         self.kappa = []
#         self.auc = []

#     def on_epoch_end(self, epoch, logs={}):
#         score = numpy.asarray(self.model.predict(self.validation_data[0]))
#         predict = numpy.round(numpy.asarray(self.model.predict(self.validation_data[0])))
#         targ = self.validation_data[1]

#         self.auc.append(sklm.roc_auc_score(targ, score))
#         self.confusion.append(sklm.confusion_matrix(targ, predict))
#         self.precision.append(sklm.precision_score(targ, predict))
#         self.recall.append(sklm.recall_score(targ, predict))
#         self.f1s.append(sklm.f1_score(targ, predict))
#         self.kappa.append(sklm.cohen_kappa_score(targ, predict))

#         return






# #-----------------following doesn't work------------------
# def second_baseline():
# 	# create model
# 	model = Sequential()
# 	model.add(keras.layers.Dense(15, activation="sigmoid", input_dim=15))
# 	model.add(keras.layers.Dense(1, activation="softmax"))

# 	# Calculate precision for the label 1.
# 	precision = km.binary_precision(label=1)

# 	# Calculate recall for the label 1.
# 	recall = km.binary_recall(label=1)
# 	# Compile model
# 	model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=[precision, recall])
# 	return model

# sec_estimators = []
# sec_estimators.append(('standardize', StandardScaler()))
# sec_estimators.append(('mlp', KerasClassifier(build_fn=second_baseline, epochs=100, batch_size=5, verbose=0)))
# sec_pipeline = Pipeline(sec_estimators)
# sec_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# sec_results = cross_val_score(sec_pipeline, X, encoded_Y, cv=sec_kfold)
# #print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print("sec_results", sec_results)
