
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


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

classes = ('micro', 'not_micro')

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


scaler = MinMaxScaler(feature_range=(0, 1)) 
X = scaler.fit_transform(X)
print(X)

features_tensor = torch.from_numpy(X).type(torch.FloatTensor)
print("X size:", features_tensor.size())
target_tensor = torch.FloatTensor(np.array(Y))
print("Y size:", target_tensor.size())

X = Variable(features_tensor, requires_grad = False).view(features_tensor.size())
y = Variable(target_tensor, requires_grad = False)

w = Variable(torch.randn(15), requires_grad=True)
b = Variable(torch.randn(145), requires_grad=True)

y_pred = torch.matmul(X,w) + b
print(y)
print(y_pred)

#f = nn.Layer(features_tensor.size())

def loss_fn(y,y_pred):
	loss = (y_pred-y).pow(2).sum()
	for param in [w,b]:
		if not param.grad is None: param.grad.data.zero_()
	loss.backward()
	return loss

def optimize(learning_rate):
	w.data -= learning_rate* w.grad.data
	b.data -= learning_rate* b.grad.data

loss = loss_fn(y,y_pred)
print("loss", loss)

optimize(1)

y_pred = torch.matmul(X,w) + b
loss = loss_fn(y,y_pred)
print("loss", loss)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)