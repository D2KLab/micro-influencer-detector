from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.svm import SVR
import datasetUtils as dsu
import numpy as np
import embeddings
import os
from pathlib import Path

#configs
method = "conc"
post_threshold = 3
dataset = "fasttext"
dataset_path = "FastText/dataset.vec" # https://fasttext.cc/docs/en/english-vectors.html 
                                            #wiki-news-300d-1M.vec.zip: 1 million word vectors trained on Wikipedia 2017,
                                            # UMBC webbase corpus and statmt.org news dataset (16B tokens).

if not os.path.exists("Models"):  
           os.makedirs("Models")
if not os.path.exists("Models/SVM/"):  
           os.makedirs("Models/SVM/")

posts = []
yO = []
yC = []
yE = []
yA = []
yN = []

print("Loading myPersonality...")
[posts, yO, yC, yE, yA, yN] = dsu.readMyPersonality()
print("Loading embeddings dataset...")
wordDictionary = dsu.parseFastText(dataset_path)
print("Data successfully loaded.")

#shuffle data
s = np.arange(posts.shape[0])
np.random.shuffle(s)
posts = posts[s]
yO = yO[s]
yC = yC[s]
yE = yE[s]
yA = yA[s]
yN = yN[s]
print("Data shuffled.")

[conE, yO, yC, yE, yA, yN] = embeddings.transformTextForTraining(wordDictionary, post_threshold, posts, yO, yC, yE, yA, yN, method, True)
print("Embeddings computed.")

l = 1
for labels in [yO, yC, yE, yA, yN]:

    if l==1:
        big5trait = "O"
        gamma = 1
        C = 1
        print("Training model for Openness...")
    elif l==2:
        big5trait = "C"
        gamma = 1
        C = 1
        print("Training model for Conscientiousness...")
    elif l==3:
        big5trait = "E"
        gamma = 1
        C = 10
        print("Training model for Extraversion...")
    elif l==4:
        big5trait = "A"
        gamma = 1
        C = 1
        print("Training model for Agreeableness...")
    elif l==5:
        big5trait = "N"
        gamma = 10
        C = 10
        print("Training model for Neuroticism...")
    l += 1

    model = SVR(kernel='rbf', gamma = gamma, C=C).fit(conE,labels)

    model_name = "SVM_"+big5trait+".pkl"
    
    f = open("Models/SVM/"+model_name, "wb")
    f.close()
    joblib.dump(model, "Models/SVM/"+model_name)
